"""Predict the winner of a UFC matchup.

The matchup is featurized with the exact training pipeline: all known fights
(UFC + DWCS history) are replayed through :class:`features.FeatureBuilder`, and
the bout is treated as one more pending fight after them. Fighters with no UFC
history (debuts) get Elo 1500, a 0.5 smoothed win rate and missing career
rates; their physical attributes and any DWCS record are still used.

Usage::

    python predict.py "Fighter A" "Fighter B" [--five-rounds] [--title] [--date YYYY-MM-DD]

A fighter can also be given by ufcstats profile URL or id to disambiguate
shared names.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from datetime import date as dt_date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import (FIGHTER_FEATURES, FeatureBuilder, build_features, fighter_key, load_data,
                      prepare_dwcs)
from ufc_common import norm_name, url_id

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"

LABELS = {
    "n_fights": "UFC fights", "wins": "UFC wins", "losses": "UFC losses", "win_rate": "Win rate (smoothed)",
    "dwcs_fights": "Contender Series bouts", "dwcs_wins": "Contender Series wins",
    "slpm": "Sig. strikes landed/min", "sapm": "Sig. strikes absorbed/min", "sig_acc": "Striking accuracy",
    "sig_def": "Striking defense", "kd_per15": "Knockdowns/15 min", "head_share": "Head-strike share",
    "body_share": "Body-strike share", "leg_share": "Leg-strike share", "dist_share": "Distance-strike share",
    "clinch_share": "Clinch-strike share", "ground_share": "Ground-strike share",
    "td_per15": "Takedowns/15 min", "td_acc": "Takedown accuracy", "td_def": "Takedown defense",
    "sub_per15": "Sub attempts/15 min", "ctrl_pct": "Control time %", "opp_ctrl_pct": "Time controlled by opponents",
    "ko_wins": "KO/TKO wins", "sub_wins": "Submission wins", "finish_rate": "Finish rate",
    "ko_losses": "KO/TKO losses", "last3_win_rate": "Win rate, last 3", "streak": "Current streak",
    "last3_sig_diff_pm": "Strike differential/min, last 3", "days_since_last": "Days since last fight",
    "age": "Age", "height": "Height (in)", "reach": "Reach (in)", "southpaw": "Southpaw",
    "elo": "Elo rating", "opp_elo_avg": "Avg. opponent Elo",
    "stance_southpaw_vs_orthodox": "Southpaw vs orthodox matchup",
    "five_round": "Five-round fight", "title_fight": "Title fight",
}


GROUPS = {
    "UFC record (W-L)": {"n_fights", "wins", "losses", "win_rate"},
    "Contender Series record (W-L)": {"dwcs_fights", "dwcs_wins"},
}


def _record(row: dict, side: str, group: str) -> str:
    if group.startswith("UFC"):
        return f"{int(row[f'{side}_wins'])}-{int(row[f'{side}_losses'])}"
    w, n = int(row[f"{side}_dwcs_wins"]), int(row[f"{side}_dwcs_fights"])
    return f"{w}-{n - w}"


@dataclass
class FighterMatch:
    key: str
    name: str
    note: str = ""


class FighterIndex:
    """Name -> fighter-key lookup over UFC bouts, DWCS bouts and fighter profiles."""

    def __init__(self, fights: pd.DataFrame, fighters: pd.DataFrame | None, dwcs: pd.DataFrame | None,
                 builder: FeatureBuilder):
        self.builder = builder
        self.by_name: dict[str, dict[str, str]] = {}   # norm name -> {key: display name}
        self.last_seen: dict[str, str] = {}
        for p in ("a", "b"):
            for k, n, d in zip(
                    (fighter_key(u, n) for u, n in zip(fights[f"fighter_{p}_url"], fights[f"fighter_{p}"])),
                    fights[f"fighter_{p}"], fights["date"]):
                self._add(k, n, d)
        if dwcs is not None:
            d = prepare_dwcs(dwcs, fighters)
            for p in ("a", "b"):
                for k, n, dd in zip(d[f"key_{p}"], d[f"fighter_{p}"], d["date"].dt.strftime("%Y-%m-%d")):
                    self._add(k, n, dd)
        if fighters is not None:
            for k, n in zip(fighters["fighter_id"], fighters["name"]):
                self._add(str(k), n, "")

    def _add(self, key, name, date):
        self.by_name.setdefault(norm_name(name), {}).setdefault(key, str(name))
        if date and date > self.last_seen.get(key, ""):
            self.last_seen[key] = date

    def resolve(self, query: str) -> FighterMatch:
        """Find a fighter by name (accent/case-insensitive), ufcstats URL or id."""
        q = query.strip()
        uid = url_id(q) if "ufcstats.com" in q else q
        if uid in self.builder.states or uid in self.builder.physical:
            name = next((n for d in self.by_name.values() for k, n in d.items() if k == uid), uid)
            return FighterMatch(uid, name)
        cands = self.by_name.get(norm_name(q))
        if not cands:
            close = difflib.get_close_matches(norm_name(q), list(self.by_name), n=3, cutoff=0.85)
            if close:
                cands = self.by_name[close[0]]
                note = f'no exact match for "{q}", using closest name'
            else:
                sugg = difflib.get_close_matches(norm_name(q), list(self.by_name), n=3, cutoff=0.6)
                hint = f" (did you mean: {', '.join(next(iter(self.by_name[s].values())) for s in sugg)}?)" if sugg else ""
                return FighterMatch(f"name:{q}", q, f"not found in the data; treated as a debut with no "
                                                    f"known stats or physicals{hint}")
        else:
            note = ""
        if len(cands) > 1:
            ranked = sorted(cands, key=lambda k: self.last_seen.get(k, ""), reverse=True)
            alts = ", ".join(f"{k} (last fight {self.last_seen.get(k) or 'none'})" for k in ranked[1:])
            note = (note + "; " if note else "") + f"name is shared, using the most recently active profile; others: {alts}"
            key = ranked[0]
        else:
            key = next(iter(cands))
        return FighterMatch(key, cands[key], note)


def predict_matchup(model, builder: FeatureBuilder, key_a: str, key_b: str, date, five_rounds=False,
                    title=False, top: int = 6) -> dict:
    """Orientation-averaged win probabilities and the top contributing factors."""
    row = builder.matchup(key_a, key_b, date, int(five_rounds), int(title))
    X = pd.DataFrame([row])
    p_a = float(model.predict_proba(X)[0])
    contrib = model.contributions(X).iloc[0]
    # Collinear features (e.g. wins/losses/fights) are summed into one factor so
    # their individual signs, which split weight arbitrarily, are never shown alone.
    grouped: dict[str, float] = {}
    for feat, c in contrib.items():
        base = feat[5:] if feat.startswith("diff_") else feat
        g = next((name for name, members in GROUPS.items() if base in members), feat)
        grouped[g] = grouped.get(g, 0.0) + float(c)
    factors = []
    for g in sorted(grouped, key=lambda k: -abs(grouped[k]))[:top]:
        c = grouped[g]
        if abs(c) < 1e-9:
            continue
        if g in GROUPS:
            label, va, vb = g, _record(row, "a", g), _record(row, "b", g)
            feat = g
        else:
            feat = g
            base = feat[5:] if feat.startswith("diff_") else feat
            label = LABELS.get(base, base)
            va, vb = (row.get(f"a_{base}"), row.get(f"b_{base}")) if base in FIGHTER_FEATURES else (row[feat], None)
        factors.append({"feature": feat, "label": label, "a": va, "b": vb,
                        "log_odds": c, "favors": "a" if c > 0 else "b"})
    return {"p_a": p_a, "p_b": 1.0 - p_a, "factors": factors, "row": row}


def _fmt(v, feat):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if feat.endswith(("acc", "def", "rate", "share", "pct")) and abs(v) <= 1:
        return f"{100 * v:.0f}%"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.2f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fighter_a")
    ap.add_argument("fighter_b")
    ap.add_argument("--five-rounds", action="store_true", help="scheduled for five rounds")
    ap.add_argument("--title", action="store_true", help="title fight")
    ap.add_argument("--date", default=dt_date.today().isoformat(), help="fight date (default: today)")
    ap.add_argument("--top", type=int, default=6, help="number of factors to show")
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} not found: run train.py first")
    bundle = joblib.load(MODEL_PATH)
    fights, fighters, dwcs = load_data()
    _, builder = build_features(fights, fighters, dwcs, elo_k=bundle["elo_k"])
    if pd.Timestamp(args.date) <= pd.to_datetime(fights["date"]).max():
        print(f"note: --date {args.date} is not after the last known fight; "
              "all known fights are still used as history", file=sys.stderr)

    index = FighterIndex(fights, fighters, dwcs, builder)
    a, b = index.resolve(args.fighter_a), index.resolve(args.fighter_b)
    if a.key == b.key:
        sys.exit("both names resolve to the same fighter")
    for m in (a, b):
        if m.note:
            print(f"[{m.name}] {m.note}", file=sys.stderr)

    res = predict_matchup(bundle["model"], builder, a.key, b.key, args.date, args.five_rounds, args.title, args.top)
    row = res["row"]
    print(f"\n{a.name} vs {b.name}  ({'5' if args.five_rounds else '3'} rounds{', title fight' if args.title else ''})")
    print(f"model: {bundle['name']} (trained through {bundle['trained_through']})\n")
    w = max(len(a.name), len(b.name))
    for m, p, side in ((a, res["p_a"], "a"), (b, res["p_b"], "b")):
        rec = f"UFC {int(row[f'{side}_wins'])}-{int(row[f'{side}_losses'])}"
        if row[f"{side}_dwcs_fights"]:
            rec += f", DWCS {int(row[f'{side}_dwcs_wins'])}-{int(row[f'{side}_dwcs_fights'] - row[f'{side}_dwcs_wins'])}"
        debut = "  (UFC debut)" if row[f"{side}_n_fights"] == 0 else ""
        print(f"  {m.name:<{w}}  {100 * p:5.1f}%   [{rec}, Elo {row[f'{side}_elo']:.0f}]{debut}")
    print("\nTop factors (log-odds contribution):")
    for f in res["factors"]:
        who = a.name if f["favors"] == "a" else b.name
        vals = f"{_fmt(f['a'], f['feature'])} vs {_fmt(f['b'], f['feature'])}" if f["b"] is not None \
            else _fmt(f["a"], f["feature"])
        print(f"  {f['label']:<34} {vals:<20} {f['log_odds']:+.3f} -> favors {who}")


if __name__ == "__main__":
    main()
