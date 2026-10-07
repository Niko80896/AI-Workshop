"""Leak-free, pre-fight feature engineering.

All features for a fight are computed from fights strictly *before* that
fight's date. ``FeatureBuilder`` walks the fight history chronologically, one
date at a time: it first computes the matchup features of every fight on a date
from the current state, and only then folds those fights into each fighter's
running state (aggregates + Elo). Prediction reuses the exact same object: after
replaying all known fights, an upcoming bout is just one more pending fight.

Dana White's Contender Series bouts (``data/dwcs_fights.csv``, results only) are
merged into the same chronological stream as extra history: they update form,
streak, layoff, finishes, Elo and the DWCS record, but not the UFC fight counts
or the per-minute striking/grappling rates (no stats exist for them). They are
never used as training rows.

The model input for a fight is the difference ``A - B`` of every per-fighter
feature (plus a stance-matchup flag and the fight context). Because ufcstats
usually lists the winner first, training uses both orientations of each fight
(see :func:`symmetrize`) and predictions average ``P(A wins)`` with
``1 - P(B wins)``.

Usage::

    python features.py      # writes data/features.csv for inspection
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ufc_common import (
    DATA_DIR, DIVISION_LBS, DWCS_CSV, FIGHTERS_CSV, FIGHTS_CSV, fight_seconds, method_group,
    name_resolver, scheduled_rounds, url_id,
)

ELO_START = 1500.0
ELO_K = 80.0  # chosen by pre-2024 Elo log loss (40: .6836, 80: .6821, 120: .6856)

# Per-fighter pre-fight features (each enters the model as ``diff_<name>`` = A - B).
FIGHTER_FEATURES = [
    # experience (UFC only) + Contender Series record
    "n_fights", "wins", "losses", "win_rate", "dwcs_fights", "dwcs_wins",
    # striking
    "slpm", "sapm", "sig_acc", "sig_def", "kd_per15",
    "head_share", "body_share", "leg_share", "dist_share", "clinch_share", "ground_share",
    # grappling
    "td_per15", "td_acc", "td_def", "sub_per15", "ctrl_pct", "opp_ctrl_pct",
    # finishing / durability
    "ko_wins", "sub_wins", "finish_rate", "ko_losses", "sub_losses", "dec_rate",
    # form
    "last3_win_rate", "streak", "last3_sig_diff_pm",
    # activity
    "days_since_last",
    # physical
    "age", "height", "reach", "southpaw",
    # weight class movement: this bout's division limit minus the previous bout's (lbs)
    "weight_change",
    # rating
    "elo", "opp_elo_avg",
]
DIFF_FEATURES = [f"diff_{f}" for f in FIGHTER_FEATURES]
# Antisymmetric extras: +1 if A is the southpaw facing an orthodox B, -1 for the reverse.
MATCHUP_FEATURES = ["stance_southpaw_vs_orthodox"]
CONTEXT_FEATURES = ["five_round", "title_fight", "division_lbs", "womens"]
# Betting-market log-odds that A wins (vig removed). Only used by the optional odds model.
ODDS_FEATURES = ["market_logit"]
ANTISYMMETRIC = DIFF_FEATURES + MATCHUP_FEATURES + ODDS_FEATURES
MODEL_FEATURES = DIFF_FEATURES + MATCHUP_FEATURES + CONTEXT_FEATURES
# Symmetric "how much finishing is in this fight" totals (A + B) for the method model.
SUM_BASES = ["slpm", "sapm", "kd_per15", "td_per15", "sub_per15", "ctrl_pct", "ko_wins", "sub_wins",
             "ko_losses", "sub_losses", "dec_rate", "finish_rate"]
SUM_FEATURES = [f"sum_{f}" for f in SUM_BASES]
METHOD_FEATURES = MODEL_FEATURES + SUM_FEATURES
METHOD_CLASSES = ["a_ko", "a_sub", "a_dec", "b_ko", "b_sub", "b_dec"]
METHOD_FLIP = {c: ("b" if c[0] == "a" else "a") + c[1:] for c in METHOD_CLASSES}

# Running sums kept per fighter, for "own" and "opp" (what the opponent did to them).
_SUM_STATS = ["kd", "sig_landed", "sig_att", "td_landed", "td_att", "sub_att", "ctrl_sec",
              "head_landed", "body_landed", "leg_landed", "dist_landed", "clinch_landed",
              "ground_landed"]


def fighter_key(url, name) -> str:
    """Stable fighter id: the ufcstats profile hash, or ``name:<name>`` when no URL is known."""
    uid = url_id(url) if isinstance(url, str) else None
    return uid or f"name:{str(name).strip()}"


def _div(a, b):
    return a / b if b and b > 0 else np.nan


@dataclass
class FighterState:
    """Running, pre-fight career aggregates for one fighter."""

    n: int = 0                        # UFC fights
    wins: int = 0
    losses: int = 0
    dwcs_fights: int = 0
    dwcs_wins: int = 0
    ko_wins: int = 0
    sub_wins: int = 0
    ko_losses: int = 0
    sub_losses: int = 0
    decisions: int = 0
    last_division: float = np.nan     # division limit (lbs) of the most recent bout with a known division
    last_weight_class: str | None = None
    stat_secs: float = 0.0            # minutes denominator: only fights with recorded stats
    own: dict = field(default_factory=lambda: dict.fromkeys(_SUM_STATS, 0.0))
    opp: dict = field(default_factory=lambda: dict.fromkeys(_SUM_STATS, 0.0))
    results: list = field(default_factory=list)                         # 'W','L','D','NC'
    recent_diffs: deque = field(default_factory=lambda: deque(maxlen=3))  # (sig diff, secs)
    last_date: pd.Timestamp | None = None
    opp_elo_sum: float = 0.0

    def features(self, date: pd.Timestamp) -> dict:
        """Feature values as of just before ``date`` (career + form + activity)."""
        mins = self.stat_secs / 60.0
        o, p = self.own, self.opp
        decided = self.wins + self.losses
        last3 = [r for r in self.results if r != "NC"][-3:]
        streak = 0
        for r in reversed(self.results):
            if r == "NC":
                continue
            if r not in ("W", "L") or (streak > 0 and r == "L") or (streak < 0 and r == "W"):
                break
            streak += 1 if r == "W" else -1
        rd_secs = sum(s for _, s in self.recent_diffs)
        all_fights = self.n + self.dwcs_fights
        return {
            "n_fights": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": (self.wins + 1) / (decided + 2),  # Laplace-smoothed, 0.5 for debuts
            "dwcs_fights": self.dwcs_fights,
            "dwcs_wins": self.dwcs_wins,
            "slpm": _div(o["sig_landed"], mins),
            "sapm": _div(p["sig_landed"], mins),
            "sig_acc": _div(o["sig_landed"], o["sig_att"]),
            "sig_def": 1 - _div(p["sig_landed"], p["sig_att"]),
            "kd_per15": _div(o["kd"] * 15, mins),
            **{f"{t}_share": _div(o[f"{t}_landed"], o["sig_landed"])
               for t in ("head", "body", "leg", "dist", "clinch", "ground")},
            "td_per15": _div(o["td_landed"] * 15, mins),
            "td_acc": _div(o["td_landed"], o["td_att"]),
            "td_def": 1 - _div(p["td_landed"], p["td_att"]),
            "sub_per15": _div(o["sub_att"] * 15, mins),
            "ctrl_pct": _div(o["ctrl_sec"], self.stat_secs),
            "opp_ctrl_pct": _div(p["ctrl_sec"], self.stat_secs),
            "ko_wins": self.ko_wins,
            "sub_wins": self.sub_wins,
            "finish_rate": _div(self.ko_wins + self.sub_wins, all_fights),
            "ko_losses": self.ko_losses,
            "sub_losses": self.sub_losses,
            "dec_rate": _div(self.decisions, all_fights),
            "last3_win_rate": (sum(r == "W" for r in last3) / len(last3)) if last3 else np.nan,
            "streak": streak,
            "last3_sig_diff_pm": _div(sum(d for d, _ in self.recent_diffs) * 60, rd_secs),
            "days_since_last": (date - self.last_date).days if self.last_date is not None else np.nan,
            "opp_elo_avg": _div(self.opp_elo_sum, all_fights),
        }


def division_lbs(weight_class) -> pd.Series:
    """Division weight limit in lbs (NaN for catch/open weight or unknown)."""
    return pd.Series(weight_class).astype(str).str.strip().map(DIVISION_LBS).astype(float)


def womens(weight_class) -> pd.Series:
    return pd.Series(weight_class).astype(str).str.contains("Women", case=False).astype(int)


def prepare_fights(fights: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (keys, duration, method group, ...) and sort chronologically."""
    f = fights.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["key_a"] = [fighter_key(u, n) for u, n in zip(f["fighter_a_url"], f["fighter_a"])]
    f["key_b"] = [fighter_key(u, n) for u, n in zip(f["fighter_b_url"], f["fighter_b"])]
    f["duration_sec"] = [fight_seconds(r, t, tf) for r, t, tf in
                         zip(f["end_round"], f["end_time"], f["time_format"])]
    f["method_group"] = f["method"].map(method_group)
    f["five_round"] = (f["time_format"].map(scheduled_rounds) == 5).astype(int)
    f["title_fight"] = f["title_fight"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    f["division_lbs"] = division_lbs(f["weight_class"])
    f["womens"] = womens(f["weight_class"])
    core = [f"{p}_{c}" for p in ("a", "b") for c in ("sig_landed", "sig_att", "td_landed", "td_att", "ctrl_sec")]
    f["has_stats"] = f[core].notna().all(axis=1) & f["duration_sec"].gt(0)
    f["label"] = f["result_a"].map({"W": 1.0, "L": 0.0})  # draws / NC -> NaN (not trained on)
    f["promotion"] = "UFC"
    return f.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)


def prepare_dwcs(dwcs: pd.DataFrame, fighters: pd.DataFrame | None) -> pd.DataFrame:
    """Map DWCS fighter names onto ufcstats fighter keys and add the derived columns.

    Fighters who later reached the UFC get their ufcstats key (so their DWCS
    record carries over); everyone else is keyed by name.
    """
    d = dwcs.copy()
    d["date"] = pd.to_datetime(d["date"])
    resolve = name_resolver(fighters) if fighters is not None and len(fighters) else (lambda n, w=None: None)
    for p in ("a", "b"):
        d[f"key_{p}"] = [fighter_key(resolve(n, w), n) for n, w in zip(d[f"fighter_{p}"], d["weight_class"])]
    d["method_group"] = d["method"].map(method_group)
    d["duration_sec"] = np.nan
    d["has_stats"] = False
    d["five_round"], d["title_fight"], d["label"] = 0, 0, np.nan
    d["division_lbs"], d["womens"] = division_lbs(d["weight_class"]), womens(d["weight_class"])
    d["promotion"] = "DWCS"
    return d.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)


class FeatureBuilder:
    """Chronological state machine producing leak-free pre-fight features."""

    def __init__(self, fighters: pd.DataFrame | None = None, elo_k: float = ELO_K):
        self.elo_k = elo_k
        self.states: dict[str, FighterState] = {}
        self.elo: dict[str, float] = {}
        self.physical: dict[str, dict] = {}
        if fighters is not None and len(fighters):
            for r in fighters.itertuples(index=False):
                self.physical[str(r.fighter_id)] = {
                    "height": r.height_in, "reach": r.reach_in, "stance": r.stance,
                    "dob": pd.to_datetime(r.dob) if isinstance(r.dob, str) else pd.NaT,
                }

    # ---------------------------------------------------------------- features
    def fighter_features(self, key: str, date: pd.Timestamp, division: float = np.nan) -> dict:
        """All per-fighter features for ``key`` as of just before ``date`` (bout at ``division`` lbs)."""
        st = self.states.get(key) or FighterState()
        feats = st.features(date)
        feats["weight_change"] = division - st.last_division  # NaN if either is unknown
        phys = self.physical.get(key, {})
        dob = phys.get("dob", pd.NaT)
        feats["age"] = (date - dob).days / 365.25 if pd.notna(dob) else np.nan
        feats["height"] = phys.get("height", np.nan)
        feats["reach"] = phys.get("reach", np.nan)
        stance = phys.get("stance")
        feats["southpaw"] = float(stance == "Southpaw") if isinstance(stance, str) else np.nan
        feats["_stance"] = stance if isinstance(stance, str) else None
        feats["elo"] = self.elo.get(key, ELO_START)
        return feats

    def matchup(self, key_a: str, key_b: str, date, five_round: int = 0, title_fight: int = 0,
                weight_class: str | None = None) -> dict:
        """Model-ready feature row for A vs B on ``date`` (plus raw per-fighter values)."""
        date = pd.Timestamp(date)
        div = float(division_lbs([weight_class]).iloc[0])
        fa, fb = self.fighter_features(key_a, date, div), self.fighter_features(key_b, date, div)
        row = {f"diff_{k}": fa[k] - fb[k] for k in FIGHTER_FEATURES}
        row.update({f"sum_{k}": fa[k] + fb[k] for k in SUM_BASES})
        sa, sb = fa.pop("_stance"), fb.pop("_stance")
        row["stance_southpaw_vs_orthodox"] = (
            float(sa == "Southpaw" and sb == "Orthodox") - float(sa == "Orthodox" and sb == "Southpaw")
            if sa and sb else np.nan)
        row["five_round"], row["title_fight"] = int(five_round), int(title_fight)
        row["division_lbs"], row["womens"] = div, int(womens([weight_class]).iloc[0])
        row.update({f"a_{k}": v for k, v in fa.items()})
        row.update({f"b_{k}": v for k, v in fb.items()})
        return row

    # ------------------------------------------------------------------ update
    def update(self, fights_on_date: pd.DataFrame) -> None:
        """Fold every fight of one date into the fighters' states and Elo ratings."""
        pre_elo = {}
        for r in fights_on_date.itertuples(index=False):
            for k in (r.key_a, r.key_b):
                pre_elo.setdefault(k, self.elo.get(k, ELO_START))
        for r in fights_on_date.itertuples(index=False):
            self._update_fighter(r, "a", "b", pre_elo[r.key_b])
            self._update_fighter(r, "b", "a", pre_elo[r.key_a])
            self._update_elo(r)

    def _update_fighter(self, r, me: str, op: str, opp_elo: float) -> None:
        key = getattr(r, f"key_{me}")
        st = self.states.setdefault(key, FighterState())
        res = getattr(r, f"result_{me}")
        res = res if res in ("W", "L", "D", "NC") else "NC"
        if r.promotion == "DWCS":
            st.dwcs_fights += 1
            st.dwcs_wins += res == "W"
        else:
            st.n += 1
            st.wins += res == "W"
            st.losses += res == "L"
        st.results.append(res)
        st.opp_elo_sum += opp_elo
        st.last_date = r.date
        if res == "W":
            st.ko_wins += r.method_group == "KO/TKO"
            st.sub_wins += r.method_group == "SUB"
        elif res == "L":
            st.ko_losses += r.method_group == "KO/TKO"
            st.sub_losses += r.method_group == "SUB"
        st.decisions += r.method_group == "DEC"
        if not np.isnan(r.division_lbs):
            st.last_division, st.last_weight_class = r.division_lbs, str(r.weight_class).strip()
        if r.has_stats:
            st.stat_secs += r.duration_sec
            for s in _SUM_STATS:
                st.own[s] += np.nan_to_num(getattr(r, f"{me}_{s}"))
                st.opp[s] += np.nan_to_num(getattr(r, f"{op}_{s}"))
            st.recent_diffs.append((getattr(r, f"{me}_sig_landed") - getattr(r, f"{op}_sig_landed"),
                                    r.duration_sec))

    def _update_elo(self, r) -> None:
        if r.result_a not in ("W", "L", "D"):
            return  # no contest: no rating change
        ra, rb = self.elo.get(r.key_a, ELO_START), self.elo.get(r.key_b, ELO_START)
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        score_a = {"W": 1.0, "L": 0.0, "D": 0.5}[r.result_a]
        delta = self.elo_k * (score_a - expected_a)
        self.elo[r.key_a], self.elo[r.key_b] = ra + delta, rb - delta


def build_features(fights: pd.DataFrame, fighters: pd.DataFrame | None = None,
                   dwcs: pd.DataFrame | None = None,
                   elo_k: float = ELO_K) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Replay all fights chronologically; return one feature row per UFC fight and the final builder.

    ``dwcs`` (optional) adds Contender Series bouts as history. The returned
    builder holds the state *after* the last known bout and is what
    ``predict.py`` uses to featurize upcoming fights.
    """
    f = prepare_fights(fights)
    stream = f
    if dwcs is not None and len(dwcs):
        stream = pd.concat([f, prepare_dwcs(dwcs, fighters)], ignore_index=True)
    builder = FeatureBuilder(fighters, elo_k=elo_k)
    rows = []
    for date, group in stream.groupby("date", sort=True):
        for r in group[group["promotion"] == "UFC"].itertuples(index=False):
            rows.append(builder.matchup(r.key_a, r.key_b, date, r.five_round, r.title_fight,
                                        r.weight_class))                                  # 1) pre-date state
        builder.update(group)                                                                  # 2) then fold in
    feats = pd.DataFrame(rows, index=f.index)
    meta = f[["fight_id", "date", "event_name", "fighter_a", "fighter_b", "key_a", "key_b",
              "weight_class", "method_group", "result_a", "label"]]
    out = pd.concat([meta, feats], axis=1)
    out["method_class"] = method_class(out["label"], out["method_group"])
    return out, builder


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Stack each fight with its mirrored copy (B vs A, features negated, label flipped).

    Removes the "winner listed first" bias of ufcstats and makes the model's
    view of A vs B consistent with B vs A.
    """
    flipped = df.copy()
    cols = [c for c in ANTISYMMETRIC if c in df]
    flipped[cols] = -df[cols]
    if "label" in df:
        flipped["label"] = 1.0 - df["label"]
    if "method_class" in df:
        flipped["method_class"] = df["method_class"].map(METHOD_FLIP)
    return pd.concat([df.assign(orientation=0), flipped.assign(orientation=1)], ignore_index=True)


def method_class(label: pd.Series, method_grp: pd.Series) -> pd.Series:
    """6-way outcome from A's view (``a_ko`` ... ``b_dec``); NaN for draws, NC, DQ and other."""
    m = method_grp.map({"KO/TKO": "ko", "SUB": "sub", "DEC": "dec"})
    side = label.map({1.0: "a", 0.0: "b"})
    return (side + "_" + m).where(side.notna() & m.notna())


def add_market(feats: pd.DataFrame, odds: pd.DataFrame | None) -> pd.DataFrame:
    """Join vig-free market probabilities (``import_odds.py``) and the ``market_logit`` feature."""
    out = feats.copy()
    if odds is None or not len(odds):
        out["p_market_a"] = np.nan
    else:
        cols = ["fight_id", "p_market_a"] + [f"p_market_{c}" for c in METHOD_CLASSES]
        out = out.merge(odds.reindex(columns=cols), on="fight_id", how="left")
        out.index = feats.index
    p = out["p_market_a"].clip(1e-4, 1 - 1e-4)
    out["market_logit"] = np.log(p / (1 - p))
    return out


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Read ``fights.csv``, ``fighters.csv`` and (if present) ``dwcs_fights.csv`` from ``data/``."""
    if not FIGHTS_CSV.exists():
        raise SystemExit(f"{FIGHTS_CSV} not found: run scrape.py or import_mirror.py first")
    fighters = pd.read_csv(FIGHTERS_CSV) if FIGHTERS_CSV.exists() else None
    dwcs = pd.read_csv(DWCS_CSV) if DWCS_CSV.exists() else None
    return pd.read_csv(FIGHTS_CSV), fighters, dwcs


def main() -> None:
    fights, fighters, dwcs = load_data()
    feats, builder = build_features(fights, fighters, dwcs)
    out = DATA_DIR / "features.csv"
    feats.to_csv(out, index=False)
    print(f"wrote {len(feats)} rows x {len(MODEL_FEATURES)} model features to {out}")
    print(f"{len(builder.states)} fighters tracked; labelled fights: {feats['label'].notna().sum()}; "
          f"DWCS bouts used as history: {0 if dwcs is None else len(dwcs)}")


if __name__ == "__main__":
    main()
