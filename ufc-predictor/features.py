"""Leak-free, pre-fight feature engineering.

All features for a fight are computed from fights strictly *before* that
fight's date. ``FeatureBuilder`` walks the fight history chronologically, one
date at a time: it first computes the matchup features of every fight on a date
from the current state, and only then folds those fights into each fighter's
running state (aggregates + Elo). Prediction reuses the exact same object: after
replaying all known fights, an upcoming bout is just one more pending fight.

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
    DATA_DIR, FIGHTERS_CSV, FIGHTS_CSV, fight_seconds, method_group,
    scheduled_rounds, url_id,
)

ELO_START = 1500.0
ELO_K = 40.0

# Per-fighter pre-fight features (each enters the model as ``diff_<name>`` = A - B).
FIGHTER_FEATURES = [
    # experience
    "n_fights", "wins", "losses", "win_rate",
    # striking
    "slpm", "sapm", "sig_acc", "sig_def", "kd_per15",
    "head_share", "body_share", "leg_share", "dist_share", "clinch_share", "ground_share",
    # grappling
    "td_per15", "td_acc", "td_def", "sub_per15", "ctrl_pct", "opp_ctrl_pct",
    # finishing / durability
    "ko_wins", "sub_wins", "finish_rate", "ko_losses",
    # form
    "last3_win_rate", "streak", "last3_sig_diff_pm",
    # activity
    "days_since_last",
    # physical
    "age", "height", "reach", "southpaw",
    # rating
    "elo", "opp_elo_avg",
]
DIFF_FEATURES = [f"diff_{f}" for f in FIGHTER_FEATURES]
# Antisymmetric extras: +1 if A is the southpaw facing an orthodox B, -1 for the reverse.
MATCHUP_FEATURES = ["stance_southpaw_vs_orthodox"]
CONTEXT_FEATURES = ["five_round", "title_fight"]
ANTISYMMETRIC = DIFF_FEATURES + MATCHUP_FEATURES
MODEL_FEATURES = ANTISYMMETRIC + CONTEXT_FEATURES

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

    n: int = 0
    wins: int = 0
    losses: int = 0
    ko_wins: int = 0
    sub_wins: int = 0
    ko_losses: int = 0
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
        return {
            "n_fights": self.n,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": (self.wins + 1) / (decided + 2),  # Laplace-smoothed, 0.5 for debuts
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
            "finish_rate": _div(self.ko_wins + self.sub_wins, self.n),
            "ko_losses": self.ko_losses,
            "last3_win_rate": (sum(r == "W" for r in last3) / len(last3)) if last3 else np.nan,
            "streak": streak,
            "last3_sig_diff_pm": _div(sum(d for d, _ in self.recent_diffs) * 60, rd_secs),
            "days_since_last": (date - self.last_date).days if self.last_date is not None else np.nan,
            "opp_elo_avg": _div(self.opp_elo_sum, self.n),
        }


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
    core = [f"{p}_{c}" for p in ("a", "b") for c in ("sig_landed", "sig_att", "td_landed", "td_att", "ctrl_sec")]
    f["has_stats"] = f[core].notna().all(axis=1) & f["duration_sec"].gt(0)
    f["label"] = f["result_a"].map({"W": 1.0, "L": 0.0})  # draws / NC -> NaN (not trained on)
    return f.sort_values(["date", "fight_id"], kind="stable").reset_index(drop=True)


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
    def fighter_features(self, key: str, date: pd.Timestamp) -> dict:
        """All per-fighter features for ``key`` as of just before ``date``."""
        st = self.states.get(key) or FighterState()
        feats = st.features(date)
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

    def matchup(self, key_a: str, key_b: str, date, five_round: int = 0, title_fight: int = 0) -> dict:
        """Model-ready feature row for A vs B on ``date`` (plus raw per-fighter values)."""
        date = pd.Timestamp(date)
        fa, fb = self.fighter_features(key_a, date), self.fighter_features(key_b, date)
        row = {f"diff_{k}": fa[k] - fb[k] for k in FIGHTER_FEATURES}
        sa, sb = fa.pop("_stance"), fb.pop("_stance")
        row["stance_southpaw_vs_orthodox"] = (
            float(sa == "Southpaw" and sb == "Orthodox") - float(sa == "Orthodox" and sb == "Southpaw")
            if sa and sb else np.nan)
        row["five_round"], row["title_fight"] = int(five_round), int(title_fight)
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
        st.n += 1
        st.results.append(res)
        st.opp_elo_sum += opp_elo
        st.last_date = r.date
        if res == "W":
            st.wins += 1
            st.ko_wins += r.method_group == "KO/TKO"
            st.sub_wins += r.method_group == "SUB"
        elif res == "L":
            st.losses += 1
            st.ko_losses += r.method_group == "KO/TKO"
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
                   elo_k: float = ELO_K) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Replay all fights chronologically; return one feature row per fight and the final builder.

    The returned builder holds the state *after* the last known fight and is what
    ``predict.py`` uses to featurize upcoming bouts.
    """
    f = prepare_fights(fights)
    builder = FeatureBuilder(fighters, elo_k=elo_k)
    rows = []
    for date, group in f.groupby("date", sort=True):
        for r in group.itertuples(index=False):  # 1) features from pre-date state
            rows.append(builder.matchup(r.key_a, r.key_b, date, r.five_round, r.title_fight))
        builder.update(group)                     # 2) then fold the date's results in
    feats = pd.DataFrame(rows, index=f.index)
    meta = f[["fight_id", "date", "event_name", "fighter_a", "fighter_b", "key_a", "key_b",
              "weight_class", "method_group", "result_a", "label"]]
    return pd.concat([meta, feats], axis=1), builder


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Stack each fight with its mirrored copy (B vs A, features negated, label flipped).

    Removes the "winner listed first" bias of ufcstats and makes the model's
    view of A vs B consistent with B vs A.
    """
    flipped = df.copy()
    flipped[ANTISYMMETRIC] = -df[ANTISYMMETRIC]
    if "label" in df:
        flipped["label"] = 1.0 - df["label"]
    return pd.concat([df.assign(orientation=0), flipped.assign(orientation=1)], ignore_index=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``data/fights.csv`` and ``data/fighters.csv``."""
    if not FIGHTS_CSV.exists():
        raise SystemExit(f"{FIGHTS_CSV} not found: run scrape.py or import_mirror.py first")
    fighters = pd.read_csv(FIGHTERS_CSV) if FIGHTERS_CSV.exists() else None
    return pd.read_csv(FIGHTS_CSV), fighters


def main() -> None:
    fights, fighters = load_data()
    feats, builder = build_features(fights, fighters)
    out = DATA_DIR / "features.csv"
    feats.to_csv(out, index=False)
    print(f"wrote {len(feats)} rows x {len(MODEL_FEATURES)} model features to {out}")
    print(f"{len(builder.states)} fighters tracked; labelled fights: {feats['label'].notna().sum()}")


if __name__ == "__main__":
    main()
