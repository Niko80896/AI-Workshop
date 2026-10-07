"""Attach historical betting odds to ``data/fights.csv`` fights.

Source: ``ufc-master.csv`` from https://github.com/shortlikeafox/ultimate_ufc_dataset
(moneylines from 2010 on, plus "wins by KO/TKO / submission / decision" prop
odds for many fights). Only the odds columns are used: that dataset's career
stat columns are cumulative snapshots and could leak future information.

Each odds row is matched to a fight by date (+/- 1 day, for time zones) and the
pair of fighter names (accent/case-insensitive, with a fuzzy fallback), then
re-oriented to the fight's a/b order. The bookmaker margin (vig) is removed by
normalising the implied probabilities.

Output: ``data/odds.csv`` with one row per matched fight::

    fight_id, ml_a, ml_b, p_market_a,
    {a,b}_{ko,sub,dec}_odds, p_market_{a,b}_{ko,sub,dec}

Usage::

    python import_odds.py              # download from GitHub
    python import_odds.py --src FILE   # use a local ufc-master.csv
"""
from __future__ import annotations

import argparse
import difflib
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scrape import make_session
from ufc_common import DATA_DIR, FIGHTS_CSV, norm_name

log = logging.getLogger("import_odds")

SOURCE = "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv"
ODDS_CSV = DATA_DIR / "odds.csv"
METHODS = ("ko", "sub", "dec")
ODDS_COLS = (["fight_id", "ml_a", "ml_b", "p_market_a"]
             + [f"{p}_{m}_odds" for p in "ab" for m in METHODS]
             + [f"p_market_{p}_{m}" for p in "ab" for m in METHODS])


def american_to_prob(odds) -> np.ndarray:
    """Implied probability (with vig) from American odds; NaN-safe."""
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 0, 100.0 / (o + 100.0), np.where(o < 0, -o / (-o + 100.0), np.nan))


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_odds(fights: pd.DataFrame, master: pd.DataFrame, min_ratio: float = 0.85) -> pd.DataFrame:
    """Match odds rows to fights; return one aligned row per matched fight (``ODDS_COLS``)."""
    f = fights[["fight_id", "date", "fighter_a", "fighter_b"]].copy()
    f["date"] = pd.to_datetime(f["date"])
    f["na"], f["nb"] = f["fighter_a"].map(norm_name), f["fighter_b"].map(norm_name)
    by_date: dict = {}
    for r in f.itertuples(index=False):
        by_date.setdefault(r.date, []).append(r)

    m = master.copy()
    m["date"] = pd.to_datetime(m["date"])
    out, unmatched = [], 0
    for r in m.itertuples(index=False):
        nr, nbl = norm_name(r.R_fighter), norm_name(r.B_fighter)
        cands = [c for d in (-1, 0, 1) for c in by_date.get(r.date + pd.Timedelta(days=d), [])]
        best, best_score, swapped = None, 0.0, False
        for c in cands:
            for sw, (x, y) in ((False, (c.na, c.nb)), (True, (c.nb, c.na))):
                s1, s2 = _similar(nr, x), _similar(nbl, y)
                score = min(s1, s2)
                # Same card, one name near-exact: accept a nickname/variant for the other
                # ("Costas" vs "Constantinos" Philippou, "Cris Cyborg" vs Cristiane Justino).
                if max(s1, s2) >= 0.95 and min(s1, s2) >= 0.3:
                    score = max(score, min_ratio)
                if score > best_score:
                    best, best_score, swapped = c, score, sw
        if best is None or best_score < min_ratio:
            unmatched += 1
            continue
        # R corner is fighter a unless swapped.
        ra, rb = ("R", "B") if not swapped else ("B", "R")
        g = lambda name: getattr(r, name, np.nan)  # noqa: E731
        row = {"fight_id": best.fight_id,
               "ml_a": g(f"{ra}_odds"), "ml_b": g(f"{rb}_odds")}
        for p, corner in (("a", ra.lower()), ("b", rb.lower())):
            for meth in METHODS:
                row[f"{p}_{meth}_odds"] = g(f"{corner}_{meth}_odds")
        out.append(row)
    log.info("matched %d of %d odds rows (%d unmatched)", len(out), len(m), unmatched)

    o = pd.DataFrame(out).drop_duplicates("fight_id")
    pa, pb = american_to_prob(o["ml_a"]), american_to_prob(o["ml_b"])
    o["p_market_a"] = pa / (pa + pb)
    props = np.column_stack([american_to_prob(o[f"{p}_{m}_odds"]) for p in "ab" for m in METHODS])
    complete = ~np.isnan(props).any(axis=1)
    norm = np.where(complete[:, None], props / props.sum(axis=1, keepdims=True), np.nan)
    for i, (p, meth) in enumerate((p, m) for p in "ab" for m in METHODS):
        o[f"p_market_{p}_{meth}"] = norm[:, i]
    return o.reindex(columns=ODDS_COLS)


def load_odds() -> pd.DataFrame | None:
    """Read ``data/odds.csv`` if it exists."""
    return pd.read_csv(ODDS_CSV) if ODDS_CSV.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="local ufc-master.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.src:
        master = pd.read_csv(args.src, low_memory=False)
    else:
        resp = make_session().get(SOURCE, timeout=120)
        resp.raise_for_status()
        master = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    fights = pd.read_csv(FIGHTS_CSV)
    odds = match_odds(fights, master)
    odds.to_csv(ODDS_CSV, index=False)
    dates = pd.to_datetime(fights.set_index("fight_id").loc[odds["fight_id"], "date"])
    log.info("wrote %d fights with odds (%s to %s); %d with full method props",
             len(odds), dates.min().date(), dates.max().date(), odds["p_market_a_ko"].notna().sum())


if __name__ == "__main__":
    main()
