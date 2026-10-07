"""Build ``data/fights.csv`` and ``data/fighters.csv`` from a public ufcstats mirror.

Fallback for environments that cannot reach ufcstats.com directly. The mirror
(https://github.com/Greco1899/scrape_ufc_stats) republishes ufcstats data as
CSVs; this script converts it into exactly the same schema ``scrape.py`` writes.

Differences from the direct scraper that this handles:

* Stats are per round -> summed into per-fight totals (all-missing stays NaN).
* Bouts list fighter *names*, not profile URLs -> names are mapped to profile
  URLs; the handful of duplicate names (e.g. two "Bruno Silva"s) are resolved
  by picking the profile whose listed weight is closest to the bout's division.
  Fighters absent from the mirror's profile list keep an empty URL and are
  identified by name downstream.
* A few fights appear twice under renamed events -> de-duplicated by fight URL.

Fighter career summary stats are not used (they would leak future information).

Usage::

    python import_mirror.py                 # download from GitHub
    python import_mirror.py --src DIR       # use already-downloaded CSVs
"""
from __future__ import annotations

import argparse
import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scrape import make_session
from ufc_common import (
    DATA_DIR, FIGHT_COLS, FIGHTER_COLS, FIGHTERS_CSV, FIGHTS_CSV,
    clean_stance, name_resolver, parse_clock, parse_date, parse_height, parse_reach,
    parse_weight, parse_weight_class, url_id,
)

log = logging.getLogger("import_mirror")

MIRROR = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = ["ufc_event_details", "ufc_fight_details", "ufc_fight_results",
         "ufc_fight_stats", "ufc_fighter_tott"]

# Mirror per-round column -> our stem.
OF_COLS = {"SIG.STR.": "sig", "TOTAL STR.": "tot", "TD": "td", "HEAD": "head", "BODY": "body",
           "LEG": "leg", "DISTANCE": "dist", "CLINCH": "clinch", "GROUND": "ground"}
INT_COLS = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev"}

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def load_mirror(src: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the mirror CSVs from a local directory or download them from GitHub."""
    out = {}
    session = None if src else make_session()
    for name in FILES:
        if src:
            out[name] = pd.read_csv(Path(src) / f"{name}.csv")
        else:
            log.info("downloading %s", name)
            r = session.get(MIRROR + f"{name}.csv", timeout=120)
            r.raise_for_status()
            out[name] = pd.read_csv(io.StringIO(r.text))
    return out


def build_fighters(tott: pd.DataFrame) -> pd.DataFrame:
    """Convert the mirror's fighter 'tale of the tape' into ``fighters.csv`` rows."""
    df = pd.DataFrame({
        "fighter_id": tott["URL"].map(url_id),
        "fighter_url": tott["URL"].str.strip(),
        "name": _norm(tott["FIGHTER"]),
        "height_in": tott["HEIGHT"].map(parse_height),
        "weight_lbs": tott["WEIGHT"].map(parse_weight),
        "reach_in": tott["REACH"].map(parse_reach),
        "stance": tott["STANCE"].map(clean_stance),
        "dob": tott["DOB"].map(parse_date),
    })
    return df.drop_duplicates("fighter_id").reindex(columns=FIGHTER_COLS)


def _fight_totals(stats: pd.DataFrame) -> pd.DataFrame:
    """Sum per-round stats into per-fight, per-fighter totals keyed by (event, bout, fighter)."""
    s = pd.DataFrame({"event": _norm(stats["EVENT"]), "bout": _norm(stats["BOUT"]),
                      "fighter": _norm(stats["FIGHTER"])})
    for col, stem in OF_COLS.items():
        parts = stats[col].astype(str).str.extract(r"^\s*(\d+)\s+of\s+(\d+)\s*$").astype(float)
        s[f"{stem}_landed"], s[f"{stem}_att"] = parts[0], parts[1]
    for col, stem in INT_COLS.items():
        s[stem] = pd.to_numeric(stats[col], errors="coerce")
    s["ctrl_sec"] = stats["CTRL"].map(parse_clock)
    s = s[stats["FIGHTER"].notna().values]
    value_cols = [c for c in s.columns if c not in ("event", "bout", "fighter")]
    return s.groupby(["event", "bout", "fighter"], sort=False)[value_cols].sum(min_count=1).reset_index()


def build_fights(m: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the mirror tables into one ``fights.csv`` row per fight."""
    events = m["ufc_event_details"].assign(event=lambda d: _norm(d["EVENT"]))
    events["date"] = events["DATE"].map(parse_date)
    ev = events.drop_duplicates("event").set_index("event")

    res = m["ufc_fight_results"].copy()
    res["event"], res["bout"] = _norm(res["EVENT"]), _norm(res["BOUT"])
    res["fight_url"] = res["URL"].str.strip()
    res = res[res["event"].isin(ev.index)]
    res = res.drop_duplicates("fight_url", keep="first")  # renamed-event duplicates

    names = res["bout"].str.split(r"\s+vs\.\s+", n=1, expand=True, regex=True)
    res["fighter_a"], res["fighter_b"] = _norm(names[0]), _norm(names[1])
    outcome = res["OUTCOME"].astype(str).str.strip().str.split("/", expand=True)
    res["result_a"], res["result_b"] = outcome[0], outcome[1]
    wc = res["WEIGHTCLASS"].map(parse_weight_class)
    res["weight_class"] = wc.str[0]
    res["title_fight"] = wc.str[1]

    resolve = name_resolver(build_fighters(m["ufc_fighter_tott"]))
    res["fighter_a_url"] = [resolve(n, w) for n, w in zip(res["fighter_a"], res["weight_class"])]
    res["fighter_b_url"] = [resolve(n, w) for n, w in zip(res["fighter_b"], res["weight_class"])]

    out = pd.DataFrame({
        "fight_id": res["fight_url"].map(url_id), "fight_url": res["fight_url"],
        "event_name": res["event"], "event_url": res["event"].map(ev["URL"]).str.strip(),
        "date": res["event"].map(ev["date"]), "weight_class": res["weight_class"],
        "title_fight": res["title_fight"], "method": _norm(res["METHOD"]),
        "method_detail": res["DETAILS"].where(res["DETAILS"].notna()).map(
            lambda x: re.sub(r"\s+", " ", str(x)).strip() if isinstance(x, str) else None),
        "end_round": pd.to_numeric(res["ROUND"], errors="coerce"),
        "end_time": _norm(res["TIME"]), "time_format": _norm(res["TIME FORMAT"]),
        "fighter_a": res["fighter_a"], "fighter_a_url": res["fighter_a_url"],
        "fighter_b": res["fighter_b"], "fighter_b_url": res["fighter_b_url"],
        "result_a": res["result_a"], "result_b": res["result_b"],
        "_event": res["event"], "_bout": res["bout"],
    })

    totals = _fight_totals(m["ufc_fight_stats"])
    stat_cols = [c for c in totals.columns if c not in ("event", "bout", "fighter")]
    for p in ("a", "b"):
        t = totals.rename(columns={c: f"{p}_{c}" for c in stat_cols})
        out = out.merge(t, how="left", left_on=["_event", "_bout", f"fighter_{p}"],
                        right_on=["event", "bout", "fighter"]).drop(columns=["event", "bout", "fighter"])
    out = out.drop_duplicates("fight_id")
    out = out.sort_values(["date", "fight_id"]).reset_index(drop=True)
    return out.reindex(columns=FIGHT_COLS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="directory with already-downloaded mirror CSVs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    m = load_mirror(args.src)
    fights = build_fights(m)
    fighters = build_fighters(m["ufc_fighter_tott"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fights.to_csv(FIGHTS_CSV, index=False)
    fighters.to_csv(FIGHTERS_CSV, index=False)

    no_url = fights[["fighter_a_url", "fighter_b_url"]].isna().any(axis=1).sum()
    no_stats = fights["a_sig_att"].isna().sum()
    log.info("wrote %d fights (%s to %s), %d fighters", len(fights), fights["date"].min(),
             fights["date"].max(), len(fighters))
    log.info("%d fights have a fighter without a profile URL; %d fights have no stats", no_url, no_stats)


if __name__ == "__main__":
    main()
