"""Shared schema and parsing helpers used by the scraper and the mirror importer.

Both data sources (``scrape.py`` and ``import_mirror.py``) produce the exact same
``data/fights.csv`` / ``data/fighters.csv`` layout defined here, so everything
downstream (features, training, prediction) is independent of the source.
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
FIGHTS_CSV = DATA_DIR / "fights.csv"
FIGHTERS_CSV = DATA_DIR / "fighters.csv"

# Per-fighter, per-fight stat columns. Each is stored as ``a_<stat>`` and ``b_<stat>``.
LANDED_ATTEMPTED = ["sig", "tot", "td", "head", "body", "leg", "dist", "clinch", "ground"]
STAT_COLS = (
    ["kd"]
    + [f"{s}_{k}" for s in LANDED_ATTEMPTED for k in ("landed", "att")]
    + ["sub_att", "rev", "ctrl_sec"]
)

FIGHT_META_COLS = [
    "fight_id", "fight_url", "event_name", "event_url", "date", "weight_class",
    "title_fight", "method", "method_detail", "end_round", "end_time", "time_format",
    "fighter_a", "fighter_a_url", "fighter_b", "fighter_b_url", "result_a", "result_b",
]
FIGHT_COLS = FIGHT_META_COLS + [f"{p}_{c}" for p in ("a", "b") for c in STAT_COLS]
FIGHTER_COLS = ["fighter_id", "fighter_url", "name", "height_in", "reach_in", "stance", "dob"]


def is_missing(text) -> bool:
    """True for ufcstats' missing-value markers (``--``, ``---``, empty, None/NaN)."""
    if text is None:
        return True
    if isinstance(text, float) and math.isnan(text):
        return True
    t = str(text).strip()
    return t == "" or set(t) <= {"-"} or t.lower() == "nan"


def url_id(url: str | None) -> str | None:
    """Return the trailing hash id of a ufcstats URL (``.../fighter-details/abc123`` -> ``abc123``)."""
    if is_missing(url):
        return None
    return str(url).rstrip("/").rsplit("/", 1)[-1]


def parse_of(text) -> tuple[float, float]:
    """Parse ``"12 of 30"`` into ``(12.0, 30.0)``; missing values become ``(nan, nan)``."""
    if is_missing(text):
        return np.nan, np.nan
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)\s*$", str(text))
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def parse_int(text) -> float:
    """Parse an integer cell, returning NaN for ``--``/blank."""
    if is_missing(text):
        return np.nan
    try:
        return float(int(str(text).strip()))
    except ValueError:
        return np.nan


def parse_clock(text) -> float:
    """Parse ``"m:ss"`` into seconds (NaN when missing)."""
    if is_missing(text):
        return np.nan
    m = re.match(r"\s*(\d+):(\d{1,2})\s*$", str(text))
    if not m:
        return np.nan
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def parse_height(text) -> float:
    """Parse ``5' 11"`` into inches (NaN when missing)."""
    if is_missing(text):
        return np.nan
    m = re.match(r"\s*(\d+)'\s*(\d+)?", str(text))
    if not m:
        return np.nan
    return float(int(m.group(1)) * 12 + int(m.group(2) or 0))


def parse_reach(text) -> float:
    """Parse ``72"`` / ``72.0"`` into inches (NaN when missing)."""
    if is_missing(text):
        return np.nan
    m = re.match(r"\s*([\d.]+)", str(text))
    return float(m.group(1)) if m else np.nan


def parse_date(text, fmts=("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d")) -> str | None:
    """Parse ufcstats date strings (``Jul 13, 1978`` / ``July 13, 1978``) into ISO ``YYYY-MM-DD``."""
    if is_missing(text):
        return None
    t = re.sub(r"\s+", " ", str(text).strip())
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def clean_stance(text) -> str | None:
    """Normalise the stance string; missing becomes None."""
    if is_missing(text):
        return None
    return str(text).strip().title()


def parse_weight_class(text) -> tuple[str, bool]:
    """Split a bout description into ``(weight_class, is_title_fight)``.

    ``"UFC Women's Flyweight Title Bout"`` -> ``("Women's Flyweight", True)``.
    """
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    title = bool(re.search(r"title\s*bout|championship", t, re.I))
    wc = re.sub(r"(Ultimate Fighter|Road to UFC)( \d+)?", "", t, flags=re.I)
    wc = re.sub(r"\b(UFC|Interim|Title|Bout|TitleBout|Tournament|Championship|Superfight)\b", "", wc, flags=re.I)
    wc = re.sub(r"\s+", " ", wc).strip()
    return (wc or "Unknown"), title


def method_group(method) -> str:
    """Collapse ufcstats methods into KO/TKO, SUB, DEC, DQ, NC/OTHER."""
    m = str(method or "").upper()
    if "KO" in m:  # KO/TKO and "TKO - Doctor's Stoppage"
        return "KO/TKO"
    if "SUB" in m:
        return "SUB"
    if "DEC" in m:
        return "DEC"
    if "DQ" in m:
        return "DQ"
    return "OTHER"  # Overturned, Could Not Continue, Other


def round_lengths(time_format) -> list[int]:
    """Round lengths in minutes from a time-format string, e.g. ``3 Rnd (5-5-5)`` -> ``[5, 5, 5]``.

    ``No Time Limit`` (and anything unparsable) returns ``[]``.
    """
    m = re.search(r"\(([\d\-\s]+)\)", str(time_format or ""))
    if not m:
        return []
    return [int(x) for x in re.findall(r"\d+", m.group(1))]


def scheduled_rounds(time_format) -> int:
    """Number of scheduled regulation rounds (``3 Rnd + OT`` counts as 3)."""
    m = re.match(r"\s*(\d+)\s*Rnd", str(time_format or ""))
    return int(m.group(1)) if m else 1


def fight_seconds(end_round, end_time, time_format) -> float:
    """Total fight duration in seconds from the ending round/time and the round structure."""
    t = parse_clock(end_time) if isinstance(end_time, str) else end_time
    try:
        r = int(end_round)
    except (TypeError, ValueError):
        return np.nan
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return np.nan
    lengths = round_lengths(time_format)
    if not lengths:
        return float(t) if r == 1 else np.nan
    prev = lengths[: r - 1]
    if len(prev) < r - 1:
        return np.nan
    return float(sum(prev) * 60 + t)
