"""Scrape Dana White's Contender Series (DWCS) results from Wikipedia.

ufcstats.com does not cover DWCS, and no public source publishes DWCS
strike/takedown stats, so this collects *results only*: date, weight class,
both fighters, winner, method, ending round and time. ``features.py`` uses these
bouts as extra history (Elo, form, streak, layoff, finishes, DWCS record) but
never in per-minute striking/grappling rates.

Each season page (``Dana White's Contender Series 1`` ... ``N``) is split into
weekly sections, each with a standard Wikipedia MMA results table
(``Weight class | winner | def. | loser | Method | Round | Time | Notes``).
The bout date is taken from the nearest date text above each table.

Output: ``data/dwcs_fights.csv``. Re-running refetches all seasons (a few pages)
and rewrites the file.

Usage::

    python scrape_dwcs.py                    # seasons 1..15 (missing pages are skipped)
    python scrape_dwcs.py --html-dir pages/  # parse saved season pages (*.html) offline
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scrape import Fetcher
from ufc_common import DATA_DIR, DWCS_COLS, DWCS_CSV, parse_date

log = logging.getLogger("scrape_dwcs")

WIKI = "https://en.wikipedia.org/wiki/"
PAGE_TITLE = "Dana White's Contender Series {n}"
DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})")
DATE_RE_DMY = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
    r",?\s+(\d{4})")


def season_url(n: int) -> str:
    return WIKI + quote(PAGE_TITLE.format(n=n).replace(" ", "_"))


def _clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)          # footnote markers like [12]
    text = re.sub(r"\s*\((?:c|ic)\)\s*$", "", text)  # champion markers
    return re.sub(r"\s+", " ", text).strip()


def _find_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if m:
        return parse_date(f"{m.group(1)} {int(m.group(2))}, {m.group(3)}")
    m = DATE_RE_DMY.search(text)
    if m:
        return parse_date(f"{m.group(2)} {int(m.group(1))}, {m.group(3)}")
    return None


def _is_results_table(table) -> bool:
    head = " ".join(th.get_text(" ", strip=True).lower() for th in table.find_all("th"))
    return "method" in head and "round" in head and "time" in head


def _parse_results_table(table) -> list[dict]:
    out = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 7:
            continue
        cells = [_clean(td.get_text(" ", strip=True)) for td in tds]
        weight, f1, sep, f2, method, rnd, time_ = cells[:7]
        if not f1 or not f2:
            continue
        sep_l = sep.lower()
        meth_l = method.lower()
        if "def" in sep_l:
            res_a, res_b = "W", "L"
        elif "draw" in meth_l:
            res_a = res_b = "D"
        else:  # "vs." with a no contest / overturned result
            res_a = res_b = "NC"
        if "no contest" in meth_l or "overturned" in meth_l:
            res_a = res_b = "NC"
        out.append({"weight_class": weight, "fighter_a": f1, "fighter_b": f2,
                    "result_a": res_a, "result_b": res_b, "method": method,
                    "end_round": pd.to_numeric(rnd, errors="coerce"), "end_time": time_})
    return out


def parse_season_page(html: str, season: int, source_url: str = "") -> list[dict]:
    """Parse one DWCS season page into result rows (one per bout).

    Walks the page in document order, remembering the most recent date seen in
    headings / paragraphs / captions, and attributes each results table to it.
    """
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("div.mw-parser-output") or soup
    rows, current_date, week, table_i = [], None, None, 0
    for el in root.find_all(["h2", "h3", "h4", "p", "caption", "table", "li", "dd", "b"]):
        if el.name == "table":
            if el.find_parent("table") is not None or not _is_results_table(el):
                continue
            bouts = _parse_results_table(el)
            if not bouts:
                continue
            table_i += 1
            # A date inside the table's own title row (common on Wikipedia) wins.
            first_th = el.find("th")
            own_date = _find_date(first_th.get_text(" ", strip=True)) if first_th else None
            ev_date = own_date or current_date
            label = week or f"Card {table_i}"
            for j, r in enumerate(bouts, 1):
                r.update(date=ev_date,
                         event_name=f"Dana White's Contender Series {season} {label}",
                         time_format="3 Rnd (5-5-5)",
                         fight_id=f"dwcs-s{season}-{table_i}-{j}",
                         source_url=source_url)
                rows.append(r)
            continue
        if el.find_parent("table") is not None:
            continue
        text = el.get_text(" ", strip=True)
        if el.name in ("h2", "h3", "h4"):
            wk = re.search(r"(Week\s*\d+|Episode\s*\d+)", text, re.I)
            if wk:
                week = wk.group(1).title()
        d = _find_date(text)
        if d:
            current_date = d
    return rows


def scrape_dwcs(fetcher: Fetcher | None = None, seasons=range(1, 16),
                html_dir: Path | None = None) -> pd.DataFrame:
    """Collect all DWCS results into a DataFrame with ``DWCS_COLS`` columns."""
    rows = []
    if html_dir is not None:
        for i, path in enumerate(sorted(Path(html_dir).glob("*.html")), 1):
            m = re.search(r"(\d+)", path.stem)
            rows += parse_season_page(path.read_text(), int(m.group(1)) if m else i, str(path))
    else:
        fetcher = fetcher or Fetcher(delay=1.0)
        for n in seasons:
            url = season_url(n)
            try:
                html = fetcher.get(url)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    log.info("season %d: no page, skipping", n)
                    continue
                raise
            season_rows = parse_season_page(html, n, url)
            log.info("season %d: %d bouts", n, len(season_rows))
            rows += season_rows
    df = pd.DataFrame(rows).reindex(columns=DWCS_COLS)
    undated = df["date"].isna()
    if undated.any():
        log.warning("dropping %d bouts with no parsable date", int(undated.sum()))
        df = df[~undated]
    return df.sort_values(["date", "fight_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-season", type=int, default=15, help="highest season number to try")
    ap.add_argument("--html-dir", type=Path, default=None, help="parse saved season pages instead of fetching")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = scrape_dwcs(seasons=range(1, args.max_season + 1), html_dir=args.html_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DWCS_CSV, index=False)
    log.info("wrote %d DWCS bouts (%s to %s) to %s", len(df), df["date"].min(), df["date"].max(), DWCS_CSV)


if __name__ == "__main__":
    main()
