"""Scrape completed UFC events, fights and fighter profiles from ufcstats.com.

Outputs ``data/fights.csv`` (one row per fight, per-fighter totals as ``a_*``/``b_*``
columns) and ``data/fighters.csv`` (height, reach, stance, DOB only).

The scraper is incremental: events already present in ``fights.csv`` are skipped
and only fighters missing from ``fighters.csv`` are fetched. Progress is written
after every event, so an interrupted run can simply be restarted.

Career summary stats on fighter pages (SLpM, Str. Acc., ...) are deliberately
ignored: they describe the fighter's *current* career and would leak future
information into features for past fights.

Usage::

    python scrape.py                 # fetch everything new
    python scrape.py --max-events 5  # smoke test on the 5 oldest missing events
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ufc_common import (
    DATA_DIR, FIGHT_COLS, FIGHTER_COLS, FIGHTERS_CSV, FIGHTS_CSV, clean_stance,
    is_missing, parse_clock, parse_date, parse_height, parse_int, parse_of,
    parse_reach, parse_weight, parse_weight_class, url_id, STAT_COLS,
)

log = logging.getLogger("scrape")

BASE = "http://ufcstats.com"
EVENTS_URL = f"{BASE}/statistics/events/completed?page=all"
USER_AGENT = "ufc-predictor research scraper (polite; contact via repo)"

# Header text in the fight-page tables -> our column stem.
TOTALS_HEADERS = {
    "kd": "kd", "sig. str.": "sig", "total str.": "tot", "td": "td",
    "sub. att": "sub_att", "rev.": "rev", "ctrl": "ctrl_sec",
}
SIG_HEADERS = {
    "head": "head", "body": "body", "leg": "leg",
    "distance": "dist", "clinch": "clinch", "ground": "ground",
}


# --------------------------------------------------------------------------- HTTP
def make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """A ``requests`` session with retry/backoff on connection errors and 429/5xx."""
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers["User-Agent"] = USER_AGENT
    return s


class Fetcher:
    """Fetch pages with a shared session and a fixed delay between requests."""

    def __init__(self, session: requests.Session | None = None, delay: float = 0.75):
        self.session = session or make_session()
        self.delay = delay
        self._last = 0.0

    def get(self, url: str) -> str:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=30)
        self._last = time.monotonic()
        resp.raise_for_status()
        return resp.text


# ------------------------------------------------------------------------ parsers
def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip() if el is not None else ""


def parse_event_list(html: str) -> list[dict]:
    """Parse the completed-events listing into ``[{name, url, date}]`` (newest first)."""
    soup = BeautifulSoup(html, "lxml")
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        a = row.select_one("a.b-link")
        if a is None or "event-details" not in a.get("href", ""):
            continue
        d = parse_date(_text(row.select_one("span.b-statistics__date")))
        events.append({"name": _text(a), "url": a["href"].strip(), "date": d})
    return events


def parse_event_page(html: str) -> dict:
    """Parse an event page into ``{name, date, fight_urls}`` (fight order as listed)."""
    soup = BeautifulSoup(html, "lxml")
    name = _text(soup.select_one("span.b-content__title-highlight"))
    ev_date = None
    for li in soup.select("li.b-list__box-list-item"):
        t = _text(li)
        if t.lower().startswith("date:"):
            ev_date = parse_date(t.split(":", 1)[1])
    urls = [tr["data-link"].strip() for tr in soup.select("tr.b-fight-details__table-row[data-link]")]
    return {"name": name, "date": ev_date, "fight_urls": urls}


def _stat_table(soup, required_header: str):
    """Return (headers, [cells]) of the fight-total table whose header contains ``required_header``.

    Per-round tables carry the ``js-fight-table`` class and are skipped.
    """
    for table in soup.find_all("table"):
        if "js-fight-table" in (table.get("class") or []):
            continue
        headers = [_text(th).lower() for th in table.select("thead th")]
        if required_header not in headers:
            continue
        row = table.select_one("tbody tr")
        if row is None:
            return headers, None
        return headers, row.find_all("td", recursive=False)
    return None, None


def _pair(cell) -> tuple[str, str]:
    ps = cell.find_all("p")
    vals = [_text(p) for p in ps] + ["", ""]
    return vals[0], vals[1]


def _empty_stats() -> dict:
    return {f"{p}_{c}": np.nan for p in ("a", "b") for c in STAT_COLS}


def parse_fight_page(html: str, fight_url: str) -> dict:
    """Parse a fight-details page into one ``fights.csv`` row (without event fields).

    Missing stats (``--`` or absent tables, common for early UFC events) become NaN.
    Table rows are matched to fighters by profile link, so the a/b orientation
    always follows the order of the fighters at the top of the page.
    """
    soup = BeautifulSoup(html, "lxml")
    persons = soup.select("div.b-fight-details__person")
    if len(persons) != 2:
        raise ValueError(f"expected 2 fighters on {fight_url}, found {len(persons)}")
    row: dict = {"fight_id": url_id(fight_url), "fight_url": fight_url}
    hrefs = []
    for p, person in zip(("a", "b"), persons):
        link = person.select_one("a.b-fight-details__person-link") or person.select_one("h3 a")
        name = _text(link) if link is not None else _text(person.select_one("h3"))
        href = link["href"].strip() if link is not None and link.get("href") else None
        row[f"fighter_{p}"] = name
        row[f"fighter_{p}_url"] = href
        row[f"result_{p}"] = _text(person.select_one("i.b-fight-details__person-status")) or None
        hrefs.append(href)

    title_el = soup.select_one("i.b-fight-details__fight-title")
    wc, title = parse_weight_class(_text(title_el))
    belt = title_el is not None and any("belt" in (img.get("src") or "") for img in title_el.find_all("img"))
    row["weight_class"], row["title_fight"] = wc, bool(title or belt)

    details = {}
    for item in soup.select("p.b-fight-details__text i.b-fight-details__text-item, "
                            "p.b-fight-details__text i.b-fight-details__text-item_first"):
        label = _text(item.select_one("i.b-fight-details__label")).rstrip(":").lower()
        value = _text(item)
        value = value.split(":", 1)[1].strip() if ":" in value and label else value
        details[label] = value
    row["method"] = details.get("method") or None
    row["end_round"] = parse_int(details.get("round"))
    row["end_time"] = details.get("time") or None
    row["time_format"] = details.get("time format") or None
    detail_p = [p for p in soup.select("p.b-fight-details__text")
                if _text(p.select_one("i.b-fight-details__label")).lower().startswith("details")]
    row["method_detail"] = (_text(detail_p[0]).split(":", 1)[-1].strip() or None) if detail_p else None

    row.update(_empty_stats())
    for required, mapping in (("kd", TOTALS_HEADERS), ("head", SIG_HEADERS)):
        headers, cells = _stat_table(soup, required)
        if not cells:
            continue
        # Figure out which <p> in each cell belongs to fighter a (match by profile link).
        first = cells[0].find_all("p")
        links = [(p.find("a") or {}).get("href", "").strip() if p.find("a") else "" for p in first]
        swap = len(links) == 2 and links[0] == hrefs[1] and links[1] == hrefs[0] and hrefs[0] != hrefs[1]
        for h, cell in zip(headers, cells):
            stem = mapping.get(h)
            if stem is None:
                continue
            va, vb = _pair(cell)
            if swap:
                va, vb = vb, va
            for p, v in (("a", va), ("b", vb)):
                if stem in ("kd", "sub_att", "rev"):
                    row[f"{p}_{stem}"] = parse_int(v)
                elif stem == "ctrl_sec":
                    row[f"{p}_{stem}"] = parse_clock(v)
                else:
                    row[f"{p}_{stem}_landed"], row[f"{p}_{stem}_att"] = parse_of(v)
    return row


def parse_fighter_page(html: str, fighter_url: str) -> dict:
    """Parse height, weight, reach, stance and DOB from a fighter page (career stats ignored)."""
    soup = BeautifulSoup(html, "lxml")
    info = {}
    for li in soup.select("li.b-list__box-list-item"):
        t = _text(li)
        if ":" in t:
            k, v = t.split(":", 1)
            info.setdefault(k.strip().lower(), v.strip())
    return {
        "fighter_id": url_id(fighter_url),
        "fighter_url": fighter_url,
        "name": _text(soup.select_one("span.b-content__title-highlight")),
        "height_in": parse_height(info.get("height")),
        "weight_lbs": parse_weight(info.get("weight")),
        "reach_in": parse_reach(info.get("reach")),
        "stance": clean_stance(info.get("stance")),
        "dob": parse_date(info.get("dob")),
    }


# ------------------------------------------------------------------------- driver
def _load(path, cols) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)


def _append(path, rows: list[dict], cols) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=cols)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def scrape(fetcher: Fetcher | None = None, max_events: int | None = None,
           today: str | None = None) -> None:
    """Incrementally scrape new completed events and any missing fighter profiles."""
    fetcher = fetcher or Fetcher()
    today = today or date.today().isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    done = set(_load(FIGHTS_CSV, FIGHT_COLS)["event_url"].dropna())
    events = [e for e in parse_event_list(fetcher.get(EVENTS_URL))
              if e["date"] and e["date"] < today and e["url"] not in done]
    events.sort(key=lambda e: e["date"])  # oldest first
    if max_events is not None:
        events = events[:max_events]
    log.info("%d new events to scrape", len(events))

    for i, ev in enumerate(events, 1):
        info = parse_event_page(fetcher.get(ev["url"]))
        rows = []
        for furl in info["fight_urls"]:
            try:
                row = parse_fight_page(fetcher.get(furl), furl)
            except Exception as exc:  # keep going; one bad page shouldn't kill the run
                log.warning("skipping fight %s: %s", furl, exc)
                continue
            row.update(event_name=info["name"] or ev["name"], event_url=ev["url"],
                       date=info["date"] or ev["date"])
            rows.append(row)
        _append(FIGHTS_CSV, rows, FIGHT_COLS)
        log.info("[%d/%d] %s %s: %d fights", i, len(events), ev["date"], ev["name"], len(rows))

    fights = _load(FIGHTS_CSV, FIGHT_COLS)
    have = set(_load(FIGHTERS_CSV, FIGHTER_COLS)["fighter_url"].dropna())
    urls = pd.concat([fights["fighter_a_url"], fights["fighter_b_url"]]).dropna().unique()
    missing = [u for u in urls if u not in have and not is_missing(u)]
    log.info("%d new fighter profiles to scrape", len(missing))
    batch = []
    for i, u in enumerate(missing, 1):
        try:
            batch.append(parse_fighter_page(fetcher.get(u), u))
        except Exception as exc:
            log.warning("skipping fighter %s: %s", u, exc)
        if len(batch) >= 50 or i == len(missing):
            _append(FIGHTERS_CSV, batch, FIGHTER_COLS)
            batch = []
            log.info("fighters %d/%d", i, len(missing))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-events", type=int, default=None, help="only scrape this many new events")
    ap.add_argument("--delay", type=float, default=0.75, help="seconds between requests")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scrape(Fetcher(delay=args.delay), max_events=args.max_events)


if __name__ == "__main__":
    main()
