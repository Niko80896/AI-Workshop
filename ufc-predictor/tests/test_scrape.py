"""Parser tests for scrape.py against saved ufcstats-style HTML fixtures."""
import math

import pandas as pd
import pytest

import scrape
from conftest import FIXTURES
from ufc_common import FIGHT_COLS, fight_seconds, method_group, parse_weight_class


def read(name):
    return (FIXTURES / name).read_text()


def test_event_list():
    events = scrape.parse_event_list(read("event_list.html"))
    assert [e["name"] for e in events] == ["UFC 333: Future vs. Event", "UFC 332: Silva vs. Wang", "UFC 2: No Way Out"]
    assert events[1] == {"name": "UFC 332: Silva vs. Wang",
                         "url": "http://ufcstats.com/event-details/ad3fdba28a7540cf", "date": "2026-10-03"}


def test_event_page():
    ev = scrape.parse_event_page(read("event.html"))
    assert ev["name"] == "UFC 332: Silva vs. Wang"
    assert ev["date"] == "2026-10-03"
    assert ev["fight_urls"] == ["http://ufcstats.com/fight-details/3f804eec9183e597",
                                "http://ufcstats.com/fight-details/514366f770553cc9"]


def test_fight_page_full():
    r = scrape.parse_fight_page(read("fight.html"), "http://ufcstats.com/fight-details/3f804eec9183e597")
    assert r["fight_id"] == "3f804eec9183e597"
    assert (r["fighter_a"], r["fighter_b"]) == ("Natalia Silva", "Wang Cong")
    assert r["fighter_a_url"].endswith("aaaa1111")
    assert (r["result_a"], r["result_b"]) == ("W", "L")
    assert r["weight_class"] == "Women's Flyweight" and r["title_fight"] is True
    assert r["method"] == "Decision - Unanimous"
    assert r["end_round"] == 5 and r["end_time"] == "5:00" and r["time_format"] == "5 Rnd (5-5-5-5-5)"
    assert r["method_detail"].startswith("Sal D'amato")
    # totals table
    assert (r["a_kd"], r["b_kd"]) == (1, 0)
    assert (r["a_sig_landed"], r["a_sig_att"], r["b_sig_landed"], r["b_sig_att"]) == (110, 260, 95, 240)
    assert (r["a_tot_landed"], r["b_tot_att"]) == (130, 250)
    assert (r["a_td_landed"], r["a_td_att"]) == (3, 7)
    assert (r["a_sub_att"], r["b_rev"]) == (1, 1)
    assert r["a_ctrl_sec"] == 372
    assert math.isnan(r["b_ctrl_sec"])  # "--" must be NaN, not 0
    # sig-strike table lists fighters in swapped order; must be re-aligned by profile link
    assert (r["a_head_landed"], r["a_head_att"], r["b_head_landed"]) == (70, 200, 60)
    assert (r["a_ground_landed"], r["b_ground_landed"], r["b_ground_att"]) == (8, 0, 0)
    assert set(FIGHT_COLS) - set(r) == {"event_name", "event_url", "date"}


def test_fight_page_without_stats_is_nan():
    r = scrape.parse_fight_page(read("fight_nostats.html"), "http://ufcstats.com/fight-details/x1")
    assert r["weight_class"] == "Open Weight" and r["title_fight"] is False
    assert r["method"] == "Submission" and r["method_detail"] == "Lapel Choke"
    assert r["time_format"] == "No Time Limit"
    for c in ("a_kd", "a_sig_landed", "b_td_att", "a_ctrl_sec", "b_leg_landed"):
        assert math.isnan(r[c]), c


def test_fighter_page_ignores_career_stats():
    f = scrape.parse_fighter_page(read("fighter.html"), "http://ufcstats.com/fighter-details/aaaa1111")
    assert f == {"fighter_id": "aaaa1111", "fighter_url": "http://ufcstats.com/fighter-details/aaaa1111",
                 "name": "Natalia Silva", "height_in": 64.0, "weight_lbs": 125.0, "reach_in": 65.0,
                 "stance": "Orthodox", "dob": "1997-07-03"}


@pytest.mark.parametrize("text,expected", [
    ("UFC Women's Flyweight Title Bout", ("Women's Flyweight", True)),
    ("UFC Interim Heavyweight Title Bout", ("Heavyweight", True)),
    ("Light Heavyweight Bout", ("Light Heavyweight", False)),
    ("Ultimate Fighter 33 Welterweight Tournament Title Bout", ("Welterweight", True)),
    ("Road to UFC 3 Women's Strawweight Tournament TitleBout", ("Women's Strawweight", True)),
    ("UFC Superfight Championship Bout", ("Unknown", True)),
])
def test_weight_class(text, expected):
    assert parse_weight_class(text) == expected


def test_duration_and_method():
    assert fight_seconds(3, "2:30", "3 Rnd (5-5-5)") == 750
    assert fight_seconds(2, "1:00", "1 Rnd + OT (12-3)") == 780
    assert fight_seconds(1, "5:08", "No Time Limit") == 308
    assert method_group("TKO - Doctor's Stoppage") == "KO/TKO"
    assert method_group("Decision - Split") == "DEC"
    assert method_group("Overturned") == "OTHER"


class FakeFetcher:
    """Serves fixtures by URL and records every request."""

    def __init__(self):
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        if "events/completed" in url:
            return read("event_list.html")
        if "event-details" in url:
            return read("event.html")
        if "fight-details" in url:
            return read("fight.html")
        if "fighter-details" in url:
            return read("fighter.html")
        raise KeyError(url)


def test_scrape_is_incremental(tmp_path, monkeypatch):
    monkeypatch.setattr(scrape, "DATA_DIR", tmp_path)
    monkeypatch.setattr(scrape, "FIGHTS_CSV", tmp_path / "fights.csv")
    monkeypatch.setattr(scrape, "FIGHTERS_CSV", tmp_path / "fighters.csv")
    f1 = FakeFetcher()
    scrape.scrape(f1, today="2026-10-07")
    fights = pd.read_csv(tmp_path / "fights.csv")
    # upcoming event skipped; 2 completed events x 2 fight links each
    assert len(fights) == 4 and list(fights.columns) == FIGHT_COLS
    assert set(fights["date"]) == {"2026-10-03"}  # date comes from the event page fixture
    assert len(pd.read_csv(tmp_path / "fighters.csv")) == 2

    f2 = FakeFetcher()
    scrape.scrape(f2, today="2026-10-07")
    assert f2.calls == [scrape.EVENTS_URL]  # nothing new: only the listing is fetched
    assert len(pd.read_csv(tmp_path / "fights.csv")) == 4
