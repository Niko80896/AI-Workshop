"""Tests for the DWCS (Wikipedia) results parser."""
from conftest import FIXTURES
from scrape_dwcs import parse_season_page, season_url


def test_season_url():
    assert season_url(10) == "https://en.wikipedia.org/wiki/Dana_White%27s_Contender_Series_season_10"
    assert season_url(8, 1) == "https://en.wikipedia.org/wiki/Dana_White%27s_Contender_Series_8"


def test_parse_season_page():
    rows = parse_season_page((FIXTURES / "dwcs_season.html").read_text(), 8, "src")
    assert len(rows) == 5  # nested infobox table ignored
    r = rows[0]
    assert (r["date"], r["fighter_a"], r["fighter_b"]) == ("2024-08-06", "Kody Steele", "Andrew Kapel")
    assert (r["result_a"], r["result_b"], r["method"], r["end_round"], r["end_time"]) == ("W", "L", "TKO (punches)", 3, "4:07")
    assert r["event_name"] == "Dana White's Contender Series 8 Week 1" and r["fight_id"] == "dwcs-s8-1-1"
    assert rows[1]["fighter_a"] == "José Pérez" and rows[1]["fighter_b"] == "Ana Smith"
    assert rows[1]["weight_class"] == "Women's Flyweight"
    # week 2 date comes from the table's own title row
    assert {x["date"] for x in rows[2:]} == {"2024-08-13"}
    assert (rows[2]["result_a"], rows[2]["result_b"]) == ("D", "D")
    assert (rows[3]["result_a"], rows[3]["result_b"]) == ("NC", "NC")
    assert rows[4]["method"].startswith("Submission") and rows[4]["result_b"] == "L"


def test_scrape_falls_back_to_old_title(monkeypatch):
    import requests
    from scrape_dwcs import scrape_dwcs

    class F:
        def get(self, url):
            if "season_" in url or not url.endswith("_8"):
                resp = requests.Response()
                resp.status_code = 404
                raise requests.HTTPError(response=resp)
            return (FIXTURES / "dwcs_season.html").read_text()

    df = scrape_dwcs(F(), seasons=[8, 9])
    assert len(df) == 5 and df["source_url"].str.endswith("_8").all()
