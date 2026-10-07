"""Tests for import_odds.py: odds conversion, vig removal and fight matching."""
import numpy as np
import pandas as pd
import pytest

from import_odds import american_to_prob, match_odds


def test_american_to_prob():
    np.testing.assert_allclose(american_to_prob([-200, 100, 300, np.nan]), [2 / 3, 0.5, 0.25, np.nan])


def test_match_reorients_and_removes_vig():
    fights = pd.DataFrame({"fight_id": ["f1", "f2", "f3"], "date": ["2020-01-01", "2020-01-01", "2020-02-01"],
                           "fighter_a": ["José Aldo", "Constantinos Philippou", "A Guy"],
                           "fighter_b": ["Max Holloway", "Tim Boetsch", "B Guy"]})
    master = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-01", "2020-03-01"],          # f1 is listed a day later (time zone)
        "R_fighter": ["Max Holloway", "Tim Boetsch", "A Guy"],          # corners swapped vs fights.csv
        "B_fighter": ["Jose Aldo", "Costas Philippou", "B Guy"],        # nickname variant; f3 date mismatch
        "R_odds": [-150, -110, -110], "B_odds": [130, -110, -110],
        "r_ko_odds": [200, np.nan, 1], "b_ko_odds": [500, 1, 1], "r_sub_odds": [1000, 1, 1],
        "b_sub_odds": [800, 1, 1], "r_dec_odds": [150, 1, 1], "b_dec_odds": [300, 1, 1]})
    o = match_odds(fights, master).set_index("fight_id")
    assert list(o.index) == ["f1", "f2"]                              # f3 not matched (wrong date)
    assert (o.loc["f1", "ml_a"], o.loc["f1", "ml_b"]) == (130, -150)  # re-oriented to fights.csv order
    pa, pb = 100 / 230, 150 / 250
    assert o.loc["f1", "p_market_a"] == pytest.approx(pa / (pa + pb))
    assert o.loc["f1", "a_ko_odds"] == 500 and o.loc["f1", "b_dec_odds"] == 150
    six = o.loc["f1", [c for c in o.columns if c.startswith("p_market_") and c != "p_market_a"]]
    assert six.sum() == pytest.approx(1.0)
    assert np.isnan(o.loc["f2", "p_market_a_ko"])                    # incomplete props -> NaN
