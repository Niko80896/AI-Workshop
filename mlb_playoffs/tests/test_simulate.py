"""Series-format, rotation and lock-in tests for the simulator."""
import numpy as np
import pandas as pd
import pytest

import config as C
from simulate import HOME_PATTERN, PlayoffSimulator

pytestmark = pytest.mark.skipif(not (C.MODEL_DIR / "through_2025_regular.pkl").exists(),
                                reason="run data.py, features.py and model.py first")


class ConstModel:
    """Stand-in model: the home team wins with fixed probability."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.full(len(X), self.p)


@pytest.fixture(scope="module")
def sim():
    return PlayoffSimulator(2025)


def _fresh(sim, p):
    sim.model = ConstModel(p)
    sim._cache = {}
    return sim


@pytest.mark.parametrize("rnd,games", [("WC", 3), ("DS", 5), ("LCS", 7), ("WS", 7)])
def test_home_patterns(rnd, games):
    pat = HOME_PATTERN[rnd]
    assert len(pat) == games
    if rnd == "WC":
        assert all(pat)                       # higher seed hosts every game
    elif rnd == "DS":
        assert pat == [True, True, False, False, True]          # 2-2-1
    else:
        assert pat == [True, True, False, False, False, True, True]  # 2-3-2


@pytest.mark.parametrize("rnd", ["WC", "DS", "LCS", "WS"])
def test_home_team_always_wins_means_home_field_team_wins(sim, rnd):
    """If the home team always wins, the side with more home games must win."""
    _fresh(sim, 1.0)
    rng = np.random.default_rng(0)
    assert sim.sim_series("SEA", "DET", rnd, dict(sim.last_pitched), rng, lock=False) == "SEA"


def test_coin_flip_series_is_even(sim):
    _fresh(sim, 0.5)
    rng = np.random.default_rng(1)
    wins = sum(sim.sim_series("SEA", "DET", "WS", dict(sim.last_pitched), rng, lock=False) == "SEA"
               for _ in range(4000))
    assert abs(wins / 4000 - 0.5) < 0.03


def test_bracket_probabilities_are_consistent(sim):
    _fresh(sim, 0.54)
    odds = sim.run(400)
    assert odds["win_ws"].sum() == pytest.approx(1.0)
    assert odds["pennant"].sum() == pytest.approx(2.0)
    assert odds["make_lcs"].sum() == pytest.approx(4.0)
    assert odds["make_ds"].sum() == pytest.approx(8.0)
    byes = odds[odds.seed <= 2]
    assert (byes["make_ds"] == 1.0).all()


def test_ace_pitches_game_one_and_returns_on_rest(sim):
    """DS calendar 10/4,10/5,10/7,10/8,10/10: ace starts G1, misses G4 (3 days rest), starts G5."""
    _fresh(sim, 0.5)
    team = "SEA"
    ace = sim.rotation_for(team)[0]
    days = sim.calendar[("DS", "AL")]
    last = dict(sim.last_pitched)
    starters = []
    for i, d in enumerate(days):
        sp, _ = sim.pick_starter(team, "DS", i + 1, d, last)
        last[sp] = d
        starters.append(sp)
    assert starters[0] == ace
    assert starters[3] != ace
    assert starters[4] == ace
    assert len(set(starters[:4])) == 4


def test_completed_games_are_locked_in():
    s = PlayoffSimulator(2025, as_of="2025-10-27")
    s.model = ConstModel(0.5)
    odds = s.run(300).set_index("code")
    # By Oct 27 only the Dodgers and Blue Jays are alive.
    assert odds.loc[["LAN", "TOR"], "pennant"].tolist() == [1.0, 1.0]
    assert odds["win_ws"].drop(["LAN", "TOR"]).sum() == 0
    played = s.played("WS", "LAN", "TOR")
    assert len(played) == 2
    assert pd.Timestamp(played[0]["date"]) == pd.Timestamp("2025-10-24")


def test_probable_override(sim):
    sim.probables[("WS", 1, "LAN")] = "ohtas001"
    sp, _ = sim.pick_starter("LAN", "WS", 1, sim.day0 + 20, dict(sim.last_pitched))
    del sim.probables[("WS", 1, "LAN")]
    assert sp == "ohtas001"
