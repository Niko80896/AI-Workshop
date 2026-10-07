"""Leakage tests: a game's features may only depend on games from earlier days.

Strategy: build features on real data, then scramble every result and stat
line on or after a cutoff date and rebuild. Features for games on the cutoff
date (and earlier) must be bit-for-bit identical; later ones should change.
"""
import numpy as np
import pandas as pd
import pytest

import config as C
import features as F

pytestmark = pytest.mark.skipif(not (C.DATA_DIR / "games.csv").exists(),
                                reason="run `python data.py` first")


@pytest.fixture(scope="module")
def season_data():
    games, pit, bat, team, players = F.load_inputs()
    keep = games["season"] == 2019
    ids = set(games.loc[keep, "game_id"])
    return (games[keep].reset_index(drop=True), pit[pit.game_id.isin(ids)], bat[bat.game_id.isin(ids)],
            team[team.game_id.isin(ids)], players)


def _scramble(df, cutoff, cols, rng):
    df = df.copy()
    mask = df["date"] >= cutoff
    for c in cols:
        if c in df.columns:
            df.loc[mask, c] = rng.integers(0, 15, mask.sum())
    return df


def test_future_data_does_not_change_features(season_data):
    games, pit, bat, team, players = season_data
    # Pick a date with a doubleheader so same-day games are covered too.
    per_day = games.groupby("date").size()
    dh_days = games[games["game_id"].str.endswith(("1", "2"))]["date"].unique()
    cutoff = sorted(d for d in dh_days if d > pd.Timestamp("2019-06-01"))[0]
    assert per_day[cutoff] > 1

    base = F.build_features(games, pit, bat, team, players).set_index("game_id")

    rng = np.random.default_rng(0)
    g2 = _scramble(games, cutoff, ["home_score", "vis_score"], rng)
    g2["home_win"] = (g2["home_score"] > g2["vis_score"]).astype(int)
    stat_cols = ["BF", "outs", "K", "BB", "HR", "H", "1B", "2B", "pitches", "er", "PA", "AB", "bip", "sb"]
    pit2, bat2, team2 = (_scramble(d, cutoff, stat_cols, rng) for d in (pit, bat, team))
    leaked = F.build_features(g2, pit2, bat2, team2, players).set_index("game_id")

    cols = F.FEATURES + ["home_elo", "away_elo", "home_pyth", "away_pyth"]
    upto = base[base["date"] <= cutoff].index
    pd.testing.assert_frame_equal(base.loc[upto, cols], leaked.loc[upto, cols])
    after = base[base["date"] > cutoff].index
    assert not np.allclose(base.loc[after, cols].values, leaked.loc[after, cols].values)


def test_state_until_matches_snapshot(season_data):
    """run_state(until=D) must give exactly the pre-game features of day D."""
    games, pit, bat, team, players = season_data
    d = pd.Timestamp("2019-08-15")
    base = F.build_features(games, pit, bat, team, players)
    st = F.run_state(games, pit, bat, team, players, until=d)
    for r in base[base["date"] == d].to_dict("records"):
        f, _ = st.matchup(r["hometeam"], r["visteam"], r["home_sp"], r["vis_sp"], d.toordinal())
        for k in F.FEATURES:
            assert f[k] == pytest.approx(r[k]), k


def test_own_result_not_in_features(season_data):
    games, pit, bat, team, players = season_data
    base = F.build_features(games, pit, bat, team, players).set_index("game_id")
    gid = games.iloc[1500]["game_id"]
    g2 = games.copy()
    g2.loc[g2.game_id == gid, ["home_score", "vis_score"]] = [0, 25]
    g2.loc[g2.game_id == gid, "home_win"] = 0
    flipped = F.build_features(g2, pit, bat, team, players).set_index("game_id")
    assert base.loc[gid, F.FEATURES].equals(flipped.loc[gid, F.FEATURES])
