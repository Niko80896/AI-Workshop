"""Tests for line-score parsing and the runs/totals model outputs."""
import numpy as np
import pandas as pd
import pytest

import config as C
from data import add_inning_splits, parse_line


def test_parse_line():
    assert parse_line("00(10)0x") == [0, 0, 10, 0, None]
    assert parse_line("120000001") == [1, 2, 0, 0, 0, 0, 0, 0, 1]


def test_inning_splits():
    g = pd.DataFrame({"vis_line": ["300100000"], "home_line": ["01000020x"]})
    g = add_inning_splits(g)
    assert (g.vis_r1[0], g.vis_f3[0], g.vis_f5[0]) == (3, 3, 4)
    assert (g.home_r1[0], g.home_f3[0], g.home_f5[0]) == (0, 1, 1)


@pytest.mark.skipif(not (C.MODEL_DIR / "runs_production.pkl").exists(), reason="run runs.py first")
def test_game_outputs_are_consistent():
    import runs
    m = runs.load("production")
    base = {c: 0.0 for c in runs.X_COLS}
    base.update(lineup_woba=.315, lineup_obp=.318, lineup_iso=.16, lineup_k=.22, opp_sp_fip=4.1,
                opp_sp_xfip=4.1, opp_sp_whip=1.28, opp_sp_kbb=.14, opp_sp_depth=5.3, opp_pen_top=3.6,
                opp_pen_all=4.0, opp_der=.70, lg_rpg=4.5, is_postseason=0)
    o = runs.game_outputs(m, {**base, "is_home": 1}, {**base, "is_home": 0}, n=20000)
    assert sum(o["f5"].values()) == pytest.approx(1.0)
    assert 0.40 < o["nrfi"] < 0.62                     # MLB games are ~50% NRFI
    assert 3.5 < o["mean_total"]["F5"] < 6.0 and 7 < o["mean_total"]["Full"] < 11
    assert o["mean_total"]["F3"] < o["mean_total"]["F5"] < o["mean_total"]["Full"]
    for (k, line), (po, pu) in o["totals"].items():
        assert po + pu <= 1.0 + 1e-9
    # Equal teams: the home side should be a slight favorite, not a big one.
    assert 0.48 < o["full_home_win"] < 0.58
