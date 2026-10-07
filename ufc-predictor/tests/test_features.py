"""Leakage and correctness tests for features.py."""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from features import (ANTISYMMETRIC, CONTEXT_FEATURES, MODEL_FEATURES, SUM_FEATURES, FeatureBuilder,
                      build_features, prepare_fights, symmetrize)
from synthetic import make_data, make_dwcs, scramble_from

FEATURE_COLS = [c for c in build_features(*make_data(n_dates=3))[0].columns
                if c in MODEL_FEATURES or c[:2] in ("a_", "b_")]


def _assert_same(x: pd.DataFrame, y: pd.DataFrame):
    pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cut_idx", [5, 20, 45])
def test_no_leakage_future_fights_do_not_change_features(cut_idx):
    """Scrambling every fight on/after a date must not change features up to and including that date."""
    fights, fighters = make_data()
    cutoff = sorted(fights["date"].unique())[cut_idx]
    base, _ = build_features(fights, fighters)
    scr, _ = build_features(scramble_from(fights, cutoff), fighters)
    upto = base["date"] <= pd.Timestamp(cutoff)  # includes the cutoff date: own result must not leak
    _assert_same(base.loc[upto, FEATURE_COLS], scr.loc[upto, FEATURE_COLS])
    # sanity: the scramble really did change later features
    later = base["date"] > pd.Timestamp(cutoff)
    assert not base.loc[later, "diff_elo"].equals(scr.loc[later, "diff_elo"])


def test_no_leakage_on_real_data():
    """Same check on the real dataset (skipped when data/fights.csv is absent)."""
    path = ROOT / "data" / "fights.csv"
    if not path.exists():
        pytest.skip("no real data")
    fights = pd.read_csv(path)
    fighters = pd.read_csv(ROOT / "data" / "fighters.csv")
    fights = fights[fights["date"] >= "2019-01-01"]  # keep the test quick
    cutoff = "2022-06-11"
    base, _ = build_features(fights, fighters)
    scr, _ = build_features(scramble_from(fights, cutoff), fighters)
    upto = base["date"] <= pd.Timestamp(cutoff)
    assert upto.sum() > 1000
    _assert_same(base.loc[upto, FEATURE_COLS], scr.loc[upto, FEATURE_COLS])


def test_prediction_state_matches_training_rows():
    """Features for a 'pending' fight built from history-only state equal the training row."""
    fights, fighters = make_data()
    full, _ = build_features(fights, fighters)
    target_date = sorted(fights["date"].unique())[30]
    _, builder = build_features(fights[fights["date"] < target_date], fighters)
    prep = prepare_fights(fights)
    for idx in prep.index[prep["date"] == pd.Timestamp(target_date)]:
        r = prep.loc[idx]
        row = builder.matchup(r.key_a, r.key_b, r.date, r.five_round, r.title_fight, r.weight_class)
        expected = full.loc[idx, list(row)]
        pd.testing.assert_series_equal(pd.Series(row, dtype=object).astype(float),
                                       expected.astype(float), check_names=False)


def test_same_day_fights_do_not_see_each_other():
    """Early-UFC tournaments: a fighter's 2nd bout of the night must not see the 1st."""
    fights, fighters = make_data(n_dates=4)
    a_url, a_name = fights.loc[0, ["fighter_a_url", "fighter_a"]]
    day = fights["date"].max()
    extra = fights[fights["date"] == day].iloc[:2].copy()
    extra["fight_id"] = ["t1", "t2"]
    extra["fighter_a_url"], extra["fighter_a"] = a_url, a_name
    feats, _ = build_features(pd.concat([fights, extra]), fighters)
    t = feats[feats["fight_id"].isin(["t1", "t2"])]
    assert t["a_n_fights"].nunique() == 1 and t["a_elo"].nunique() == 1


def test_hand_computed_values():
    """Two fights for fighter X; check the pre-fight features of the third by hand."""
    fights, fighters = make_data(n_dates=1, fights_per_date=1)
    f = pd.concat([fights] * 3, ignore_index=True)
    f["fight_id"] = ["a", "b", "c"]
    f["date"] = ["2020-01-01", "2020-06-01", "2020-09-01"]
    f["result_a"], f["result_b"] = ["W", "L", "W"], ["L", "W", "L"]
    f["method"] = ["KO/TKO", "Decision - Unanimous", "Submission"]
    f["end_round"], f["end_time"], f["time_format"] = [1, 3, 1], ["2:00", "5:00", "1:00"], "3 Rnd (5-5-5)"
    f["a_sig_landed"], f["a_sig_att"], f["b_sig_landed"], f["b_sig_att"] = [10, 50, 1], [20, 100, 1], [5, 30, 1], [10, 60, 1]
    f["a_td_landed"], f["a_td_att"], f["b_td_landed"], f["b_td_att"] = [1, 1, 0], [2, 3, 0], [0, 2, 0], [1, 4, 0]
    f["a_kd"], f["a_ctrl_sec"], f["b_ctrl_sec"] = [1, 0, 0], [60, 120, 0], [0, 60, 0]
    feats, _ = build_features(f, fighters)
    x = feats.iloc[2]
    mins = (120 + 900) / 60
    assert x["a_n_fights"] == 2 and x["a_wins"] == 1 and x["a_losses"] == 1
    assert x["a_win_rate"] == pytest.approx(2 / 4)
    assert x["a_slpm"] == pytest.approx(60 / mins)
    assert x["a_sapm"] == pytest.approx(35 / mins)
    assert x["a_sig_acc"] == pytest.approx(60 / 120)
    assert x["a_sig_def"] == pytest.approx(1 - 35 / 70)
    assert x["a_kd_per15"] == pytest.approx(15 / mins)
    assert x["a_td_acc"] == pytest.approx(2 / 5) and x["a_td_def"] == pytest.approx(1 - 2 / 5)
    assert x["a_ctrl_pct"] == pytest.approx(180 / 1020) and x["a_opp_ctrl_pct"] == pytest.approx(60 / 1020)
    assert x["a_ko_wins"] == 1 and x["a_ko_losses"] == 0 and x["a_finish_rate"] == pytest.approx(0.5)
    assert x["a_streak"] == -1 and x["a_last3_win_rate"] == pytest.approx(0.5)
    assert x["a_last3_sig_diff_pm"] == pytest.approx((5 + 20) * 60 / 1020)
    assert x["a_days_since_last"] == (pd.Timestamp("2020-09-01") - pd.Timestamp("2020-06-01")).days
    assert x["a_elo"] < x["b_elo"] or x["a_elo"] > 1500  # sanity: Elo moved
    assert x["diff_elo"] == pytest.approx(x["a_elo"] - x["b_elo"])


def test_missing_stats_use_only_recorded_minutes():
    """A fight without stats adds to experience but not to the per-minute denominator."""
    fights, fighters = make_data(n_dates=1, fights_per_date=1)
    f = pd.concat([fights] * 3, ignore_index=True)
    f["fight_id"], f["date"] = ["a", "b", "c"], ["2020-01-01", "2020-06-01", "2020-09-01"]
    f["result_a"], f["result_b"] = "W", "L"
    f["end_round"], f["end_time"], f["time_format"] = 3, "5:00", "3 Rnd (5-5-5)"
    f["a_sig_landed"] = [30.0, np.nan, 0.0]
    f.loc[1, [c for c in f if c[:2] in ("a_", "b_")]] = np.nan
    x = build_features(f, fighters)[0].iloc[2]
    assert x["a_n_fights"] == 2 and x["a_slpm"] == pytest.approx(30 / 15)


def test_debut_fighter_gracefully_handled():
    fights, fighters = make_data()
    _, builder = build_features(fights, fighters)
    row = builder.matchup("never-seen", fights.loc[0, "fighter_a_url"].rsplit("/", 1)[1], "2030-01-01")
    assert row["a_n_fights"] == 0 and row["a_elo"] == 1500 and row["a_win_rate"] == 0.5
    assert np.isnan(row["a_slpm"]) and np.isnan(row["a_age"])


def test_symmetrize_flips_antisymmetric_features_only():
    feats, _ = build_features(*make_data(n_dates=10))
    feats = feats[feats["label"].notna()]
    s = symmetrize(feats)
    n = len(feats)
    a, b = s.iloc[:n].reset_index(drop=True), s.iloc[n:].reset_index(drop=True)
    anti = [c for c in ANTISYMMETRIC if c in feats]
    _assert_same(b[anti], -a[anti])
    _assert_same(b[CONTEXT_FEATURES + SUM_FEATURES], a[CONTEXT_FEATURES + SUM_FEATURES])
    assert ((a["label"] + b["label"]) == 1).all()
    m = a["method_class"].notna()
    assert (a.loc[m, "method_class"].str[0] != b.loc[m, "method_class"].str[0]).all()
    assert (a.loc[m, "method_class"].str[2:] == b.loc[m, "method_class"].str[2:]).all()
    assert ((a.loc[m, "method_class"].str[0] == "a") == (a.loc[m, "label"] == 1)).all()


def _with_dwcs(seed=2):
    fights, fighters = make_data()
    dates = sorted(fights["date"].unique())[::3]
    return fights, fighters, make_dwcs(fighters, [pd.Timestamp(d) - pd.Timedelta(days=7) for d in dates], seed=seed)


def test_no_leakage_with_dwcs_history():
    """Scrambling UFC *and* DWCS results on/after a date leaves earlier features unchanged."""
    fights, fighters, dwcs = _with_dwcs()
    cutoff = pd.Timestamp(sorted(fights["date"].unique())[25])
    base, _ = build_features(fights, fighters, dwcs)
    dw2 = dwcs.copy()
    late = pd.to_datetime(dw2["date"]) >= cutoff
    dw2.loc[late, "result_a"] = "W"
    dw2.loc[late, "result_b"] = "L"
    dw2.loc[late, ["fighter_a", "fighter_b"]] = dw2.loc[late, ["fighter_b", "fighter_a"]].values
    scr, _ = build_features(scramble_from(fights, cutoff), fighters, dw2)
    upto = base["date"] <= cutoff
    _assert_same(base.loc[upto, FEATURE_COLS], scr.loc[upto, FEATURE_COLS])
    assert len(base) == len(fights)  # DWCS bouts are history only, never rows


def test_dwcs_counts_as_history_not_ufc_experience():
    fights, fighters = make_data(n_dates=1, fights_per_date=1)
    ufc = fights.assign(date="2021-01-01")
    name_a = ufc.loc[0, "fighter_a"]
    dwcs = pd.DataFrame([{
        "fight_id": "d1", "event_name": "DWCS", "date": "2020-08-01", "weight_class": "Lightweight",
        "method": "KO (head kick)", "end_round": 1, "end_time": "0:30", "time_format": "3 Rnd (5-5-5)",
        "fighter_a": name_a.upper(), "fighter_b": "Prospect X", "result_a": "W", "result_b": "L", "source_url": ""}])
    without, _ = build_features(ufc, fighters)
    with_, b = build_features(ufc, fighters, dwcs)
    x = with_.iloc[0]
    assert x["a_n_fights"] == 0 and x["a_dwcs_fights"] == 1 and x["a_dwcs_wins"] == 1
    assert x["a_ko_wins"] == 1 and x["a_streak"] == 1 and x["a_finish_rate"] == 1
    assert x["a_elo"] > without.iloc[0]["a_elo"]          # Elo credited to the ufcstats key
    assert x["a_days_since_last"] == (pd.Timestamp("2021-01-01") - pd.Timestamp("2020-08-01")).days
    assert "name:Prospect X" in b.states                  # DWCS-only fighter keyed by name


def test_weight_class_movement_and_durability():
    """Moving up from lightweight to welterweight gives +15 lbs; sub losses / decisions tracked."""
    fights, fighters = make_data(n_dates=1, fights_per_date=1)
    f = pd.concat([fights] * 3, ignore_index=True)
    f["fight_id"], f["date"] = ["a", "b", "c"], ["2020-01-01", "2020-06-01", "2020-09-01"]
    f["weight_class"] = ["Lightweight", "Catch Weight", "Welterweight"]
    f["result_a"], f["result_b"] = ["L", "W", "W"], ["W", "L", "L"]
    f["method"] = ["Submission", "Decision - Split", "KO/TKO"]
    x = build_features(f, fighters)[0]
    assert np.isnan(x.loc[0, "a_weight_change"])           # debut: no previous division
    assert np.isnan(x.loc[1, "a_weight_change"])           # catch weight bout: unknown limit
    assert x.loc[2, "a_weight_change"] == 15               # last known division was lightweight
    assert x.loc[2, "division_lbs"] == 170 and x.loc[2, "womens"] == 0
    assert x.loc[2, "a_sub_losses"] == 1 and x.loc[2, "a_dec_rate"] == pytest.approx(0.5)
    _, b = build_features(f, fighters)
    key = x.loc[0, "key_a"]
    row = b.matchup(key, x.loc[0, "key_b"], "2021-01-01", weight_class="Women's Flyweight")
    assert row["a_weight_change"] == 125 - 170 and row["womens"] == 1
