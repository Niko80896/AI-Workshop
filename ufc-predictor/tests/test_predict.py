"""End-to-end tests for models.py and predict.py on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from features import build_features
from models import EloBaseline, GBMModel, LogisticModel, mirror
from predict import FighterIndex, predict_matchup
from synthetic import make_data, make_dwcs


@pytest.fixture(scope="module")
def world():
    fights, fighters = make_data(n_dates=80)
    dwcs = make_dwcs(fighters, ["2014-08-01", "2014-09-01"])
    feats, builder = build_features(fights, fighters, dwcs)
    return fights, fighters, dwcs, feats, builder


@pytest.mark.parametrize("cls", [EloBaseline, LogisticModel, lambda: GBMModel(n_estimators=30)])
def test_models_are_orientation_symmetric(world, cls):
    feats = world[3]
    m = cls().fit(feats)
    p = m.predict_proba(feats)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(m.predict_proba(mirror(feats[m.features])), 1 - p, atol=1e-12)


def test_logistic_contributions_sum_to_logit(world):
    feats = world[3]
    m = LogisticModel().fit(feats)
    p = m.predict_proba(feats.head(20))
    c = m.contributions(feats.head(20)).sum(axis=1).to_numpy()
    # orientation-averaged probabilities are close to sigmoid(sum of averaged contributions)
    np.testing.assert_allclose(1 / (1 + np.exp(-c)), p, atol=0.02)


def test_predict_matchup_symmetric_and_handles_debuts(world):
    fights, fighters, dwcs, feats, builder = world
    m = LogisticModel().fit(feats)
    ka, kb = "f001", "f002"
    r1 = predict_matchup(m, builder, ka, kb, "2030-01-01", five_rounds=True, title=True)
    r2 = predict_matchup(m, builder, kb, ka, "2030-01-01", five_rounds=True, title=True)
    assert r1["p_a"] == pytest.approx(r2["p_b"], abs=1e-12)
    assert r1["factors"] and all(f["favors"] in ("a", "b") for f in r1["factors"])
    debut = predict_matchup(m, builder, ka, "name:Nobody", "2030-01-01")
    assert 0 < debut["p_a"] < 1 and debut["row"]["b_n_fights"] == 0


def test_fighter_index_resolution(world):
    fights, fighters, dwcs, feats, builder = world
    idx = FighterIndex(fights, fighters, dwcs, builder)
    assert idx.resolve("fighter 3").key == "f003"                      # case-insensitive
    assert idx.resolve("http://ufcstats.com/fighter-details/f004").key == "f004"
    m = idx.resolve("Fighter 3x")                                      # close typo
    assert m.key in ("f003", "f033", "f030") and "closest" in m.note
    assert idx.resolve("Prospect 1").key == "name:Prospect 1"          # DWCS-only fighter
    new = idx.resolve("Completely Unknown Person")
    assert new.key.startswith("name:") and "debut" in new.note
