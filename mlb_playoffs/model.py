"""Single-game win-probability models.

Time-based evaluation only (never a random split):
    train 2016-2022  ->  validate/tune 2023  ->  test 2024-2025
2015 is a burn-in season for Elo and the decayed stats, so it is not used.

Compares four baselines (home team always wins, higher Pythagorean win%,
Log5, Elo) against logistic regression and gradient boosting, reports
accuracy / log loss / Brier score, draws a calibration plot, ranks feature
importance, and walk-forward backtests every postseason 2017-2025 (each
season predicted by a model trained only on earlier seasons).

Usage:
    python model.py                 # evaluate + backtest + save production model
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from features import ELO_HFA, FEATURE_LABELS, FEATURES

log = logging.getLogger("model")

TRAIN = range(2016, 2023)
VALID = [2023]
TEST = [2024, 2025]


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

def log5(pa, pb):
    """Probability A beats B from each team's win% (Bill James' Log5)."""
    return (pa - pa * pb) / (pa + pb - 2 * pa * pb)


def baseline_probs(df: pd.DataFrame, home_rate: float) -> dict[str, np.ndarray]:
    return {
        "Home team always wins": np.full(len(df), home_rate),
        "Higher Pythag win% wins": np.where(df["home_pyth"] >= df["away_pyth"], 0.55, 0.45),
        "Log5 (Pythag win%)": log5(df["home_pyth"].values, df["away_pyth"].values),
        "Elo": 1 / (1 + 10 ** (-(df["elo_diff"].values + ELO_HFA) / 400)),
    }


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def make_logreg(c=0.05):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def make_gbm(lr=0.03, iters=250, depth=3, leaf=200):
    return HistGradientBoostingClassifier(learning_rate=lr, max_iter=iters, max_depth=depth,
                                          min_samples_leaf=leaf, l2_regularization=1.0,
                                          random_state=0)


class EnsembleModel:
    """Average of logistic regression and gradient boosting probabilities.

    The logistic regression part also provides per-prediction explanations
    (coefficient x standardized feature value = log-odds contribution).
    """

    def __init__(self, lr_c=0.05, gbm_kwargs=None, weight_lr=0.5):
        self.lr = make_logreg(lr_c)
        self.gbm = make_gbm(**(gbm_kwargs or {}))
        self.w = weight_lr
        self.features = list(FEATURES)

    def fit(self, X: pd.DataFrame, y):
        self.lr.fit(X[self.features], y)
        self.gbm.fit(X[self.features], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        return self.w * self.lr.predict_proba(X)[:, 1] + (1 - self.w) * self.gbm.predict_proba(X)[:, 1]

    def explain(self, row: dict, top: int = 6) -> list[tuple[str, float, float]]:
        """Top factors for one game: (label, log-odds contribution to home, raw value)."""
        scaler, lr = self.lr[0], self.lr[-1]
        x = np.array([[row[f] for f in self.features]])
        z = (x - scaler.mean_) / scaler.scale_
        contrib = (lr.coef_[0] * z[0])
        order = np.argsort(-np.abs(contrib))[:top]
        return [(FEATURE_LABELS[self.features[i]], float(contrib[i]), float(x[0, i])) for i in order]


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"accuracy": accuracy_score(y, p >= 0.5), "log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p), "n": len(y)}


def load_features() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / "features.csv", parse_dates=["date"])
    return df[df["season"] >= 2016].reset_index(drop=True)


def tune(train, valid) -> tuple[float, dict, float]:
    """Pick LR regularization, GBM settings and blend weight by validation log loss."""
    best_c = min([0.003, 0.01, 0.03, 0.1, 1.0], key=lambda c: log_loss(
        valid.home_win, make_logreg(c).fit(train[FEATURES], train.home_win).predict_proba(valid[FEATURES])[:, 1]))
    grid = [dict(lr=0.03, iters=250, depth=3, leaf=200), dict(lr=0.02, iters=400, depth=2, leaf=300),
            dict(lr=0.05, iters=150, depth=4, leaf=400)]
    scores = []
    for g in grid:
        m = make_gbm(**g).fit(train[FEATURES], train.home_win)
        scores.append(log_loss(valid.home_win, m.predict_proba(valid[FEATURES])[:, 1]))
    best_g = grid[int(np.argmin(scores))]
    ens = EnsembleModel(best_c, best_g).fit(train, train.home_win)
    p_lr = ens.lr.predict_proba(valid[FEATURES])[:, 1]
    p_gb = ens.gbm.predict_proba(valid[FEATURES])[:, 1]
    best_w = min(np.linspace(0, 1, 11), key=lambda w: log_loss(valid.home_win, w * p_lr + (1 - w) * p_gb))
    return best_c, best_g, float(best_w)


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Train on TRAIN, tune on VALID, report everything on TEST."""
    tr, va, te = (df[df.season.isin(s)] for s in (TRAIN, VALID, TEST))
    c, g, w = tune(tr, va)
    log.info("tuned: LR C=%s, GBM %s, LR weight %.1f", c, g, w)
    trva = pd.concat([tr, va])
    home_rate = trva.home_win.mean()
    lr = make_logreg(c).fit(trva[FEATURES], trva.home_win)
    gb = make_gbm(**g).fit(trva[FEATURES], trva.home_win)
    ens = EnsembleModel(c, g, w).fit(trva, trva.home_win)
    probs = baseline_probs(te, home_rate)
    probs["Logistic regression"] = lr.predict_proba(te[FEATURES])[:, 1]
    probs["Gradient boosting"] = gb.predict_proba(te[FEATURES])[:, 1]
    probs["Ensemble (LR + GBM)"] = ens.predict_proba(te)
    table = pd.DataFrame({k: metrics(te.home_win, p) for k, p in probs.items()}).T
    return table, {"probs": probs, "test": te, "lr": lr, "gb": gb, "params": (c, g, w)}


def feature_importance(res) -> pd.DataFrame:
    te = res["test"]
    lr = res["lr"][-1]
    perm = permutation_importance(res["gb"], te[FEATURES], te.home_win, scoring="neg_log_loss",
                                  n_repeats=5, random_state=0)
    return pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in FEATURES],
        "lr_std_coef": lr.coef_[0],
        "gbm_perm_importance": perm.importances_mean,
    }).sort_values("lr_std_coef", key=np.abs, ascending=False)


def postseason_backtest(df: pd.DataFrame, params) -> pd.DataFrame:
    """Walk-forward: predict each postseason with a model trained on all earlier games."""
    c, g, w = params
    rows = []
    for season in range(2017, 2026):
        train = df[(df.season < season) | ((df.season == season) & (df["round"] == "REG"))]
        test = df[(df.season == season) & (df["round"] != "REG")]
        if test.empty:
            continue
        ens = EnsembleModel(c, g, w).fit(train, train.home_win)
        out = test[["game_id", "date", "season", "round", "hometeam", "visteam", "home_win",
                    "elo_diff", "home_pyth", "away_pyth"]].copy()
        out["p_model"] = ens.predict_proba(test)
        out["p_elo"] = 1 / (1 + 10 ** (-(test.elo_diff + ELO_HFA) / 400))
        out["p_log5"] = log5(test.home_pyth.values, test.away_pyth.values)
        out["p_home"] = train.home_win.mean()
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def calibration_plot(y, probs: dict, path, title):
    import matplotlib.pyplot as plt

    import plotstyle as S
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0.2, 0.8], [0.2, 0.8], color=S.NEUTRAL, lw=1, zorder=1)
    bins = np.linspace(0.2, 0.8, 9)
    for i, (name, p) in enumerate(probs.items()):
        idx = np.digitize(p, bins)
        xs, ys, ns = [], [], []
        for b in np.unique(idx):
            m = idx == b
            if m.sum() >= 40:
                xs.append(p[m].mean()); ys.append(np.mean(np.asarray(y)[m])); ns.append(m.sum())
        ax.plot(xs, ys, marker="o", ms=6, color=S.SERIES[i], label=name, zorder=3,
                markeredgecolor=S.SURFACE, markeredgewidth=1.5)
    ax.set_xlim(0.2, 0.8); ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title(title)
    ax.text(0.79, 0.22, "perfect calibration on the diagonal", ha="right", color=S.TEXT_2, fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_production(df: pd.DataFrame, params, until=None, name="production") -> EnsembleModel:
    """Fit on every game before ``until`` and save to data/models/<name>.pkl."""
    c, g, w = params
    train = df if until is None else df[df.date < pd.Timestamp(until)]
    ens = EnsembleModel(c, g, w).fit(train, train.home_win)
    with open(C.MODEL_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(ens, f)
    (C.MODEL_DIR / f"{name}.json").write_text(pd.Series(
        {"train_games": len(train), "through": str(train.date.max().date()), "lr_c": c, "lr_weight": w,
         "gbm": str(g)}).to_json(indent=1))
    return ens


def load_model(name="production") -> EnsembleModel:
    with open(C.MODEL_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    df = load_features()
    table, res = evaluate(df)
    pd.set_option("display.width", 140)
    print("\n=== Single-game test set: 2024-2025 regular season + postseason ===")
    print(table.round(4).to_string())
    table.to_csv(C.DATA_DIR / "model_eval.csv")

    imp = feature_importance(res)
    print("\n=== Feature importance (LR standardized coef; GBM permutation, log-loss) ===")
    print(imp.round(4).to_string(index=False))
    imp.to_csv(C.DATA_DIR / "feature_importance.csv", index=False)

    te = res["test"]
    calibration_plot(te.home_win.values, {k: res["probs"][k] for k in
                     ("Ensemble (LR + GBM)", "Elo", "Log5 (Pythag win%)")},
                     C.CHART_DIR / "calibration_test.png", "Calibration, 2024-25 test games")

    bt = postseason_backtest(df, res["params"])
    bt.to_csv(C.DATA_DIR / "postseason_backtest.csv", index=False)
    summary = pd.DataFrame({k: metrics(bt.home_win, bt[c]) for k, c in
                            [("Model", "p_model"), ("Elo", "p_elo"), ("Log5", "p_log5"), ("Home rate", "p_home")]}).T
    print(f"\n=== Walk-forward postseason backtest 2017-2025 ({len(bt)} games) ===")
    print(summary.round(4).to_string())
    by_season = bt.groupby("season").apply(lambda d: pd.Series(
        {"games": len(d), "model_acc": ((d.p_model >= .5) == d.home_win).mean(),
         "model_ll": log_loss(d.home_win, d.p_model, labels=[0, 1]),
         "elo_ll": log_loss(d.home_win, d.p_elo, labels=[0, 1])}), include_groups=False)
    print(by_season.round(3).to_string())
    calibration_plot(bt.home_win.values, {"Model": bt.p_model.values, "Elo": bt.p_elo.values},
                     C.CHART_DIR / "calibration_postseason.png", "Calibration, postseason 2017-25 (walk-forward)")

    # Production models: one frozen before the 2025 postseason (honest 2025 demo),
    # one trained on everything (for 2026).
    train_production(df, res["params"], until="2025-09-29", name="through_2025_regular")
    train_production(df, res["params"], name="production")
    print("\nSaved models to", C.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
