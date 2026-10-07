"""Train and evaluate fight-outcome models with a time-based split.

* Train on fights before ``--test-start`` (default 2024-01-01), test on fights
  from that date on. Never a random split.
* Models: Elo baseline (pick the higher-rated fighter), logistic regression,
  LightGBM. LightGBM's number of trees and the logistic C are chosen on a later
  validation block *inside* the training period (``--val-start``).
* Reports accuracy, log loss and Brier score on the test set, a calibration plot
  and permutation importance (test set) for the best model.
* The best model (by test log loss) is then refit on all labelled fights and
  saved to ``models/model.joblib`` for ``predict.py``.

Usage::

    python train.py
    python train.py --test-start 2025-01-01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from features import ELO_K, build_features, load_data
from models import EloBaseline, GBMModel, LogisticModel, mirror

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"

# Reference categorical palette (validated slots 1-3, light surface) + chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3dc"


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    """Accuracy (a 0.5 prediction earns half credit), log loss, Brier score."""
    pred = np.where(p > 0.5, 1.0, np.where(p < 0.5, 0.0, 0.5))
    acc = np.mean(np.where(pred == 0.5, 0.5, pred == y))
    return {"accuracy": float(acc), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "n_fights": int(len(y))}


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5,
                           seed: int = 0) -> pd.DataFrame:
    """Increase in test log loss when one feature is shuffled across fights.

    The shuffled column is fed through the same orientation-averaged prediction,
    so the mirrored (B vs A) copy stays consistent with the permuted A-vs-B value.
    """
    rng = np.random.default_rng(seed)
    base = log_loss(y, model.predict_proba(X), labels=[0, 1])
    out = []
    for f in model.features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(log_loss(y, model.predict_proba(Xp), labels=[0, 1]) - base)
        out.append({"feature": f, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(out).sort_values("importance", ascending=False).reset_index(drop=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_calibration(preds: dict[str, np.ndarray], y: np.ndarray, path: Path, n_bins: int = 10) -> None:
    """Reliability diagram on both orientations of each test fight (so it is symmetric)."""
    fig, ax = plt.subplots(figsize=(6.4, 6), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Perfectly calibrated")
    yy = np.concatenate([y, 1 - y])
    edges = np.linspace(0, 1, n_bins + 1)
    for color, (name, p) in zip(SERIES, preds.items()):
        pp = np.concatenate([p, 1 - p])
        idx = np.clip(np.digitize(pp, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= 20:
                xs.append(pp[m].mean())
                ys.append(yy[m].mean())
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted win probability", color=INK, fontsize=10)
    ax.set_ylabel("Observed win rate", color=INK, fontsize=10)
    ax.set_title("Calibration on the test set (bins with ≥20 fighter-sides)", color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, model_name: str, path: Path, top: int = 20) -> None:
    """Horizontal bar chart of the top permutation importances."""
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(d["feature"], d["importance"], color=SERIES[0], height=0.7,
            xerr=d["std"], error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_xlabel("Increase in test log loss when shuffled", color=INK, fontsize=10)
    ax.set_title(f"Permutation importance ({model_name}), top {len(d)}", color=INK, fontsize=11, loc="left")
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--val-start", default="2021-01-01", help="validation block inside the training period")
    ap.add_argument("--min-train-date", default="2001-01-01",
                    help="earliest fight used as a training row (all fights still feed history)")
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    fights, fighters, dwcs = load_data()
    print("building features ...")
    feats, _ = build_features(fights, fighters, dwcs)
    data = feats[feats["label"].notna() & (feats["date"] >= args.min_train_date)].reset_index(drop=True)
    train = data[data["date"] < args.test_start]
    test = data[data["date"] >= args.test_start]
    inner_tr, inner_va = train[train["date"] < args.val_start], train[train["date"] >= args.val_start]
    print(f"train {len(train)} fights ({train['date'].min():%Y-%m-%d}..{train['date'].max():%Y-%m-%d}), "
          f"test {len(test)} fights ({test['date'].min():%Y-%m-%d}..{test['date'].max():%Y-%m-%d})")

    # --- hyperparameters chosen on the validation block inside the training period
    yv = inner_va["label"].to_numpy()
    c_scores = {c: log_loss(yv, LogisticModel(C=c).fit(inner_tr).predict_proba(inner_va)) for c in (0.001, 0.003, 0.01, 0.03, 0.1, 1.0)}
    best_c = min(c_scores, key=c_scores.get)
    n_trees = GBMModel().best_iterations(inner_tr, inner_va)
    print(f"logistic C={best_c} (val log loss {c_scores[best_c]:.4f}); lightgbm trees={n_trees}")

    models = {"Elo baseline": EloBaseline(), "Logistic regression": LogisticModel(C=best_c),
              "LightGBM": GBMModel(n_estimators=n_trees)}
    y = test["label"].to_numpy()
    preds, metrics = {}, {}
    for name, m in models.items():
        m.fit(train)
        preds[name] = m.predict_proba(test)
        metrics[name] = evaluate(y, preds[name])
        # Symmetry check: swapping corners must give exactly 1 - p.
        assert np.allclose(m.predict_proba(mirror(test[m.features])), 1 - preds[name])

    table = pd.DataFrame(metrics).T[["accuracy", "log_loss", "brier", "n_fights"]]
    print("\nTest-set results (fights on/after", args.test_start + "):")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    plot_calibration(preds, y, REPORTS / "calibration.png")

    best_name = min((n for n in models if n != "Elo baseline"), key=lambda n: metrics[n]["log_loss"])
    best = models[best_name]
    print(f"\nbest model: {best_name}; computing permutation importance on the test set ...")
    imp = permutation_importance(best, test, y)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    plot_importance(imp, best_name, REPORTS / "feature_importance.png")
    print(imp.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- refit the best model on every labelled fight and save it
    final = type(best)(**({"C": best_c} if best_name == "Logistic regression" else {"n_estimators": n_trees}))
    final.fit(data)
    joblib.dump({"model": final, "name": best_name, "elo_k": ELO_K, "test_metrics": metrics,
                 "trained_through": str(data["date"].max().date()), "n_train_fights": int(len(data)),
                 "test_start": args.test_start}, MODEL_PATH)
    with open(REPORTS / "metrics.json", "w") as fh:
        json.dump({"test_start": args.test_start, "n_train": len(train), "n_test": len(test),
                   "logistic_C": best_c, "lightgbm_trees": n_trees, "best_model": best_name,
                   "metrics": metrics}, fh, indent=2)
    print(f"\nsaved {best_name} (refit on {len(data)} fights through {data['date'].max():%Y-%m-%d}) to {MODEL_PATH}")
    print(f"reports written to {REPORTS}/")


if __name__ == "__main__":
    main()
