"""Train and evaluate fight-outcome models with a time-based split.

* Train on fights before ``--test-start`` (default 2024-01-01), test on fights
  from that date on. Never a random split.
* Models: Elo baseline (pick the higher-rated fighter), logistic regression,
  LightGBM. LightGBM's number of trees and the logistic C are chosen on a later
  validation block *inside* the training period (``--val-start``).
* Reports accuracy, log loss and Brier score on the test set, a calibration plot
  and permutation importance (test set) for the best model.
* Betting odds (``import_odds.py``, optional): the vig-free market probability is
  reported as a benchmark on the test fights that have odds, and a second
  logistic model that also sees the market's log-odds is trained and evaluated
  on the fights with odds.
* Method of victory: 6-way models (A/B x KO/SUB/DEC) vs method frequencies and,
  where available, the market's method props. The reported "combined" model is
  what ``predict.py`` shows: P(A wins) from the winner model x P(method | A wins).
* The best models (by test log loss) are refit on all labelled fights and saved
  to ``models/model.joblib`` for ``predict.py``.

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

from features import (ELO_K, METHOD_CLASSES, MODEL_FEATURES, ODDS_FEATURES, add_market, build_features,
                      load_data)
from import_odds import load_odds
from models import EloBaseline, GBMModel, LogisticModel, MethodModel, combine, mirror

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


def plot_calibration(preds: dict[str, np.ndarray], y: np.ndarray, path: Path, n_bins: int = 10,
                     title: str = "Calibration on the test set") -> None:
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
    ax.set_title(f"{title} (bins with ≥20 fighter-sides)", color=INK, fontsize=11, loc="left")
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

def evaluate_method(df: pd.DataFrame, P: np.ndarray) -> dict:
    """6-way log loss / accuracy and 3-way method accuracy (who wins ignored)."""
    y = df["method_class"].map(METHOD_CLASSES.index).to_numpy()
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(axis=1, keepdims=True)
    how = P[:, :3] + P[:, 3:]
    return {"log_loss_6way": float(log_loss(y, P, labels=list(range(6)))),
            "accuracy_6way": float(np.mean(P.argmax(1) == y)),
            "accuracy_method": float(np.mean(how.argmax(1) == y % 3)), "n_fights": int(len(y))}


def _print_table(title: str, metrics: dict, cols) -> None:
    print(f"\n{title}")
    print(pd.DataFrame(metrics).T[cols].to_string(float_format=lambda v: f"{v:.4f}"))


def _tune_c(make, tr, va, y_col="label"):
    yv = va[y_col].to_numpy()
    scores = {c: log_loss(yv, make(c).fit(tr).predict_proba(va)) for c in (0.001, 0.003, 0.01, 0.03, 0.1, 1.0)}
    return min(scores, key=scores.get)


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
    feats = add_market(feats, load_odds())
    data = feats[feats["label"].notna() & (feats["date"] >= args.min_train_date)].reset_index(drop=True)
    train = data[data["date"] < args.test_start]
    test = data[data["date"] >= args.test_start]
    inner_tr, inner_va = train[train["date"] < args.val_start], train[train["date"] >= args.val_start]
    print(f"train {len(train)} fights ({train['date'].min():%Y-%m-%d}..{train['date'].max():%Y-%m-%d}), "
          f"test {len(test)} fights ({test['date'].min():%Y-%m-%d}..{test['date'].max():%Y-%m-%d})")
    report: dict = {"test_start": args.test_start, "n_train": len(train), "n_test": len(test)}

    # ------------------------------------------------------------------ winner
    best_c = _tune_c(lambda c: LogisticModel(C=c), inner_tr, inner_va)
    n_trees = GBMModel().best_iterations(inner_tr, inner_va)
    print(f"logistic C={best_c}; lightgbm trees={n_trees} (chosen on {args.val_start[:4]}+ validation block)")

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
    _print_table(f"Winner, all test fights (on/after {args.test_start}):", metrics,
                 ["accuracy", "log_loss", "brier", "n_fights"])
    plot_calibration(preds, y, REPORTS / "calibration.png")
    report.update(logistic_C=best_c, lightgbm_trees=n_trees, metrics=metrics)

    best_name = min((n for n in models if n != "Elo baseline"), key=lambda n: metrics[n]["log_loss"])
    best = models[best_name]
    report["best_model"] = best_name

    # --------------------------------------------------- betting-odds benchmark
    odds_model, odds_c = None, None
    has = data["p_market_a"].notna()
    if has.any():
        tr_o, te_o = train[has.loc[train.index]], test[has.loc[test.index]]
        itr_o, iva_o = tr_o[tr_o["date"] < args.val_start], tr_o[tr_o["date"] >= args.val_start]
        feats_o = MODEL_FEATURES + ODDS_FEATURES
        odds_c = _tune_c(lambda c: LogisticModel(features=feats_o, C=c), itr_o, iva_o)
        odds_model = LogisticModel(features=feats_o, C=odds_c).fit(tr_o)
        yo = te_o["label"].to_numpy()
        po = {"Betting market (vig-free)": te_o["p_market_a"].to_numpy(),
              f"{best_name} (no odds)": best.predict_proba(te_o),
              "Logistic + market odds": odds_model.predict_proba(te_o)}
        m_o = {k: evaluate(yo, v) for k, v in po.items()}
        m_o["Elo baseline"] = evaluate(yo, models["Elo baseline"].predict_proba(te_o))
        _print_table(f"Winner, test fights with odds ({len(te_o)} of {len(test)}, "
                     f"{te_o['date'].min():%Y-%m-%d}..{te_o['date'].max():%Y-%m-%d}):", m_o,
                     ["accuracy", "log_loss", "brier", "n_fights"])
        plot_calibration(po, yo, REPORTS / "calibration_odds.png", title="Test fights with odds")
        report.update(odds_metrics=m_o, odds_logistic_C=odds_c, n_train_with_odds=len(tr_o))

    # -------------------------------------------------------- method of victory
    mtrain, mtest = train[train["method_class"].notna()], test[test["method_class"].notna()]
    imtr, imva = mtrain[mtrain["date"] < args.val_start], mtrain[mtrain["date"] >= args.val_start]
    yv = imva["method_class"].map(METHOD_CLASSES.index)
    m_c = min((0.001, 0.003, 0.01, 0.03, 0.1),
              key=lambda c: log_loss(yv, MethodModel("logistic", C=c).fit(imtr).predict_proba(imva), labels=range(6)))
    m_trees = MethodModel("lightgbm").best_iterations(imtr, imva)
    method_models = {"Method frequencies": MethodModel("prior"),
                     "Multinomial logistic": MethodModel("logistic", C=m_c),
                     "LightGBM multiclass": MethodModel("lightgbm", n_estimators=m_trees)}
    m_metrics, m_preds = {}, {}
    p_win = best.predict_proba(mtest)
    for name, mm in method_models.items():
        mm.fit(mtrain)
        m_preds[name] = mm.predict_proba(mtest)
        m_metrics[name] = evaluate_method(mtest, m_preds[name])
        m_metrics[f"{best_name} x {name}"] = evaluate_method(mtest, combine(p_win, m_preds[name]))
    _print_table(f"Method of victory, test fights ({len(mtest)}; draws/NC/DQ excluded):", m_metrics,
                 ["log_loss_6way", "accuracy_6way", "accuracy_method", "n_fights"])
    props = [f"p_market_{c}" for c in METHOD_CLASSES]
    hp = mtest[props].notna().all(axis=1).to_numpy()
    if hp.any():
        sub = mtest[hp]
        mp = {"Betting market props (vig-free)": evaluate_method(sub, sub[props].to_numpy())}
        for name in method_models:
            mp[f"{best_name} x {name}"] = evaluate_method(sub, combine(p_win[hp], m_preds[name][hp]))
        _print_table(f"Method of victory, test fights with method props ({hp.sum()}):", mp,
                     ["log_loss_6way", "accuracy_6way", "accuracy_method", "n_fights"])
        report["method_props_metrics"] = mp
    best_method = min((n for n in method_models if n != "Method frequencies"),
                      key=lambda n: m_metrics[f"{best_name} x {n}"]["log_loss_6way"])
    report.update(method_metrics=m_metrics, method_logistic_C=m_c, method_lightgbm_trees=m_trees,
                  best_method_model=best_method)

    # ------------------------------------------------------ feature importance
    print(f"\nbest winner model: {best_name}; method model: {best_method}")
    print("computing permutation importance on the test set ...")
    imp = permutation_importance(best, test, y)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)
    plot_importance(imp, best_name, REPORTS / "feature_importance.png")
    print(imp.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------- refit on all data and save
    final = type(best)(**({"C": best_c} if best_name == "Logistic regression" else {"n_estimators": n_trees}))
    final.fit(data)
    final_odds = None
    if odds_model is not None:
        final_odds = LogisticModel(features=odds_model.features, C=odds_c).fit(data[data["p_market_a"].notna()])
    bm = method_models[best_method]
    final_method = MethodModel(bm.kind, C=bm.C, n_estimators=bm.n_estimators).fit(data[data["method_class"].notna()])
    joblib.dump({"model": final, "name": best_name, "odds_model": final_odds, "method_model": final_method,
                 "method_name": best_method, "elo_k": ELO_K, "test_metrics": metrics,
                 "trained_through": str(data["date"].max().date()), "n_train_fights": int(len(data)),
                 "test_start": args.test_start}, MODEL_PATH)
    with open(REPORTS / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved {best_name}{' + odds model' if final_odds else ''} + {best_method} "
          f"(refit on {len(data)} fights through {data['date'].max():%Y-%m-%d}) to {MODEL_PATH}")
    print(f"reports written to {REPORTS}/")


if __name__ == "__main__":
    main()
