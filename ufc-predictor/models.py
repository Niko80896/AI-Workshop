"""Orientation-symmetric model wrappers shared by ``train.py`` and ``predict.py``.

Every model is fit on both orientations of each fight (see
:func:`features.symmetrize`) and predicts ``P(A wins)`` as the average of
``P(A beats B)`` and ``1 - P(B beats A)``, so swapping the corner order can
never change the answer.

``MethodModel`` does the same for the 6-way outcome (A or B, by KO/TKO,
submission or decision): the mirrored prediction's classes are swapped
(``a_ko`` <-> ``b_ko`` ...) before averaging.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (ANTISYMMETRIC, METHOD_CLASSES, METHOD_FEATURES, METHOD_FLIP, MODEL_FEATURES,
                      symmetrize)


def mirror(X: pd.DataFrame) -> pd.DataFrame:
    """B-vs-A view of A-vs-B feature rows (antisymmetric features negated)."""
    Xm = X.copy()
    cols = [c for c in ANTISYMMETRIC if c in X]
    Xm[cols] = -X[cols]
    return Xm


class SymmetricModel:
    """Base class: subclasses implement ``_fit``, ``_proba`` and ``_contrib``."""

    name = "base"

    def __init__(self, features: list[str] | None = None):
        self.features = list(features or MODEL_FEATURES)

    def fit(self, df: pd.DataFrame) -> "SymmetricModel":
        """Fit on labelled per-fight rows (both orientations are generated internally)."""
        s = symmetrize(df[df["label"].notna()])
        self._fit(s[self.features], s["label"].astype(int).to_numpy())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Orientation-averaged P(A wins) for each row of A-vs-B features."""
        X = X[self.features]
        return 0.5 * (self._proba(X) + 1.0 - self._proba(mirror(X)))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature log-odds contributions towards A winning (orientation-averaged)."""
        X = X[self.features]
        c = 0.5 * (self._contrib(X) - self._contrib(mirror(X)))
        return pd.DataFrame(c, columns=self.features, index=X.index)

    # subclass hooks -----------------------------------------------------
    def _fit(self, X, y):
        raise NotImplementedError

    def _proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def _contrib(self, X) -> np.ndarray:
        raise NotImplementedError


class EloBaseline(SymmetricModel):
    """Always pick the higher-Elo fighter; probability from the Elo expected score."""

    name = "elo"

    def __init__(self):
        super().__init__(["diff_elo"])

    def _fit(self, X, y):
        return self

    def _proba(self, X):
        return 1.0 / (1.0 + 10 ** (-X["diff_elo"].to_numpy() / 400.0))

    def _contrib(self, X):
        return (X["diff_elo"].to_numpy() * np.log(10) / 400.0)[:, None]


class LogisticModel(SymmetricModel):
    """Standardized logistic regression; missing diffs imputed as 0 (= no difference)."""

    name = "logistic"

    def __init__(self, features=None, C: float = 0.1):
        super().__init__(features)
        self.C = C

    def _fit(self, X, y):
        self.pipe = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
            StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        self.pipe.fit(X, y)

    def _proba(self, X):
        return self.pipe.predict_proba(X)[:, 1]

    def _contrib(self, X):
        imp, sc, lr = self.pipe.named_steps.values()
        z = sc.transform(imp.transform(X))
        return z * lr.coef_[0]


class GBMModel(SymmetricModel):
    """LightGBM gradient boosting (handles missing values natively)."""

    name = "lightgbm"
    DEFAULT_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=60,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=2.0, verbose=-1)

    def __init__(self, features=None, n_estimators: int = 400, **params):
        super().__init__(features)
        self.n_estimators = n_estimators
        self.params = {**self.DEFAULT_PARAMS, **params}

    def _fit(self, X, y):
        self.model = lgb.LGBMClassifier(n_estimators=self.n_estimators, random_state=0, **self.params)
        self.model.fit(X, y)

    def best_iterations(self, train: pd.DataFrame, valid: pd.DataFrame, max_rounds: int = 3000) -> int:
        """Pick ``n_estimators`` by early stopping on a later, time-based validation block."""
        tr, va = symmetrize(train[train["label"].notna()]), symmetrize(valid[valid["label"].notna()])
        m = lgb.LGBMClassifier(n_estimators=max_rounds, random_state=0, **self.params)
        m.fit(tr[self.features], tr["label"].astype(int), eval_set=[(va[self.features], va["label"].astype(int))],
              eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
        return int(m.best_iteration_ or max_rounds)

    def _proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def _contrib(self, X):
        return self.model.predict(X, pred_contrib=True)[:, :-1]  # drop the bias column


_FLIP_IDX = [METHOD_CLASSES.index(METHOD_FLIP[c]) for c in METHOD_CLASSES]


class MethodModel:
    """6-way method-of-victory model: P(a_ko, a_sub, a_dec, b_ko, b_sub, b_dec).

    ``kind`` is ``"prior"`` (training-set method frequencies, split evenly
    between the two corners), ``"logistic"`` (multinomial) or ``"lightgbm"``.
    """

    def __init__(self, kind: str = "logistic", features: list[str] | None = None,
                 C: float = 0.01, n_estimators: int = 300):
        self.kind, self.C, self.n_estimators = kind, C, n_estimators
        self.features = list(features or METHOD_FEATURES)
        self.name = {"prior": "Method frequencies", "logistic": "Multinomial logistic",
                     "lightgbm": "LightGBM multiclass"}[kind]

    @staticmethod
    def _xy(df: pd.DataFrame, features):
        s = symmetrize(df[df["method_class"].notna()])
        return s[features], s["method_class"].map(METHOD_CLASSES.index).to_numpy()

    def _new_lgbm(self, n):
        return lgb.LGBMClassifier(objective="multiclass", n_estimators=n, random_state=0,
                                  **GBMModel.DEFAULT_PARAMS)

    def fit(self, df: pd.DataFrame) -> "MethodModel":
        X, y = self._xy(df, self.features)
        if self.kind == "prior":
            self.prior = np.bincount(y, minlength=len(METHOD_CLASSES)) / len(y)
        elif self.kind == "logistic":
            self.pipe = make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                StandardScaler(), LogisticRegression(C=self.C, max_iter=3000))
            self.pipe.fit(X, y)
        else:
            self.model = self._new_lgbm(self.n_estimators).fit(X, y)
        return self

    def best_iterations(self, train: pd.DataFrame, valid: pd.DataFrame, max_rounds: int = 2000) -> int:
        """Early-stopped tree count on a later validation block (LightGBM only)."""
        Xt, yt = self._xy(train, self.features)
        Xv, yv = self._xy(valid, self.features)
        m = self._new_lgbm(max_rounds)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return int(m.best_iteration_ or max_rounds)

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "prior":
            return np.tile(self.prior, (len(X), 1))
        est = self.pipe if self.kind == "logistic" else self.model
        return est.predict_proba(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Orientation-averaged probabilities, columns in ``METHOD_CLASSES`` order."""
        X = X[self.features]
        return 0.5 * (self._raw(X) + self._raw(mirror(X))[:, _FLIP_IDX])


def combine(p_win: np.ndarray, P_method: np.ndarray) -> np.ndarray:
    """P(A wins) x P(method | A wins) and likewise for B, from a 6-way method model."""
    a = P_method[:, :3] / P_method[:, :3].sum(axis=1, keepdims=True)
    b = P_method[:, 3:] / P_method[:, 3:].sum(axis=1, keepdims=True)
    return np.hstack([p_win[:, None] * a, (1 - p_win)[:, None] * b])
