"""Run-scoring model: first inning (NRFI/YRFI), first 3 / first 5 innings, full-game totals.

Each team's runs are modeled in four segments: inning 1, innings 2-3,
innings 4-5 and innings 6-end. For each segment a Poisson regression
(log link) predicts the expected runs from that team's lineup against the
opposing starter, bullpen and defense; the spread around that mean is a
negative binomial whose dispersion is fit on the training seasons (real run
distributions have more shutout innings *and* more big innings than a
Poisson). Simulating the four segments gives scores that are consistent
across NRFI, F3, F5 and full-game markets.

Calibrated against real scoring (NRFI happens in ~50% of games, F5 totals
average ~5.1 runs), unlike a simulator tuned to suppress scoring. Features
come from features.csv, so the same no-leakage guarantee applies.

Usage:
    python runs.py              # fit, backtest 2024-2026 (out of sample), save models
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import data

log = logging.getLogger("runs")

SEGMENTS = ["s1", "s23", "s45", "s6p"]          # innings 1, 2-3, 4-5, 6-end
SEG_LABEL = {"s1": "1st inning", "s23": "innings 2-3", "s45": "innings 4-5", "s6p": "innings 6+"}
X_COLS = ["lineup_woba", "lineup_obp", "lineup_iso", "lineup_k",
          "opp_sp_fip", "opp_sp_xfip", "opp_sp_whip", "opp_sp_kbb", "opp_sp_depth",
          "opp_pen_top", "opp_pen_all", "opp_der", "is_home", "lg_rpg", "is_postseason"]
TRAIN = range(2016, 2024)
TEST = [2024, 2025, 2026]
N_SIMS = 10000


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def side_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: one per batting team, with the opponent's pitching/defense."""
    rows = []
    for me, opp, home in (("home", "away", 1), ("away", "home", 0)):
        d = pd.DataFrame({
            "game_id": feats["game_id"], "season": feats["season"], "date": feats["date"],
            "round": feats["round"], "is_home": home, "lg_rpg": feats["lg_rpg"],
            "is_postseason": feats["is_postseason"],
            "lineup_woba": feats[f"{me}_lineup_woba"], "lineup_obp": feats[f"{me}_lineup_obp"],
            "lineup_iso": feats[f"{me}_lineup_iso"], "lineup_k": feats[f"{me}_lineup_k"],
            "opp_sp_fip": feats[f"{opp}_sp_fip"], "opp_sp_xfip": feats[f"{opp}_sp_xfip"],
            "opp_sp_whip": feats[f"{opp}_sp_whip"], "opp_sp_kbb": feats[f"{opp}_sp_kbb"],
            "opp_sp_depth": feats[f"{opp}_sp_depth"], "opp_pen_top": feats[f"{opp}_pen_top"],
            "opp_pen_all": feats[f"{opp}_pen_all"], "opp_der": feats[f"{opp}_der"],
        })
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_training() -> pd.DataFrame:
    """Side rows joined with actual runs by segment (games with a full first 5 innings)."""
    feats = pd.read_csv(C.DATA_DIR / "features.csv", parse_dates=["date"])
    feats = feats[feats["season"] >= 2016]
    g = data.load("games").set_index("game_id")
    sides = side_rows(feats)
    side = np.where(sides["is_home"] == 1, "home", "vis")
    def col(name):
        h = g.loc[sides["game_id"], f"home_{name}"].values
        v = g.loc[sides["game_id"], f"vis_{name}"].values
        return np.where(side == "home", h, v)
    r1, f3, f5, final = col("r1"), col("f3"), col("f5"), col("score")
    sides["s1"], sides["s23"], sides["s45"], sides["s6p"] = r1, f3 - r1, f5 - f3, final - f5
    sides = sides.dropna(subset=SEGMENTS)
    return sides[(sides[SEGMENTS] >= 0).all(axis=1)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def nb_loglik(y, mu, r):
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))).sum()


class RunsModel:
    """Per-segment Poisson mean models + negative-binomial dispersion."""

    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.models, self.r = {}, {}

    def fit(self, df: pd.DataFrame):
        for s in SEGMENTS:
            m = make_pipeline(StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=1000))
            m.fit(df[X_COLS], df[s])
            mu = m.predict(df[X_COLS])
            grid = np.exp(np.linspace(np.log(0.05), np.log(20), 80))
            self.r[s] = float(grid[np.argmax([nb_loglik(df[s].values, mu, r) for r in grid])])
            self.models[s] = m
        return self

    def means(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {s: self.models[s].predict(df[X_COLS]) for s in SEGMENTS}

    def p_zero(self, mu, s):
        r = self.r[s]
        return (r / (r + mu)) ** r

    def simulate(self, mu: dict, n=N_SIMS, rng=None) -> dict[str, np.ndarray]:
        """Draw segment runs; ``mu`` maps segment -> array of means (one per team-row)."""
        rng = rng or np.random.default_rng(0)
        out = {}
        for s in SEGMENTS:
            m = np.asarray(mu[s], dtype=float)[:, None]
            r = self.r[s]
            out[s] = rng.negative_binomial(r, r / (r + m), size=(m.shape[0], n)).astype(np.int16)
        return out


# --------------------------------------------------------------------------- #
# Game-level outputs
# --------------------------------------------------------------------------- #

def game_outputs(model: RunsModel, home_row: dict, away_row: dict, n=N_SIMS, seed=0,
                 lines=(("F5", 4.5), ("Full", 8.5))) -> dict:
    """All first-inning / F3 / F5 / full-game numbers for one matchup."""
    df = pd.DataFrame([home_row, away_row])
    mu = model.means(df)
    sim = model.simulate(mu, n=n, rng=np.random.default_rng(seed))
    h = {s: sim[s][0] for s in SEGMENTS}
    a = {s: sim[s][1] for s in SEGMENTS}
    hf3, af3 = h["s1"] + h["s23"], a["s1"] + a["s23"]
    hf5, af5 = hf3 + h["s45"], af3 + a["s45"]
    hfull, afull = hf5 + h["s6p"], af5 + a["s6p"]
    # Extra innings: a full-game tie is decided by a coin flip weighted toward the home team.
    tie = hfull == afull
    home_wins_full = (hfull > afull) | (tie & (np.random.default_rng(seed + 1).random(n) < 0.52))
    p_nrfi = model.p_zero(mu["s1"][0], "s1") * model.p_zero(mu["s1"][1], "s1")
    f5_margin = hf5 - af5
    out = {
        "exp_runs": {"home": {"1st": mu["s1"][0], "F3": mu["s1"][0] + mu["s23"][0],
                              "F5": mu["s1"][0] + mu["s23"][0] + mu["s45"][0], "Full": sum(m[0] for m in mu.values())},
                     "away": {"1st": mu["s1"][1], "F3": mu["s1"][1] + mu["s23"][1],
                              "F5": mu["s1"][1] + mu["s23"][1] + mu["s45"][1], "Full": sum(m[1] for m in mu.values())}},
        "f5_avg_incl0": (hf5.mean(), af5.mean()),
        "f5_avg_excl0": (hf5[hf5 > 0].mean(), af5[af5 > 0].mean()),
        "nrfi": float(p_nrfi),
        "f3": {"home": (hf3 > af3).mean(), "tie": (hf3 == af3).mean(), "away": (hf3 < af3).mean()},
        "f5": {"home": (f5_margin > 0).mean(), "tie": (f5_margin == 0).mean(), "away": (f5_margin < 0).mean()},
        "f5_margin_mode": int(pd.Series(f5_margin[f5_margin != 0]).mode().iloc[0]) if (f5_margin != 0).any() else 0,
        "full_home_win": home_wins_full.mean(),
        "totals": {},
    }
    tot = {"F3": hf3 + af3, "F5": hf5 + af5, "Full": hfull + afull}
    out["median_total"] = {k: float(np.median(v)) for k, v in tot.items()}
    out["mean_total"] = {k: float(v.mean()) for k, v in tot.items()}
    for k, v in tot.items():
        base = {"F3": 2.5, "F5": 4.5, "Full": 8.5}[k]
        for line in (base - 1, base, base + 1):
            out["totals"][(k, line)] = ((v > line).mean(), (v < line).mean())
    for k, line in lines:
        v = tot[k]
        out["totals"][(k, line)] = ((v > line).mean(), (v < line).mean())
    out["team_totals_f5"] = {"home": {l: (hf5 > l).mean() for l in (1.5, 2.5)},
                             "away": {l: (af5 > l).mean() for l in (1.5, 2.5)}}
    return out


def matchup_rows(row: dict, is_post: int = 1) -> tuple[dict, dict]:
    """Home and away batting rows from a LeagueState.matchup() feature row."""
    feats = pd.DataFrame([{**row, "game_id": "x", "season": 0, "date": pd.NaT, "round": "",
                           "is_postseason": is_post}])
    s = side_rows(feats)
    return s.iloc[0].to_dict(), s.iloc[1].to_dict()


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def backtest(model: RunsModel, test: pd.DataFrame, n=2000) -> dict:
    """Out-of-sample checks on game level: NRFI, totals, F5 winner and F5 'lean' records."""
    home = test[test.is_home == 1].set_index("game_id")
    away = test[test.is_home == 0].set_index("game_id")
    ids = home.index.intersection(away.index)
    home, away = home.loc[ids], away.loc[ids]
    mh, ma = model.means(home), model.means(away)
    rng = np.random.default_rng(0)
    sh, sa = model.simulate(mh, n, rng), model.simulate(ma, n, rng)
    hf5 = sh["s1"] + sh["s23"] + sh["s45"]
    af5 = sa["s1"] + sa["s23"] + sa["s45"]
    hfull, afull = hf5 + sh["s6p"], af5 + sa["s6p"]
    act = {
        "nrfi": ((home.s1 == 0) & (away.s1 == 0)).values,
        "hf5": (home.s1 + home.s23 + home.s45).values, "af5": (away.s1 + away.s23 + away.s45).values,
    }
    act["hfull"] = act["hf5"] + home.s6p.values
    act["afull"] = act["af5"] + away.s6p.values
    p_nrfi = model.p_zero(mh["s1"], "s1") * model.p_zero(ma["s1"], "s1")
    res = {"games": len(ids)}

    def ll(y, p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    base = act["nrfi"].mean()
    res["nrfi"] = {"actual_rate": base, "pred_rate": float(p_nrfi.mean()),
                   "logloss": ll(act["nrfi"], p_nrfi), "logloss_const": ll(act["nrfi"], np.full(len(ids), base))}
    bins = pd.qcut(p_nrfi, 5, duplicates="drop")
    res["nrfi_bins"] = pd.DataFrame({"pred": p_nrfi, "act": act["nrfi"]}).groupby(bins, observed=True).mean()

    for name, ph, pa, line in (("F5 total", hf5, af5, 4.5), ("Full total", hfull, afull, 8.5)):
        p_over = ((ph + pa) > line).mean(axis=1)
        key = "hf5" if name == "F5 total" else "hfull"
        y = ((act[key] + act["a" + key[1:]]) > line).astype(float)
        res[name] = {"line": line, "pred_mean_total": float((ph + pa).mean()),
                     "act_mean_total": float((act[key] + act["a" + key[1:]]).mean()),
                     "pred_over": float(p_over.mean()), "act_over": float(y.mean()),
                     "logloss": ll(y, p_over), "logloss_const": ll(y, np.full(len(y), y.mean()))}

    # F5 winner leans (ties = push), like the "F5 win % discounting ties >= X" rule.
    m = hf5 - af5
    p_home = (m > 0).mean(axis=1)
    p_away = (m < 0).mean(axis=1)
    p_home_nt = p_home / np.clip(p_home + p_away, 1e-9, None)
    act_m = act["hf5"] - act["af5"]
    act_full_home = act["hfull"] > act["afull"]
    rows = []
    for t in (0.50, 0.55, 0.58, 0.60, 0.62, 0.65):
        pick_home = p_home_nt >= t
        pick_away = (1 - p_home_nt) >= t
        picks = pick_home | pick_away
        won = np.where(pick_home, act_m > 0, act_m < 0)[picks]
        push = (act_m == 0)[picks]
        full_won = np.where(pick_home, act_full_home, ~act_full_home)[picks]
        n_dec = (~push).sum()
        rows.append({"threshold": t, "plays": int(picks.sum()), "W": int((won & ~push).sum()),
                     "L": int((~won & ~push).sum()), "P": int(push.sum()),
                     "win%_ex_push": float((won & ~push).sum() / max(n_dec, 1)),
                     "full_game_win%": float(full_won.mean()) if picks.any() else np.nan})
    res["f5_leans"] = pd.DataFrame(rows)
    return res


def save(model, name):
    with open(C.MODEL_DIR / f"runs_{name}.pkl", "wb") as f:
        pickle.dump(model, f)


def load(name="production") -> RunsModel:
    with open(C.MODEL_DIR / f"runs_{name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    df = load_training()
    train, test = df[df.season.isin(TRAIN)], df[df.season.isin(TEST)]
    model = RunsModel().fit(train)
    print("Negative-binomial dispersion by segment:", {s: round(r, 2) for s, r in model.r.items()})
    mu = model.means(test)
    print("\nExpected vs actual runs per team, 2024-2026 (out of sample):")
    for s in SEGMENTS:
        print(f"  {SEG_LABEL[s]:12s} predicted {mu[s].mean():.3f}   actual {test[s].mean():.3f}   "
              f"P(0) predicted {model.p_zero(mu[s], s).mean():.3f} actual {(test[s] == 0).mean():.3f}")
    res = backtest(model, test)
    print(f"\nBacktest on {res['games']} games, 2024-2026:")
    n = res["nrfi"]
    print(f"  NRFI: actual {n['actual_rate']:.1%}, predicted {n['pred_rate']:.1%}; "
          f"log loss {n['logloss']:.4f} vs constant-rate {n['logloss_const']:.4f}")
    print("  NRFI calibration (quintiles of predicted NRFI%):")
    print(res["nrfi_bins"].round(3).to_string())
    for k in ("F5 total", "Full total"):
        r = res[k]
        print(f"  {k} o/u {r['line']}: mean total predicted {r['pred_mean_total']:.2f} actual {r['act_mean_total']:.2f}; "
              f"P(over) predicted {r['pred_over']:.1%} actual {r['act_over']:.1%}; "
              f"log loss {r['logloss']:.4f} vs constant {r['logloss_const']:.4f}")
    print("\n  F5 moneyline leans (pick the side whose F5 win% excluding ties >= threshold):")
    print(res["f5_leans"].round(3).to_string(index=False))
    res["f5_leans"].to_csv(C.DATA_DIR / "f5_leans_backtest.csv", index=False)
    save(RunsModel().fit(df[df.date < pd.Timestamp("2026-09-29")]), "through_2026_regular")
    save(RunsModel().fit(df), "production")
    print("\nSaved runs models to", C.MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import runs
    runs.main()
