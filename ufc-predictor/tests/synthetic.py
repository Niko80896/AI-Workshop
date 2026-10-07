"""Random but well-formed fights/fighters tables in the project schema (for tests)."""
import numpy as np
import pandas as pd

from ufc_common import FIGHT_COLS, STAT_COLS


def make_data(n_fighters=40, n_dates=60, fights_per_date=4, seed=0):
    """Return (fights, fighters) with random results/stats; ~10% of fights lack stats."""
    rng = np.random.default_rng(seed)
    ids = [f"f{i:03d}" for i in range(n_fighters)]
    fighters = pd.DataFrame({
        "fighter_id": ids,
        "fighter_url": [f"http://ufcstats.com/fighter-details/{i}" for i in ids],
        "name": [f"Fighter {i}" for i in range(n_fighters)],
        "height_in": rng.integers(64, 78, n_fighters).astype(float),
        "reach_in": rng.integers(64, 82, n_fighters).astype(float),
        "stance": rng.choice(["Orthodox", "Southpaw", "Switch"], n_fighters, p=[.7, .2, .1]),
        "dob": [f"{y}-0{m}-15" for y, m in zip(rng.integers(1980, 2000, n_fighters), rng.integers(1, 9, n_fighters))],
    })
    dates = pd.date_range("2015-01-01", periods=n_dates, freq="21D")
    rows = []
    for d in dates:
        for j in range(fights_per_date):
            a, b = rng.choice(n_fighters, 2, replace=False)
            res = rng.choice(["W", "L", "D", "NC"], p=[.6, .37, .02, .01])
            five = rng.random() < .15
            row = {
                "fight_id": f"{d:%Y%m%d}{j}", "fight_url": f"u{d:%Y%m%d}{j}", "event_name": f"E {d:%Y-%m-%d}",
                "event_url": f"e{d:%Y%m%d}", "date": d.strftime("%Y-%m-%d"), "weight_class": "Lightweight",
                "title_fight": bool(five and rng.random() < .5),
                "method": rng.choice(["KO/TKO", "Submission", "Decision - Unanimous"]), "method_detail": None,
                "end_round": int(rng.integers(1, 4)), "end_time": f"{rng.integers(0, 5)}:{rng.integers(10, 60)}",
                "time_format": "5 Rnd (5-5-5-5-5)" if five else "3 Rnd (5-5-5)",
                "fighter_a": f"Fighter {a}", "fighter_a_url": fighters.fighter_url[a],
                "fighter_b": f"Fighter {b}", "fighter_b_url": fighters.fighter_url[b],
                "result_a": res, "result_b": {"W": "L", "L": "W"}.get(res, res),
            }
            has = rng.random() > .1
            for p in ("a", "b"):
                for c in STAT_COLS:
                    if c.endswith("_att"):
                        continue
                    if c.endswith("_landed"):
                        att = float(rng.integers(0, 120))
                        row[f"{p}_{c}"] = float(rng.integers(0, att + 1)) if has else np.nan
                        row[f"{p}_{c[:-7]}_att"] = att if has else np.nan
                    else:
                        row[f"{p}_{c}"] = float(rng.integers(0, 300 if c == "ctrl_sec" else 3)) if has else np.nan
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=FIGHT_COLS), fighters


def scramble_from(fights, cutoff, seed=1):
    """Return a copy where every fight on/after ``cutoff`` has random results, methods and stats."""
    rng = np.random.default_rng(seed)
    f = fights.copy()
    m = pd.to_datetime(f["date"]) >= pd.Timestamp(cutoff)
    n = int(m.sum())
    f.loc[m, "result_a"] = rng.choice(["W", "L", "D", "NC"], n)
    f.loc[m, "result_b"] = f.loc[m, "result_a"].map({"W": "L", "L": "W"}).fillna(f.loc[m, "result_a"])
    f.loc[m, "method"] = rng.choice(["KO/TKO", "Submission", "Decision - Split"], n)
    f.loc[m, "end_round"] = rng.integers(1, 4, n)
    stat = [c for c in f.columns if c[:2] in ("a_", "b_")]
    f.loc[m, stat] = rng.integers(0, 500, (n, len(stat))).astype(float)
    return f
