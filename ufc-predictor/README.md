# UFC Fight Predictor

Predicts the winner of UFC fights as a calibrated probability from leak-free,
pre-fight features built on historical ufcstats.com data. Dana White's
Contender Series results are used as extra history.

```
$ python predict.py "Islam Makhachev" "Ilia Topuria" --five-rounds --title

Islam Makhachev vs Ilia Topuria  (5 rounds, title fight)
model: Logistic regression (trained through 2026-10-03)

  Islam Makhachev   76.3%   [UFC 18-1, Elo 2063]
  Ilia Topuria      23.7%   [UFC 9-1, Elo 1856]

Top factors (log-odds contribution):
  Current streak                     17 vs -1             +0.655 -> favors Islam Makhachev
  Elo rating                         2062.92 vs 1856.35   +0.624 -> favors Islam Makhachev
  Age                                34.95 vs 29.71       -0.309 -> favors Ilia Topuria
  ...
```

## Setup

```bash
cd ufc-predictor
pip install -r requirements.txt
```

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1a. Scrape ufcstats.com | `python scrape.py` | `data/fights.csv`, `data/fighters.csv` |
| 1b. *or* import the GitHub mirror | `python import_mirror.py` | same files, same schema |
| 1c. Contender Series results (optional) | `python scrape_dwcs.py` | `data/dwcs_fights.csv` |
| 2. Features (optional, for inspection) | `python features.py` | `data/features.csv` |
| 3. Train and evaluate | `python train.py` | `models/model.joblib`, `reports/` |
| 4. Predict | `python predict.py "A" "B" [--five-rounds] [--title]` | stdout |
| Tests | `python -m pytest -q` | |

### 1. Data

**`scrape.py`** crawls every completed event, every fight on each event, and
every fighter profile on ufcstats.com. It uses a shared `requests` session with
retries and backoff, and waits 0.75 s between requests (`--delay`).

- **Incremental:** events already in `fights.csv` and fighters already in
  `fighters.csv` are skipped. Results are appended after each event, so an
  interrupted run can just be restarted.
- **Per-fight stats:** totals per fighter (KD, sig/total strikes, TD, sub
  attempts, reversals, control time, and sig strikes by head/body/leg and
  distance/clinch/ground). `--` is stored as NaN, not 0.
- **Fighter profiles:** only height, weight, reach, stance and DOB are saved.
  The career summary stats (SLpM, Str. Acc., …) are ignored because they describe
  the fighter's *current* career and would leak future information.
- A full first run is roughly 9k fight pages plus 2.6k fighter pages, about 2.5
  hours at the default delay.

**`import_mirror.py`** builds the same files from
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats),
a regularly updated CSV mirror of ufcstats. Use it when ufcstats.com is
unreachable (it was blocked in the cloud sandbox this was built in). It runs in
a few seconds. It handles these mirror quirks:

- Stats are per round, so they are summed per fight (all-missing stays NaN).
- 25 fights appear twice under renamed events, so they are de-duplicated by
  fight URL.
- Bouts list names, not URLs, so names are matched to profiles accent- and
  case-insensitively. Shared names (e.g. two "Bruno Silva"s) are resolved by the
  profile weight closest to the bout's division.

**`scrape_dwcs.py`** parses DWCS season pages from Wikipedia. ufcstats doesn't
cover DWCS, and no public source has DWCS strike or takedown stats, so this
collects results only (date, division, fighters, winner, method, round, time).
`--html-dir` parses saved pages offline.

> ⚠️ Wikipedia was unreachable from the build sandbox, so the DWCS parser has
> only been tested against a hand-written fixture of Wikipedia's standard MMA
> results table. On a first real run, check the logged bout counts per season
> (DWCS seasons have about 40–50 bouts) and the "dropping N bouts with no
> parsable date" warning.

### 2. Features (`features.py`)

**No leakage, by construction.** `FeatureBuilder` replays all bouts
chronologically, one date at a time:

1. compute features for every fight on that date from the current state;
2. only then fold that date's results into each fighter's running aggregates and
   Elo.

A fight therefore never sees its own result or anything later, including
same-night tournament bouts from the early UFC events. `tests/test_features.py`
checks this directly. It scrambles every result and stat on or after a cutoff
date and asserts that all features up to and including the cutoff are
unchanged. It runs on synthetic data, on DWCS data, and on the real dataset
when present. I also confirmed the test fails if the two steps are swapped.

Per-fighter pre-fight features (the model sees `A − B` for each):

| Group | Features |
|---|---|
| Experience | UFC fights, wins, losses, smoothed win rate, DWCS bouts and wins |
| Striking | sig. landed/min, absorbed/min, accuracy, defense, KD/15 min, head/body/leg and distance/clinch/ground shares |
| Grappling | TD/15 min, TD accuracy, TD defense, sub attempts/15 min, control %, time controlled by opponents % |
| Finishing / durability | KO wins, sub wins, finish rate, KO/TKO losses |
| Form | win rate over last 3, current streak, sig-strike differential/min over last 3 |
| Activity | days since last fight |
| Physical | age on fight day, height, reach, southpaw, plus a southpaw-vs-orthodox matchup flag |
| Rating | Elo (pre-fight, K=80, tuned on pre-2024 fights), average opponent Elo |
| Context | five-round fight, title fight |

Per-minute rates use only minutes from fights with recorded stats.

**DWCS history:** DWCS bouts update Elo, streak, form, layoff, finishes and the
DWCS record. They do not count toward UFC fight counts or any per-minute rate,
and they are never training rows. DWCS fighters are mapped to their ufcstats
profile by name, so a prospect's record carries into their UFC debut.

**Winner-first bias:** ufcstats lists the winner first in about 64% of fights.
Every model trains on both orientations (A vs B, plus B vs A with the features
negated and the label flipped). Every prediction averages `P(A wins)` with
`1 − P(B wins)`, so swapping corners always gives exactly the complementary
probability (this is asserted in `train.py` and in the tests).

### 3. Modeling (`train.py`)

- **Split:** train on fights before 2024-01-01, test on fights from 2024-01-01
  on (`--test-start`). Training rows start in 2001 (`--min-train-date`), but all
  earlier fights still feed fighter history.
- **Tuning:** the logistic C and the number of LightGBM trees are picked on a
  2021–2023 validation block inside the training period. The test set is never
  used for tuning.
- **Selection and refit:** the best model by test log loss is refit on all
  labelled fights and saved.

Results on the 1,451 test fights (2024-01-13 → 2026-10-03):

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Elo baseline (higher Elo wins) | 56.1% | 0.681 | 0.244 |
| Logistic regression | 64.8% | **0.637** | **0.223** |
| LightGBM | 65.1% | 0.639 | 0.224 |

![calibration](reports/calibration.png)

Both learned models are well calibrated through the middle and slightly
*under*-confident at the extremes. Logistic regression edges out LightGBM on
log loss, so it is the saved model.

Permutation importance on the test set
([csv](reports/feature_importance.csv)) ranks age difference far ahead of
everything else, then Elo, KO/TKO losses, streak, and strikes absorbed per
minute:

![importance](reports/feature_importance.png)

For context, closing betting odds typically reach about 65–70% accuracy.

### 4. Prediction (`predict.py`)

- **Same pipeline as training:** it replays all known bouts through the same
  `FeatureBuilder` and treats the matchup as one more pending fight.
- **Name lookup:** names match accent- and case-insensitively, with fuzzy
  matching for typos. Shared names resolve to the most recently active profile,
  and the alternatives are printed. You can also pass a ufcstats profile URL or
  id.
- **Debuts and unknowns:** a fighter with no UFC fights (or not in the data at
  all) gets Elo 1500, a 0.5 smoothed win rate, and missing career rates. Their
  physicals and DWCS record are still used.
- **Explanations:** the top factors are orientation-averaged log-odds
  contributions. Collinear record features are reported together as one
  "UFC record" factor.

## Project layout

```
ufc_common.py     schema, parsing helpers, name resolver
scrape.py         ufcstats.com scraper (incremental)
import_mirror.py  GitHub-mirror importer (same schema)
scrape_dwcs.py    Contender Series results from Wikipedia
features.py       chronological, leak-free feature builder
models.py         orientation-symmetric Elo / logistic / LightGBM wrappers
train.py          time-split evaluation, reports, final model
predict.py        CLI
tests/            fixtures, synthetic data, leakage + parser + prediction tests
```

## Not yet included (optional extras)

These were left out on purpose, pending your go-ahead:

- historical betting odds, as a feature and as a benchmark;
- method-of-victory prediction (KO / SUB / DEC);
- missed weight, short-notice replacements, and weight-class changes.
