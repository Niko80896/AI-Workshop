# MLB Playoff Analysis & Prediction

Estimates win probabilities for single playoff games, full series, and the
whole bracket (each team's odds to win the pennant and World Series), and
explains what drives each prediction.

```bash
pip install -r requirements.txt
python data.py        # ~1 min: download (cached) + parse 2015-2025 into data/*.csv
python features.py    # ~2 min: leakage-free features for every game -> data/features.csv
python model.py       # ~20 s: evaluate, backtest, save models
python report.py bracket
python report.py game "Cubs" "Padres"               # first team is home
python report.py series "Dodgers" "Blue Jays" --round WS
python -m pytest -q tests
```

## Data sources (and why)

The network this was built on blocks FanGraphs, Baseball Savant,
Baseball-Reference and the MLB Stats API, so the core pipeline uses
**Retrosheet** (via the Chadwick Bureau GitHub mirror): every regular-season
and postseason game 2015-2025 with full play-by-play, game logs and player
handedness. Statcast-only metrics (xwOBA, barrel rate, xERA, OAA, framing) and
FanGraphs-only ones (SIERA, DRS, BsR) are therefore replaced with
play-by-play equivalents (see below). `python data.py --external` tries the
pybaseball/statsapi pulls anyway and saves them to `data/external/` when the
hosts are reachable.

| Wanted | Used here |
|---|---|
| wRC+, OBP/SLG | wOBA (fixed linear weights), regressed, per batter and vs LHP/RHP |
| ISO, K%, BB% | computed per batter from play-by-play |
| FIP, xFIP, K-BB%, HR/9 | computed per pitcher (xFIP uses fly balls x league HR/FB) |
| xERA / SIERA | regressed ERA as a results check (no Statcast) |
| OAA / DRS | defensive efficiency (share of balls in play turned into outs) |
| Catcher framing | not available without pitch locations |
| BsR | stolen-base runs (0.2 x SB - 0.41 x CS) |

## Step 1: Data (`data.py`, `retrosheet.py`)

Every download is cached in `data/cache/` and never fetched again. Outputs:

- `games.csv`: date, teams, score, starting pitchers, round (REG/WC/DS/LCS/WS)
- `pitcher_games.csv`: per pitcher per game (BF, outs, K, BB, HBP, HR, batted-ball types, pitches, ER, SP/RP)
- `batter_games.csv`: per batter per game, split by opposing pitcher hand
- `team_games.csv`: batting totals, SB/CS, balls in play and outs for defense
- `players.csv`, `standings.csv`

The parser is checked two ways. Exactly 3 outs are counted in 99.5% of all
half-innings (the rest are walk-offs). Season totals match published stats:
Skubal 2025 195.1 IP and 241 K, Raleigh 60 HR.

## Step 2: Features (`features.py`)

`LeagueState` is fed games in date order. Features for a game on day D are
read **before** any game on day D is added, so doubleheaders can't see each
other. The same `matchup()` call builds training rows and simulated playoff
games. Stats use exponential time decay, so recent games count more and last
season fades out naturally, and they are regressed toward league average.

Features (home minus away): Elo; Pythagorean win%; recent run differential;
lineup wOBA / ISO / K% against the opposing starter's hand; platoon share;
starter FIP / xFIP / K-BB% / regressed ERA / innings per start / short rest;
**top-5 bullpen** FIP and full-bullpen FIP; bullpen pitches in the last 3 days
and top relievers likely unavailable; defensive efficiency; baserunning; team
rest days; postseason flag.

**Roster changes are automatic.** Lineups are built from the players who
actually played for the team in the last 21 days, weighted by playing time,
so deadline acquisitions count with their full history and departed players
drop out.

**Injuries are manual.** Add rows to `data/manual/injuries.csv` (see
`injuries_example.csv`). An injured starter leaves the rotation, an injured
reliever leaves the top-5 bullpen, and an injured hitter's playing time goes
to replacement level.

**Leakage tests** (`tests/test_leakage.py`) scramble every score and stat line
from a cutoff date onward and assert that features up to that date are
bit-for-bit unchanged. They also check that a game's own result never affects
its features.

## Step 3: Models (`model.py`)

Time-based split: train 2016-22, tune on 2023, test on 2024-25. 2015 is a
burn-in year. Test results on 4,949 games:

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Home team always wins | 53.2% | 0.6911 | 0.2490 |
| Higher Pythag win% wins | 56.0% | 0.6862 | 0.2465 |
| Log5 (Pythag) | 56.0% | 0.6817 | 0.2444 |
| Elo | 56.0% | 0.6821 | 0.2446 |
| Logistic regression | **56.6%** | **0.6764** | **0.2418** |
| Gradient boosting | 56.1% | 0.6789 | 0.2430 |
| Ensemble (used) | 56.4% | 0.6775 | 0.2423 |

The ensemble blend weight was chosen on the 2023 validation season, not on
the test set. See `charts/calibration_test.png`: the ensemble sits on the
diagonal, while Elo is overconfident at the extremes.

**Postseason backtest** (walk-forward: each October predicted by a model
trained only on earlier games, 369 games, 2017-25): model log loss 0.690,
Elo 0.684, Log5 0.689, home-rate 0.690. Playoff games are between evenly
matched teams, and with ~40 games a year none of these differences is
statistically meaningful. Expect coin-flip-plus odds; the value is in
calibration and in the explanations, not in picking winners.

Feature importance (logistic regression standardized coefficients and
gradient-boosting permutation importance) is saved to
`data/feature_importance.csv`. The biggest drivers are lineup wOBA against
the starter's hand, starter xFIP, Elo, top-5 bullpen FIP and defense.

Two models are saved. `through_2025_regular` is frozen before the 2025
postseason, so the 2025 demo is an honest out-of-sample prediction.
`production` uses all games.

## Step 4: Simulation (`simulate.py`)

Uses the current format: Wild Card best-of-3 (higher seed hosts all games),
Division Series 2-2-1, LCS and World Series 2-3-2 (World Series home field
goes to the better record). Each game is simulated with that day's starter.
The rotation is ranked best-first, and the best starter with at least 4 days
of rest pitches. Off days therefore let an ace pitch Games 1 and 5, and Wild
Card teams start the Division Series without their ace.

Bracket inputs live in `data/manual/bracket_<season>.json`:

- `seeds`: per league, 1-6
- `calendar`: game dates per round; leave it out to use the default template
- `rotations`: e.g. `{"LAN": ["Shohei Ohtani", "Blake Snell", ...]}`
- `probables`: `[{"round": "WS", "game": 1, "team": "LAN", "pitcher": "Shohei Ohtani"}]`

`--asof YYYY-MM-DD` locks in every game played before that date. For 2025 the
results come from Retrosheet. For 2026, enter games in
`data/manual/results_2026.csv`.

```bash
python simulate.py --asof 2025-10-10 --sims 20000
```

## Step 5: Reports (`report.py`)

Each command prints win probabilities, the top factors (each factor's
contribution to the log-odds in the logistic regression), and a head-to-head
stat table. Charts are saved to `charts/`:

- `bracket_*.png`: odds of reaching each round
- `game_*.png` / `series_*.png`: what drives the prediction
- `staff_*.png`: rotation and top-5 bullpen breakdown

## Known limitations

- **No 2026 data offline.** Retrosheet's newest season is 2025. To predict
  2026, data for that season has to be added (for example by allowing
  statsapi.mlb.com and the FanGraphs/Savant hosts); until then
  `bracket_2026.json` runs on ratings through 2025.
- Bullpen fatigue is known for the first simulated day only; later simulated
  games assume a rested bullpen.
- Auto-rotations need at least 4.4 innings per start, so pitchers on a
  ramp-up (e.g. Ohtani in 2025) need a manual `rotations` entry.
- Runs and ERA are not park-adjusted (ballpark factors are an optional extra).

## Layout

```
config.py      paths and constants
retrosheet.py  event-file parser
data.py        download/cache/clean -> data/*.csv (+ optional pybaseball/statsapi)
features.py    LeagueState, leakage-free features
model.py       baselines, LR, GBM, evaluation, backtest
simulate.py    Monte Carlo series/bracket simulation
report.py      CLI + charts
plotstyle.py   chart palette/style
tests/         parser, leakage, simulator tests
data/manual/   bracket, results, injuries (you edit these)
```
