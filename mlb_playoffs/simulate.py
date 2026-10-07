"""Monte Carlo series and bracket simulation (2022+ 12-team format).

Rounds and home-field patterns:
    Wild Card   best-of-3, higher seed hosts every game (seeds 3v6, 4v5)
    Division    best-of-5, 2-2-1 (seeds 1 and 2 host the WC winners)
    LCS         best-of-7, 2-3-2, better seed hosts games 1, 2, 6, 7
    World Series best-of-7, 2-3-2, better regular-season record hosts

Games are simulated one at a time with the projected starter for that game:
each team's rotation is ordered best-first and the first starter with
enough rest (default 4 days) on that calendar date takes the ball, so off
days let aces pitch Games 1 and 4/5 and teams coming out of the Wild Card
round start their DS without their ace.

Completed games (before --asof) are locked in from actual results, and the
bracket JSON in data/manual/ can override rotations and individual probable
pitchers. Injured players listed in data/manual/injuries.csv are removed
from rotations, bullpens and lineups.

Usage:
    python simulate.py                          # 2025 bracket from the start
    python simulate.py --asof 2025-10-10        # lock in games played before Oct 10
    python simulate.py --season 2026 --sims 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C
import data
import features as F
from model import load_model

log = logging.getLogger("simulate")

ROUNDS = ["WC", "DS", "LCS", "WS"]
BEST_OF = {"WC": 3, "DS": 5, "LCS": 7, "WS": 7}
# Which side hosts game i (True = higher seed / home-field team).
HOME_PATTERN = {
    "WC": [True, True, True],
    "DS": [True, True, False, False, True],
    "LCS": [True, True, False, False, False, True, True],
    "WS": [True, True, False, False, False, True, True],
}
# Default calendar (day offsets from the Wild Card opener) when none is given.
DEFAULT_OFFSETS = {"WC": [0, 1, 2], "DS": [4, 5, 7, 8, 10], "LCS": [12, 13, 15, 16, 17, 19, 20],
                   "WS": [24, 25, 27, 28, 29, 31, 32]}


# --------------------------------------------------------------------------- #
# Name resolution
# --------------------------------------------------------------------------- #

def resolve_team(name: str) -> str:
    """Accept 'Dodgers', 'LAD', 'Los Angeles Dodgers' or Retrosheet 'LAN'."""
    n = name.strip().lower()
    for code in C.TEAM_NAMES:
        if n in (code.lower(), C.TEAM_DISPLAY[code].lower(), C.TEAM_NAMES[code].lower()) \
                or n.endswith(" " + C.TEAM_NAMES[code].lower()):
            return C.FRANCHISE_ALIASES.get(code, code)
    raise SystemExit(f"Unknown team: {name!r}")


_players = None


def resolve_player(name: str, team: str | None = None, state: F.LeagueState | None = None) -> str:
    """Accept a Retrosheet id or a player name (prefers players on ``team``)."""
    global _players
    if _players is None:
        _players = data.load("players")
    if name in set(_players["player_id"]):
        return name
    m = _players[_players["name"].str.lower() == name.strip().lower()]
    if m.empty:
        raise SystemExit(f"Unknown player: {name!r}")
    if len(m) > 1 and team and state is not None:
        on_team = [p for p in m["player_id"]
                   if p in state.team_starters[team] or p in state.team_relievers[team]]
        if on_team:
            return on_team[0]
    return m["player_id"].iloc[0]


def display(team: str) -> str:
    return C.TEAM_DISPLAY.get(team, team)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def load_bracket(season: int) -> dict:
    path = C.MANUAL_DIR / f"bracket_{season}.json"
    b = json.loads(path.read_text())
    b["seeds"] = {lg: [resolve_team(t) for t in ts] for lg, ts in b["seeds"].items() if all(ts)}
    if not b["seeds"]:
        raise SystemExit(f"Fill in the seeds in {path} first.")
    return b


def load_injuries(season: int, as_of, state, path=None) -> frozenset:
    """Player ids marked 'out' on the as-of date in data/manual/injuries.csv."""
    path = C.MANUAL_DIR / "injuries.csv" if path is None else Path(path)
    if not path.exists():
        return frozenset()
    inj = pd.read_csv(path, dtype=str).fillna("")
    out = set()
    d = pd.Timestamp(as_of)
    for r in inj.to_dict("records"):
        if str(r["season"]) != str(season) or r["status"].lower() != "out":
            continue
        if r["from_date"] and pd.Timestamp(r["from_date"]) > d:
            continue
        if r["to_date"] and pd.Timestamp(r["to_date"]) < d:
            continue
        out.add(resolve_player(r["player"], resolve_team(r["team"]), state))
    return frozenset(out)


def actual_postseason_games(season: int, as_of) -> pd.DataFrame:
    """Completed postseason games before ``as_of`` (Retrosheet or manual CSV)."""
    g = data.load("games")
    g = g[(g.season == season) & (g["round"] != "REG") & (g.date < pd.Timestamp(as_of))]
    g = g.rename(columns={"visteam": "away", "hometeam": "home", "vis_score": "away_score",
                          "vis_sp": "away_sp"})
    g = g[["date", "round", "away", "home", "away_score", "home_score", "away_sp", "home_sp"]]
    manual = C.MANUAL_DIR / f"results_{season}.csv"
    if manual.exists():
        m = pd.read_csv(manual, parse_dates=["date"])
        if len(m):
            for side in ("away", "home"):
                m[side] = m[side].map(resolve_team)
            m = m[m.date < pd.Timestamp(as_of)]
            g = pd.concat([g, m], ignore_index=True)
    return g.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #

class PlayoffSimulator:
    """Simulate playoff series game-by-game with projected starting pitchers."""

    def __init__(self, season: int, as_of=None, model_name: str | None = None, seed: int = 0,
                 injuries=None):
        self.bracket = load_bracket(season)
        self.season = season
        cal = self.bracket.get("calendar") or {}
        wc_start = (pd.Timestamp(cal["WC"]["AL"][0]) if cal.get("WC") else
                    pd.Timestamp(as_of or f"{season}-09-30"))
        self.as_of = pd.Timestamp(as_of) if as_of else wc_start
        self.state = F.state_as_of(self.as_of)
        self.model = load_model(model_name or self.bracket.get("model", "production"))
        self.out = load_injuries(season, self.as_of, self.state, injuries)
        self.min_rest = self.bracket.get("min_rest_days", 4)
        self.day0 = self.as_of.toordinal()
        self.calendar = self._calendar(cal, wc_start)
        self.rotations = self._rotations()
        self.probables = self._probables()
        self.actual = actual_postseason_games(season, self.as_of)
        self.records = self._records()
        self.rng = np.random.default_rng(seed)
        self._cache: dict = {}
        # Last appearance day per pitcher, from real games through as-of.
        self.last_pitched = {}
        for team, rot in self.rotations.items():
            for p in rot:
                app = self.state.appearances.get(p)
                if app:
                    self.last_pitched[p] = app[-1][0]

    # -- setup helpers ----------------------------------------------------
    def _calendar(self, cal, wc_start) -> dict:
        out = {}
        for rnd in ROUNDS:
            for lg in ("AL", "NL", "MLB"):
                if cal.get(rnd, {}).get(lg):
                    out[(rnd, lg)] = [pd.Timestamp(d).toordinal() for d in cal[rnd][lg]]
                else:
                    out[(rnd, lg)] = [wc_start.toordinal() + o for o in DEFAULT_OFFSETS[rnd]]
        return out

    def _rotations(self) -> dict:
        rot = {}
        manual = self.bracket.get("rotations", {})
        for lg, teams in self.bracket["seeds"].items():
            for t in teams:
                key = next((k for k in manual if resolve_team(k) == t), None)
                if key:
                    rot[t] = [resolve_player(p, t, self.state) for p in manual[key]]
                    rot[t] = [p for p in rot[t] if p not in self.out]
                else:
                    rot[t] = self.state.rotation(t, self.day0, n=4, out=self.out)
        return rot

    def rotation_for(self, team) -> list[str]:
        """Rotation for any team (bracket teams are precomputed; others on demand)."""
        if team not in self.rotations:
            self.rotations[team] = self.state.rotation(team, self.day0, n=4, out=self.out)
            for p in self.rotations[team]:
                app = self.state.appearances.get(p)
                if app:
                    self.last_pitched[p] = app[-1][0]
        return self.rotations[team]

    def _probables(self) -> dict:
        out = {}
        for p in self.bracket.get("probables", []):
            t = resolve_team(p["team"])
            out[(p["round"], int(p["game"]), t)] = resolve_player(p["pitcher"], t, self.state)
        return out

    def _records(self) -> dict:
        st = data.load("standings")
        st = st[st.season == self.season]
        if st.empty:
            st = data.load("standings")
            st = st[st.season == st.season.max()]
        return dict(zip(st.team, st.pct))

    def league_of(self, team) -> str:
        return next((lg for lg, ts in self.bracket["seeds"].items() if team in ts), "MLB")

    def seed_of(self, team) -> int:
        return self.bracket["seeds"][self.league_of(team)].index(team) + 1

    # -- single game ------------------------------------------------------
    def game_prob(self, home, away, home_sp, away_sp, day, home_rest=1, away_rest=1,
                  home_sp_rest=5, away_sp_rest=5) -> float:
        """P(home wins), using state as of --asof with game-specific rest values."""
        fresh = day <= self.day0
        key = (home, away, home_sp, away_sp, min(home_rest, 4), min(away_rest, 4),
               home_sp_rest < 4, away_sp_rest < 4, fresh)
        p = self._cache.get(key)
        if p is None:
            row, _ = self.state.matchup(home, away, home_sp, away_sp, self.day0, is_post=1,
                                        rest=(min(home_rest, 4), min(away_rest, 4)),
                                        out=self.out, sp_rest=(home_sp_rest, away_sp_rest))
            if not fresh:   # bullpen fatigue beyond today is unknown
                row["pen_fatigue_diff"] = 0.0
                row["pen_tired_top_diff"] = 0
            p = float(self.model.predict_proba(pd.DataFrame([row]))[0])
            self._cache[key] = p
        return p

    def pick_starter(self, team, rnd, game_no, day, last) -> tuple[str, int]:
        """Override, else best-ranked starter with enough rest, else the most rested."""
        forced = self.probables.get((rnd, game_no, team))
        rot = self.rotation_for(team)
        if forced:
            sp = forced
        else:
            rested = [p for p in rot if day - last.get(p, -99) - 1 >= self.min_rest]
            sp = rested[0] if rested else max(rot, key=lambda p: day - last.get(p, -99))
        return sp, min(day - last.get(sp, -99) - 1, 10)

    # -- series -----------------------------------------------------------
    def played(self, rnd, a, b) -> list[dict]:
        if not hasattr(self, "_played"):
            self._played = defaultdict(list)
            for gm in self.actual.to_dict("records"):
                self._played[(gm["round"], frozenset((gm["home"], gm["away"])))].append(gm)
        return self._played.get((rnd, frozenset((a, b))), [])

    def sim_series(self, hi, lo, rnd, last, rng, lock=True) -> str:
        """Simulate one series; ``hi`` holds home-field advantage. Mutates ``last``."""
        need = BEST_OF[rnd] // 2 + 1
        wins = {hi: 0, lo: 0}
        days = self.calendar[(rnd, "MLB" if rnd == "WS" else self.league_of(hi))]
        played = self.played(rnd, hi, lo) if lock else []
        team_last_day = {hi: None, lo: None}
        for i, gm in enumerate(played):
            winner = gm["home"] if gm["home_score"] > gm["away_score"] else gm["away"]
            wins[winner] += 1
            d = pd.Timestamp(gm["date"]).toordinal()
            team_last_day[hi] = team_last_day[lo] = d
        n = len(played)
        while max(wins.values()) < need:
            day = days[min(n, len(days) - 1)]
            if day <= self.day0:   # scheduled before as-of but not in results: play "today"
                day = self.day0
            home, away = (hi, lo) if HOME_PATTERN[rnd][n] else (lo, hi)
            hsp, hrest = self.pick_starter(home, rnd, n + 1, day, last)
            asp, arest = self.pick_starter(away, rnd, n + 1, day, last)
            trest = {t: (day - team_last_day[t] - 1) if team_last_day[t] else 2 for t in (hi, lo)}
            p = self.game_prob(home, away, hsp, asp, day, trest[home], trest[away], hrest, arest)
            winner = home if rng.random() < p else away
            wins[winner] += 1
            last[hsp] = last[asp] = day
            team_last_day[hi] = team_last_day[lo] = day
            n += 1
        return max(wins, key=wins.get)

    def higher(self, a, b, rnd) -> tuple[str, str]:
        """(home-field team, other) for a matchup."""
        seeded = all(any(t in ts for ts in self.bracket["seeds"].values()) for t in (a, b))
        if rnd == "WS" or not seeded or self.league_of(a) != self.league_of(b):
            ra, rb = self.records.get(a, .5), self.records.get(b, .5)
            return (a, b) if ra >= rb else (b, a)
        return (a, b) if self.seed_of(a) < self.seed_of(b) else (b, a)

    def sim_bracket(self, rng) -> dict:
        """One full postseason. Returns the furthest round reached by each team."""
        last = dict(self.last_pitched)
        reached = {}
        pennant = {}
        for lg, s in self.bracket["seeds"].items():
            for t in s:
                reached[t] = "WC"
            reached[s[0]] = reached[s[1]] = "DS"
            w36 = self.sim_series(s[2], s[5], "WC", last, rng)
            w45 = self.sim_series(s[3], s[4], "WC", last, rng)
            reached[w36] = reached[w45] = "DS"
            ds1 = self.sim_series(*self.higher(s[0], w45, "DS"), "DS", last, rng)
            ds2 = self.sim_series(*self.higher(s[1], w36, "DS"), "DS", last, rng)
            reached[ds1] = reached[ds2] = "LCS"
            champ = self.sim_series(*self.higher(ds1, ds2, "LCS"), "LCS", last, rng)
            reached[champ] = "WS"
            pennant[lg] = champ
        al, nl = pennant.get("AL"), pennant.get("NL")
        ws = self.sim_series(*self.higher(al, nl, "WS"), "WS", last, rng)
        reached[ws] = "CHAMP"
        return reached

    def run(self, n: int = 10000) -> pd.DataFrame:
        """Odds of each team reaching each round, over ``n`` simulated postseasons."""
        order = ["WC", "DS", "LCS", "WS", "CHAMP"]
        counts = defaultdict(lambda: np.zeros(len(order)))
        for _ in range(n):
            for t, r in self.sim_bracket(self.rng).items():
                counts[t][: order.index(r) + 1] += 1
        rows = []
        for t, c in counts.items():
            rows.append({"team": display(t), "code": t, "league": self.league_of(t), "seed": self.seed_of(t),
                         "make_ds": c[1] / n, "make_lcs": c[2] / n, "pennant": c[3] / n, "win_ws": c[4] / n})
        return (pd.DataFrame(rows).sort_values(["win_ws", "pennant", "make_lcs", "make_ds"], ascending=False)
                .reset_index(drop=True))

    def series_odds(self, a, b, rnd="WS", n=10000, lock=False) -> dict:
        """Head-to-head series probability plus the game-1 starters used."""
        hi, lo = self.higher(a, b, rnd)
        wins = defaultdict(int)
        for _ in range(n):
            last = dict(self.last_pitched)
            wins[self.sim_series(hi, lo, rnd, last, self.rng, lock=lock)] += 1
        return {"home_field": hi, "p": {t: wins[t] / n for t in (hi, lo)}}

    def projected_games(self, a, b, rnd="WS") -> list[dict]:
        """Deterministic preview of every possible game: venue, starters, P(win)."""
        hi, lo = self.higher(a, b, rnd)
        last = dict(self.last_pitched)
        days = self.calendar[(rnd, "MLB" if rnd == "WS" else self.league_of(hi))]
        out = []
        played = self.played(rnd, a, b)
        for i in range(BEST_OF[rnd]):
            if i < len(played):   # completed: show the real starters and date
                gm = played[i]
                d = pd.Timestamp(gm["date"])
                last[gm["home_sp"]] = last[gm["away_sp"]] = d.toordinal()
                out.append({"game": i + 1, "date": d.date(), "home": gm["home"], "away": gm["away"],
                            "home_sp": gm["home_sp"], "away_sp": gm["away_sp"], "p_home": float("nan"),
                            "sp_rest": (5, 5)})
                continue
            day = max(days[i], self.day0)
            home, away = (hi, lo) if HOME_PATTERN[rnd][i] else (lo, hi)
            hsp, hrest = self.pick_starter(home, rnd, i + 1, day, last)
            asp, arest = self.pick_starter(away, rnd, i + 1, day, last)
            p = self.game_prob(home, away, hsp, asp, day, 1, 1, hrest, arest)
            last[hsp] = last[asp] = day
            out.append({"game": i + 1, "date": pd.Timestamp.fromordinal(day).date(), "home": home,
                        "away": away, "home_sp": hsp, "away_sp": asp, "p_home": p,
                        "sp_rest": (hrest, arest)})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--asof", default=None, help="lock in games played before this date (YYYY-MM-DD)")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--injuries", default=None, help="injuries CSV (default data/manual/injuries.csv)")
    args = ap.parse_args()
    sim = PlayoffSimulator(args.season, args.asof, seed=args.seed, injuries=args.injuries)
    for t, rot in sim.rotations.items():
        names = [data.load("players").set_index("player_id").name.get(p, p) for p in rot]
        print(f"{display(t):4s} rotation: {', '.join(names)}")
    odds = sim.run(args.sims)
    print(f"\nPostseason odds as of {sim.as_of.date()} ({args.sims:,} simulations)")
    print(odds.to_string(index=False, float_format=lambda x: f"{x:6.1%}"))
    odds.to_csv(C.DATA_DIR / f"bracket_odds_{args.season}_{sim.as_of.date()}.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
