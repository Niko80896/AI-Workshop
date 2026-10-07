"""Leakage-free feature engineering.

All features come from ``LeagueState``, which is fed games strictly in date
order. For any game on day D, features are read from the state *before* any
game on day D is added (so doubleheaders never see each other either). The
same ``LeagueState.matchup()`` call is used for historical training rows and
for simulating future playoff games, so training and prediction can't drift.

Stats are accumulated with exponential time decay, which handles both
"weight recent performance more" and the regular-season-to-next-season
carryover in one mechanism, then regressed toward league average.

Usage:
    python features.py      # writes data/features.csv
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C
import data

log = logging.getLogger("features")

# Half-lives in days.
HL_TEAM = 120        # season-level team strength (Pythagorean)
HL_FORM = 15         # "last 30 days" form
HL_PITCHER = 200
HL_BATTER = 240
HL_ROSTER = 20       # who is playing regularly for a team right now
HL_RELIEF = 30
ROSTER_WINDOW = 21   # a player is on the active roster if he played in the last N days

ELO_K = 4.0
ELO_HFA = 24.0
ELO_REGRESS = 1 / 3

# Columns used by the models. Every one is home-minus-away unless noted.
FEATURES = [
    "elo_diff", "pyth_diff", "form_rd_diff",
    "lineup_woba_diff", "lineup_iso_diff", "lineup_k_diff", "platoon_adv_diff",
    "sp_fip_diff", "sp_xfip_diff", "sp_kbb_diff", "sp_era_diff", "sp_depth_diff", "sp_short_rest_diff",
    "pen_top_fip_diff", "pen_all_fip_diff", "pen_fatigue_diff", "pen_tired_top_diff",
    "def_eff_diff", "baserun_diff", "rest_diff", "is_postseason",
]

FEATURE_LABELS = {
    "elo_diff": "Elo rating", "pyth_diff": "Pythagorean win%", "form_rd_diff": "Recent run differential",
    "lineup_woba_diff": "Lineup wOBA vs opposing SP hand", "lineup_iso_diff": "Lineup power (ISO)",
    "lineup_k_diff": "Lineup strikeout rate", "platoon_adv_diff": "Platoon advantage vs SP",
    "sp_fip_diff": "Starter FIP", "sp_xfip_diff": "Starter xFIP", "sp_kbb_diff": "Starter K-BB%",
    "sp_era_diff": "Starter ERA (regressed)", "sp_depth_diff": "Starter innings per start",
    "sp_short_rest_diff": "Starter on short rest", "pen_top_fip_diff": "Top-5 bullpen FIP",
    "pen_all_fip_diff": "Full bullpen FIP", "pen_fatigue_diff": "Bullpen pitches, last 3 days",
    "pen_tired_top_diff": "Top relievers likely unavailable", "def_eff_diff": "Defensive efficiency",
    "baserun_diff": "Baserunning (SB/CS runs)", "rest_diff": "Team rest days",
    "is_postseason": "Postseason game",
}

WOBA_W = C.WOBA_WEIGHTS


class Decayed:
    """Exponentially time-decayed sums of a fixed-length stat vector, by key."""

    def __init__(self, half_life: float, width: int):
        self.lam = math.log(2) / half_life
        self.width = width
        self.vals: dict = {}

    def add(self, key, day: int, vec) -> None:
        cur = self.vals.get(key)
        if cur is None:
            self.vals[key] = [np.asarray(vec, dtype=float).copy(), day]
        else:
            cur[0] = cur[0] * math.exp(-self.lam * (day - cur[1])) + vec
            cur[1] = day

    def get(self, key, day: int) -> np.ndarray:
        cur = self.vals.get(key)
        if cur is None:
            return np.zeros(self.width)
        return cur[0] * math.exp(-self.lam * (day - cur[1]))


def shrink(num, den, prior_rate, k):
    """Regress an observed rate toward ``prior_rate`` with ``k`` pseudo-observations."""
    return (num + k * prior_rate) / (den + k)


# Pitching vector layout.
PV = ["BF", "outs", "K", "BB", "HBP", "HR", "F", "P", "er", "IBB"]
PI = {k: i for i, k in enumerate(PV)}
# Batting vector layout (per batter per pitcher hand).
BV = ["woba_num", "woba_den", "PA", "K", "AB", "iso_num"]
BI = {k: i for i, k in enumerate(BV)}


def _bat_vec(r) -> np.ndarray:
    ubb = r.get("BB", 0)
    woba = (WOBA_W["BB"] * ubb + WOBA_W["HBP"] * r.get("HBP", 0) + WOBA_W["1B"] * r.get("1B", 0)
            + WOBA_W["2B"] * r.get("2B", 0) + WOBA_W["3B"] * r.get("3B", 0) + WOBA_W["HR"] * r.get("HR", 0))
    den = r.get("AB", 0) + ubb + r.get("SF", 0) + r.get("HBP", 0)
    iso = r.get("2B", 0) + 2 * r.get("3B", 0) + 3 * r.get("HR", 0)
    return np.array([woba, den, r.get("PA", 0), r.get("K", 0), r.get("AB", 0), iso], dtype=float)


def _pit_vec(r) -> np.ndarray:
    return np.array([r.get(k, 0) for k in PV], dtype=float)


@dataclass
class Unavailable:
    """Players to treat as unavailable (injuries / manual roster moves)."""
    players: frozenset = frozenset()


class LeagueState:
    """Everything known about the league as of a given day."""

    def __init__(self):
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)
        self.elo_season: dict[str, int] = {}
        self.team_runs = Decayed(HL_TEAM, 3)          # RS, RA, G
        self.team_form = Decayed(HL_FORM, 2)          # RD, G
        self.team_misc = Decayed(HL_TEAM, 5)          # bip, bip_reached, SB, CS, G
        self.league_misc = Decayed(HL_TEAM, 5)
        self.league_runs = Decayed(HL_TEAM, 3)
        self.bat = Decayed(HL_BATTER, len(BV))        # key (player, hand)
        self.league_bat = Decayed(HL_BATTER, len(BV))  # key "same"/"opp" platoon
        self.roster_pa = Decayed(HL_ROSTER, 1)        # key (team, player)
        self.last_played: dict = {}                   # (team, player) -> day
        self.team_players: dict = defaultdict(set)
        self.bats: dict = {}
        self.throws: dict = {}
        self.pit = Decayed(HL_PITCHER, len(PV))       # key player
        self.league_pit = Decayed(HL_PITCHER, len(PV))
        self.starts = Decayed(HL_PITCHER, 3)          # starts, outs, pitches
        self.relief = Decayed(HL_RELIEF, 1)           # key (team, player): relief appearances
        self.team_relief = Decayed(HL_TEAM, len(PV))  # team bullpen aggregate
        self.last_relief: dict = {}                   # (team, player) -> day
        self.team_relievers: dict = defaultdict(set)
        self.team_starters: dict = defaultdict(set)
        self.last_start: dict = {}                    # (team, player) -> day
        self.appearances: dict = defaultdict(lambda: deque(maxlen=8))  # player -> (day, pitches)
        self.team_last_game: dict = {}
        self.day = 0

    # ------------------------------------------------------------------ #
    # League baselines
    # ------------------------------------------------------------------ #
    def lg_pitch_rates(self, day):
        v = self.league_pit.get("all", day)
        bf = max(v[PI["BF"]], 1)
        if bf < 1000:   # first days of the data: use typical MLB rates
            return {"K": .225, "BB": .08, "HBP": .011, "HR": .031, "F": .27, "P": .07,
                    "outs": .69, "er": .115, "IBB": .005}
        r = {k: v[PI[k]] / bf for k in PV if k != "BF"}
        return r

    def lg_fip_const(self, day):
        r = self.lg_pitch_rates(day)
        lg_era = 27 * r["er"] / r["outs"]
        raw = (13 * r["HR"] + 3 * (r["BB"] + r["HBP"]) - 2 * r["K"]) / (r["outs"] / 3)
        return lg_era - raw

    def lg_woba(self, day, kind=None):
        if kind is None:
            v = self.league_bat.get("same", day) + self.league_bat.get("opp", day)
        else:
            v = self.league_bat.get(kind, day)
        return v[BI["woba_num"]] / v[BI["woba_den"]] if v[BI["woba_den"]] > 500 else 0.315

    def lg_rate(self, day, stat, denom):
        v = self.league_bat.get("same", day) + self.league_bat.get("opp", day)
        return v[BI[stat]] / v[BI[denom]] if v[BI[denom]] > 500 else {"K": .225, "iso_num": .16}[stat]

    # ------------------------------------------------------------------ #
    # Pitchers
    # ------------------------------------------------------------------ #
    def pitcher_line(self, pid, day) -> dict:
        """Regressed rate stats (FIP, xFIP, K-BB%, ERA, depth) for one pitcher."""
        lg = self.lg_pitch_rates(day)
        v = self.pit.get(pid, day)
        bf = v[PI["BF"]]
        K = shrink(v[PI["K"]], bf, lg["K"], 70)
        BB = shrink(v[PI["BB"]], bf, lg["BB"], 170)
        HBP = shrink(v[PI["HBP"]], bf, lg["HBP"], 300)
        HR = shrink(v[PI["HR"]], bf, lg["HR"], 500)
        FB = shrink(v[PI["F"]] + v[PI["P"]], bf, lg["F"] + lg["P"], 150)
        outs = shrink(v[PI["outs"]], bf, lg["outs"], 300)
        er = shrink(v[PI["er"]], bf, lg["er"], 600)
        c = self.lg_fip_const(day)
        ip_per_bf = outs / 3
        hr_per_fb = lg["HR"] / (lg["F"] + lg["P"])
        fip = (13 * HR + 3 * (BB + HBP) - 2 * K) / ip_per_bf + c
        xfip = (13 * FB * hr_per_fb + 3 * (BB + HBP) - 2 * K) / ip_per_bf + c
        st = self.starts.get(pid, day)
        depth = shrink(st[1], st[0], 15.5, 4) / 3  # innings per start, prior ~5.1
        return {"fip": fip, "xfip": xfip, "kbb": K - BB, "k": K, "era": 27 * er / outs,
                "depth": depth, "bf": bf, "throws": self.throws.get(pid, "R")}

    def days_rest(self, pid, day) -> int:
        app = self.appearances.get(pid)
        if not app:
            return 10
        return min(day - app[-1][0] - 1, 10)

    def rotation(self, team, day, n=4, out: frozenset = frozenset()) -> list[str]:
        """Projected playoff rotation: recent starters ranked by regressed FIP."""
        cands = [p for p in self.team_starters[team]
                 if self.last_start.get((team, p), -999) >= day - 35 and p not in out]
        cands.sort(key=lambda p: self.pitcher_line(p, day)["fip"])
        return cands[:n]

    def bullpen(self, team, day, top_n=5, out: frozenset = frozenset()) -> dict:
        """Top-N relievers by regressed FIP among those used recently, plus fatigue."""
        cands = []
        for p in self.team_relievers[team]:
            if p in out or self.last_relief.get((team, p), -999) < day - 30:
                continue
            apps = self.relief.get((team, p), day)[0]
            if apps < 2.5:
                continue
            cands.append((self.pitcher_line(p, day)["fip"], p))
        cands.sort()
        top = [p for _, p in cands[:top_n]]
        top_fip = np.mean([f for f, _ in cands[:top_n]]) if cands else 4.3
        if len(cands) < top_n:   # pad thin bullpens with replacement level
            top_fip = (top_fip * len(cands) + 4.8 * (top_n - len(cands))) / top_n
        v = self.team_relief.get(team, day)
        lg = self.lg_pitch_rates(day)
        bf = v[PI["BF"]]
        K, BB = shrink(v[PI["K"]], bf, lg["K"], 200), shrink(v[PI["BB"]], bf, lg["BB"], 300)
        HBP, HR = shrink(v[PI["HBP"]], bf, lg["HBP"], 400), shrink(v[PI["HR"]], bf, lg["HR"], 800)
        outs = shrink(v[PI["outs"]], bf, lg["outs"], 400)
        all_fip = (13 * HR + 3 * (BB + HBP) - 2 * K) / (outs / 3) + self.lg_fip_const(day)
        fatigue, tired = 0.0, 0
        for p in self.team_relievers[team]:
            for d, pitches, t, gs in self.appearances.get(p, ()):
                if day - 3 <= d < day and t == team and not gs:
                    fatigue += pitches
        for p in top:
            recent = {}
            for d, pc, _t, _gs in self.appearances.get(p, ()):
                recent[d] = recent.get(d, 0) + pc
            if (day - 1 in recent and day - 2 in recent) or recent.get(day - 1, 0) >= 30:
                tired += 1
        return {"top_fip": top_fip, "all_fip": all_fip, "fatigue": fatigue, "tired_top": tired,
                "top": top}

    # ------------------------------------------------------------------ #
    # Lineups
    # ------------------------------------------------------------------ #
    def roster(self, team, day, out: frozenset = frozenset()) -> dict[str, float]:
        """Active position players and their playing-time weights."""
        w = {}
        for p in self.team_players[team]:
            if self.last_played.get((team, p), -999) >= day - ROSTER_WINDOW:
                wt = self.roster_pa.get((team, p), day)[0]
                if wt > 0.5:
                    w[p] = 0.0 if p in out else wt
        return w

    def batter_rate(self, pid, hand, day, stat="woba") -> float:
        """Regressed wOBA / K% / ISO for a batter vs a pitcher hand (with platoon prior)."""
        both = self.bat.get((pid, "L"), day) + self.bat.get((pid, "R"), day)
        vs = self.bat.get((pid, hand), day)
        bats = self.bats.get(pid, "R")
        if bats == "B":
            same = False
        else:
            same = bats == hand
        if stat == "woba":
            lg_all = self.lg_woba(day)
            lg_split = self.lg_woba(day, "same" if same else "opp")
            overall = shrink(both[BI["woba_num"]], both[BI["woba_den"]], lg_all, 220)
            prior = overall + (lg_split - lg_all)
            return shrink(vs[BI["woba_num"]], vs[BI["woba_den"]], prior, 400)
        if stat == "k":
            return shrink(both[BI["K"]], both[BI["PA"]], self.lg_rate(day, "K", "PA"), 60)
        return shrink(both[BI["iso_num"]], both[BI["AB"]], self.lg_rate(day, "iso_num", "AB"), 160)

    def lineup(self, team, opp_hand, day, out: frozenset = frozenset()) -> dict:
        """Playing-time weighted lineup quality against a pitcher hand.

        Unavailable players' playing time goes to replacement level.
        """
        w = self.roster(team, day, out)
        lost = sum(self.roster_pa.get((team, p), day)[0] for p in out
                   if self.last_played.get((team, p), -999) >= day - ROSTER_WINDOW)
        tot = sum(w.values()) + lost
        lg = self.lg_woba(day)
        if tot <= 0:
            return {"woba": lg, "iso": .16, "k": .225, "platoon_adv": .5}
        woba = sum(wt * self.batter_rate(p, opp_hand, day) for p, wt in w.items())
        iso = sum(wt * self.batter_rate(p, opp_hand, day, "iso") for p, wt in w.items())
        k = sum(wt * self.batter_rate(p, opp_hand, day, "k") for p, wt in w.items())
        adv = sum(wt for p, wt in w.items()
                  if self.bats.get(p, "R") == "B" or self.bats.get(p, "R") != opp_hand)
        repl_woba = lg - 0.03
        return {"woba": (woba + lost * repl_woba) / tot, "iso": (iso + lost * 0.13) / tot,
                "k": (k + lost * 0.25) / tot, "platoon_adv": adv / tot}

    # ------------------------------------------------------------------ #
    # Team-level
    # ------------------------------------------------------------------ #
    def team_line(self, team, day) -> dict:
        lgv = self.league_runs.get("all", day)
        lg_rpg = lgv[0] / lgv[2] if lgv[2] > 50 else 4.5
        v = self.team_runs.get(team, day)
        rs = shrink(v[0], v[2], lg_rpg, 25)
        ra = shrink(v[1], v[2], lg_rpg, 25)
        pyth = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)
        f = self.team_form.get(team, day)
        form = shrink(f[0], f[1], 0.0, 8)
        m, lm = self.team_misc.get(team, day), self.league_misc.get("all", day)
        lg_der = 1 - lm[1] / lm[0] if lm[0] > 100 else 0.69
        der = shrink(m[0] - m[1], m[0], lg_der, 400)
        baserun = shrink(0.2 * m[2] - 0.41 * m[3], m[4], 0.0, 20)
        return {"elo": self.elo[team], "pyth": pyth, "form": form, "der": der, "baserun": baserun,
                "rs_pg": rs, "ra_pg": ra}

    # ------------------------------------------------------------------ #
    # Matchup
    # ------------------------------------------------------------------ #
    def matchup(self, home, away, home_sp, away_sp, day, is_post=0, rest=None,
                out: frozenset = frozenset(), sp_rest=None) -> dict:
        """Feature row for a (possibly hypothetical) game on ``day``.

        Args:
            rest: optional (home_rest_days, away_rest_days) override (simulation).
            out: player ids to treat as unavailable (injuries).
            sp_rest: optional (home_sp_rest, away_sp_rest) override.
        """
        th, ta = self.team_line(home, day), self.team_line(away, day)
        sh, sa = self.pitcher_line(home_sp, day), self.pitcher_line(away_sp, day)
        lh = self.lineup(home, sa["throws"], day, out)
        la = self.lineup(away, sh["throws"], day, out)
        bh, ba = self.bullpen(home, day, out=out), self.bullpen(away, day, out=out)
        if rest is None:
            rest = (min(day - self.team_last_game.get(home, day - 5) - 1, 4),
                    min(day - self.team_last_game.get(away, day - 5) - 1, 4))
        if sp_rest is None:
            sp_rest = (self.days_rest(home_sp, day), self.days_rest(away_sp, day))
        row = {
            "elo_diff": th["elo"] - ta["elo"],
            "pyth_diff": th["pyth"] - ta["pyth"],
            "form_rd_diff": th["form"] - ta["form"],
            "lineup_woba_diff": lh["woba"] - la["woba"],
            "lineup_iso_diff": lh["iso"] - la["iso"],
            "lineup_k_diff": lh["k"] - la["k"],
            "platoon_adv_diff": lh["platoon_adv"] - la["platoon_adv"],
            "sp_fip_diff": sh["fip"] - sa["fip"],
            "sp_xfip_diff": sh["xfip"] - sa["xfip"],
            "sp_kbb_diff": sh["kbb"] - sa["kbb"],
            "sp_era_diff": sh["era"] - sa["era"],
            "sp_depth_diff": sh["depth"] - sa["depth"],
            "sp_short_rest_diff": int(sp_rest[0] < 4) - int(sp_rest[1] < 4),
            "pen_top_fip_diff": bh["top_fip"] - ba["top_fip"],
            "pen_all_fip_diff": bh["all_fip"] - ba["all_fip"],
            "pen_fatigue_diff": bh["fatigue"] - ba["fatigue"],
            "pen_tired_top_diff": bh["tired_top"] - ba["tired_top"],
            "def_eff_diff": th["der"] - ta["der"],
            "baserun_diff": th["baserun"] - ta["baserun"],
            "rest_diff": rest[0] - rest[1],
            "is_postseason": is_post,
            # Not model features: used by baselines and reports.
            "home_elo": th["elo"], "away_elo": ta["elo"], "home_pyth": th["pyth"], "away_pyth": ta["pyth"],
        }
        detail = {"home_team": th, "away_team": ta, "home_sp": sh, "away_sp": sa,
                  "home_lineup": lh, "away_lineup": la, "home_pen": bh, "away_pen": ba}
        return row, detail

    # ------------------------------------------------------------------ #
    # Updates (only ever called with a game's own, completed data)
    # ------------------------------------------------------------------ #
    def _elo_update(self, g, day):
        season = g["season"]
        for t in (g["hometeam"], g["visteam"]):
            if self.elo_season.get(t, season) != season:
                self.elo[t] = 1500 + (1 - ELO_REGRESS) * (self.elo[t] - 1500)
            self.elo_season[t] = season
        h, a = g["hometeam"], g["visteam"]
        exp = 1 / (1 + 10 ** (-(self.elo[h] + ELO_HFA - self.elo[a]) / 400))
        mov = abs(g["home_score"] - g["vis_score"])
        k = ELO_K * math.log(mov + 1) * 1.6
        delta = k * (g["home_win"] - exp)
        self.elo[h] += delta
        self.elo[a] -= delta

    def update(self, g, pitchers, batters, teams) -> None:
        """Add one completed game (game row + its pitcher/batter/team stat lines)."""
        day = g["day"]
        self._elo_update(g, day)
        for team, rs, ra in ((g["hometeam"], g["home_score"], g["vis_score"]),
                             (g["visteam"], g["vis_score"], g["home_score"])):
            self.team_runs.add(team, day, [rs, ra, 1])
            self.team_form.add(team, day, [rs - ra, 1])
            self.league_runs.add("all", day, [rs, ra, 1])
            self.team_last_game[team] = day
        for t in teams:
            vec = [t.get("bip", 0), t.get("bip_reached", 0), t.get("sb", 0), t.get("cs", 0), 1]
            self.team_misc.add(t["team"], day, vec)
            self.league_misc.add("all", day, vec)
        for p in pitchers:
            pid, team = p["player_id"], p["team"]
            v = _pit_vec(p)
            self.pit.add(pid, day, v)
            self.league_pit.add("all", day, v)
            self.appearances[pid].append((day, p.get("pitches", 0), team, int(p.get("gs", 0))))
            if p.get("gs", 0) == 1:
                self.starts.add(pid, day, [1, p.get("outs", 0), p.get("pitches", 0)])
                self.team_starters[team].add(pid)
                self.last_start[(team, pid)] = day
            else:
                self.relief.add((team, pid), day, [1])
                self.team_relief.add(team, day, v)
                self.team_relievers[team].add(pid)
                self.last_relief[(team, pid)] = day
        for b in batters:
            pid, team, hand = b["player_id"], b["team"], b["vs_hand"]
            v = _bat_vec(b)
            self.bat.add((pid, hand), day, v)
            # 'bats' in the stat line is the side he actually hit from.
            self.league_bat.add("same" if b.get("bats") == hand else "opp", day, v)
            self.roster_pa.add((team, pid), day, [b.get("PA", 0)])
            self.last_played[(team, pid)] = day
            self.team_players[team].add(pid)
        self.day = day


def day_number(d) -> int:
    return pd.Timestamp(d).toordinal()


def load_inputs():
    """Load the clean CSVs and bucket stat lines by game for fast iteration."""
    games = data.load("games")
    pit = data.load("pitcher_games")
    bat = data.load("batter_games")
    team = data.load("team_games")
    players = data.load("players")
    return games, pit, bat, team, players


def _bucket(df):
    out = defaultdict(list)
    for r in df.to_dict("records"):
        out[r["game_id"]].append(r)
    return out


def run_state(games, pit, bat, team, players, until=None, on_day=None) -> LeagueState:
    """Feed games into a fresh LeagueState in date order.

    Args:
        until: stop before this date (exclusive) — the state then reflects
            exactly what was known on the morning of ``until``.
        on_day: optional callback(state, day_games_df) called *before* each
            day's games are added; used to snapshot pre-game features.
    """
    st = LeagueState()
    hands = players.set_index("player_id")[["bats", "throws"]]
    st.bats = hands["bats"].to_dict()
    st.throws = hands["throws"].to_dict()
    g = games.sort_values(["date", "game_id"]).copy()
    if until is not None:
        g = g[g["date"] < pd.Timestamp(until)]
    g["day"] = g["date"].map(lambda d: d.toordinal())
    pb, bb, tb = _bucket(pit), _bucket(bat), _bucket(team)
    for day, dg in g.groupby("day", sort=True):
        if on_day is not None:
            on_day(st, dg)
        for r in dg.to_dict("records"):
            st.update(r, pb.get(r["game_id"], []), bb.get(r["game_id"], []), tb.get(r["game_id"], []))
    return st


def build_features(games, pit, bat, team, players) -> pd.DataFrame:
    """One leakage-free feature row per game (home-team perspective)."""
    rows = []

    def snap(st, dg):
        for r in dg.to_dict("records"):
            if not isinstance(r["home_sp"], str) or not isinstance(r["vis_sp"], str):
                continue
            f, _ = st.matchup(r["hometeam"], r["visteam"], r["home_sp"], r["vis_sp"], r["day"],
                              is_post=int(r["is_postseason"]))
            f.update({k: r[k] for k in ("game_id", "date", "season", "round", "hometeam", "visteam",
                                         "home_sp", "vis_sp", "home_score", "vis_score", "home_win")})
            rows.append(f)

    run_state(games, pit, bat, team, players, on_day=snap)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import time
    t0 = time.time()
    feats = build_features(*load_inputs())
    feats.to_csv(C.DATA_DIR / "features.csv", index=False)
    log.info("wrote %d feature rows in %.0fs", len(feats), time.time() - t0)
