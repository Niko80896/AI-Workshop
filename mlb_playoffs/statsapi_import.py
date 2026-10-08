"""Import a season from the MLB Stats API into the same tables Retrosheet produces.

Retrosheet publishes a season only after it ends, so the current season
(2026) comes from statsapi.mlb.com instead. Each final game's live feed is
converted into rows with exactly the columns of games.csv, pitcher_games.csv,
batter_games.csv and team_games.csv, and players are mapped to Retrosheet
ids with the Chadwick Bureau register (new players get ``mlb<id>``), so
2026 stats join each player's 2015-2025 history.

Only the parsed rows are cached (data/cache/statsapi/<season>/<gamePk>.json.gz);
raw feeds are not kept. Games are fetched once, after they are final.

Usage:
    python statsapi_import.py --season 2026      # then: python data.py --live
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import logging
import urllib.request

import pandas as pd

import config as C
import data

log = logging.getLogger("statsapi")
API = "https://statsapi.mlb.com"

# MLBAM team id -> Retrosheet franchise code.
TEAM_IDS = {
    108: "ANA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHN", 113: "CIN", 114: "CLE",
    115: "COL", 116: "DET", 117: "HOU", 118: "KCA", 119: "LAN", 120: "WAS", 121: "NYN",
    133: "ATH", 134: "PIT", 135: "SDN", 136: "SEA", 137: "SFN", 138: "SLN", 139: "TBA",
    140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI", 144: "ATL", 145: "CHA", 146: "MIA",
    147: "NYA", 158: "MIL",
}
LEAGUES = {103: "AL", 104: "NL"}
ROUND = {"R": "REG", "F": "WC", "D": "DS", "L": "LCS", "W": "WS"}

# statsapi eventType -> Retrosheet-style batter outcome (None = not a plate appearance).
EVENTS = {
    "strikeout": "K", "strikeout_double_play": "K", "strikeout_triple_play": "K",
    "walk": "BB", "intent_walk": "IBB", "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "field_error": "ROE", "fielders_choice": "FC", "fielders_choice_out": "FC",
    "catcher_interf": "CI", "field_out": "OUT", "force_out": "OUT",
    "grounded_into_double_play": "OUT", "double_play": "OUT", "triple_play": "OUT",
    "sac_fly": "OUT", "sac_fly_double_play": "OUT", "sac_bunt": "OUT",
    "sac_bunt_double_play": "OUT",
}
TRAJ = {"ground_ball": "G", "bunt_grounder": "G", "fly_ball": "F", "line_drive": "L",
        "bunt_line_drive": "L", "popup": "P", "bunt_popup": "P"}


def get_json(path: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(API + path, headers={"Accept-Encoding": "gzip", "User-Agent": "mlb-playoffs"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return json.loads(raw)


def reachable() -> bool:
    try:
        get_json("/api/v1/sports", timeout=15)
        return True
    except Exception as e:
        log.warning("statsapi.mlb.com unreachable: %s", str(e)[:120])
        return False


# --------------------------------------------------------------------------- #
# Player id mapping
# --------------------------------------------------------------------------- #

_idmap: dict | None = None


def mlbam_to_retro() -> dict[int, str]:
    """MLBAM id -> Retrosheet id, from the Chadwick register (cached)."""
    global _idmap
    if _idmap is None:
        frames = []
        for h in "0123456789abcdef":
            text = data.fetch(f"data/people-{h}.csv",
                              base="https://raw.githubusercontent.com/chadwickbureau/register/master")
            df = pd.read_csv(io.StringIO(text), usecols=["key_mlbam", "key_retro"], dtype=str)
            frames.append(df.dropna())
        reg = pd.concat(frames)
        _idmap = {int(m): r for m, r in zip(reg.key_mlbam, reg.key_retro)}
    return _idmap


def pid(mlbam: int) -> str:
    return mlbam_to_retro().get(int(mlbam), f"mlb{int(mlbam)}")


# --------------------------------------------------------------------------- #
# Feed -> rows
# --------------------------------------------------------------------------- #

def parse_feed(feed: dict) -> dict:
    """Convert one final game's live feed into games/pitcher/batter/team/player rows."""
    gd, ld = feed["gameData"], feed["liveData"]
    date = gd["datetime"]["officialDate"]
    home_t, away_t = gd["teams"]["home"], gd["teams"]["away"]
    home, away = TEAM_IDS[home_t["id"]], TEAM_IDS[away_t["id"]]
    dh = gd["game"].get("doubleHeader", "N")
    game_num = str(gd["game"].get("gameNumber", 1)) if dh in ("Y", "S") else "0"
    game_id = f"{home}{date.replace('-', '')}{game_num}"
    box = ld["boxscore"]["teams"]
    side_team = {"home": home, "away": away}

    players = []
    for p in gd["players"].values():
        players.append({"player_id": pid(p["id"]), "name": p.get("fullName", ""),
                        "bats": p.get("batSide", {}).get("code", "R"),
                        "throws": p.get("pitchHand", {}).get("code", "R")})

    # Pitcher lines: totals from the boxscore, the rest from play-by-play.
    pit = {}
    starters = {}
    for side in ("home", "away"):
        order = box[side].get("pitchers", [])
        if order:
            starters[side] = pid(order[0])
        for mid in order:
            st = box[side]["players"][f"ID{mid}"]["stats"].get("pitching", {})
            pit[pid(mid)] = {"team": side_team[side], "gs": int(mid == order[0]), "g": 1,
                             "outs": st.get("outs", 0), "pitches": st.get("numberOfPitches", 0),
                             "er": st.get("earnedRuns", 0)}

    bat: dict = {}
    teams = {home: {}, away: {}}

    def inc(d, k, v=1):
        d[k] = d.get(k, 0) + v

    for play in ld["plays"]["allPlays"]:
        res = play.get("result", {})
        top = play["about"]["halfInning"] == "top"
        bat_team, fld_team = (away, home) if top else (home, away)
        for r in play.get("runners", []):
            et = r.get("details", {}).get("eventType", "")
            if et.startswith("stolen_base"):
                inc(teams[bat_team], "sb")
            elif et.startswith("caught_stealing") or et.startswith("pickoff_caught_stealing"):
                inc(teams[bat_team], "cs")
        o = EVENTS.get(res.get("eventType"))
        if o is None:
            continue
        m = play["matchup"]
        p_id, b_id = pid(m["pitcher"]["id"]), pid(m["batter"]["id"])
        hand, bats = m["pitchHand"]["code"], m["batSide"]["code"]
        et = res["eventType"]
        stat = {"PA": 1}
        if o in ("K", "BB", "IBB", "HBP", "1B", "2B", "3B", "HR", "ROE"):
            stat[o] = 1
        if o in ("1B", "2B", "3B", "HR"):
            stat["H"] = 1
        sf, sh = et.startswith("sac_fly"), et.startswith("sac_bunt")
        if sf:
            stat["SF"] = 1
        if sh:
            stat["SH"] = 1
        if o not in ("BB", "IBB", "HBP", "CI") and not sf and not sh:
            stat["AB"] = 1
        traj = None
        for ev in play.get("playEvents", []):
            if ev.get("hitData", {}).get("trajectory"):
                traj = TRAJ.get(ev["hitData"]["trajectory"])
        if traj and o not in ("K", "BB", "IBB", "HBP"):
            stat[traj] = 1
        if o in ("1B", "2B", "3B", "ROE", "FC", "OUT"):
            inc(teams[fld_team], "bip")
            if o in ("1B", "2B", "3B", "ROE"):
                inc(teams[fld_team], "bip_reached")
        ps = pit.setdefault(p_id, {"team": fld_team, "gs": 0, "g": 1, "outs": 0, "pitches": 0, "er": 0})
        inc(ps, "BF")
        inc(ps, "BF_vsL" if bats == "L" else "BF_vsR")
        for k, v in stat.items():
            if k != "PA":
                inc(ps, k, v)
        bs = bat.setdefault((b_id, hand), {"team": bat_team, "bats": bats})
        for k, v in stat.items():
            inc(bs, k, v)
        for k in ("PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K", "SF"):
            if k in stat:
                inc(teams[bat_team], k, stat[k])

    for side in ("home", "away"):
        f = box[side].get("teamStats", {}).get("fielding", {})
        teams[side_team[side]]["errors"] = f.get("errors", 0)

    ls = ld["linescore"]["teams"]
    hs, as_ = ls["home"].get("runs", 0), ls["away"].get("runs", 0)
    names = {pid(p["id"]): p.get("fullName", "") for p in gd["players"].values()}
    game = {
        "date": date, "game_num": game_num, "visteam": away, "vis_league": LEAGUES.get(away_t.get("league", {}).get("id"), ""),
        "hometeam": home, "home_league": LEAGUES.get(home_t.get("league", {}).get("id"), ""),
        "vis_score": as_, "home_score": hs, "outs": None, "park": gd.get("venue", {}).get("name", ""),
        "vis_sp": starters.get("away"), "vis_sp_name": names.get(starters.get("away"), ""),
        "home_sp": starters.get("home"), "home_sp_name": names.get(starters.get("home"), ""),
        "game_id": game_id, "round": ROUND.get(gd["game"]["type"], "REG"),
        "season": int(date[:4]), "home_win": int(hs > as_), "is_postseason": int(gd["game"]["type"] != "R"),
    }
    base = {"game_id": game_id, "date": date}
    return {
        "game": game,
        "pitcher": [{**base, "player_id": k, **v} for k, v in pit.items()],
        "batter": [{**base, "player_id": k[0], "vs_hand": k[1], **v} for k, v in bat.items()],
        "team": [{**base, "team": t, **v} for t, v in teams.items()],
        "players": players,
    }


# --------------------------------------------------------------------------- #
# Season import
# --------------------------------------------------------------------------- #

def final_games(season: int, end: str | None = None) -> list[dict]:
    """Final regular-season and postseason games for a season (schedule endpoint)."""
    end = end or f"{season}-11-30"
    sched = get_json(f"/api/v1/schedule?sportId=1&startDate={season}-03-01&endDate={end}"
                     f"&gameType=R,F,D,L,W")
    out = []
    for d in sched.get("dates", []):
        for g in d["games"]:
            if g["status"].get("abstractGameState") == "Final" and g["status"].get("codedGameState") in ("F", "O"):
                out.append(g)
    return out


def import_game(pk: int, season: int) -> dict | None:
    path = C.CACHE_DIR / "statsapi" / str(season) / f"{pk}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    feed = get_json(f"/api/v1.1/game/{pk}/feed/live")
    try:
        rows = parse_feed(feed)
    except KeyError as e:
        log.warning("game %s skipped: missing %s", pk, e)
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(rows).encode()))
    return rows


def import_season(season: int, workers: int = 8) -> dict[str, pd.DataFrame]:
    """All final games of ``season`` as DataFrames with the Retrosheet-derived schemas."""
    mlbam_to_retro()
    games = final_games(season)
    log.info("%d final games in %d", len(games), season)
    out = {"game": [], "pitcher": [], "batter": [], "team": [], "players": []}
    with cf.ThreadPoolExecutor(workers) as ex:
        for rows in ex.map(lambda g: import_game(g["gamePk"], season), games):
            if rows is None:
                continue
            out["game"].append(rows["game"])
            for k in ("pitcher", "batter", "team", "players"):
                out[k] += rows[k]
    frames = {k: pd.DataFrame(v) for k, v in out.items()}
    for k in ("game", "pitcher", "batter", "team"):
        frames[k]["date"] = pd.to_datetime(frames[k]["date"])
    frames["players"] = frames["players"].drop_duplicates("player_id")
    return frames


def fetch_seeds(season: int) -> dict:
    """Playoff seeds 1-6 per league from statsapi standings (division winners, then wild cards)."""
    st = get_json(f"/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason")
    seeds = {"AL": [], "NL": []}
    by_lg: dict = {"AL": [], "NL": []}
    for rec in st["records"]:
        lg = LEAGUES[rec["league"]["id"]]
        for t in rec["teamRecords"]:
            by_lg[lg].append(t)
    for lg, teams in by_lg.items():
        leaders = sorted((t for t in teams if t.get("divisionLeader")), key=lambda t: int(t.get("leagueRank", 99)))
        wcs = sorted((t for t in teams if t.get("wildCardRank") and not t.get("divisionLeader")),
                     key=lambda t: int(t["wildCardRank"]))[:3]
        seeds[lg] = [TEAM_IDS[t["team"]["id"]] for t in leaders + wcs]
    return seeds


def postseason_calendar(season: int) -> dict:
    """Scheduled postseason dates per round/league from the schedule (played or not)."""
    sched = get_json(f"/api/v1/schedule?sportId=1&startDate={season}-09-25&endDate={season}-11-15&gameType=F,D,L,W")
    cal: dict = {}
    for d in sched.get("dates", []):
        for g in d["games"]:
            rnd = ROUND[g["gameType"]]
            lg = "MLB" if rnd == "WS" else LEAGUES.get(
                g["teams"]["home"]["team"].get("league", {}).get("id"), None)
            if lg is None:
                lg = "AL" if TEAM_IDS.get(g["teams"]["home"]["team"]["id"]) in (
                    "ANA", "ATH", "BAL", "BOS", "CHA", "CLE", "DET", "HOU", "KCA", "MIN", "NYA",
                    "SEA", "TBA", "TEX", "TOR") else "NL"
            n = g.get("seriesGameNumber", 1)
            slot = cal.setdefault(rnd, {}).setdefault(lg, {})
            slot.setdefault(n, g["officialDate"])
    return {r: {lg: [v[k] for k in sorted(v)] for lg, v in lgs.items()} for r, lgs in cal.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=C.LAST_SEASON + 1)
    args = ap.parse_args()
    if not reachable():
        raise SystemExit("statsapi.mlb.com is not reachable from this environment (network policy).")
    f = import_season(args.season)
    log.info("imported %d games, %d pitcher lines, %d batter lines",
             len(f["game"]), len(f["pitcher"]), len(f["batter"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
