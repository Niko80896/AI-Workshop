"""Data collection: download, cache and clean MLB data into CSVs under data/.

Primary source is Retrosheet (via the Chadwick Bureau GitHub mirror), which
provides every regular-season and postseason game since 1871 with full
play-by-play. From it we build:

    data/games.csv          one row per game: date, teams, score, starters, round
    data/pitcher_games.csv  one row per pitcher per game (BF, outs, K, BB, HR, GB/FB, pitches, ER)
    data/batter_games.csv   one row per batter per game per opposing-pitcher hand
    data/team_games.csv     one row per team per game (batting totals, SB/CS, defense)
    data/players.csv        id, name, bats, throws
    data/standings.csv      final regular-season standings with derived playoff seeds

Optional sources (pybaseball: FanGraphs, Baseball Savant, Baseball Reference;
statsapi: MLB Stats API) are fetched by ``fetch_external()``. They add
Statcast/defense/baserunning columns when those hosts are reachable and are
skipped cleanly otherwise. Every download is cached under data/cache/.

Usage:
    python data.py               # build everything (re-runs use the cache)
    python data.py --live        # also import the current season from the MLB Stats API
    python data.py --external    # also try pybaseball / statsapi sources
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import logging
import urllib.error
import urllib.request

import pandas as pd

import config as C
from retrosheet import parse_event_file

log = logging.getLogger("data")

# --------------------------------------------------------------------------- #
# Download + cache
# --------------------------------------------------------------------------- #

def fetch(rel_path: str, base: str = C.RETRO_BASE, timeout: int = 60) -> str | None:
    """Return the text of ``base/rel_path``, caching it under data/cache.

    A missing file (HTTP 404) is cached as a ``.missing`` marker so it is not
    requested again. Network errors raise.
    """
    dest = C.CACHE_DIR / rel_path
    missing = dest.with_name(dest.name + ".missing")
    if dest.exists():
        return dest.read_text(encoding="latin-1")
    if missing.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(f"{base}/{rel_path}", timeout=timeout) as r:
            text = r.read().decode("latin-1")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            missing.touch()
            return None
        raise
    dest.write_text(text, encoding="latin-1")
    return text


def season_team_codes(season: int) -> list[str]:
    """Retrosheet team codes for a season (from the TEAMyyyy file)."""
    text = fetch(f"seasons/{season}/TEAM{season}")
    return [ln.split(",")[0] for ln in text.splitlines() if ln and ln.split(",")[1] in ("A", "N")]


def team_leagues(season: int) -> dict[str, str]:
    """Map team code -> 'AL'/'NL' for a season."""
    text = fetch(f"seasons/{season}/TEAM{season}")
    out = {}
    for ln in text.splitlines():
        parts = ln.split(",")
        if len(parts) >= 2 and parts[1] in ("A", "N"):
            out[parts[0]] = "AL" if parts[1] == "A" else "NL"
    return out


def event_file_paths(season: int) -> list[str]:
    """All regular-season and postseason event-file paths for a season."""
    leagues = team_leagues(season)
    paths = [f"seasons/{season}/{season}{t}.EV{'A' if lg == 'AL' else 'N'}" for t, lg in leagues.items()]
    paths += [f"seasons/{season}/{season}{s}.EVE" for s in C.POSTSEASON_EVENT_FILES]
    return paths


def gamelog_path(season: int) -> str:
    # Retrosheet's 2024 log is stored lowercase in the mirror.
    return f"seasons/{season}/{'gl' if season == 2024 else 'GL'}{season}.{'txt' if season == 2024 else 'TXT'}"


def download_all(seasons=C.SEASONS, workers: int = 8) -> None:
    """Fetch every file the pipeline needs (cached; safe to re-run)."""
    paths = ["reference/biofile.csv"] + [f"gamelog/{f}" for f in C.POSTSEASON_GAMELOGS.values()]
    for s in seasons:
        paths.append(gamelog_path(s))
        paths += event_file_paths(s)
    todo = [p for p in paths if not (C.CACHE_DIR / p).exists()
            and not (C.CACHE_DIR / (p + ".missing")).exists()]
    log.info("%d files needed, %d not cached yet", len(paths), len(todo))
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, todo))


# --------------------------------------------------------------------------- #
# Game logs
# --------------------------------------------------------------------------- #

# Zero-based column positions in the Retrosheet game-log format.
GL_COLS = {0: "date", 1: "game_num", 3: "visteam", 4: "vis_league", 6: "hometeam",
           7: "home_league", 9: "vis_score", 10: "home_score", 11: "outs", 16: "park",
           19: "vis_line", 20: "home_line",
           101: "vis_sp", 102: "vis_sp_name", 103: "home_sp", 104: "home_sp_name"}


def read_gamelog(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)
    df = df[list(GL_COLS)].rename(columns=GL_COLS)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in ("vis_score", "home_score", "outs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["game_id"] = df["hometeam"] + df["date"].dt.strftime("%Y%m%d") + df["game_num"]
    return df


def parse_line(line: str) -> list:
    """Retrosheet line score ('00(10)0x') -> runs per inning; 'x' (not batted) -> None."""
    out, i = [], 0
    while i < len(line):
        ch = line[i]
        if ch == "(":
            j = line.index(")", i)
            out.append(int(line[i + 1:j]))
            i = j + 1
            continue
        out.append(None if ch.lower() == "x" else int(ch))
        i += 1
    return out


def add_inning_splits(games: pd.DataFrame) -> pd.DataFrame:
    """Runs in the 1st inning, first 3 and first 5 innings per side (NaN if not played)."""
    for side in ("vis", "home"):
        lines = games[f"{side}_line"].fillna("").astype(str).map(parse_line)
        for name, n in (("r1", 1), ("f3", 3), ("f5", 5)):
            games[f"{side}_{name}"] = lines.map(
                lambda ln, n=n: sum(ln[:n]) if len(ln) >= n and None not in ln[:n] else float("nan"))
    return games


def postseason_round(row) -> str:
    """Postseason games come from per-round game logs; label them."""
    return row["round"]


def build_games(seasons=C.SEASONS) -> pd.DataFrame:
    """All regular-season and postseason games with results and starters."""
    frames = []
    for s in seasons:
        g = read_gamelog(fetch(gamelog_path(s)))
        g["round"] = "REG"
        frames.append(g)
    for rnd, fname in C.POSTSEASON_GAMELOGS.items():
        g = read_gamelog(fetch(f"gamelog/{fname}"))
        g = g[g["date"].dt.year.isin(seasons)]
        g["round"] = rnd
        frames.append(g)
    games = pd.concat(frames, ignore_index=True)
    games["season"] = games["date"].dt.year
    for col in ("visteam", "hometeam"):
        games[col] = games[col].replace(C.FRANCHISE_ALIASES)
    games["home_win"] = (games["home_score"] > games["vis_score"]).astype(int)
    games = games[games["home_score"] != games["vis_score"]]  # drop suspended/tied
    games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
    games["is_postseason"] = (games["round"] != "REG").astype(int)
    return games


# --------------------------------------------------------------------------- #
# Play-by-play
# --------------------------------------------------------------------------- #

def load_hands() -> pd.DataFrame:
    """Player id, name, bats, throws from the Retrosheet biofile."""
    bio = pd.read_csv(io.StringIO(fetch("reference/biofile.csv")), dtype=str, keep_default_na=False)
    bio["name"] = (bio["NICKNAME"].where(bio["NICKNAME"] != "", bio["FIRST"]) + " " + bio["LAST"]).str.strip()
    return bio.rename(columns={"PLAYERID": "player_id", "BATS": "bats", "THROWS": "throws"})[
        ["player_id", "name", "bats", "throws"]]


def parse_seasons(seasons=C.SEASONS) -> dict[str, pd.DataFrame]:
    """Parse every event file into pitcher/batter/team/game-level frames."""
    players = load_hands()
    hands = dict(zip(players["player_id"], zip(players["bats"], players["throws"])))
    out = {"pitcher": [], "batter": [], "team": [], "game": []}
    for s in seasons:
        n = 0
        for path in event_file_paths(s):
            text = fetch(path)
            if text is None:
                continue
            p, b, t, g = parse_event_file(text, hands)
            out["pitcher"] += p
            out["batter"] += b
            out["team"] += t
            out["game"] += g
            n += 1
        log.info("season %d: parsed %d event files", s, n)
    frames = {k: pd.DataFrame(v).fillna(0) for k, v in out.items()}
    for k in ("pitcher", "batter", "team"):
        frames[k]["date"] = pd.to_datetime(frames[k]["date"])
    for col in ("team",):
        for k in ("pitcher", "batter", "team"):
            frames[k][col] = frames[k][col].replace(C.FRANCHISE_ALIASES)
    frames["players"] = players
    return frames


# --------------------------------------------------------------------------- #
# Standings and playoff seeds
# --------------------------------------------------------------------------- #

def build_standings(games: pd.DataFrame) -> pd.DataFrame:
    """Final regular-season W/L, run differential and league for each team-season."""
    reg = games[games["round"] == "REG"]
    rows = []
    for side, opp in (("home", "vis"), ("vis", "home")):
        team_col = "hometeam" if side == "home" else "visteam"
        lg_col = "home_league" if side == "home" else "vis_league"
        d = reg[[ "season", team_col, lg_col, f"{side}_score", f"{opp}_score"]].copy()
        d.columns = ["season", "team", "league", "rs", "ra"]
        rows.append(d)
    d = pd.concat(rows)
    d["w"] = (d["rs"] > d["ra"]).astype(int)
    st = d.groupby(["season", "team", "league"], as_index=False).agg(
        g=("w", "size"), w=("w", "sum"), rs=("rs", "sum"), ra=("ra", "sum"))
    st["l"] = st["g"] - st["w"]
    st["pct"] = st["w"] / st["g"]
    return st.sort_values(["season", "league", "pct"], ascending=[True, True, False])


# --------------------------------------------------------------------------- #
# Optional external sources (pybaseball / statsapi)
# --------------------------------------------------------------------------- #

def _try(name: str, fn):
    """Run an external fetch; log and return None if the host is unreachable."""
    try:
        return fn()
    except Exception as e:  # network policy blocks, API changes, etc.
        log.warning("external source %s unavailable: %s", name, str(e).splitlines()[0][:120])
        return None


def fetch_external(seasons=C.SEASONS) -> dict[str, pd.DataFrame]:
    """Best-effort pull of FanGraphs / Statcast / Baseball Reference team metrics.

    Each table is saved to data/external/<name>.csv with a ``season`` column.
    Season-level stats are only used by features.py for games played *after*
    that season's regular season ended (no leakage).
    """
    import pybaseball as pb
    pb.cache.enable()
    results = {}
    jobs = {
        "fg_team_batting": lambda: pb.team_batting(min(seasons), max(seasons)),
        "fg_team_pitching": lambda: pb.team_pitching(min(seasons), max(seasons)),
        "fg_pitching_players": lambda: pb.pitching_stats(min(seasons), max(seasons), qual=10),
        "fg_team_fielding": lambda: pb.team_fielding(min(seasons), max(seasons)),
        "savant_oaa": lambda: pd.concat(
            [pb.statcast_outs_above_average(s, "all").assign(season=s) for s in seasons if s >= 2016]),
        "savant_framing": lambda: pd.concat(
            [pb.statcast_catcher_framing(s).assign(season=s) for s in seasons]),
    }
    for name, job in jobs.items():
        df = _try(name, job)
        if df is not None and len(df):
            if "Season" in df.columns and "season" not in df.columns:
                df = df.rename(columns={"Season": "season"})
            df.to_csv(C.EXTERNAL_DIR / f"{name}.csv", index=False)
            results[name] = df
    return results


def fetch_live_bracket(season: int) -> dict | None:
    """Pull the current postseason schedule and probable pitchers from the MLB Stats API.

    Returns a dict in the same shape as data/manual/bracket_<season>.json, or
    None if statsapi.mlb.com is unreachable.
    """
    def job():
        import statsapi
        sched = statsapi.schedule(start_date=f"{season}-09-28", end_date=f"{season}-11-10",
                                  sportId=1, include_series_status=True)
        post = [g for g in sched if g.get("game_type") in ("F", "D", "L", "W")]
        return {"season": season, "source": "statsapi", "games": post}
    data = _try("statsapi schedule", job)
    if data:
        (C.EXTERNAL_DIR / f"statsapi_postseason_{season}.json").write_text(json.dumps(data, indent=1))
    return data


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_all(seasons=C.SEASONS, external: bool = False, live: bool = False) -> None:
    """Download (cached) and write all clean CSVs to data/.

    With ``live=True`` the current season (C.LIVE_SEASON) is imported from the
    MLB Stats API and appended with the same columns.
    """
    download_all(seasons)
    games = build_games(seasons)
    frames = parse_seasons(seasons)
    pbp_games = frames["game"].copy()
    # Keep only games present in both sources.
    games = games[games["game_id"].isin(set(pbp_games["game_id"]))].reset_index(drop=True)
    if live:
        import statsapi_import as SA
        if not SA.reachable():
            raise SystemExit("--live needs statsapi.mlb.com, which this environment's network policy blocks.")
        cur = SA.import_season(C.LIVE_SEASON)
        games = pd.concat([games, cur["game"]], ignore_index=True).sort_values(["date", "game_id"])
        games = games.reset_index(drop=True)
        for k in ("pitcher", "batter", "team"):
            frames[k] = pd.concat([frames[k], cur[k]], ignore_index=True).fillna(0)
        new = cur["players"][~cur["players"]["player_id"].isin(set(frames["players"]["player_id"]))]
        frames["players"] = pd.concat([frames["players"], new], ignore_index=True)
        log.info("added %d %d games from the MLB Stats API", len(cur["game"]), C.LIVE_SEASON)
    games = add_inning_splits(games)
    games.to_csv(C.DATA_DIR / "games.csv", index=False)
    pitch = frames["pitcher"]
    pitch["ip"] = pitch["outs"] / 3
    pitch["role"] = pitch["gs"].map({1: "SP", 0: "RP"})
    pitch.to_csv(C.DATA_DIR / "pitcher_games.csv", index=False)
    frames["batter"].to_csv(C.DATA_DIR / "batter_games.csv", index=False)
    frames["team"].to_csv(C.DATA_DIR / "team_games.csv", index=False)
    frames["players"].to_csv(C.DATA_DIR / "players.csv", index=False)
    build_standings(games).to_csv(C.DATA_DIR / "standings.csv", index=False)
    qa = pbp_games["half_innings_3_outs"].sum() / pbp_games["half_innings"].sum()
    log.info("games=%d  pitcher lines=%d  batter lines=%d  half-innings with exactly 3 outs: %.2f%%",
             len(games), len(pitch), len(frames["batter"]), 100 * qa)
    if external:
        fetch_external(seasons)
        fetch_live_bracket(C.LAST_SEASON + 1)


def load(name: str) -> pd.DataFrame:
    """Read one of the clean CSVs written by build_all()."""
    df = pd.read_csv(C.DATA_DIR / f"{name}.csv", low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", action="store_true", help="also try pybaseball/statsapi sources")
    ap.add_argument("--live", action="store_true",
                    help=f"also import the {C.LIVE_SEASON} season from the MLB Stats API")
    ap.add_argument("--seasons", nargs="*", type=int, default=C.SEASONS)
    args = ap.parse_args()
    build_all(args.seasons, external=args.external, live=args.live)
