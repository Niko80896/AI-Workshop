"""Tests for converting an MLB Stats API live feed into pipeline rows."""
import statsapi_import as SA


def _play(half, batter, pitcher, bats, throws, event, traj=None, runners=()):
    ev = [{"isPitch": True}]
    if traj:
        ev.append({"isPitch": True, "hitData": {"trajectory": traj}})
    return {"about": {"halfInning": half, "inning": 1},
            "matchup": {"batter": {"id": batter}, "pitcher": {"id": pitcher},
                        "batSide": {"code": bats}, "pitchHand": {"code": throws}},
            "result": {"eventType": event, "type": "atBat"},
            "playEvents": ev, "runners": list(runners)}


def _player(i, name, bats, throws):
    return {"id": i, "fullName": name, "batSide": {"code": bats}, "pitchHand": {"code": throws}}


FEED = {
    "gameData": {
        "datetime": {"officialDate": "2026-10-04"},
        "game": {"type": "D", "doubleHeader": "N", "gameNumber": 1},
        "teams": {"home": {"id": 158, "league": {"id": 104}}, "away": {"id": 135, "league": {"id": 104}}},
        "venue": {"name": "American Family Field"},
        "players": {f"ID{p['id']}": p for p in [
            _player(1, "Home Ace", "R", "R"), _player(2, "Home Reliever", "L", "L"),
            _player(3, "Away Starter", "R", "L"), _player(10, "Away Hitter", "L", "R"),
            _player(20, "Home Hitter", "S", "R")]},
    },
    "liveData": {
        "linescore": {"teams": {"home": {"runs": 3}, "away": {"runs": 2}}},
        "boxscore": {"teams": {
            "home": {"pitchers": [1, 2], "teamStats": {"fielding": {"errors": 1}}, "players": {
                "ID1": {"stats": {"pitching": {"outs": 18, "numberOfPitches": 95, "earnedRuns": 2}}},
                "ID2": {"stats": {"pitching": {"outs": 3, "numberOfPitches": 12, "earnedRuns": 0}}}}},
            "away": {"pitchers": [3], "teamStats": {"fielding": {"errors": 0}}, "players": {
                "ID3": {"stats": {"pitching": {"outs": 24, "numberOfPitches": 101, "earnedRuns": 3}}}}},
        }},
        "plays": {"allPlays": [
            _play("top", 10, 1, "L", "R", "strikeout"),
            _play("top", 10, 1, "L", "R", "home_run", "fly_ball"),
            _play("top", 10, 2, "L", "L", "walk"),
            _play("top", 10, 2, "L", "L", "sac_fly", "fly_ball"),
            _play("bottom", 20, 3, "R", "L", "single", "line_drive",
                  runners=[{"details": {"eventType": "stolen_base_2b"}}]),
            _play("bottom", 20, 3, "R", "L", "grounded_into_double_play", "ground_ball"),
            _play("bottom", 20, 3, "R", "L", "field_error", "ground_ball"),
        ]},
    },
}


def test_parse_feed(monkeypatch):
    monkeypatch.setattr(SA, "_idmap", {1: "aceh001", 3: "stara001"})
    rows = SA.parse_feed(FEED)
    g = rows["game"]
    assert g["game_id"] == "MIL202610040"
    assert (g["hometeam"], g["visteam"], g["round"]) == ("MIL", "SDN", "DS")
    assert (g["home_score"], g["vis_score"], g["home_win"]) == (3, 2, 1)
    assert g["home_sp"] == "aceh001" and g["vis_sp"] == "stara001"   # register mapping
    pit = {r["player_id"]: r for r in rows["pitcher"]}
    assert pit["aceh001"]["gs"] == 1 and pit["aceh001"]["BF"] == 2
    assert pit["aceh001"]["K"] == 1 and pit["aceh001"]["HR"] == 1 and pit["aceh001"]["F"] == 1
    assert pit["mlb2"]["gs"] == 0 and pit["mlb2"]["BB"] == 1 and pit["mlb2"]["SF"] == 1
    assert pit["aceh001"]["outs"] == 18 and pit["aceh001"]["er"] == 2
    bat = {(r["player_id"], r["vs_hand"]): r for r in rows["batter"]}
    assert bat[("mlb10", "R")]["PA"] == 2 and bat[("mlb10", "R")]["AB"] == 2
    assert bat[("mlb10", "L")]["PA"] == 2 and bat[("mlb10", "L")].get("AB", 0) == 0   # BB + SF
    assert bat[("mlb20", "L")]["bats"] == "R"   # switch hitter batting right vs a lefty
    team = {r["team"]: r for r in rows["team"]}
    assert team["MIL"]["sb"] == 1 and team["MIL"]["errors"] == 1
    assert team["SDN"]["bip"] == 3 and team["SDN"]["bip_reached"] == 2   # 1B, GIDP, ROE
    assert {p["player_id"] for p in rows["players"]} >= {"aceh001", "mlb10"}


def test_doubleheader_game_id(monkeypatch):
    monkeypatch.setattr(SA, "_idmap", {})
    feed = {**FEED, "gameData": {**FEED["gameData"], "game": {"type": "R", "doubleHeader": "Y", "gameNumber": 2}}}
    g = SA.parse_feed(feed)["game"]
    assert g["game_id"] == "MIL202610042" and g["round"] == "REG"
