"""Unit tests for the Retrosheet event parser."""
import pytest

from retrosheet import count_pitches, parse_event, parse_event_file


@pytest.mark.parametrize("ev,outcome,outs", [
    ("K", "K", 1),
    ("K.BX1(23)", "K", 1),            # dropped third strike, thrown out
    ("K+WP.B-1", "K", 0),             # reached on wild pitch on strike three
    ("K+CS2(26)/DP", "K", 2),
    ("63/G6", "OUT", 1),
    ("64(1)3/GDP", "OUT", 2),
    ("8(B)84(2)/LDP", "OUT", 2),
    ("54(1)/FO/G56.B-1", "FC", 1),
    ("S9/L9LS.3-H;2-H;1-3", "1B", 0),
    ("D7/L78M.1-H;BX3(76354/TH)", "2B", 1),
    ("HR/F7D.2-H", "HR", 0),
    ("W.1-2", "BB", 0),
    ("IW", "IBB", 0),
    ("HP.2-3;1-2", "HBP", 0),
    ("E6/G6.1-2", "ROE", 0),
    ("FC/G4.2-3;1X2(4E6)(6);B-1", "FC", 1),
    ("S5/G56S.BX3(25)(E5/TH)", "1B", 1),
    ("7/SF/F7D+.3-H", "OUT", 1),
    ("PO1(E1/TH).1-3", None, 0),
    ("POCS2(1361)", None, 1),
    ("SB2.1XH(2)(E2/TH)", None, 1),
    ("CS2(E2).1-3", None, 0),
    ("SB2;SB3", None, 0),
])
def test_parse_event(ev, outcome, outs):
    p = parse_event(ev)
    assert p.outcome == outcome
    assert p.outs == outs


def test_batted_ball_and_steals():
    assert parse_event("43/G34").bb_type == "G"
    assert parse_event("8/F89D").bb_type == "F"
    assert parse_event("S8/L78").bb_type == "L"
    assert parse_event("SB2;SB3").sb == 2
    assert parse_event("K+CS2(26)").cs == 1


def test_count_pitches():
    assert count_pitches("CBBBC.F*B") == 7
    assert count_pitches("BCFX") == 4
    assert count_pitches("") == 0


GAME = """id,TST202501010
info,visteam,AAA
info,hometeam,BBB
info,date,2025/01/01
start,v1,"Vis One",0,1,8
start,vp,"Vis Pitcher",0,0,1
start,h1,"Home One",1,1,8
start,hp,"Home Pitcher",1,0,1
play,1,0,v1,32,BBCFB,W
play,1,0,v1,00,X,63/G6.1-2
play,1,0,v1,02,CCS,K
play,1,0,v1,10,BX,8/F8
play,1,1,h1,00,X,HR/F7
sub,rp,"Relief",1,0,1
play,2,0,v1,22,BBCC,K
data,er,vp,1
"""


def test_parse_event_file_attribution():
    hands = {"v1": ("L", "R"), "vp": ("R", "R"), "h1": ("B", "R"), "hp": ("R", "L"), "rp": ("R", "R")}
    p, b, t, g = parse_event_file(GAME, hands)
    pit = {r["player_id"]: r for r in p}
    assert pit["hp"]["BF"] == 4 and pit["hp"]["outs"] == 3 and pit["hp"]["K"] == 1
    assert pit["hp"]["pitches"] == 5 + 1 + 3 + 2
    assert pit["vp"]["HR"] == 1 and pit["vp"]["er"] == 1
    assert pit["rp"]["K"] == 1 and pit["rp"]["gs"] == 0
    assert g[0]["home_sp"] == "hp" and g[0]["vis_sp"] == "vp"
    # Switch hitter h1 bats left against a right-handed pitcher.
    hb = [r for r in b if r["player_id"] == "h1"][0]
    assert hb["bats"] == "L" and hb["vs_hand"] == "R"
    # v1 faced the left-handed starter four times, then the right-handed reliever.
    pa = {r["vs_hand"]: r["PA"] for r in b if r["player_id"] == "v1"}
    assert pa == {"L": 4, "R": 1}
