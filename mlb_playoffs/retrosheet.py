"""Minimal Retrosheet event-file parser.

Turns play-by-play event files into per-game stat lines for pitchers, batters
and teams. Only what the model needs is decoded: plate-appearance outcomes,
batted-ball types, outs (for innings pitched), pitch counts, stolen bases,
errors and earned runs. Base/out state is not reconstructed.

Event-string grammar reference: https://www.retrosheet.org/eventfile.htm
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

PITCH_CHARS = set("BCFHIKLMOPQRSTUVXY")
_PAREN = re.compile(r"\(([^)]*)\)")
_BB_TYPE = re.compile(r"^(BG|BP|BL|G|F|L|P)(\d.*)?$")
_ERR = re.compile(r"E\d")

# Batter outcome codes produced by classify().
PA_OUTCOMES = ("K", "BB", "IBB", "HBP", "1B", "2B", "3B", "HR", "ROE", "FC", "OUT", "CI")
NON_PA = ("SB", "CS", "PO", "POCS", "WP", "PB", "BK", "DI", "OA", "NP", "FLE", "OBS")


@dataclass
class ParsedEvent:
    """Decoded pieces of one Retrosheet event string."""
    outcome: str | None          # one of PA_OUTCOMES, or None for non-PA events
    outs: int = 0
    bb_type: str | None = None   # G, F, L, P (bunts folded into G/P/L)
    sb: int = 0
    cs: int = 0
    errors: int = 0
    sac_fly: bool = False
    sac_hit: bool = False


def _split_top(s: str, sep: str) -> list[str]:
    """Split on ``sep`` only outside parentheses (``PO1(E1/TH)`` stays whole)."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _is_out(s: str) -> bool:
    """A caught-stealing / pickoff / X-advance is an out when some paren group is a
    pure fielding sequence, e.g. ``1X2(46)`` or ``1X2(4E6)(6)``; ``2X3(5E4)`` is safe.
    Groups such as ``(E5/TH)``, ``(NR)`` or ``(UR)`` are annotations."""
    g = _PAREN.findall(s)
    if not g:
        return True
    return any(x[:1].isdigit() and "E" not in x for x in g)


def _paren_outs(s: str) -> int:
    """Count runner outs written as (1)/(2)/(3)/(B) without an error inside."""
    return sum(1 for g in _PAREN.findall(s) if g in ("B", "1", "2", "3"))


def parse_event(ev: str) -> ParsedEvent:
    """Decode a Retrosheet event string (e.g. ``"64(1)3/GDP.2-3"``)."""
    ev = ev.replace("#", "").replace("!", "").replace("?", "")
    basic_part, _, adv_part = ev.partition(".")
    advances = [a for a in adv_part.split(";") if a]
    pieces = _split_top(basic_part, "/")
    basic, mods = pieces[0], pieces[1:]
    parts = basic.split("+")
    main = parts[0]

    p = ParsedEvent(outcome=None)
    p.errors = len(_ERR.findall(ev))
    for m in mods:
        mm = _BB_TYPE.match(m)
        if mm and p.bb_type is None:
            p.bb_type = mm.group(1)[-1]
        if m == "SF":
            p.sac_fly = True
        if m == "SH":
            p.sac_hit = True

    batter_adv_safe = any(a.startswith("B-") for a in advances)
    # Batter put out on the bases (e.g. K.BX1(23)): the advance records the out.
    batter_out_on_bases = any(a.startswith("BX") for a in advances)
    outs = 0

    def runner_events(seg: str) -> int:
        """Outs and SB/CS for stolen-base style segments (SB2;CS3(25) etc.)."""
        o = 0
        for s in seg.split(";"):
            if s.startswith("SB"):
                p.sb += 1
            elif s.startswith("POCS") or s.startswith("CS"):
                p.cs += 1
                o += _is_out(s)
            elif s.startswith("PO"):
                o += _is_out(s)
        return o

    if main.startswith("K"):
        p.outcome = "K"
        if not batter_adv_safe and not batter_out_on_bases:
            outs += 1
        for extra in parts[1:]:
            outs += runner_events(extra)
    elif main in ("W", "IW", "I") or main.startswith("IW"):
        p.outcome = "BB" if main == "W" else "IBB"
        for extra in parts[1:]:
            outs += runner_events(extra)
    elif main.startswith("HP"):
        p.outcome = "HBP"
    elif main.startswith("HR") or re.fullmatch(r"H\d*", main):
        p.outcome = "HR"
    elif main.startswith("DGR") or re.match(r"D\d*$", main) or main == "D":
        p.outcome = "2B"
    elif re.match(r"S\d*$", main):
        p.outcome = "1B"
    elif re.match(r"T\d*$", main):
        p.outcome = "3B"
    elif main.startswith("FLE"):
        p.outcome = None
    elif main.startswith("FC"):
        p.outcome = "FC"
    elif main.startswith("E"):
        p.outcome = "ROE"
    elif main.startswith("C") and not main.startswith("CS"):
        p.outcome = "CI"
    elif main[:1].isdigit():
        # Fielded ball. Error in the sequence (outside parens) => batter safe.
        no_paren = _PAREN.sub("", main)
        if "E" in no_paren:
            p.outcome = "ROE"
        else:
            outs += _paren_outs(main)
            batter_out_explicit = "(B)" in main
            if (not batter_out_explicit and not main.endswith(")") and not batter_adv_safe
                    and not batter_out_on_bases):
                outs += 1
                p.outcome = "OUT"
            elif batter_out_explicit:
                p.outcome = "OUT"
            else:
                p.outcome = "FC" if not batter_adv_safe or main.endswith(")") else "OUT"
    else:
        # Non-PA event: SB, CS, PO, WP, PB, BK, DI, OA, NP ...
        for piece in parts:
            outs += runner_events(piece)

    for a in advances:
        if len(a) >= 3 and a[1] == "X":
            if _is_out(a):
                outs += 1
    p.outs = min(outs, 3)
    return p


def count_pitches(pitches: str) -> int:
    """Number of pitches thrown in a Retrosheet pitch-sequence string."""
    return sum(1 for c in pitches if c in PITCH_CHARS)


@dataclass
class GameAccumulator:
    """Per-game stat lines being accumulated while reading one game."""
    game_id: str
    date: str = ""
    visteam: str = ""
    hometeam: str = ""
    gametype: str = ""
    pitcher: dict = field(default_factory=lambda: {0: None, 1: None})  # by fielding side
    starters: dict = field(default_factory=dict)
    pitchers: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    batters: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    teams: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    half_outs: dict = field(default_factory=lambda: defaultdict(int))
    pitcher_order: list = field(default_factory=list)


def parse_event_file(text: str, hands: dict) -> tuple[list, list, list, list]:
    """Parse one Retrosheet event file.

    Args:
        text: contents of an .EVA/.EVN/.EVE file.
        hands: player_id -> (bats, throws) from the Retrosheet biofile.

    Returns:
        (pitcher_rows, batter_rows, team_rows, game_rows) lists of dicts.
    """
    prow, brow, trow, grow = [], [], [], []
    g: GameAccumulator | None = None
    badj: dict = {}
    padj: dict = {}
    last = None  # (inning, side, batter, pitches, was_pa, fielding pitcher)

    def flush_pending(new_key):
        nonlocal last
        if last is not None:
            key, pitches, was_pa, pit = last
            if not was_pa and key != new_key and pit is not None:
                g.pitchers[pit]["pitches"] += count_pitches(pitches)
        last = None

    def finish():
        if g is None:
            return
        flush_pending(None)
        for pid, st in g.pitchers.items():
            row = {"game_id": g.game_id, "date": g.date, "player_id": pid}
            row.update(st)
            prow.append(row)
        for (pid, phand), st in g.batters.items():
            row = {"game_id": g.game_id, "date": g.date, "player_id": pid, "vs_hand": phand}
            row.update(st)
            brow.append(row)
        for team, st in g.teams.items():
            row = {"game_id": g.game_id, "date": g.date, "team": team}
            row.update(st)
            trow.append(row)
        complete = sum(1 for v in g.half_outs.values() if v == 3)
        grow.append({
            "game_id": g.game_id, "date": g.date, "visteam": g.visteam, "hometeam": g.hometeam,
            "gametype": g.gametype, "vis_sp": g.starters.get(0), "home_sp": g.starters.get(1),
            "half_innings": len(g.half_outs), "half_innings_3_outs": complete,
        })

    for line in text.splitlines():
        if not line:
            continue
        rec = line.split(",")
        kind = rec[0]
        if kind == "id":
            finish()
            g = GameAccumulator(game_id=rec[1])
            badj, padj, last = {}, {}, None
        elif g is None:
            continue
        elif kind == "info":
            key, val = rec[1], ",".join(rec[2:])
            if key == "date":
                g.date = val.replace("/", "-")
            elif key == "visteam":
                g.visteam = val
            elif key == "hometeam":
                g.hometeam = val
            elif key == "gametype":
                g.gametype = val
        elif kind in ("start", "sub"):
            # start,playerid,"name",side,batpos,fieldpos  (name may contain commas)
            pid = rec[1]
            side, fieldpos = int(rec[-3]), int(rec[-1])
            if fieldpos == 1:
                fside = side  # side here = team of the player (0 vis, 1 home)
                g.pitcher[fside] = pid
                team = g.visteam if fside == 0 else g.hometeam
                st = g.pitchers[pid]
                st["team"] = team
                if kind == "start":
                    g.starters[fside] = pid
                    st["gs"] = 1
                st.setdefault("gs", 0)
                st["g"] = 1
                if pid not in g.pitcher_order:
                    g.pitcher_order.append(pid)
        elif kind == "badj":
            badj[rec[1]] = rec[2]
        elif kind == "padj":
            padj[rec[1]] = rec[2]
        elif kind == "data" and rec[1] == "er":
            g.pitchers[rec[2]]["er"] += int(rec[3])
        elif kind == "play":
            inning, bside, batter, _count, pitches, event = rec[1], int(rec[2]), rec[3], rec[4], rec[5], rec[6]
            fside = 1 - bside
            pit = g.pitcher[fside]
            key = (inning, bside, batter)
            flush_pending(key)
            if event == "NP":
                continue
            ev = parse_event(event)
            bat_team = g.visteam if bside == 0 else g.hometeam
            fld_team = g.hometeam if bside == 0 else g.visteam
            g.half_outs[(int(inning), bside)] += ev.outs
            tb, tf = g.teams[bat_team], g.teams[fld_team]
            tb["sb"] += ev.sb
            tb["cs"] += ev.cs
            tf["errors"] += ev.errors
            if pit is not None:
                g.pitchers[pit]["outs"] += ev.outs
            is_pa = ev.outcome is not None
            last = (key, pitches, is_pa, pit)
            if not is_pa:
                continue
            if pit is not None:
                g.pitchers[pit]["pitches"] += count_pitches(pitches)
            last = None
            o = ev.outcome
            p_throw = padj.pop(pit, None) or hands.get(pit, ("R", "R"))[1] or "R"
            b_bats = badj.pop(batter, None) or hands.get(batter, ("R", "R"))[0] or "R"
            if b_bats == "B":
                b_bats = "L" if p_throw == "R" else "R"
            stat = {"PA": 1}
            if o in ("K",):
                stat["K"] = 1
            elif o == "BB":
                stat["BB"] = 1
            elif o == "IBB":
                stat["IBB"] = 1
            elif o == "HBP":
                stat["HBP"] = 1
            elif o in ("1B", "2B", "3B", "HR"):
                stat[o] = 1
                stat["H"] = 1
            elif o == "ROE":
                stat["ROE"] = 1
            if ev.sac_fly:
                stat["SF"] = 1
            if ev.sac_hit:
                stat["SH"] = 1
            if o not in ("BB", "IBB", "HBP", "CI") and not ev.sac_fly and not ev.sac_hit:
                stat["AB"] = 1
            if ev.bb_type and o not in ("K", "BB", "IBB", "HBP"):
                stat[ev.bb_type] = 1
            # Ball in play (for defensive efficiency): not K/BB/HBP/HR/CI.
            if o in ("1B", "2B", "3B", "ROE", "FC", "OUT"):
                tf["bip"] += 1
                if o in ("1B", "2B", "3B", "ROE"):
                    tf["bip_reached"] += 1
            if pit is not None:
                ps = g.pitchers[pit]
                ps["BF"] += 1
                for k, v in stat.items():
                    if k != "PA":
                        ps[k] += v
                ps["BF_vsL" if b_bats == "L" else "BF_vsR"] += 1
            bs = g.batters[(batter, p_throw)]
            bs["team"] = bat_team
            bs["bats"] = b_bats
            for k, v in stat.items():
                bs[k] += v
            for k, v in stat.items():
                if k in ("PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K", "SF"):
                    tb[k] += v
    finish()
    # Textual 'team'/'bats' fields were stored in float dicts; normalise.
    return prow, brow, trow, grow
