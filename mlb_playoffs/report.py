"""Command-line reports and charts.

    python report.py game "Team A" "Team B"      # single game, Team A at home
    python report.py series "Team A" "Team B"    # series odds (default: World Series format)
    python report.py bracket                     # full playoff odds table

Common options:
    --season 2025          bracket file data/manual/bracket_<season>.json
    --asof YYYY-MM-DD      state of the world / games locked in before this date
    --sims N               Monte Carlo simulations (default 10000)
    --injuries PATH        injuries CSV (default data/manual/injuries.csv)

game options:   --home-sp NAME  --away-sp NAME   (override probable starters)
                --total 8.5  --f5-total 4.5      (price specific total lines)
series options: --round WC|DS|LCS|WS

Charts are written to charts/.
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
import data
import plotstyle as S
from simulate import PlayoffSimulator, display, resolve_player, resolve_team

_names = None


def pname(pid: str) -> str:
    global _names
    if _names is None:
        _names = data.load("players").set_index("player_id")["name"].to_dict()
    return _names.get(pid, pid)


def pct(x: float) -> str:
    return f"{x:.1%}"


# --------------------------------------------------------------------------- #
# Head-to-head comparison
# --------------------------------------------------------------------------- #

def comparison_table(sim: PlayoffSimulator, a: str, b: str, sp_a: str, sp_b: str) -> pd.DataFrame:
    """Key stats side by side (as of the simulator's as-of date)."""
    st, day = sim.state, sim.day0
    rows = []
    ta, tb = st.team_line(a, day), st.team_line(b, day)
    la_line, lb_line = st.pitcher_line(sp_a, day), st.pitcher_line(sp_b, day)
    lin_a = st.lineup(a, lb_line["throws"], day, sim.out)
    lin_b = st.lineup(b, la_line["throws"], day, sim.out)
    pa, pb = st.bullpen(a, day, out=sim.out), st.bullpen(b, day, out=sim.out)
    rec = sim.records

    def add(label, va, vb, fmt, better="high"):
        rows.append({"stat": label, display(a): fmt.format(va), display(b): fmt.format(vb),
                     "edge": "" if va == vb else display(a if (va > vb) == (better == "high") else b)})

    add("Regular-season win%", rec.get(a, np.nan), rec.get(b, np.nan), "{:.3f}")
    add("Elo rating", ta["elo"], tb["elo"], "{:.0f}")
    add("Pythagorean win% (regressed)", ta["pyth"], tb["pyth"], "{:.3f}")
    add("Runs scored / game", ta["rs_pg"], tb["rs_pg"], "{:.2f}")
    add("Runs allowed / game", ta["ra_pg"], tb["ra_pg"], "{:.2f}", "low")
    add("Recent run diff / game", ta["form"], tb["form"], "{:+.2f}")
    add(f"Lineup wOBA vs opp. SP hand", lin_a["woba"], lin_b["woba"], "{:.3f}")
    add("Lineup ISO", lin_a["iso"], lin_b["iso"], "{:.3f}")
    add("Lineup K%", lin_a["k"], lin_b["k"], "{:.1%}", "low")
    add("Starter", 0, 0, "")
    rows[-1].update({display(a): f"{pname(sp_a)} ({la_line['throws']})",
                     display(b): f"{pname(sp_b)} ({lb_line['throws']})", "edge": ""})
    add("  Starter FIP", la_line["fip"], lb_line["fip"], "{:.2f}", "low")
    add("  Starter xFIP", la_line["xfip"], lb_line["xfip"], "{:.2f}", "low")
    add("  Starter K-BB%", la_line["kbb"], lb_line["kbb"], "{:.1%}")
    add("  Starter IP / start", la_line["depth"], lb_line["depth"], "{:.1f}")
    add("Top-5 bullpen FIP", pa["top_fip"], pb["top_fip"], "{:.2f}", "low")
    add("Full bullpen FIP", pa["all_fip"], pb["all_fip"], "{:.2f}", "low")
    add("Defensive efficiency", ta["der"], tb["der"], "{:.3f}")
    add("Baserunning runs / game", ta["baserun"], tb["baserun"], "{:+.3f}")
    return pd.DataFrame(rows)


def explain_lines(sim, row: dict, home: str, away: str, top=6) -> list[str]:
    out = []
    for label, contrib, _ in sim.model.explain(row, top=top):
        if abs(contrib) < 0.005:
            continue
        side = display(home) if contrib > 0 else display(away)
        out.append(f"  {label:34s} favors {side:4s} ({contrib:+.3f} log-odds for {display(home)})")
    return out


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def chart_factors(sim, row, home, away, path, title):
    """Diverging bar chart of each feature's contribution to the home team's log-odds."""
    contribs = sim.model.explain(row, top=10)
    contribs = [c for c in contribs if abs(c[1]) >= 0.003][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(contribs) + 1.4))
    vals = [c[1] for c in contribs]
    ax.barh(range(len(vals)), vals, color=[S.POS if v > 0 else S.NEG for v in vals], height=0.6)
    ax.set_yticks(range(len(vals)), [c[0] for c in contribs])
    ax.axvline(0, color=S.TEXT_2, lw=0.8)
    lim = max(abs(v) for v in vals) * 1.25 if vals else 1
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to win probability (log-odds)")
    ax.text(1, 1.01, f"favors {display(home)} (home) →", transform=ax.transAxes, ha="right", va="bottom",
            color=S.TEXT_2, fontsize=9)
    ax.text(0, 1.01, f"← favors {display(away)}", transform=ax.transAxes, ha="left", va="bottom",
            color=S.TEXT_2, fontsize=9)
    ax.set_title(title, pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_staff(sim, a, b, path):
    """Rotation and top-5 bullpen FIP for both teams (lower is better)."""
    st, day = sim.state, sim.day0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, (kind, title) in zip(axes, [("rot", "Projected rotation, FIP"), ("pen", "Top-5 bullpen, FIP")]):
        labels, vals, colors = [], [], []
        for i, t in enumerate((a, b)):
            ps = sim.rotation_for(t) if kind == "rot" else st.bullpen(t, day, out=sim.out)["top"]
            for p in ps:
                labels.append(f"{display(t)}  {pname(p)}")
                vals.append(st.pitcher_line(p, day)["fip"])
                colors.append(S.SERIES[i])
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=colors, height=0.5)
        for yi, v in zip(y, vals):
            ax.text(v + 0.04, yi, f"{v:.2f}", va="center", fontsize=8, color=S.TEXT_2)
        ax.set_yticks(y, labels)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel("FIP (regressed; lower is better)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.SERIES[i]) for i in range(2)]
    fig.legend(handles, [display(a), display(b)], loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_bracket(odds: pd.DataFrame, path, title):
    """Heatmap of each team's odds to reach each round (single-hue sequential ramp)."""
    from matplotlib.colors import LinearSegmentedColormap
    cols = ["make_ds", "make_lcs", "pennant", "win_ws"]
    heads = ["Reach DS", "Reach LCS", "Win pennant", "Win World Series"]
    o = odds.copy()
    o["label"] = o["team"] + "  (" + o["league"] + " " + o["seed"].astype(str) + ")"
    o = o.sort_values(["win_ws", "pennant"], ascending=False)
    m = o[cols].values
    cmap = LinearSegmentedColormap.from_list("seq", [S.SURFACE] + S.SEQ)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(o) + 1.6))
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, f"{v:.0%}" if v >= 0.005 else "–", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else S.TEXT)
    ax.set_xticks(range(len(cols)), heads)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(o)), o["label"])
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(o)), minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_game(args):
    sim = PlayoffSimulator(args.season, args.asof, seed=args.seed, injuries=args.injuries)
    home, away = resolve_team(args.team_a), resolve_team(args.team_b)
    day = sim.day0
    last = dict(sim.last_pitched)
    sim.rotation_for(home), sim.rotation_for(away)
    last.update(sim.last_pitched)
    if args.home_sp:
        hsp = resolve_player(args.home_sp, home, sim.state)
    else:
        hsp, _ = sim.pick_starter(home, "", 0, day, last)
    if args.away_sp:
        asp = resolve_player(args.away_sp, away, sim.state)
    else:
        asp, _ = sim.pick_starter(away, "", 0, day, last)
    row, _ = sim.state.matchup(home, away, hsp, asp, day, is_post=1, out=sim.out)
    p = float(sim.model.predict_proba(pd.DataFrame([row]))[0])

    print(f"\n{display(away)} @ {display(home)}  —  as of {sim.as_of.date()}")
    print(f"Starters: {pname(asp)} vs {pname(hsp)}")
    print(f"\n  {display(home):4s} {pct(p):>6s}   {'█' * round(p * 40)}")
    print(f"  {display(away):4s} {pct(1 - p):>6s}   {'█' * round((1 - p) * 40)}")
    print("\nTop factors:")
    print("\n".join(explain_lines(sim, row, home, away)))
    print("\nHead-to-head:")
    print(comparison_table(sim, home, away, hsp, asp).to_string(index=False))
    path = C.CHART_DIR / f"game_{display(away)}_at_{display(home)}.png"
    chart_factors(sim, row, home, away, path,
                  f"{display(away)} @ {display(home)}: {display(home)} {pct(p)}  ({pname(asp)} vs {pname(hsp)})")
    staff = C.CHART_DIR / f"staff_{display(home)}_{display(away)}.png"
    chart_staff(sim, home, away, staff)
    print_runs(sim, row, home, away, args)
    print(f"\nCharts: {path}\n        {staff}")


def load_runs_model(sim):
    """Runs model matching the win model's training cutoff (falls back to production)."""
    import runs
    name = sim.bracket.get("model", "production")
    try:
        return runs.load(name)
    except FileNotFoundError:
        return runs.load("production")


def print_runs(sim, row, home, away, args):
    """First inning, F3, F5 and full-game run projections (runs.py)."""
    import runs
    model = load_runs_model(sim)
    h_row, a_row = runs.matchup_rows(row, is_post=int(row.get("is_postseason", 1)))
    lines = [("F5", args.f5_total)] if args.f5_total is not None else []
    if args.total is not None:
        lines.append(("Full", args.total))
    o = runs.game_outputs(model, h_row, a_row, n=args.sims, seed=args.seed, lines=lines)
    H, A = display(home), display(away)
    print(f"\nRuns, first five and totals ({args.sims:,} simulations):")
    e = o["exp_runs"]
    print(f"  Expected runs        {'1st':>6s} {'F3':>6s} {'F5':>6s} {'Full':>6s}")
    for side, t in (("away", A), ("home", H)):
        print(f"    {t:4s}               {e[side]['1st']:6.2f} {e[side]['F3']:6.2f} {e[side]['F5']:6.2f} {e[side]['Full']:6.2f}")
    print(f"  F5 average runs (incl. 0):  {H} {o['f5_avg_incl0'][0]:.2f}   {A} {o['f5_avg_incl0'][1]:.2f}   "
          f"DIF {o['f5_avg_incl0'][0] - o['f5_avg_incl0'][1]:+.2f}")
    print(f"  F5 average runs (excl. 0):  {H} {o['f5_avg_excl0'][0]:.2f}   {A} {o['f5_avg_excl0'][1]:.2f}   "
          f"DIF {o['f5_avg_excl0'][0] - o['f5_avg_excl0'][1]:+.2f}")
    print(f"  NRFI {pct(o['nrfi'])}   YRFI {pct(1 - o['nrfi'])}   (MLB average ~50%)")
    for k in ("f3", "f5"):
        d = o[k]
        nt = d["home"] / (d["home"] + d["away"])
        fav, pfav = (H, nt) if nt >= 0.5 else (A, 1 - nt)
        print(f"  {k.upper()} 3-way: {H} {pct(d['home'])}  tie {pct(d['tie'])}  {A} {pct(d['away'])}   "
              f"-> {fav} {pct(pfav)} excluding ties")
    m = o["f5_margin_mode"]
    print(f"  Most common F5 margin: {abs(m)} run(s) in favor of {H if m > 0 else A}")
    print(f"  Full game (run model): {H} {pct(o['full_home_win'])}   (win model: {H} {pct(sim.model.predict_proba(pd.DataFrame([row]))[0])})")
    print(f"  Projected totals (median / mean):  F3 {o['median_total']['F3']:.1f}/{o['mean_total']['F3']:.2f}   "
          f"F5 {o['median_total']['F5']:.1f}/{o['mean_total']['F5']:.2f}   Full {o['median_total']['Full']:.1f}/{o['mean_total']['Full']:.2f}")
    print("  Over / under:")
    for (k, line), (po, pu) in sorted(o["totals"].items()):
        print(f"    {k:4s} {line:4.1f}   over {pct(po):>6s}   under {pct(pu):>6s}")
    for side, t in (("away", A), ("home", H)):
        tt = o["team_totals_f5"][side]
        print(f"  {t} F5 team total: over 1.5 {pct(tt[1.5])}, over 2.5 {pct(tt[2.5])}")


def cmd_series(args):
    sim = PlayoffSimulator(args.season, args.asof, seed=args.seed, injuries=args.injuries)
    a, b = resolve_team(args.team_a), resolve_team(args.team_b)
    res = sim.series_odds(a, b, args.round, n=args.sims, lock=True)
    hi = res["home_field"]
    lo = b if hi == a else a
    print(f"\n{args.round} series: {display(a)} vs {display(b)}  (home field: {display(hi)})  —  as of {sim.as_of.date()}")
    for t in (a, b):
        print(f"  {display(t):4s} {pct(res['p'][t]):>6s}   {'█' * round(res['p'][t] * 40)}")
    print("\nProjected games (if needed):")
    games = sim.projected_games(a, b, args.round)
    played = sim.played(args.round, a, b)
    for g in games:
        tag = ""
        if g["game"] <= len(played):
            r = played[g["game"] - 1]
            tag = f"   FINAL {display(r['away'])} {r['away_score']:.0f} - {display(r['home'])} {r['home_score']:.0f}"
        print(f"  G{g['game']} {g['date']}  {display(g['away']):4s} @ {display(g['home']):4s}  "
              f"{pname(g['away_sp']):22s} vs {pname(g['home_sp']):22s}  "
              f"{display(g['home']) + ' ' + pct(g['p_home']) if not tag else ''}{tag}")
    g1 = next((g for g in games if g["p_home"] == g["p_home"]), games[0])  # next unplayed game
    row, _ = sim.state.matchup(g1["home"], g1["away"], g1["home_sp"], g1["away_sp"], sim.day0, is_post=1,
                               rest=(1, 1), out=sim.out, sp_rest=g1["sp_rest"])
    print(f"\nTop factors (Game {g1['game']}, {display(g1['home'])} perspective):")
    print("\n".join(explain_lines(sim, row, g1["home"], g1["away"])))
    print("\nHead-to-head:")
    print(comparison_table(sim, hi, lo, sim.rotation_for(hi)[0], sim.rotation_for(lo)[0]).to_string(index=False))
    path = C.CHART_DIR / f"series_{display(hi)}_{display(lo)}.png"
    chart_factors(sim, row, g1["home"], g1["away"], path,
                  f"{args.round} {display(a)} vs {display(b)}: {display(hi)} {pct(res['p'][hi])} to win series")
    staff = C.CHART_DIR / f"staff_{display(hi)}_{display(lo)}.png"
    chart_staff(sim, hi, lo, staff)
    print(f"\nCharts: {path}\n        {staff}")


def cmd_bracket(args):
    sim = PlayoffSimulator(args.season, args.asof, seed=args.seed, injuries=args.injuries)
    print(f"\nProjected rotations (as of {sim.as_of.date()}):")
    for t, rot in sim.rotations.items():
        print(f"  {display(t):4s} {', '.join(pname(p) for p in rot)}")
    if sim.out:
        print("Unavailable (injuries.csv): " + ", ".join(pname(p) for p in sim.out))
    odds = sim.run(args.sims)
    show = odds.copy()
    for c in ("make_ds", "make_lcs", "pennant", "win_ws"):
        show[c] = show[c].map(lambda x: f"{x:6.1%}")
    show = show.rename(columns={"make_ds": "Reach DS", "make_lcs": "Reach LCS", "pennant": "Pennant",
                                "win_ws": "Win WS"}).drop(columns=["code"])
    print(f"\n{args.season} postseason odds, {args.sims:,} simulations, as of {sim.as_of.date()}:")
    print(show.to_string(index=False))
    odds.to_csv(C.DATA_DIR / f"bracket_odds_{args.season}_{sim.as_of.date()}.csv", index=False)
    path = C.CHART_DIR / f"bracket_{args.season}_{sim.as_of.date()}.png"
    chart_bracket(odds, path, f"{args.season} postseason odds as of {sim.as_of.date()}")
    print(f"\nChart: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--injuries", default=None, help="injuries CSV (default data/manual/injuries.csv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("game", help="single-game prediction (first team is home)")
    g.add_argument("team_a"); g.add_argument("team_b")
    g.add_argument("--home-sp"); g.add_argument("--away-sp")
    g.add_argument("--total", type=float, help="full-game total line to price, e.g. 8.5")
    g.add_argument("--f5-total", type=float, help="first-five total line to price, e.g. 4.5")
    s = sub.add_parser("series", help="series odds")
    s.add_argument("team_a"); s.add_argument("team_b")
    s.add_argument("--round", default="WS", choices=["WC", "DS", "LCS", "WS"])
    sub.add_parser("bracket", help="full playoff odds")
    args = ap.parse_args()
    # Allow global options after the subcommand too.
    {"game": cmd_game, "series": cmd_series, "bracket": cmd_bracket}[args.cmd](args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
