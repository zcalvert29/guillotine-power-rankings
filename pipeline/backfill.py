"""
Rebuild pre-kickoff chop odds for every finished week, for the Track record tab.

For week W it uses only what existed before W's games:
  - each surviving team's roster in week W (Sleeper's matchup roster for that week)
  - Sleeper's player projections for week W
  - score spread fitted on stats from weeks before W (position defaults in week 1)
and plays week W 50,000 times: chance each team posts the lowest score.
FAAB and waivers don't enter a one-week chop call, and no future-week projections are used, so there's no look-ahead.

    python -m pipeline.backfill                   # live (GitHub Action)
    python -m pipeline.backfill --dir saved_pull  # saved pull (only weeks it has projections for)
"""
import argparse
import collections
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import numpy as np

from pipeline.model import DEFAULT_SD, SKILL, Engine, _team_name, build_context, move_impacts, simulate, summarize
from pipeline.sleeper import DirSource, LiveSource

DEFAULT_LEAGUE = "1389750406673883136"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAWS = 50000


def fit_sd(stat_weeks, P):
    hist = collections.defaultdict(list)
    for rows in stat_weeks:
        for x in rows or []:
            s = x.get("stats") or {}
            if s.get("gp", 0) >= 1 or s.get("gms_active", 0) >= 1:
                hist[x["player_id"]].append(float(s.get("pts_ppr") or 0.0))
    sd = dict(DEFAULT_SD)
    for pos in ["QB", "RB", "WR", "TE"]:
        M, S = [], []
        for pid, v in hist.items():
            if P.get(pid, {}).get("pos") == pos and len(v) >= 3 and np.mean(v) >= 5:
                M.append(np.mean(v)); S.append(np.std(v, ddof=1))
        if len(M) >= 15:
            b, a = np.polyfit(M, S, 1)
            sd[pos] = (float(a), float(b))
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default=os.environ.get("LEAGUE_ID", DEFAULT_LEAGUE))
    ap.add_argument("--dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "history", "backfill.json"))
    ap.add_argument("--snap-dir", default=os.path.join(ROOT, "snapshots"))
    ap.add_argument("--sims", type=int, default=int(os.environ.get("N_SIMS_REBUILD", 3000)))
    ap.add_argument("--impact-sims", type=int, default=int(os.environ.get("N_IMPACT_SIMS", 1000)))
    ap.add_argument("--no-snapshots", action="store_true", help="only grade, don't rebuild full weekly snapshots")
    ap.add_argument("--force", action="store_true", help="rebuild snapshots even if one exists for that week")
    a = ap.parse_args()
    src = DirSource(a.dir) if a.dir else LiveSource(a.league)

    league, state = src.league(), src.state()
    season = str(league["season"])
    cur = int(state.get("week") or 1)
    users = {u["user_id"]: u for u in src.users() or []}
    rosters = {r["roster_id"]: r for r in src.rosters()}
    names = {rid: _team_name(users.get(r.get("owner_id"), {})) if r.get("owner_id") else f"Team {rid}" for rid, r in rosters.items()}
    slots = [s for s in league["roster_positions"] if s not in ("BN", "IR", "TAXI")]

    P = {}
    for pid, p in (src.players() or {}).items():
        pos = "DEF" if pid.isalpha() else p.get("position")
        if pos in SKILL:
            P[pid] = dict(name=p.get("full_name") or pid, pos=pos, team=p.get("team") or (pid if pos == "DEF" else None), injury=None)

    tx = []
    for w in range(1, cur + 2):
        tx += src.transactions(w) or []
    chops = {t["leg"]: t["roster_ids"][0] for t in tx if t.get("type") == "chopped"}

    stats_cache = {}
    out_weeks = []
    for W in sorted(chops):
        proj_rows = src.projections(season, W)
        mu_rows = src.matchups(W)
        if not proj_rows or not mu_rows:
            print(f"week {W}: missing projections or matchups, skipped")
            continue
        alive = [rid for rid in rosters if not any(chops[l] == rid for l in chops if l < W)]
        roster_w = {m["roster_id"]: [p for p in (m.get("players") or []) if p in P] for m in mu_rows}
        proj = collections.defaultdict(dict)
        for x in proj_rows:
            proj[x["player_id"]][W] = float((x.get("stats") or {}).get("pts_ppr") or 0.0)
        for w in range(1, W):
            if w not in stats_cache:
                stats_cache[w] = src.stats(season, w)
        sd = fit_sd([stats_cache[w] for w in range(1, W)], P)
        taken = {p for rid in alive for p in roster_w.get(rid, [])}
        repl = {}
        for pos in SKILL:
            vals = sorted((proj[p].get(W, 0.0) for p in proj if p not in taken and P.get(p, {}).get("pos") == pos), reverse=True)
            repl[(pos, W)] = vals[1] if len(vals) > 1 else 0.0
        E = Engine(dict(P=P, proj=proj, repl=repl, slots=slots, sd_fit=sd, final=W))
        mus, sds = [], []
        for rid in alive:
            mu, var = E.lineup(roster_w.get(rid, []), W)
            mus.append(mu); sds.append(np.sqrt(var))
        rng = np.random.default_rng(1000 + W)
        draws = rng.normal(np.array(mus)[None, :], np.array(sds)[None, :], size=(DRAWS, len(alive)))
        losers = np.bincount(draws.argmin(axis=1), minlength=len(alive)) / DRAWS
        out_weeks.append(dict(
            week=W, chopped_roster_id=chops[W], chopped=names[chops[W]],
            teams=[dict(roster_id=rid, name=names[rid], p_chop=float(losers[i]), proj=round(float(mus[i]), 1),
                        sd=round(float(sds[i]), 1)) for i, rid in enumerate(alive)],
            sd_fit=sd))
        r = sorted(alive, key=lambda x: -losers[alive.index(x)])
        print(f"week {W}: {len(alive)} alive, chopped {names[chops[W]]} "
              f"(model {losers[alive.index(chops[W])]:.0%}, rank {r.index(chops[W]) + 1}); top pick {names[r[0]]} {losers[alive.index(r[0])]:.0%}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(generated_at=dt.datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="minutes"),
                       season=season, draws=DRAWS, weeks=out_weeks), fh, separators=(",", ":"))
    print(f"wrote {a.out}: {len(out_weeks)} weeks")
    if not a.no_snapshots:
        rebuild_snapshots(src, a)


def rebuild_snapshots(src, a):
    """Full pre-kickoff snapshot (rankings, FAAB, waivers...) for each week that has started but has no pre-kickoff snapshot."""
    league = src.league()
    season = str(league["season"])
    schedule = src.schedule(season) or []
    first_day = {}
    for g in schedule:
        first_day[g["week"]] = min(first_day.get(g["week"], g["date"]), g["date"])
    today = dt.datetime.now(ZoneInfo("America/Chicago")).date().isoformat()
    existing = {}
    os.makedirs(a.snap_dir, exist_ok=True)
    for f in os.listdir(a.snap_dir):
        if f.endswith(".json"):
            with open(os.path.join(a.snap_dir, f)) as fh:
                s = json.load(fh)
            if not s.get("week_in_progress"):
                existing[s["start_week"]] = f
    live_start = build_context(src)["start"]
    for W in sorted(w for w in first_day if w <= live_start and first_day[w] <= today):
        if W in existing and not a.force:
            continue
        ctx = build_context(src, as_of_week=W)
        if len(ctx["alive"]) <= 1:
            continue
        label = (dt.date.fromisoformat(first_day[W]) - dt.timedelta(days=1)).isoformat()
        res = simulate(ctx, nsim=a.sims, seed=2026)
        snap = summarize(ctx, res, generated_at=f"{label}T23:59-05:00")
        snap["snapshot_date"] = label
        snap["moves"] = move_impacts(ctx, ctx["prev_cutoff_ms"], ctx["as_of_ms"], nsim=a.impact_sims, top=6)
        path = os.path.join(a.snap_dir, f"{label}.json")
        with open(path, "w") as fh:
            json.dump(snap, fh, separators=(",", ":"))
        top = snap["teams"][0]
        big = max(snap["moves"], key=lambda m: max(abs(x["d_win"]) for x in m["teams"]), default=None)
        print(f"rebuilt Week {W} -> {path}: {len(ctx['alive'])} alive, #1 {top['name']} ({top['p_win']:.0%} win); "
              f"{len(snap['moves'])} moves scored" + (f", biggest: {big['teams'][0]['name']} {big['teams'][0]['got']} {big['teams'][0]['d_win']:+.1%}" if big else ""))


if __name__ == "__main__":
    main()
