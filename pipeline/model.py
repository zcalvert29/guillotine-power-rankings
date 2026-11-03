"""
Guillotine league forward-looking model.

Each simulated season, week by week:
  1. run a FAAB auction for the most recently chopped roster (if its waivers haven't run yet)
  2. draw every surviving team's score from its best projected lineup
  3. chop the lowest score; that roster becomes next week's auction pool
Repeat until one team is left. Rankings come from thousands of simulated seasons.
"""
import collections
import math
import multiprocessing as mp
import random

import numpy as np

SKILL = ["QB", "RB", "WR", "TE", "K", "DEF"]
FLEX_ELIG = {
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
FLEX_ORDER = ["REC_FLEX", "WRRB_FLEX", "FLEX", "SUPER_FLEX"]  # most restrictive first
DEFAULT_SD = {"QB": (4.0, .16), "RB": (3.3, .27), "WR": (2.8, .37), "TE": (4.8, .19),
              "K": (4.5, .10), "DEF": (4.5, .30)}

PARAMS = dict(
    tau=75.0,          # FAAB aggressiveness: bid share of budget = 1 - exp(-ROS points gained / tau)
    bid_noise=0.35,    # lognormal noise on each bid
    min_gain=3.0,      # teams don't bid on players who add < this many ROS points
    pool_min_ros=15.0, # ignore dumped players projected for less than this ROS
    pool_top=7,        # only auction the top-N dumped players
    repl_rank=2,       # free-agent streamer level = Nth best unrostered projection
    max_active=None,   # set from league roster size
)


# ----------------------------------------------------------------------------- context

def _team_name(u):
    md = u.get("metadata") or {}
    return md.get("team_name") or u.get("display_name")


def build_context(src):
    league, state = src.league(), src.state()
    season = str(league["season"])
    cur = int(state.get("week") or state.get("display_week") or 1)
    users = {u["user_id"]: u for u in (src.users() or [])}
    rosters = src.rosters()
    roster_positions = league["roster_positions"]

    players_raw = src.players() or {}
    P = {}
    for pid, p in players_raw.items():
        pos = p.get("position")
        if pid.isalpha():
            pos = "DEF"
        if pos in SKILL:
            P[pid] = dict(name=p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip() or pid,
                          pos=pos, team=p.get("team") or (pid if pos == "DEF" else None),
                          injury=p.get("injury_status"))

    # transactions -> chops, last waiver processing time
    tx = []
    for w in range(1, cur + 2):
        tx += src.transactions(w) or []
    chops = sorted([(t["leg"], t["roster_ids"][0], list((t.get("drops") or {}).keys()), t["created"])
                    for t in tx if t.get("type") == "chopped"])
    chopped_legs = {c[0] for c in chops}
    last_waiver_run = max([t.get("status_updated") or 0 for t in tx
                           if t.get("type") == "waiver" and t.get("status") == "complete"] or [0])

    alive = [r["roster_id"] for r in rosters if not (r.get("settings") or {}).get("eliminated")]
    start = cur + 1 if cur in chopped_legs else cur
    final = start + len(alive) - 2

    schedule = src.schedule(season) or []
    games = collections.defaultdict(dict)  # week -> team -> status
    for g in schedule:
        for side in ("home", "away"):
            games[g["week"]][g[side]] = g.get("status")
    nfl_teams = {t for w in games for t in games[w]}
    byes = {t: [w for w in range(1, 19) if t not in games.get(w, {})] for t in nfl_teams}

    proj = collections.defaultdict(dict)
    for w in range(start, min(max(final, start), 18) + 1):
        for x in src.projections(season, w) or []:
            proj[x["player_id"]][w] = float((x.get("stats") or {}).get("pts_ppr") or 0.0)
            if x["player_id"] not in P and (x.get("player") or {}).get("position") in SKILL:
                pl = x["player"]
                P[x["player_id"]] = dict(name=f"{pl.get('first_name','')} {pl.get('last_name','')}".strip(),
                                         pos=pl["position"], team=x.get("team"), injury=pl.get("injury_status"))

    # weekly variance fit from this season's actual scores
    hist = collections.defaultdict(list)
    for w in range(1, start):
        for x in src.stats(season, w) or []:
            s = x.get("stats") or {}
            if s.get("gp", 0) >= 1 or s.get("gms_active", 0) >= 1:
                hist[x["player_id"]].append(float(s.get("pts_ppr") or 0.0))
    sd_fit = dict(DEFAULT_SD)
    for pos in ["QB", "RB", "WR", "TE"]:
        M, S = [], []
        for pid, v in hist.items():
            if P.get(pid, {}).get("pos") == pos and len(v) >= 4 and np.mean(v) >= 5:
                M.append(np.mean(v)); S.append(np.std(v, ddof=1))
        if len(M) >= 15:
            b, a = np.polyfit(M, S, 1)
            sd_fit[pos] = (float(a), float(b))

    rmap = {r["roster_id"]: r for r in rosters}
    roster_players = {rid: [p for p in (rmap[rid].get("players") or []) if p in P] for rid in rmap}
    reserve = {rid: set(rmap[rid].get("reserve") or []) for rid in rmap}
    faab_budget = int((league.get("settings") or {}).get("waiver_budget") or 100)
    faab = {rid: faab_budget - int((rmap[rid].get("settings") or {}).get("waiver_budget_used") or 0) for rid in rmap}

    on_alive = {p for rid in alive for p in roster_players[rid]}
    pending = []
    pending_from = None
    if chops:
        leg, rid, drops, created = chops[-1]
        if created > last_waiver_run:
            pending = [p for p in drops if p in P and p not in on_alive]
            pending_from = dict(week=leg, roster_id=rid)

    taken = on_alive | set(pending)
    repl = {}
    for w in range(start, final + 1):
        for pos in SKILL:
            vals = sorted((proj[p].get(w, 0.0) for p in proj if p not in taken and P.get(p, {}).get("pos") == pos), reverse=True)
            k = PARAMS["repl_rank"] - 1
            repl[(pos, w)] = vals[k] if len(vals) > k else 0.0

    # current-week live scoring
    live = None
    wk_status = games.get(start, {})
    started = any(s and s != "pre_game" for s in wk_status.values())
    if started:
        mu = {x["roster_id"]: x for x in (src.matchups(start) or [])}
        live = {}
        for rid in alive:
            m = mu.get(rid) or {}
            base = float(m.get("points") or 0.0)
            rem = []
            for p, sp in zip(m.get("starters") or [], m.get("starters_points") or []):
                if p not in P:
                    continue
                st = wk_status.get(P[p]["team"])
                if st == "pre_game":
                    rem.append((p, 1.0))
                elif st and st != "complete":
                    rem.append((p, 0.5))
            live[rid] = dict(base=base, remaining=rem, starters=list(m.get("starters") or []))

    slots = [s for s in roster_positions if s not in ("BN", "IR", "TAXI")]
    active_slots = len([s for s in roster_positions if s not in ("IR", "TAXI")])
    params = dict(PARAMS, max_active=active_slots)

    return dict(
        season=season, current_week=cur, start=start, final=final, alive=alive,
        league_name=league.get("name"), slots=slots, params=params, sd_fit=sd_fit,
        P=P, proj={k: dict(v) for k, v in proj.items()}, repl=repl, byes=byes,
        roster_players=roster_players, reserve={k: list(v) for k, v in reserve.items()}, faab=faab,
        faab_budget=faab_budget,
        names={rid: _team_name(users.get(rmap[rid]["owner_id"], {})) if rmap[rid].get("owner_id") else f"Team {rid}" for rid in rmap},
        usernames={rid: (users.get(rmap[rid]["owner_id"], {}) or {}).get("display_name") for rid in rmap},
        eliminated=[dict(week=c[0], roster_id=c[1]) for c in chops],
        pending=pending, pending_from=pending_from, live=live, tx=tx,
    )


# ----------------------------------------------------------------------------- lineup math

class Engine:
    def __init__(self, ctx):
        self.c = ctx
        self.P, self.proj, self.repl = ctx["P"], ctx["proj"], ctx["repl"]
        slots = ctx["slots"]
        self.dedicated = collections.Counter(s for s in slots if s in SKILL)
        self.flex = [s for f in FLEX_ORDER for s in slots if s == f]
        self.sd = ctx["sd_fit"]
        self._ros = {}

    def pw(self, p, w):
        return self.proj.get(p, {}).get(w, 0.0)

    def sd_of(self, pos, mu):
        a, b = self.sd[pos]
        return a + b * mu if mu > 0.5 else 0.0

    def lineup(self, players, w, detail=False):
        by = {pos: [] for pos in SKILL}
        for p in players:
            v = self.pw(p, w)
            if v > 0:
                by[self.P[p]["pos"]].append((v, p))
        for pos in SKILL:
            by[pos].append((self.repl.get((pos, w), 0.0), "FA"))
            by[pos].sort(reverse=True)
        starters = []
        left = {pos: list(by[pos]) for pos in SKILL}
        for pos, n in self.dedicated.items():
            take, left[pos] = left[pos][:n], left[pos][n:]
            starters += [(pos, v, p) for v, p in take]
        for f in self.flex:
            best = None
            for pos in FLEX_ELIG[f]:
                if left[pos] and (best is None or left[pos][0][0] > best[1]):
                    best = (pos, left[pos][0][0])
            if best:
                v, p = left[best[0]].pop(0)
                starters.append((best[0], v, p))
        mu = sum(v for _, v, _ in starters)
        var = sum(self.sd_of(pos, v) ** 2 for pos, v, _ in starters)
        return (mu, var, starters) if detail else (mu, var)

    def team_weeks(self, players, w0):
        return {w: self.lineup(players, w) for w in range(w0, self.c["final"] + 1)}

    def ros(self, p, w0):
        k = (p, w0)
        if k not in self._ros:
            self._ros[k] = sum(self.pw(p, w) for w in range(w0, self.c["final"] + 1))
        return self._ros[k]


# ----------------------------------------------------------------------------- simulation

_CTX = None


def _sim_chunk(args):
    nsim, seed = args
    c = _CTX
    E = Engine(c)
    prm = c["params"]
    rng = np.random.default_rng(seed)
    pyr = random.Random(seed)
    start, final = c["start"], c["final"]
    alive0 = c["alive"]
    reserve = {k: set(v) for k, v in c["reserve"].items()}
    base_tw = {rid: E.team_weeks(c["roster_players"][rid], start) for rid in alive0}

    chop = collections.defaultdict(collections.Counter)
    spend = collections.Counter()
    first_auction = collections.defaultdict(lambda: collections.defaultdict(list))  # player -> rid -> [bids]
    first_label = None

    def auction(pool, w, alive, rost, faab, tw, record):
        cand = sorted([p for p in pool if E.ros(p, w) > prm["pool_min_ros"]], key=lambda p: -E.ros(p, w))[:prm["pool_top"]]
        for p in cand:
            bids = []
            for rid in alive:
                if faab[rid] <= 0:
                    continue
                new = E.team_weeks(rost[rid] + [p], w)
                gain = sum(new[t][0] - tw[rid][t][0] for t in range(w, final + 1))
                if gain < prm["min_gain"]:
                    continue
                share = (1 - math.exp(-gain / prm["tau"])) * math.exp(pyr.gauss(0, prm["bid_noise"]))
                bid = max(1, min(faab[rid], int(round(faab[rid] * min(share, 1.0)))))
                bids.append((bid, pyr.random(), rid, new))
            if not bids:
                if record is not None:
                    record[p]["none"].append(0)
                continue
            bid, _, rid, new = max(bids)
            if record is not None:
                record[p][rid].append(bid)
            faab[rid] -= bid
            spend[rid] += bid
            rost[rid].append(p)
            active = [q for q in rost[rid] if q not in reserve.get(rid, set())]
            if prm["max_active"] and len(active) > prm["max_active"]:
                worst = min(active, key=lambda q: E.ros(q, w))
                rost[rid].remove(worst)
                new = E.team_weeks(rost[rid], w)
            tw[rid] = new

    for _ in range(nsim):
        rost = {rid: list(c["roster_players"][rid]) for rid in alive0}
        faab = {rid: c["faab"][rid] for rid in alive0}
        tw = {rid: dict(base_tw[rid]) for rid in alive0}
        alive = list(alive0)
        pool = list(c["pending"])
        recorded = False
        for w in range(start, final + 1):
            if pool:
                auction(pool, w, alive, rost, faab, tw, None if recorded else first_auction)
                if not recorded:
                    recorded = True
            if w == start and c["live"]:
                sc = {}
                for rid in alive:
                    lv = c["live"][rid]
                    x = lv["base"]
                    for p, frac in lv["remaining"]:
                        mu = E.pw(p, w) * frac
                        x += max(0.0, rng.normal(mu, E.sd_of(E.P[p]["pos"], mu) * math.sqrt(frac)))
                    sc[rid] = x
            else:
                sc = {rid: rng.normal(tw[rid][w][0], math.sqrt(tw[rid][w][1])) for rid in alive}
            out = min(sc, key=sc.get)
            chop[out][w] += 1
            pool = rost[out]
            alive.remove(out)
            if not recorded and w == start:
                first_label = w
        chop[alive[0]]["champion"] += 1
    return dict(chop={k: dict(v) for k, v in chop.items()}, spend=dict(spend),
                first_auction={p: {r: b for r, b in d.items()} for p, d in first_auction.items()}, n=nsim)


def simulate(ctx, nsim=10000, seed=1, workers=None):
    global _CTX
    _CTX = ctx
    workers = workers or max(1, mp.cpu_count())
    per = [nsim // workers + (1 if i < nsim % workers else 0) for i in range(workers)]
    jobs = [(n, seed * 1000 + i) for i, n in enumerate(per) if n]
    if workers == 1:
        parts = [_sim_chunk(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_sim_chunk, jobs)
    chop = collections.defaultdict(collections.Counter)
    spend = collections.Counter()
    fa = collections.defaultdict(lambda: collections.defaultdict(list))
    for part in parts:
        for k, v in part["chop"].items():
            chop[k].update(v)
        spend.update(part["spend"])
        for p, d in part["first_auction"].items():
            for r, b in d.items():
                fa[p][r] += b
    return dict(chop=chop, spend=spend, first_auction=fa, n=nsim)


# ----------------------------------------------------------------------------- summary

def summarize(ctx, res, generated_at):
    E = Engine(ctx)
    N = res["n"]
    start, final = ctx["start"], ctx["final"]
    weeks = list(range(start, final + 1))
    P = ctx["P"]

    def pinfo(p, w0):
        info = P[p]
        return dict(id=p, name=info["name"], pos=info["pos"], team=info["team"], injury=info["injury"],
                    bye=[b for b in ctx["byes"].get(info["team"], []) if start <= b <= final],
                    proj={str(w): round(E.pw(p, w), 1) for w in weeks}, ros=round(E.ros(p, w0), 1))

    teams = []
    for rid in ctx["alive"]:
        c = res["chop"].get(rid, {})
        dist = {str(w): c.get(w, 0) / N for w in weeks}
        dist["champion"] = c.get("champion", 0) / N
        exp_weeks = sum(c.get(w, 0) * (w - start) for w in weeks) / N + dist["champion"] * len(weeks)
        tw = E.team_weeks(ctx["roster_players"][rid], start)
        star_wk = start + 1 if (ctx["live"] and start + 1 <= final) else start
        starters_now = E.lineup(ctx["roster_players"][rid], star_wk, detail=True)[2]
        live = (ctx["live"] or {}).get(rid)
        teams.append(dict(
            roster_id=rid, name=ctx["names"][rid], username=ctx["usernames"][rid], faab=ctx["faab"][rid],
            p_survive_week=1 - dist[str(start)],
            p_survive_3=1 - sum(dist.get(str(w), 0) for w in range(start, start + 3)),
            exp_weeks=exp_weeks, p_win=dist["champion"], chop_dist=dist,
            proj_by_week={str(w): round(tw[w][0], 1) for w in weeks},
            sd_week=round(math.sqrt(tw[start][1]), 1),
            exp_faab_spend=res["spend"].get(rid, 0) / N,
            live_points=(live or {}).get("base"),
            live_remaining=[P[p]["name"] for p, _ in (live or {}).get("remaining", [])],
            projected_starters=[p for _, _, p in starters_now if p != "FA"],
            roster=sorted([pinfo(p, start) for p in ctx["roster_players"][rid]], key=lambda x: -x["ros"]),
            reserve=ctx["reserve"].get(rid, []),
        ))
    teams.sort(key=lambda t: (-t["exp_weeks"], -t["p_win"]))
    for i, t in enumerate(teams, 1):
        t["rank"] = i

    pool = []
    for p, d in res["first_auction"].items():
        tot = sum(len(b) for b in d.values())
        winners = sorted(((r, len(b) / tot, float(np.median(b)) if b else 0) for r, b in d.items() if r != "none"),
                         key=lambda x: -x[1])
        pool.append(dict(player=pinfo(p, start), unclaimed=len(d.get("none", [])) / tot,
                         winners=[dict(roster_id=r, name=ctx["names"][r], p=pw_, median_bid=mb) for r, pw_, mb in winners[:4]]))
    pool.sort(key=lambda x: -x["player"]["ros"])

    names = ctx["names"]
    def pname(p):
        return P.get(p, {}).get("name", p)
    recent = []
    for t in sorted(ctx["tx"], key=lambda t: -(t.get("status_updated") or t.get("created") or 0)):
        if t.get("status") != "complete" or t.get("type") == "chopped":
            continue
        recent.append(dict(
            time=t.get("status_updated") or t.get("created"), type=t["type"], week=t.get("leg"),
            teams=[names.get(r) for r in t.get("roster_ids") or []],
            adds=[dict(player=pname(p), to=names.get(r)) for p, r in (t.get("adds") or {}).items()],
            drops=[dict(player=pname(p), frm=names.get(r)) for p, r in (t.get("drops") or {}).items()],
            bid=(t.get("settings") or {}).get("waiver_bid"),
            faab_moves=[dict(amount=b["amount"], frm=names.get(b["sender"]), to=names.get(b["receiver"])) for b in t.get("waiver_budget") or []],
        ))
        if len(recent) >= 60:
            break

    return dict(
        generated_at=generated_at, league_name=ctx["league_name"], season=ctx["season"],
        current_week=ctx["current_week"], start_week=start, final_week=final, n_sims=N,
        week_in_progress=bool(ctx["live"]), params=ctx["params"], sd_fit=ctx["sd_fit"],
        faab_budget=ctx["faab_budget"], teams=teams,
        eliminated=[dict(week=e["week"], roster_id=e["roster_id"], name=names[e["roster_id"]]) for e in ctx["eliminated"]],
        pending_from=(dict(week=ctx["pending_from"]["week"], name=names[ctx["pending_from"]["roster_id"]])
                      if ctx["pending_from"] else None),
        waiver_pool=pool, pool_is_pending=bool(ctx["pending"]), transactions=recent,
    )
