"""Guillotine League Power Rankings - Streamlit app. Reads daily snapshots written by pipeline/run.py."""
import datetime as dt
import glob
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.environ.get("SNAP_DIR", os.path.join(ROOT, "snapshots"))
MY_USERNAME = os.environ.get("MY_USERNAME", "zcalvert29")

BLUE = "#2a78d6"     # highlighted team
GRAY = "#b9b8b2"     # other teams
INK2 = "#52514e"

st.set_page_config(page_title="Guillotine Power Rankings", page_icon="🪓", layout="wide")


# ----------------------------------------------------------------------------- data

@st.cache_data(ttl=900)
def load_snapshots(sig):
    snaps = {}
    for f in sorted(glob.glob(os.path.join(SNAP_DIR, "*.json"))):
        with open(f) as fh:
            s = json.load(fh)
        snaps[s.get("snapshot_date") or os.path.basename(f)[:-5]] = s
    return snaps


def snap_signature():
    return tuple((f, os.path.getmtime(f)) for f in sorted(glob.glob(os.path.join(SNAP_DIR, "*.json"))))


snaps = load_snapshots(snap_signature())
if not snaps:
    st.title("Guillotine Power Rankings")
    st.info("No snapshots yet. Run `python -m pipeline.run` (or the GitHub Action) to create the first one.")
    st.stop()

dates = sorted(snaps)
week_dates = {}
for d in dates:
    week_dates.setdefault(snaps[d]["start_week"], []).append(d)
week_list = sorted(week_dates, reverse=True)
current_week = snaps[dates[-1]]["start_week"]

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("View")
    view_week = st.selectbox(
        "Week", week_list, index=0,
        format_func=lambda w: f"Week {w}" + (" (current)" if w == current_week else ""),
        help="Rankings as they stood during any week. Defaults to the latest.")
    wk_days = week_dates[view_week][::-1]
    with st.expander("As of a specific day", expanded=False):
        view_date = st.selectbox("Day", wk_days, index=0,
                                 format_func=lambda d: dt.date.fromisoformat(d).strftime("%a %b %d")
                                 + (" · rebuilt, pre-kickoff" if snaps[d].get("rebuilt") else "")
                                 + (" (latest)" if d == wk_days[0] else ""))
    S = snaps[view_date]
    earlier = [d for d in dates if d < view_date]
    base_opts = {}
    prev_weeks = [w for w in week_list if w < view_week]
    if prev_weeks:
        _pd = week_dates[prev_weeks[0]][-1]
        _lab = ("before kickoff" if snaps[_pd].get("rebuilt")
                else ("latest, " + dt.date.fromisoformat(_pd).strftime("%b %d")))
        base_opts[f"Week {prev_weeks[0]} ({_lab})"] = _pd
    if earlier:
        base_opts["Previous day"] = earlier[-1]
    week_start = week_dates[view_week][0]
    if week_start < view_date:
        base_opts[f"Start of Week {view_week}"] = week_start
    base_label = st.radio("Compare against", list(base_opts) or ["(nothing earlier yet)"], index=0)
    B = snaps.get(base_opts.get(base_label)) if base_opts else None
    team_names = [t["name"] for t in S["teams"]]
    mine = next((t["name"] for t in S["teams"] if t.get("username") == MY_USERNAME), team_names[0])
    focus = st.selectbox("Highlight team", team_names, index=team_names.index(mine))
    if S.get("rebuilt"):
        st.caption(f"Rebuilt as of just before Week {S['start_week']} kickoff, from {S['n_sims']:,} simulated seasons.")
    else:
        st.caption(f"Snapshot built {S['generated_at'].replace('T', ' ')} CT from {S['n_sims']:,} simulated seasons.")

teams = {t["name"]: t for t in S["teams"]}
bteams = {t["roster_id"]: t for t in (B or {}).get("teams", [])}
start, final = S["start_week"], S["final_week"]
weeks = [str(w) for w in range(start, final + 1)]


def alive_after(t, w):
    """Chance team t is still alive after week w (weeks already played count as survived)."""
    return 1 - sum(v for k, v in t["chop_dist"].items() if k != "champion" and int(k) <= w)


def as_of(snap):
    return snap.get("as_of_ms") or int(dt.datetime.fromisoformat(snap["generated_at"]).timestamp() * 1000)


def snap_label(snap):
    if snap.get("rebuilt"):
        return f"before Week {snap['start_week']} kickoff"
    return dt.date.fromisoformat(snap["snapshot_date"]).strftime("%a %b %d")


NOLAP = dict(labelOverlap=False, labelLimit=240)

# if this week's chop is already decided, headline next week instead
live_decided = S["week_in_progress"] and max(t["chop_dist"][str(start)] for t in S["teams"]) > 0.95
fw = start + 1 if live_decided and start + 1 <= final else start
proj_weeks = [w for w in weeks if not (S["week_in_progress"] and int(w) == start)]


def delta(cur, rid, key, scale=100.0):
    b = bteams.get(rid)
    if b is None:
        return None
    return (cur - b[key]) * scale


def fmt_delta(x, unit="", digits=1):
    if x is None:
        return ""
    if abs(x) < 0.05:
        return "–"
    return f"{'▲' if x > 0 else '▼'} {abs(x):.{digits}f}{unit}"


# ----------------------------------------------------------------------------- header
st.title(f"🪓 {S['league_name'].strip()}")
status = (f"Week {start} in progress" if S["week_in_progress"] else f"Week {start} up next")
st.markdown(f"**{status}** · {len(S['teams'])} teams alive · season ends Week {final}"
            + (f" · Δ vs {base_label}" if B else ""))
latest_elim = {e["week"]: e for e in snaps[dates[-1]]["eliminated"]}
if S.get("rebuilt"):
    st.caption(f"🔁 Rebuilt view: the league as it stood just before Week {start} kicked off (rosters, FAAB and that week's projections). "
               "Odds for later weeks use Sleeper's projections as they stand now, so long-range numbers carry some hindsight; "
               f"the Week {start} chop odds don't.")
if view_week != current_week and view_week in latest_elim:
    e = latest_elim[view_week]
    p_out = next((t["chop_dist"].get(str(view_week), 0) for t in S["teams"]
                  if t["roster_id"] == e.get("roster_id") or t["name"] == e["name"]), None)
    st.warning(f"Looking back at Week {view_week} as of {dt.date.fromisoformat(view_date).strftime('%b %d')}. "
               f"**{e['name']}** was chopped that week" + (f" (this view gave them {p_out:.0%})." if p_out is not None else ".")
               + " See **Track record** for every week.")
elif S.get("pool_is_pending") and S.get("pending_from"):
    st.info(f"{S['pending_from']['name']}'s roster (chopped Week {S['pending_from']['week']}) is on waivers. "
            f"See **Waivers** for who's likely to land each player.")

tab_rank, tab_chop, tab_team, tab_waiv, tab_chg, tab_track, tab_about = st.tabs(
    ["Power rankings", "Chop watch", "Team", "Waivers", "What changed", "Track record", "Method"])

# ----------------------------------------------------------------------------- rankings
with tab_rank:
    rows = []
    for t in S["teams"]:
        b = bteams.get(t["roster_id"])
        d_wk = None if b is None else (alive_after(t, fw) - alive_after(b, fw)) * 100
        d_win = None if b is None else (t["p_win"] - b["p_win"]) * 100
        rows.append({
            "Rank": t["rank"],
            "Move": ("" if b is None else ("–" if b["rank"] == t["rank"] else
                     (f"▲{b['rank'] - t['rank']}" if b["rank"] > t["rank"] else f"▼{t['rank'] - b['rank']}"))),
            "Team": t["name"] + ("  ⭐" if t["name"] == focus else ""),
            "surv": alive_after(t, fw) * 100,
            "Δ surv": fmt_delta(d_wk),
            "alive3": alive_after(t, fw + 2) * 100,
            "Exp. weeks": t["exp_weeks"],
            "Win": t["p_win"] * 100,
            "Δ win": fmt_delta(d_win),
            "FAAB": t["faab"],
            "Proj/wk": sum(t["proj_by_week"][w] for w in proj_weeks) / max(1, len(proj_weeks)),
        })
    df = pd.DataFrame(rows)
    pct = lambda label, help_=None: st.column_config.ProgressColumn(label, format="%.0f%%", min_value=0, max_value=100, help=help_)
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("#", width="small"),
            "Move": st.column_config.TextColumn("", width="small"),
            "surv": pct(f"Survive Wk {fw}", f"Chance of still being alive after Week {fw}"),
            "Δ surv": st.column_config.TextColumn("Δ", width="small"),
            "alive3": pct(f"Alive after Wk {fw + 2}", f"Chance of still being alive after Week {fw + 2}"),
            "Win": pct("Win", "Chance of being the last team standing"),
            "Δ win": st.column_config.TextColumn("Δ", width="small"),
            "Exp. weeks": st.column_config.NumberColumn("Exp. wks", format="%.2f",
                                                        help=f"Expected weeks survived out of {len(weeks)} left - the ranking stat"),
            "FAAB": st.column_config.NumberColumn("FAAB", format="$%d"),
            "Proj/wk": st.column_config.NumberColumn("Proj/wk", format="%.1f", help="Best projected lineup, avg over remaining weeks"),
        })
    st.caption("Ranked by expected weeks survived. Δ = percentage-point change vs the comparison snapshot."
               + (f" Week {start}'s chop is all but decided, so survival columns look ahead to Week {fw}." if fw != start else ""))

    shown = [d for d in dates if d <= view_date]
    if len(shown) > 1:
        hist = []
        for d in shown:
            for t in snaps[d]["teams"]:
                hist.append(dict(date=pd.Timestamp(d), team=t["name"], win=t["p_win"], exp=t["exp_weeks"], alive=True))
        hdf = pd.DataFrame(hist)
        metric = st.segmented_control("Trend", ["Win %", "Expected weeks"], default="Win %")
        col = "win" if metric == "Win %" else "exp"
        hdf["is_focus"] = hdf["team"] == focus
        base = alt.Chart(hdf).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", grid=False)),
            y=alt.Y(f"{col}:Q", title=None, axis=alt.Axis(format=".0%" if col == "win" else ".1f")),
            detail="team:N",
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("date:T", format="%b %d"),
                     alt.Tooltip(f"{col}:Q", format=".1%" if col == "win" else ".2f", title=metric)])
        others = base.transform_filter("!datum.is_focus").mark_line(strokeWidth=1.5, color=GRAY, point=alt.OverlayMarkDef(size=30, color=GRAY))
        me = base.transform_filter("datum.is_focus").mark_line(strokeWidth=2.5, color=BLUE, point=alt.OverlayMarkDef(size=70, color=BLUE))
        st.markdown(f"**{metric} over time** — <span style='color:{BLUE}'>■</span> {focus} vs. the rest of the league",
                    unsafe_allow_html=True)
        st.altair_chart((others + me).properties(height=280), width="stretch")

# ----------------------------------------------------------------------------- chop watch
with tab_chop:
    cw_opts = list(range(start, final + 1))
    cw = st.segmented_control("Week", cw_opts, default=fw, format_func=lambda w: f"Wk {w}") or fw
    st.subheader(f"Who gets chopped in Week {cw}?")
    in_play = [t for t in S["teams"] if alive_after(t, cw - 1) > 0.005]   # drop teams already (all but) chopped
    if cw > start:
        st.caption(f"Chance each team is the one chopped in Week {cw}, from today's view. It already accounts for the "
                   f"chance a team is gone before then. Hover a bar for the odds if they make it to Week {cw}.")
    cdf = pd.DataFrame([dict(team=t["name"], p=t["chop_dist"][str(cw)], focus=t["name"] == focus,
                             cond=t["chop_dist"][str(cw)] / max(alive_after(t, cw - 1), 1e-9),
                             live=t.get("live_points"), left=", ".join(t.get("live_remaining") or []))
                        for t in in_play]).sort_values("p", ascending=False)
    bars = alt.Chart(cdf).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("team:N", sort="-x", title=None, axis=alt.Axis(**NOLAP)),
        x=alt.X("p:Q", title=f"Chance of being chopped in Week {cw}", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, max(0.05, cdf["p"].max())])),
        color=alt.condition("datum.focus", alt.value(BLUE), alt.value(GRAY)),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("p:Q", format=".1%", title="Chop chance"),
                 alt.Tooltip("cond:Q", format=".1%", title=f"If alive entering Wk {cw}")])
    labels = bars.mark_text(align="left", dx=4, color=INK2).encode(text=alt.Text("p:Q", format=".0%"), color=alt.value(INK2))
    st.altair_chart((bars + labels).properties(height=34 * len(cdf)), width="stretch")
    if S["week_in_progress"] and cw == start:
        st.markdown("**Live scores this week**")
        st.dataframe(cdf.sort_values("live")[["team", "live", "left"]].rename(
            columns={"team": "Team", "live": "Points so far", "left": "Starters still to play"}),
            hide_index=True, width="stretch")

    st.subheader("Danger by week")
    st.caption("Chance each team is the one chopped in each week (unconditional). Dark = danger week.")
    hm = []
    for t in S["teams"]:
        for w in weeks:
            if alive_after(t, int(w) - 1) > 0.005:   # no cells after a team is (all but) chopped
                hm.append(dict(team=t["name"], week=f"Wk {w}", wnum=int(w), p=t["chop_dist"][w], proj=t["proj_by_week"][w]))
    hm = pd.DataFrame(hm)
    order = [t["name"] for t in S["teams"]]
    heat = alt.Chart(hm).mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        x=alt.X("week:N", sort=alt.SortField("wnum"), title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("team:N", sort=order, title=None, axis=alt.Axis(**NOLAP)),
        color=alt.Color("p:Q", title="Chop chance", scale=alt.Scale(range=["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95"]),
                        legend=alt.Legend(format=".0%", orient="bottom")),
        tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("week:N", title="Week"),
                 alt.Tooltip("p:Q", format=".1%", title="Chop chance"), alt.Tooltip("proj:Q", format=".1f", title="Projected pts")])
    txt = alt.Chart(hm).mark_text(fontSize=11).encode(
        x=alt.X("week:N", sort=alt.SortField("wnum")), y=alt.Y("team:N", sort=order),
        text=alt.Text("p:Q", format=".0%"),
        color=alt.condition(f"datum.p > {0.55 * float(hm['p'].max()):.4f}", alt.value("white"), alt.value(INK2)))
    st.altair_chart((heat + txt).properties(height=32 * len(order)), width="stretch")

# ----------------------------------------------------------------------------- team
with tab_team:
    tname = st.selectbox("Team", team_names, index=team_names.index(focus), key="team_pick")
    T = teams[tname]
    bT = bteams.get(T["roster_id"])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rank", f"#{T['rank']}", delta=(bT["rank"] - T["rank"]) if bT else None)
    c2.metric(f"Survive Wk {fw}", f"{alive_after(T, fw):.0%}",
              delta=f"{(alive_after(T, fw) - alive_after(bT, fw)) * 100:+.1f} pts" if bT else None)
    c3.metric("Exp. weeks", f"{T['exp_weeks']:.2f}",
              delta=f"{delta(T['exp_weeks'], T['roster_id'], 'exp_weeks', 1):+.2f}" if bT else None)
    c4.metric("Win", f"{T['p_win']:.1%}",
              delta=f"{delta(T['p_win'], T['roster_id'], 'p_win'):+.1f} pts" if bT else None)
    c5.metric("FAAB", f"${T['faab']}", delta=(T["faab"] - bT["faab"]) if bT else None)

    # projected lineup vs league median by week
    lw = []
    for w in proj_weeks:
        vals = sorted(t["proj_by_week"][w] for t in S["teams"])
        lw.append(dict(week=int(w), series=tname, pts=T["proj_by_week"][w]))
        lw.append(dict(week=int(w), series="League median", pts=vals[len(vals) // 2]))
        lw.append(dict(week=int(w), series="League low", pts=vals[0]))
    lw = pd.DataFrame(lw)
    st.markdown(f"**Projected lineup by week** — dips are bye/injury holes to plug")
    line = alt.Chart(lw).mark_line(point=alt.OverlayMarkDef(size=60)).encode(
        x=alt.X("week:O", title="Week", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pts:Q", title="Projected points", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=[tname, "League median", "League low"], range=[BLUE, GRAY, "#e34948"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=[tname, "League median", "League low"],
                                                             range=[[1, 0], [4, 3], [2, 2]]), legend=None),
        strokeWidth=alt.condition(f"datum.series == '{tname}'", alt.value(2.5), alt.value(1.5)),
        tooltip=[alt.Tooltip("series:N", title=""), alt.Tooltip("week:O", title="Week"), alt.Tooltip("pts:Q", format=".1f", title="Points")])
    st.altair_chart(line.properties(height=260), width="stretch")

    st.markdown(f"**Roster** — weekly projections (BYE = bye week, ★ = projected starter in Week {proj_weeks[0]})")
    rr = []
    starters = set(T["projected_starters"])
    for p in T["roster"]:
        row = {"": "★" if p["id"] in starters else "", "Player": p["name"], "Pos": p["pos"], "NFL": p["team"] or "",
               "Status": ("IR slot" if p["id"] in T["reserve"] else (p["injury"] or "")), "ROS": p["ros"]}
        for w in proj_weeks:
            row[f"W{w}"] = "BYE" if int(w) in p["bye"] else (f"{p['proj'][w]:.1f}" if p["proj"][w] else "–")
        rr.append(row)
    st.dataframe(pd.DataFrame(rr), hide_index=True, width="stretch",
                 column_config={"ROS": st.column_config.NumberColumn("ROS pts", format="%.0f")})

    st.markdown("**When does this team get chopped?**")
    cd = pd.DataFrame([dict(week=("Wins it all" if k == "champion" else f"Wk {k}"), order=(99 if k == "champion" else int(k)), p=v)
                       for k, v in T["chop_dist"].items()])
    chop_bar = alt.Chart(cd).mark_bar(cornerRadiusEnd=4, color=BLUE, size=26).encode(
        x=alt.X("week:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("p:Q", title=None, axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("week:N", title="Outcome"), alt.Tooltip("p:Q", format=".1%", title="Chance")])
    st.altair_chart(chop_bar.properties(height=220), width="stretch")
    st.caption(f"Expected FAAB spend over the rest of the season in the simulations: ${T['exp_faab_spend']:.0f}.")

# ----------------------------------------------------------------------------- waivers
with tab_waiv:
    if S.get("pool_is_pending"):
        st.subheader(f"On waivers now: {S['pending_from']['name']}'s roster")
        st.caption("Simulated blind-bid auction. Bids scale with how many rest-of-season points a player adds to each "
                   "team's best lineup and with that team's remaining FAAB.")
    else:
        st.subheader(f"Waiver outlook after the Week {start} chop")
        st.caption("Whoever gets chopped this week dumps their roster. For each player, if his team is chopped: "
                   "who most likely lands him and for how much.")
    wrows = []
    for x in S["waiver_pool"]:
        p = x["player"]
        w = x["winners"]
        row = {"Player": p["name"], "Pos": p["pos"], "NFL": p["team"] or "", "ROS pts": p["ros"]}
        for i in range(2):
            if i < len(w):
                row[f"#{i + 1} landing spot"] = f"{w[i]['name']} ({w[i]['p']:.0%})"
                row[f"#{i + 1} median bid"] = f"${w[i]['median_bid']:.0f}"
            else:
                row[f"#{i + 1} landing spot"], row[f"#{i + 1} median bid"] = "", ""
        row["Goes unclaimed"] = f"{x['unclaimed']:.0%}"
        wrows.append(row)
    if wrows:
        st.dataframe(pd.DataFrame(wrows), hide_index=True, width="stretch",
                     column_config={"ROS pts": st.column_config.NumberColumn(format="%.0f")})
    else:
        st.write("No meaningful players in the pool.")

    st.subheader("Recent moves")
    since = None
    if B:
        since = as_of(B)
    tx = S.get("transactions", [])
    show_all = st.toggle("Show all recent moves", value=since is None)
    lines = []
    for t in tx:
        if not show_all and since and (t["time"] or 0) <= since:
            continue
        when = dt.datetime.fromtimestamp((t["time"] or 0) / 1000).strftime("%a %b %d")
        adds = ", ".join(f"+{a['player']}" for a in t["adds"])
        drops = ", ".join(f"−{d['player']}" for d in t["drops"])
        who = " ↔ ".join(x for x in t["teams"] if x)
        kind = {"waiver": "Waiver", "free_agent": "Free agent", "trade": "Trade"}.get(t["type"], t["type"])
        bid = f" · ${t['bid']}" if t["type"] == "waiver" and t.get("bid") is not None else ""
        faab = "; ".join(f"${m['amount']} FAAB {m['frm']} → {m['to']}" for m in t["faab_moves"])
        lines.append(dict(When=when, Type=kind, Team=who, Moves=" ".join(x for x in [adds, drops] if x) + bid + (f" · {faab}" if faab else "")))
    if lines:
        st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch")
    else:
        st.write("No moves since the comparison snapshot.")

# ----------------------------------------------------------------------------- what changed
with tab_chg:
    if not B:
        st.info("Changes appear once there is an earlier snapshot to compare against.")
    else:
        t0, t1 = as_of(B), as_of(S)
        pre_view = lambda x: not x.get("week_in_progress") and (x.get("rebuilt") or x.get("as_of_ms"))
        during = (f" — i.e. moves made during Week {B['start_week']}"
                  if S["start_week"] == B["start_week"] + 1 and pre_view(B) and S.get("rebuilt") else "")
        st.subheader(f"Between {snap_label(B)} and {snap_label(S)}")
        if during:
            st.caption(during.strip(" —").capitalize() + ", from one pre-kickoff view to the next.")
        gone = [t["name"] for t in B["teams"] if t["roster_id"] not in {x["roster_id"] for x in S["teams"]}]
        if gone:
            st.markdown("🪓 **Chopped:** " + ", ".join(gone))
        id_by_name = {t["name"]: t["roster_id"] for t in B["teams"] + S["teams"]}
        period_tx = [x for x in S.get("transactions", []) if t0 < (x["time"] or 0) <= t1]
        waiver_spend, trade_net, trade_notes = {}, {}, {}
        for x in period_tx:
            rids = x.get("rids") or [id_by_name.get(n) for n in x["teams"]]
            if x["type"] == "waiver" and x.get("bid"):
                for r in rids:
                    waiver_spend[r] = waiver_spend.get(r, 0) + x["bid"]
            for m in x.get("faab_moves") or []:
                fr, to = m.get("frm_id") or id_by_name.get(m["frm"]), m.get("to_id") or id_by_name.get(m["to"])
                trade_net[fr] = trade_net.get(fr, 0) - m["amount"]
                trade_net[to] = trade_net.get(to, 0) + m["amount"]
            if x["type"] == "trade":
                for r, nm in zip(rids, x["teams"]):
                    got = [a_["player"] for a_ in x["adds"] if a_["to"] == nm]
                    sent = [d["player"] for d in x["drops"] if d["frm"] == nm]
                    money = sum((m["amount"] if m["to"] == nm else -m["amount"]) for m in x.get("faab_moves") or []
                                if nm in (m["to"], m["frm"]))
                    note = "got " + (", ".join(got) or "nothing") + "; sent " + (", ".join(sent) or "nothing")
                    if money:
                        note += f" ({'+' if money > 0 else '−'}${abs(money)} FAAB)"
                    trade_notes.setdefault(r, []).append(note)
        money = lambda v: "" if not v else f"{'+' if v > 0 else '−'}${abs(v)}"
        crow = []
        for t in S["teams"]:
            b = bteams.get(t["roster_id"])
            if not b:
                continue
            rid = t["roster_id"]
            now_ids = {p["id"]: p["name"] for p in t["roster"]}
            then_ids = {p["id"]: p["name"] for p in b["roster"]}
            same = [w for w in weeks if w in b["proj_by_week"]]
            crow.append({
                "Team": t["name"],
                "Δ win": (t["p_win"] - b["p_win"]) * 100,
                "Δ exp. weeks": t["exp_weeks"] - b["exp_weeks"],
                "Δ proj/wk": (sum(t["proj_by_week"][w] - b["proj_by_week"][w] for w in same) / len(same)) if same else None,
                "FAAB now": t["faab"],
                "FAAB change": money(t["faab"] - b["faab"]),
                "Waiver bids won": f"${waiver_spend[rid]}" if waiver_spend.get(rid) else "",
                "Trade FAAB": money(trade_net.get(rid, 0)),
                "Added": ", ".join(now_ids[i] for i in now_ids if i not in then_ids),
                "Dropped": ", ".join(then_ids[i] for i in then_ids if i not in now_ids),
                "Trades": " | ".join(trade_notes.get(rid, [])),
            })
        cdf2 = pd.DataFrame(crow).sort_values("Δ win", ascending=False)
        st.dataframe(cdf2, hide_index=True, width="stretch", column_config={
            "Δ win": st.column_config.NumberColumn("Δ win (pts)", format="%+.1f"),
            "Δ exp. weeks": st.column_config.NumberColumn(format="%+.2f"),
            "Δ proj/wk": st.column_config.NumberColumn("Δ proj/wk", format="%+.1f", help="Change in best projected lineup, same weeks compared"),
            "FAAB now": st.column_config.NumberColumn(format="$%d"),
            "FAAB change": st.column_config.TextColumn(help="Net change: waiver bids won plus FAAB sent/received in trades"),
            "Trade FAAB": st.column_config.TextColumn(help="FAAB received (+) or sent (−) in trades"),
        })
        st.caption("Probability changes come from roster moves, projection/injury updates and results. Every run reuses the same "
                   "random seed, so simulation noise mostly cancels out between days.")

# ----------------------------------------------------------------------------- track record
with tab_track:
    st.subheader("How did the forecasts do?")
    st.caption("For each finished week: the model's chop odds before kickoff vs. who actually got chopped. "
               "**Live** = the first snapshot of that week taken before kickoff. **Rebuilt** = weeks before the app existed, "
               "reconstructed from that week's rosters, Sleeper's projections for that week, and score spread fitted only on earlier weeks, so no look-ahead. "
               "Weeks marked * only had mid-week snapshots, so part of the result was already known.")
    bf_path = os.path.join(os.path.dirname(SNAP_DIR.rstrip("/")), "history", "backfill.json")
    rebuilt = {}
    if os.path.exists(bf_path):
        with open(bf_path) as fh:
            rebuilt = {x["week"]: x for x in json.load(fh)["weeks"]}

    def grade(w, probs, nm, out, label, source, forecast):
        p = probs[out]
        n = len(probs)
        fav = max(probs, key=probs.get)
        return dict(week=w, label=label, clean=not label.endswith("*"), source=source, forecast=forecast,
                    chopped=nm[out], p=p, rank=1 + sum(1 for v in probs.values() if v > p), n=n,
                    fav=nm[fav], fav_p=probs[fav], base=1 / n,
                    brier=sum((v - (1.0 if r == out else 0.0)) ** 2 for r, v in probs.items()),
                    brier_base=(1 - 1 / n) ** 2 + (n - 1) * (1 / n) ** 2)

    recs = []
    for w in sorted(latest_elim):
        e = latest_elim[w]
        wd = week_dates.get(w, [])
        pre_live = [d for d in wd if not snaps[d]["week_in_progress"] and not snaps[d].get("rebuilt")]
        pre = pre_live or [d for d in wd if not snaps[d]["week_in_progress"]]
        rec = None
        if pre or (wd and w not in rebuilt):
            d0 = pre[0] if pre else wd[0]
            F = snaps[d0]
            probs = {t["roster_id"]: t["chop_dist"].get(str(w), 0.0) for t in F["teams"]}
            nm = {t["roster_id"]: t["name"] for t in F["teams"]}
            out = e.get("roster_id") if e.get("roster_id") in probs else next((r for r, n in nm.items() if n == e["name"]), None)
            if out is not None:
                rec = grade(w, probs, nm, out, f"Wk {w}" + ("" if pre else "*"),
                            ("Rebuilt" if snaps[d0].get("rebuilt") else "Live") if pre else "Live (mid-week)",
                            dt.date.fromisoformat(d0).strftime("%b %d"))
        if rec is None and w in rebuilt:
            R = rebuilt[w]
            probs = {t["roster_id"]: t["p_chop"] for t in R["teams"]}
            nm = {t["roster_id"]: t["name"] for t in R["teams"]}
            if R["chopped_roster_id"] in probs:
                rec = grade(w, probs, nm, R["chopped_roster_id"], f"Wk {w}", "Rebuilt", "pre-kickoff")
        if rec:
            recs.append(rec)
    if not recs:
        st.info("Nothing to grade yet. Each week's forecast is scored once that week's chop happens, so this fills in as the season goes. "
                "The first fully clean grade is the first week that has a snapshot from before kickoff.")
    else:
        rd = pd.DataFrame(recs)
        clean = rd[rd["clean"]]
        g = clean if len(clean) else rd
        m1, m2, m3 = st.columns(3)
        m1.metric("Chopped team was in the model's top 3", f"{int((g['rank'] <= 3).sum())} of {len(g)}",
                  help="Weeks where the team that got chopped was one of the model's three most likely picks.")
        m1.caption(f"Random guess: {(3 / g['n']).clip(upper=1).sum():.1f} of {len(g)}")
        m2.metric("Avg. chop odds given to the chopped team", f"{g['p'].mean():.0%}",
                  help="Higher is better. Compare with picking at random (next to it).")
        m2.caption(f"Random guess: {g['base'].mean():.0%}")
        skill = 1 - g["brier"].sum() / g["brier_base"].sum()
        m3.metric("Skill vs. random guessing", f"{skill:+.0%}",
                  help="Brier skill score: 0% = no better than giving every team equal odds; 100% = perfect.")
        if not len(clean):
            st.caption("All graded weeks so far only have mid-week snapshots (*), so these numbers are optimistic.")
        bars = alt.Chart(rd).mark_bar(cornerRadiusEnd=4, size=28, color=BLUE).encode(
            x=alt.X("label:N", sort=alt.SortField("week"), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("p:Q", title="Odds given to chopped team", axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, max(0.4, float(rd["p"].max()))])),
            tooltip=[alt.Tooltip("label:N", title="Week"), alt.Tooltip("chopped:N", title="Chopped"),
                     alt.Tooltip("p:Q", format=".1%", title="Model odds"), alt.Tooltip("rank:Q", title="Model rank"),
                     alt.Tooltip("fav:N", title="Model's top pick"), alt.Tooltip("source:N", title="Source")])
        base_tick = alt.Chart(rd).mark_tick(color=INK2, thickness=2, size=34).encode(
            x=alt.X("label:N", sort=alt.SortField("week")), y="base:Q",
            tooltip=[alt.Tooltip("base:Q", format=".1%", title="Random-guess odds")])
        st.altair_chart((bars + base_tick).properties(height=260), width="stretch")
        st.caption("Bars: what the model gave the team that went home. Gray tick: what a random guess would have given.")
        st.dataframe(rd.sort_values("week", ascending=False).assign(
            p=lambda x: (x["p"] * 100).round(0).astype(int).astype(str) + "%",
            fav_p=lambda x: (x["fav_p"] * 100).round(0).astype(int).astype(str) + "%",
            rank=lambda x: x["rank"].astype(str) + " of " + x["n"].astype(str))[
            ["label", "chopped", "p", "rank", "fav", "fav_p", "source"]].rename(columns={
                "label": "Week", "chopped": "Chopped", "p": "Model odds", "rank": "Model rank",
                "fav": "Model's top pick", "fav_p": "Top pick odds", "source": "Source"}),
            hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- method
with tab_about:
    p = S["params"]
    st.markdown(f"""
**How the rankings work**

Each day the pipeline pulls the league from Sleeper and plays out the rest of the season {S['n_sims']:,} times:

1. **Waivers.** The most recently chopped roster goes up for a blind FAAB auction. Every surviving team values each player by
   how many rest-of-season points he adds to its *best projected lineup*, and bids
   `FAAB × (1 − e^(−gain / {p['tau']:.0f}))` with random noise. Highest bid wins; the winner drops its least valuable player.
2. **Scores.** Each team starts its best projected lineup (one free-agent streamer per position allowed, at replacement level).
   Player projections are Sleeper's weekly PPR projections, which also handle byes and injuries. Weekly spread comes from this
   season's actual scoring: SD ≈ a + b × projection, fitted by position.
3. **Chop.** Lowest score is eliminated; their roster becomes the next auction pool. During a live week, completed games use
   actual points and only unplayed starters are simulated.

**Ranking stat:** expected weeks survived from here (the champion survives all {len(weeks)}). Win % is the chance of being last standing.

**Not modeled:** trades (including FAAB trades), managers starting sub-optimal lineups, and correlations between players.
""")
