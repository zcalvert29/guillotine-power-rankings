# 🪓 Guillotine League Power Rankings

A Streamlit app with forward-looking power rankings for the Sleeper Guillotine league. Every morning a GitHub Action pulls the league, plays out the rest of the season 10,000 times (weekly chops plus FAAB auctions for each chopped roster) and saves a dated snapshot. The app reads those snapshots, so it shows how rosters and odds change from day to day.

## What's in the app
- **Week picker:** the sidebar defaults to the current week's latest snapshot; pick any earlier week (or a specific day inside it) to see the rankings as they stood then. Weeks before the app existed are rebuilt with rosters and FAAB as that week was actually played (game-day trades count in that week).
- **Power rankings:** chance of surviving the next week, chance of being alive in 3 weeks, expected weeks survived (the ranking stat), win %, FAAB left and a change column, plus a win % trend chart.
- **Chop watch:** this week's and next week's chop odds, live scores during game days, and a danger-by-week heatmap.
- **Team:** projected lineup by week against the league median and low, the roster with weekly projections and byes, and when the team is likely to be chopped.
- **Waivers:** who most likely lands each player off the chopped roster and the median winning bid, plus recent moves (waivers with bids, free agents, trades and FAAB trades).
- **Trends:** one small chart per team, week by week: win %, chance to survive the week, expected weeks, or FAAB left (plus a FAAB-spent-by-week table).
- **Big moves:** every significant waiver, pickup and trade scored by re-running the simulation with that move undone; shows this week's and the season's biggest swings in win % and survival.
- **What changed:** adds, drops, FAAB spent and the change in odds for each team since any earlier snapshot.
- **Track record:** each finished week's pre-kickoff chop odds vs. who actually got chopped, with hit rate and a skill score vs. random guessing. Weeks before the app existed are rebuilt by `pipeline/backfill.py` (that week's rosters + that week's projections, no look-ahead); the daily job re-grades finished weeks automatically.

## Setup (about 10 minutes, one time)
1. **Create a GitHub repo** (private is fine) and upload this folder's contents, including `.github/` and `snapshots/`.
2. **Allow the Action to push:** go to the repo's Settings → Actions → General → Workflow permissions and choose **Read and write**.
3. **Test it:** go to Actions → *Daily refresh* → **Run workflow**. A new `snapshots/YYYY-MM-DD.json` should appear in about 3 minutes.
4. **Deploy:** at [share.streamlit.io](https://share.streamlit.io), create an app from the repo with main file `app.py`. Streamlit Cloud redeploys automatically whenever the Action commits a new snapshot.

Optional (repo Settings → Secrets and variables → Actions → Variables):
- `LEAGUE_ID`: a different league; the default is this one.
- `N_SIMS`: number of simulated seasons; the default is 10000.

App setting: the `MY_USERNAME` environment variable sets which team is highlighted (default `zcalvert29`).

## Schedule
The Action runs daily at about 5:17 AM Central, which catches overnight waiver processing, free-agent moves, and projection and injury updates. To refresh more often (for example every couple of hours on Sundays), add another line under `schedule:` in `.github/workflows/refresh.yml`, like `- cron: "0 18-23/2 * * 0"`. A same-day run overwrites that day's snapshot.

## Run locally
```bash
pip install -r requirements.txt
python -m pipeline.run            # live pull + simulation -> snapshots/
streamlit run app.py
```
`python -m pipeline.run --dir <folder>` runs on a saved pull (the folder layout from `pull_sleeper_guillotine.py`).

## Model notes
- **Lineups:** each team starts its best projected lineup from Sleeper's weekly PPR projections. Those projections already account for byes and injuries. Each team can add one free-agent streamer per position at replacement level.
- **Weekly variance:** fitted by position from this season's actual scores.
- **Live weeks:** during a week in progress, completed games use actual points and only unplayed starters are simulated.
- **Waivers:** a blind-bid auction. Each team's bid grows with the rest-of-season points the player adds to its lineup and with its remaining FAAB, using `tau` in `pipeline/model.py`. That setting is roughly calibrated to this league's past bids.
- **Not modeled:** trades, suboptimal lineups and correlation between players.
- **Day-to-day stability:** every run uses the same random seed, so changes mostly reflect real moves, not simulation noise.
