"""
Daily refresh: pull league from Sleeper, simulate the rest of the season, write snapshots/<date>.json.

    python -m pipeline.run                    # live pull (league ID from LEAGUE_ID env or default)
    python -m pipeline.run --dir saved_pull   # use a saved pull instead (testing)
"""
import argparse
import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

from pipeline.model import build_context, simulate, summarize
from pipeline.sleeper import DirSource, LiveSource

DEFAULT_LEAGUE = "1389750406673883136"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default=os.environ.get("LEAGUE_ID", DEFAULT_LEAGUE))
    ap.add_argument("--dir", help="read a saved pull instead of the live API")
    ap.add_argument("--sims", type=int, default=int(os.environ.get("N_SIMS", 10000)))
    ap.add_argument("--seed", type=int, default=2026,
                    help="fixed by default so day-to-day changes reflect real moves, not simulation noise")
    ap.add_argument("--out", default=os.path.join(ROOT, "snapshots"))
    ap.add_argument("--date", help="snapshot date label (YYYY-MM-DD); default = today in America/Chicago")
    a = ap.parse_args()

    t0 = time.time()
    src = DirSource(a.dir) if a.dir else LiveSource(a.league)
    ctx = build_context(src)
    if len(ctx["alive"]) <= 1:
        print("Season is over - nothing to simulate.")
        return
    now = dt.datetime.now(ZoneInfo("America/Chicago"))
    label = a.date or now.date().isoformat()
    seed = a.seed
    print(f"{ctx['league_name']}: {len(ctx['alive'])} alive, simulating weeks {ctx['start']}-{ctx['final']} "
          f"({'live week' if ctx['live'] else 'pre-week'}; pending waiver pool: {len(ctx['pending'])} players)")
    res = simulate(ctx, nsim=a.sims, seed=seed)
    snap = summarize(ctx, res, generated_at=now.isoformat(timespec="minutes"))
    snap["snapshot_date"] = label
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{label}.json")
    with open(path, "w") as fh:
        json.dump(snap, fh, separators=(",", ":"))
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    for t in snap["teams"]:
        print(f"{t['rank']:>2} {t['name'][:22]:<22} survive wk {snap['start_week']}: {t['p_survive_week']:6.1%}  "
              f"exp wks {t['exp_weeks']:.2f}  win {t['p_win']:5.1%}  FAAB {t['faab']}")


if __name__ == "__main__":
    main()
