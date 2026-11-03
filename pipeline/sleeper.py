"""Sleeper data access. LiveSource hits the public API; DirSource reads a saved pull (for testing)."""
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://api.sleeper.app"
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
_POS_Q = "&".join(f"position[]={p}" for p in POSITIONS)


class LiveSource:
    def __init__(self, league_id):
        self.league_id = league_id
        self._cache = {}

    def _get(self, path):
        if path in self._cache:
            return self._cache[path]
        url = BASE + path
        data = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "guillotine-power-rankings"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        self._cache[path] = data
        return data

    def league(self):
        return self._get(f"/v1/league/{self.league_id}")

    def state(self):
        return self._get("/v1/state/nfl")

    def users(self):
        return self._get(f"/v1/league/{self.league_id}/users")

    def rosters(self):
        return self._get(f"/v1/league/{self.league_id}/rosters")

    def matchups(self, w):
        return self._get(f"/v1/league/{self.league_id}/matchups/{w}")

    def transactions(self, w):
        return self._get(f"/v1/league/{self.league_id}/transactions/{w}")

    def stats(self, season, w):
        return self._get(f"/stats/nfl/{season}/{w}?season_type=regular&{_POS_Q}")

    def projections(self, season, w):
        return self._get(f"/projections/nfl/{season}/{w}?season_type=regular&{_POS_Q}")

    def schedule(self, season):
        return self._get(f"/schedule/nfl/regular/{season}")

    def players(self):
        return self._get("/v1/players/nfl")


class DirSource:
    """Reads the folder layout written by pull_sleeper_guillotine.py."""

    def __init__(self, root):
        self.root = root

    def _f(self, name):
        p = os.path.join(self.root, name)
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            return json.load(fh)

    def league(self): return self._f("league.json")
    def state(self): return self._f("state.json")
    def users(self): return self._f("users.json")
    def rosters(self): return self._f("rosters.json")
    def matchups(self, w): return self._f(f"matchups/week_{w:02d}.json")
    def transactions(self, w): return self._f(f"transactions/week_{w:02d}.json")
    def stats(self, season, w): return self._f(f"stats/week_{w:02d}.json")
    def projections(self, season, w): return self._f(f"projections/week_{w:02d}.json")
    def schedule(self, season): return self._f("schedule.json")
    def players(self): return self._f("players.json")
