import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
from Freshness import already_fresh
from paths import csv_path

# PWHL (Professional Women's Hockey League) stats, straight from the feed thepwhl.com itself uses (HockeyTech).
# No API key of your own needed - this is the public key built into the PWHL website.
#   data/pwhl_games.csv    every regular-season and playoff game (scores, overtime / shootout), incl. upcoming ones
#   data/pwhl_teams.csv    team code / name / logo
#   data/pwhl_skaters.csv  every skater's goals, assists, points, shots, hits, blocks and ice time in every game
#   data/pwhl_goalies.csv  every goalie's shots against, saves, goals against and ice time in every game
# The FIRST run downloads every game since the league started (2024) - a few minutes. After that only new games.

FEED = "https://lscluster.hockeytech.com/feed/index.php"
BASE = {"key": "446521baf8c38984", "client_code": "pwhl", "fmt": "json", "lang": "en"}
TZ = "America/Toronto"


def get(params):
    """one request to the PWHL feed - waits and tries again if the site hangs up (up to 5 times)"""
    for attempt in range(5):
        try:
            text = requests.get(FEED, params=BASE | params, timeout=30).text
            return json.loads(text[text.find("{"):text.rfind("}") + 1])   # some views wrap the JSON in ( )
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return {}


def minutes(clock):
    """ice time "19:57" -> 19.95 minutes"""
    try:
        m, s = str(clock).split(":")[:2]
        return round(int(m) + int(s) / 60, 2)
    except ValueError:
        return 0.0


# ---- 1. every regular-season and playoff schedule (skip preseason) ----
seasons = get({"feed": "modulekit", "view": "seasons"}).get("SiteKit", {}).get("Seasons", [])
seasons = [s for s in seasons if s["career"] == "1"]   # career = counts in the record books (regular + playoffs)
games, teams = [], {}
for s in seasons:
    for g in get({"feed": "modulekit", "view": "schedule", "season_id": s["season_id"]}).get("SiteKit", {}).get("Schedule", []):
        final = g["final"] == "1"
        games.append({"game_id": g["game_id"], "season_id": int(s["season_id"]), "season": s["season_name"],
                      "playoffs": int(s["playoff"] == "1"), "date": g["date_played"],
                      "start": g["GameDateISO8601"], "completed": final,
                      "home_team": g["home_team_code"], "home_name": g["home_team_name"],
                      "away_team": g["visiting_team_code"], "away_name": g["visiting_team_name"],
                      "home_score": float(g["home_goal_count"]) if final else None,
                      "away_score": float(g["visiting_goal_count"]) if final else None,
                      "overtime": int(g["overtime"] == "1"), "shootout": int(g["shootout"] not in ("0", "")),
                      "status": g["game_status"]})
    for t in get({"feed": "modulekit", "view": "teamsbyseason", "season_id": s["season_id"]}).get(
            "SiteKit", {}).get("Teamsbyseason", []):
        if t["code"] != "TBD":   # placeholder for playoff games whose teams aren't decided yet
            teams[t["code"]] = {"team": t["code"], "name": t["name"], "logo": t["team_logo_url"]}
    print(f"  {s['season_name']}: {sum(g['season_id'] == int(s['season_id']) for g in games)} games")
games = pd.DataFrame(games).drop_duplicates("game_id").sort_values("start")

# already downloaded today and no game has started since? reuse it
today = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d")
if already_fresh(csv_path("pwhl_games.csv"), list(games.loc[games["date"] == today, "start"])):
    raise SystemExit(0)
pd.DataFrame(teams.values()).to_csv(csv_path("pwhl_teams.csv"), index=False)
print(len(games), "games |", len(teams), "teams")


# ---- 2. box scores of finished games we don't have yet ----
def box_rows(game):
    """(skater rows, goalie rows) for one game"""
    data = get({"feed": "statviewfeed", "view": "gameSummary", "game_id": game["game_id"]})
    skaters, goalies = [], []
    for side, team, opponent in (("homeTeam", game["home_team"], game["away_team"]),
                                 ("visitingTeam", game["away_team"], game["home_team"])):
        common = {"game_id": game["game_id"], "season_id": game["season_id"], "playoffs": game["playoffs"],
                  "date": game["date"], "team": team, "opponent": opponent, "home": int(side == "homeTeam")}
        for p in data.get(side, {}).get("skaters", []):
            st = p["stats"]
            skaters.append(common | {
                "player_id": p["info"]["id"], "player": f'{p["info"]["firstName"]} {p["info"]["lastName"]}',
                "position": p["info"]["position"], "goals": st["goals"], "assists": st["assists"],
                "points": st["points"], "shots": st["shots"], "hits": st["hits"], "blocks": st["blockedShots"],
                "pim": st["penaltyMinutes"], "plus_minus": st["plusMinus"], "toi": minutes(st["toi"])})
        for p in data.get(side, {}).get("goalies", []):
            st = p["stats"]
            if not st.get("timeOnIce"):
                continue   # dressed as the backup but never played
            goalies.append(common | {
                "player_id": p["info"]["id"], "player": f'{p["info"]["firstName"]} {p["info"]["lastName"]}',
                "starter": p["starting"], "shots_against": st["shotsAgainst"], "saves": st["saves"],
                "goals_against": st["goalsAgainst"], "toi": minutes(st["timeOnIce"])})
    return skaters, goalies


def safe_box_rows(game):
    """one weird box score shouldn't crash the whole download - skip it (the next run tries it again)"""
    try:
        return box_rows(game)
    except Exception as error:
        print(f"  skipped game {game['game_id']} ({game['date']}): {error!r}")
        return [], []


old_sk = pd.read_csv(csv_path("pwhl_skaters.csv")) if os.path.exists(csv_path("pwhl_skaters.csv")) else pd.DataFrame()
old_gl = pd.read_csv(csv_path("pwhl_goalies.csv")) if os.path.exists(csv_path("pwhl_goalies.csv")) else pd.DataFrame()
have = set(old_sk["game_id"].astype(str)) if len(old_sk) else set()
todo = games[games["completed"] & ~games["game_id"].astype(str).isin(have)]
print(len(todo), "new finished games to download")

sk_rows, gl_rows = [], []
with ThreadPoolExecutor(max_workers=4) as pool:   # 4 at a time: faster, and still polite to the server
    for i, (sk, gl) in enumerate(pool.map(lambda g: safe_box_rows(g[1]), todo.iterrows()), start=1):
        sk_rows += sk
        gl_rows += gl
        if i % 100 == 0 or i == len(todo):
            for old, new, path in ((old_sk, sk_rows, csv_path("pwhl_skaters.csv")), (old_gl, gl_rows, csv_path("pwhl_goalies.csv"))):
                everyone = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
                everyone.sort_values(["player_id", "date"]).to_csv(path, index=False)
            print(f"  {i}/{len(todo)} games done")

# saved LAST, on purpose: the "already up to date today" check looks at this file,
# so a download that crashes halfway never counts as finished
games.to_csv(csv_path("pwhl_games.csv"), index=False)
print(len(games), "games saved to data/pwhl_games.csv - all done")
