import os
import time
import requests
import pandas as pd

# College football (FBS) stats from ESPN's free feed - no key needed.
#   cfb_games.csv    every game (scores, rankings), including this week's upcoming ones
#   cfb_teams.csv    team id / abbreviation / name / logo
#   cfb_players.csv  every player's passing, rushing and receiving in every finished game
# The FIRST run downloads ~3,000 box scores (~25 min). After that it only fetches games it doesn't have yet.

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
SEASONS = range(2023, 2027)   # 2023 to this season
WEEKS = range(1, 17)          # regular season weeks

# already updated today and no game has started since? nothing new to download
from Freshness import already_fresh, football_game_starts
if already_fresh("cfb_games.csv", football_game_starts("cfb")):
    raise SystemExit(0)

# ---- 1. every game, week by week ----
# Only re-check what can still change - that's what keeps the daily update fast:
#   finished seasons: reused from last time (they never change)
#   this season: weeks where every game is already final are reused; the rest (this week, upcoming weeks -
#   their rankings and kick-off times change) are checked again
now = pd.Timestamp.now()
this_season = now.year if now.month >= 3 else now.year - 1   # January bowl games still belong to last season
saved = pd.read_csv("cfb_games.csv", dtype={"game_id": str}) if os.path.exists("cfb_games.csv") \
    else pd.DataFrame(columns=["season", "week", "completed"])
games = []
teams = {t["team"]: t for t in pd.read_csv("cfb_teams.csv").to_dict("records")} if os.path.exists("cfb_teams.csv") else {}
for season in SEASONS:
    old_season = saved[saved["season"] == season]
    if season < this_season and len(old_season) > 500:
        games += old_season.to_dict("records")
        print(season, "->", len(old_season), "games (already saved)")
        continue
    done_weeks = {w for w, g in old_season.groupby("week") if g["completed"].astype(bool).all()}
    games += old_season[old_season["week"].isin(done_weeks)].to_dict("records")
    for week in WEEKS:
        if week in done_weeks:
            continue   # every game that week is final - nothing can change
        # groups=80 = FBS (the top division) - limit=300 so we get every game that week in one request
        events = requests.get(f"{ESPN}/scoreboard", params={"dates": season, "seasontype": 2, "week": week,
                                                            "groups": 80, "limit": 300}, timeout=30).json().get("events", [])
        for e in events:
            row = {"game_id": str(e["id"]), "season": season, "week": week,
                   "date": pd.Timestamp(e["date"]).tz_convert("America/Toronto").strftime("%Y-%m-%d"),
                   "completed": e["status"]["type"]["completed"]}
            for t in e["competitions"][0]["competitors"]:
                side = t["homeAway"]   # "home" or "away"
                team = t["team"]
                row[f"{side}_team"] = team.get("abbreviation", team["id"])
                row[f"{side}_name"] = team["displayName"]
                row[f"{side}_score"] = float(t["score"]) if t.get("score") not in (None, "") else None
                row[f"{side}_rank"] = t.get("curatedRank", {}).get("current", 99)   # 99 = not ranked
                teams[row[f"{side}_team"]] = {"team": row[f"{side}_team"], "id": team["id"],
                                              "name": team["displayName"], "logo": team.get("logo", "")}
            games.append(row)
        time.sleep(0.2)
    print(season, "->", sum(g["season"] == season for g in games), "games")

games = pd.DataFrame(games).drop_duplicates("game_id")
games.to_csv("cfb_games.csv", index=False)
pd.DataFrame(teams.values()).to_csv("cfb_teams.csv", index=False)
print(len(games), "games saved to cfb_games.csv |", len(teams), "teams")

# ---- 2. player stats from each finished game's box score (only the ones we don't have yet) ----
WANTED = {("passing", "passingYards"): "passing_yards", ("passing", "completions/passingAttempts"): "attempts",
          ("rushing", "rushingAttempts"): "carries", ("rushing", "rushingYards"): "rushing_yards",
          ("receiving", "receptions"): "receptions", ("receiving", "receivingYards"): "receiving_yards"}


def box_rows(game):
    """one row per player in one game: attempts, passing yards, carries, rushing yards, receptions, receiving yards"""
    data = requests.get(f"{ESPN}/summary", params={"event": game["game_id"]}, timeout=30).json()
    sides = data.get("boxscore", {}).get("players", [])
    team_abbrevs = [s["team"].get("abbreviation", s["team"]["id"]) for s in sides]
    players = {}
    for side, abbrev in zip(sides, team_abbrevs):
        opponent = [a for a in team_abbrevs if a != abbrev][0] if len(team_abbrevs) == 2 else None
        for category in side["statistics"]:
            for athlete in category["athletes"]:
                pid = athlete["athlete"]["id"]
                row = players.setdefault(pid, {"game_id": game["game_id"], "season": game["season"],
                                               "week": game["week"], "date": game["date"], "player_id": pid,
                                               "player": athlete["athlete"]["displayName"], "team": abbrev,
                                               "opponent": opponent, "attempts": 0, "passing_yards": 0, "carries": 0,
                                               "rushing_yards": 0, "receptions": 0, "receiving_yards": 0})
                for (cat, key), column in WANTED.items():
                    if category["name"] == cat and key in category["keys"]:
                        value = athlete["stats"][category["keys"].index(key)]
                        if key == "completions/passingAttempts":   # "21/40" -> 40 attempts
                            value = value.split("/")[-1]
                        row[column] = int(float(value)) if value not in ("", "--") else 0
    return list(players.values())


old = pd.read_csv("cfb_players.csv") if os.path.exists("cfb_players.csv") else pd.DataFrame()
have = set(old["game_id"].astype(str)) if len(old) else set()
todo = games[games["completed"] & ~games["game_id"].astype(str).isin(have)]
print(len(todo), "new finished games to download")

rows, done = [], 0
for _, game in todo.iterrows():
    try:
        rows += box_rows(game)
    except Exception as error:   # one bad game shouldn't stop the whole download
        print("  skipped game", game["game_id"], error)
    done += 1
    if done % 200 == 0 or done == len(todo):   # save progress every 200 games
        new = pd.DataFrame(rows)
        all_players = pd.concat([old, new], ignore_index=True) if len(old) else new
        all_players = all_players.sort_values(["player_id", "season", "week"])
        all_players.to_csv("cfb_players.csv", index=False)
        print(f"  {done}/{len(todo)} games done - {len(all_players)} player-games saved")
    time.sleep(0.1)
