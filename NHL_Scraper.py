import time
import requests
import pandas as pd

url = "https://api.nhle.com/stats/rest/en/skater/summary"
toi_url = "https://api.nhle.com/stats/rest/en/skater/timeonice"   # ice-time report: has power-play time
realtime_url = "https://api.nhle.com/stats/rest/en/skater/realtime"   # shot attempts (shots + misses + blocked)
goalie_url = "https://api.nhle.com/stats/rest/en/goalie/summary"  # goalies are a separate report from skaters
team_url = "https://api.nhle.com/stats/rest/en/team/summary"      # team results: who won, goals and shots

# every season to download: season id -> (first day to check, last day to check)
# the current season runs up to today, so new players and rookies are included
# already updated today and no game has started since? nothing new to download
from Freshness import already_fresh, nhl_game_starts
if already_fresh("skaters.csv", nhl_game_starts()):
    raise SystemExit(0)

today = pd.Timestamp.today().normalize()
seasons = {
    "20242025": ("2024-10-01", "2025-04-30"),   # more history to learn from (tested: makes the model more accurate)
    "20252026": ("2025-10-01", "2026-04-30"),
    "20262027": ("2026-10-01", today),
}

tables = []          # empty list, one table per chunk gets added here
goalie_tables = []   # same idea for goalies
team_tables = []     # and for team results

for season, (first_day, last_day) in seasons.items():
    # start date of every 2-week chunk of this season
    chunk_starts = pd.date_range(first_day, last_day, freq="14D")

    for start in chunk_starts:
        end = start + pd.Timedelta(days=14)    # chunk ends 14 days after it starts
        start_text = start.strftime("%Y-%m-%d")   # date -> text like "2025-10-07"
        end_text = end.strftime("%Y-%m-%d")

        paramaters = {
            "isAggregate": "false",
            "isGame": "true",
            "limit": "-1",
            # f-string: {season}, {start_text} and {end_text} get replaced with this chunk's values
            "cayenneExp": f'gameTypeId=2 and seasonId={season} and gameDate>="{start_text}" and gameDate<"{end_text}"',
        }

        response = requests.get(url, params=paramaters)
        data = response.json()
        chunk = pd.DataFrame(data["data"])
        print(season, start_text, "to", end_text, "->", len(chunk), "rows of", data["total"])   # should match each time

        if len(chunk) > 0:
            # same filter, different report: power-play ice time for the same player-games
            toi = pd.DataFrame(requests.get(toi_url, params=paramaters).json()["data"])
            # match rows by player + game and copy ppTimeOnIce across (how="left" keeps every summary row)
            chunk = chunk.merge(toi[["playerId", "gameId", "ppTimeOnIce"]], on=["playerId", "gameId"], how="left")
            # and the shot-attempt report: shows how much a player is TRYING to shoot (steadier than shots on goal)
            rt = pd.DataFrame(requests.get(realtime_url, params=paramaters).json()["data"])
            chunk = chunk.merge(rt[["playerId", "gameId", "totalShotAttempts"]], on=["playerId", "gameId"], how="left")
            chunk["season"] = season   # the API doesn't include the season in game rows, so add it
            tables.append(chunk)       # add this chunk's table to the list

            # same filter again, goalie report: saves, shots against, and whether he started
            goalies = pd.DataFrame(requests.get(goalie_url, params=paramaters).json()["data"])
            goalies["season"] = season
            goalie_tables.append(goalies)

            # and the team report: one row per team per game, with the result
            teams = pd.DataFrame(requests.get(team_url, params=paramaters).json()["data"])
            teams["season"] = season
            team_tables.append(teams)
        time.sleep(1)                  # wait 1 second so we don't hammer the NHL's server

# glue all the chunks into one big table (ignore_index renumbers the rows 0, 1, 2, ...)
df = pd.concat(tables, ignore_index=True)

# ---- clean up ----
df["toiMinutes"] = (df["timeOnIcePerGame"] / 60).round(1)   # ice time: seconds -> minutes (1210 -> 20.2)
df["ppMinutes"] = (df["ppTimeOnIce"] / 60).round(1)          # power-play time: seconds -> minutes
df["home"] = (df["homeRoad"] == "H").astype(int)             # "H"/"R" -> 1 for home, 0 for away
df["attempts"] = df["totalShotAttempts"]                     # shorter name: every shot he tried

# keep only the columns the model needs, in a readable order
df = df[["season", "gameDate", "gameId", "playerId", "skaterFullName", "positionCode",
         "teamAbbrev", "opponentTeamAbbrev", "home", "toiMinutes", "ppMinutes",
         "goals", "assists", "points", "shots", "ppPoints", "attempts"]]

# each player's games together, oldest to newest
df = df.sort_values(["playerId", "gameDate"])

df.to_csv("skaters.csv", index=False)   # replaces the old file, so it's always one fresh, up-to-date copy
print(len(df), "rows |", df["playerId"].nunique(), "players |", df["gameId"].nunique(), "games")
print("saved skaters.csv")

# ---- goalies ----
gl = pd.concat(goalie_tables, ignore_index=True)
gl["home"] = (gl["homeRoad"] == "H").astype(int)
gl["toiMinutes"] = (gl["timeOnIce"] / 60).round(1)
gl = gl[["season", "gameDate", "gameId", "playerId", "goalieFullName", "teamAbbrev", "opponentTeamAbbrev",
         "home", "gamesStarted", "toiMinutes", "shotsAgainst", "saves", "goalsAgainst"]]
gl = gl.sort_values(["playerId", "gameDate"])
gl.to_csv("goalies.csv", index=False)   # also replaced every run
print(len(gl), "goalie games |", gl["playerId"].nunique(), "goalies | saved goalies.csv")

# ---- teams ----
tm = pd.concat(team_tables, ignore_index=True)
# the team report has the full name but not the abbreviation - but the OTHER team's row in the same game
# lists this team as its opponent, so: my abbreviation = my opponent's "opponentTeamAbbrev"
other = tm[["gameId", "teamId", "opponentTeamAbbrev"]].rename(columns={"teamId": "otherId", "opponentTeamAbbrev": "team"})
tm = tm.merge(other, on="gameId")
tm = tm[tm["teamId"] != tm["otherId"]]   # keep the pairing with the OTHER team, not itself
tm["home"] = (tm["homeRoad"] == "H").astype(int)
tm = tm.rename(columns={"opponentTeamAbbrev": "opponent", "wins": "win",
                        "shotsForPerGame": "shotsFor", "shotsAgainstPerGame": "shotsAgainst"})
tm = tm[["season", "gameDate", "gameId", "team", "opponent", "home", "win",
         "goalsFor", "goalsAgainst", "shotsFor", "shotsAgainst"]]
tm = tm.sort_values(["team", "gameDate"])
tm.to_csv("teams.csv", index=False)   # also replaced every run
print(len(tm), "team games | saved teams.csv")
