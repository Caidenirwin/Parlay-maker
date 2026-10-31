import os
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
from Freshness import already_fresh, football_game_starts
from paths import csv_path

# WNBA stats from ESPN's free feed - no key needed.
#   data/wnba_games.csv   every regular-season and playoff game (scores), including upcoming ones
#   data/wnba_teams.csv    team abbreviation / name / logo
#   data/wnba_players.csv  every player's points, rebounds, assists, 3-pointers and minutes in every finished game
# The FIRST run downloads ~800 box scores (several at a time, ~5 min). After that only new games.

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
# the WNBA plays May to October, all in one calendar year. Each season: (first day, last day) to check
SEASONS = {2024: ("2024-05-10", "2024-10-25"), 2025: ("2025-05-12", "2025-10-25"),
           2026: ("2026-05-05", min("2026-10-31", pd.Timestamp.now().strftime("%Y-%m-%d")))}
if pd.Timestamp.now() >= pd.Timestamp("2027-05-01"):   # next season: picked up automatically
    SEASONS[2027] = ("2027-05-01", pd.Timestamp.now().strftime("%Y-%m-%d"))

if already_fresh(csv_path("wnba_games.csv"), football_game_starts("wnba")):   # same "nothing new today" check as the others
    raise SystemExit(0)


def get(url, params):
    """ESPN sometimes hangs up mid-download - wait a moment and try again (up to 5 times)"""
    for attempt in range(5):
        try:
            return requests.get(url, params=params, timeout=30).json()
        except (requests.RequestException, ValueError):
            time.sleep(2 * (attempt + 1))
    return {}   # still failing -> treat as "nothing there" and move on


def num(value):
    """ESPN's text stat -> whole number ("--" or blank means 0)"""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0

# ---- 1. every regular-season game, day by day ----
games, teams = [], {}
for season, (first, last) in SEASONS.items():
    for day in pd.date_range(first, last):
        events = get(f"{ESPN}/scoreboard", {"dates": day.strftime("%Y%m%d")})
        for e in events.get("events", []):
            if e["season"]["type"] not in (2, 3):   # 2 = regular season, 3 = playoffs (skip preseason)
                continue
            row = {"game_id": e["id"], "season": season,
                   "date": pd.Timestamp(e["date"]).tz_convert("America/Toronto").strftime("%Y-%m-%d"),
                   "completed": e["status"]["type"]["completed"]}
            for t in e["competitions"][0]["competitors"]:
                side = t["homeAway"]
                row[f"{side}_team"] = t["team"]["abbreviation"]
                row[f"{side}_name"] = t["team"]["displayName"]
                row[f"{side}_score"] = float(t["score"]) if t.get("score") not in (None, "") else None
                teams[t["team"]["abbreviation"]] = {"team": t["team"]["abbreviation"], "name": t["team"]["displayName"],
                                                    "logo": t["team"].get("logo", "")}
            games.append(row)
        time.sleep(0.05)
        if day.day == 1:   # a sign of life once a month - this part takes a few minutes
            print(f"  checked up to {day.strftime('%b %Y')} - {len(games)} games so far")
    print(season, "->", sum(g["season"] == season for g in games), "games")

games = pd.DataFrame(games).drop_duplicates("game_id")
pd.DataFrame(teams.values()).to_csv(csv_path("wnba_teams.csv"), index=False)
print(len(games), "regular-season games |", len(teams), "teams")


# ---- 2. player stats from each finished game's box score (only the ones we don't have yet) ----
def box_rows(game):
    """one row per player who played: minutes, points, rebounds, assists, 3-pointers (made / tried), shots tried"""
    data = get(f"{ESPN}/summary", {"event": game["game_id"]})
    sides = data.get("boxscore", {}).get("players", [])
    abbrevs = [s["team"]["abbreviation"] for s in sides]
    rows = []
    for side, team in zip(sides, abbrevs):
        opponent = [a for a in abbrevs if a != team][0] if len(abbrevs) == 2 else None
        group = side["statistics"][0]
        keys = group["keys"]
        for athlete in group["athletes"]:
            if athlete.get("didNotPlay") or not athlete.get("stats") or "id" not in athlete.get("athlete", {}):
                continue   # injured / didn't get in / ESPN left out who he is
            s = dict(zip(keys, athlete["stats"]))
            made3, tried3 = (s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "0-0").split("-") + ["0"])[:2]
            fga = (s.get("fieldGoalsMade-fieldGoalsAttempted", "0-0").split("-") + ["0"])[1]
            rows.append({"game_id": game["game_id"], "season": game["season"], "date": game["date"],
                         "player_id": athlete["athlete"]["id"], "player": athlete["athlete"]["displayName"],
                         "team": team, "opponent": opponent, "home": int(team == game["home_team"]),
                         "starter": int(bool(athlete.get("starter"))),
                         "minutes": num(s.get("minutes")), "points": num(s.get("points")),
                         "rebounds": num(s.get("rebounds")), "assists": num(s.get("assists")),
                         "threes": num(made3), "threes_tried": num(tried3), "shots_tried": num(fga)})
    return rows


def safe_box_rows(game):
    """one weird box score shouldn't crash the whole download - skip it (the next run tries it again)"""
    try:
        return box_rows(game)
    except Exception as error:
        print(f"  skipped game {game['game_id']} ({game['date']}): {error!r}")
        return []


old = pd.read_csv(csv_path("wnba_players.csv")) if os.path.exists(csv_path("wnba_players.csv")) else pd.DataFrame()
have = set(old["game_id"].astype(str)) if len(old) else set()
todo = games[games["completed"] & ~games["game_id"].astype(str).isin(have)]
print(len(todo), "new finished games to download")

rows = []
# 4 at a time: much faster than one by one, and still polite to ESPN's servers
with ThreadPoolExecutor(max_workers=4) as pool:
    for i, result in enumerate(pool.map(lambda g: safe_box_rows(g[1]), todo.iterrows()), start=1):
        rows += result
        if i % 300 == 0 or i == len(todo):
            new = pd.DataFrame(rows)
            everyone = pd.concat([old, new], ignore_index=True) if len(old) else new
            everyone.sort_values(["player_id", "date"]).to_csv(csv_path("wnba_players.csv"), index=False)
            print(f"  {i}/{len(todo)} games done - {len(everyone)} player-games saved")

# saved LAST, on purpose: the "already up to date today" check looks at this file,
# so a download that crashes halfway never counts as finished
games.to_csv(csv_path("wnba_games.csv"), index=False)
print(len(games), "games saved to data/wnba_games.csv - all done")
