import os
import time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
from Freshness import already_fresh, football_game_starts

# MLB stats from ESPN's free feed - no key needed.
#   mlb_games.csv    every regular-season and playoff game (scores + both starting pitchers), incl. upcoming ones
#   mlb_teams.csv    team abbreviation / name / logo
#   mlb_players.csv  one row per player per game: batters (hits, home runs, RBIs...) and pitchers (innings, strikeouts...)
# The FIRST run downloads ~5,000 box scores (several at a time, ~30 min). After that only new games.

ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
# (first day, last day) to check for each season - opening day in late March through the World Series
SEASONS = {2025: ("2025-03-15", "2025-11-05"),
           2026: ("2026-03-20", min("2026-11-10", pd.Timestamp.now().strftime("%Y-%m-%d")))}
if pd.Timestamp.now() >= pd.Timestamp("2027-03-01"):   # next season: picked up automatically
    SEASONS[2027] = ("2027-03-15", pd.Timestamp.now().strftime("%Y-%m-%d"))

if already_fresh("mlb_games.csv", football_game_starts("mlb")):   # same "nothing new today" check as the others
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
    """ESPN's text stat -> number ("--" or blank means 0)"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def innings(value):
    """ESPN writes innings as 5.2 = 5 innings and 2 outs -> 5.67"""
    whole, _, outs = str(value).partition(".")
    return num(whole) + num(outs or 0) / 3


# ---- 1. every regular-season and playoff game, day by day ----
games, teams = [], {}
for season, (first, last) in SEASONS.items():
    for day in pd.date_range(first, last):
        events = get(f"{ESPN}/scoreboard", {"dates": day.strftime("%Y%m%d")})
        for e in events.get("events", []):
            if e["season"]["type"] not in (2, 3):   # 2 = regular season, 3 = playoffs (skip spring training)
                continue
            row = {"game_id": e["id"], "season": season, "playoffs": int(e["season"]["type"] == 3),
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
pd.DataFrame(teams.values()).to_csv("mlb_teams.csv", index=False)
print(len(games), "games |", len(teams), "teams")


# ---- 2. player stats from each finished game's box score (only the ones we don't have yet) ----
def box_rows(game):
    """one row per batter and one per pitcher who played"""
    data = get(f"{ESPN}/summary", {"event": game["game_id"]})
    sides = data.get("boxscore", {}).get("players", [])
    abbrevs = [s["team"]["abbreviation"] for s in sides]
    rows = []
    for side, team in zip(sides, abbrevs):
        opponent = [a for a in abbrevs if a != team][0] if len(abbrevs) == 2 else None
        for group in side["statistics"]:
            kind = group.get("type")   # "batting" or "pitching"
            if kind not in ("batting", "pitching"):
                continue
            for athlete in group["athletes"]:
                if not athlete.get("stats") or "id" not in athlete.get("athlete", {}):
                    continue
                s = dict(zip(group["keys"], athlete["stats"]))
                row = {"game_id": game["game_id"], "season": game["season"], "playoffs": game["playoffs"],
                       "date": game["date"], "player_id": athlete["athlete"]["id"],
                       "player": athlete["athlete"]["displayName"], "team": team, "opponent": opponent,
                       "home": int(team == game["home_team"]), "role": "batter" if kind == "batting" else "pitcher",
                       "starter": int(bool(athlete.get("starter")))}
                if kind == "batting":
                    row |= {"bat_order": athlete.get("batOrder", 0), "at_bats": num(s.get("atBats")),
                            "hits": num(s.get("hits")), "home_runs": num(s.get("homeRuns")),
                            "rbis": num(s.get("RBIs")), "walks": num(s.get("walks")),
                            "batter_k": num(s.get("strikeouts"))}
                else:
                    row |= {"innings": round(innings(s.get("fullInnings.partInnings")), 2),
                            "pitches": num(s.get("pitches")), "strikeouts": num(s.get("strikeouts")),
                            "hits_allowed": num(s.get("hits")), "runs_allowed": num(s.get("runs")),
                            "walks_allowed": num(s.get("walks")), "hr_allowed": num(s.get("homeRuns"))}
                rows.append(row)
    return rows


def safe_box_rows(game):
    """one weird box score shouldn't crash the whole download - skip it (the next run tries it again)"""
    try:
        return box_rows(game)
    except Exception as error:
        print(f"  skipped game {game['game_id']} ({game['date']}): {error!r}")
        return []


old = pd.read_csv("mlb_players.csv") if os.path.exists("mlb_players.csv") else pd.DataFrame()
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
            everyone.sort_values(["player_id", "date"]).to_csv("mlb_players.csv", index=False)
            print(f"  {i}/{len(todo)} games done - {len(everyone)} player-games saved")

# saved LAST, on purpose: the "already up to date today" check looks at this file,
# so a download that crashes halfway never counts as finished
games.to_csv("mlb_games.csv", index=False)
print(len(games), "games saved to mlb_games.csv - all done")
