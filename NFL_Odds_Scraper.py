import os
import time
import requests
import pandas as pd
from Odds_Scraper import flatten   # same "nested odds -> simple rows" function the NHL odds use
from Freshness import already_fresh, football_game_starts
from paths import csv_path

# NFL odds from FanDuel, DraftKings and BetMGM, through The Odds API - with its OWN key,
# so the NFL never uses up your hockey credits.
#   setx NFL_ODDS_API_KEY "your-second-key"     (then restart VS Code)
# Cost: about 5 credits per game (one per market) - a full Sunday is ~70 credits, so don't run it every hour.

# the NFL's own key if you've set one - for now it falls back to the hockey key (they share its credits).
# when you get a second key: setx NFL_ODDS_API_KEY "your-second-key", restart VS Code, and the NFL uses that instead
API_KEY = os.getenv("NFL_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
base_url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
books = "fanduel,draftkings,betmgm"
markets = "h2h,player_pass_yds,player_rush_yds,player_reception_yds,player_receptions"
DAYS_AHEAD = 4   # only games starting in the next 4 days (run Sunday morning -> Sunday + Monday night)

if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit('No API key. Get a free key at the-odds-api.com, then run: setx ODDS_API_KEY "your-key" '
                         "and restart VS Code.")
    if not os.getenv("NFL_ODDS_API_KEY"):
        print("Using your hockey key for NFL odds (they share its credits).")

    # already downloaded today and no game has started since? reuse it - saves credits
    if already_fresh(csv_path("nfl_odds.csv"), football_game_starts("nfl")):
        raise SystemExit(0)

    # 1. the list of upcoming games (this call is free)
    events = requests.get(f"{base_url}/events", params={"apiKey": API_KEY}).json()
    now = pd.Timestamp.now(tz="America/Toronto")
    upcoming = [e for e in events
                if now < pd.Timestamp(e["commence_time"]) <= now + pd.Timedelta(days=DAYS_AHEAD)]
    print(f"{len(upcoming)} games in the next {DAYS_AHEAD} days - about {len(upcoming) * 5} credits")

    # 2. the odds for each game (this costs credits)
    rows = []
    for event in upcoming:
        response = requests.get(f"{base_url}/events/{event['id']}/odds", params={
            "apiKey": API_KEY,
            "bookmakers": books,
            "markets": markets,
            "oddsFormat": "decimal",
        })
        if response.status_code != 200:
            print("  skipped", event["away_team"], "@", event["home_team"], "->", response.status_code, response.text[:100])
            continue
        game_rows = flatten(response.json())
        start = pd.Timestamp(event["commence_time"]).tz_convert("America/Toronto")
        for row in game_rows:
            row["game_date"] = start.strftime("%Y-%m-%d")   # NFL games span several days, so remember each one's date
        rows += game_rows
        print(" ", start.strftime("%a %I:%M %p"), event["away_team"], "@", event["home_team"], "->",
              len(game_rows), "prices | credits left:", response.headers.get("x-requests-remaining"))
        time.sleep(1)   # be polite to the server

    odds = pd.DataFrame(rows)
    if odds.empty:
        raise SystemExit("No NFL odds yet - books post player props a few days before each game.")

    # the book's chance = 1 / decimal odds (still includes the book's cut - the model removes it later)
    odds["book_chance"] = (1 / odds["price"]).round(3)
    odds.to_csv(csv_path("nfl_odds.csv"), index=False)   # replaced every run, like data/odds.csv for hockey
    print(len(odds), "prices saved to data/nfl_odds.csv")
