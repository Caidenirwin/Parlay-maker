import os
import sys
import time
import requests
import pandas as pd
from Freshness import already_fresh, nhl_game_starts

# Player-prop odds for tonight's NHL games from FanDuel, DraftKings and BetMGM,
# through The Odds API (free key at https://the-odds-api.com - 500 credits a month).
# Cost: about 5 credits per game (one per market), so a 10-game night is about 50 credits.
#
# Ontario: all three books operate in Ontario, but the API only has their US prices.
# Ontario prices are usually the same or very close - always check the price in your app before betting.
# (Hard Rock isn't available in Ontario, so it's left out.)

# read the key from your computer, never paste it in the code:
# NHL_ODDS_API_KEY if you've set one (setx NHL_ODDS_API_KEY "your-key"), else your first key
API_KEY = os.getenv("NHL_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
# This week: run with --week (the dashboard's "This week" switch) for every game in the next 7 days -
# about 5 credits per game, so a full NHL week (~50 games) is ~250 credits.
WEEK = "--week" in sys.argv
base_url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
books = "fanduel,draftkings,betmgm"
markets = "h2h,player_points,player_goal_scorer_anytime,player_shots_on_goal,player_total_saves"   # h2h = moneyline


def flatten(game):
    """Turn one game's nested odds data into simple rows: one row per book, per player, per bet."""
    rows = []
    for book in game.get("bookmakers", []):              # each sportsbook
        for market in book["markets"]:                   # each bet type (points, goals, shots)
            for outcome in market["outcomes"]:           # each player's Over / Under / Yes price
                rows.append({
                    "game": f"{game['away_team']} @ {game['home_team']}",
                    "book": book["key"],
                    "market": market["key"],
                    "player": outcome.get("description"),  # the player's name
                    "side": outcome["name"],               # "Over", "Under" or "Yes"
                    "line": outcome.get("point", 0.5),     # e.g. 2.5 shots (anytime goal has no line, it's 0.5)
                    "price": outcome["price"],             # decimal odds: 2.50 means a $1 bet returns $2.50
                })
    return rows


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("No API key. Get a free one at the-odds-api.com, then run: setx ODDS_API_KEY \"your-key\" "
                         "and restart VS Code.")

    # already downloaded today and no game has started since? reuse it - saves credits
    fresh = already_fresh("odds.csv", nhl_game_starts())
    if fresh and WEEK and ("game_date" not in pd.read_csv("odds.csv", nrows=1).columns
                           or pd.read_csv("odds.csv")["game_date"].nunique() < 2):
        print("...but that download was tonight only - getting the whole week now")
        fresh = False
    if fresh:
        raise SystemExit(0)

    # 1. the list of upcoming games (this call is free)
    events = requests.get(f"{base_url}/events", params={"apiKey": API_KEY}).json()

    # keep only TODAY's games (Ontario time) that haven't started yet - or, with --week, the next 7 days' games.
    # (not "the next 24 hours" - late at night that grabs tomorrow's games and mixes them up with tonight's)
    now = pd.Timestamp.now(tz="America/Toronto")
    last_day = (now + pd.Timedelta(days=6 if WEEK else 0)).date()
    tonight = []
    for e in events:
        start = pd.Timestamp(e["commence_time"]).tz_convert("America/Toronto")
        if now.date() <= start.date() <= last_day and start > now - pd.Timedelta(minutes=15):
            tonight.append(e)
    print(len(tonight), "games that haven't started" + (" in the next 7 days" if WEEK else " today"),
          f"- about {len(tonight) * 5} credits")

    # 2. the props for each game (this costs credits)
    rows = []
    for event in tonight:
        response = requests.get(f"{base_url}/events/{event['id']}/odds", params={
            "apiKey": API_KEY,
            "bookmakers": books,
            "markets": markets,
            "oddsFormat": "decimal",
        })
        if response.status_code != 200:
            print("  skipped", event["away_team"], "@", event["home_team"], "->", response.status_code, response.text[:100])
            continue
        game_day = pd.Timestamp(event["commence_time"]).tz_convert("America/Toronto").strftime("%Y-%m-%d")
        game_rows = [r | {"game_date": game_day} for r in flatten(response.json())]   # which day each price is for
        rows += game_rows
        print(" ", event["away_team"], "@", event["home_team"], "->", len(game_rows), "prices | credits left:",
              response.headers.get("x-requests-remaining"))
        time.sleep(1)   # be polite to the server

    odds = pd.DataFrame(rows)
    if odds.empty:
        raise SystemExit("No props posted yet. Books usually post NHL props a few hours before puck drop.")

    # the book's chance = 1 / decimal odds (this still includes the book's cut, so it's a bit high)
    odds["book_chance"] = (1 / odds["price"]).round(3)
    odds.to_csv("odds.csv", index=False)
    print(len(odds), "prices saved to odds.csv")

    # 3. best price for each bet across the 3 books (higher decimal odds = bigger payout)
    best = odds.sort_values("price", ascending=False).drop_duplicates(["player", "market", "side", "line"])
    print("\nBest price per bet (first 15):")
    print(best[["player", "market", "side", "line", "book", "price"]].head(15).to_string(index=False))
