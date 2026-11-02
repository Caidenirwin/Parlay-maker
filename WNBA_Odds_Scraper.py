import os
import sys
import time
import requests
import pandas as pd
from Odds_Scraper import flatten
from Freshness import already_fresh, football_game_starts

# WNBA odds from FanDuel, DraftKings and BetMGM (The Odds API) - tonight's games only, like hockey.
# A busy night has ~6 games - props for all of them would cost ~24 credits - so:
#   moneylines: ALL of tonight's games, in one request (~1 credit)
#   player props: only the MAX_PROP_GAMES closest games, ~4 credits each
#   (close game = starters play the whole 4th quarter; in a blowout they sit, and their props die)
# Key: WNBA_ODDS_API_KEY if you set one, else your hockey key (they'd share credits).
# This week: run with --week (the dashboard's "This week" switch) for every game in the next 7 days -
#   moneylines are still one request, but props cost ~4 credits per game, so a week is ~100-200 credits.

API_KEY = os.getenv("WNBA_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
base_url = "https://api.the-odds-api.com/v4/sports/basketball_wnba"
books = "fanduel,draftkings,betmgm"
prop_markets = "player_points,player_rebounds,player_assists,player_threes"
MAX_PROP_GAMES = 4    # player props for at most this many games - raise it if you have credits to spare

WEEK = "--week" in sys.argv   # every game in the next 7 days

if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit('No API key. Get a free key at the-odds-api.com, then run: setx ODDS_API_KEY "your-key" '
                         "and restart VS Code.")

    # already downloaded today and no game has started since? reuse it - saves credits
    fresh = already_fresh("wnba_odds.csv", football_game_starts("wnba"))
    if fresh and WEEK and pd.read_csv("wnba_odds.csv")["game_date"].nunique() < 2:
        print("...but that download was tonight only - getting the whole week now")
        fresh = False
    if fresh:
        raise SystemExit(0)

    now = pd.Timestamp.now(tz="America/Toronto")
    day_of = lambda t: pd.Timestamp(t).tz_convert("America/Toronto").strftime("%Y-%m-%d")

    # 0. which days? (this list is free - don't pay a credit to find out there's nothing)
    #    normal: tonight's games - or, with none left tonight, tomorrow's
    #    --week (dashboard "This week"): every game in the next 7 days
    listed = [e for e in requests.get(f"{base_url}/events", params={"apiKey": API_KEY}).json()
              if pd.Timestamp(e["commence_time"]) > now]
    tomorrow = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    week_end = (now + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    days = sorted({day_of(e["commence_time"]) for e in listed})
    next_day = f" Next: {pd.Timestamp(days[0]).strftime('%a %b %d')}." if days else " (off-season)"
    if WEEK:
        game_days = [d for d in days if d <= week_end]
        if not game_days:
            raise SystemExit("No WNBA games with odds in the next 7 days." + next_day)
        print(f"This week: {len(game_days)} game days ({', '.join(pd.Timestamp(d).strftime('%a') for d in game_days)})")
    else:
        if not days or days[0] > tomorrow:
            raise SystemExit("No WNBA games with odds tonight or tomorrow." + next_day)
        game_days = days[:1]
        if game_days[0] != now.strftime("%Y-%m-%d"):
            print(f"No games left tonight - getting odds for {pd.Timestamp(game_days[0]).strftime('%A %b %d')} instead")
    wanted = lambda t: now < pd.Timestamp(t) and day_of(t) in game_days

    # 1. moneylines for every game - one request, however many days
    response = requests.get(f"{base_url}/odds", params={"apiKey": API_KEY, "bookmakers": books, "markets": "h2h",
                                                         "oddsFormat": "decimal"})
    response.raise_for_status()
    events = [e for e in response.json() if wanted(e["commence_time"])]
    rows = []
    for e in events:
        rows += [r | {"game_date": day_of(e["commence_time"])} for r in flatten(e)]
    print(f"moneylines: {len(events)} games | credits left: {response.headers.get('x-requests-remaining')}")

    # 2. pick each day's closest games for props: the favourite's price nearest to even money
    def closeness(e):
        prices = [r["price"] for r in rows if r["game"] == f"{e['away_team']} @ {e['home_team']}"]
        return abs(1 / min(prices) - 0.5) if prices else 1   # 1/price of the favourite: 0.5 = coin flip

    prop_games = [e for d in game_days
                  for e in sorted([x for x in events if day_of(x["commence_time"]) == d], key=closeness)[:MAX_PROP_GAMES]]
    print(f"player props for {len(prop_games)} games - about {len(prop_games) * 4} credits")

    # 3. player props for those games
    for e in prop_games:
        r = requests.get(f"{base_url}/events/{e['id']}/odds", params={"apiKey": API_KEY, "bookmakers": books,
                                                                      "markets": prop_markets, "oddsFormat": "decimal"})
        if r.status_code != 200:
            print("  skipped", e["away_team"], "@", e["home_team"], "->", r.status_code, r.text[:100])
            continue
        game_rows = [x | {"game_date": day_of(e["commence_time"])} for x in flatten(r.json()) if x["market"] != "h2h"]
        rows += game_rows
        print(f"  {day_of(e['commence_time'])[5:]}  {e['away_team']} @ {e['home_team']} -> {len(game_rows)} prop prices"
              f" | credits left: {r.headers.get('x-requests-remaining')}")
        time.sleep(1)

    odds = pd.DataFrame(rows)
    if odds.empty:
        raise SystemExit("No WNBA odds for tonight - no games, or the books haven't posted yet.")
    odds["book_chance"] = (1 / odds["price"]).round(3)
    odds.to_csv("wnba_odds.csv", index=False)
    print(len(odds), "prices saved to wnba_odds.csv")
