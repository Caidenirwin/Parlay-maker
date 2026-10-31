import os
import time
import requests
import pandas as pd
from Odds_Scraper import flatten
from NFL_Live import clean_name
from Freshness import already_fresh, football_game_starts

# College football odds from FanDuel, DraftKings and BetMGM (The Odds API).
# A Saturday has ~50 games - player props for all of them would cost ~250 credits - so:
#   moneylines: ALL games, in one request (~1 credit)
#   player props: only the MAX_PROP_GAMES biggest games (ranked teams first), ~4 credits each
# Key: CFB_ODDS_API_KEY if you set one, else the NFL key, else your hockey key (they'd share credits).

API_KEY = os.getenv("CFB_ODDS_API_KEY") or os.getenv("NFL_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
base_url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf"
books = "fanduel,draftkings,betmgm"
prop_markets = "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions"
DAYS_AHEAD = 4        # games starting in the next 4 days (run it Saturday morning)
MAX_PROP_GAMES = 8    # player props for at most this many games - raise it if you have credits to spare

if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit('No API key. Get a free key at the-odds-api.com, then run: setx ODDS_API_KEY "your-key" '
                         "and restart VS Code.")
    now = pd.Timestamp.now(tz="America/Toronto")
    in_window = lambda t: now < pd.Timestamp(t) <= now + pd.Timedelta(days=DAYS_AHEAD)

    # already downloaded today and no game has started since? reuse it - saves credits
    if already_fresh("cfb_odds.csv", football_game_starts("cfb")):
        raise SystemExit(0)

    # 1. moneylines for every game - one request
    response = requests.get(f"{base_url}/odds", params={"apiKey": API_KEY, "bookmakers": books, "markets": "h2h",
                                                         "oddsFormat": "decimal"})
    response.raise_for_status()
    events = [e for e in response.json() if in_window(e["commence_time"])]
    rows = []
    for e in events:
        date = pd.Timestamp(e["commence_time"]).tz_convert("America/Toronto").strftime("%Y-%m-%d")
        rows += [r | {"game_date": date} for r in flatten(e)]
    print(f"moneylines: {len(events)} games | credits left: {response.headers.get('x-requests-remaining')}")

    # 2. pick the biggest games for player props: games with ranked teams first (rank 1 = best)
    # rankings come from cfb_games.csv (CFB_Scraper.py saves each team's current AP ranking)
    rank = {}
    if os.path.exists("cfb_games.csv"):
        listed = pd.read_csv("cfb_games.csv")
        for side in ("home", "away"):
            for name, r in zip(listed[f"{side}_name"], listed[f"{side}_rank"]):
                rank[clean_name(name)] = r   # latest season's rank wins (file is in date order)

    def importance(e):
        ranks = sorted(rank.get(clean_name(e[t]), 99) for t in ("home_team", "away_team"))
        return ranks[0], ranks[1]   # best-ranked team first, then the other team's rank

    prop_games = sorted(events, key=importance)[:MAX_PROP_GAMES]
    print(f"player props for {len(prop_games)} games - about {len(prop_games) * 4} credits")

    # 3. player props for those games
    for e in prop_games:
        r = requests.get(f"{base_url}/events/{e['id']}/odds", params={"apiKey": API_KEY, "bookmakers": books,
                                                                      "markets": prop_markets, "oddsFormat": "decimal"})
        if r.status_code != 200:
            print("  skipped", e["away_team"], "@", e["home_team"], "->", r.status_code, r.text[:100])
            continue
        date = pd.Timestamp(e["commence_time"]).tz_convert("America/Toronto").strftime("%Y-%m-%d")
        game_rows = [x | {"game_date": date} for x in flatten(r.json()) if x["market"] != "h2h"]
        rows += game_rows
        print(f"  {e['away_team']} @ {e['home_team']} -> {len(game_rows)} prop prices | credits left: "
              f"{r.headers.get('x-requests-remaining')}")
        time.sleep(1)

    odds = pd.DataFrame(rows)
    if odds.empty:
        raise SystemExit("No college odds in the next few days.")
    odds["book_chance"] = (1 / odds["price"]).round(3)
    odds.to_csv("cfb_odds.csv", index=False)
    print(len(odds), "prices saved to cfb_odds.csv")
