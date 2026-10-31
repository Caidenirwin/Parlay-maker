import json
import re
import requests
import pandas as pd

# Tonight's starting goalies from Daily Faceoff (https://www.dailyfaceoff.com/starting-goalies/).
# The NHL doesn't say who starts until the game begins, but Daily Faceoff tracks it all day:
#   "Confirmed" = the team said so    "Likely" = strong reports    blank = nobody knows yet
# Their robots.txt allows this page. ML.py uses this automatically, so you don't need to run it yourself.

url = "https://www.dailyfaceoff.com/starting-goalies/"


def get_starting_goalies(date):
    """date like "2026-10-09" -> table with one row per team: team name, goalie, status"""
    response = requests.get(url + date, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()   # stop with an error if the page didn't load

    # the page has all its data inside a <script id="__NEXT_DATA__"> tag as JSON - grab it instead of reading the HTML
    raw = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', response.text, re.S).group(1)
    games = json.loads(raw)["props"]["pageProps"]["data"]

    rows = []
    for game in games:
        for side in ("home", "away"):   # each game has a home and an away goalie
            rows.append({
                "team_name": game[f"{side}TeamName"],
                "goalie": game[f"{side}GoalieName"],
                "status": game[f"{side}NewsStrengthName"] or "Unconfirmed",   # blank -> "Unconfirmed"
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    print(get_starting_goalies(today).to_string(index=False))
