import os
import sys
import pandas as pd

# "Did I already download this today, with no game starting since?" - used by every scraper so that running
# them again the same day reuses what you have instead of spending API credits / minutes downloading again.
# Add --force when you run a scraper to download fresh anyway:   python Odds_Scraper.py --force

TZ = "America/Toronto"


def already_fresh(path, game_starts):
    """True if `path` was saved TODAY and none of `game_starts` (start times) happened since then.
    Prints why it's reusing the file. Always False with --force."""
    if "--force" in sys.argv or not os.path.exists(path):
        return False
    saved = pd.Timestamp.fromtimestamp(os.path.getmtime(path)).tz_localize(TZ)
    now = pd.Timestamp.now(tz=TZ)
    if saved.date() != now.date():
        return False   # it's from another day - download again
    started_since = [t for t in game_starts if saved < pd.Timestamp(t).tz_convert(TZ) <= now]
    if started_since:
        return False   # a game has started since - things may have changed
    print(f"Reusing {path} from {saved:%I:%M %p} - no game has started since, so nothing new to download "
          f"(no credits used). Run with --force to download fresh anyway.")
    return True


def nhl_game_starts():
    """start times of today's NHL games (free NHL schedule)"""
    import requests
    today = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d")
    day = requests.get(f"https://api-web.nhle.com/v1/schedule/{today}", timeout=15).json()["gameWeek"][0]
    return [pd.Timestamp(g["startTimeUTC"]) for g in day["games"]] if day["date"] == today else []


def football_game_starts(league):
    """start times of this week's NFL ("nfl") or college ("cfb") games (free ESPN feed)"""
    from NFL_Live import get_nfl_scoreboard
    return [g["date"] for g in get_nfl_scoreboard(None, league)]
