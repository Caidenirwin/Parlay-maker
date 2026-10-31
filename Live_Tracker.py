import json
import math
import re
import unicodedata
import pandas as pd
import requests
from scipy.stats import norm, poisson

# Live scores and live player stats from the NHL (free, no key), and each bet's chance of winning RIGHT NOW.
# The dashboard uses this - you don't need to run it yourself.

LEAGUE_GOALS_PER_GAME = 3.0   # about how many goals one NHL team scores in a full game


def get_scoreboard(date):
    """every game on a date: teams, logos, score, period, clock, state (FUT / PRE / LIVE / CRIT / FINAL / OFF)"""
    return requests.get(f"https://api-web.nhle.com/v1/score/{date}", timeout=15).json().get("games", [])


def get_boxscore(game_id):
    """live player stats for one game"""
    return requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore", timeout=15).json()


def short_name(name):
    """'Auston Matthews' and 'A. Matthews' -> 'a matthews' (the live boxscore only gives the first initial)"""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    parts = [re.sub(r"[^a-z]", "", p) for p in name.split()]
    parts = [p for p in parts if p]
    return f"{parts[0][0]} {parts[-1]}" if parts else ""


def time_left(game):
    """share of the game still to play: 1.0 = not started, 0.5 = halfway, 0 = over"""
    state = game["gameState"]
    if state in ("FUT", "PRE"):
        return 1.0
    if state in ("FINAL", "OFF"):
        return 0.0
    period = game.get("period", 1)
    clock = game.get("clock", {})
    seconds_in_period = 0 if clock.get("inIntermission") else clock.get("secondsRemaining", 0)
    if period > 3:                                           # overtime: almost nothing left
        return 0.02
    return ((3 - period) * 1200 + seconds_in_period) / 3600  # 3 periods x 20 minutes = 3600 seconds


def game_status(game):
    """short text for the ticker: '7:00 PM', '2nd 06:23', '2nd INT', 'FINAL/OT'"""
    state = game["gameState"]
    if state in ("FUT", "PRE"):   # start time in Ontario time, like "7:00 PM"
        return pd.Timestamp(game["startTimeUTC"]).tz_convert("America/Toronto").strftime("%I:%M %p").lstrip("0")
    if state in ("FINAL", "OFF"):
        period_type = game.get("periodDescriptor", {}).get("periodType", "REG")
        return "FINAL" if period_type == "REG" else f"FINAL/{period_type}"
    period = game.get("period", 1)
    name = {1: "1st", 2: "2nd", 3: "3rd"}.get(period, "OT")
    clock = game.get("clock", {})
    return f"{name} INT" if clock.get("inIntermission") else f"{name} {clock.get('timeRemaining', '')}"


def find_player(box, team, player):
    """a player's live line in the boxscore (or None if he's not in the lineup)"""
    side = "homeTeam" if box["homeTeam"]["abbrev"] == team else "awayTeam"
    for group in ("forwards", "defense", "goalies"):
        for p in box["playerByGameStats"][side].get(group, []):
            if short_name(p["name"]["default"]) == short_name(player):
                return p
    return None


def live_leg(leg, games_by_team, boxscores):
    """how one bet leg is doing right now: current stat, live chance to win, and a status"""
    game = games_by_team.get(leg["team"])
    if game is None:
        return {"now": "-", "chance": None, "status": "Game not found"}
    left = time_left(game)
    over = game["gameState"] in ("FINAL", "OFF")
    started = game["gameState"] not in ("FUT", "PRE")
    mean = float(leg.get("mean") or leg["line"])   # older bets saved before 'mean' existed: rough guess
    line = float(leg["line"])

    # ---- moneyline: who's winning, and how much time is left for the other team to come back ----
    if leg["stat"] == "moneyline":
        us = game["homeTeam"] if game["homeTeam"]["abbrev"] == leg["team"] else game["awayTeam"]
        them = game["awayTeam"] if us is game["homeTeam"] else game["homeTeam"]
        mine, theirs = us.get("score", 0), them.get("score", 0)
        if over:
            chance = 1.0 if mine > theirs else 0.0
        else:
            # remaining goals for each team ~ Poisson, tilted a bit toward whoever was favoured before the game
            lam_us = LEAGUE_GOALS_PER_GAME * left * (0.8 + 0.4 * mean)
            lam_them = LEAGUE_GOALS_PER_GAME * left * (0.8 + 0.4 * (1 - mean))
            chance = 0.0
            for a in range(12):
                for b in range(12):
                    p = poisson.pmf(a, lam_us) * poisson.pmf(b, lam_them)
                    final = (mine + a) - (theirs + b)
                    chance += p if final > 0 else (p * mean if final == 0 else 0)   # tied -> overtime coin-ish flip
        now = f"{us['abbrev']} {mine}-{theirs} {them['abbrev']}" if started else "not started"
        return {"now": now, "chance": chance, "status": status_text(chance, over, started)}

    # ---- player props: his stat so far + what he'll probably add in the time left ----
    if not started:
        current = 0
    else:
        box = boxscores.get(game["id"])
        p = find_player(box, leg["team"], leg["player"]) if box else None
        if p is None:
            return {"now": "not in lineup", "chance": None, "status": "Not playing (refund)"}
        current = {"points": p.get("points"), "goals": p.get("goals"), "shots": p.get("sog"),
                   "saves": p.get("saves")}[leg["stat"]] or 0

    wants_over = leg["side"] in ("Over", "Yes")
    if leg["stat"] == "saves":
        # bell curve for the saves still to come, narrower as the game runs out
        rest_mean, rest_sd = mean * left, max(float(leg.get("sd") or 6.5) * math.sqrt(left), 1e-6)
        p_over = 1.0 if current > line else (0.0 if left == 0 else 1 - norm.cdf(line - current, rest_mean, rest_sd))
    else:
        need = math.floor(line) + 1 - current                    # how many more he needs to go over
        p_over = 1.0 if need <= 0 else 1 - poisson.cdf(need - 1, mean * left)
    chance = p_over if wants_over else 1 - p_over
    return {"now": f"{current:g} {leg['stat']} (line {line:g})", "chance": chance,
            "status": status_text(chance, over, started)}


def status_text(chance, over, started):
    if chance >= 0.999:
        return "Hit"
    if chance <= 0.001:
        return "Lost"
    if not started:
        return "Not started"
    if chance >= 0.6:
        return "On track"
    if chance >= 0.35:
        return "Toss-up"
    return "In trouble"


def live_bet(bet, games_by_team, boxscores, nfl_lookup=None):
    """every leg of a bet + the whole bet's live chance and what it's fairly worth right now.
    nfl_lookup(date, team) -> (game, box) handles NFL legs (they use ESPN's feed instead of the NHL's)"""
    from NFL_Live import live_nfl_leg   # imported here so the NHL side never depends on it

    def one(leg):
        if leg.get("sport") in ("nfl", "cfb", "nba", "wnba", "mlb") and nfl_lookup:   # ESPN legs (all but NHL)
            return live_nfl_leg(leg, *nfl_lookup(leg.get("game_date") or bet["game_date"], leg["team"], leg["sport"]))
        return live_leg(leg, games_by_team, boxscores)

    legs = [one(leg) | {"leg": leg} for leg in json.loads(bet["legs"])]
    chances = [l["chance"] for l in legs if l["chance"] is not None]   # refunded legs drop out
    chance = math.prod(chances) if chances else None                   # all legs must hit (different games)
    fair_value = bet["stake"] * bet["price"] * chance if chance is not None else None
    return {"legs": legs, "chance": chance, "fair_value": fair_value}
