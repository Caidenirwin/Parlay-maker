import math
import re
import unicodedata
import pandas as pd
import requests
from scipy.stats import norm, poisson

# Live NFL, college football, NBA, WNBA and MLB scores and player stats from ESPN's free feed - used to grade bets,
# track them mid-game, and fill the score ticker. The dashboard and Bet_Tracker use this; you don't run it yourself.
# league = "nfl" (default), "cfb" (college football), "nba", "wnba" or "mlb" - they all use ESPN's format.

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
LEAGUE_URL = {"nfl": ESPN, "cfb": "https://site.api.espn.com/apis/site/v2/sports/football/college-football",
              "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba",
              "wnba": "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba",
              "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"}
# college: only FBS games (groups=80), and up to 300 of them in one request
LEAGUE_PARAMS = {"nfl": {}, "cfb": {"groups": 80, "limit": 300}, "nba": {}, "wnba": {}, "mlb": {}}
# football quarters are 15 min, NBA 12, WNBA 10
QUARTER_SECONDS = {"nfl": 900, "cfb": 900, "nba": 720, "wnba": 600}
COUNT_STATS = {"receptions", "rebounds", "assists", "threes", "hits", "home_runs", "rbis", "strikeouts"}   # whole-number counts -> Poisson; yards / points -> bell curve
# ESPN and nflverse spell two teams differently - always convert ESPN's to nflverse's
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
NFLVERSE_TO_ESPN = {v: k for k, v in ESPN_TO_NFLVERSE.items()}
# how much a final score margin typically varies (points / runs) - used for live moneyline win chances
MARGIN_SPREAD = {"nfl": 13.5, "cfb": 16.0, "nba": 12.5, "wnba": 11.0, "mlb": 4.4}
# games that were never played on their date - ESPN keeps them on the schedule with one of these labels
VOID_STATUSES = {"STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED"}


def clean_name(name):
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def num(value):
    """ESPN's text stat -> whole number ("--" or blank means 0)"""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def nfl_logo(team):
    """nflverse abbreviation -> ESPN's team logo"""
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{NFLVERSE_TO_ESPN.get(team, team).lower()}.png"


def get_nfl_scoreboard(date=None, league="nfl"):
    """NFL games on a date ("2026-10-18"), or this week's games if no date. Each game is a simple dict:
    id, state (FUT / LIVE / FINAL), status_text, left (share of the game still to play), home and away
    teams with abbrev (nflverse style), score, logo_url and name."""
    params = dict(LEAGUE_PARAMS[league], **({"dates": date.replace("-", "")} if date else {}))
    events = requests.get(f"{LEAGUE_URL[league]}/scoreboard", params=params, timeout=15).json().get("events", [])
    games = []
    for e in events:
        status = e["status"]
        state = {"pre": "FUT", "in": "LIVE", "post": "FINAL"}.get(status["type"]["state"], "FUT")
        if status["type"].get("name") in VOID_STATUSES:
            state = "VOID"   # postponed (rain-out) / cancelled: never played on this date -> the books void bets
        elif state == "FINAL" and not status["type"].get("completed", True):
            state = "LIVE"   # suspended mid-game: it gets finished another day, so bets stay open until then
        game = {"id": e["id"], "state": state, "date": pd.Timestamp(e["date"]).tz_convert("America/Toronto")}
        for t in e["competitions"][0]["competitors"]:
            if league == "nfl":
                abbrev = ESPN_TO_NFLVERSE.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
                logo_url = nfl_logo(abbrev)
            else:   # college: ESPN's own abbreviation and logo
                abbrev = t["team"].get("abbreviation", t["team"]["id"])
                logo_url = t["team"].get("logo", "")
            game[t["homeAway"]] = {"abbrev": abbrev, "score": int(t.get("score") or 0),
                                   "logo_url": logo_url, "name": t["team"]["displayName"]}
        if state == "FUT":
            game["status_text"] = game["date"].strftime("%a %I:%M %p").replace(" 0", " ")
            game["left"] = 1.0
        elif state == "FINAL":
            game["status_text"] = status["type"].get("shortDetail", "Final").replace("Final/OT", "FINAL/OT").replace("Final", "FINAL")
            game["left"] = 0.0
        elif state == "VOID":
            game["status_text"] = status["type"].get("description", "Postponed")   # "Postponed" / "Canceled"
            game["left"] = 0.0
        elif league == "mlb":   # baseball: innings, not a clock. "Top 5th" = 4 innings done + a bit
            inning = status.get("period", 1)
            detail = status["type"].get("shortDetail", "")
            part = {"Top": 0.25, "Mid": 0.5, "Bot": 0.75, "End": 1.0}.get(detail.split(" ")[0], 0.5)
            game["status_text"] = detail
            game["left"] = max(0.02, (9 - (inning - 1 + part)) / 9)   # extra innings -> almost over
        else:
            quarter = status.get("period", 1)
            clock = status.get("displayClock", "0:00")
            minutes, seconds = (clock.split(":") + ["0"])[:2] if ":" in clock else ("0", clock)
            clock_left = int(float(minutes)) * 60 + int(float(seconds))
            q = QUARTER_SECONDS[league]
            game["status_text"] = f"Q{quarter} {clock}" if quarter <= 4 else "OT"
            game["left"] = 0.02 if quarter > 4 else ((4 - quarter) * q + clock_left) / (4 * q)
        game["preseason"] = e.get("season", {}).get("type") == 1   # exhibition games: no betting odds, stars sit
        if game["preseason"] and state == "FUT":
            game["status_text"] = "Preseason · " + game["status_text"]
        games.append(game)
    return games


def get_nfl_box(event_id, league="nfl"):
    """everyone's stats in one game: clean player name -> passing_yards, rushing_yards, receiving_yards, receptions
    (NBA: points, rebounds, assists, threes)"""
    data = requests.get(f"{LEAGUE_URL[league]}/summary", params={"event": event_id}, timeout=15).json()
    players = {}
    if league in ("nba", "wnba"):   # basketball: one stat group per team, players who sat out are marked didNotPlay
        for team in data.get("boxscore", {}).get("players", []):
            group = team["statistics"][0]
            for athlete in group["athletes"]:
                if athlete.get("didNotPlay") or not athlete["stats"]:
                    continue
                s = dict(zip(group["keys"], athlete["stats"]))
                made3 = s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "0-0").split("-")[0]
                players[clean_name(athlete["athlete"]["displayName"])] = {
                    "points": num(s.get("points")), "rebounds": num(s.get("rebounds")),
                    "assists": num(s.get("assists")), "threes": num(made3)}
        return players
    if league == "mlb":   # baseball: a batting group (hits, homers, RBIs) and a pitching group (strikeouts)
        for team in data.get("boxscore", {}).get("players", []):
            for group in team["statistics"]:
                kind = group.get("type")
                for athlete in group["athletes"]:
                    if not athlete.get("stats"):
                        continue
                    s = dict(zip(group["keys"], athlete["stats"]))
                    row = players.setdefault(clean_name(athlete["athlete"]["displayName"]),
                                             {"hits": 0, "home_runs": 0, "rbis": 0, "strikeouts": 0})
                    if kind == "batting":
                        row.update(hits=num(s.get("hits")), home_runs=num(s.get("homeRuns")), rbis=num(s.get("RBIs")))
                    elif kind == "pitching":   # "strikeouts" bets are the PITCHER's strikeouts
                        row["strikeouts"] = num(s.get("strikeouts"))
        return players
    wanted = {("passing", "passingYards"): "passing_yards", ("rushing", "rushingYards"): "rushing_yards",
              ("receiving", "receivingYards"): "receiving_yards", ("receiving", "receptions"): "receptions"}
    for team in data.get("boxscore", {}).get("players", []):
        for category in team["statistics"]:
            for athlete in category["athletes"]:
                key = clean_name(athlete["athlete"]["displayName"])
                row = players.setdefault(key, {"passing_yards": 0, "rushing_yards": 0, "receiving_yards": 0,
                                               "receptions": 0})
                for (cat, stat_key), column in wanted.items():
                    if category["name"] == cat and stat_key in category["keys"]:
                        value = athlete["stats"][category["keys"].index(stat_key)]
                        row[column] = int(float(value)) if value not in ("", "--") else 0
    return players


def team_of(game, team):
    us = game["home"] if game["home"]["abbrev"] == team else game["away"]
    them = game["away"] if us is game["home"] else game["home"]
    return us, them


def grade_nfl_leg(leg, game, box):
    """won / lost / push / void (didn't play, or the game was postponed) / open (not over yet)"""
    if game is not None and game["state"] == "VOID":
        return "void"   # postponed / cancelled -> refunded (in a parlay the leg just drops out)
    if game is None or game["state"] != "FINAL":
        return "open"
    if leg["stat"] == "moneyline":
        us, them = team_of(game, leg["team"])
        return "won" if us["score"] > them["score"] else ("push" if us["score"] == them["score"] else "lost")
    player = box.get(clean_name(leg["player"]))
    if player is None:
        return "void"
    value = player[leg["stat"]]
    if value == leg["line"]:
        return "push"
    return "won" if (value > leg["line"]) == (leg["side"] in ("Over", "Yes")) else "lost"


def status_text(chance, over, started):
    if chance >= 0.999:
        return "Hit"
    if chance <= 0.001:
        return "Lost"
    if not started:
        return "Not started"
    return "On track" if chance >= 0.6 else ("Toss-up" if chance >= 0.35 else "In trouble")


def live_nfl_leg(leg, game, box):
    """how an NFL leg is doing right now: current stat, live chance, status"""
    if game is None:
        return {"now": "-", "chance": None, "status": "Game not found"}
    if game["state"] == "VOID":
        return {"now": game["status_text"].lower(), "chance": None, "status": f"{game['status_text']} - void (refund)"}
    left, started, over = game["left"], game["state"] != "FUT", game["state"] == "FINAL"
    mean = float(leg.get("mean") or leg["line"])

    if leg["stat"] == "moneyline":
        us, them = team_of(game, leg["team"])
        diff = us["score"] - them["score"]
        if over:
            chance = 1.0 if diff > 0 else 0.0
        else:
            # the rest of the game's margin ~ bell curve, centred on what the pre-game win chance implied
            margin_spread = MARGIN_SPREAD.get(leg.get("sport"), 13.5)
            expected = norm.ppf(min(max(mean, 0.01), 0.99)) * margin_spread * left
            spread = max(margin_spread * math.sqrt(left), 1e-6)
            chance = 1 - norm.cdf(-diff, expected, spread)
        now = f"{us['abbrev']} {us['score']}-{them['score']} {them['abbrev']}" if started else "not started"
        return {"now": now, "chance": chance, "status": status_text(chance, over, started)}

    current = 0
    if started:
        player = box.get(clean_name(leg["player"])) if box else None
        if player is None:
            if over:
                return {"now": "didn't play", "chance": None, "status": "Not playing (refund)"}
        else:
            current = player[leg["stat"]]
    line = float(leg["line"])
    if leg["stat"] in COUNT_STATS:
        need = math.floor(line) + 1 - current
        p_over = 1.0 if need <= 0 else 1 - poisson.cdf(need - 1, mean * left)
    else:   # yards: bell curve for the yards still to come, narrower as the game runs out
        sd = max(float(leg.get("sd") or 25) * math.sqrt(left), 1e-6)
        p_over = 1.0 if current > line else (0.0 if left == 0 else 1 - norm.cdf(line - current, mean * left, sd))
    chance = p_over if leg["side"] in ("Over", "Yes") else 1 - p_over
    label = leg["stat"].replace("_", " ")
    return {"now": f"{current:g} {label} (line {line:g})", "chance": chance, "status": status_text(chance, over, started)}
