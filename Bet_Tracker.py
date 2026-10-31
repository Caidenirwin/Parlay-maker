import json
import os
import re
import unicodedata
import pandas as pd
from Live_Tracker import find_player, get_boxscore, get_scoreboard
from NFL_Live import get_nfl_box, get_nfl_scoreboard, grade_nfl_leg
from paths import csv_path

# Keeps track of the bets you actually placed (data/my_bets.csv) and every bet the model FOUND (data/found_bets.csv,
# whether you took it or not), and grades both from the NHL results that NHL_Scraper.py downloads.
# The dashboard and ML.py use this - you don't need to run it yourself.
#
# Every bet is a list of "legs": a single bet has 1 leg, a parlay has 2-3.

BETS_FILE = csv_path("my_bets.csv")
FOUND_FILE = csv_path("found_bets.csv")
COLUMNS = ["id", "placed", "game_date", "type", "book", "description", "legs", "price", "stake", "status", "profit"]


def describe_legs(legs):
    """'Dylan Larkin Over 2.5 shots + Kyle Connor anytime goal' - the same text for placed and found bets,
    so the Fumbles page can tell which found bets you took"""
    return " + ".join(f"{leg['player']} {leg['bet']}" for leg in legs)


def save_found(game_date, found):
    """save every bet the model found today. found = list of dicts with legs, book, price, stake.
    Running the model again the same day adds new finds and updates old ones instead of duplicating them."""
    old = load_bets(FOUND_FILE)
    rows = [{
        "placed": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "game_date": f.get("game_date", game_date),   # NFL bets carry their own game day (Thu / Sun / Mon)
        "type": "single" if len(f["legs"]) == 1 else f"{len(f['legs'])}-leg parlay",
        "book": f["book"],
        "description": describe_legs(f["legs"]),
        "legs": json.dumps(f["legs"], default=float),
        "price": round(float(f["price"]), 3),
        "stake": round(float(f["stake"]), 2),   # the model's suggested stake
        "status": "open",
        "profit": 0.0,
    } for f in found]
    both = pd.concat([old, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
    both = both.drop_duplicates(["game_date", "description", "book"], keep="last")   # latest price wins
    both["id"] = range(1, len(both) + 1)
    both[COLUMNS].to_csv(FOUND_FILE, index=False)


def clean_name(name):
    """'Alexis Lafrenière' -> 'alexis lafreniere', so names from different sites match"""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def american_to_decimal(american):
    """+150 -> 2.50, -120 -> 1.83"""
    american = float(american)
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def load_bets(path=BETS_FILE):
    """your placed bets (default) or, with FOUND_FILE, every bet the model found"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def add_bet(legs, book, price, stake, game_date):
    """save a bet you placed. legs = list of dicts with player, team, stat, side, line, price, bet"""
    bets = load_bets()
    new = {
        "id": int(bets["id"].max()) + 1 if len(bets) else 1,
        "placed": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "game_date": game_date,
        "type": "single" if len(legs) == 1 else f"{len(legs)}-leg parlay",
        "book": book,
        "description": describe_legs(legs),
        "legs": json.dumps(legs, default=float),
        "price": round(price, 3),
        "stake": round(stake, 2),
        "status": "open",
        "profit": 0.0,
    }
    pd.concat([bets, pd.DataFrame([new])], ignore_index=True).to_csv(BETS_FILE, index=False)
    return new["id"]   # so the dashboard can offer "Undo"


def delete_bet(bet_id):
    bets = load_bets()
    bets[bets["id"] != bet_id].to_csv(BETS_FILE, index=False)


# every finished result - "cashed out" = you took the book's early offer instead of waiting
SETTLED = ["won", "lost", "push", "cashed out"]


def cash_out_bet(bet_id, amount):
    """you cashed out early: the book paid you `amount` (CA$) back. profit = what you got - what you bet.
    The bet is closed from then on - grading skips it, since only open bets get graded."""
    bets = load_bets()
    row = bets["id"] == bet_id
    bets.loc[row, "status"] = "cashed out"
    bets.loc[row, "profit"] = round(amount - bets.loc[row, "stake"], 2)
    bets.to_csv(BETS_FILE, index=False)


def grade_leg(leg, game, box):
    """won / lost / push / void (didn't play -> money back) / open (game not over yet).
    Uses the NHL's live game feed: final stats are there the minute a game ends
    (the stats files NHL_Scraper.py downloads can lag hours behind)."""
    if game is None or game["gameState"] not in ("FINAL", "OFF"):
        return "open"                                   # not played yet, or still going
    if leg["stat"] == "moneyline":
        us = game["homeTeam"] if game["homeTeam"]["abbrev"] == leg["team"] else game["awayTeam"]
        them = game["awayTeam"] if us is game["homeTeam"] else game["homeTeam"]
        return "won" if us.get("score", 0) > them.get("score", 0) else "lost"   # final score includes OT/shootout

    player = find_player(box, leg["team"], leg["player"])
    if player is None:
        return "void"                                   # his team played but he didn't -> books refund the bet
    value = {"points": player.get("points"), "goals": player.get("goals"),
             "shots": player.get("sog"), "saves": player.get("saves")}[leg["stat"]] or 0

    if value == leg["line"]:
        return "push"                                   # landed exactly on the line -> money back
    went_over = value > leg["line"]
    wanted_over = leg["side"] in ("Over", "Yes")
    return "won" if went_over == wanted_over else "lost"


def grade_bets(path=BETS_FILE):
    """grade every open bet (in data/my_bets.csv, or data/found_bets.csv) whose games are over. returns how many got graded."""
    bets = load_bets(path)
    if bets.empty:
        return 0
    scoreboards, boxscores = {}, {}   # download each day / each game only once, even with lots of bets

    def game_and_box(date, team):
        if date not in scoreboards:
            scoreboards[date] = {}
            for g in get_scoreboard(date):
                scoreboards[date][g["homeTeam"]["abbrev"]] = g
                scoreboards[date][g["awayTeam"]["abbrev"]] = g
        game = scoreboards[date].get(team)
        if game is None or game["gameState"] not in ("FINAL", "OFF"):
            return game, None
        if game["id"] not in boxscores:
            boxscores[game["id"]] = get_boxscore(game["id"])
        return game, boxscores[game["id"]]

    nfl_scoreboards, nfl_boxes = {}, {}

    def nfl_game_and_box(date, team, league="nfl"):
        """the same as game_and_box, but for NFL / college legs (ESPN's feed)"""
        if (league, date) not in nfl_scoreboards:
            nfl_scoreboards[(league, date)] = {}
            for g in get_nfl_scoreboard(date, league):
                nfl_scoreboards[(league, date)][g["home"]["abbrev"]] = g
                nfl_scoreboards[(league, date)][g["away"]["abbrev"]] = g
        game = nfl_scoreboards[(league, date)].get(team)
        if game is None or game["state"] != "FINAL":
            return game, None
        if game["id"] not in nfl_boxes:
            nfl_boxes[game["id"]] = get_nfl_box(game["id"], league)
        return game, nfl_boxes[game["id"]]

    def grade_any(leg, bet_date):
        date = leg.get("game_date") or bet_date   # NFL legs carry their own game day
        if leg.get("sport") in ("nfl", "cfb", "nba", "wnba", "mlb"):   # ESPN's feed (every sport but the NHL)
            return grade_nfl_leg(leg, *nfl_game_and_box(date, leg["team"], leg["sport"]))
        return grade_leg(leg, *game_and_box(date, leg["team"]))

    graded = 0
    for i, bet in bets[bets["status"] == "open"].iterrows():
        legs = json.loads(bet["legs"])
        results = [grade_any(leg, bet["game_date"]) for leg in legs]
        if "lost" in results:
            status, price = "lost", 0                   # one losing leg sinks the whole parlay - no need to wait
        elif "open" in results:
            continue                                    # nothing lost yet, but some games aren't over
        else:
            # pushed / void legs drop out of the parlay; what's left pays at the remaining legs' odds
            won_legs = [leg for leg, r in zip(legs, results) if r == "won"]
            if not won_legs:
                status, price = "push", 1.0
            else:
                status = "won"
                price = bet["price"]                    # the odds you actually got
                if len(won_legs) < len(legs):           # some legs dropped out -> shrink the payout by their odds
                    for leg, r in zip(legs, results):
                        if r != "won":
                            price /= float(leg["price"])
        bets.loc[i, "status"] = status
        bets.loc[i, "profit"] = round(-bet["stake"] if status == "lost" else bet["stake"] * (price - 1), 2)
        graded += 1
    bets.to_csv(path, index=False)
    return graded
