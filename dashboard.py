import base64
import copy
import html
import itertools
from collections import Counter
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import altair as alt   # chart library that comes with Streamlit - used for the profit/loss pop-up
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import SGDClassifier   # the baccarat AI (learns one hand at a time)
from Bet_Tracker import (FOUND_FILE, SETTLED, add_bet, american_to_decimal, cash_out_bet, delete_bet, grade_bets,
                         load_bets)
from Live_Tracker import game_status, get_boxscore, get_scoreboard, live_bet
from NFL_Live import clean_name, get_nfl_box, get_nfl_scoreboard, nfl_logo

# Start it with:   streamlit run dashboard.py
# The colours and font live in .streamlit/config.toml

st.set_page_config(page_title="Parlay Quant", page_icon=":material/sports_hockey:", layout="wide")
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # so the scripts and csv files are always found
TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")
LEG_COLUMNS = ["player", "team", "game", "stat", "side", "line", "bet", "price", "mean", "sd"]
EXTRA_LEG_COLUMNS = ["sport", "game_date"]   # NFL legs carry these (NHL legs default to nhl / today)
GREEN, RED, AMBER, BLUE, MUTED = "#22C55E", "#EF4444", "#F59E0B", "#3B82F6", "#94A3B8"
BLUE = "#FFFFFF" if globals().get("PQ_THEME", "v2") != "classic" else BLUE   # 2.0: white chart lines


# ---------------------------------------------------------------- small helpers
def american(decimal_price):
    """2.50 -> +150, 1.83 -> -120"""
    if decimal_price >= 2:
        return f"{(decimal_price - 1) * 100:+.0f}"
    return f"{-100 / (decimal_price - 1):+.0f}"


def cad(amount):
    """12.5 -> 'CA$12.50', -5 -> '-CA$5.00' (Canadian dollars)"""
    return f"{'-' if amount < 0 else ''}CA${abs(amount):,.2f}"


def md_cad(amount):
    """cad() for markdown text: in markdown two $ signs mean 'math formula', so escape them as \\$"""
    return cad(amount).replace("$", "\\$")


@st.cache_data(ttl=3600, show_spinner=False)
def team_logos(sport):
    """college / NBA team abbreviation -> ESPN logo (CFB_Scraper.py / NBA_Scraper.py save the list)"""
    if not os.path.exists(f"{sport}_teams.csv"):
        return {}
    teams = pd.read_csv(f"{sport}_teams.csv")
    return dict(zip(teams["team"], teams["logo"]))


def logo(team, sport="nhl"):
    """a team's official logo: the NHL's (made for dark backgrounds), or ESPN's for NFL, college and NBA teams"""
    if str(sport).lower() == "nfl":
        return nfl_logo(team)
    if str(sport).lower() in ("cfb", "nba", "wnba", "mlb"):
        return team_logos(str(sport).lower()).get(team, "")
    return f"https://assets.nhle.com/logos/nhl/svg/{team}_dark.svg"


# player photos: each sport's stats file -> (id column, name column) and the photo address built from the id
PHOTO_SOURCES = {
    "nhl": ([("skaters.csv", "playerId", "skaterFullName"), ("goalies.csv", "playerId", "goalieFullName")],
            "https://assets.nhle.com/mugs/nhl/latest/{}.png"),
    "nba": ([("nba_players.csv", "player_id", "player")], "https://a.espncdn.com/i/headshots/nba/players/full/{}.png"),
    "wnba": ([("wnba_players.csv", "player_id", "player")], "https://a.espncdn.com/i/headshots/wnba/players/full/{}.png"),
    "mlb": ([("mlb_players.csv", "player_id", "player")], "https://a.espncdn.com/i/headshots/mlb/players/full/{}.png"),
    "cfb": ([("cfb_players.csv", "player_id", "player")],
            "https://a.espncdn.com/i/headshots/college-football/players/full/{}.png"),
}


@st.cache_data(ttl=86400, show_spinner=False)
def photo_index(sport):
    """clean player name -> photo address, for one sport (worked out once a day)"""
    try:
        if sport == "nfl":   # NFL stats use nflverse ids - nflverse's player list has the photo links
            players = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/players/players.csv",
                                  usecols=["display_name", "headshot"]).dropna()
            return {clean_name(n): u for n, u in zip(players["display_name"], players["headshot"])}
        files, address = PHOTO_SOURCES[sport]
        index = {}
        for path, id_col, name_col in files:
            if os.path.exists(path):
                ids = pd.read_csv(path, usecols=[id_col, name_col]).drop_duplicates(name_col, keep="last")
                index |= {clean_name(n): address.format(int(i)) for n, i in zip(ids[name_col], ids[id_col])}
        return index
    except Exception:
        return {}   # no photos is fine - the card shows the team logo instead


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def photo_exists(url):
    """does this photo really exist? (some players - lots of college ones - have none) - checked once a week"""
    try:
        return requests.head(url, timeout=4, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


def player_photo(name, sport):
    """the player's photo address, or None (moneylines, unknown players, no photo)"""
    url = photo_index(str(sport).lower()).get(clean_name(name)) if name else None
    return url if url and photo_exists(url) else None


def face(player, team, sport, size=48):
    """the player's photo with his team's logo in the corner - or just the team logo when there's no photo"""
    photo = player_photo(player, sport)
    if not photo:
        return f'<img src="{logo(team, sport)}" style="width:{size - 4}px;height:{size - 4}px;flex-shrink:0">'
    badge = max(16, size * 9 // 20)
    return (f'<span style="position:relative;width:{size}px;height:{size}px;flex-shrink:0;display:inline-block">'
            f'<img src="{photo}" style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;'
            f'background:#141414;border:1px solid #262626">'
            f'<img src="{logo(team, sport)}" style="position:absolute;right:-3px;bottom:-3px;width:{badge}px;'
            f'height:{badge}px;border-radius:50%;background:#000;padding:2px;border:1px solid #262626"></span>')


def leg_dict(row, extra=()):
    """the columns a saved bet leg needs, from a row of a bets / lines file"""
    return {c: row[c] for c in LEG_COLUMNS + EXTRA_LEG_COLUMNS + list(extra) if c in row.index and pd.notna(row[c])}


def updated_at(path):
    return pd.Timestamp.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None


def made_today(path, days=0):
    """made today - or, for NFL files (a week of games), within the last `days` days"""
    when = updated_at(path)
    return when is not None and (pd.Timestamp.today().normalize() - when.normalize()).days <= days


def run_script(script, label, force=False, args=()):
    """run one of your scripts like typing 'python script' in the terminal, and show what it printed.
    force=True adds --force: download fresh even if it already did today. args: extra options (e.g. --week)"""
    with st.status(label + "...", expanded=False) as box:
        command = [sys.executable, script, *args] + (["--force"] if force else [])
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        st.code((result.stdout + result.stderr)[-5000:] or "(nothing printed)")
        ok = result.returncode == 0
        box.update(label=f"{label}: {'done' if ok else 'stopped - open for details'}",
                   state="complete" if ok else "error")
    return ok


# ---------------------------------------------------------------- background jobs (so the page never freezes)
# The Tonight tab's buttons start their scripts in the BACKGROUND: the page stays usable (blackjack!) while they
# run, a progress box updates every 2 seconds, and the whole page refreshes with the new bets when they finish.
def start_job(plan):
    """plan = [(script, label, extra options, force), ...] - run them one after another in the background"""
    job = {"steps": [{"label": label, "state": "waiting", "output": ""} for _, label, _, _ in plan], "done": False}

    def work():   # runs on its own thread - only touches the job dict, never Streamlit itself
        for step, (script, label, args, force) in zip(job["steps"], plan):
            step["state"] = "running"
            command = [sys.executable, script, *args] + (["--force"] if force else [])
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            step["output"] = (result.stdout + result.stderr)[-5000:] or "(nothing printed)"
            step["state"] = "complete" if result.returncode == 0 else "error"
            if result.returncode != 0:
                break   # stop if a step failed (for example, no props posted yet)
        job["done"] = True

    threading.Thread(target=work, daemon=True).start()
    st.session_state.job = job
    st.rerun()   # show the buttons as busy right away


def job_running():
    job = st.session_state.get("job")
    return bool(job) and not job["done"]


def show_job():
    """one box per step: waiting / running / done / stopped (open it to see what the script printed)"""
    job = st.session_state.job
    for step in job["steps"]:
        if step["state"] == "waiting":
            st.caption(f":material/schedule: {step['label']} - waiting")
            continue
        label = {"running": f"{step['label']}...", "complete": f"{step['label']}: done",
                 "error": f"{step['label']}: stopped - open for details"}[step["state"]]
        with st.status(label, state=step["state"], expanded=False):
            st.code(step["output"] or "working...")


@st.fragment(run_every=2)
def job_progress_live():
    """refreshes every 2 seconds while the job runs - and the whole page once it's finished (new bets)"""
    show_job()
    if st.session_state.job["done"]:
        st.rerun()


def job_progress():
    if job_running():
        job_progress_live()
    else:
        show_job()


# ---------------------------------------------------------------- blackjack (fake chips - no real money, just for fun)
BJ_SUITS = {"♠": False, "♥": True, "♦": True, "♣": False}   # suit -> red?
BJ_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def bj_shoe():
    """a fresh 6-deck shoe, shuffled - like a real casino table"""
    shoe = [(rank, suit) for _ in range(6) for rank in BJ_RANKS for suit in BJ_SUITS]
    random.shuffle(shoe)
    return shoe


def bj_total(hand):
    """best total: aces count 11 unless that busts, then 1"""
    total = sum(11 if r == "A" else 10 if r in ("10", "J", "Q", "K") else int(r) for r, _ in hand)
    aces = sum(r == "A" for r, _ in hand)
    while total > 21 and aces:
        total, aces = total - 10, aces - 1
    return total


def bj_game():
    return st.session_state.setdefault("bj", {"chips": 1000, "bet": 25, "stake": 0, "shoe": bj_shoe(), "player": [],
                                              "dealer": [], "phase": "bet", "msg": "", "record": [0, 0, 0]})


def bj_draw(g):
    if len(g["shoe"]) < 52:   # cut card: shuffle a new shoe when it runs low
        g["shoe"] = bj_shoe()
        g["used"] = []        # a fresh shoe - the card count starts again at 0
        g["note"] = "New shoe shuffled - the count starts again at 0."
    card = g["shoe"].pop()
    g.setdefault("used", []).append(card)   # every card out of this shoe (for the card-counting trainer)
    return card


# ---- card-counting trainer: the Hi-Lo system (for learning only - see the README) ----
def hilo(card):
    """Hi-Lo value of a card: low cards (2-6) +1, middle (7-9) 0, tens and aces -1"""
    rank = card[0]
    return 1 if rank in ("2", "3", "4", "5", "6") else -1 if rank in ("10", "J", "Q", "K", "A") else 0


def bj_count(g):
    """(running count, true count, decks left) for every card you've SEEN from this shoe.
    The dealer's face-down card doesn't count until it's turned over - just like at a real table."""
    running = sum(hilo(c) for c in g.get("used", []))
    if g["phase"] == "play" and len(g["dealer"]) > 1:
        running -= hilo(g["dealer"][1])   # still face down
    decks_left = max(len(g["shoe"]) / 52, 0.5)
    return running, running / decks_left, decks_left


def bj_finish(g, msg=None):
    """dealer plays out (stands on 17), then every hand is paid (more than one after a split)"""
    hands = g.get("hands") or [{"cards": g["player"], "stake": g["stake"]}]
    split = len(hands) > 1
    natural = not split and len(hands[0]["cards"]) == 2 and bj_total(hands[0]["cards"]) == 21
    dealer_natural = len(g["dealer"]) == 2 and bj_total(g["dealer"]) == 21
    if any(bj_total(h["cards"]) <= 21 for h in hands) and not natural:
        while bj_total(g["dealer"]) < 17:
            g["dealer"].append(bj_draw(g))
    d = bj_total(g["dealer"])
    paid, staked, texts = 0, 0, []
    for i, h in enumerate(hands):
        p, stake = bj_total(h["cards"]), h["stake"]
        if p > 21:
            pay, text = 0, f"Bust with {p} - you lose {stake:,}."
        elif natural and not dealer_natural:
            pay, text = stake * 2.5, f"Blackjack! You win {stake * 1.5:,.0f} (pays 3 to 2)."
        elif dealer_natural and not natural:
            pay, text = 0, f"Dealer has blackjack - you lose {stake:,}."
        elif d > 21 or p > d:
            pay, text = stake * 2, f"You win {stake:,}! " + ("Dealer busts." if d > 21 else f"{p} beats {d}.")
        elif p == d:
            pay, text = stake, f"Push - {p} each, you get your {stake:,} back."
        else:
            pay, text = 0, f"Dealer wins, {d} to {p} - you lose {stake:,}."
        paid, staked = paid + pay, staked + stake
        texts.append(f"Hand {i + 1}: {text}" if split else text)
    net = paid - staked
    result = 0 if net > 0 else 1 if net < 0 else 2
    text = " ".join(texts) + (f" Overall: {net:+,.0f}." if split else "")
    g["chips"] += int(paid)
    g["record"][result] += 1
    g["hands_played"] = g.get("hands_played", 0) + 1   # the counting trainer's quiz asks every few hands
    followed = g.pop("hand_followed", False)
    track = g.setdefault("ai_track", {"followed": 0, "total": 0})
    if followed:    # you followed the AI on every move this hand -> your record when following it
        track.setdefault("record", [0, 0, 0])[result] += 1
    if followed or "ai_start" in g:   # the AI's own record: same as yours if you followed it, else replayed its way
        track.setdefault("ai_record", [0, 0, 0])[result if followed else bj_ai_replay(g)] += 1
    g["phase"], g["msg"] = "done", msg or text


def bj_ai_replay(g):
    """the AI's own result on this hand: the same cards from the same shoe, but every move is its advice
    (it still only knows what you knew - your cards, the dealer's up card and the unseen pile)"""
    player, dealer, shoe = (list(x) for x in g["ai_start"])
    trained = bj_coach_trained()   # the coach you picked: your trained bot, or the simulation AI

    def choose(cards, can_double, can_split):
        spot = bj_trained_move(g, cards, can_double, can_split) if trained else None
        if spot:
            return spot["best"]
        h = {"player": cards, "dealer": dealer, "shoe": shoe, "stake": 1}
        return bj_ai_move(h, can_double, can_split)["best"]

    result, _ = bj_round(player, dealer, shoe.pop, choose)
    return 0 if result > 0 else 1 if result < 0 else 2


def bj_pair(cards):
    """two cards of the same value (any two 10-value cards count) - a hand you can split"""
    return len(cards) == 2 and bj_up_value(cards[0]) == bj_up_value(cards[1])


def bj_round(player, dealer, draw, choose):
    """one round after the deal (no blackjacks), with splitting: choose(cards, can_double, can_split) -> "H" / "S" /
    "D" / "P" plays every hand. Split up to 4 hands, double after splitting, split aces get one card each.
    Returns (chips won or lost, chips bet) for a 1-chip starting bet."""
    hands, i = [{"cards": list(player), "bet": 1, "aces": False}], 0
    while i < len(hands):
        h = hands[i]
        if len(h["cards"]) == 1:   # a hand from a split gets its 2nd card when its turn comes
            h["cards"].append(draw())
        while not h["aces"] and bj_total(h["cards"]) < 21:
            cards = h["cards"]
            can_double = len(cards) == 2
            can_split = bj_pair(cards) and len(hands) < 4
            move = choose(cards, can_double, can_split)
            if move == "P" and can_split:
                aces = cards[0][0] == "A"
                hands.insert(i + 1, {"cards": [cards.pop()], "bet": 1, "aces": aces})
                cards.append(draw())
                h["aces"] = aces
                continue
            if move == "S":
                break
            cards.append(draw())
            if move == "D" and can_double:
                h["bet"] = 2
                break
        i += 1
    if any(bj_total(h["cards"]) <= 21 for h in hands):
        while bj_total(dealer) < 17:
            dealer.append(draw())
    d = bj_total(dealer)
    result = 0
    for h in hands:
        p = bj_total(h["cards"])
        result += -h["bet"] if p > 21 else h["bet"] if d > 21 or p > d else 0 if p == d else -h["bet"]
    return result, sum(h["bet"] for h in hands)


def bj_add_chip(value):
    """drop a chip into the betting circle"""
    g = bj_game()
    g["bet"] = (g.get("bet") or 0) + value
    g["drop"] = True


def bj_clear():
    bj_game()["bet"] = 0


def bj_deal():
    g = bj_game()
    if g["bet"] < 10:
        g["msg"] = "Put some chips in the betting circle first."
        return
    if g["bet"] > g["chips"]:
        g["msg"] = "Not enough chips for that bet - pick a smaller one."
        return
    g["chips"] -= g["bet"]
    g["stake"] = g["bet"]
    # dealt like a real table: you, dealer, you, dealer
    first, second, third, fourth = (bj_draw(g) for _ in range(4))
    g["player"], g["dealer"] = [first, third], [second, fourth]
    # every hand you're playing (more than one after a split); g["player"] is always the one you're on now
    g["hands"], g["active"] = [{"cards": g["player"], "stake": g["stake"], "aces": False}], 0
    g["ai_start"] = (list(g["player"]), list(g["dealer"]), list(g["shoe"]))   # so the AI can play this hand its own way
    g["seen"] = {"player": 0, "dealer": 0, "hole": False}   # nothing on the table has been shown yet -> animate it all
    g["hand_followed"] = True   # until you go against the AI's advice this hand
    g["phase"], g["msg"] = "play", ""
    g["quiz_msg"] = None   # the last quiz answer goes away with the new hand
    if bj_total(g["player"]) == 21 or bj_total(g["dealer"]) == 21:
        bj_finish(g)   # a blackjack on the deal ends the hand right away


def bj_track(move):
    """note whether your move matched the AI's advice (for the 'you followed the AI' count)"""
    g = bj_game()
    if g["phase"] != "play":
        return
    track = g.setdefault("ai_track", {"followed": 0, "total": 0})
    track["total"] += 1
    trained = bj_trained_move(g) if bj_coach_trained() else None
    adv = trained or bj_ai_move(g)
    # a move worth (almost) exactly as much as the best one counts as right - some spots are a dead heat
    followed = move == adv["best"] or adv["evs"].get(move, -9) >= adv["evs"][adv["best"]] - 0.005
    track["followed"] += followed
    soft = "soft " if bj_is_soft(g["player"]) else ""
    evs = ", ".join(f"{BJ_MOVE[m]} {ev:+.2f}" for m, ev in sorted(adv["evs"].items(), key=lambda kv: -kv[1]))
    trainer_grade("bj", followed, True,
                  f"With {soft}{bj_total(g['player'])} against a dealer {g['dealer'][0][0]}, the best play is "
                  f"**{BJ_MOVE[adv['best']]}** (you chose {BJ_MOVE[move]}). Average result per chip: {evs}"
                  + (f" - what your trained bot learned from {adv['tries']:,} tries of this exact spot at a count "
                     f"of {adv['buckets'][adv['bucket']]}." if trained else
                     f". The basic-strategy chart says {BJ_MOVE[adv['chart']]}."))
    if not followed:
        g["hand_followed"] = False   # this hand no longer counts toward "hands where you followed the AI"


def bj_next_hand(g):
    """this hand is done: move on to the next split hand (it gets its 2nd card now) - or, after the last, the dealer"""
    hands = g.get("hands") or [{"cards": g["player"], "stake": g["stake"], "aces": False}]
    g["active"] = g.get("active", 0) + 1
    while g["active"] < len(hands):
        h = hands[g["active"]]
        g["player"], g["stake"] = h["cards"], h["stake"]
        if len(h["cards"]) == 1:
            h["cards"].append(bj_draw(g))
        if not h["aces"] and bj_total(h["cards"]) < 21:
            return   # your move on this hand
        g["active"] += 1   # split aces (one card only) or 21: nothing to decide
    bj_finish(g)


def bj_can_split(g):
    return (g["phase"] == "play" and bj_pair(g["player"]) and len(g.get("hands", [])) < 4
            and g["chips"] >= g["stake"])


def bj_hit():
    bj_track("H")
    g = bj_game()
    g["player"].append(bj_draw(g))
    if bj_total(g["player"]) >= 21:
        bj_next_hand(g)


def bj_stand():
    bj_track("S")
    bj_next_hand(bj_game())


def bj_double():
    bj_track("D")
    g = bj_game()
    g["chips"] -= g["stake"]
    g["stake"] *= 2
    g["hands"][g["active"]]["stake"] = g["stake"]
    g["player"].append(bj_draw(g))   # double down: double the bet, exactly one more card
    bj_next_hand(g)


def bj_split():
    """split a pair into two hands - the same bet again on the new one; each gets a 2nd card in turn"""
    g = bj_game()
    if not bj_can_split(g):
        return
    bj_track("P")
    g["chips"] -= g["stake"]
    aces = g["player"][0][0] == "A"
    g["hands"].insert(g["active"] + 1, {"cards": [g["player"].pop()], "stake": g["stake"], "aces": aces})
    g["hands"][g["active"]]["aces"] = aces
    g["player"].append(bj_draw(g))
    if aces or bj_total(g["player"]) >= 21:   # split aces get one card each - no more moves
        bj_next_hand(g)


def bj_reset():
    st.session_state.bj = {**bj_game(), "chips": 1000, "phase": "bet", "msg": "Fresh 1,000 chips - good luck!",
                           "player": [], "dealer": [], "hands": []}


DEAL_GAP = 0.22   # seconds between cards as they come out of the shoe


def bj_cards(g, who, hide_hole=False):
    """one hand's cards. Cards you haven't seen yet slide in from the shoe (one after another);
    the dealer's face-down card flips over when it's turned up. Cards already on the table just sit still."""
    hand, seen = g[who], g.get("seen", {}).get(who, len(g[who]))
    flipping = who == "dealer" and not hide_hole and len(hand) > 1 and not g.get("seen", {}).get("hole", True)
    out, n_new = "", 0
    for i, (rank, suit) in enumerate(hand):
        if i < 2 and seen == 0:
            delay = (2 * i + (who == "dealer")) * DEAL_GAP   # the opening deal: you, dealer, you, dealer
        else:
            delay = (DEAL_GAP * 1.6 if flipping else 0) + n_new * DEAL_GAP * 1.6   # later cards, after any flip
        if i >= seen and not (i < 2 and seen == 0):
            n_new += 1
        style = f' style="animation-delay:{delay:.2f}s"'
        if hide_hole and i == 1:
            out += f'<span class="bj-card back{" new" if i >= seen else ""}"{style}></span>'
            continue
        anim = "new" if i >= seen else ("flip" if flipping and i == 1 else "")
        style = ' style="animation-delay:0s"' if anim == "flip" else (style if anim else "")   # flip first
        out += (f'<span class="bj-card {"red" if BJ_SUITS[suit] else ""} {anim}"{style}><b>{rank}</b>'
                f'<span class="bj-suit">{suit}</span></span>')
    return out


# the card table's look - shared by every card game (blackjack, baccarat, ...)
CARD_CSS = """<style>
  /* the green felt - the games' tables (keyed containers, so chips and buttons can sit ON the table) */
  .bj-felt, .st-key-bj_felt, .st-key-bac_felt, .st-key-vp_table {
             background:radial-gradient(ellipse at 50% 0%,#14532D 0%,#0B2E1A 45%,#04120A 100%);
             border:2px solid #14532D; border-radius:22px; padding:26px 32px 22px;
             box-shadow:inset 0 0 60px rgba(0,0,0,.55); overflow:hidden; position:relative; }
  .bj-mark { text-align:center; color:rgba(167,243,208,.25); font-weight:800; font-size:13px;
             letter-spacing:.25em; margin:-10px 0 16px; }
  .bj-who  { display:flex; align-items:center; gap:12px; color:#A7F3D0; font-size:13px; text-transform:uppercase;
             letter-spacing:.12em; margin:2px 0 10px; }
  .bj-who b { color:#000; background:#FFF; border-radius:999px; padding:2px 11px; font-size:15px; letter-spacing:0; }
  .bj-hand { display:flex; flex-wrap:wrap; gap:10px; min-height:112px; margin-bottom:26px; perspective:700px; }
  .bj-card { display:inline-flex; flex-direction:column; justify-content:space-between; width:76px; height:108px;
             padding:7px 9px; border-radius:10px; background:#FAFAFA; color:#111; font-size:24px;
             box-shadow:0 6px 14px rgba(0,0,0,.55); }
  .bj-card.red { color:#DC2626; }
  .bj-card .bj-suit { align-self:flex-end; font-size:30px; line-height:1; }
  .bj-card.back { background:repeating-linear-gradient(45deg,#14532D 0 8px,#166534 8px 16px);
                  border:3px solid #22C55E; }
  /* dealing: a new card slides in from the shoe (top right) with a little spin */
  @keyframes bj-deal { from { transform:translate(420px,-260px) rotate(-30deg); opacity:0; }
                       to   { transform:none; opacity:1; } }
  .bj-card.new  { animation:bj-deal .5s cubic-bezier(.2,.8,.25,1) both; }
  /* the dealer's hole card turning face up */
  @keyframes bj-flip { from { transform:rotateY(90deg); } to { transform:rotateY(0); } }
  .bj-card.flip { animation:bj-flip .4s ease-out both; }
  .bj-msg  { color:#FDE68A; font-weight:700; font-size:18px; min-height:26px; text-align:center; }
    
  /* totals and the result message fade in once the cards have landed */
  @keyframes bj-fadein { from { opacity:0; } to { opacity:1; } }
  .bj-fade { animation:bj-fadein .3s ease-out both; }
    </style>"""


# ---- casino chips: a rack to pick them from, and stacks of them sitting on the betting spots ----
CHIPS = [(10, "#2563EB"), (25, "#16A34A"), (100, "#1F1F1F"), (500, "#7C3AED")]   # value, colour (casino-style)
CHIP_CSS = "<style>" + "".join(f".st-key-chip_{game}_{value} button {{ background:{color} !important; }}"
                               for game in ("bj", "bac", "vp", "rl", "tcp", "lir", "cr") for value, color in CHIPS) + """
  [class*="st-key-chip_"] { display:flex; justify-content:center; }
  [class*="st-key-chip_"] button { width:58px !important; height:58px; min-height:58px; border-radius:50% !important;
      border:4px dashed rgba(255,255,255,.85) !important; color:#FFF !important; font-weight:800; padding:0 !important;
      box-shadow:0 4px 10px rgba(0,0,0,.55); transition:transform .12s ease; }
  [class*="st-key-chip_"] button:hover { transform:translateY(-3px); }
  [class*="st-key-chip_"] button:disabled { opacity:.3; }
  .pq-stack { position:relative; width:56px; margin:0 auto; }
  .pq-chip  { position:absolute; left:5px; width:46px; height:46px; border-radius:50%;
              border:4px dashed rgba(255,255,255,.85); box-shadow:0 3px 6px rgba(0,0,0,.6); }
  .pq-stack b { position:absolute; bottom:0; left:50%; transform:translateX(-50%); color:#FFF; white-space:nowrap;
                background:rgba(0,0,0,.6); border-radius:999px; padding:1px 9px; font-size:13px; }
  @keyframes pq-drop { from { transform:translateY(-46px); opacity:0; } to { transform:none; opacity:1; } }
  .pq-chip.drop { animation:pq-drop .28s ease-out both; }
  .pq-empty { height:70px; display:flex; align-items:center; justify-content:center; color:rgba(167,243,208,.4);
              font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
  .pq-circle { width:132px; height:132px; margin:2px auto 14px; border:2px dashed rgba(167,243,208,.45);
               border-radius:50%; display:flex; align-items:flex-end; justify-content:center; padding-bottom:8px; }
  [class*="st-key-spot_"] button { min-height:58px; border-radius:36px !important; background:rgba(0,0,0,.28) !important;
      border:2px solid rgba(167,243,208,.5) !important; color:#A7F3D0 !important; font-weight:800; letter-spacing:.1em; }
  [class*="st-key-spot_"] button:hover { background:rgba(34,197,94,.18) !important; border-color:#22C55E !important; }
</style>"""


def ai_rate_caption(record, what, how):
    """the same 'AI win rate' line in every game's AI panel. record = [won, lost, pushed]; pushes are left out"""
    w, l, p = record
    if w + l + p:
        st.caption(f"**AI win rate: {w / max(w + l, 1):.0%}** ({w}-{l}-{p} over {w + l + p} {what}) - {how}, "
                   f"on the same cards / spins you got.")


# ---- Trainer mode (every game): the AI keeps its advice to itself until you decide, then grades you ----
def trainer_on(game):
    return bool(st.session_state.get(f"trainer_{game}"))


def trainer_switch(game, what="made your move"):
    st.toggle("Trainer mode", key=f"trainer_{game}",
              help=f"The AI hides its advice until you've {what}, then grades you and explains the best play.")


def trainer_grade(game, yours, best, lesson):
    """score one decision (only while Trainer mode is on): right = the same as the AI's best play"""
    if not trainer_on(game):
        return
    t = st.session_state.setdefault("trainer", {}).setdefault(game, {"right": 0, "asked": 0, "streak": 0, "best": 0})
    ok = yours == best
    t["right"], t["asked"] = t["right"] + ok, t["asked"] + 1
    t["streak"] = t["streak"] + 1 if ok else 0
    t["best"] = max(t["best"], t["streak"])
    t["last"] = (ok, lesson)


def ai_box(headline, why):
    """the green 'The AI says' box at the top of an AI panel"""
    st.html(f'<div style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.45);'
            f'border-radius:10px;padding:10px 12px;margin:2px 0 6px">'
            f'<div style="font-size:12px;color:#A7F3D0;letter-spacing:.08em;text-transform:uppercase">'
            f'The AI says</div><div style="font-size:18px;font-weight:800;color:#FFF;margin:2px 0">{headline}</div>'
            f'<div style="font-size:12px;color:#C7C7C7;line-height:1.55">{why}</div></div>')


def trainer_quiet():
    """what the AI panel shows in Trainer mode while you're deciding"""
    st.html('<div style="border:1px dashed rgba(251,191,36,.6);border-radius:10px;padding:10px 12px;margin:2px 0 6px">'
            '<div style="font-size:12px;color:#FCD34D;letter-spacing:.08em;text-transform:uppercase">Trainer mode</div>'
            '<div style="font-size:18px;font-weight:800;color:#FFF;margin:2px 0">Your call</div>'
            '<div style="font-size:12px;color:#C7C7C7;line-height:1.55">The AI keeps its advice to itself until you '
            'decide - then it grades you and shows the best play.</div></div>')


def trainer_panel(game):
    """the grade for your last decision, and your score so far"""
    t = st.session_state.get("trainer", {}).get(game)
    if not t or not t["asked"]:
        st.caption("Trainer mode is on: make your decision and the AI will grade it.")
        return
    ok, lesson = t["last"]
    (st.success if ok else st.error)(("**Correct!** " if ok else "**Not the best play.** ") + lesson,
                                     icon=":material/check_circle:" if ok else ":material/cancel:")
    st.caption(f"Trainer score: **{t['right']} of {t['asked']}** ({t['right'] / t['asked']:.0%}) · "
               f"streak {t['streak']} · best streak {t['best']}")


def chip_stack_html(amount, drop=False, empty="Bet"):
    """the chips sitting on a betting spot - biggest chips at the bottom - with the amount underneath"""
    if amount <= 0:
        return f'<div class="pq-empty">{empty}</div>'
    pieces, left = [], amount
    for value, color in reversed(CHIPS):
        while left >= value and len(pieces) < 10:
            pieces.append(color)
            left -= value
    chips = "".join(f'<span class="pq-chip{" drop" if drop and i == len(pieces) - 1 else ""}" '
                    f'style="background:{color};bottom:{24 + i * 5}px"></span>' for i, color in enumerate(pieces))
    return f'<div class="pq-stack" style="height:{70 + 5 * (len(pieces) - 1)}px">{chips}<b>{amount:,}</b></div>'


def chip_rack(game, on_chip, can_add, selected=None):
    """the chip rack: one round chip per value. Tapping one calls on_chip(value); can_add(value) -> allowed?"""
    for col, (value, _) in zip(st.columns(len(CHIPS)), CHIPS):
        col.button(f"{value}", key=f"chip_{game}_{value}", on_click=on_chip, args=(value,), disabled=not can_add(value))
    if selected:   # baccarat: the chip you'll drop next gets a gold ring
        st.html(f"<style>.st-key-chip_{game}_{selected} button {{ box-shadow:0 0 0 4px #FBBF24, "
                f"0 4px 10px rgba(0,0,0,.55) !important; }}</style>")


# ---- the blackjack AI: what to do with your hand (simulated on the real shoe) and how much to bet (the count) ----
def bj_up_value(card):
    """the dealer's face-up card as a number: 2-10, ace = 11"""
    return 11 if card[0] == "A" else 10 if card[0] in ("10", "J", "Q", "K") else int(card[0])


def bj_is_soft(hand):
    """a soft hand has an ace still counting as 11 (A-6 = soft 17)"""
    hard = sum(1 if r == "A" else 10 if r in ("10", "J", "Q", "K") else int(r) for r, _ in hand)
    return any(r == "A" for r, _ in hand) and hard + 10 <= 21


def bj_basic(hand, up, can_double=True):
    """the standard basic-strategy chart for this table (6 decks, dealer stands on soft 17, no splitting):
    "H" hit, "S" stand, "D" double"""
    total, soft = bj_total(hand), bj_is_soft(hand)
    if soft:
        if total >= 19:
            return "S"
        if total == 18:   # soft 18: double vs 3-6 (stand there when you can't double), stand vs 2 / 7 / 8, hit vs 9-A
            return "D" if can_double and 3 <= up <= 6 else "S" if up <= 8 else "H"
        double_vs = {13: (5, 6), 14: (5, 6), 15: (4, 6), 16: (4, 6), 17: (3, 6)}.get(total)
        return "D" if can_double and double_vs and double_vs[0] <= up <= double_vs[1] else "H"
    if total >= 17:
        return "S"
    if total >= 13:
        return "S" if up <= 6 else "H"
    if total == 12:
        return "S" if 4 <= up <= 6 else "H"
    if total == 11:
        return "D" if can_double and up <= 10 else "H"
    if total == 10:
        return "D" if can_double and up <= 9 else "H"
    if total == 9:
        return "D" if can_double and 3 <= up <= 6 else "H"
    return "H"


def bj_settle(player, dealer, draw, bet=1):
    """dealer plays out (stands on 17) and the hand is scored: +bet / -bet / 0"""
    p = bj_total(player)
    if p > 21:
        return -bet
    while bj_total(dealer) < 17:
        dealer.append(draw())
    d = bj_total(dealer)
    return bet if d > 21 or p > d else 0 if p == d else -bet


@st.cache_data(max_entries=200, show_spinner=False)
def bj_simulate_moves(player, up, pool, can_double, can_split=False, n=3000):
    """expected result per chip of Hit / Stand / Double (/ Split) for THIS hand, simulated n times each from the
    cards really left (pool = the shoe + the dealer's hidden card, which you can't see). After a hit it keeps
    playing by the basic-strategy chart; after a split each hand is played by the chart."""
    rng = random.Random(str((player, up, len(pool))))   # same situation -> same answer
    draw = lambda: rng.choice(pool)
    results = {}
    if can_split:   # each half of the split: the pair card + a new card, played by the chart (split aces: 1 card)
        total, upv, card = 0, bj_up_value(up), player[0]
        aces = card[0] == "A"
        for _ in range(n):
            hole = draw()
            while bj_total([up, hole]) == 21:
                hole = draw()
            result, _ = bj_round([card, draw()], [up, hole], draw,
                                 lambda cards, dbl, spl: "S" if aces else bj_chart(cards, upv, dbl, False))
            total += result
        results["P"] = 2 * total / n   # two hands, each worth the same on average
    for move in ("S", "H", "D") if can_double else ("S", "H"):
        total = 0
        for _ in range(n):
            hole = draw()
            while bj_total([up, hole]) == 21:   # the dealer checked for blackjack already - it isn't one
                hole = draw()
            hand, bet = list(player), 1
            if move == "D":
                hand.append(draw())
                bet = 2
            elif move == "H":
                hand.append(draw())
                while bj_total(hand) < 21 and bj_basic(hand, bj_up_value(up), can_double=False) == "H":
                    hand.append(draw())
            total += bj_settle(hand, [up, hole], draw, bet)
        results[move] = total / n
    return results


BJ_MOVE = {"H": "Hit", "S": "Stand", "D": "Double down", "P": "Split"}


def bj_chart_split(card, up):
    """the textbook pair-splitting chart for this table (6 decks, double after split allowed) - up: 2-11 (ace = 11)"""
    v = bj_up_value(card)
    if v in (11, 8):
        return True        # always split aces and 8s
    if v in (10, 5):
        return False       # never split 10s (20 is great) or 5s (10 is a double)
    if v == 9:
        return up not in (7, 10, 11)
    if v == 7:
        return up <= 7
    if v == 6:
        return up <= 6
    if v == 4:
        return up in (5, 6)
    return up <= 7         # 2s and 3s


def bj_chart(hand, up, can_double=True, can_split=False):
    """the full chart: split a pair when the splitting chart says so, otherwise basic strategy"""
    return "P" if can_split and bj_chart_split(hand[0], up) else bj_basic(hand, up, can_double)

# ---- your own blackjack AI (the "blackjack training" folder): it taught itself by playing hundreds of millions of
# hands at this exact table (Monte Carlo reinforcement learning). The dashboard reads its strategy.json. ----
BJ_BOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blackjack training", "checkpoints",
                           "strategy.json")


@st.cache_data(show_spinner=False, max_entries=2)
def bj_trained_load(saved_at):
    with open(BJ_BOT_FILE) as f:
        return json.load(f)


def bj_trained():
    """your trained bot's strategy - re-read whenever training saves a new checkpoint (None if there isn't one)"""
    try:
        return bj_trained_load(os.path.getmtime(BJ_BOT_FILE))
    except (OSError, ValueError):
        return None


def bj_coach_trained():
    return bj_trained() is not None and st.session_state.get("bj_coach", "Your trained bot") == "Your trained bot"


def bj_bucket(true):
    """the bot's 5 count buckets: -2 or less / -2 to 0 / 0 to +2 / +2 to +4 / +4 or more"""
    return 0 if true <= -2 else 1 if true < 0 else 2 if true < 2 else 3 if true < 4 else 4


def bj_trained_move(g, player=None, can_double=None, can_split=None):
    """what your bot learned for this exact spot: your total, soft / hard (or the pair you can split), the dealer's
    up card, whether you can double, and the count (it only knows what you can see - never the face-down card)"""
    bot = bj_trained()
    player = player if player is not None else g["player"]
    if can_double is None:
        can_double = len(player) == 2 and bj_game()["chips"] >= g["stake"]
    if can_split is None:
        can_split = bj_can_split(g) if player is g["player"] else False
    up = 1 if g["dealer"][0][0] == "A" else min(10, bj_up_value(g["dealer"][0]))
    bucket = bj_bucket(bj_count(g)[1])
    card = 1 if player[0][0] == "A" else min(10, bj_up_value(player[0]))
    for b in (bucket, 2):   # a count it's rarely seen in this spot -> what it does at a neutral count
        key = (f"pair,{card},{up},{b}" if can_split else
               f"{bj_total(player)},{int(bj_is_soft(player))},{up},{int(can_double)},{b}")
        spot = bot["states"].get(key)
        if spot and can_split and not can_double:   # a split you can't afford to double after: drop the double
            spot = {**spot, "ev": {m: v for m, v in spot["ev"].items() if m != "D"}}
            spot["best"] = max(spot["ev"], key=spot["ev"].get)
        if spot:
            return {"best": spot["best"], "evs": spot["ev"], "tries": sum(spot["tries"].values()), "bucket": b,
                    "hands": bot["meta"]["hands"], "buckets": bot["count_buckets"]}
    return None


def bj_trained_bet(g):
    """the bet size your bot learned for the count right now (1 unit = about 1% of your chips)"""
    bot = bj_trained()
    _, true, _ = bj_count(g)
    row = bot["betting"].get(str(min(max(round(true), -6), 10)), {"units": 1, "edge": 0})
    unit = max(10, round(g["chips"] * 0.01 / 5) * 5)
    return {"true": true, "units": row["units"], "edge": row["edge"] * 100, "bet": min(row["units"] * unit, g["chips"])}


def bj_ai_move(g, can_double=None, can_split=None):
    """the AI's advice for the hand in play: the best move by simulation, plus what the chart says"""
    up = g["dealer"][0]
    # what the AI is allowed to know: only which cards HAVEN'T been seen yet - the rest of the shoe with the
    # dealer's face-down card mixed in, sorted into a plain pile. It never learns which one is the dealer's,
    # or the order the cards will come out in - exactly what a sharp player at a real table could know.
    pool = tuple(sorted(g["shoe"] + [g["dealer"][1]], key=lambda c: (BJ_RANKS.index(c[0]), c[1])))
    if can_double is None:
        can_double = len(g["player"]) == 2 and bj_game()["chips"] >= g["stake"]
    if can_split is None:
        can_split = bj_can_split(g) if "hands" in g else False
    evs = bj_simulate_moves(tuple(g["player"]), up, pool, can_double, can_split)
    chart = bj_chart(g["player"], bj_up_value(up), can_double, can_split)
    best = max(evs, key=evs.get)
    # 3,000 runs still has a little luck in it: if the simulation only beats the chart by a hair, trust the chart
    close = best != chart and evs[best] - evs[chart] < 0.03
    return {"evs": evs, "best": chart if close else best, "chart": chart, "close": close}


def bj_ai_bet(g):
    """the AI's bet advice from the true count: the minimum when it's neutral or low, more when it's high"""
    _, true, _ = bj_count(g)
    unit = max(10, round(g["chips"] * 0.01 / 5) * 5)   # 1 betting unit = about 1% of your chips
    units = 1 if true < 2 else min(8, int(true))        # the classic "bet spread": 1 unit, up to 8 when it's hot
    edge = -0.6 + 0.5 * true   # rough player edge %: about -0.6% off the top, +0.5% per point of true count
    return {"true": true, "bet": min(units * unit, g["chips"]), "units": units, "edge": edge}


def bj_play_strategy(strategy, hands, count_betting=False):
    """play `hands` hands of this table's blackjack with one strategy; returns (profit per 100 chips bet,
    win rate of the hands that won or lost). strategy(hand, up_value, can_double, can_split, true_count) ->
    "H" / "S" / "D" / "P". count_betting = bet 1-8 units by the true count (or a function: true count -> units)."""
    shoe, running, staked, profit, won, lost = [], 0, 0, 0, 0, 0

    def draw():
        nonlocal shoe, running
        if len(shoe) < 52:
            shoe, running = bj_shoe(), 0
        card = shoe.pop()
        running += hilo(card)
        return card

    for _ in range(hands):
        if len(shoe) < 52:
            shoe, running = bj_shoe(), 0
        true = running / max(len(shoe) / 52, 0.5)
        bet = (count_betting(true) if callable(count_betting) else      # a bot's own learned bet sizes
               (1 if true < 2 else min(8, int(true))) if count_betting else 1)
        player, dealer = [draw(), draw()], [draw()]
        dealer.append(shoe.pop())   # the hole card: face down, so it isn't counted until it's turned over
        staked += bet
        if bj_total(player) == 21 or bj_total(dealer) == 21:   # blackjack on the deal
            p, d = bj_total(player) == 21, bj_total(dealer) == 21
            result = 0 if p and d else 1.5 * bet if p else -bet
        else:
            up = bj_up_value(dealer[0])
            # every hand (splits too) played by the strategy - it also gets the true count of the cards seen so far
            chips, bets = bj_round(player, dealer, draw, lambda cards, dbl, spl: strategy(
                cards, up, dbl, spl, running / max(len(shoe) / 52, 0.5)))
            result = chips * bet
            staked += (bets - 1) * bet   # extra chips put out for doubles and splits
        running += hilo(dealer[1])       # the hole card is turned over now - count it
        profit += result
        won += result > 0
        lost += result < 0
    return profit / staked * 100, won / max(won + lost, 1)


def bj_simulate(hands):
    """the blackjack Strategy lab: four ways of playing, `hands` hands each"""
    chart = lambda h, up, d, s, t: bj_chart(h, up, d, s)   # basic strategy, pair splitting included
    styles = {"Basic strategy + counting (the AI's way)": (chart, True),
              "Basic strategy, same bet every hand": (chart, False),
              # never risk busting: stand on hard 12+, but soft hands (A-2...) can't bust, so hit them up to 18
              "Never bust (stand on 12 or more)": (lambda h, up, d, s, t: ("H" if bj_total(h) < 18 else "S")
                                                   if bj_is_soft(h) else "S" if bj_total(h) >= 12 else "H", False),
              "Mimic the dealer (hit to 17)": (lambda h, up, d, s, t: "S" if bj_total(h) >= 17 else "H", False)}
    bot = bj_trained()
    if bot is not None:   # your trained bot: its learned moves (count-aware, splits too) and its learned bet sizes
        def trained(hand, up, can_double, can_split, true):
            up = 1 if up == 11 else up
            card = 1 if hand[0][0] == "A" else min(10, bj_up_value(hand[0]))
            key = (f"pair,{card},{up}," if can_split else
                   f"{bj_total(hand)},{int(bj_is_soft(hand))},{up},{int(can_double)},")
            spot = bot["states"].get(key + str(bj_bucket(true))) or bot["states"].get(key + "2")
            return spot["best"] if spot else bj_chart(hand, 11 if up == 1 else up, can_double, can_split)
        bets = lambda true: bot["betting"].get(str(min(max(round(true), -6), 10)), {"units": 1})["units"]
        styles = {f"Your trained bot ({bot['meta']['hands'] / 1e6:,.0f}M hands of self-play)": (trained, bets),
                  "Your trained bot, same bet every hand": (trained, False)} | styles
    rows = []
    for name, (play, counting) in styles.items():
        profit, win_rate = bj_play_strategy(play, hands, counting)
        rows.append({"Strategy": name, "Profit per 100 chips bet": profit, "Win rate": win_rate})
    return pd.DataFrame(rows)


def counting_trainer(g):
    """the card-counting trainer under the table - one switch turns all of it on or off (off = plain blackjack)"""
    if not st.toggle("Card counting trainer", key="bj_trainer",
                     help="Learn the Hi-Lo card counting system while you play. For learning only - see the README."):
        return
    note = g.pop("note", None)
    if note:
        st.caption(f":material/shuffle: {note}")
    show_col, quiz_col = st.columns(2)
    show = show_col.toggle("Show the count", key="bj_show_count")
    quiz = quiz_col.toggle("Practice: quiz me every 3 hands", key="bj_quiz",
                           help="Turn 'Show the count' off for a real test")
    running, true, decks_left = bj_count(g)

    if show:
        a, b, c = st.columns(3)
        a.metric("Running count", f"{running:+d}", border=True,
                 help="Add up every card you've seen this shoe: 2-6 = +1, 7-9 = 0, 10 / J / Q / K / A = -1")
        b.metric("True count", f"{true:+.1f}", border=True, help="Running count ÷ decks still in the shoe")
        c.metric("Decks left", f"{decks_left:.1f}", border=True)
        if true >= 2:
            st.success(f"The count is high ({true:+.1f}): lots of tens and aces are left, so the table favours you. "
                       "A counter would bet more here.", icon=":material/trending_up:")
        elif true <= -2:
            st.warning(f"The count is low ({true:+.1f}): mostly small cards are left, so the house edge is bigger. "
                       "A counter would bet the minimum.", icon=":material/trending_down:")
        else:
            st.caption("The count is close to neutral - a counter would make a normal-sized bet.")

    # the quiz: after every 3rd hand, what's the running count?
    hands = g.get("hands_played", 0)
    right, asked = g.setdefault("quiz_score", [0, 0])
    if quiz and g["phase"] == "done" and hands and hands % 3 == 0 and g.get("quiz_hand") != hands:
        ask, check = st.columns([3, 1], vertical_alignment="bottom")
        guess = ask.number_input("Quiz: what's the running count right now?", step=1, value=0,
                                 key=f"bj_guess_{hands}")
        if check.button("Check", key=f"bj_check_{hands}", width="stretch", type="primary"):
            g["quiz_hand"] = hands
            g["quiz_score"] = [right + (guess == running), asked + 1]
            g["quiz_msg"] = (True, f"Correct - the running count is {running:+d} (true count {true:+.1f}).") \
                if guess == running else (False, f"Not quite - it's {running:+d}, you said {guess:+d}. "
                                                 f"(True count {true:+.1f}.) Keep going, it gets easier!")
    if g.get("quiz_msg"):
        good, text = g["quiz_msg"]
        (st.success if good else st.error)(text, icon=":material/check_circle:" if good else ":material/cancel:")
    if quiz and g["quiz_score"][1]:
        st.caption(f"Quiz score: {g['quiz_score'][0]} of {g['quiz_score'][1]} right")

    with st.expander("How card counting works (Hi-Lo)", icon=":material/school:"):
        st.markdown(
            "**The idea:** tens and aces help *you* (more blackjacks, and the dealer busts more), small cards help "
            "*the dealer*. Keep track of which have been played and you know what's left in the shoe.\n\n"
            "**Hi-Lo values:** 2, 3, 4, 5, 6 = **+1** · 7, 8, 9 = **0** · 10, J, Q, K, A = **-1**\n\n"
            "1. Start at 0 when the shoe is shuffled.\n"
            "2. Add up every card you see - yours, the dealer's, all of them. That's the **running count**.\n"
            "3. Divide by the decks left in the shoe: the **true count**. +2 or higher means the cards left favour "
            "you; a counter bets more. Negative means they favour the house; a counter bets the minimum.\n\n"
            "**For learning only.** It isn't illegal, but casinos can refuse service to people who count cards - "
            "and online blackjack in Ontario reshuffles automatically, so counting doesn't work there at all. "
            "This table uses fake chips. If gambling stops being fun, ConnexOntario offers free 24/7 help at "
            "1-866-531-2600.")


@st.fragment
def blackjack():
    """the table (its own tab) - a fragment, so clicking Hit / Stand only redraws the game, not the whole page"""
    g = bj_game()
    st.html(CARD_CSS + CHIP_CSS + """<style>
      .bj-splits { display:flex; justify-content:center; gap:14px; flex-wrap:wrap; }
      .bj-split  { padding:6px 10px 2px; border-radius:14px; border:2px solid transparent; }
      .bj-split.on { border-color:#FBBF24; box-shadow:0 0 16px rgba(251,191,36,.35); }
    </style>""")

    playing = g["phase"] == "play"
    if not playing and g["bet"] > g["chips"]:
        g["bet"] = 0   # can't cover last hand's bet any more - start the circle empty
    hands = g.get("hands") or []
    split = len(hands) > 1 and g["player"] is not None and any(h["cards"] is g["player"] for h in hands)
    dealer_total = "?" if playing else (bj_total(g["dealer"]) if g["dealer"] else "")
    player_total = bj_total(g["player"]) if g["player"] else ""
    if split:   # every hand side by side - the one you're playing is outlined; new cards slide in
        seen = g.get("seen", {}).get("hands", [])
        you = ""
        for i, h in enumerate(hands):
            n_seen = seen[i] if i < len(seen) else 0
            cards = "".join(
                f'<span class="bj-card {"red" if BJ_SUITS[s] else ""}{" new" if j >= n_seen else ""}" '
                f'style="animation-delay:{max(0, j - n_seen) * DEAL_GAP:.2f}s"><b>{r}</b><span class="bj-suit">{s}</span></span>'
                for j, (r, s) in enumerate(h["cards"]))
            on = playing and i == g.get("active")
            you += (f'<div class="bj-split{" on" if on else ""}"><div class="bj-who">Hand {i + 1} '
                    f'<b>{bj_total(h["cards"])}</b> · {h["stake"]:,} chips</div><div class="bj-hand">{cards}</div></div>')
        you = f'<div class="bj-splits">{you}</div>'
    else:
        you = (f'<div class="bj-who">You{f" <b>{player_total}</b>" if player_total != "" else ""}</div>'
               f'<div class="bj-hand">{bj_cards(g, "player")}</div>')
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="bj_felt"):
            st.html(f'<div>'
                    f'<div class="bj-who">Dealer{f" <b>{dealer_total}</b>" if dealer_total != "" else ""}</div>'
                    f'<div class="bj-hand">{bj_cards(g, "dealer", hide_hole=playing)}</div>'
                    f'<div class="bj-mark">BLACKJACK PAYS 3 TO 2 &nbsp;·&nbsp; DEALER STANDS ON 17 &nbsp;·&nbsp; '
                    f'SPLIT UP TO 4 HANDS</div>{you}'
                    f'<div class="bj-msg">{html.escape(g["msg"]) or ("Tap chips to bet, then deal." if not g["player"] else "")}'
                    f'</div></div>')
            # the betting circle: your chips sit here (all of them while a hand is on - every split hand's bet too)
            in_play = sum(h["stake"] for h in hands) if playing and hands else g["stake"]
            st.html(f'<div class="pq-circle">{chip_stack_html(in_play if playing else g["bet"], g.pop("drop", False))}</div>')
            if playing:   # your moves, right on the table
                hit_col, stand_col, double_col, split_col = st.columns(4)
                hit_col.button("Hit", on_click=bj_hit, key="bj_hit", width="stretch", type="primary",
                               icon=":material/add:")
                stand_col.button("Stand", on_click=bj_stand, key="bj_stand", width="stretch",
                                 icon=":material/front_hand:")
                double_col.button("Double down", on_click=bj_double, key="bj_double", width="stretch",
                                  icon=":material/exposure_plus_2:",
                                  disabled=len(g["player"]) != 2 or g["chips"] < g["stake"],
                                  help="Double your bet and take exactly one more card")
                split_col.button("Split", on_click=bj_split, key="bj_split", width="stretch",
                                 icon=":material/call_split:", disabled=not bj_can_split(g),
                                 help="Two cards of the same value: split them into two hands (the same bet again "
                                      "on the new one). Up to 4 hands; split aces get one card each.")
            elif g["chips"] >= 10:
                chip_rack("bj", bj_add_chip, lambda v: g["bet"] + v <= g["chips"])
                clear_col, deal_col = st.columns(2)
                clear_col.button("Clear", on_click=bj_clear, key="bj_clear", width="stretch", icon=":material/close:",
                                 disabled=not g["bet"])
                deal_col.button(f"Deal ({g['bet']:,} chips)" if g["bet"] else "Deal", on_click=bj_deal, key="bj_deal",
                                width="stretch", type="primary", icon=":material/playing_cards:", disabled=g["bet"] < 10)
        # everything on the table has now been shown - only cards dealt after this animate
        g["seen"] = {"player": len(g["player"]), "dealer": len(g["dealer"]), "hole": not playing,
                     "hands": [len(h["cards"]) for h in hands]}
        counting_trainer(g)
        strategy_lab("bj")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{g['chips']:,}", f"{g['chips'] - 1000:+,} since you started" if g["chips"] != 1000 else None,
                  border=True)
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True)
        # ---- the AI: advice only - you make every move yourself ----
        with st.container(border=True):
            st.markdown("**:material/psychology: Blackjack AI**")
            if bj_trained() is not None:
                st.segmented_control("Coach", ["Your trained bot", "Simulation AI"], default="Your trained bot",
                                     key="bj_coach", width="stretch",
                                     help="Your trained bot = the one in the 'blackjack training' folder: it taught "
                                          "itself by playing hundreds of millions of hands at this table. Simulation "
                                          "AI = 3,000 simulated runs of this exact hand, checked against the chart.")
            trainer_switch("bj", "chosen Hit, Stand or Double")
            spot = bj_trained_move(g) if playing and bj_coach_trained() else None
            if playing and trainer_on("bj"):
                trainer_quiet()
            elif spot:   # your trained bot: what it learned for this exact spot at this count
                headline = BJ_MOVE[spot["best"]]
                why = (f"Your trained bot learned this from <b>{spot['tries']:,}</b> tries of this exact spot at a "
                       f"count of {spot['buckets'][spot['bucket']]} (it has played {spot['hands'] / 1e6:,.0f} million "
                       f"hands in all).")
                details = " &nbsp;".join(
                    f'{BJ_MOVE[m]}: <b style="color:{"#86EFAC" if m == spot["best"] else "#E5E7EB"}">{ev:+.2f}</b>'
                    for m, ev in sorted(spot["evs"].items(), key=lambda kv: -kv[1]))
                details += ' <span style="color:#7A7A7A">average result per chip</span>'
                chart = bj_chart(g["player"], bj_up_value(g["dealer"][0]),
                                 len(g["player"]) == 2 and g["chips"] >= g["stake"], bj_can_split(g))
                if chart != spot["best"]:
                    details += (f'<br>The basic-strategy chart says {BJ_MOVE[chart]} - the bot found '
                                f'{BJ_MOVE[spot["best"]].lower()} does better here'
                                + (" at this count." if spot["bucket"] != 2 else "."))
            elif not playing and bj_coach_trained():   # its learned bet sizes
                adv = bj_trained_bet(g)
                headline = f"Bet {adv['bet']:,} chips"
                why = (f"True count {adv['true']:+.1f}: at this count your trained bot measured an edge of "
                       f"<b>{adv['edge']:+.2f}%</b>, so it bets <b>{adv['units']} unit{'s' if adv['units'] > 1 else ''}"
                       f"</b> (1 unit = ~1% of your chips). It worked out by itself that hot shoes pay - card "
                       f"counting, learned from scratch.")
                details = ""
            elif playing:
                adv = bj_ai_move(g)
                if adv["close"]:
                    why = "Too close to call by simulation - so it goes with the basic-strategy chart."
                elif adv["best"] != adv["chart"]:
                    why = (f"The cards left in this shoe change the answer - the basic-strategy chart says "
                           f"{BJ_MOVE[adv['chart']]}, but the simulation of this shoe says {BJ_MOVE[adv['best']]}.")
                else:
                    why = "The simulation of this shoe and the basic-strategy chart agree."
                headline = BJ_MOVE[adv["best"]]
                details = " &nbsp;".join(
                    f'{BJ_MOVE[m]}: <b style="color:{"#86EFAC" if m == adv["best"] else "#E5E7EB"}">{ev:+.2f}</b>'
                    for m, ev in sorted(adv["evs"].items(), key=lambda kv: -kv[1]))
                details += ' <span style="color:#7A7A7A">per chip bet (3,000 simulated runs each)</span>'
            else:
                adv = bj_ai_bet(g)
                headline = f"Bet {adv['bet']:,} chips"
                why = (f"True count {adv['true']:+.1f}: lots of tens and aces left - the cards favour you, so a "
                       f"bigger bet ({adv['units']} units)." if adv["units"] > 1 else
                       f"True count {adv['true']:+.1f}: the cards left don't favour you - bet the minimum "
                       f"(1 unit, ~1% of your chips).")
                details = (f'Rough edge right now: <b style="color:{"#86EFAC" if adv["edge"] > 0 else "#FCA5A5"}">'
                           f'{adv["edge"]:+.1f}%</b> (you vs the house)')
            if not (playing and trainer_on("bj")):
                st.html(f'<div style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.45);'
                        f'border-radius:10px;padding:10px 12px;margin:2px 0 6px">'
                        f'<div style="font-size:12px;color:#A7F3D0;letter-spacing:.08em;text-transform:uppercase">'
                        f'The AI says</div><div style="font-size:18px;font-weight:800;color:#FFF;margin:2px 0">{headline}</div>'
                        f'<div style="font-size:12px;color:#C7C7C7;line-height:1.55">{why}<br>{details}</div></div>')
            if trainer_on("bj"):
                trainer_panel("bj")
            track = g.get("ai_track")
            if track and track["total"]:
                st.caption(f"You followed its advice on {track['followed']} of {track['total']} moves.")
            if track and sum(track.get("ai_record", [])):
                ai_rate_caption(track["ai_record"], "hands", "playing every hand its own way")
                st.caption("Even perfect play wins only ~47% of decided hands at this table - blackjacks paying 3 to 2 "
                           "and doubling are what make up the difference.")
            if track and sum(track.get("record", [])):
                w, l, p = track["record"]
                st.caption(f"Your win rate when you followed it all the way: **{w / max(w + l, 1):.0%}** ({w}-{l}-{p}).")
            st.caption("The AI can't see the dealer's face-down card or what's coming next - it only knows which "
                       "cards haven't been seen yet, like you would. Moves: simulated from those unseen cards, "
                       "checked against the expert chart. Bets: the true count - the one real edge in "
                       "blackjack. See the Strategy lab under the table for proof. For learning only.")
        if playing:
            st.caption(f"In play: {g['stake']:,} chips - Hit, Stand or Double down on the table.")
        elif g["chips"] < 10:
            st.button("Out of chips - get 1,000 more", on_click=bj_reset, key="bj_reset", icon=":material/refresh:",
                      width="stretch", type="primary")
        else:
            st.caption("Tap chips in the rack to put them in the betting circle. Your bet stays for the next hand - "
                       "just hit Deal again, or Clear to take it back.")

# ---------------------------------------------------------------- baccarat (punto banco) - fake chips, just for fun
# Bet on Player, Banker or Tie; the cards are dealt by fixed rules (no decisions after the bet).
# Uses the same chips as blackjack. 8-deck shoe, like a real baccarat table.
BAC_PAYS = {"Player": "pays 1 to 1", "Banker": "pays 0.95 to 1 (5% commission)", "Tie": "pays 8 to 1"}


def bac_value(card):
    """baccarat card value: ace = 1, 2-9 = face value, 10 / J / Q / K = 0"""
    rank = card[0]
    return 1 if rank == "A" else 0 if rank in ("10", "J", "Q", "K") else int(rank)


def bac_total(hand):
    return sum(bac_value(c) for c in hand) % 10   # only the last digit counts: 7 + 8 = 15 -> 5


def bac_game():
    g = st.session_state.setdefault("bac", {"shoe": [], "player": [], "banker": [], "order": [], "msg": "",
                                            "new": False, "record": [0, 0, 0], "history": []})
    g.setdefault("bets", {"Player": 0, "Tie": 0, "Banker": 0})   # chips sitting on each spot
    g.setdefault("chip", 25)                                       # the chip you'll drop next
    return g


def bac_pick_chip(value):
    bac_game()["chip"] = value


def bac_place(spot):
    """drop the picked chip on Player, Tie or Banker"""
    g, wallet = bac_game(), bj_game()
    if sum(g["bets"].values()) + g["chip"] > wallet["chips"]:
        g["msg"] = "Not enough chips for that - pick a smaller chip or clear the table."
        return
    g["bets"][spot] += g["chip"]
    g["drop"], g["new"] = spot, False


def bac_clear():
    bac_game()["bets"] = {"Player": 0, "Tie": 0, "Banker": 0}


def bac_draw(g):
    if len(g["shoe"]) < 52:   # cut card: a fresh 8-deck shoe
        g["shoe"] = [(rank, suit) for _ in range(8) for rank in BJ_RANKS for suit in BJ_SUITS]
        random.shuffle(g["shoe"])
    return g["shoe"].pop()


def bac_coup(draw):
    """one hand of baccarat by the official rules: (player cards, banker cards, deal order, natural?).
    draw() hands out the next card - the real table and the simulator both use this, so the rules are identical."""
    # dealt Player, Banker, Player, Banker
    p1, b1, p2, b2 = draw(), draw(), draw(), draw()
    player, banker = [p1, p2], [b1, b2]
    order = [("player", 0), ("banker", 0), ("player", 1), ("banker", 1)]
    natural = bac_total(player) >= 8 or bac_total(banker) >= 8   # an 8 or 9 on the first two cards: nobody draws
    if not natural:
        third = None
        if bac_total(player) <= 5:   # Player draws on 0-5, stands on 6-7
            third = draw()
            player.append(third)
            order.append(("player", 2))
        b = bac_total(banker)
        if third is None:            # Player stood -> Banker plays the same rule
            banker_draws = b <= 5
        else:                        # Player drew -> the official Banker table, based on Player's third card
            t = bac_value(third)
            banker_draws = (b <= 2 or (b == 3 and t != 8) or (b == 4 and 2 <= t <= 7)
                            or (b == 5 and 4 <= t <= 7) or (b == 6 and t in (6, 7)))
        if banker_draws:
            banker.append(draw())
            order.append(("banker", 2))
    return player, banker, order, natural


def bac_results(hands):
    """deal `hands` real baccarat hands (official rules, 8-deck shoes) -> the winners in order: "P", "B" or "T" """
    shoe = []

    def draw():
        nonlocal shoe
        if len(shoe) < 52:
            shoe = [(rank, suit) for _ in range(8) for rank in BJ_RANKS for suit in BJ_SUITS]
            random.shuffle(shoe)
        return shoe.pop()

    results = []
    for _ in range(hands):
        player, banker, _, _ = bac_coup(draw)
        p, b = bac_total(player), bac_total(banker)
        results.append("P" if p > b else "B" if b > p else "T")
    return results


@st.cache_resource(show_spinner="Simulating 500,000 baccarat hands for the AI...")
def bac_base_rates(hands=500_000):
    """the simulation the AI leans on: how often Banker / Player / Tie win over 500,000 hands (done once)"""
    counts = Counter(bac_results(hands))
    return {r: counts[r] / hands for r in "PBT"}, hands


def bac_simulate(hands):
    """deal `hands` hands from fresh 8-deck shoes and score six ways of betting 1 chip a hand on the SAME cards"""
    results = bac_results(hands)

    def profit(pick, winner):   # 1 chip bet on `pick`
        if winner == "T":
            return 8 if pick == "T" else 0      # Player / Banker bets come back on a tie
        if pick == winner:
            return 0.95 if pick == "B" else 1
        return -1

    styles = {"Always Banker": lambda last: "B", "Always Player": lambda last: "P", "Always Tie": lambda last: "T",
              "Follow the streak": lambda last: last or "B",                      # bet whoever won last
              "Chop (bet against the streak)": lambda last: {"P": "B", "B": "P"}.get(last, "B"),
              "Random Player / Banker": lambda last: random.choice("PB")}
    # the exact long-run result per 100 chips, from the math of an 8-deck shoe. The streak / chop / random styles
    # just switch between Banker and Player, so they land in between the two (past hands don't change the odds)
    exact = {"Always Banker": -1.06, "Always Player": -1.24, "Always Tie": -14.36, "Follow the streak": -1.15,
             "Chop (bet against the streak)": -1.15, "Random Player / Banker": -1.15}
    rows = []
    for name, pick_for in styles.items():
        last, total, won, lost = None, 0.0, 0, 0
        for winner in results:
            result = profit(pick_for(last), winner)
            total += result
            won += result > 0
            lost += result < 0
            if winner != "T":
                last = winner
        rows.append({"Betting style": name, "Return": 1 + total / hands, "Profit per 100 chips bet": total / hands * 100,
                     "Long run (exact math)": exact[name],
                     "Win rate": won / max(won + lost, 1)})   # of the hands that won or lost (ties push)
    return pd.DataFrame(rows).sort_values("Long run (exact math)", ascending=False), Counter(results)


# ---- the baccarat AI: a machine-learning model that tries to predict the next result, and learns from every hand ----
# Honest note: baccarat hands are almost independent (the cards are shuffled), so a good model should settle near
# "Banker 46% / Player 45% / Tie 9%". The scoreboard compares it with simply always picking Banker.
BAC_CLASSES = ["P", "B", "T"]
BAC_NAME = {"P": "Player", "B": "Banker", "T": "Tie"}


def bac_features(history, shoe):
    """what the AI sees before a hand: the last 3 results, the current streak, and which cards are left in the shoe"""
    last = (["-"] * 3 + list(history))[-3:]
    recent = [1.0 if r == c else 0.0 for r in last for c in BAC_CLASSES]
    streak = 0
    for r in reversed(history):
        if r != history[-1]:
            break
        streak += 1
    if len(shoe) < 52:   # it's about to be reshuffled - a fresh shoe
        shoe = [(rank, suit) for _ in range(8) for rank in BJ_RANKS for suit in BJ_SUITS]
    values = Counter(bac_value(c) for c in shoe)
    return recent + [min(streak, 6) / 6] + [values.get(v, 0) / len(shoe) for v in range(10)]


@st.cache_resource(show_spinner="Training the baccarat AI on 40,000 simulated hands...")
def bac_ai_base(hands=40_000):
    """the AI's starting experience: trained once on simulated hands (shared by everyone, then copied per player)"""
    rng_shoe, history, X, y = [], [], [], []

    def draw():
        return rng_shoe.pop()

    for _ in range(hands):
        if len(rng_shoe) < 52:
            rng_shoe[:] = [(rank, suit) for _ in range(8) for rank in BJ_RANKS for suit in BJ_SUITS]
            random.shuffle(rng_shoe)
            history = []
        X.append(bac_features(history, rng_shoe))
        player, banker, _, _ = bac_coup(draw)
        p, b = bac_total(player), bac_total(banker)
        y.append("P" if p > b else "B" if b > p else "T")
        history.append(y[-1])
    # alpha = how cautious it is. Tested on 20,000 fresh hands: a loose model (0.001) predicted WORSE than plain
    # base rates (it "found" patterns that aren't there); 0.03 is as good as the base rates - there's no real
    # pattern in baccarat results to find, so the honest model mostly agrees with the simulation
    model = SGDClassifier(loss="log_loss", alpha=0.03, random_state=0)
    model.fit(X, y)
    # from now on each real hand nudges it only a little - otherwise it drifts after random streaks
    model.set_params(learning_rate="constant", eta0=0.0005)
    return model


def bac_ai(g):
    """this player's own copy of the AI - it keeps learning from their hands"""
    if "ai" not in g:
        g["ai"] = copy.deepcopy(bac_ai_base())
        g["ai_score"] = {"right": 0, "total": 0, "banker": 0, "learned": 0}
    return g["ai"]


def bac_ai_predict(g):
    """{"P": chance, "B": chance, "T": chance} for the NEXT hand"""
    model = bac_ai(g)
    probs = model.predict_proba([bac_features(g["history"], g["shoe"])])[0]
    return dict(zip(model.classes_, probs))


AI_WEIGHT = 0.35       # blended chances = 35% the AI's read of this shoe + 65% the 500,000-hand simulation
# only advise against the math's best bet if another spot is ahead by 3+ chips per 100. Tested: the AI's read has
# no real predictive edge, so with a 1-chip margin it still switched to Player ~8% of hands for nothing
SWITCH_MARGIN = 3.0


def bac_spot_profit(chances):
    """expected profit per 100 chips on each spot, from chances {"P", "B", "T"}.
    Player: win 1, lose 1, a tie comes back. Banker: win 0.95 (commission), lose 1, a tie comes back. Tie: win 8, lose 1."""
    p, b, t = chances["P"], chances["B"], chances["T"]
    return {"Player": (p - b) * 100, "Banker": (0.95 * b - p) * 100, "Tie": (8 * t - (1 - t)) * 100}


# the exact long-run odds of 8-deck baccarat (worked out by math over every possible deal - no luck in them)
BAC_EXACT = {"P": 0.4462, "B": 0.4586, "T": 0.0952}


def bac_ai_advice(g):
    """the AI's full advice for the next hand: the exact odds (checked by the simulation), the AI's own read,
    the blend, and what to bet. Same idea as the betting models: lean on the solid numbers, let the model nudge.
    (The baseline is the EXACT odds, not the simulation: Banker and Player are only 0.18 per 100 apart, and even
    500,000 simulated hands can swap them by luck - which once made it advise Player for no reason.)"""
    sim, sim_hands = bac_base_rates()
    ai = bac_ai_predict(g)
    blend = {r: AI_WEIGHT * ai[r] + (1 - AI_WEIGHT) * BAC_EXACT[r] for r in "PBT"}
    exact_profit, profit = bac_spot_profit(BAC_EXACT), bac_spot_profit(blend)
    math_best = max(exact_profit, key=exact_profit.get)              # Banker - the exact long-run best bet
    challenger = max((s for s in profit if s != math_best), key=profit.get)
    switch = profit[challenger] - profit[math_best] >= SWITCH_MARGIN  # only for a clear difference, not noise
    return {"sim": sim, "sim_hands": sim_hands, "ai": ai, "blend": blend, "profit": profit,
            "bet": challenger if switch else math_best, "math_best": math_best, "switched": switch,
            "skip": min(profit, key=profit.get)}


def bac_deal():
    g, wallet = bac_game(), bj_game()   # one chip balance for every card game
    bets = g["bets"]
    total = sum(bets.values())
    if not total:
        g["msg"] = "Put some chips on Player, Tie or Banker first."
        return
    if total > wallet["chips"]:
        g["msg"] = "Not enough chips for everything on the table - clear it and bet smaller."
        return
    wallet["chips"] -= total
    if trainer_on("bac"):   # grade the spot with the most chips on it against the AI's advice
        adv = bac_ai_advice(g)
        main = max(bets, key=bets.get)
        per100 = ", ".join(f"{spot} {ev:+.2f}" for spot, ev in adv["profit"].items())
        trainer_grade("bac", main, adv["bet"],
                      f"The best bet was **{adv['bet']}** (you bet mostly on {main}). Long-run profit per 100 chips: "
                      f"{per100}. Banker loses the least even after its 5% commission; Tie is the worst bet.")
    # the AI's prediction for this hand, made BEFORE the cards come out
    features = bac_features(g["history"], g["shoe"])
    probs = bac_ai_predict(g)
    ai_pick = max(probs.items(), key=lambda kv: kv[1])[0]
    advised = bac_ai_advice(g)["bet"]   # what the AI told you to bet, before the cards came out
    player, banker, order, natural = bac_coup(lambda: bac_draw(g))
    p, b = bac_total(player), bac_total(banker)
    winner = "Player" if p > b else "Banker" if b > p else "Tie"
    # score the AI, then let it learn from what really happened
    score = g["ai_score"]
    score["total"] += 1
    score["right"] += ai_pick == winner[0]
    score["banker"] += winner == "Banker"
    bac_ai(g).partial_fit([features], [winner[0]])
    score["learned"] += 1
    g["ai_last"] = (ai_pick, winner[0])

    # how its ADVICE did: 1 chip on the advised spot every hand, next to 1 chip on Banker every hand
    def one_chip(spot):
        if winner == "Tie":
            return 8 if spot == "Tie" else 0
        return (0.95 if spot == "Banker" else 1) if spot == winner else -1
    score["advice_profit"] = score.get("advice_profit", 0) + one_chip(advised)
    score["banker_profit"] = score.get("banker_profit", 0) + one_chip("Banker")
    # win rates: of the hands its advice won or lost (a tie gives Player / Banker bets back)
    for who, spot in (("advice", advised), ("banker", "Banker")):
        result = one_chip(spot)
        score[f"{who}_won"] = score.get(f"{who}_won", 0) + (result > 0)
        score[f"{who}_lost"] = score.get(f"{who}_lost", 0) + (result < 0)
    score.setdefault("advised", Counter())[advised] += 1
    g["advice_last"] = advised
    # every spot is paid on its own, like a real table
    if winner == "Tie":
        pay = bets["Tie"] * 9 + bets["Player"] + bets["Banker"]   # Tie pays 8 to 1; Player / Banker come back
    else:
        pay = bets[winner] + (bets[winner] if winner == "Player" else round(bets[winner] * 0.95))
    net = pay - total
    score = f"Tie, {p} each" if winner == "Tie" else \
        f"{winner} wins {max(p, b)} to {min(p, b)}" + (" (natural)" if natural else "")
    text = f"{score} - " + (f"you win {net:,}!" if net > 0 else f"you lose {-net:,}." if net < 0 else "you break even.")
    wallet["chips"] += pay
    g["record"][0 if net > 0 else 2 if net == 0 else 1] += 1
    g["history"] = (g["history"] + [winner[0]])[-24:]   # the "road": the last 24 results
    g.update(player=player, banker=banker, order=order, msg=text, new=True, winner=winner)

def bac_cards(g, who):
    """a hand's cards - straight after a deal they slide in one by one, in the order they were dealt"""
    out = ""
    for i, (rank, suit) in enumerate(g[who]):
        anim = ""
        if g["new"]:
            step = g["order"].index((who, i))
            anim = f' new" style="animation-delay:{step * DEAL_GAP * 1.4:.2f}s'
        out += (f'<span class="bj-card {"red" if BJ_SUITS[suit] else ""}{anim}"><b>{rank}</b>'
                f'<span class="bj-suit">{suit}</span></span>')
    return out


def strategy_lab(kind):
    """'Strategy lab': simulate thousands of hands to see which way of playing gives back the most.
    kind = "bac" (baccarat) or "vp" (video poker). Fake chips - it's about learning the math, not beating anyone."""
    with st.expander("Strategy lab: simulate thousands of hands", icon=":material/science:"):
        if kind == "bac":
            st.caption("Deals real baccarat hands (same rules as the table) and scores six ways of betting on the "
                       "SAME cards - so the only difference is the betting style.")
            sizes, simulate = [100_000, 250_000, 500_000, 1_000_000], bac_simulate
        elif kind == "bj":
            st.caption("Plays real blackjack hands (this table's rules: 6 decks, dealer stands on 17, blackjack "
                       "pays 3 to 2) four different ways. Counting = bet 1 unit normally, up to 8 when the true "
                       "count is high.")
            sizes, simulate = [20_000, 100_000, 250_000, 500_000], bj_simulate
        elif kind == "rl":
            st.caption("Spins the wheel and scores five ways of betting on the SAME spins - including the famous "
                       "Martingale (double after every loss) and chasing the 'hot' number.")
            sizes, simulate = [50_000, 200_000, 500_000, 1_000_000], rl_simulate
        elif kind == "tcp":
            st.caption("Deals real three card poker hands and scores four ways of deciding Play / Fold on the SAME "
                       "cards, plus the Pair Plus side bet on its own.")
            sizes, simulate = [20_000, 100_000, 250_000, 500_000], tcp_simulate
        elif kind == "pk":
            st.caption("Five styles each play the SAME deals heads-up against Ace, 100 big blinds deep. Slow-ish: "
                       "every decision deals out the unknown cards to work out the chance to win.")
            sizes, simulate = [100, 300, 600, 1000], pk_simulate
        elif kind == "cr":
            st.caption("Rolls the dice and scores six ways of betting on the SAME rolls, 1 chip a bet (plus full "
                       "odds for the first one).")
            sizes, simulate = [100_000, 500_000, 1_000_000, 2_000_000], cr_simulate
        elif kind == "lir":
            st.caption("Deals real let it ride hands and scores four ways of deciding when to pull bets back, on "
                       "the SAME cards. The first run takes a little while - the AI works every hand out exactly.")
            sizes, simulate = [5_000, 20_000, 50_000, 100_000], lir_simulate
        else:
            st.caption("Plays the same deals three different ways, next to perfect play (the best hold every hand, "
                       "worked out exactly by math).")
            sizes, simulate = [5_000, 20_000, 50_000], vp_simulate
        n_col, run_col = st.columns([3, 1], vertical_alignment="bottom")
        hands = n_col.select_slider("Hands to simulate", sizes, value=sizes[1], key=f"{kind}_sim_n",
                                    format_func=lambda n: f"{n:,}")
        if run_col.button("Run simulation", key=f"{kind}_sim_go", type="primary", width="stretch",
                          icon=":material/play_arrow:"):
            with st.spinner(f"Playing {hands:,} hands..."):
                st.session_state[f"{kind}_sim"] = simulate(hands)
        result = st.session_state.get(f"{kind}_sim")
        if result is None:
            return
        table = result[0] if kind in ("bac", "rl", "cr") else result
        name_col = "Betting style" if kind == "bac" else "Strategy"
        # only the columns that matter, so the table fits next to the side panel
        shown = ([name_col, "Win rate", "Profit per 100 chips bet", "Long run (exact math)"] if kind in ("bac", "rl", "cr")
                 else [name_col, "Win rate", "Profit per 100 chips bet"] if kind in ("bj", "tcp")
                 else [name_col, "Win rate", "Profit per 100 chips bet", "Per hand (per 100 of one bet)"] if kind == "lir"
                 else [name_col, "Win rate", "Big blinds won per 100 hands"] if kind == "pk"
                 else [name_col, "Win rate", "Return", "How it's known"])
        st.dataframe(table, hide_index=True, width="stretch", column_order=shown,
                     column_config={"Win rate": st.column_config.NumberColumn(
                         format="percent", help={"bac": "Of the hands that won or lost (a tie returns Player / Banker bets)",
                                                 "bj": "Of the hands that won or lost (pushes left out)",
                                                 "rl": "Spins that made a profit",
                                                 "tcp": "Of the hands that won or lost (folds count as losses)",
                                                 "lir": "Hands ending in a pair of tens or better",
                                                 "cr": "Of the bets that won or lost (pushes left out)",
                                                 "pk": "Hands won, of the hands won or lost",
                                                 "vp": "Hands ending in two pair or better (jacks or better only "
                                                       "gives your bet back)"}[kind]),
                                    "Return": st.column_config.NumberColumn(
                         "Gives back per chip", format="%.4f", help="1.0000 = break even. Below 1 = the house keeps some"),
                         "Profit per 100 chips bet": st.column_config.NumberColumn(
                             "Simulated profit per 100", format="%+.2f"),
                         "Per hand (per 100 of one bet)": st.column_config.NumberColumn(
                             format="%+.2f", help="Profit per hand for every 100 chips on ONE circle - the usual "
                                                  "way the house edge is quoted (about -3.5 played perfectly)"),
                         "Long run (exact math)": st.column_config.NumberColumn(
                             format="%+.2f", help="What each style makes per 100 chips over millions of hands - "
                                                  "worked out exactly, no luck involved"),
                         "Big blinds won per 100 hands": st.column_config.NumberColumn(
                             format="%+.1f", help="How poker players measure a style: big blinds (10 chips) won or "
                                                  "lost per 100 hands. Above 0 = beating Ace")})
        # baccarat: chart the exact long-run numbers (the simulated ones are too close to call by luck alone)
        field = ("Long run (exact math)" if kind == "bac" else "Big blinds won per 100 hands" if kind == "pk"
                 else "Profit per 100 chips bet")
        chart = alt.Chart(table).mark_bar(cornerRadius=3).encode(
            x=alt.X(f"{field}:Q", title="big blinds won per 100 hands vs Ace (below 0 = losing)" if kind == "pk"
                    else "profit per 100 chips bet, long run (below 0 = losing)"),
            y=alt.Y(f"{name_col}:N", sort="-x", title=None),
            color=alt.condition(alt.datum[field] >= 0, alt.value(GREEN), alt.value(RED)),
            tooltip=[name_col, alt.Tooltip(f"{field}:Q", format="+.2f")])
        st.altair_chart(chart.properties(height=40 * len(table) + 20), width="stretch")
        if kind == "bac":
            counts = result[1]
            total = sum(counts.values())
            luck = 95 / total ** 0.5 * 2   # about 2 standard errors, per 100 chips
            st.caption(f"Who won: Banker {counts['B'] / total:.1%} · Player {counts['P'] / total:.1%} · "
                       f"Tie {counts['T'] / total:.1%}. Luck still moves each simulated result by about "
                       f"±{luck:.2f} per 100 at {total:,} hands - Banker and Player are only 0.18 apart, so a "
                       f"simulation can put them in either order. The exact-math column is the true ranking.")
            st.info("**What it shows:** **Banker is the best bet** - it loses the least (about 1 chip per 100). "
                    "Following the streak or betting against it lands right where Player / Banker do: past results "
                    "don't predict the next hand, so the \"road\" can't give you an edge. **Tie is the worst bet** by "
                    "far. And every style still loses over time - that's the house edge.",
                    icon=":material/lightbulb:")
        elif kind == "rl":
            st.info("**What it shows:** every roulette bet - and every system - ends up near **-5.26 per 100**, the "
                    "two green pockets (0 and 00). **Win rate is not profit:** two dozens wins about 65% of spins and Martingale "
                    "wins about half of them (and most of its *sequences*), yet both still lose the same 5.26% - "
                    "Martingale just hides it until a long losing streak wipes out many small wins. The 'hot' number "
                    "does no better than any other number: the wheel has no memory.", icon=":material/lightbulb:")
            st.markdown("**A different question: start with 100 chips - what gets you to 200 most often?**")
            st.dataframe(result[1], hide_index=True, width="stretch", column_config={
                "Reached 200": st.column_config.NumberColumn(format="percent",
                                                             help="Out of 4,000 tries, each played until 200 or broke")})
            st.caption("This is the famous *bold play* result (Dubins & Savage): when the house has an edge, the "
                       "fewer and bigger your bets, the better your chance of hitting a goal - one all-in bet on Red "
                       "doubles you 47.4% of the time, while 5-chip bets give the 5.26% edge hundreds of spins to "
                       "grind you down. It doesn't make roulette profitable - it's the least-bad way to chase a target. "
                       "Fake chips only.")
        elif kind == "bj":
            st.caption(f"Luck still moves each result by about ±{230 / hands ** 0.5:.2f} per 100 at {hands:,} hands "
                       "(more for counting - its bets vary more). Run more hands for a steadier answer.")
            st.info("**What it shows:** in blackjack your decisions matter a lot - **basic strategy** loses only "
                    "about 1 chip per 100, while copying the dealer or never risking a bust loses 4-6x more. "
                    "**Adding the count** (betting more when lots of tens and aces are left) gets close to even or "
                    "slightly ahead - that's the AI's advice. It's for learning only: casinos can refuse counters, and "
                    "online blackjack reshuffles every hand, so counting doesn't work there.",
                    icon=":material/lightbulb:")
        elif kind == "pk":
            st.caption(f"Poker swings a lot: at {hands:,} hands, luck alone moves each result by roughly "
                       f"±{100 * 25 / hands ** 0.5:.0f} big blinds per 100 (big all-in pots swing it). Run more hands "
                       f"for a steadier answer.")
            st.info("**What it shows:** unlike the casino games, poker has **no house edge** - you're playing "
                    "Ace, not the casino, so a better style really does win chips. Calling everything or raising "
                    "everything gets punished; folding too much gives away the blinds. Weighing your chance to win "
                    "against the price of calling (the advisor) is the backbone of good poker.",
                    icon=":material/lightbulb:")
        elif kind == "cr":
            st.info("**What it shows:** the dice are the same for every style - only the bets differ, and that "
                    "changes everything. **Pass + full odds** loses the least (about 0.37 per 100) because the odds "
                    "part has no house edge. Pass, Don't Pass and Place 6 / 8 are all cheap; the Field costs about "
                    "twice as much, and **Any 7 loses about 17 per 100** - more than 10x the Pass Line. Every style "
                    "still loses over time.", icon=":material/lightbulb:")
        elif kind == "lir":
            st.info("**What it shows:** the cards are the same for everyone, so **win rate barely changes** - what "
                    "changes is how much is riding when you win. The AI pulls back weak hands and rides strong ones, "
                    "losing the least (about 3.5 per 100 of one bet, per hand). Always riding loses much more, and "
                    "always pulling back still pays the house on the $ bet. Every way loses over time.",
                    icon=":material/lightbulb:")
        elif kind == "tcp":
            st.info("**What it shows:** the **Q-6-4 rule** loses the least - about 2 chips per 100 bet. Always "
                    "playing or only playing pairs costs more, and Pair Plus is the worst bet on the table (about "
                    "-7.3 per 100). Every way still loses over time - that's the house edge.",
                    icon=":material/lightbulb:")
        else:
            st.info("**What it shows:** in video poker your choices really matter. **Perfect play gives back about "
                    "99.5%** - turn on **Show the best hold** to play that way. A simple strategy gets close; just "
                    "drawing 5 new cards every time loses badly. Even perfect play still loses a little on this pay "
                    "table - the house keeps its edge.", icon=":material/lightbulb:")


@st.fragment
def baccarat():
    """the baccarat table - a fragment, so dealing only redraws the game, not the whole page"""
    g, wallet = bac_game(), bj_game()
    st.html(CARD_CSS + CHIP_CSS + """<style>
      .bac-road { display:flex; flex-wrap:wrap; gap:5px; margin-top:4px; }
      .bac-dot  { width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center;
                  justify-content:center; font-size:11px; font-weight:800; }
      .bac-dot.P { background:#FFFFFF; color:#000; }
      .bac-dot.B { background:#EF4444; color:#FFF; }
      .bac-dot.T { background:#22C55E; color:#000; }
    </style>""")
    landed = (len(g["order"]) * DEAL_GAP * 1.4 + 0.35) if g["new"] else 0   # when the last card has landed
    fade = f' class="bj-fade" style="animation-delay:{landed:.2f}s"' if g["new"] else ""
    totals = {who: (f' <b{fade}>{bac_total(g[who])}</b>' if g[who] else "") for who in ("player", "banker")}
    msg = html.escape(g["msg"]) or "Tap a chip in the rack, then tap Player, Tie or Banker to bet - then deal."
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="bac_felt"):
            st.html(f'<div>'
                    f'<div class="bj-who">Banker{totals["banker"]}</div>'
                    f'<div class="bj-hand">{bac_cards(g, "banker")}</div>'
                    f'<div class="bj-who">Player{totals["player"]}</div>'
                    f'<div class="bj-hand">{bac_cards(g, "player")}</div>'
                    f'<div class="bj-msg"><span{fade}>{msg}</span></div></div>')
            # the three betting spots, with your chips sitting on them
            drop = g.pop("drop", None)
            if g.get("winner") and not drop:   # the winning spot lights up after a deal
                st.html(f"<style>.st-key-spot_{g['winner']} button {{ background:rgba(34,197,94,.35) !important; "
                        f"border-color:#22C55E !important; color:#FFF !important; }}</style>")
            spots = {"Player": "PLAYER  1:1", "Tie": "TIE  8:1", "Banker": "BANKER  0.95:1"}
            for col, (spot, label) in zip(st.columns(3), spots.items()):
                with col:
                    # same height for every spot, so the three buttons line up however tall the stacks are
                    st.html(f'<div style="height:126px;display:flex;align-items:flex-end;justify-content:center">'
                            f'{chip_stack_html(g["bets"][spot], drop == spot, empty="")}</div>')
                    st.button(label, key=f"spot_{spot}", on_click=bac_place, args=(spot,), width="stretch",
                              help=f"Drop a {g['chip']} chip on {spot} - it {BAC_PAYS[spot]}")
            if wallet["chips"] >= 10:
                chip_rack("bac", bac_pick_chip, lambda v: v <= wallet["chips"], selected=g["chip"])
                total = sum(g["bets"].values())
                clear_col, deal_col = st.columns(2)
                clear_col.button("Clear", on_click=bac_clear, key="bac_clear", width="stretch", icon=":material/close:",
                                 disabled=not total)
                deal_col.button(f"Deal ({total:,} chips)" if total else "Deal", on_click=bac_deal, key="bac_deal",
                                width="stretch", type="primary", icon=":material/playing_cards:", disabled=not total)
    g["new"] = False   # cards on the table now - they don't animate again until the next deal

    with table:
        if g["history"]:
            st.html('<div class="bj-who" style="color:#94A3B8">Last results</div><div class="bac-road">'
                    + "".join(f'<span class="bac-dot {r}">{r}</span>' for r in g["history"]) + "</div>")
        with st.expander("How baccarat works", icon=":material/school:"):
            st.markdown(
                "**Goal:** bet on which hand ends closer to **9** - Player or Banker - or that they **tie**.\n\n"
                "**Card values:** ace = 1, 2-9 = face value, 10 / J / Q / K = 0. Only the last digit of the total "
                "counts (7 + 8 = 15 -> **5**).\n\n"
                "**The deal follows fixed rules** - you don't make any choices after betting:\n"
                "- An 8 or 9 on the first two cards is a **natural** - nobody draws.\n"
                "- Player draws a third card on 0-5 and stands on 6-7.\n"
                "- Banker's draw depends on its total and Player's third card (the official table).\n\n"
                "**Payouts:** Player 1 to 1 · Banker 0.95 to 1 (5% commission, because Banker wins slightly more "
                "often) · Tie 8 to 1 - Player / Banker bets come back on a tie.\n\n"
                "**House edge:** Banker about 1.06%, Player about 1.24%, Tie about 14% - that's why the Tie bet's "
                "big payout is still the worst bet on the table. Fake chips, no real money - just for fun.")
        strategy_lab("bac")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every card game")
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True)
        if wallet["chips"] < 10:
            st.button("Out of chips - get 1,000 more", on_click=bj_reset, key="bac_reset", icon=":material/refresh:",
                      width="stretch", type="primary")
            return
        # ---- the AI: its prediction for the next hand, and how it's really doing ----
        with st.container(border=True):
            st.markdown("**:material/psychology: Baccarat AI** · next hand")
            probs = bac_ai_predict(g)
            pick = max(probs.items(), key=lambda kv: kv[1])[0]
            st.html("".join(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:13px">'
                f'<span style="width:52px;color:{"#FFF" if r == pick else "#94A3B8"};font-weight:{800 if r == pick else 500}">'
                f'{BAC_NAME[r]}</span><span style="flex:1;height:8px;background:#161616;border-radius:4px;overflow:hidden">'
                f'<span style="display:block;height:100%;width:{probs[r] * 100:.0f}%;'
                f'background:{"#22C55E" if r == pick else "#3A3A3A"}"></span></span>'
                f'<span style="width:44px;text-align:right;color:#E5E7EB">{probs[r]:.1%}</span></div>'
                for r in BAC_CLASSES))
            # its advice - you place the chips yourself. Built on the simulation, nudged by the AI.
            adv = bac_ai_advice(g)
            size = max(10, round(wallet["chips"] * 0.02 / 5) * 5)   # about 2% of your chips
            why = (f"The AI sees {adv['bet']} ahead by {adv['profit'][adv['bet']] - adv['profit'][adv['math_best']]:.1f}"
                   f" per 100 - enough to switch from {adv['math_best']}." if adv["switched"] else
                   f"The exact odds (confirmed by a {adv['sim_hands']:,}-hand simulation) say {adv['math_best']} "
                   f"loses the least, and the AI doesn't see a big enough reason to change that.")
            trainer_switch("bac", "placed your bet and dealt")
            if trainer_on("bac"):
                trainer_quiet()
                trainer_panel("bac")
            else:
                ai_box(f'Bet {adv["bet"]} · about {size:,} chips',
                       f'{why}<br>'
                       + " &nbsp;".join(f'{spot}: <b style="color:{"#86EFAC" if spot == adv["bet"] else "#FCA5A5" if spot == adv["skip"] else "#E5E7EB"}">'
                                        f'{ev:+.2f}</b>' for spot, ev in adv["profit"].items())
                       + f' <span style="color:#7A7A7A">per 100 chips</span><br>Skip <b>{adv["skip"]}</b>. Every spot '
                         f'loses a little over time, so keep bets small (~2% of your chips).')
            # where the advice comes from: the simulation, the AI's read of this shoe, and the blend of the two
            rows = [("Exact odds", BAC_EXACT), ("Simulation", adv["sim"]), ("AI's read", adv["ai"]),
                    ("Blend", adv["blend"])]
            st.html('<table style="width:100%;font-size:12px;border-collapse:collapse;color:#C7C7C7">'
                    '<tr style="color:#7A7A7A"><td></td><td style="text-align:right">Player</td>'
                    '<td style="text-align:right">Banker</td><td style="text-align:right">Tie</td></tr>'
                    + "".join(f'<tr><td style="padding:2px 0;{"font-weight:700;color:#FFF" if name == "Blend" else ""}">'
                              f'{name}</td>' + "".join(f'<td style="text-align:right">{ch[r]:.1%}</td>' for r in "PBT")
                              + "</tr>" for name, ch in rows) + "</table>")
            st.caption(f"Exact odds: worked out by math. Simulation: {adv['sim_hands']:,} hands, dealt once - it agrees "
                       f"with the exact odds within luck. AI: a model that reads the last results, the streak and how "
                       f"many of each card value are left (never their order or what's next), learning after every "
                       f"hand. Blend = {AI_WEIGHT:.0%} AI + {1 - AI_WEIGHT:.0%} exact odds, and it only advises "
                       f"against {adv['math_best']} for a {SWITCH_MARGIN:.0f}+ chip edge.")
            score = g["ai_score"]
            if score["total"]:
                last_pick, last_win = g.get("ai_last", (None, None))
                st.caption(f"Last hand: it advised **{g.get('advice_last', '-')}** - it was **{BAC_NAME[last_win]}**")
                rate = lambda who: score.get(f"{who}_won", 0) / max(score.get(f"{who}_won", 0) + score.get(f"{who}_lost", 0), 1)
                won, lost = score.get("advice_won", 0), score.get("advice_lost", 0)
                ai_rate_caption([won, lost, max(score["total"] - won - lost, 0)], "hands",
                                "betting its advice every hand")
                st.caption(f"Always Banker: **{rate('banker'):.0%}**. Long run, Banker wins ~50.7% of the hands that "
                           f"don't tie - but its 5% commission still leaves the house ahead.")
                st.caption(f"Following its advice (1 chip a hand): **{score.get('advice_profit', 0):+.2f}** · "
                           f"always Banker: **{score.get('banker_profit', 0):+.2f}** over {score['total']} hands. "
                           f"Its plain guess was right {score['right']} of {score['total']}.")
        st.caption("Tap a chip in the rack (gold ring = picked), then tap a spot on the table to drop it there. "
                   "You can bet more than one spot at once. Your chips stay for the next hand - Clear takes them back.")
        st.html('<table class="vp-pay" style="width:100%;font-size:13px">' + "".join(
            f'<tr><td style="padding:4px 8px;color:#C7C7C7">{spot}</td>'
            f'<td style="padding:4px 8px;text-align:right;font-weight:700">{pays}</td></tr>'
            for spot, pays in BAC_PAYS.items()) + "</table>")


# ---------------------------------------------------------------- video poker (Jacks or Better) - fake chips, for fun
# Deal 5 cards, hold the ones you want, draw once. Paid on the final hand (the standard "9/6" pay table).
# A fresh 52-card deck every hand, like the machines. Uses the same chips as the other games.
VP_PAYS = [("Royal Flush", 250), ("Straight Flush", 50), ("Four of a Kind", 25), ("Full House", 9),
           ("Flush", 6), ("Straight", 4), ("Three of a Kind", 3), ("Two Pair", 2), ("Jacks or Better", 1)]
VP_VALUE = {r: v for v, r in enumerate(BJ_RANKS[1:], start=2)} | {"A": 14}   # 2-10, J=11, Q=12, K=13, A=14


def vp_hand_name(hand):
    """the best poker hand in 5 cards that pays (None = no win)"""
    values = [VP_VALUE[r] for r, _ in hand]
    counts = sorted(Counter(values).values(), reverse=True)
    flush = len({s for _, s in hand}) == 1
    unique = sorted(set(values))
    straight = len(unique) == 5 and (unique[-1] - unique[0] == 4 or unique == [2, 3, 4, 5, 14])   # A-2-3-4-5 counts
    if straight and flush:
        return "Royal Flush" if unique[0] == 10 else "Straight Flush"
    if counts[0] == 4:
        return "Four of a Kind"
    if counts == [3, 2]:
        return "Full House"
    if flush:
        return "Flush"
    if straight:
        return "Straight"
    if counts[0] == 3:
        return "Three of a Kind"
    if counts == [2, 2, 1]:
        return "Two Pair"
    pairs = [v for v, n in Counter(values).items() if n == 2]
    return "Jacks or Better" if pairs and pairs[0] >= 11 else None


def vp_game():
    return st.session_state.setdefault("vp", {"deck": [], "hand": [], "held": [False] * 5, "phase": "bet", "bet": 25,
                                              "stake": 0, "msg": "", "record": [0, 0, 0], "new": [], "win": None})


def vp_add_chip(value):
    g = vp_game()
    g["bet"] = (g.get("bet") or 0) + value
    g["drop"], g["new"] = True, []


def vp_clear():
    g = vp_game()
    g["bet"], g["new"] = 0, []


def vp_deal():
    g, wallet = vp_game(), bj_game()
    if g["bet"] < 10:
        g["msg"] = "Put some chips in the betting circle first."
        return
    if g["bet"] > wallet["chips"]:
        g["msg"] = "Not enough chips for that bet - pick a smaller one."
        return
    wallet["chips"] -= g["bet"]
    g["deck"] = [(rank, suit) for rank in BJ_RANKS for suit in BJ_SUITS]
    random.shuffle(g["deck"])
    g["hand"] = [g["deck"].pop() for _ in range(5)]
    name = vp_hand_name(g["hand"])
    g.update(stake=g["bet"], held=[False] * 5, phase="hold", new=list(range(5)), win=None,
             msg=f"You have {name} - hold your cards and draw." if name else "Pick the cards to hold, then draw.")


def vp_toggle(i):
    g = vp_game()
    g["held"][i] = not g["held"][i]
    g["new"] = []   # nothing new dealt - don't replay the deal animation


def vp_draw():
    g, wallet = vp_game(), bj_game()
    # did you hold what the AI advised? (it only ever looked at your 5 cards - never the deck)
    held_mask = sum(1 << i for i in range(5) if g["held"][i])
    holds = vp_best_holds(tuple(g["hand"]))
    (best_ev, ai_mask), your_ev = holds[0], next(ev for ev, m in holds if m == held_mask)
    followed = held_mask == ai_mask
    # trainer: any hold worth as much as the best one counts as right (some hands have two equal holds)
    trainer_grade("vp", your_ev >= best_ev - 1e-9, True,
                  f"Best hold: **{vp_mask_cards(g['hand'], ai_mask)}** - gives back {best_ev:.2f} per chip. "
                  + ("That's what you held." if followed else
                     f"You held {vp_mask_cards(g['hand'], held_mask)}, which gives back {your_ev:.2f}."))
    # the AI's own hand: its hold, with the SAME next cards off the deck that you'd get
    upcoming = list(g["deck"])
    ai_hand = [card if ai_mask >> i & 1 else upcoming.pop() for i, card in enumerate(g["hand"])]
    ai_multiple = dict(VP_PAYS).get(vp_hand_name(ai_hand), 0)
    g["new"] = [i for i in range(5) if not g["held"][i]]   # replace every card you didn't hold
    for i in g["new"]:
        g["hand"][i] = g["deck"].pop()
    name = vp_hand_name(g["hand"])
    multiple = dict(VP_PAYS).get(name, 0)
    pay = g["stake"] * multiple
    wallet["chips"] += pay
    result = 0 if pay > g["stake"] else 2 if pay == g["stake"] else 1
    g["record"][result] += 1
    track = g.setdefault("ai_track", {"followed": 0, "total": 0, "record": [0, 0, 0]})
    track["total"] += 1
    track.setdefault("ai_record", [0, 0, 0])[0 if ai_multiple > 1 else 2 if ai_multiple == 1 else 1] += 1
    if followed:
        track["followed"] += 1
        track["record"][result] += 1   # your record on the hands where you held what the AI said
    g["win"] = name
    g["phase"] = "done"
    g["msg"] = (f"{name}! It pays {multiple} for 1 - you get {pay:,} back." if name == "Jacks or Better"
                else f"{name}! Pays {multiple} for 1 - you win {pay - g['stake']:,}." if name
                else f"No win - you lose {g['stake']:,}.")


# ---- video poker strategy: the best hold for any hand, and simulations of simple strategies ----
VP_PAY = dict(VP_PAYS)
VP_DECK = [(rank, suit) for rank in BJ_RANKS for suit in BJ_SUITS]


@st.cache_data(max_entries=300, show_spinner=False)
def vp_best_holds(hand):
    """[(expected return "for 1", which cards to hold as a 5-bit mask), ...] for all 32 ways to hold this hand,
    best first. Exact when 0-2 cards get replaced; estimated from 3,000 random draws when 3-5 do."""
    rest = [c for c in VP_DECK if c not in hand]
    rng = random.Random(str(hand))   # same hand -> same estimate every time
    holds = []
    for mask in range(32):
        held = [hand[i] for i in range(5) if mask >> i & 1]
        need = 5 - len(held)
        if need <= 2:   # every possible draw (at most 1,081)
            pays = [VP_PAY.get(vp_hand_name(held + list(d)), 0) for d in itertools.combinations(rest, need)]
        else:           # too many to try them all - a big random sample instead
            pays = [VP_PAY.get(vp_hand_name(held + rng.sample(rest, need)), 0) for _ in range(3000)]
        holds.append((sum(pays) / len(pays), mask))
    return sorted(holds, reverse=True)


def vp_mask_cards(hand, mask):
    return " ".join(f"{r}{s}" for i, (r, s) in enumerate(hand) if mask >> i & 1) or "nothing (draw 5 new cards)"


def vp_simple_hold(hand):
    """a simple strategy most people can remember: keep paying straights and better, keep any pair / trips /
    quads, keep 4 cards to a flush, otherwise keep your jacks, queens, kings and aces"""
    if vp_hand_name(hand) in ("Royal Flush", "Straight Flush", "Four of a Kind", "Full House", "Flush", "Straight"):
        return 31
    counts = Counter(r for r, _ in hand)
    if max(counts.values()) >= 2:
        return sum(1 << i for i, (r, _) in enumerate(hand) if counts[r] >= 2)
    suits = Counter(s for _, s in hand)
    suit, n = suits.most_common(1)[0]
    if n == 4:
        return sum(1 << i for i, (_, s) in enumerate(hand) if s == suit)
    return sum(1 << i for i, (r, _) in enumerate(hand) if r in ("J", "Q", "K", "A"))


def vp_simulate(hands):
    """play `hands` hands three ways on the SAME deals and compare how much each gives back per chip bet"""
    styles = {"Simple strategy (pairs, 4-flushes, high cards)": vp_simple_hold,
              "Keep paying hands and pairs only": lambda h: 31 if vp_hand_name(h) in (
                  "Royal Flush", "Straight Flush", "Four of a Kind", "Full House", "Flush", "Straight")
              else sum(1 << i for i, (r, _) in enumerate(h) if Counter(x for x, _ in h)[r] >= 2),
              "Hold nothing - always draw 5 new": lambda h: 0}
    totals, wins = dict.fromkeys(styles, 0), dict.fromkeys(styles, 0)
    for _ in range(hands):
        cards = random.sample(VP_DECK, 10)   # the 5 dealt + the next 5 off the deck
        hand, extra = cards[:5], cards[5:]
        for name, strategy in styles.items():
            mask = strategy(hand)
            final = [hand[i] for i in range(5) if mask >> i & 1]
            final += extra[:5 - len(final)]
            pays = VP_PAY.get(vp_hand_name(final), 0)
            totals[name] += pays
            wins[name] += pays >= 2   # a real win (two pair or better) - jacks or better only gives the bet back
    # perfect play's numbers are known exactly for this pay table: 99.54% return, and it ends in two pair or better
    # about 24.0% of the time (two pair 12.9%, three of a kind 7.4%, straights / flushes / full houses ~3.4%, ...)
    rows = [{"Strategy": "Perfect play (the best hold every hand)", "Return": 0.9954,
             "Profit per 100 chips bet": -0.46, "Win rate": 0.240,
             "How it's known": "worked out exactly by math, for this pay table"}]
    rows += [{"Strategy": name, "Return": total / hands, "Profit per 100 chips bet": (total / hands - 1) * 100,
              "Win rate": wins[name] / hands, "How it's known": f"simulated, {hands:,} hands"}
             for name, total in totals.items()]
    return pd.DataFrame(rows)


def vp_use_best():
    g = vp_game()
    best_mask = vp_best_holds(tuple(g["hand"]))[0][1]
    g["held"] = [bool(best_mask >> i & 1) for i in range(5)]
    g["new"] = []


@st.fragment
def video_poker():
    """the video poker machine - a fragment, so holding / drawing only redraws the game"""
    g, wallet = vp_game(), bj_game()
    if g["phase"] != "hold" and g["bet"] > wallet["chips"]:
        g["bet"] = 0   # can't cover last hand's bet any more
    st.html(CARD_CSS + CHIP_CSS + """<style>
      .vp-slot { display:flex; justify-content:center; padding-top:6px; }
      .vp-slot .bj-card { width:96px; height:136px; font-size:30px; padding:8px 10px; }
      .vp-slot .bj-card .bj-suit { font-size:38px; }
      .vp-slot .bj-card.held { outline:3px solid #22C55E; outline-offset:3px; }
      .vp-slot .bj-card.tip  { box-shadow:0 0 0 3px #FBBF24, 0 0 18px rgba(251,191,36,.6); }
      .vp-empty { width:96px; height:136px; border-radius:10px; border:2px dashed rgba(167,243,208,.25); }
      .vp-pay { width:100%; border-collapse:collapse; font-size:13px; }
      .vp-pay td { padding:4px 8px; border-bottom:1px solid #161616; color:#C7C7C7; }
      .vp-pay td:last-child { text-align:right; font-weight:700; }
      .vp-pay tr.hit td { background:#22C55E; color:#000; font-weight:800; }
    </style>""")
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="vp_table"):
            msg = html.escape(g["msg"]) or "Pick a bet and deal 5 cards."
            st.html(f'<div class="bj-msg" style="margin-bottom:6px">{msg}</div>')
            # the strategy helper: which cards give back the most, on average?
            best_mask = None
            if g["phase"] == "hold" and not trainer_on("vp") and st.toggle("Show the best hold", key="vp_hint",
                                                  help="Works out the expected return of all 32 ways to hold this "
                                                       "hand and suggests the best one - the gold-outlined cards"):
                with st.spinner("Working out every way to hold..."):
                    holds = vp_best_holds(tuple(g["hand"]))
                best_mask = holds[0][1]
                current = sum(1 << i for i in range(5) if g["held"][i])
                yours = next(ev for ev, m in holds if m == current)
                lines = "".join(f'<div>{"Best" if n == 0 else f"#{n + 1}"}: hold <b>{vp_mask_cards(g["hand"], m)}</b>'
                                f' - gives back <b>{ev:.2f}</b> per chip</div>' for n, (ev, m) in enumerate(holds[:3]))
                st.html(f'<div style="color:#E5E7EB;font-size:13px;line-height:1.7;text-align:center">{lines}'
                        f'<div style="color:#A7F3D0">Your hold right now: {yours:.2f} per chip</div></div>')
                if current != best_mask:
                    st.button("Use the best hold", on_click=vp_use_best, key="vp_use_best",
                              icon=":material/auto_fix_high:", width="stretch")
            slots = st.columns(5)
            for i, slot in enumerate(slots):
                with slot:
                    if not g["hand"]:
                        st.html('<div class="vp-slot"><div class="vp-empty"></div></div>')
                        continue
                    rank, suit = g["hand"][i]
                    order = g["new"].index(i) if i in g["new"] else None
                    anim = f' new" style="animation-delay:{order * DEAL_GAP:.2f}s' if order is not None else ""
                    held = " held" if g["held"][i] and g["phase"] != "bet" else ""
                    if best_mask is not None and best_mask >> i & 1:
                        held += " tip"   # the advisor says: hold this one
                    st.html(f'<div class="vp-slot"><span class="bj-card {"red" if BJ_SUITS[suit] else ""}{held}'
                            f'{anim}"><b>{rank}</b><span class="bj-suit">{suit}</span></span></div>')
                    if g["phase"] == "hold":
                        st.button("Held" if g["held"][i] else "Hold", key=f"vp_hold_{i}", on_click=vp_toggle,
                                  args=(i,), width="stretch", type="primary" if g["held"][i] else "secondary")
            g["new"] = []   # shown - these cards won't animate again
            # the betting circle: your chips sit here
            st.html(f'<div class="pq-circle">'
                    f'{chip_stack_html(g["stake"] if g["phase"] == "hold" else g["bet"], g.pop("drop", False))}</div>')
            if g["phase"] == "hold":
                st.button("Draw", on_click=vp_draw, key="vp_draw", type="primary", width="stretch",
                          icon=":material/autorenew:")
            elif wallet["chips"] >= 10:
                chip_rack("vp", vp_add_chip, lambda v: g["bet"] + v <= wallet["chips"])
                clear_col, deal_col = st.columns(2)
                clear_col.button("Clear", on_click=vp_clear, key="vp_clear", width="stretch", icon=":material/close:",
                                 disabled=not g["bet"])
                deal_col.button(f"Deal ({g['bet']:,} chips)" if g["bet"] else "Deal", on_click=vp_deal, key="vp_deal",
                                width="stretch", type="primary", icon=":material/playing_cards:", disabled=g["bet"] < 10)
        with st.expander("How video poker works", icon=":material/school:"):
            st.markdown(
                "**Jacks or Better:** you're dealt 5 cards from a fresh deck. Click **Hold** on the ones you want to "
                "keep, then **Draw** - every card you didn't hold is replaced, once. You're paid on the final hand.\n\n"
                "**The lowest paying hand is a pair of jacks or better** (J-J, Q-Q, K-K or A-A) - it gives your bet "
                "back. Pays are \"for 1\": 2 for 1 on a 10-chip bet returns 20.\n\n"
                "**Basic tips:** always keep a paying hand; keep any pair (even small ones); keep 4 cards to a flush "
                "or royal flush; with nothing, keep your jacks, queens, kings and aces. With perfect play this pay table "
                "returns about 99.5% - so even played perfectly, the house still keeps a little. Fake chips - just for fun.")
        strategy_lab("vp")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every card game")
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True)
        # ---- the AI: advice only - you pick the holds yourself ----
        with st.container(border=True):
            st.markdown("**:material/psychology: Video Poker AI**")
            trainer_switch("vp", "picked your holds and drawn")
            quiet = g["phase"] == "hold" and trainer_on("vp")
            if quiet:
                headline = why = ""
            elif g["phase"] == "hold":
                holds = vp_best_holds(tuple(g["hand"]))
                best_ev, best = holds[0]
                current = sum(1 << i for i in range(5) if g["held"][i])
                yours = next(ev for ev, m in holds if m == current)
                headline = f"Hold {vp_mask_cards(g['hand'], best)}" if best else "Draw 5 new cards"
                why = (f"Of all 32 ways to hold this hand, this gives back the most: <b>{best_ev:.2f}</b> per chip. "
                       + ("You're holding exactly that." if current == best else
                          f"What you're holding now gives back {yours:.2f} - tap Hold on the cards it names."))
            else:
                unit = max(10, round(wallet["chips"] * 0.01 / 5) * 5)
                headline = f"Bet {unit:,} chips"
                why = ("Same size every hand, about 1% of your chips. Video poker deals from a fresh deck every hand, "
                       "so there's nothing to count - the only edge is holding the right cards, and that's what "
                       "this AI helps with once you deal.")
            if quiet:
                trainer_quiet()
            else:
                ai_box(headline, why)
            if trainer_on("vp"):
                trainer_panel("vp")
            track = g.get("ai_track")
            if track and sum(track.get("ai_record", [])):
                ai_rate_caption(track["ai_record"], "hands", "holding its own pick every hand (Jacks or Better = a "
                                "push; perfect play wins about 31% of the hands that don't push)")
            if track and track["total"]:
                w, l, p = track["record"]
                st.caption(f"You held what it said on **{track['followed']} of {track['total']}** hands"
                           + (f" - on those: **{w}-{l}-{p}**, a real win (two pair or better) "
                              f"**{w / max(w + l + p, 1):.0%}** of the time (perfect play: ~24%)." if sum(track["record"]) else "."))
            st.caption("The AI only ever looks at your 5 cards - never the deck or what you'd draw next. For each way "
                       "to hold, it works out every possible draw (or 3,000 random ones) and picks the best. "
                       "Turn on 'Show the best hold' on the table to see its picks outlined in gold.")
        st.html('<table class="vp-pay">' + "".join(
            f'<tr class="{"hit" if g["phase"] == "done" and g["win"] == name else ""}"><td>{name}</td><td>{mult}</td></tr>'
            for name, mult in VP_PAYS) + "</table>")
        if g["phase"] == "hold":
            st.caption(f"In play: {g['stake']:,} chips")
        elif wallet["chips"] < 10:
            st.button("Out of chips - get 1,000 more", on_click=bj_reset, key="vp_reset", icon=":material/refresh:",
                      width="stretch", type="primary")


# ---------------------------------------------------------------- roulette (American: 0 and 00) - fake chips
# Put chips on numbers or outside bets, spin the wheel. 38 pockets: 1-36 plus TWO green ones (0 and 00), so almost
# every bet loses 5.26% in the long run. Inside the code 00 is pocket number 37 (shown as "00" everywhere).
RL_POCKETS = 38
RL_ORDER = [0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22, 34, 15, 3, 24, 36, 13, 1, 37, 27, 10, 25, 29, 12, 8, 19, 31,
            18, 6, 21, 33, 16, 4, 23, 35, 14, 2]   # the real order of the pockets around an American wheel (37 = 00)
RL_RED = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
# every bet: (wins on pocket n?, pays X to 1, label on the table)
def rl_name(n):
    return "00" if n == 37 else str(n)


def rl_num(n):   # 1-36 (both green pockets, 0 and 00, lose every outside bet)
    return 1 <= n <= 36


RL_BETS = {f"n{k}": ((lambda n, k=k: n == k), 35, rl_name(k)) for k in range(RL_POCKETS)} | {
    "red": (lambda n: n in RL_RED, 1, "RED"), "black": (lambda n: rl_num(n) and n not in RL_RED, 1, "BLACK"),
    "odd": (lambda n: rl_num(n) and n % 2 == 1, 1, "ODD"), "even": (lambda n: rl_num(n) and n % 2 == 0, 1, "EVEN"),
    "low": (lambda n: 1 <= n <= 18, 1, "1-18"), "high": (lambda n: 19 <= n <= 36, 1, "19-36"),
    "d1": (lambda n: 1 <= n <= 12, 2, "1st 12"), "d2": (lambda n: 13 <= n <= 24, 2, "2nd 12"),
    "d3": (lambda n: 25 <= n <= 36, 2, "3rd 12"),
    "c1": (lambda n: rl_num(n) and n % 3 == 1, 2, "2:1"), "c2": (lambda n: rl_num(n) and n % 3 == 2, 2, "2:1"),
    "c3": (lambda n: rl_num(n) and n % 3 == 0, 2, "2:1")}


def rl_color(n):
    return "#15803D" if n in (0, 37) else "#B91C1C" if n in RL_RED else "#141414"


def rl_wheel_parts(labels=True):
    """the wheel drawing (SVG pieces, 300 x 300): 38 pockets in the real order, numbers, a hub"""
    c, r, step = 150, 144, 360 / RL_POCKETS
    parts = [f'<circle cx="{c}" cy="{c}" r="{r + 4}" fill="#5B3A1E" stroke="#C8A24A" stroke-width="3"/>']
    for i, n in enumerate(RL_ORDER):
        a0, a1 = math.radians(i * step - step / 2 - 90), math.radians(i * step + step / 2 - 90)
        parts.append(f'<path d="M{c} {c} L{c + r * math.cos(a0):.2f} {c + r * math.sin(a0):.2f} '
                     f'A{r} {r} 0 0 1 {c + r * math.cos(a1):.2f} {c + r * math.sin(a1):.2f} Z" '
                     f'fill="{rl_color(n)}" stroke="#C8A24A" stroke-width=".8"/>')
        if labels:
            parts.append(f'<text x="{c}" y="{c - r * 0.84:.1f}" transform="rotate({i * step:.2f} {c} {c})" '
                         f'text-anchor="middle" font-size="12" font-weight="800" fill="#FFF" '
                         f'font-family="Inter,sans-serif">{rl_name(n)}</text>')
    parts.append(f'<circle cx="{c}" cy="{c}" r="{r * 0.62:.1f}" fill="#0B2E1A" stroke="#C8A24A" stroke-width="3"/>'
                 f'<circle cx="{c}" cy="{c}" r="{r * 0.16:.1f}" fill="#C8A24A"/>')
    return "".join(parts)


RL_WHEEL_IMG = "data:image/svg+xml;base64," + base64.b64encode(
    f'<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">{rl_wheel_parts()}</svg>'.encode()).decode()


def rl_game():
    g = st.session_state.setdefault("rl", {"bets": {}, "chip": 25, "angle": 0.0, "spins": 0, "result": None,
                                           "spun": False, "msg": "", "record": [0, 0, 0], "history": []})
    return g


def rl_pick_chip(value):
    rl_game()["chip"] = value


# opposite even-money bets: you can back one side only - betting both just loses on 0 and 00 (house rule here)
RL_OPPOSITE = {"red": "black", "black": "red", "odd": "even", "even": "odd", "low": "high", "high": "low"}


def rl_place(key):
    """drop the picked chip on a bet"""
    g, wallet = rl_game(), bj_game()
    other = RL_OPPOSITE.get(key)
    if other and g["bets"].get(other):
        g["msg"] = (f"You already have chips on {RL_BETS[other][2].title()} - you can only back one side. "
                    f"Clear the table to switch.")
        return
    if sum(g["bets"].values()) + g["chip"] > wallet["chips"]:
        g["msg"] = "Not enough chips for that - pick a smaller chip or clear the table."
        return
    g["bets"][key] = g["bets"].get(key, 0) + g["chip"]
    g["spun"], g["last_chip"] = False, key


def rl_clear():
    g = rl_game()
    g["bets"], g["spun"] = {}, False


def rl_payout(bets, n):
    """chips back if the ball lands on n (your stake on each winning bet + its winnings)"""
    return sum(amount * (RL_BETS[key][1] + 1) for key, amount in bets.items() if RL_BETS[key][0](n))


def rl_quiz_key(g):
    """one quiz question per spin and set of bets"""
    return g["spins"], tuple(sorted(g["bets"].items()))


def rl_quiz_pocket(g):
    """the pocket the quiz asks about - one where your bets win, most of the time"""
    rng = random.Random(str(rl_quiz_key(g)))
    wins = [n for n in range(RL_POCKETS) if rl_payout(g["bets"], n)]
    return rng.choice(wins) if wins and rng.random() < 0.75 else rng.randrange(RL_POCKETS)


def rl_quiz_check(pocket):
    g = rl_game()
    answer, guess = rl_payout(g["bets"], pocket), int(st.session_state.get("rl_quiz_guess") or 0)
    winners = [f"{RL_BETS[k][2]} ({amount:,} chips, pays {RL_BETS[k][1]} to 1)"
               for k, amount in g["bets"].items() if RL_BETS[k][0](pocket)]
    trainer_grade("rl", guess, answer,
                  f"On {rl_name(pocket)} you get back **{answer:,}** chips (you said {guess:,}). "
                  + ("Winning bets: " + "; ".join(winners) + " - each one pays its odds AND gives your chips back."
                     if winners else "None of your bets win on that pocket - everything on the table is lost."))
    g["quiz_asked"] = rl_quiz_key(g)


def rl_spin():
    g, wallet = rl_game(), bj_game()
    total = sum(g["bets"].values())
    if not total:
        g["msg"] = "Put some chips on the table first."
        return
    if total > wallet["chips"]:
        g["msg"] = "Not enough chips for everything on the table - clear it and bet smaller."
        return
    wallet["chips"] -= total
    n = random.randrange(RL_POCKETS)   # the ball - decided only now, after every bet (and the AI's advice) is in
    pay = rl_payout(g["bets"], n)
    net = pay - total
    wallet["chips"] += pay
    g["record"][0 if net > 0 else 2 if net == 0 else 1] += 1
    if g.get("ai_bet"):   # how the AI's suggested bet would have done on this same spin
        ai_net = rl_payout(g["ai_bet"], n) - sum(g["ai_bet"].values())
        g.setdefault("ai_record", [0, 0, 0])[0 if ai_net > 0 else 2 if ai_net == 0 else 1] += 1
    plan = g.get("plan")
    if plan and plan.get("next"):   # the strategy you're following moves on (Martingale doubles after a loss, ...)
        won = rl_payout(plan["next"], n) > sum(plan["next"].values())
        plan["level"] = float(rl_level(plan["plan"][2], plan["level"], won))
        plan["spins"] += 1
    g["history"] = (g["history"] + [n])[-30:]
    # turn the wheel: at least 5 full turns, ending with pocket n under the pointer at the top
    step = 360 / RL_POCKETS
    target = (360 - RL_ORDER.index(n) * step) % 360
    g["from"] = g["angle"]
    g["angle"] = g["angle"] + 360 * 5 + (target - g["angle"] % 360) % 360
    g["spins"] += 1
    g["result"], g["spun"] = n, True
    colour = "green" if n in (0, 37) else "red" if n in RL_RED else "black"
    g["msg"] = (f"{rl_name(n)} {colour} - " + (f"you win {net:,}!" if net > 0 else f"you lose {-net:,}." if net < 0
                                      else "you break even."))


def rl_analysis(bets):
    """the AI's look at the bets on the table: for every pocket, what you'd get back"""
    total = sum(bets.values())
    backs = [rl_payout(bets, n) for n in range(RL_POCKETS)]
    return {"total": total, "profit_pockets": sum(b > total for b in backs), "any_back": sum(b > 0 for b in backs),
            "best": max(backs) - total, "expected": sum(backs) / RL_POCKETS - total}


def rl_simulate(spins):
    """the Strategy lab: five ways of betting on the SAME spins"""
    results = [random.randrange(RL_POCKETS) for _ in range(spins)]
    rows = []

    def flat(name, bets):
        total, staked, won = 0, 0, 0
        for n in results:
            net = rl_payout(bets, n) - sum(bets.values())
            total, staked, won = total + net, staked + sum(bets.values()), won + (net > 0)
        rows.append({"Strategy": name, "Win rate": won / spins, "Profit per 100 chips bet": total / staked * 100})

    flat("Flat bet on Red", {"red": 1})
    flat("Flat bet on one number (17)", {"n17": 1})
    flat("Two dozens (1st 12 + 2nd 12)", {"d1": 1, "d2": 1})
    # Martingale: double the bet after every loss, back to 1 after a win (stops doubling at the 500-chip table max)
    bet, total, staked, won = 1, 0, 0, 0
    for n in results:
        staked += bet
        if n in RL_RED:
            total, won, bet = total + bet, won + 1, 1
        else:
            total, bet = total - bet, min(bet * 2, 500)
    rows.append({"Strategy": "Martingale on Red (double after a loss)", "Win rate": won / spins,
                 "Profit per 100 chips bet": total / staked * 100})
    # "hot number": always bet the number that came up most in the last 38 spins
    total, won, seen = 0, 0, []
    for n in results:
        hot = Counter(seen[-38:]).most_common(1)[0][0] if seen else 17
        total, won = total + (35 if n == hot else -1), won + (n == hot)
        seen.append(n)
    rows.append({"Strategy": "Bet the 'hot' number (most common lately)", "Win rate": won / spins,
                 "Profit per 100 chips bet": total / spins * 100})
    # ---- the famous "progression" systems, all on Red: they change the bet size, never the odds ----
    def progression(name, next_bet):
        """next_bet(bet, won, state) -> the next bet size (1-500 chips)"""
        bet, total, staked, won_spins, state = 1, 0, 0, 0, {}
        for n in results:
            staked += bet
            win = n in RL_RED
            total += bet if win else -bet
            won_spins += win
            bet = max(1, min(500, next_bet(bet, win, state)))
        rows.append({"Strategy": name, "Win rate": won_spins / spins, "Profit per 100 chips bet": total / staked * 100})

    def paroli(bet, win, state):   # let a win ride: double after a win, back to 1 after 3 wins in a row or a loss
        state["streak"] = state.get("streak", 0) + 1 if win else 0
        if state["streak"] >= 3:
            state["streak"] = 0
            return 1
        return bet * 2 if win else 1

    def fibonacci(bet, win, state):   # 1,1,2,3,5,8...: one step up after a loss, two steps back after a win
        seq, i = [1, 1], state.get("i", 0)
        while len(seq) < 30:
            seq.append(seq[-1] + seq[-2])
        state["i"] = min(i + 1, 29) if not win else max(i - 2, 0)
        return seq[state["i"]]

    progression("Paroli on Red (let wins ride, 3 times)", paroli)
    progression("D'Alembert on Red (+1 after a loss, -1 after a win)", lambda bet, win, s: bet - 1 if win else bet + 1)
    progression("Fibonacci on Red", fibonacci)
    # James Bond: 14 on 19-36, 5 on the 13-18 six-line (pays 5 to 1), 1 on 0 - 20 chips a spin
    total, won = 0, 0
    for n in results:
        back = (28 if 19 <= n <= 36 else 0) + (30 if 13 <= n <= 18 else 0) + (36 if n == 0 else 0)
        total, won = total + back - 20, won + (back > 20)
    rows.append({"Strategy": "James Bond (14 on 19-36, 5 on 13-18, 1 on 0)", "Win rate": won / spins,
                 "Profit per 100 chips bet": total / (20 * spins) * 100})
    for row in rows:
        row["Long run (exact math)"] = -5.26   # every bet: 2 green pockets out of 38 = 5.26% house edge
    return pd.DataFrame(rows), rl_goal_sim()


def rl_goal_sim(trials=4000, start=100, goal=200):
    """a different question: start with 100 chips - which way of betting most often reaches 200 before going broke?
    (the bold-play result: with a house edge, FEW BIG bets beat many small ones)"""
    def run(next_bet):
        reached = 0
        for _ in range(trials):
            chips, state = start, {}
            for _ in range(20_000):
                if chips <= 0 or chips >= goal:
                    break
                bet = max(1, min(chips, next_bet(chips, state)))
                win = random.randrange(RL_POCKETS) in RL_RED
                chips += bet if win else -bet
                state["won"] = win
            reached += chips >= goal
        return reached / trials

    def martingale(chips, s):
        s["bet"] = 5 if s.get("won", True) else s["bet"] * 2
        return s["bet"]

    styles = {"Bold play: bet everything you need on Red (all-in)": lambda chips, s: min(chips, goal - chips),
              "Big bets: 25 chips on Red each spin": lambda chips, s: 25,
              "Small bets: 5 chips on Red each spin": lambda chips, s: 5,
              "Martingale (5 chips, double after a loss)": martingale}
    return pd.DataFrame([{"Way of betting": name, "Reached 200": run(fn)} for name, fn in styles.items()])


# ---- the AI strategy finder: invent strategies, play each through thousands of sessions, keep the best for a goal ----
# A strategy = which bet + how big + a betting system + when to stop. No strategy beats the 5.26% edge - but they
# differ hugely in HOW you lose or win: how often a session ends ahead, how often you double, how often you go broke.
RL_RED_ARR = np.array(sorted(RL_RED))
RL_PLAN_BETS = {   # name: (wins when the ball lands..., net win per chip, the table bets it means)
    "Red": (lambda r: np.isin(r, RL_RED_ARR), 1, lambda s: {"red": s}),
    "One dozen (1st 12)": (lambda r: (r >= 1) & (r <= 12), 2, lambda s: {"d1": s}),
    "Two dozens (1st + 2nd 12)": (lambda r: (r >= 1) & (r <= 24), 0.5,
                                  lambda s: {"d1": max(10, round(s / 10) * 5), "d2": max(10, round(s / 10) * 5)}),
    "One number (17)": (lambda r: r == 17, 35, lambda s: {"n17": s}),
}
RL_SYSTEMS = {
    "Flat": "the same bet every spin",
    "Martingale": "double the bet after every loss, back to the start after a win",
    "Paroli": "double after a win (up to 3 in a row), back to the start after a loss",
    "D'Alembert": "one unit more after a loss, one unit less after a win",
    "Fibonacci": "one step up the 1-1-2-3-5-8-13 ladder after a loss, two steps down after a win",
    "Bold": "bet exactly what you need to reach your target in one win (or everything you have left)",
}
RL_FIB = np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377])
RL_GOALS = {   # goal: (score of a strategy's results, what it measures)
    "Finish ahead most often": (lambda m: m["ahead"], "sessions that end with more chips than they started"),
    "Lose the least": (lambda m: m["average"], "the average result of a session"),
    "Double my chips": (lambda m: m["double"], "sessions that end with at least double the chips"),
    "Ahead, but safe": (lambda m: m["ahead"] - 4 * max(0, m["bust"] - 0.01),
                        "sessions that end ahead, while going broke 1% of the time or less"),
}


def rl_stake(system, base, level, bank, target, pays):
    """how much a betting system bets next (works on one session, or on a numpy array of them at once)"""
    stake = {"Flat": lambda: base + 0 * level, "Martingale": lambda: base * 2.0 ** level,
             "Paroli": lambda: base * 2.0 ** level, "D'Alembert": lambda: base * (1 + level),
             "Fibonacci": lambda: base * RL_FIB[np.minimum(level, len(RL_FIB) - 1).astype(int)],
             "Bold": lambda: (target - bank) / pays}[system]()
    return np.minimum(np.maximum(stake, 10), bank)   # the table minimum is 10 - and you can't bet more than you have


def rl_level(system, level, won):
    """where a betting system moves after a spin it won / lost"""
    if system in ("Martingale",):
        return np.where(won, 0, level + 1)
    if system == "Paroli":
        return np.where(won, np.where(level >= 2, 0, level + 1), 0)
    if system == "D'Alembert":
        return np.where(won, np.maximum(level - 1, 0), level + 1)
    if system == "Fibonacci":
        return np.where(won, np.maximum(level - 2, 0), level + 1)
    return level * 0


def rl_session_sim(plan, sessions, seed, start=1000.0):
    """play one strategy through many sessions at once - every session spins together, one spin at a time"""
    bet, size, system, stop_win, stop_loss, spins = plan
    hit, pays, _ = RL_PLAN_BETS[bet]
    rng = np.random.default_rng(seed)
    bank, level = np.full(sessions, start), np.zeros(sessions)
    wagered, played, active = np.zeros(sessions), np.zeros(sessions), np.ones(sessions, bool)
    target = start * (1 + stop_win) if stop_win else np.inf
    floor = start * (1 - stop_loss) if stop_loss else 0
    for _ in range(spins):
        if not active.any():
            break
        stake = rl_stake(system, start * size, level, bank, target, pays)
        won = hit(rng.integers(0, RL_POCKETS, sessions))
        bank = bank + np.where(won, stake * pays, -stake) * active
        wagered, played = wagered + stake * active, played + active
        level = rl_level(system, level, won)
        active &= (bank < target) & (bank > floor) & (bank >= 10)
    result = (bank - start) / start
    # the average is EXACT, not simulated: every bet on this wheel loses 2 chips per 38 bet on average, so a session
    # loses 2/38 of everything it bets - simulating it instead just adds luck (and lucky long shots look like winners)
    return {"ahead": (bank > start + 1e-9).mean(), "average": -2 / 38 * wagered.mean() / start,
            "median": float(np.median(result)),
            "double": (bank >= 2 * start - 1e-9).mean(), "bust": (bank < 10).mean(), "spins": played.mean(),
            "per_bet": (bank - start).sum() / max(wagered.sum(), 1)}


def rl_plan_text(plan):
    bet, size, system, stop_win, stop_loss, spins = plan
    first = (f"Bet just enough on **{bet}** to reach your target in one win" if system == "Bold" else
             f"Start with **{size:.0%} of your chips** on **{bet}**, {RL_SYSTEMS[system]} ({system})")
    stops = ([f"up {stop_win:.0%}"] if stop_win else []) + ([f"down {stop_loss:.0%}"] if stop_loss else [])
    return (f"{first}. Stop when you're {' or '.join(stops)}, or after **{spins} spins**." if stops else
            f"{first}. Play **{spins} spins** (or until the chips run out).")


@st.cache_data(show_spinner=False, max_entries=8)
def rl_find_strategies(goal, tries=600, sessions=2000):
    """the search: random strategies, then the best ones tweaked (a small genetic search), then the top 5
    re-tested on 10,000 brand-new sessions - a strategy that only looked good by luck gets found out there"""
    score = RL_GOALS[goal][0]
    rng = random.Random(7)
    pick = lambda: (rng.choice(list(RL_PLAN_BETS)), rng.choice([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]),
                    rng.choice(list(RL_SYSTEMS)), rng.choice([0.1, 0.25, 0.5, 1.0, None]),
                    rng.choice([0.25, 0.5, None]), rng.choice([20, 50, 100, 200]))

    def fix(plan):   # Bold play needs a target to aim at
        plan = list(plan)
        if plan[2] == "Bold" and not plan[3]:
            plan[3] = 1.0
        return tuple(plan)

    tried = {}

    def test(plan):
        plan = fix(plan)
        if plan not in tried:
            tried[plan] = rl_session_sim(plan, sessions, seed=11)   # the same dice for every strategy - a fair race
        return plan

    # the classics go in first (tiny flat bets, bold play, Martingale ...) so the search never misses them
    for bet in RL_PLAN_BETS:
        for system in RL_SYSTEMS:
            for size, stop_win, stop_loss, spins in ((0.01, None, None, 20), (0.05, 0.25, 0.5, 50), (1.0, 1.0, None, 200)):
                test((bet, size, system, stop_win, stop_loss, spins))
    for _ in range(tries):
        test(pick())
    for _ in range(3):   # breed: tweak one part of each of the best 12
        for plan in sorted(tried, key=lambda p: score(tried[p]), reverse=True)[:12]:
            for _ in range(4):
                child, new = list(plan), pick()
                field = rng.randrange(6)
                child[field] = new[field]
                test(child)
    ranked = sorted(tried, key=lambda p: score(tried[p]), reverse=True)
    retested = [(plan, rl_session_sim(plan, 10_000, seed=99)) for plan in ranked[:5]]
    retested.sort(key=lambda pm: score(pm[1]), reverse=True)
    everything = pd.DataFrame([{"bet": p[0], "system": p[2], **m} for p, m in tried.items()])
    return retested, everything


def rl_follow(plan):
    rl_game()["plan"] = {"plan": plan, "start": bj_game()["chips"], "level": 0, "spins": 0}


def rl_unfollow():
    rl_game().pop("plan", None)


def rl_plan_now(g, chips):
    """the next bet the strategy you're following calls for: (table bets, None) - or (None, why it says stop)"""
    p = g["plan"]
    bet, size, system, stop_win, stop_loss, spins = p["plan"]
    start = p["start"]
    target = start * (1 + stop_win) if stop_win else None
    if target and chips >= target:
        return None, f"You reached your target: {chips:,} chips ({chips - start:+,}). The strategy says **stop and keep it**."
    if chips <= (start * (1 - stop_loss) if stop_loss else 0) or chips < 10:
        return None, f"You hit the stop-loss ({chips - start:+,}). The strategy says **stop here** - that's its job."
    if p["spins"] >= spins:
        return None, f"That's all {spins} spins ({chips - start:+,}). The strategy says **stop**."
    stake = float(rl_stake(system, start * size, p["level"], chips, target or np.inf, RL_PLAN_BETS[bet][1]))
    stake = int(max(10, min(chips, round(stake / 5) * 5)))
    return RL_PLAN_BETS[bet][2](stake), None


def rl_strategy_finder():
    with st.expander("AI strategy finder: it invents strategies and tests them", icon=":material/auto_awesome:"):
        st.caption("The AI builds strategies out of a bet (Red, a dozen, two dozens, one number), a bet size, a "
                   "betting system (flat, Martingale, Paroli, D'Alembert, Fibonacci, bold play) and when to stop. It "
                   "plays each one through 1,500 simulated sessions of 1,000 chips, breeds the best ones into new "
                   "variations, then re-tests the top 5 on 10,000 brand-new sessions.")
        goal_col, run_col = st.columns([3, 1], vertical_alignment="bottom")
        goal = goal_col.selectbox("What do you want from a session?", list(RL_GOALS), key="rl_find_goal",
                                  help="Each goal has a different best strategy")
        if run_col.button("Find strategies", key="rl_find_go", type="primary", width="stretch",
                          icon=":material/auto_awesome:"):
            with st.spinner("Inventing strategies and playing thousands of sessions..."):
                st.session_state["rl_found"] = (goal, rl_find_strategies(goal))
        found = st.session_state.get("rl_found")
        if not found:
            return
        goal, (retested, everything) = found
        best, m = retested[0]
        with st.container(border=True):
            st.markdown(f"**Best strategy for \"{goal}\"** ({RL_GOALS[goal][1]})")
            st.markdown(rl_plan_text(best))
            a, b, c, d = st.columns(4)
            a.metric("Finishes ahead", f"{m['ahead']:.0%}", border=True, help="Sessions that end with more chips")
            b.metric("Average session", f"{m['average'] * 1000:+,.0f}", border=True,
                     help="Average chips won or lost per 1,000-chip session")
            c.metric("Doubles up", f"{m['double']:.0%}", border=True)
            d.metric("Goes broke", f"{m['bust']:.0%}", border=True)
            st.caption(f"Tested on 10,000 fresh sessions it never trained on · about {m['spins']:.0f} spins a session "
                       f"· half the sessions end at {m['median'] * 1000:+,.0f} or better")
            st.button("Follow this strategy", key="rl_follow", on_click=rl_follow, args=(best,), type="primary",
                      icon=":material/route:", help="The Roulette AI will tell you the exact bet every spin")
        st.markdown("**The top 5** (after the fresh re-test)")
        st.dataframe(pd.DataFrame([{"Strategy": rl_plan_text(p).replace("**", ""), "Finishes ahead": r["ahead"],
                                    "Average session": r["average"] * 1000, "Doubles up": r["double"],
                                    "Goes broke": r["bust"]} for p, r in retested]),
                     hide_index=True, width="stretch", column_config={
                         "Finishes ahead": st.column_config.NumberColumn(format="percent"),
                         "Doubles up": st.column_config.NumberColumn(format="percent"),
                         "Goes broke": st.column_config.NumberColumn(format="percent"),
                         "Average session": st.column_config.NumberColumn(format="%+.0f",
                                                                          help="Chips per 1,000-chip session")})
        chart = alt.Chart(everything.assign(average=everything["average"] * 1000)).mark_circle(size=55, opacity=0.7).encode(
            x=alt.X("ahead:Q", title="chance a session finishes ahead", axis=alt.Axis(format="%")),
            y=alt.Y("average:Q", title="average chips per session"),
            color=alt.Color("system:N", title="System"),
            tooltip=["bet", "system", alt.Tooltip("ahead:Q", format=".0%"), alt.Tooltip("average:Q", format="+.0f"),
                     alt.Tooltip("bust:Q", format=".0%", title="goes broke")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#7A7A7A", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((chart + zero).properties(height=300), width="stretch")
        st.info(f"**What the AI learned from {len(everything):,} strategies:** every one sits below the dashed line. "
                f"Each session loses exactly 5.26% of the chips it bets, on average - so the only way to lose less "
                f"is to bet less. What a strategy CAN change is the shape: Martingale-style systems finish "
                  f"ahead often but sometimes lose big; bold play doubles most often; tiny flat bets lose the least "
                  f"because they bet the least. Pick the shape you like - it's still fake chips.",
                icon=":material/lightbulb:")


def rl_chip_css(key, amount, dropping=False):
    """draws a casino chip ON a betting spot (over its button), coloured like the biggest chip in the bet,
    with the amount printed on it - a few chips' edges show underneath when it's more than one chip"""
    color = [c for v, c in CHIPS if v <= amount][-1] if amount >= CHIPS[0][0] else CHIPS[0][1]
    label = (f"{amount / 1000:.1f}k".replace(".0k", "k")) if amount >= 1000 else str(amount)
    stack = "".join(f", 0 {3 * i}px 0 -1px {color}, 0 {3 * i}px 0 0 rgba(255,255,255,.55)"
                    for i in range(1, min(4, amount // max(v for v, _ in CHIPS if v <= max(amount, 10)))))
    return (f".st-key-rlb_{key} button {{ position:relative; overflow:visible !important; }}"
            f".st-key-rlb_{key} button::after {{ content:'{label}'; position:absolute; left:50%; top:50%; width:30px; "
            f"height:30px; margin:-17px 0 0 -15px; border-radius:50%; background:{color}; box-sizing:border-box; "
            f"border:3px dashed rgba(255,255,255,.9); color:#FFF; font-size:10px; font-weight:800; display:flex; "
            f"align-items:center; justify-content:center; z-index:3; box-shadow:0 4px 6px rgba(0,0,0,.6){stack}; "
            + ("animation:pq-drop .28s ease-out both; " if dropping else "") + "}")


def rl_bet_button(key, col):
    """one spot on the betting layout - your chips sit on top of it (see rl_chip_css)"""
    label = RL_BETS[key][2]
    other = RL_OPPOSITE.get(key)
    locked = bool(other and rl_game()["bets"].get(other))   # the other side already has your chips
    col.button(label, key=f"rlb_{key}", on_click=rl_place, args=(key,), width="stretch", disabled=locked,
               help=(f"Locked - you already bet {RL_BETS[other][2].title()}. One side only (clear the table to switch)."
                     if locked else f"{RL_BETS[key][2]} pays {RL_BETS[key][1]} to 1"))


@st.fragment
def roulette():
    """the roulette table - a fragment, so betting / spinning only redraws the game"""
    g, wallet = rl_game(), bj_game()
    spin_now = g.pop("spun", False) and g["result"] is not None
    last_chip = g.pop("last_chip", None)   # the chip you just put down drops onto its spot
    st.html(CARD_CSS + CHIP_CSS + "<style>" + "".join(
        f".st-key-rlb_n{n} button {{ background:{rl_color(n)} !important; }}" for n in range(RL_POCKETS)) + """
      [class*="st-key-rlb_"] button { min-height:40px; padding:0 2px !important; border-radius:6px !important;
          font-size:12px !important; font-weight:800; color:#FFF !important; border:1px solid rgba(255,255,255,.22) !important; }
      [class*="st-key-rlb_"] button p { font-size:12px !important; font-weight:800; }
      .st-key-rlb_red button { background:#B91C1C !important; } .st-key-rlb_black button { background:#141414 !important; }
      [class*="st-key-rlb_"]:not([class*="rlb_n"]):not(.st-key-rlb_red):not(.st-key-rlb_black) button {
          background:rgba(0,0,0,.28) !important; color:#A7F3D0 !important; }
      .st-key-rl_felt { background:radial-gradient(ellipse at 50% 0%,#14532D 0%,#0B2E1A 45%,#04120A 100%);
                        border:2px solid #14532D; border-radius:22px; padding:22px 24px 18px;
                        box-shadow:inset 0 0 60px rgba(0,0,0,.55); }
      .rl-wheel-box { position:relative; width:250px; height:250px; margin:0 auto; }
      .rl-wheel { width:250px; height:250px; display:block; filter:drop-shadow(0 8px 18px rgba(0,0,0,.6)); }
      .rl-orbit { position:absolute; inset:0; pointer-events:none; z-index:2; }
      .rl-ball  { position:absolute; left:50%; top:50%; width:15px; height:15px; border-radius:50%;
                  background:radial-gradient(circle at 35% 30%, #FFFFFF 0%, #E5E5E5 45%, #9A9A9A 100%);
                  box-shadow:0 2px 5px rgba(0,0,0,.75); transform:translate(-50%,-50%) translateY(-88px); }
      .rl-result { width:96px; height:96px; border-radius:50%; margin:6px auto; display:flex; align-items:center;
                   justify-content:center; font-size:40px; font-weight:800; color:#FFF; border:4px solid #C8A24A;
                   box-shadow:0 6px 16px rgba(0,0,0,.6); }
      .rl-dots { display:flex; flex-wrap:wrap; gap:4px; justify-content:center; margin-top:6px; }
      .rl-dot { width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
                font-size:10px; font-weight:800; color:#FFF; }
    </style>"""
        # chips ON the table: a casino chip sitting on every spot you've bet, with the amount on it
        + "<style>" + "".join(rl_chip_css(k, amount, k == last_chip) for k, amount in g["bets"].items()) + "</style>")
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="rl_felt"):
            wheel_col, info_col = st.columns([3, 2], vertical_alignment="center")
            spin_time = 3.4
            if spin_now:   # spin from where the wheel was to where it lands (a fresh animation each spin)
                name = f"rl-spin-{g['spins']}"
                wheel = (f'<style>@keyframes {name} {{ from {{ transform:rotate({g["from"]:.1f}deg); }} '
                         f'to {{ transform:rotate({g["angle"]:.1f}deg); }} }}</style>'
                         f'<img class="rl-wheel" src="{RL_WHEEL_IMG}" '
                         f'style="animation:{name} {spin_time}s cubic-bezier(.15,.62,.1,1) both">')
            else:
                wheel = f'<img class="rl-wheel" src="{RL_WHEEL_IMG}" style="transform:rotate({g["angle"]:.1f}deg)">'
            # the ball: it runs around the rim the OTHER way, slows down and drops into the winning pocket (which
            # the wheel brings to the top), then sits there until the next spin
            ball = ""
            if g["result"] is not None:
                if spin_now:
                    orbit, drop = f"rl-orbit-{g['spins']}", f"rl-drop-{g['spins']}"
                    ball = (f'<style>@keyframes {orbit} {{ from {{ transform:rotate(2160deg); }} to {{ transform:rotate(0deg); }} }}'
                            f'@keyframes {drop} {{ 0%, 72% {{ transform:translate(-50%,-50%) translateY(-114px); }} '
                            f'86% {{ transform:translate(-50%,-50%) translateY(-84px); }} '
                            f'100% {{ transform:translate(-50%,-50%) translateY(-88px); }} }}</style>'
                            f'<div class="rl-orbit" style="animation:{orbit} {spin_time}s cubic-bezier(.22,.6,.18,1) both">'
                            f'<span class="rl-ball" style="animation:{drop} {spin_time}s linear both"></span></div>')
                else:
                    ball = '<div class="rl-orbit"><span class="rl-ball"></span></div>'
            wheel_col.html(f'<div class="rl-wheel-box">{wheel}{ball}</div>')
            fade = f' class="bj-fade" style="animation-delay:{spin_time:.1f}s"' if spin_now else ""
            n = g["result"]
            info_col.html(
                (f'<div{fade}><div class="rl-result" style="background:{rl_color(n)}">{rl_name(n)}</div></div>' if n is not None
                 else '<div class="pq-empty" style="height:96px">No spin yet</div>')
                + f'<div class="bj-msg" style="font-size:16px"><span{fade}>'
                  f'{html.escape(g["msg"]) or "Tap a chip, then tap bets on the layout - then spin."}</span></div>'
                + ('<div class="rl-dots">' + "".join(
                    f'<span class="rl-dot" style="background:{rl_color(h)}">{rl_name(h)}</span>'
                    for h in reversed(g["history"][-12 if not spin_now else -13:][:-1] if spin_now else g["history"][-12:]))
                   + "</div>" if g["history"] else ""))

            # the betting layout: 0, then 3 rows of 12 numbers with a 2:1 column bet at the end of each row
            for r, col_key in ((3, "c3"), (2, "c2"), (1, "c1")):
                cols = st.columns([1.15] + [1] * 12 + [1.2], gap="small")
                if r == 3:
                    rl_bet_button("n0", cols[0])
                if r == 1:
                    rl_bet_button("n37", cols[0])   # 00
                for i in range(12):
                    rl_bet_button(f"n{3 * i + r}", cols[i + 1])
                rl_bet_button(col_key, cols[13])
            cols = st.columns([1.15, 4, 4, 4, 1.2], gap="small")
            for key, col in zip(("d1", "d2", "d3"), cols[1:4]):
                rl_bet_button(key, col)
            cols = st.columns([1.15, 2, 2, 2, 2, 2, 2, 1.2], gap="small")
            for key, col in zip(("low", "even", "red", "black", "odd", "high"), cols[1:7]):
                rl_bet_button(key, col)

            if g["bets"]:   # every bet on the table, with its chips
                names = {k: ("Number " + RL_BETS[k][2] if k.startswith("n") else
                             {"c1": "Column 1-34", "c2": "Column 2-35", "c3": "Column 3-36"}.get(k, RL_BETS[k][2]))
                         for k in g["bets"]}
                st.html('<div style="text-align:center;color:#A7F3D0;font-size:13px;margin:4px 0">On the table: '
                        + " &nbsp;·&nbsp; ".join(f"<b style='color:#FFF'>{names[k]}</b> {a}" for k, a in g["bets"].items())
                        + "</div>")
            if wallet["chips"] >= 10:
                chip_rack("rl", rl_pick_chip, lambda v: v <= wallet["chips"], selected=g["chip"])
                total = sum(g["bets"].values())
                clear_col, spin_col = st.columns(2)
                clear_col.button("Clear", on_click=rl_clear, key="roul_clear", width="stretch", icon=":material/close:",
                                 disabled=not total)
                spin_col.button(f"Spin ({total:,} chips)" if total else "Spin", on_click=rl_spin, key="roul_spin",
                                width="stretch", type="primary", icon=":material/autorenew:", disabled=not total)
        with st.expander("How roulette works", icon=":material/school:"):
            st.markdown(
                "**This is the American wheel - 38 pockets:** 1-36 (half red, half black) and TWO greens, **0 and "
                "00**. Bet on where the ball "
                "lands. **Inside bets:** a single number (or 0, or 00) pays **35 to 1**. **Outside bets:** Red / Black, Odd / Even, "
                "1-18 / 19-36 pay **1 to 1**; a dozen or a column pays **2 to 1**. On 0 or 00, every outside bet loses. "
                "**Table rule:** you can only back one side of Red / Black, Odd / Even and 1-18 / 19-36 - betting both "
                "sides can never win, it just loses on 0 and 00.\n\n"
                "**The house edge is 5.26% on every bet** - a single number pays as if there were 36 pockets, but "
                "there are 38 (that second green pocket is why American roulette costs about twice as much as the "
                "single-zero European wheel). No betting system changes that: doubling after losses (Martingale), \"hot\" numbers, "
                "streaks - the wheel has no memory. Fake chips, no real money - just for fun.")
        strategy_lab("rl")
        rl_strategy_finder()

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every game")
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True)
        # ---- the AI: it looks at your bets (never the ball - that's only decided when you spin) ----
        with st.container(border=True):
            st.markdown("**:material/psychology: Roulette AI**")
            # every roulette bet has the SAME 5.26% edge - so the AI asks what you're going for, then names the bet
            goal = st.segmented_control("Your goal", ["Win more spins", "Balanced", "Big payout", "Double my chips"],
                                        default="Win more spins", key="rl_goal", width="stretch") or "Win more spins"
            size = max(10, round(wallet["chips"] * 0.02 / 5) * 5)   # about 2% of your chips
            small = max(10, round(wallet["chips"] * 0.01 / 5) * 5)  # about 1%
            colour = random.Random(g["spins"]).choice(["Red", "Black"])   # a coin flip - they're exactly equal
            if goal == "Win more spins":
                headline = f"Bet {size:,} on {colour}"
                why = (f"Red and Black are exactly even: 18 of 38 pockets each (<b>47.4%</b>), paying 1 to 1 - so the "
                       f"AI flipped a coin and got {colour}. Past spins don't change it. Odd / Even and 1-18 / 19-36 "
                       f"are the same bet in disguise.")
            elif goal == "Balanced":
                headline = f"Bet {size:,} on 1st 12 + {size:,} on 2nd 12"
                why = (f"Two dozens cover 24 of 38 pockets: you win <b>63%</b> of spins (+{size:,} net) and lose "
                       f"{2 * size:,} the rest of the time. Any two dozens or two columns work the same.")
            elif goal == "Big payout":
                pick = random.Random(g["spins"] + 1).choice(["0", "00"])
                headline = f"Bet {small:,} on {pick} (green)"
                why = (f"A single number pays <b>35 to 1</b> (+{small * 35:,}) but hits only 1 in 38 (<b>2.6%</b>). "
                       f"Green 0 and 00 are no better or worse than any other number - they're just as likely. "
                       f"Keep these bets small: you'll usually miss.")
            else:
                headline = f"Bet {wallet['chips']:,} on {colour} (all-in)"
                why = (f"The math answer for doubling: <b>bold play</b> - one big bet doubles you 47.4% of the time, "
                       f"while lots of small bets give the 5.26% edge time to grind you down (see the Strategy lab: "
                       f"5-chip bets get from 100 to 200 only about 1 time in 10). It's the least-bad way to chase a "
                       f"target, not a way to win long run. Fake chips only!")
            # the exact bet it just named - scored against your next spin for its own win rate
            g["ai_bet"] = ({colour.lower(): size} if goal == "Win more spins" else {"d1": size, "d2": size}
                           if goal == "Balanced" else {"n0" if pick == "0" else "n37": small} if goal == "Big payout"
                           else {colour.lower(): wallet["chips"]})
            if g.get("plan"):   # following a strategy from the AI strategy finder: its exact bet, every spin
                bet, size, system, stop_win, stop_loss, spins = g["plan"]["plan"]
                nxt, stop = rl_plan_now(g, wallet["chips"])
                g["plan"]["next"] = nxt
                if nxt:
                    labels = {"red": "Red", "d1": "1st 12", "d2": "2nd 12", "n17": "17"}
                    headline = "Bet " + " + ".join(f"{amt:,} on {labels[k]}" for k, amt in nxt.items())
                    g["ai_bet"] = nxt
                else:
                    headline, g["ai_bet"] = "Stop - strategy finished", None
                why = (f"<b>Your strategy:</b> {html.escape(rl_plan_text(g['plan']['plan'])).replace('**', '')}<br>"
                       f"Spin {g['plan']['spins'] + 1} of {spins} · you started with {g['plan']['start']:,} chips"
                       + (f" · {system} step {g['plan']['level']:.0f}" if system not in ("Flat", "Bold") else "")
                       + (f"<br>{html.escape(stop).replace('**', '')}" if stop else ""))
            why += " <span style='color:#7A7A7A'>Every bet loses 5.26% long run.</span>"
            a = rl_analysis(g["bets"])
            if a["total"]:   # and a look at what's on the table right now
                why += (f"<br><b>Your bets now:</b> {a['total']:,} chips, profit on {a['profit_pockets']} of 38 "
                        f"pockets ({a['profit_pockets'] / RL_POCKETS:.0%}), best case +{a['best']:,}, expected "
                        f"{a['expected']:+.2f} a spin."
                        + (" That's more than 5% of your chips on one spin." if a["total"] > wallet["chips"] * 0.05
                           and goal != "Double my chips" else ""))
            st.html(f'<div style="background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.45);'
                    f'border-radius:10px;padding:10px 12px;margin:2px 0 6px">'
                    f'<div style="font-size:12px;color:#A7F3D0;letter-spacing:.08em;text-transform:uppercase">'
                    f'The AI says</div><div style="font-size:18px;font-weight:800;color:#FFF;margin:2px 0">{headline}</div>'
                    f'<div style="font-size:12px;color:#C7C7C7;line-height:1.55">{why}</div></div>')
            if g.get("plan"):
                st.button("Stop following the strategy", key="rl_unfollow", on_click=rl_unfollow,
                          icon=":material/close:", width="stretch")
            if sum(g.get("ai_record", [])):
                ai_rate_caption(g["ai_record"], "spins", "betting what it suggested for your goal each spin "
                                "(Red/Black ~47%, two dozens ~63%, one number ~3%)")
            # trainer: in roulette no choice changes the 5.26% edge - so it drills the payouts instead
            st.toggle("Trainer mode", key="trainer_rl",
                      help="A payout quiz: the AI picks a pocket and asks how many chips your bets would give back.")
            if trainer_on("rl"):
                if g["bets"] and g.get("quiz_asked") != rl_quiz_key(g):
                    pocket = rl_quiz_pocket(g)
                    st.markdown(f"**Payout quiz:** if the ball lands on **{rl_name(pocket)}**, how many chips do "
                                f"you get back?")
                    ask, check = st.columns([3, 1], vertical_alignment="bottom")
                    ask.number_input("Chips back", min_value=0, step=5, key="rl_quiz_guess",
                                     label_visibility="collapsed")
                    check.button("Check", on_click=rl_quiz_check, args=(pocket,), key="rl_quiz_check",
                                 type="primary", width="stretch")
                elif not g["bets"]:
                    st.caption("Put chips on the table - the AI will quiz you on what they'd pay.")
                trainer_panel("rl")
            # streaks / hot numbers: say plainly that they mean nothing
            hist = g["history"]
            if len(hist) >= 4:
                last = ["green" if h in (0, 37) else "red" if h in RL_RED else "black" for h in hist]
                streak = len(last) - next((i for i in range(len(last) - 1, -1, -1) if last[i] != last[-1]), -1) - 1
                hot, times = Counter(hist).most_common(1)[0]
                st.caption(f"{streak} {last[-1]} in a row" + (" - " if streak >= 3 else " · ")
                           + (f"that doesn't make {'black' if last[-1] == 'red' else 'red'} more likely. "
                              if streak >= 3 else "")
                           + f"Most common lately: {rl_name(hot)} ({times}x) - it's no more likely to come up again. "
                             "The wheel has no memory: every spin is a fresh 1 in 38.")
            if won + lost + pushed:
                st.caption(f"**Win rate:** {won / max(won + lost, 1):.0%} of your spins made a profit ({won}-{lost}).")
            st.caption("The AI only looks at your bets and past spins - the ball is only decided when you press Spin.")


# ---------------------------------------------------------------- three card poker - fake chips, just for fun
# Ante (+ optional Pair Plus), 3 cards each, then Play (match the Ante) or Fold. The dealer needs Queen-high to
# qualify. A fresh 52-card deck every hand (like a casino shuffling machine).
TCP_DECK = [(rank, suit) for rank in BJ_RANKS for suit in BJ_SUITS]
TCP_VALUE = {r: v for v, r in enumerate(BJ_RANKS[1:], start=2)} | {"A": 14}
TCP_NAMES = ["High card", "Pair", "Flush", "Straight", "Three of a kind", "Straight flush"]
TCP_ANTE_BONUS = {5: 5, 4: 4, 3: 1}            # straight flush 5, three of a kind 4, straight 1 (paid when you play)
TCP_PAIR_PLUS = {5: 40, 4: 30, 3: 6, 2: 3, 1: 1}   # the usual "1-3-6-30-40" Pair Plus table


def tcp_rank(hand):
    """(category, tie-break values) - category 0 high card ... 5 straight flush. In 3-card poker a STRAIGHT beats a
    flush (it's rarer with 3 cards), and A-2-3 is a straight (the lowest one)."""
    values = sorted((TCP_VALUE[r] for r, _ in hand), reverse=True)
    flush = len({s for _, s in hand}) == 1
    if values == [14, 3, 2]:
        straight, values = True, [3, 2, 1]
    else:
        straight = values[0] - values[2] == 2 and len(set(values)) == 3
    if straight and flush:
        return 5, values
    if values[0] == values[2]:
        return 4, values
    if straight:
        return 3, values
    if flush:
        return 2, values
    if len(set(values)) == 2:
        pair = values[1]   # the middle card is always part of the pair
        return 1, [pair, next(v for v in values if v != pair)]
    return 0, values


def tcp_qualifies(rank):
    return rank[0] > 0 or rank[1][0] >= 12   # Queen-high or better


def tcp_result(player, dealer):
    """per 1 chip of Ante, if you PLAY: (ante + play result, ante bonus). Dealer doesn't qualify -> Ante wins, Play pushes"""
    me, them = tcp_rank(player), tcp_rank(dealer)
    bonus = TCP_ANTE_BONUS.get(me[0], 0)
    if not tcp_qualifies(them):
        return 1, bonus
    return (2 if me > them else -2 if me < them else 0), bonus


@st.cache_data(max_entries=300, show_spinner=False)
def tcp_play_ev(hand):
    """the AI: average result per chip of Ante if you PLAY this hand - checked against every possible dealer hand
    (all 18,424 of them, from the 49 cards you can't see). Folding always loses exactly 1."""
    rest = [c for c in TCP_DECK if c not in hand]
    total = sum(sum(tcp_result(list(hand), list(d))) for d in itertools.combinations(rest, 3))
    return total / 18424


def tcp_q64(hand):
    """the famous shortcut: play Queen-6-4 or better, fold anything worse"""
    rank = tcp_rank(hand)
    return rank[0] > 0 or rank[1] >= [12, 6, 4]


def tcp_game():
    g = st.session_state.setdefault("tcp", {"player": [], "dealer": [], "phase": "bet", "ante": 25, "pp": 0,
                                            "chip": 25, "msg": "", "record": [0, 0, 0], "new": [], "reveal": False,
                                            "ai_record": [0, 0, 0], "followed": [0, 0]})
    return g


def tcp_pick_chip(value):
    tcp_game()["chip"] = value


def tcp_place(spot):
    """drop the picked chip on Ante or Pair Plus"""
    g, wallet = tcp_game(), bj_game()
    if g["phase"] == "decide":
        return
    if g["ante"] + g["pp"] + g["chip"] > wallet["chips"]:
        g["msg"] = "Not enough chips for that - pick a smaller chip or clear the table."
        return
    g[spot] += g["chip"]
    g["drop"], g["new"] = spot, []


def tcp_clear():
    g = tcp_game()
    g["ante"], g["pp"], g["new"] = 0, 0, []


def tcp_deal():
    g, wallet = tcp_game(), bj_game()
    if g["ante"] < 10:
        g["msg"] = "Put chips on the Ante first."
        return
    if g["ante"] * 2 + g["pp"] > wallet["chips"]:
        g["msg"] = "Keep enough chips to match your Ante if you Play - bet smaller."
        return
    wallet["chips"] -= g["ante"] + g["pp"]
    deck = random.sample(TCP_DECK, 6)
    g.update(player=deck[:3], dealer=deck[3:], phase="decide", reveal=False, msg="", new=list(range(6)), win=None)


def tcp_settle(play):
    """you played or folded: turn the dealer's cards over and pay everything"""
    g, wallet = tcp_game(), bj_game()
    ante, pp = g["ante"], g["pp"]
    # the AI's own call on the SAME cards (it only looked at your 3 cards)
    play_ev = tcp_play_ev(tuple(g["player"]))
    ai_play = play_ev > -1
    trainer_grade("tcp", play, ai_play,
                  f"The best play was **{'Play' if ai_play else 'Fold'}**: playing this hand averages {play_ev:+.2f} "
                  f"per chip of Ante (over all 18,424 dealer hands), folding loses -1.00. Shortcut: play Queen-6-4 "
                  f"or better - this hand {'is' if tcp_q64(g['player']) else 'is not'}.")
    g["followed"][0] += play == ai_play
    g["followed"][1] += 1
    me, them = tcp_rank(g["player"]), tcp_rank(g["dealer"])
    back = 0
    if pp:   # Pair Plus is paid on your hand alone, play or fold
        back += pp * (TCP_PAIR_PLUS[me[0]] + 1) if me[0] in TCP_PAIR_PLUS else 0
    if play:
        wallet["chips"] -= ante   # the Play bet matches the Ante
        main, bonus = tcp_result(g["player"], g["dealer"])
        back += ante * 2 + ante * main + ante * bonus   # both bets back, +/- the result, + any Ante bonus
        staked = ante * 2
    else:
        main, bonus, staked = -1, 0, ante
    wallet["chips"] += back
    net = back - staked - pp
    g["record"][0 if net > 0 else 2 if net == 0 else 1] += 1
    ai_main = sum(tcp_result(g["player"], g["dealer"])) if ai_play else -1
    g["ai_record"][0 if ai_main > 0 else 2 if ai_main == 0 else 1] += 1
    qual = "the dealer qualifies" if tcp_qualifies(them) else "the dealer doesn't qualify (Queen-high needed)"
    g["msg"] = (f"You: {TCP_NAMES[me[0]]} · Dealer: {TCP_NAMES[them[0]]} - {qual}. " if play else
                f"You folded {TCP_NAMES[me[0]].lower()}. Dealer had {TCP_NAMES[them[0]].lower()}. ") + \
        (f"You win {net:,}!" if net > 0 else f"You lose {-net:,}." if net < 0 else "You break even.")
    g.update(phase="done", reveal=True, played=play, new=[])


def tcp_simulate(hands):
    """the Strategy lab: four ways of deciding Play / Fold on the SAME hands, plus the Pair Plus side bet alone"""
    deals = [random.sample(TCP_DECK, 6) for _ in range(hands)]
    styles = {"Q-6-4 rule (what the AI does)": tcp_q64, "Always play": lambda h: True,
              "Play a pair or better only": lambda h: tcp_rank(h)[0] >= 1,
              "Play like the dealer (Queen-high or better)": lambda h: tcp_qualifies(tcp_rank(h))}
    rows = []
    for name, decide in styles.items():
        total, staked, won, lost = 0, 0, 0, 0
        for d in deals:
            player, dealer = d[:3], d[3:]
            if decide(player):
                result = sum(tcp_result(player, dealer))
                staked += 2
            else:
                result = -1
                staked += 1
            total, won, lost = total + result, won + (result > 0), lost + (result < 0)
        rows.append({"Strategy": name, "Win rate": won / max(won + lost, 1), "Profit per 100 chips bet": total / staked * 100})
    total, won = 0, 0
    for d in deals:
        cat = tcp_rank(d[:3])[0]
        total, won = total + TCP_PAIR_PLUS.get(cat, -1), won + (cat in TCP_PAIR_PLUS)
    rows.append({"Strategy": "Pair Plus side bet (on its own)", "Win rate": won / hands,
                 "Profit per 100 chips bet": total / hands * 100})
    return pd.DataFrame(rows)


def tcp_cards(g, who):
    """a hand's 3 cards: dealt in (sliding in), the dealer's face down until you play or fold, then flipped over"""
    out = ""
    for i, (rank, suit) in enumerate(g[who]):
        slot = i if who == "player" else i + 3
        delay = (i * 2 + (who == "dealer")) * DEAL_GAP
        if who == "dealer" and not g["reveal"]:
            out += f'<span class="bj-card back{" new" if slot in g["new"] else ""}" style="animation-delay:{delay:.2f}s"></span>'
            continue
        anim = " new" if slot in g["new"] else (" flip" if who == "dealer" and g.get("flipping") else "")
        style = f' style="animation-delay:{delay if anim == " new" else i * 0.15:.2f}s"' if anim else ""
        out += f'<span class="bj-card {"red" if BJ_SUITS[suit] else ""}{anim}"{style}><b>{rank}</b><span class="bj-suit">{suit}</span></span>'
    return out


@st.fragment
def three_card_poker():
    """the three card poker table - a fragment, so betting / playing only redraws the game"""
    g, wallet = tcp_game(), bj_game()
    deciding, done = g["phase"] == "decide", g["phase"] == "done"
    g["flipping"] = done and g.pop("just_settled", False)
    st.html(CARD_CSS + CHIP_CSS + """<style>
      .st-key-tcp_felt { background:radial-gradient(ellipse at 50% 0%,#14532D 0%,#0B2E1A 45%,#04120A 100%);
                         border:2px solid #14532D; border-radius:22px; padding:24px 30px 18px;
                         box-shadow:inset 0 0 60px rgba(0,0,0,.55); }
      .tcp-spot-label { text-align:center; color:#A7F3D0; font-size:12px; font-weight:800; letter-spacing:.12em; }
    </style>""")
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="tcp_felt"):
            me = tcp_rank(g["player"]) if g["player"] else None
            them = tcp_rank(g["dealer"]) if g["dealer"] and g["reveal"] else None
            st.html(f'<div><div class="bj-who">Dealer{f" <b>{TCP_NAMES[them[0]]}</b>" if them else ""}</div>'
                    f'<div class="bj-hand">{tcp_cards(g, "dealer")}</div>'
                    f'<div class="bj-mark">DEALER QUALIFIES WITH QUEEN HIGH &nbsp;·&nbsp; PLAY Q-6-4 OR BETTER</div>'
                    f'<div class="bj-who">You{f" <b>{TCP_NAMES[me[0]]}</b>" if me else ""}</div>'
                    f'<div class="bj-hand">{tcp_cards(g, "player")}</div>'
                    f'<div class="bj-msg">{html.escape(g["msg"]) or ("Pick your bet and decide: Play or Fold." if deciding else "Put chips on the Ante (and Pair Plus if you like), then deal.")}</div></div>')
            g["new"] = []
            # the three betting circles: Pair Plus, Ante, Play - your chips sit on them
            drop = g.pop("drop", None)
            play_amount = g["ante"] if done and g.get("played") else 0
            for col, (label, amount, spot) in zip(st.columns(3), (("PAIR PLUS", g["pp"], "pp"), ("ANTE", g["ante"], "ante"),
                                                                 ("PLAY", play_amount, None))):
                col.html(f'<div class="tcp-spot-label">{label}</div>'
                         f'<div class="pq-circle" style="width:110px;height:110px">{chip_stack_html(amount, drop == spot, empty="")}</div>')
                if spot and not deciding:
                    col.button(f"Bet {label.title()}", key=f"tcp_spot_{spot}", on_click=tcp_place, args=(spot,),
                               width="stretch", help=f"Drop a {g['chip']} chip on {label.title()}")
            if deciding:
                play_col, fold_col = st.columns(2)
                play_col.button(f"Play ({g['ante']:,} more)", on_click=lambda: (g.update(just_settled=True), tcp_settle(True)),
                                key="tcp_play", type="primary", width="stretch", icon=":material/play_arrow:",
                                disabled=wallet["chips"] < g["ante"])
                fold_col.button("Fold", on_click=lambda: (g.update(just_settled=True), tcp_settle(False)),
                                key="tcp_fold", width="stretch", icon=":material/close:")
            elif wallet["chips"] >= 10:
                chip_rack("tcp", tcp_pick_chip, lambda v: v <= wallet["chips"], selected=g["chip"])
                clear_col, deal_col = st.columns(2)
                clear_col.button("Clear", on_click=tcp_clear, key="tcp_clear", width="stretch", icon=":material/close:",
                                 disabled=not (g["ante"] or g["pp"]))
                deal_col.button(f"Deal ({g['ante'] + g['pp']:,} chips)" if g["ante"] else "Deal", on_click=tcp_deal,
                                key="tcp_deal", width="stretch", type="primary", icon=":material/playing_cards:",
                                disabled=g["ante"] < 10)
        with st.expander("How three card poker works", icon=":material/school:"):
            st.markdown(
                "**Bet the Ante** (and the optional **Pair Plus**). You and the dealer get 3 cards; the dealer's stay "
                "face down. Look at yours and choose: **Play** (put in a bet equal to your Ante) or **Fold** (lose the "
                "Ante).\n\n**If you play:** the dealer needs **Queen-high or better** to qualify. If it doesn't, your "
                "Ante wins 1 to 1 and Play comes back. If it does, the better hand wins both bets 1 to 1 (a tie "
                "pushes).\n\n**Hands, best first:** straight flush, three of a kind, **straight**, **flush**, pair, "
                "high card - with 3 cards a straight is rarer than a flush, so it ranks higher. A-2-3 is a straight.\n\n"
                "**Ante bonus** (paid when you play, win or lose): straight 1, three of a kind 4, straight flush 5. "
                "**Pair Plus** pays on your hand alone: pair 1, flush 3, straight 6, three of a kind 30, straight "
                "flush 40.\n\n**Best strategy:** play **Queen-6-4 or better**, fold the rest - it costs about 2 chips "
                "per 100 bet. Pair Plus loses about 7.3 per 100. Fake chips - just for fun.")
        strategy_lab("tcp")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every game")
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True)
        with st.container(border=True):
            st.markdown("**:material/psychology: Three Card Poker AI**")
            trainer_switch("tcp", "chosen Play or Fold")
            if deciding and trainer_on("tcp"):
                headline = None
            elif deciding:
                ev = tcp_play_ev(tuple(g["player"]))
                headline = "Play" if ev > -1 else "Fold"
                why = (f"It checked all 18,424 hands the dealer could have. Playing gives back <b>{ev:+.2f}</b> per chip "
                       f"of Ante; folding loses <b>-1.00</b>. " +
                       ("Play wins - or loses less." if ev > -1 else "Folding loses less.") +
                       f" (The Q-6-4 rule says {'Play' if tcp_q64(g['player']) else 'Fold'} too.)")
            else:
                size = max(10, round(wallet["chips"] * 0.02 / 5) * 5)
                headline = f"Ante {size:,} · skip Pair Plus"
                why = ("Keep enough chips to match your Ante. Pair Plus is fun but loses about 7.3 per 100 - more than "
                       "twice the main game (about 2 per 100 when you play Q-6-4 or better).")
            if headline is None:
                trainer_quiet()
            else:
                ai_box(headline, why)
            if trainer_on("tcp"):
                trainer_panel("tcp")
            ai_rate_caption(g["ai_record"], "hands", "if it had decided Play / Fold on every hand you were dealt")
            if g["followed"][1]:
                st.caption(f"You did what it said on {g['followed'][0]} of {g['followed'][1]} hands.")
            st.caption("The AI only sees your 3 cards - the dealer's are face down, so it checks every hand they "
                       "could be.")


# ---------------------------------------------------------------- let it ride - fake chips, just for fun
# Three equal bets (1, 2 and $). You get 3 cards, plus 2 face-down community cards. After seeing your 3 you can pull
# bet 1 back; after the first community card, bet 2. The $ bet always stays. Paid on the 5-card hand: tens or better.
LIR_PAYS = [("Royal flush", 1000), ("Straight flush", 200), ("Four of a kind", 50), ("Full house", 11),
            ("Flush", 8), ("Straight", 5), ("Three of a kind", 3), ("Two pair", 2), ("Pair of tens or better", 1)]


def lir_hand(hand):
    """(name, pays 'to 1') for a 5-card hand - (None, -1) when it loses"""
    values = sorted((VP_VALUE[r] for r, _ in hand), reverse=True)
    counts = Counter(values)
    shape = sorted(counts.values(), reverse=True)
    flush = len({s for _, s in hand}) == 1
    straight = len(counts) == 5 and (values[0] - values[4] == 4 or values == [14, 5, 4, 3, 2])
    if straight and flush:
        return ("Royal flush", 1000) if values[4] == 10 else ("Straight flush", 200)
    if shape[0] == 4:
        return "Four of a kind", 50
    if shape[:2] == [3, 2]:
        return "Full house", 11
    if flush:
        return "Flush", 8
    if straight:
        return "Straight", 5
    if shape[0] == 3:
        return "Three of a kind", 3
    if shape[:2] == [2, 2]:
        return "Two pair", 2
    if shape[0] == 2 and max(v for v, n in counts.items() if n == 2) >= 10:
        return "Pair of tens or better", 1
    return None, -1


def lir_ride_ev_exact(cards):
    """average result per chip of letting a bet ride, knowing these cards (3 or 4 of the final 5) - over EVERY way
    the unseen cards could come out (1,176 ways after 3 cards, 48 after 4). Pulling the bet back = 0."""
    rest = [c for c in VP_DECK if c not in cards]
    results = [lir_hand(cards + extra)[1] for extra in itertools.combinations(rest, 5 - len(cards))]
    return sum(results) / len(results)


@st.cache_data(max_entries=5000, show_spinner=False)
def lir_ride_ev(cards):
    """the AI (cached): it only knows the cards that are face up - never the face-down community cards"""
    return lir_ride_ev_exact(tuple(cards))


def lir_game():
    return st.session_state.setdefault("lir", {"bet": 10, "player": [], "community": [], "phase": "bet", "msg": "",
                                               "riding": [True, True], "record": [0, 0, 0], "ai_record": [0, 0, 0],
                                               "new": [], "followed": [0, 0]})


def lir_add_chip(value):
    g = lir_game()
    g["bet"] = (g["bet"] or 0) + value
    g["drop"] = True


def lir_clear():
    lir_game()["bet"] = 0


def lir_deal():
    g, wallet = lir_game(), bj_game()
    if g["bet"] < 10:
        g["msg"] = "Put chips in the circle first - the same bet goes on 1, 2 and $."
        return
    if g["bet"] * 3 > wallet["chips"]:
        g["msg"] = "Not enough chips for three bets that size - bet smaller."
        return
    wallet["chips"] -= g["bet"] * 3
    cards = random.sample(VP_DECK, 5)
    g.update(player=cards[:3], community=cards[3:], phase="d1", riding=[True, True], msg="", new=[0, 1, 2],
             stake=g["bet"], flip=None)


def lir_decide(ride):
    """you let bet 1 (then bet 2) ride, or pulled it back"""
    g, wallet = lir_game(), bj_game()
    step = 0 if g["phase"] == "d1" else 1
    seen = tuple(g["player"] + g["community"][:step])
    ev = lir_ride_ev(seen)
    tie = abs(ev) < 1e-9   # e.g. 4 to an open straight like 5-6-7-8: riding is worth exactly 0
    ai_ride = ev > 1e-9
    g["followed"][0] += ride == ai_ride or tie
    g["followed"][1] += 1
    trainer_grade("lir", ai_ride if tie else ride, ai_ride,
                  (f"Bet {step + 1}: a dead heat - riding these cards is worth exactly 0, the same as pulling back, "
                   f"so either is right. " if tie else
                   f"Bet {step + 1}: the best play was **{'Let it ride' if ai_ride else 'Pull it back'}**. Riding "
                   f"with these {len(seen)} cards averages {ev:+.2f} per chip (over every way the unseen cards could "
                   f"come), pulling back = 0. ") + LIR_TIPS[step])
    g.setdefault("ai_rides", [True, True])[step] = ai_ride
    g["riding"][step] = ride
    if not ride:
        wallet["chips"] += g["stake"]   # the bet comes back to you
    g["new"], g["flip"] = [], step
    if step == 0:
        g["phase"] = "d2"
        return
    # the last community card turns over: pay what's still riding
    name, pays = lir_hand(g["player"] + g["community"])
    left = sum(g["riding"]) + 1   # the $ bet always stays
    back = left * g["stake"] * (pays + 1) if pays > 0 else 0
    wallet["chips"] += back
    net = back - left * g["stake"]
    g["record"][0 if net > 0 else 2 if net == 0 else 1] += 1
    ai_net = (sum(g["ai_rides"]) + 1) * pays   # the AI's own result on the same cards, per chip
    g["ai_record"][0 if ai_net > 0 else 2 if ai_net == 0 else 1] += 1
    g["phase"] = "done"
    g["msg"] = (f"{name}! Pays {pays} to 1 on {left} bet{'s' if left > 1 else ''} - you win {net:,}." if pays > 0
                else f"No pair of tens or better - you lose {-net:,}.")


LIR_TIPS = ["The short rule after 3 cards: ride with a paying pair (tens or better) or three of a kind, three cards "
            "to a royal flush, or three suited cards in a row (3-4-5 or higher) - a few other suited hands with "
            "high cards too.",
            "The short rule after 4 cards: ride with a paying hand, four to a flush, or four to an open straight "
            "with a high card (like 8-9-10-J) - four to an inside straight only when all four are 10 or higher."]


def lir_simulate(hands):
    """the Strategy lab: four ways to decide on the SAME deals. Profit is per chip you actually had riding."""
    memo = {}

    def ev(cards):   # the exact AI, remembered by the hand's ranks + suit pattern (suits are all alike)
        cards = sorted(cards, key=lambda c: (VP_VALUE[c[0]], c[1]))
        names = {}
        key = tuple((r, names.setdefault(s, len(names))) for r, s in cards)
        if key not in memo:
            memo[key] = lir_ride_ev_exact(tuple(cards))
        return memo[key]

    def paying(cards):   # already holding a paying hand: tens or better, two pair, or three of a kind
        counts = Counter(VP_VALUE[r] for r, _ in cards)
        return (any(n >= 3 or (n == 2 and v >= 10) for v, n in counts.items())
                or sum(n == 2 for n in counts.values()) == 2)

    styles = {"Exact math (what the AI does)": lambda c: ev(c) > 0, "Always let it ride": lambda c: True,
              "Always pull both back": lambda c: False, "Ride only with a paying hand already": paying}
    deals = [random.sample(VP_DECK, 5) for _ in range(hands)]
    rows = []
    for name, ride in styles.items():
        profit = staked = won = lost = 0
        for d in deals:
            left = 1 + ride(tuple(d[:3])) + ride(tuple(d[:4]))
            pays = lir_hand(d)[1]
            profit, staked = profit + left * pays, staked + left
            won, lost = won + (pays > 0), lost + (pays < 0)
        rows.append({"Strategy": name, "Win rate": won / hands, "Profit per 100 chips bet": profit / staked * 100,
                     "Per hand (per 100 of one bet)": profit / hands * 100})
    return pd.DataFrame(rows)


def lir_cards(g):
    """your 3 cards, then the 2 community cards (face down until they turn over, one per decision)"""
    out = ""
    for i, (rank, suit) in enumerate(g["player"]):
        anim = f' new" style="animation-delay:{i * DEAL_GAP:.2f}s' if i in g["new"] else ""
        out += f'<span class="bj-card {"red" if BJ_SUITS[suit] else ""}{anim}"><b>{rank}</b><span class="bj-suit">{suit}</span></span>'
    out += '<span style="width:26px;display:inline-block"></span>'
    shown = {"d1": 0, "d2": 1}.get(g["phase"], 2)
    for i, (rank, suit) in enumerate(g["community"]):
        if i >= shown:
            anim = f' new" style="animation-delay:{(3 + i) * DEAL_GAP:.2f}s' if g["new"] else ""
            out += f'<span class="bj-card back{anim}"></span>'
            continue
        flip = " flip" if g.get("flip") is not None and i == g["flip"] else ""
        out += f'<span class="bj-card {"red" if BJ_SUITS[suit] else ""}{flip}"><b>{rank}</b><span class="bj-suit">{suit}</span></span>'
    return out


@st.fragment
def let_it_ride():
    """the let it ride table - a fragment, so each decision only redraws the game"""
    g, wallet = lir_game(), bj_game()
    deciding = g["phase"] in ("d1", "d2")
    if not deciding and g["bet"] * 3 > wallet["chips"]:
        g["bet"] = 0
    st.html(CARD_CSS + CHIP_CSS + """<style>
      .st-key-lir_felt { background:radial-gradient(ellipse at 50% 0%,#3B1D5E 0%,#1E0F33 45%,#0B0614 100%);
                         border:2px solid #4C1D95; border-radius:22px; padding:24px 30px 18px;
                         box-shadow:inset 0 0 60px rgba(0,0,0,.55); }
      .lir-label { text-align:center; color:#DDD6FE; font-size:14px; font-weight:800; letter-spacing:.12em; }
      .lir-pay { width:100%; border-collapse:collapse; font-size:13px; }
      .lir-pay td { padding:3px 8px; border-bottom:1px solid #161616; color:#C7C7C7; }
      .lir-pay td:last-child { text-align:right; font-weight:700; }
      .lir-pay tr.hit td { background:#22C55E; color:#000; font-weight:800; }
    </style>""")
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="lir_felt"):
            done = g["phase"] == "done"
            name = lir_hand(g["player"] + g["community"])[0] if done else None
            prompt = {"d1": "Look at your 3 cards: pull back bet 1, or let it ride?",
                      "d2": "One community card is up: pull back bet 2, or let it ride?"}.get(g["phase"])
            st.html(f'<div><div class="bj-who">Your cards &nbsp;·&nbsp; community cards</div>'
                    f'<div class="bj-hand">{lir_cards(g) if g["player"] else ""}</div>'
                    f'<div class="bj-mark">PAIR OF TENS OR BETTER WINS &nbsp;·&nbsp; THE $ BET ALWAYS RIDES</div>'
                    f'<div class="bj-msg">{html.escape(g["msg"]) or prompt or "Put chips in the circle - the same bet goes on 1, 2 and $ - then deal."}</div></div>')
            g["new"], g["flip"] = [], None   # shown - don't replay the animations
            drop = g.pop("drop", False)
            stake = g.get("stake", 0) if g["phase"] != "bet" else g["bet"]
            for col, (label, i) in zip(st.columns(3), (("1", 0), ("2", 1), ("$", None))):
                amount = stake if i is None or g["phase"] == "bet" or g["riding"][i] else 0
                col.html(f'<div class="lir-label">{label}</div>'
                         f'<div class="pq-circle" style="width:110px;height:110px">{chip_stack_html(amount, drop, empty="")}</div>')
            if deciding:
                pull, ride = st.columns(2)
                n = 1 if g["phase"] == "d1" else 2
                pull.button(f"Pull back bet {n}", on_click=lir_decide, args=(False,), key="lir_pull", width="stretch",
                            icon=":material/undo:")
                ride.button("Let it ride", on_click=lir_decide, args=(True,), key="lir_ride", width="stretch",
                            type="primary", icon=":material/trending_flat:")
            elif wallet["chips"] >= 30:
                chip_rack("lir", lir_add_chip, lambda v: (g["bet"] + v) * 3 <= wallet["chips"])
                clear_col, deal_col = st.columns(2)
                clear_col.button("Clear", on_click=lir_clear, key="lir_clear", width="stretch", icon=":material/close:",
                                 disabled=not g["bet"])
                deal_col.button(f"Deal ({g['bet'] * 3:,} chips)" if g["bet"] else "Deal", on_click=lir_deal,
                                key="lir_deal", width="stretch", type="primary", icon=":material/playing_cards:",
                                disabled=g["bet"] < 10)
            st.html('<table class="lir-pay">' + "".join(
                f'<tr class="{"hit" if hand == name else ""}"><td>{hand}</td><td>{pays:,} to 1</td></tr>'
                for hand, pays in LIR_PAYS) + "</table>")
        with st.expander("How let it ride works", icon=":material/school:"):
            st.markdown(
                "**Put three equal bets out: 1, 2 and $.** You get 3 cards; 2 community cards lie face down.\n\n"
                "1. Look at your 3 cards: **pull back bet 1** (you get it back) or **let it ride**.\n"
                "2. The first community card turns over: pull back **bet 2** or let it ride.\n"
                "3. The last card turns over. The **$ bet always stays.**\n\n"
                "Every bet still riding is paid on your 5-card hand: a **pair of tens or better** pays 1 to 1, up to "
                "a royal flush at 1,000 to 1. There's no dealer to beat - just the pay table.\n\n"
                "**Best strategy:** only let a bet ride when it's worth more than pulling it back - that's what the "
                "AI works out exactly. Played perfectly the game loses about 3.5 chips per 100 of one bet per hand. "
                "Fake chips - just for fun.")
        strategy_lab("lir")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every game")
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True)
        if not deciding and wallet["chips"] < 30:
            st.button("Out of chips - get 1,000 more", on_click=bj_reset, key="lir_reset", icon=":material/refresh:",
                      width="stretch", type="primary")
        with st.container(border=True):
            st.markdown("**:material/psychology: Let It Ride AI**")
            trainer_switch("lir", "pulled back or let it ride")
            if deciding and trainer_on("lir"):
                trainer_quiet()
            elif deciding:
                seen = g["player"] + g["community"][:0 if g["phase"] == "d1" else 1]
                ev = lir_ride_ev(tuple(seen))
                ways = 1176 if len(seen) == 3 else 48
                ai_box("Let it ride" if ev > 1e-9 else "Either - a dead heat" if abs(ev) < 1e-9 else "Pull it back",
                       f"It checked all {ways:,} ways the face-down cards could come. Riding averages <b>{ev:+.2f}</b> "
                       f"per chip; pulling back is 0. " + ("Riding wins more than it costs." if ev > 0 else
                                                           "Riding loses more than it wins - take the bet back."))
            else:
                size = max(10, round(wallet["chips"] * 0.01 / 5) * 5)
                ai_box(f"Bet {size:,} a circle",
                       "About 1% of your chips on each of the three circles. Every hand deals from a fresh deck, so "
                       "there's nothing to count - the edge is in when you pull back, and that's what the AI checks.")
            if trainer_on("lir"):
                trainer_panel("lir")
            ai_rate_caption(g["ai_record"], "hands", "deciding every pull-back itself")
            if g["followed"][1]:
                st.caption(f"You did what it said on {g['followed'][0]} of {g['followed'][1]} decisions.")
            st.caption("The AI only sees the cards that are face up - never the face-down community cards.")


# ---------------------------------------------------------------- craps (the dice game) - fake chips, just for fun
# Nothing you do changes the dice - but WHICH bets you make changes the house edge from 0% to almost 17%. That's
# what the AI advises on. The come-out roll: Pass wins on 7 / 11, loses on 2 / 3 / 12; anything else is the point.
# Then Pass wins if the point comes again before a 7.
CR_BETS = {   # bet: (name on the table, pays, house edge %)
    "pass": ("Pass Line", "1 to 1", 1.41), "dont": ("Don't Pass", "1 to 1", 1.36),
    "odds": ("Odds", "true odds", 0.0), "place6": ("Place 6", "7 to 6", 1.52), "place8": ("Place 8", "7 to 6", 1.52),
    "field": ("Field", "1 to 1 · 2 pays 2 · 12 pays 3", 2.78), "any7": ("Any 7", "4 to 1", 16.67)}
CR_ODDS_MAX = {4: 3, 10: 3, 5: 4, 9: 4, 6: 5, 8: 5}           # "3-4-5x odds": the most you can put behind Pass
CR_ODDS_PAY = {4: 2, 10: 2, 5: 1.5, 9: 1.5, 6: 1.2, 8: 1.2}   # true odds: exactly the chance of the point vs a 7
CR_POINT_WIN = {4: 1 / 3, 10: 1 / 3, 5: 2 / 5, 9: 2 / 5, 6: 5 / 11, 8: 5 / 11}   # point before 7
CR_PIPS = {1: [4], 2: [0, 8], 3: [0, 4, 8], 4: [0, 2, 6, 8], 5: [0, 2, 4, 6, 8], 6: [0, 2, 3, 5, 6, 8]}


def cr_game():
    return st.session_state.setdefault("cr", {"bets": dict.fromkeys(CR_BETS, 0), "chip": 25, "point": None,
                                              "dice": None, "rolls": 0, "history": [], "msg": "", "record": [0, 0, 0],
                                              "ai_record": [0, 0, 0], "rolled": False})


def cr_odds_max(g):
    return g["bets"]["pass"] * CR_ODDS_MAX[g["point"]] if g["point"] else 0


def cr_pick_chip(value):
    cr_game()["chip"] = value


def cr_place(bet):
    """drop the picked chip on a bet - the chips leave your stack right away and stay until the bet is decided"""
    g, wallet = cr_game(), bj_game()
    chip = g["chip"]
    if bet in ("pass", "dont") and g["point"]:
        g["msg"] = "Pass / Don't Pass go on before the come-out roll - wait for this point to finish."
        return
    if bet == "odds":
        room = cr_odds_max(g) - g["bets"]["odds"]
        if room <= 0:
            g["msg"] = ("Odds go behind a Pass Line bet once there's a point." if not (g["point"] and g["bets"]["pass"])
                        else f"That's the most odds allowed ({CR_ODDS_MAX[g['point']]}x your Pass bet on a {g['point']}).")
            return
        chip = min(chip, room)
    if chip > wallet["chips"]:
        g["msg"] = "Not enough chips for that - pick a smaller chip."
        return
    wallet["chips"] -= chip
    g["bets"][bet] += chip
    g["drop"], g["msg"] = bet, ""


def cr_clear():
    """take back every bet you're allowed to - Pass / Don't Pass are locked in once there's a point"""
    g, wallet = cr_game(), bj_game()
    for bet, amount in g["bets"].items():
        if amount and not (bet in ("pass", "dont") and g["point"]):
            wallet["chips"] += amount
            g["bets"][bet] = 0


def cr_roll():
    g, wallet = cr_game(), bj_game()
    b = g["bets"]
    # trainer: grade the table you're rolling with (once per new set of bets)
    table = tuple(b.values())
    if any(table) and table != g.get("graded"):
        g["graded"] = table
        bad = [CR_BETS[k][0] + f" ({CR_BETS[k][2]:.2f}%)" for k in ("field", "any7") if b[k]]
        missed = g["point"] and b["pass"] and b["odds"] < cr_odds_max(g)
        lesson = ("Every bet on your table is one of the cheapest in the casino." if not (bad or missed) else
                  (f"{' and '.join(bad)} cost far more than the Pass Line (1.41%) or Place 6 / 8 (1.52%). "
                   if bad else "")
                  + (f"You left free odds on the table: up to {cr_odds_max(g):,} behind your Pass, at a 0% house "
                     f"edge - the only bet in the casino the house has no edge on." if missed else ""))
        trainer_grade("cr", not (bad or missed), True, lesson)
    d1, d2 = random.randint(1, 6), random.randint(1, 6)   # the dice - nobody knows them before this
    s, point = d1 + d2, g["point"]
    back, staked, notes = 0, 0, []

    def settle(bet, pays):   # pays = winnings per chip, or -1 for a loss; the bet comes down either way
        nonlocal back, staked
        amount = b[bet]
        staked += amount
        if pays > 0:
            back += amount + int(amount * pays)
            notes.append(f"{CR_BETS[bet][0]} wins {int(amount * pays):,}")
        elif pays < 0:
            notes.append(f"{CR_BETS[bet][0]} loses")
        else:
            back += amount
            notes.append(f"{CR_BETS[bet][0]} pushes")
        b[bet] = 0

    # one-roll bets
    if b["field"]:
        settle("field", 2 if s == 2 else 3 if s == 12 else 1 if s in (3, 4, 9, 10, 11) else -1)
    if b["any7"]:
        settle("any7", 4 if s == 7 else -1)
    # the line bets and the point
    if point is None:
        if s in (7, 11, 2, 3, 12):
            pass_wins = s in (7, 11)
            g["ai_record"][0 if pass_wins else 1] += 1   # the AI's own Pass Line bet, every round
            if b["pass"]:
                settle("pass", 1 if pass_wins else -1)
            if b["dont"]:
                settle("dont", 0 if s == 12 else -1 if pass_wins else 1)
            text = {7: "Seven - a natural", 11: "Yo-leven - a natural", 2: "Snake eyes - craps", 3: "Craps",
                    12: "Boxcars - craps"}[s]
        else:
            g["point"] = s
            text = f"The point is {s}"
    else:
        if s in (6, 8) and b[f"place{s}"]:   # a place bet wins and stays up - only the winnings are paid
            back += int(b[f"place{s}"] * 7 / 6)
            staked += b[f"place{s}"]
            notes.append(f"Place {s} wins {int(b[f'place{s}'] * 7 / 6):,}")
        if s == point or s == 7:
            pass_wins = s == point
            g["ai_record"][0 if pass_wins else 1] += 1
            if b["odds"]:
                settle("odds", CR_ODDS_PAY[point] if pass_wins else -1)
            if b["pass"]:
                settle("pass", 1 if pass_wins else -1)
            if b["dont"]:
                settle("dont", -1 if pass_wins else 1)
            if s == 7:
                for bet in ("place6", "place8"):
                    if b[bet]:
                        settle(bet, -1)
            g["point"] = None
            text = f"{s} - the point is made! Pass wins" if pass_wins else "Seven out - the point is lost"
        else:
            text = f"{s} - no decision, roll again (the point is {point})"
    wallet["chips"] += back
    net = back - staked
    if notes:
        g["record"][0 if net > 0 else 2 if net == 0 else 1] += 1
    g.update(dice=(d1, d2), rolled=True, rolls=g["rolls"] + 1, history=(g["history"] + [s])[-20:])
    g["msg"] = f"{d1} + {d2} = {s}. {text}." + (f" {' · '.join(notes)}." if notes else "")


def cr_simulate(rolls):
    """the Strategy lab: six ways to bet on the SAME dice, flat 1 chip a bet"""
    dice = [random.randint(1, 6) + random.randint(1, 6) for _ in range(rolls)]

    def line(side, odds=False):   # side 1 = Pass, -1 = Don't Pass, a bet on every come-out
        point, profit, staked, won, lost = None, 0, 0, 0, 0
        for s in dice:
            if point is None:
                if s in (4, 5, 6, 8, 9, 10):
                    point = s
                    continue
                r = (1 if s in (7, 11) else -1) * side if s != 12 else (-1 if side == 1 else 0)
                profit, staked, won, lost = profit + r, staked + 1, won + (r > 0), lost + (r < 0)
            elif s in (point, 7):
                win = (s == point) == (side == 1)
                r, bet = (1 if win else -1), 1
                if odds:
                    r += CR_ODDS_MAX[point] * (CR_ODDS_PAY[point] if win else -1)
                    bet += CR_ODDS_MAX[point]
                profit, staked, won, lost = profit + r, staked + bet, won + win, lost + (not win)
                point = None
        return profit, staked, won, lost

    def place():   # 1 chip on each of 6 and 8 while there's a point
        point, profit, staked, won, lost = None, 0, 0, 0, 0
        for s in dice:
            if point is None:
                point = s if s in (4, 5, 6, 8, 9, 10) else None
                continue
            if s in (6, 8):
                profit, staked, won = profit + 7 / 6, staked + 1, won + 1
            elif s == 7:
                profit, staked, lost = profit - 2, staked + 2, lost + 2
            if s in (point, 7):
                point = None
        return profit, staked, won, lost

    def one_roll(pays):
        results = [pays(s) for s in dice]
        return sum(results), len(results), sum(r > 0 for r in results), sum(r < 0 for r in results)

    styles = [("Pass Line + full odds (what the AI does)", line(1, True), -0.37), ("Pass Line", line(1), -1.41),
              ("Don't Pass", line(-1), -1.36), ("Place 6 and 8", place(), -1.52),
              ("Field every roll", one_roll(lambda s: 2 if s == 2 else 3 if s == 12 else 1 if s in (3, 4, 9, 10, 11) else -1), -2.78),
              ("Any 7 every roll", one_roll(lambda s: 4 if s == 7 else -1), -16.67)]
    return pd.DataFrame([{"Strategy": name, "Win rate": won / max(won + lost, 1),
                          "Profit per 100 chips bet": profit / staked * 100, "Long run (exact math)": exact}
                         for name, (profit, staked, won, lost), exact in styles]), None


def cr_die(value, delay):
    pips = "".join(f'<span class="{"cr-pip" if i in CR_PIPS[value] else ""}"></span>' for i in range(9))
    return f'<div class="cr-die{" roll" if delay is not None else ""}" style="animation-delay:{delay or 0}s">{pips}</div>'


@st.fragment
def craps():
    """the craps table - a fragment, so betting / rolling only redraws the game"""
    g, wallet = cr_game(), bj_game()
    b, point = g["bets"], g["point"]
    st.html(CHIP_CSS + """<style>
      .st-key-cr_felt { background:radial-gradient(ellipse at 50% 0%,#7F1D1D 0%,#3B0A0A 45%,#140303 100%);
                        border:2px solid #7F1D1D; border-radius:22px; padding:22px 26px 16px;
                        box-shadow:inset 0 0 60px rgba(0,0,0,.55); }
      .cr-dice { display:flex; justify-content:center; gap:18px; min-height:84px; align-items:center; }
      .cr-die { width:66px; height:66px; background:#FAFAFA; border-radius:13px; padding:9px; box-sizing:border-box;
                display:grid; grid-template:repeat(3,1fr)/repeat(3,1fr); box-shadow:0 6px 16px rgba(0,0,0,.55); }
      .cr-pip { width:12px; height:12px; border-radius:50%; background:#111; place-self:center; }
      .cr-die.roll { animation:cr-roll .95s cubic-bezier(.2,.8,.25,1) both; }
      @keyframes cr-roll { 0% { transform:translate(320px,-60px) rotate(0); opacity:0; }
                           15% { opacity:1; } 70% { transform:translate(-12px,8px) rotate(-640deg); }
                           100% { transform:none; } }
      .cr-puck { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:800; font-size:13px;
                 letter-spacing:.08em; }
      .cr-spot { text-align:center; color:#FECACA; font-size:13px; font-weight:800; letter-spacing:.06em; }
      .cr-pays { text-align:center; color:#A3A3A3; font-size:11px; min-height:16px; }
      .cr-hist { display:flex; flex-wrap:wrap; gap:4px; justify-content:center; margin-top:6px; }
      .cr-hist span { min-width:22px; height:22px; border-radius:6px; background:#262626; color:#E5E7EB;
                      font-size:11px; display:inline-flex; align-items:center; justify-content:center; }
      .cr-hist span.seven { background:#B91C1C; }
    </style>""")
    table, side = st.columns([5, 2], gap="large")
    with table:
        with st.container(key="cr_felt"):
            rolled = g.pop("rolled", False)
            dice = (cr_die(g["dice"][0], 0 if rolled else None) + cr_die(g["dice"][1], 0.08 if rolled else None)
                    if g["dice"] else '<div style="color:#A3A3A3">Put chips down, then roll the dice.</div>')
            puck = (f'<span class="cr-puck" style="background:#FAFAFA;color:#111">ON · POINT {point}</span>' if point
                    else '<span class="cr-puck" style="background:#111;color:#A3A3A3;border:1px solid #444">OFF · COME-OUT ROLL</span>')
            hist = "".join(f'<span class="{"seven" if s == 7 else ""}">{s}</span>' for s in g["history"])
            st.html(f'<div style="text-align:center">{puck}</div><div class="cr-dice">{dice}</div>'
                    f'<div class="bj-msg">{html.escape(g["msg"]) or "Every bet waits for the dice. Start with the Pass Line."}</div>'
                    f'<div class="cr-hist">{hist}</div>')
            drop = g.pop("drop", None)
            for row in (("pass", "dont", "odds"), ("place6", "place8", "field", "any7")):
                for col, bet in zip(st.columns(len(row)), row):
                    name, pays, edge = CR_BETS[bet]
                    locked = (bet in ("pass", "dont") and point) or (bet == "odds" and not (point and b["pass"]))
                    if bet == "odds" and point:
                        pays = f"{CR_ODDS_PAY[point]:g} to 1 · up to {cr_odds_max(g):,}"
                    col.html(f'<div class="cr-spot">{name.upper()}</div><div class="cr-pays">{pays}</div>'
                             f'<div class="pq-circle" style="width:86px;height:86px">{chip_stack_html(b[bet], drop == bet, empty="")}</div>')
                    col.button("Bet", key=f"crb_{bet}", on_click=cr_place, args=(bet,), width="stretch",
                               disabled=bool(locked), help=f"House edge {edge:.2f}%")
            chip_rack("cr", cr_pick_chip, lambda v: v <= wallet["chips"], selected=g["chip"])
            clear_col, roll_col = st.columns(2)
            clear_col.button("Take bets back", on_click=cr_clear, key="cr_clear", width="stretch",
                             icon=":material/undo:", help="Pass / Don't Pass stay locked in while there's a point")
            roll_col.button("Roll the dice", on_click=cr_roll, key="cr_roll", width="stretch", type="primary",
                            icon=":material/casino:")
        with st.expander("How craps works", icon=":material/school:"):
            st.markdown(
                "**The come-out roll** (the puck says OFF): a **Pass Line** bet wins on **7 or 11** and loses on "
                "**2, 3 or 12**. Any other number (4, 5, 6, 8, 9, 10) becomes **the point**.\n\n"
                "**Then** you keep rolling: Pass wins if the **point comes again before a 7**, and loses on a 7 "
                "(\"seven out\"). **Don't Pass** is the opposite (and a 12 on the come-out is a push).\n\n"
                "**Odds** (once there's a point): extra chips behind your Pass bet, paid at **true odds** - 2 to 1 "
                "on 4 / 10, 3 to 2 on 5 / 9, 6 to 5 on 6 / 8. **0% house edge** - the only fair bet in the "
                "casino. Up to 3x / 4x / 5x your Pass bet.\n\n"
                "**Place 6 / 8:** wins 7 to 6 whenever a 6 / 8 rolls before a 7, and stays up. Off on the come-out. "
                "**Field** and **Any 7** are one-roll bets.\n\n"
                "**House edge:** Odds 0% · Don't Pass 1.36% · Pass 1.41% · Place 6 / 8 1.52% · Field 2.78% · "
                "Any 7 16.67%. Real tables pay in whole chips (bet multiples of 6 on Place, 5 on 6 / 8 odds) - this "
                "one rounds down too. Fake chips - just for fun.")
        strategy_lab("cr")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every game - chips on the craps table come back when a bet is decided")
        st.metric("Record (W-L-P)", f"{won}-{lost}-{pushed}", border=True, help="Rolls that decided at least one bet")
        if wallet["chips"] < 10 and not any(b.values()):
            st.button("Out of chips - get 1,000 more", on_click=bj_reset, key="cr_reset", icon=":material/refresh:",
                      width="stretch", type="primary")
        with st.container(border=True):
            st.markdown("**:material/psychology: Craps AI**")
            trainer_switch("cr", "rolled with your bets down")
            size = max(10, round(wallet["chips"] * 0.01 / 5) * 5)
            if trainer_on("cr"):
                trainer_quiet()
                trainer_panel("cr")
            else:
                if point and b["pass"] and b["odds"] < cr_odds_max(g):
                    add = cr_odds_max(g) - b["odds"]
                    headline = f"Add {add:,} odds behind your Pass"
                    why = (f"Odds pay true odds ({CR_ODDS_PAY[point]:g} to 1 on a {point}) - a <b>0% house edge</b>, the "
                           f"only fair bet in the casino. Full odds bring your whole bet down to about <b>0.37%</b>.")
                elif point and b["pass"]:
                    headline = "You're set - roll"
                    why = (f"Full odds are down. From here Pass wins <b>{CR_POINT_WIN[point]:.0%}</b> of the time on a "
                           f"{point} - and when it does, the odds pay enough to make it a fair bet.")
                elif point:
                    headline = "Wait for the come-out"
                    why = ("Pass / Don't Pass go on before the come-out roll. If you want action now, Place 6 / 8 is "
                           "the cheapest (1.52%).")
                else:
                    headline = f"Bet {size:,} on the Pass Line"
                    why = ("1.41% house edge, and it wins 49.3% of rounds. When a point comes up, add full odds. "
                           "(Don't Pass is a hair cheaper at 1.36% - betting with the dice or against them is up to you.)")
                bad = [f"{CR_BETS[k][0]} ({CR_BETS[k][2]:.2f}%)" for k in ("field", "any7") if b[k]]
                if bad:
                    why += f"<br><b>Take down:</b> {', '.join(bad)} - those lose much faster."
                ai_box(headline, why)
            ai_rate_caption(g["ai_record"], "rounds", "betting the Pass Line every round (about 49% long run)")
            st.caption("The AI can't know the dice - nobody can. It only knows the math of each bet: the edge, the "
                       "payouts and the chance of a point. The dice have no memory, so there are no hot or cold "
                       "streaks to follow.")


# ---------------------------------------------------------------- texas hold'em poker - fake chips, just for fun
# Heads-up No-Limit Hold'em against "Ace", a bot. Blinds 5 / 10. Ace only knows its own 2 cards, the board and how you've
# bet - the advisor only knows yours. Both work out "equity" (the chance to win at showdown) by dealing out the unknown
# cards hundreds of times, then compare it with the price of calling ("pot odds").
PK_SB, PK_BB = 5, 10
PK_DECK = [(rank, suit) for rank in BJ_RANKS for suit in BJ_SUITS]
PK_NAMES = ["High card", "Pair", "Two pair", "Three of a kind", "Straight", "Flush", "Full house", "Four of a kind",
            "Straight flush"]
PK_WHO = {"you": "You", "bot": "Ace"}


def pk_straight(values):
    """the top card of the best straight in these values (0 = none); an ace also plays low (A-2-3-4-5)"""
    v = sorted(set(values), reverse=True)
    if 14 in v:
        v.append(1)
    for i in range(len(v) - 4):
        if v[i] - v[i + 4] == 4:
            return v[i]
    return 0


def pk_eval(cards):
    """the best 5-card hand in 5-7 cards, as a tuple that compares the right way: (category, tie-breaks...)"""
    vals = sorted((VP_VALUE[r] for r, _ in cards), reverse=True)
    suits = [s for _, s in cards]
    flush = next((s for s in BJ_SUITS if suits.count(s) >= 5), None)
    if flush:
        fvals = sorted((VP_VALUE[r] for r, s in cards if s == flush), reverse=True)
        top = pk_straight(fvals)
        if top:
            return (8, top)
    groups = sorted(Counter(vals).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    v1, c1 = groups[0]
    if c1 == 4:
        return (7, v1, max(v for v in vals if v != v1))
    if c1 == 3:
        pair = max((v for v, c in groups[1:] if c >= 2), default=0)
        if pair:
            return (6, v1, pair)
    if flush:
        return (5, *fvals[:5])
    top = pk_straight(vals)
    if top:
        return (4, top)
    if c1 == 3:
        return (3, v1, *[v for v in vals if v != v1][:2])
    pairs = [v for v, c in groups if c == 2]
    if len(pairs) >= 2:
        return (2, pairs[0], pairs[1], max(v for v in vals if v not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *[v for v in vals if v != pairs[0]][:3])
    return (0, *vals[:5])


def pk_chen(hand):
    """the Chen formula: a quick score for two starting cards (high cards, pairs, suited, connected)"""
    (r1, s1), (r2, s2) = hand
    a, b = sorted((VP_VALUE[r1], VP_VALUE[r2]), reverse=True)
    score = {14: 10, 13: 8, 12: 7, 11: 6}.get(a, a / 2)
    if a == b:
        return max(score * 2, 5)
    gap = a - b - 1
    return score + 2 * (s1 == s2) - ([0, 1, 2, 4][gap] if gap < 4 else 5) + (gap <= 1 and a < 12)


PK_RANGE = sorted(itertools.combinations(PK_DECK, 2), key=pk_chen, reverse=True)   # all 1,326 starting hands, best first
PK_PREFLOP = {}   # preflop equity by hand shape (AKs, 72o...) - the same for every suit, so it's worked out once


def pk_equity(hole, board, top=1.0, n=400, seed=None):
    """the chance these 2 cards win at showdown against one opponent holding any hand from the best `top` share of
    starting hands - every unknown card dealt out n times (a tie counts half)"""
    if not board:
        (r1, s1), (r2, s2) = hole
        shape = (*sorted((VP_VALUE[r1], VP_VALUE[r2])), s1 == s2, top, n)
        if shape in PK_PREFLOP:
            return PK_PREFLOP[shape]
    rng = random.Random(seed)
    known = set(hole) | set(board)
    hands = [h for h in PK_RANGE[:max(40, int(len(PK_RANGE) * top))] if not (set(h) & known)]
    deck = [c for c in PK_DECK if c not in known]
    need, score = 5 - len(board), 0.0
    for _ in range(n):
        opp = rng.choice(hands)
        run = board + rng.sample([c for c in deck if c not in opp], need)
        me, them = pk_eval(hole + run), pk_eval(list(opp) + run)
        score += 1 if me > them else 0.5 if me == them else 0
    if not board:
        PK_PREFLOP[shape] = score / n
    return score / n


@st.cache_data(max_entries=600, show_spinner=False)
def pk_equity_cached(hole, board, top):
    return pk_equity(list(hole), list(board), top, n=700, seed=f"{hole}{board}{top}")


def pk_read(h, who):
    """what `who`'s betting says: any two cards, or (after raising) mostly the better hands"""
    return {0: 1.0, 1: 0.5}.get(h["raises"][who], 0.3)


def pk_other(who):
    return "bot" if who == "you" else "you"


def pk_to_call(h, who):
    return h["bet"][pk_other(who)] - h["bet"][who]


def pk_put(h, who, amount):
    amount = max(0, min(int(amount), h["stack"][who]))
    h["stack"][who] -= amount
    h["bet"][who] += amount
    h["pot"] += amount
    return amount


def pk_raise_add(h, who, frac):
    """chips `who` puts in to bet / raise by frac x the pot (None = all-in) - never more than the other can match"""
    other = pk_other(who)
    cap = min(h["stack"][who], h["bet"][other] + h["stack"][other] - h["bet"][who])
    if frac is None:
        return cap
    to_call = pk_to_call(h, who)
    raise_by = max(frac * (h["pot"] + to_call), h["last_raise"], PK_BB)
    return int(min(cap, to_call + round(raise_by / 5) * 5))


def pk_can_raise(h, who):
    return pk_raise_add(h, who, None) > pk_to_call(h, who)


def pk_start(stacks, button, deck=None, names=None):
    """a new hand: deal 2 cards each, post the blinds (heads-up the button is the small blind and acts first)"""
    deck = list(deck or random.sample(PK_DECK, 52))
    h = {"deck0": list(deck), "deck": deck, "hole": {"you": [], "bot": []}, "board": [], "street": 0,
         "start": dict(stacks), "stack": dict(stacks), "bet": {"you": 0, "bot": 0}, "pot": 0, "button": button,
         "acted": [], "last_raise": PK_BB, "raises": {"you": 0, "bot": 0}, "log": [], "done": False,
         "actions": [], "names": names or PK_WHO}   # actions: (street, seat, kind, total in) - your CFR bot's format
    for _ in range(2):
        for who in (pk_other(button), button):
            h["hole"][who].append(h["deck"].pop())
    pk_put(h, button, PK_SB)
    pk_put(h, pk_other(button), PK_BB)
    h["to_act"] = button
    return h


def pk_in(h, who):
    """everything `who` has put in this hand (all streets)"""
    return h["start"][who] - h["stack"][who]


def pk_add_to(h, who, total):
    """chips to add so `who` has `total` in this hand - at least a legal min-raise, at most all-in"""
    cap, to_call = pk_raise_add(h, who, None), pk_to_call(h, who)
    min_add = to_call + max(h["last_raise"], PK_BB)
    return int(max(min(total - pk_in(h, who), cap), min(min_add, cap)))


def pk_act(h, who, action, frac=None, to=None):
    """one move: fold / call (a check when there's nothing to call) / raise (a bet when nobody has bet yet).
    A raise is sized by `frac` of the pot (None = all-in) - or `to` = the total to have in this hand."""
    to_call, verb = pk_to_call(h, who), (lambda word: word if who == "you" else word + "s")
    name, seat = h["names"][who], 0 if h["button"] == who else 1
    if action == "fold":
        h["log"].append(f"{name} {verb('fold')}")
        h["actions"].append((h["street"], seat, "fold", pk_in(h, who)))
        return pk_finish(h, pk_other(who))
    if action == "call":
        put = pk_put(h, who, to_call)
        h["log"].append(f"{name} {verb('call')} {put:,}" if to_call else f"{name} {verb('check')}")
        h["actions"].append((h["street"], seat, "call", pk_in(h, who)))
        if who not in h["acted"]:
            h["acted"].append(who)
    else:
        put = pk_put(h, who, pk_raise_add(h, who, frac) if to is None else pk_add_to(h, who, to))
        h["last_raise"] = max(put - to_call, PK_BB)
        h["raises"][who] += 1
        h["acted"] = [who]
        h["actions"].append((h["street"], seat, "raise", pk_in(h, who)))
        word = "raise" if to_call or h["street"] == 0 else "bet"
        h["log"].append(f"{name} {verb(word)} {'to ' if word == 'raise' else ''}{h['bet'][who]:,}"
                        + (" (all-in)" if h["stack"][who] == 0 else ""))
    h["to_act"] = pk_other(who)
    b = h["bet"]
    if set(h["acted"]) != {"you", "bot"} or (b["you"] != b["bot"] and min(h["stack"].values()) > 0):
        return   # the other player still has to act
    big = max(b, key=b.get)   # a short all-in call: the part that couldn't be matched goes back
    extra = b[big] - b[pk_other(big)]
    h["stack"][big], h["pot"], b[big] = h["stack"][big] + extra, h["pot"] - extra, b[big] - extra
    if h["street"] == 3 or min(h["stack"].values()) == 0:   # river done, or someone's all-in: deal it out
        h["board"] += [h["deck"].pop() for _ in range(5 - len(h["board"]))]
        return pk_finish(h)
    h["street"] += 1
    h["board"] += [h["deck"].pop() for _ in range(3 if h["street"] == 1 else 1)]
    h["log"].append(f"- {['Flop', 'Turn', 'River'][h['street'] - 1]} -")
    h.update(bet={"you": 0, "bot": 0}, acted=[], last_raise=PK_BB, to_act=pk_other(h["button"]))


def pk_finish(h, winner="showdown"):
    if winner == "showdown":
        you, bot = (pk_eval(h["hole"][w] + h["board"]) for w in ("you", "bot"))
        winner = "you" if you > bot else "bot" if bot > you else None
        h["showdown"] = (you, bot)
    if winner:
        h["stack"][winner] += h["pot"]
    else:   # a tie splits the pot
        h["stack"]["you"] += h["pot"] // 2
        h["stack"]["bot"] += h["pot"] - h["pot"] // 2
    h.update(done=True, winner=winner)


def pk_best(eq, odds, to_call, can_raise):
    """the advisor's rule: raise strong hands, call when your chance beats the price, otherwise check / fold"""
    if to_call:
        if eq >= 0.7 and can_raise:
            return "raise", 1.0 if eq >= 0.82 else 0.5
        return ("call", None) if eq >= odds else ("fold", None)
    if eq >= 0.6 and can_raise:
        return "raise", 1.0 if eq >= 0.78 else 0.5
    return "call", None


def pk_decide(h, who, style, rng, n=300):
    """a player's move, knowing only their own cards, the board and the other's betting"""
    to_call, can_raise = pk_to_call(h, who), pk_can_raise(h, who)
    if style == "cfr":
        return pk_cfr_move(h, who)
    if style == "station":
        return "call", None
    if style == "maniac":
        return ("raise", 1.0) if can_raise else ("call", None)
    eq = pk_equity(h["hole"][who], h["board"], pk_read(h, pk_other(who)), n=n, seed=rng.random())
    odds = to_call / (h["pot"] + to_call) if to_call else 0
    if style == "rock":   # waits for a strong hand
        if eq < 0.58:
            return ("fold", None) if to_call else ("call", None)
        return pk_best(eq, odds, to_call, can_raise)
    if style == "ace":    # Ace: the same math, a little unpredictable, and it bluffs now and then
        if h["street"] >= 1 and can_raise and rng.random() < 0.08:
            return "raise", 0.5
        eq += rng.uniform(-0.05, 0.05)
    return pk_best(eq, odds, to_call, can_raise)


def pk_play_out(h, you_style, rng, n=300, bot_style="ace"):
    while not h["done"]:
        who = h["to_act"]
        pk_act(h, who, *pk_decide(h, who, you_style if who == "you" else bot_style, rng, n))
    return h


# ---- your own poker AI (the "poker training" folder): a CFR bot that taught itself over ~24 million hands ----
PK_BOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poker training")
PK_SUIT_CODE = {"♣": 0, "♦": 1, "♥": 2, "♠": 3}   # its cards are ints: rank (2..A = 0..12) * 4 + suit (c d h s)
PK_STACK = 100 * PK_BB   # it was trained 100 big blinds deep, so every hand is played 1,000 chips deep


@st.cache_resource(show_spinner="Loading your poker bot...")
def pk_cfr():
    """your trained bot (checkpoints/blueprint.pkl) - or None if it isn't there"""
    path = os.path.join(PK_BOT_DIR, "checkpoints", "blueprint.pkl")
    if not os.path.exists(path):
        return None
    if PK_BOT_DIR not in sys.path:
        sys.path.insert(0, PK_BOT_DIR)
    from pokerbot.bots import CFRBot
    return CFRBot(path)


def pk_card_int(card):
    return (VP_VALUE[card[0]] - 2) * 4 + PK_SUIT_CODE[card[1]]


def pk_obs(h, who):
    """the table as your bot sees it: only `who`'s own cards, the board and the betting (seat 0 = the button)"""
    seat, other = 0 if h["button"] == who else 1, pk_other(who)
    contrib = [0, 0]
    contrib[seat], contrib[1 - seat] = pk_in(h, who), pk_in(h, other)
    max_to = contrib[seat] + pk_raise_add(h, who, None)
    return {"seat": seat, "hole": [pk_card_int(c) for c in h["hole"][who]],
            "board": [pk_card_int(c) for c in h["board"]], "street": h["street"], "contrib": contrib,
            "to_call": pk_to_call(h, who), "can_raise": pk_can_raise(h, who),
            "min_raise_to": min(contrib[1 - seat] + max(h["last_raise"], PK_BB), max_to), "max_raise_to": max_to,
            "stack": min(h["start"].values()), "sb": PK_SB, "bb": PK_BB, "actions": list(h["actions"])}


def pk_cfr_move(h, who):
    """your bot's move for `who` - its trained strategy (or its own equity fallback off the tree)"""
    kind, amount = pk_cfr().act(pk_obs(h, who))
    if kind == "raise" and pk_can_raise(h, who):
        return "raise", None, int(round(amount))
    return ("fold", None) if kind == "fold" and pk_to_call(h, who) else ("call", None)


def pk_cfr_strategy(h, who="you"):
    """what your bot plays in this exact spot, as a mix: [(kind, move name, probability, move args)] -
    or None when the spot is off its training tree (unusual bet sizes, very short stacks)"""
    bot, obs = pk_cfr(), pk_obs(h, who)
    node = bot._sync(obs)
    if node is None or node.term or node.player != obs["seat"] or node.street != obs["street"]:
        return None
    probs = bot.bp.get(f"{bot._bucket(obs['hole'], obs['board'])}:{node.hist}")
    if probs is None:
        return None
    mix = {}
    for tok, p in zip(node.acts, probs):
        kind, amount = bot.token_to_action(tok, node, obs)
        if kind == "raise" and obs["can_raise"]:
            to = int(round(amount))
            street_to = to - (pk_in(h, who) - h["bet"][who])   # the table shows this street's bet
            word = "Raise to" if obs["to_call"] or h["street"] == 0 else "Bet"
            name = f"{word} {street_to:,}" + (" (all-in)" if to >= obs["max_raise_to"] else "")
            args = ("raise", None, to)
        elif kind == "fold":
            name, args = "Fold", ("fold", None)
        else:
            name, args = (f"Call {min(obs['to_call'], h['stack'][who]):,}" if obs["to_call"] else "Check"), ("call", None)
        old = mix.get(name, (kind, name, 0.0, args))
        mix[name] = (kind, name, old[2] + p, args)
    return sorted(mix.values(), key=lambda m: -m[2])


def pk_game():
    return st.session_state.setdefault("pk", {"hand": None, "button": "you", "record": [0, 0, 0],
                                              "ai_record": [0, 0, 0], "followed": [0, 0], "seen_board": 0, "msg": ""})


def pk_coach_is_cfr():
    """the coach (advisor + trainer): your CFR bot when it's there and picked, else the equity advisor"""
    return pk_cfr() is not None and st.session_state.get("pk_coach", "Your CFR bot") == "Your CFR bot"


def pk_advice(h):
    """the advisor: your chance to win (against the hands Ace's betting suggests) vs the price of calling"""
    top = pk_read(h, "bot")
    eq = pk_equity_cached(tuple(h["hole"]["you"]), tuple(h["board"]), top)
    to_call = pk_to_call(h, "you")
    odds = to_call / (h["pot"] + to_call) if to_call else 0.0
    action, frac = pk_best(eq, odds, to_call, pk_can_raise(h, "you"))
    return {"equity": eq, "odds": odds, "action": action, "frac": frac, "top": top, "to_call": to_call}


def pk_move_name(h, action, frac):
    to_call = pk_to_call(h, "you")
    if action == "fold":
        return "Fold"
    if action == "call":
        return f"Call {min(to_call, h['stack']['you']):,}" if to_call else "Check"
    total = h["bet"]["you"] + pk_raise_add(h, "you", frac)
    return f"{'Raise to' if to_call or h['street'] == 0 else 'Bet'} {total:,}" + (" (all-in)" if frac is None else "")


def pk_bot_turns(g):
    h = g["hand"]
    while h and not h["done"] and h["to_act"] == "bot":
        pk_act(h, "bot", *pk_decide(h, "bot", h.get("bot_style", "ace"), random))


def pk_sync(g):
    """your chips follow the hand as it goes; at the end, the result is recorded once"""
    h, wallet = g["hand"], bj_game()
    wallet["chips"] = h.get("reserve", 0) + h["stack"]["you"]   # chips you didn't bring to the table stay put
    if not h["done"] or h.get("counted"):
        return
    h["counted"] = True
    net = h["stack"]["you"] - h["start"]["you"]
    result = 0 if net > 0 else 2 if net == 0 else 1
    g["record"][result] += 1
    if g["hand_followed"]:   # you did what the coach said every time this hand
        g["followed"][0] += net > 0
        g["followed"][1] += net != 0
    # the AI's own result: the same deck and the same opponent, with the coach making every one of your moves
    replay = pk_play_out(pk_start(h["start"], h["button"], h["deck0"]), "cfr" if h.get("coach") == "cfr" else "advisor",
                         random.Random(str(h["deck0"][:4])), 250, bot_style=h.get("bot_style", "ace"))
    ai_net = replay["stack"]["you"] - h["start"]["you"]
    g["ai_record"][0 if ai_net > 0 else 2 if ai_net == 0 else 1] += 1
    them = h["names"]["bot"]
    if "showdown" in h:
        you, bot = h["showdown"]
        g["msg"] = (f"Showdown: your {PK_NAMES[you[0]].lower()} vs {them}'s {PK_NAMES[bot[0]].lower()} - "
                    + ("you win" if h["winner"] == "you" else f"{them} wins" if h["winner"] == "bot" else "a split pot")
                    + f" ({net:+,}).")
    else:
        g["msg"] = (f"{them} folds - you win the pot" if h["winner"] == "you" else f"You fold - {them} wins the pot") \
            + f" ({net:+,})."


def pk_deal():
    g, wallet = pk_game(), bj_game()
    if wallet["chips"] < PK_BB:
        g["msg"] = "You need at least 10 chips for the big blind."
        return
    g["button"] = pk_other(g["button"])   # the button moves every hand
    style = "cfr" if pk_cfr() is not None and st.session_state.get("pk_opp") == "Your CFR bot" else "ace"
    you = min(wallet["chips"], PK_STACK)   # a cash game, 100 big blinds deep - how your bot was trained
    h = pk_start({"you": you, "bot": PK_STACK}, g["button"],
                 names={"you": "You", "bot": "Your bot" if style == "cfr" else "Ace"})
    h.update(reserve=wallet["chips"] - you, bot_style=style, coach="cfr" if pk_coach_is_cfr() else "equity")
    g.update(hand=h, hand_followed=True, seen_board=0, fresh=True, msg="")
    pk_bot_turns(g)
    pk_sync(g)


def pk_kinds(mix):
    """a strategy mix added up by kind of move: {'fold': p, 'call': p, 'raise': p}"""
    out = {}
    for kind, _, p, _ in mix:
        out[kind] = out.get(kind, 0) + p
    return out


def pk_you(action, frac=None, to=None):
    """your move - graded by the coach first (it only ever sees your cards), then the opponent answers"""
    g = pk_game()
    h = g["hand"]
    if not h or h["done"] or h["to_act"] != "you":
        return
    mix = pk_cfr_strategy(h) if pk_coach_is_cfr() else None
    if mix:   # coach = your CFR bot: a move it plays at least a quarter of the time here (or its favourite) is right
        kinds = pk_kinds(mix)
        mine = "raise" if action == "raise" else action
        share = kinds.get(mine, 0)
        ok = share >= 0.25 or mine == max(kinds, key=kinds.get)
        trainer_grade("pk", ok, True,
                      "Your CFR bot plays this spot: " + " · ".join(f"**{name}** {p:.0%}" for _, name, p, _ in mix
                                                                    if p >= 0.01)
                      + f". You chose {pk_move_name(h, action, frac).lower()} - it does that **{share:.0%}** of the "
                        f"time. " + ("Good poker mixes moves on purpose, so you're hard to read." if ok else
                                     "That's a move it almost never makes here."))
        if not ok:
            g["hand_followed"] = False
        pk_act(h, "you", action, frac, to)
        pk_bot_turns(g)
        return pk_sync(g)
    adv = pk_advice(h)
    kind = lambda a: "raise" if a == "raise" else a if a == "fold" else "check" if not adv["to_call"] else "call"
    yours, best = kind(action), kind(adv["action"])
    close = abs(adv["equity"] - adv["odds"]) < 0.03 and {yours, best} <= {"call", "fold"}   # a coin flip either way
    trainer_grade("pk", best if close else yours, best,
                  f"Your chance to win was **{adv['equity']:.0%}**"
                  + (f" and calling cost {adv['to_call']:,} into a pot of {h['pot']:,}, so you needed "
                     f"**{adv['odds']:.0%}**. " if adv["to_call"] else " with nothing to call. ")
                  + f"Best play: **{pk_move_name(h, adv['action'], adv['frac'])}**. "
                  + {"fold": "Your chance was below the price - folding saves chips.",
                     "call": "Your chance beat the price, so calling makes money over time.",
                     "check": "Not strong enough to bet for value - check and see the next card free.",
                     "raise": "A strong hand: bet so the pot is bigger when you win."}[best])
    if yours != best and not close:
        g["hand_followed"] = False
    pk_act(h, "you", action, frac, to)
    pk_bot_turns(g)
    pk_sync(g)


def pk_simulate(hands):
    """the Strategy lab: each style plays the SAME deals against Ace (100 big blinds deep, a fresh stack each hand)"""
    decks = [random.sample(PK_DECK, 52) for _ in range(hands)]
    styles = ({"Your CFR bot (poker training folder)": "cfr"} if pk_cfr() is not None else {}) | {
        "Equity + pot odds (the equity advisor)": "advisor", "Rock (strong hands only)": "rock",
        "Calling station (always calls)": "station", "Maniac (always raises the pot)": "maniac",
        "Ace's own style (a mirror match)": "ace"}
    rows = []
    for name, style in styles.items():
        rng, profit, won, lost = random.Random(3), 0, 0, 0
        for i, deck in enumerate(decks):
            h = pk_play_out(pk_start({"you": 1000, "bot": 1000}, "you" if i % 2 else "bot", deck), style, rng, n=120)
            net = h["stack"]["you"] - 1000
            profit, won, lost = profit + net, won + (net > 0), lost + (net < 0)
        rows.append({"Strategy": name, "Win rate": won / max(won + lost, 1),
                     "Big blinds won per 100 hands": profit / PK_BB / hands * 100})
    return pd.DataFrame(rows)


def pk_cards(cards, hidden=False, new_from=None, flip=False):
    out = ""
    for i, (rank, suit) in enumerate(cards):
        anim = (f' new" style="animation-delay:{(i - new_from) * DEAL_GAP:.2f}s' if new_from is not None and i >= new_from
                else " flip" if flip else "")
        out += (f'<span class="bj-card back{anim}"></span>' if hidden else
                f'<span class="bj-card {"red" if BJ_SUITS[suit] else ""}{anim}"><b>{rank}</b>'
                f'<span class="bj-suit">{suit}</span></span>')
    return out


@st.fragment
def poker():
    """the hold'em table - a fragment, so each move only redraws the game"""
    g, wallet = pk_game(), bj_game()
    h = g["hand"]
    live = h is not None and not h["done"]
    your_turn = live and h["to_act"] == "you"
    st.html(CARD_CSS + CHIP_CSS + """<style>
      .st-key-pk_felt { background:radial-gradient(ellipse at 50% 0%,#1E3A8A 0%,#0F1F4A 45%,#050A1A 100%);
                        border:2px solid #1E3A8A; border-radius:22px; padding:22px 26px 16px;
                        box-shadow:inset 0 0 60px rgba(0,0,0,.55); }
      .pk-row { display:flex; align-items:center; justify-content:center; gap:22px; flex-wrap:wrap; }
      .pk-pot { color:#FDE68A; font-weight:800; font-size:15px; letter-spacing:.06em; text-align:center; margin:4px 0; }
      .pk-log { color:#94A3B8; font-size:12px; text-align:center; min-height:18px; }
      .pk-slot { width:64px; height:92px; border-radius:9px; border:2px dashed rgba(191,219,254,.25); display:inline-block; }
    </style>""")
    has_cfr = pk_cfr() is not None
    them = h["names"]["bot"] if h else ("Your bot" if has_cfr and st.session_state.get("pk_opp") == "Your CFR bot" else "Ace")
    table, side = st.columns([5, 2], gap="large")
    with table:
        if has_cfr:   # who you play: Ace, or the bot you trained yourself (only between hands)
            st.segmented_control("Opponent", ["Ace", "Your CFR bot"], default="Ace", key="pk_opp", disabled=live,
                                 help="Your CFR bot = the one in the 'poker training' folder: it taught itself by "
                                      "playing ~24 million hands against itself. Changes from the next hand.")
        with st.container(key="pk_felt"):
            if h:
                fresh = g.pop("fresh", False)
                shown = "showdown" in h
                new_board = g["seen_board"] if len(h["board"]) > g["seen_board"] else None
                g["seen_board"] = len(h["board"])
                you_now = PK_NAMES[pk_eval(h["hole"]["you"] + h["board"])[0]] if h["board"] else ""
                bot_now = PK_NAMES[h["showdown"][1][0]] if shown else ""
                board = pk_cards(h["board"], new_from=new_board) + '<span class="pk-slot"></span>' * (5 - len(h["board"]))
                st.html(f'<div class="bj-who">{them} · {h["stack"]["bot"]:,} chips{f" <b>{bot_now}</b>" if bot_now else ""}'
                        f'{" · button" if h["button"] == "bot" else ""}</div>'
                        f'<div class="pk-row"><div class="bj-hand">{pk_cards(h["hole"]["bot"], hidden=not shown, new_from=0 if fresh else None, flip=shown and h.get("flip_shown") is None)}</div>'
                        f'<div class="pq-circle" style="width:78px;height:78px">{chip_stack_html(h["bet"]["bot"], empty="")}</div></div>'
                        f'<div class="pk-pot">POT {h["pot"]:,}</div><div class="bj-hand">{board}</div>'
                        f'<div class="pk-log">{html.escape(" · ".join(h["log"][-5:]))}</div>'
                        f'<div class="pk-row"><div class="bj-hand">{pk_cards(h["hole"]["you"], new_from=0 if fresh else None)}</div>'
                        f'<div class="pq-circle" style="width:78px;height:78px">{chip_stack_html(h["bet"]["you"], empty="")}</div></div>'
                        f'<div class="bj-who">You · {h["stack"]["you"]:,} chips{f" <b>{you_now}</b>" if you_now else ""}'
                        f'{" · button" if h["button"] == "you" else ""}</div>'
                        f'<div class="bj-msg">{html.escape(g["msg"]) or ("Your move." if your_turn else "")}</div>')
                if shown:
                    h["flip_shown"] = True
            else:
                st.html(f'<div class="bj-msg" style="padding:40px 0">Heads-up Texas Hold\'em against {them}. Blinds '
                        f'5 / 10, 1,000 chips deep (100 big blinds) - deal a hand to start.</div>')
            if your_turn:
                to_call = pk_to_call(h, "you")
                cols = st.columns(5)
                cols[0].button("Fold", key="pk_fold", on_click=pk_you, args=("fold",), width="stretch",
                               disabled=not to_call, help="Nothing to call - check instead" if not to_call else None)
                cols[1].button(pk_move_name(h, "call", None), key="pk_call", on_click=pk_you, args=("call",),
                               width="stretch")
                for col, (label, frac) in zip(cols[2:], (("Half pot", 0.5), ("Pot", 1.0), ("All-in", None))):
                    col.button(pk_move_name(h, "raise", frac).replace(" (all-in)", ""), key=f"pk_raise_{label}",
                               on_click=pk_you, args=("raise", frac), width="stretch",
                               type="primary" if frac == 1.0 else "secondary", help=label,
                               disabled=not pk_can_raise(h, "you"))
            elif not live:
                if wallet["chips"] >= PK_BB:
                    st.button("Deal a hand", key="pk_deal", on_click=pk_deal, type="primary", width="stretch",
                              icon=":material/playing_cards:")
                else:
                    st.button("Out of chips - get 1,000 more", on_click=bj_reset, key="pk_reset",
                              icon=":material/refresh:", width="stretch", type="primary")
        with st.expander("How Texas Hold'em works", icon=":material/school:"):
            st.markdown(
                "**Goal:** win the chips in the pot - with the best 5-card hand at the end, or by making the other "
                "player fold.\n\n**A hand:** you and Ace get 2 private cards. Blinds go in (5 and 10, swapping each "
                "hand). Then betting rounds: **before the flop**, after **the flop** (3 shared cards), **the turn** "
                "(a 4th) and **the river** (a 5th). On your turn: **fold**, **check / call**, or **bet / raise**.\n\n"
                "**Hands, best first:** straight flush, four of a kind, full house, flush, straight, three of a "
                "kind, two pair, pair, high card. You use any 5 of your 2 cards + the 5 on the board.\n\n"
                "**The key idea - pot odds:** if calling costs 20 to win a pot of 80, you're paying 20 for a shot at "
                "100, so you need to win **20%** of the time. If your chance is better than that, calling makes money "
                "over time. That's exactly what the advisor checks. Unlike the casino games, there's **no house "
                "edge** here - you can beat Ace by playing better. Fake chips - just for fun.")
        strategy_lab("pk")

    with side:
        won, lost, pushed = g["record"]
        st.metric("Chips", f"{wallet['chips']:,}",
                  f"{wallet['chips'] - 1000:+,} since you started" if wallet["chips"] != 1000 else None, border=True,
                  help="One chip balance for every game")
        st.metric("Hands (W-L-T)", f"{won}-{lost}-{pushed}", border=True)
        with st.container(border=True):
            st.markdown("**:material/psychology: Poker AI**")
            if has_cfr:
                st.segmented_control("Coach", ["Your CFR bot", "Equity advisor"], default="Your CFR bot",
                                     key="pk_coach", width="stretch",
                                     help="Who advises you and grades you in Trainer mode: your trained bot (its "
                                          "strategy for your exact spot) or the equity + pot odds advisor")
            trainer_switch("pk", "made your move")
            mix = pk_cfr_strategy(h) if your_turn and pk_coach_is_cfr() else None
            if your_turn and trainer_on("pk"):
                trainer_quiet()
            elif mix:   # your bot's strategy for YOUR seat right now - it only knows your cards and the betting
                best = mix[0]
                bars = "".join(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px">'
                    f'<span style="width:118px;color:{"#FFF" if m is best else "#C7C7C7"}">{html.escape(name)}</span>'
                    f'<span style="flex:1;height:8px;background:#161616;border-radius:4px;overflow:hidden">'
                    f'<span style="display:block;height:100%;width:{p * 100:.0f}%;'
                    f'background:{"#22C55E" if m is best else "#3A3A3A"}"></span></span>'
                    f'<span style="width:34px;text-align:right">{p:.0%}</span></div>'
                    for m in mix if (name := m[1]) and (p := m[2]) >= 0.01)
                ai_box(best[1], f"What <b>your CFR bot</b> plays in your seat here - a mix, so you're hard to read:"
                                f"{bars}")
                st.button(f"Play its move ({best[1]})", key="pk_play_cfr", on_click=pk_you, args=best[3],
                          icon=":material/smart_toy:", width="stretch")
            elif your_turn:
                if pk_coach_is_cfr():
                    st.caption("This spot is off your bot's training tree (an unusual bet size or short stack) - "
                               "showing the equity advisor instead.")
                adv = pk_advice(h)
                vs = "any two cards" if adv["top"] == 1 else f"the best {adv['top']:.0%} of hands (Ace has raised)"
                why = (f"Your chance to win: <b>{adv['equity']:.0%}</b> against {vs} - from dealing out the unknown "
                       f"cards 700 times. ")
                if adv["to_call"]:
                    why += (f"Calling costs {adv['to_call']:,} to win {h['pot']:,}: you need <b>{adv['odds']:.0%}</b>. "
                            + ("You're ahead of the price." if adv["equity"] >= adv["odds"] else "You're short of it."))
                else:
                    why += "Nothing to call. " + ("Strong enough to bet for value." if adv["action"] == "raise"
                                                  else "Not strong enough to bet - check for free.")
                ai_box(pk_move_name(h, adv["action"], adv["frac"]), why)
            else:
                ai_box("Deal when ready" if not live else f"{them} is thinking...",
                       ("On your turn your CFR bot shows what it would play in your seat." if pk_coach_is_cfr() else
                        "On your turn the AI works out your chance to win and the price of calling, then names the "
                        "move.") + f" It never sees {them}'s cards - and {them} never sees yours.")
            if trainer_on("pk"):
                trainer_panel("pk")
            ai_rate_caption(g["ai_record"], "hands", "making every one of your moves itself (the coach you picked)")
            if g["followed"][1]:
                st.caption(f"Hands where you did what it said every time: you won {g['followed'][0]} of "
                           f"{g['followed'][1]}.")
            st.caption("Poker has no house edge, so better play really wins. Ace uses equity + pot odds with a little "
                       "randomness and the odd bluff; your CFR bot plays the strategy it learned by self-play.")


# ---------------------------------------------------------------- styles for the custom cards (one place)
st.html("""
<style>
  .pq-header { display:flex; justify-content:space-between; align-items:flex-end; margin:4px 0 2px; }
  .pq-title  { font-size:28px; font-weight:700; letter-spacing:-.5px; }
  .pq-sub    { color:#94A3B8; font-size:14px; margin-top:2px; }
  .pq-date   { color:#94A3B8; font-size:14px; }
  .muted     { color:#94A3B8; font-weight:400; }

  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .chip  { font-size:12px; font-weight:500; padding:3px 9px; border-radius:999px;
           background:#1F2937; color:#CBD5E1; white-space:nowrap; }
  .chip.blue  { background:rgba(59,130,246,.15); color:#93C5FD; font-weight:600; }
  .chip.green { background:rgba(34,197,94,.15);  color:#86EFAC; font-weight:600; }

  .bet       { display:flex; gap:14px; align-items:flex-start; }
  .bet img   { width:44px; height:44px; flex-shrink:0; }
  .bet-title { font-size:16px; font-weight:600; }
  .bet-sub   { color:#94A3B8; font-size:13px; margin-top:2px; }
  .leg       { display:flex; align-items:center; gap:10px; padding:6px 0; border-top:1px solid #1F2937; }
  .leg img   { width:24px; height:24px; }

  .fo-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:12px; }
  .fo-card { background:#121826; border:1px solid #1F2937; border-radius:10px; padding:12px 14px; }
  .fo-row  { display:flex; align-items:center; gap:10px; font-weight:600; }
  .fo-row img { width:30px; height:30px; }
  .fo-row .pct { margin-left:auto; }
  .fo-row.fav .pct { color:#22C55E; }
  .fo-bar  { display:flex; height:6px; border-radius:3px; overflow:hidden; margin:8px 0; background:#1F2937; }
  .fo-goalies { color:#94A3B8; font-size:12px; margin-top:6px; }

  .sb-game  { display:flex; align-items:center; gap:8px; padding:7px 0; border-bottom:1px solid #1F2937;
              font-size:13px; font-weight:600; }
  .sb-team  { display:flex; align-items:center; gap:5px; }
  .sb-team img { width:20px; height:20px; }
  .sb-team .sc { color:#94A3B8; min-width:10px; }
  .sb-team .sc.up { color:#22C55E; }
  .sb-at    { color:#475569; font-weight:400; }
  .sb-time  { margin-left:auto; color:#94A3B8; font-weight:500; font-size:12px; white-space:nowrap; }
  .sb-time.live { color:#EF4444; }
</style>
""")

# ---- Parlay Quant 2.0: the black + green look (the default). dashboard_classic.py turns it off for the old blue look ----
V2 = globals().get("PQ_THEME", "v2") != "classic"
if V2:
    st.html("""
<style>
  /* page + sidebar: true black, thin borders */
  .stApp, [data-testid="stAppViewContainer"] { background:#000 !important; }
  [data-testid="stHeader"] { background:rgba(0,0,0,.85) !important; backdrop-filter:blur(8px); }
  [data-testid="stDecoration"] { display:none; }
  [data-testid="stSidebar"] { background:#050505 !important; border-right:1px solid #161616 !important; }
  [data-testid="stSidebar"] hr, [data-testid="stMain"] hr { border-color:#161616 !important; }
  .block-container { padding-top:4.2rem !important; max-width:1400px; }

  /* header */
  .pq-title { font-size:30px !important; font-weight:800 !important; letter-spacing:-1px !important;
              background:linear-gradient(90deg,#FFFFFF 0%,#86EFAC 100%); -webkit-background-clip:text;
              background-clip:text; color:transparent; }
  .pq-sub, .pq-date { color:#7A7A7A !important; }
  .muted { color:#7A7A7A !important; }

  /* tabs: pill style */
  [data-baseweb="tab-list"] { gap:6px !important; border-bottom:1px solid #161616 !important; padding-bottom:8px; }
  [data-baseweb="tab"] { background:#0B0B0B !important; border:1px solid #1C1C1C !important; border-radius:999px !important;
                         padding:6px 16px !important; height:auto !important; color:#A3A3A3 !important;
                         transition:all .15s ease; }
  [data-baseweb="tab"]:hover { border-color:#2E2E2E !important; color:#FFF !important; }
  [data-baseweb="tab"][aria-selected="true"] { background:#FFFFFF !important; color:#000 !important;
                                               border-color:#FFFFFF !important; font-weight:600; }
  [data-baseweb="tab"][aria-selected="true"] * { color:#000 !important; }
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; }

  /* cards (bordered containers), metrics, expanders */
  [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details {
      background:#0A0A0A !important; border:1px solid #1C1C1C !important; border-radius:14px !important;
      transition:border-color .15s ease; }
  [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:#2A2A2A !important; }
  [data-testid="stMetric"] { background:#0A0A0A; border-color:#1C1C1C !important; border-radius:14px !important;
                             padding:14px 16px !important; }
  [data-testid="stMetricLabel"] p { text-transform:uppercase; letter-spacing:.08em; font-size:11px !important;
                                    color:#7A7A7A !important; }
  [data-testid="stMetricValue"] { font-size:28px !important; font-weight:700 !important;
                                  font-variant-numeric:tabular-nums; letter-spacing:-.5px; }

  /* buttons */
  .stButton button, [data-testid="stPopover"] button { border-radius:10px !important; border-color:#262626 !important;
                                                       background:#0E0E0E !important; transition:all .15s ease; }
  .stButton button:hover, [data-testid="stPopover"] button:hover { border-color:#22C55E !important; }
  .stButton button[kind="primary"] { background:#22C55E !important; border-color:#22C55E !important; color:#000 !important;
                                     font-weight:600; }
  .stButton button[kind="primary"]:hover { background:#4ADE80 !important; }

  /* the custom bet / face-off cards and chips */
  .chip { background:#141414 !important; color:#C7C7C7 !important; border:1px solid #1F1F1F; }
  .chip.blue  { background:#FFFFFF !important; color:#000000 !important; border-color:#FFFFFF; font-weight:700; }
  .chip.green { background:rgba(34,197,94,.10) !important; color:#86EFAC !important; border-color:rgba(34,197,94,.25); }
  .bet-title { font-size:16px; letter-spacing:-.2px; }
  .leg { border-top-color:#161616 !important; }
  .fo-card { background:#0A0A0A !important; border-color:#1C1C1C !important; border-radius:14px !important; }
  .fo-bar  { background:#161616 !important; }
  .fo-bar div:last-child { background:#22C55E !important; }   /* home side of the win-chance bar */
  .fo-bar div:first-child { background:#3A3A3A !important; }
  .sb-game { border-bottom-color:#141414 !important; }

  /* live score ticker */
  .ticker { background:#050505 !important; border-color:#1C1C1C !important; border-radius:14px !important; }

  /* numbers line up in tables and scores */
  .score, .sc, .pct, [data-testid="stMetricDelta"] { font-variant-numeric:tabular-nums; }
</style>
""")


# ---------------------------------------------------------------- live data (refreshed every 30 seconds)
@st.cache_data(ttl=25, show_spinner=False)
def scoreboard(date):
    return get_scoreboard(date)


@st.cache_data(ttl=25, show_spinner=False)
def boxscore(game_id):
    return get_boxscore(game_id)


def ticker_html(games):
    """a stock-market style strip that scrolls tonight's scores sideways"""
    items = ""
    for g in games:
        away, home = g["awayTeam"], g["homeTeam"]
        live = g["gameState"] in ("LIVE", "CRIT")
        started = g["gameState"] not in ("FUT", "PRE")

        def side(team, other):
            score = team.get("score", "") if started else ""
            leading = started and team.get("score", 0) > other.get("score", 0)
            return (f'<img src="{team.get("logo_url") or logo(team["abbrev"])}"><span class="abbr">{team["abbrev"]}</span>'
                    f'<span class="score {"up" if leading else ""}">{score}</span>')

        status = g.get("status_text") or game_status(g)
        items += (f'<span class="game">{side(away, home)}<span class="at">@</span>{side(home, away)}'
                  f'<span class="state {"live" if live else ""}">{"LIVE · " if live else ""}{status}</span></span>')
    seconds = max(20, 6 * len(games))   # more games -> scroll a bit slower
    return f"""
<style>
  .ticker {{ background:#121826; border:1px solid #1F2937; border-radius:10px; overflow:hidden; white-space:nowrap; }}
  .track {{ display:inline-block; padding:9px 0; animation: scroll {seconds}s linear infinite; }}
  .ticker:hover .track {{ animation-play-state: paused; }}
  @keyframes scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
  .game {{ display:inline-flex; align-items:center; gap:7px; margin:0 26px; color:#E5E7EB; font-weight:600; font-size:14px; }}
  .game img {{ height:24px; width:24px; }}
  .score {{ min-width:12px; color:#94A3B8; }}
  .score.up {{ color:#22C55E; }}
  .at {{ color:#475569; margin:0 3px; font-weight:400; }}
  .state {{ color:#94A3B8; font-weight:500; font-size:12px; margin-left:6px; }}
  .state.live {{ color:#EF4444; }}
</style>
<div class="ticker"><div class="track">{items}{items}</div></div>"""   # items twice -> seamless loop


@st.cache_data(ttl=25, show_spinner=False)
def nfl_scoreboard(date=None, league="nfl"):
    return get_nfl_scoreboard(date, league)


@st.cache_data(ttl=25, show_spinner=False)
def nfl_box(game_id, league="nfl"):
    return get_nfl_box(game_id, league)


def nfl_as_nhl(g):
    """turn an ESPN NFL game into the same shape as an NHL game, so the ticker and sidebar can show both"""
    return {"awayTeam": g["away"], "homeTeam": g["home"], "status_text": g["status_text"],
            "gameState": {"FUT": "FUT", "LIVE": "LIVE", "FINAL": "FINAL", "VOID": "FINAL"}[g["state"]]}


def games_for(sport, date=None):
    """NHL / NBA: one day's games. NFL / college: this week's games (or one day's)."""
    if sport in ("NFL", "NCAAF"):
        return [nfl_as_nhl(g) for g in nfl_scoreboard(date, "nfl" if sport == "NFL" else "cfb")]
    if sport in ("NBA", "WNBA", "MLB"):
        return [nfl_as_nhl(g) for g in nfl_scoreboard(date or TODAY, sport.lower())]
    return scoreboard(date or TODAY)


@st.fragment(run_every=30)
def live_ticker(sport):
    try:
        games = games_for(sport)
    except Exception:
        st.caption("Live scores unavailable right now.")
        return
    if games:
        st.html(ticker_html(games))
    else:
        st.caption(f"No {sport} games {'today' if sport in ('NHL', 'NBA', 'WNBA', 'MLB') else 'this week'}.")


def settled_only(bets):
    return bets[bets["status"].isin(SETTLED)]


def pnl_chart(settled):
    """green/red bar = that day's profit or loss, blue line = running total"""
    daily = settled.groupby("game_date")["profit"].sum().reset_index()
    daily["running"] = daily["profit"].cumsum()
    daily["result"] = ["Profit" if p >= 0 else "Loss" for p in daily["profit"]]
    x = alt.X("game_date:N", title="date", axis=alt.Axis(labelAngle=0))
    bars = alt.Chart(daily).mark_bar(cornerRadius=3).encode(
        x=x, y=alt.Y("profit:Q", title="CA$"),
        color=alt.Color("result:N", scale=alt.Scale(domain=["Profit", "Loss"], range=[GREEN, RED]), legend=None),
        tooltip=[alt.Tooltip("game_date:N", title="date"), alt.Tooltip("profit:Q", title="that day", format="+.2f"),
                 alt.Tooltip("running:Q", title="running total", format="+.2f")])
    line = alt.Chart(daily).mark_line(color=BLUE, strokeWidth=2.5, point=True).encode(
        x=x, y="running:Q", tooltip=[alt.Tooltip("running:Q", title="running total", format="+.2f")])
    return (bars + line).properties(height=300)


def pnl_section(title, bets):
    """one graph + the numbers under it (used inside the pop-up)"""
    settled = settled_only(bets)
    st.markdown(f"**{title}**")
    if settled.empty:
        open_count = (bets["status"] == "open").sum() if len(bets) else 0
        st.info(f"No graded bets yet ({open_count} still open). After the games finish, click "
                "**Grade finished bets** and the graph fills in.", icon=":material/hourglass_top:")
        return
    st.altair_chart(pnl_chart(settled), width="stretch")
    daily = settled.groupby("game_date")["profit"].sum()
    staked = settled["stake"].sum()
    a, b, c, d = st.columns(4)
    a.metric("Total", cad(settled["profit"].sum()))
    b.metric("ROI", f"{settled['profit'].sum() / staked:+.1%}" if staked else "-")
    c.metric("Best day", cad(daily.max()), daily.idxmax(), delta_color="off", delta_arrow="off")
    d.metric("Worst day", cad(daily.min()), daily.idxmin(), delta_color="off", delta_arrow="off")


@st.dialog("Profit / loss", width="large")
def pnl_popup(sections):
    """the pop-up: sections = list of (title, bets table)"""
    st.caption("Bars: each day's profit (green) or loss (red). Line: your running total. Hover for exact numbers.")
    for i, (title, bets) in enumerate(sections):
        if i:
            st.divider()
        pnl_section(title, bets)


def status_color(status):
    """colour for a live status word in the table"""
    color = {"Hit": GREEN, "On track": GREEN, "Toss-up": AMBER, "In trouble": RED, "Lost": RED}.get(status, MUTED)
    return f"color: {color}; font-weight: 600"


def nfl_game_and_box(date, team, league="nfl"):
    """(game, live player stats) for an NFL or college team on a date - used by the live tracker for football legs"""
    game = next((g for g in nfl_scoreboard(date, league) if team in (g["home"]["abbrev"], g["away"]["abbrev"])), None)
    box = nfl_box(game["id"], league) if game and game["state"] in ("LIVE", "FINAL") else None
    return game, box


@st.fragment(run_every=30)
def live_bets():
    bets = load_bets()
    open_bets = bets[bets["status"] == "open"]
    st.caption(f"Updates every 30 seconds · last update {pd.Timestamp.now().strftime('%I:%M:%S %p')}")
    if open_bets.empty:
        st.info("No open bets. Bets you mark as placed show up here while their games are on.", icon=":material/info:")
        return

    for _, bet in open_bets.iterrows():
        games_by_team = {}
        for g in scoreboard(bet["game_date"]):
            games_by_team[g["homeTeam"]["abbrev"]] = g
            games_by_team[g["awayTeam"]["abbrev"]] = g
        legs = json.loads(bet["legs"])
        boxscores = {}
        for leg in legs:
            g = games_by_team.get(leg["team"])
            if g and g["gameState"] not in ("FUT", "PRE"):
                boxscores[g["id"]] = boxscore(g["id"])
        live = live_bet(bet, games_by_team, boxscores, nfl_lookup=nfl_game_and_box)

        with st.container(border=True):
            logos = "".join(f'<img src="{logo(leg["team"], leg.get("sport", "nhl"))}" style="width:30px;height:30px">'
                            for leg in legs)
            st.html(f'<div class="bet"><div style="display:flex;gap:4px">{logos}</div><div>'
                    f'<div class="bet-title">{html.escape(bet["description"])}</div>'
                    f'<div class="chips"><span class="chip blue">{bet["book"].upper()} {american(bet["price"])}</span>'
                    f'<span class="chip">{bet["type"]}</span>'
                    f'<span class="chip">{cad(bet["stake"])} to win {cad(bet["stake"] * (bet["price"] - 1))}</span>'
                    f'</div></div></div>')
            table = pd.DataFrame([{"Player / team": l["leg"]["player"], "Bet": l["leg"]["bet"], "Right now": l["now"],
                                   "Chance": f"{l['chance']:.0%}" if l["chance"] is not None else "-",
                                   "Status": l["status"]} for l in live["legs"]])
            st.dataframe(table.style.map(status_color, subset=["Status"]), hide_index=True, width="stretch")

            chance, fair = live["chance"], live["fair_value"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Chance to win now", f"{chance:.0%}" if chance is not None else "-")
            m2.metric("Fair value now", cad(fair) if fair is not None else "-",
                      help="stake × odds × chance to win now - what the bet is really worth this minute")
            m3.metric("Pays if it hits", cad(bet["stake"] * bet["price"]))

            # took the book's cash-out offer? record what they paid
            cash_col, cancel_col, _ = st.columns([1, 1, 3])
            with cash_col.popover("I cashed out", icon=":material/payments:", width="stretch"):
                amount = st.number_input("The book paid me (CA$)", min_value=0.0, step=1.0,
                                         value=round(float(fair), 2) if fair else 0.0, key=f"cash_amt_{bet['id']}",
                                         help="Starts at the fair value - change it to what your book actually paid")
                if amount and amount < bet["stake"]:
                    st.caption(f"That's a loss of {md_cad(bet['stake'] - amount)} on this bet.")
                elif amount:
                    st.caption(f"That's a profit of {md_cad(amount - bet['stake'])} on this bet.")
                if st.button("Save cash-out", key=f"cash_{bet['id']}", type="primary", icon=":material/payments:",
                             disabled=amount <= 0):
                    cash_out_bet(bet["id"], amount)
                    st.rerun(scope="app")

            # placed it by mistake? remove it (asks to confirm first)
            with cancel_col.popover("Cancel bet", icon=":material/delete:", width="stretch"):
                st.markdown("Remove this bet from your dashboard?")
                st.caption("This only removes it from your tracking here - it doesn't cancel it at the sportsbook.")
                if st.button("Yes, remove it", key=f"cancel_{bet['id']}", type="primary", icon=":material/delete:"):
                    delete_bet(bet["id"])
                    st.rerun(scope="app")   # refresh the whole page, not just this live section

            # should I cash out?
            if chance is None:
                st.info("Every player in this bet is out - it should be refunded.", icon=":material/undo:")
            elif chance >= 0.999:
                st.success("It's hit - nothing to do, it settles after the game.", icon=":material/check_circle:")
            elif chance <= 0.001:
                st.error("This one's lost.", icon=":material/cancel:")
            else:
                st.warning(f"**Cash out only if your book offers more than {md_cad(fair)}.** If the offer is lower, "
                           f"keep the bet - books usually offer less than fair value, so most of the time: hold.",
                           icon=":material/account_balance:")


def place_bet_box(container, key, legs, book, suggested_price, suggested_stake, game_date=None):
    """odds + stake boxes that show the payout as you type, and the 'placed it' button
    (game_date: for logging a bet late - otherwise it's the earliest leg's day, or today)"""
    odds_got = container.number_input("Odds you got (American)", value=float(american(suggested_price)),
                                      step=5.0, format="%.0f", key=f"odds_{key}")   # whole numbers: 118 / -120
    stake = container.number_input("Stake (CA$)", value=float(round(suggested_stake, 2)), min_value=0.0,
                                   step=1.0, key=f"stake_{key}",
                                   help=f"Suggested: {cad(suggested_stake)} - a quarter of the Kelly formula, capped "
                                        "so no single bet risks much of your bankroll")
    if suggested_stake <= 0:
        container.caption("The model suggests **no stake** here - it doesn't see enough value. Type an amount only "
                          "if you're placing it anyway (a bonus / free bet is the right use for these).")
    if -100 < odds_got < 100:   # American odds are always +100 or more, or -100 or less
        container.error("American odds must be +100 or higher, or -100 or lower (like +150 or -120).",
                        icon=":material/error:")
        return
    price = american_to_decimal(odds_got)
    # decimal odds already include your stake: CA$10 at 2.50 -> CA$25 back = CA$10 stake + CA$15 profit
    container.markdown(f"To win **{md_cad(stake * (price - 1))}** · total back **{md_cad(stake * price)}**")
    if container.button("I placed this", key=f"place_{key}", icon=":material/check_circle:",
                        disabled=stake <= 0, width="stretch"):
        # NFL games aren't always today; late-logged bets bring their own date
        game_day = game_date or min((leg.get("game_date") or TODAY) for leg in legs)
        new_id = add_bet(legs, book, price, stake, game_day)
        # remember it, so an "Undo" banner can appear at the top of the page
        description = " + ".join(f"{leg['player']} {leg['bet']}" for leg in legs)
        st.session_state.last_placed = {"id": new_id, "text": f"{description} · {cad(stake)} at {book}"}
        st.toast(f"Saved: {md_cad(stake)} to win {md_cad(stake * (price - 1))}", icon=":material/check_circle:")
        return True
    return False


def bet_card_html(team, title, subtitle, chips, sport="nhl", player=None):
    """player photo (or team logo) + bet name + small info chips"""
    chip_html = "".join(f'<span class="chip {kind}">{html.escape(text)}</span>' for kind, text in chips)
    return (f'<div class="bet">{face(player, team, sport, 52)}<div><div class="bet-title">{title}</div>'
            f'<div class="bet-sub">{html.escape(subtitle)}</div><div class="chips">{chip_html}</div></div></div>')


# ---------------------------------------------------------------- page
# ---- which sport the page shows (live tracking, profit/loss and fumbles always cover both) ----
# ---- at the very top of the sidebar: the betting dashboard, or a full-screen game of blackjack ----
VIEW = st.sidebar.segmented_control("View", [":material/insights: Bets", ":material/playing_cards: Card games"],
                                    default=":material/insights: Bets", key="view_mode", width="stretch") or "Bets"
BLACKJACK = "Card games" in VIEW
SPORT = st.sidebar.segmented_control("Sport", ["NHL", "NBA", "WNBA", "MLB", "NFL", "NCAAF"], default="NHL", key="sport",
                                     width="stretch") or "NHL"
CFG = {
    "NHL": {"bets": "bets_today.csv", "parlays": "parlays_today.csv", "lines": "lines_today.csv",
            "faceoffs": "faceoffs_today.csv", "days": 0, "when": "tonight", "default_type": "shots",
            "steps": [("NHL_Scraper.py", "Updating NHL stats"), ("Odds_Scraper.py", "Getting odds"),
                      ("ML.py", "Running the model")],
            "updated": [("skaters.csv", "NHL stats", "query_stats"), ("odds.csv", "Odds", "request_quote"),
                        ("bets_today.csv", "Bets", "insights")],
            "sub": "NHL player props, moneylines and parlays", "note": "Check lineups and starting goalies"},
    "NBA": {"bets": "nba_bets_today.csv", "parlays": "nba_parlays_today.csv", "lines": "nba_lines_today.csv",
            "faceoffs": "nba_faceoffs_today.csv", "days": 0, "when": "tonight", "default_type": "points",
            "steps": [("NBA_Scraper.py", "Updating NBA stats"), ("NBA_Odds_Scraper.py", "Getting NBA odds"),
                      ("NBA_ML.py", "Running the NBA model")],
            "updated": [("nba_players.csv", "NBA stats", "query_stats"), ("nba_odds.csv", "Odds", "request_quote"),
                        ("nba_bets_today.csv", "Bets", "insights")],
            "sub": "NBA player props, moneylines and parlays",
            "note": "Check the injury report - stars get rested, especially on back-to-backs"},
    "WNBA": {"bets": "wnba_bets_today.csv", "parlays": "wnba_parlays_today.csv", "lines": "wnba_lines_today.csv",
             "faceoffs": "wnba_faceoffs_today.csv", "days": 0, "when": "tonight", "default_type": "points",
             "steps": [("WNBA_Scraper.py", "Updating WNBA stats"), ("WNBA_Odds_Scraper.py", "Getting WNBA odds"),
                       ("WNBA_ML.py", "Running the WNBA model")],
             "updated": [("wnba_players.csv", "WNBA stats", "query_stats"), ("wnba_odds.csv", "Odds", "request_quote"),
                         ("wnba_bets_today.csv", "Bets", "insights")],
             "sub": "WNBA player props, moneylines and parlays",
             "note": "Check the injury report before betting"},
    "MLB": {"bets": "mlb_bets_today.csv", "parlays": "mlb_parlays_today.csv", "lines": "mlb_lines_today.csv",
            "faceoffs": "mlb_faceoffs_today.csv", "days": 0, "when": "today", "default_type": "strikeouts",
            "steps": [("MLB_Scraper.py", "Updating MLB stats"), ("MLB_Odds_Scraper.py", "Getting MLB odds"),
                      ("MLB_ML.py", "Running the MLB model")],
            "updated": [("mlb_players.csv", "MLB stats", "query_stats"), ("mlb_odds.csv", "Odds", "request_quote"),
                        ("mlb_bets_today.csv", "Bets", "insights")],
            "sub": "MLB player props, moneylines and parlays",
            "note": "Check the posted lineups and starting pitchers - a batter who sits is a refund, "
                    "a pitcher swap changes everything"},
    "NFL": {"bets": "nfl_bets_today.csv", "parlays": "nfl_parlays_today.csv", "lines": "nfl_lines_today.csv",
            "faceoffs": "nfl_faceoffs_today.csv", "days": 6, "when": "this week", "default_type": "receiving_yards",
            "steps": [("NFL_Scraper.py", "Updating NFL stats"), ("NFL_Odds_Scraper.py", "Getting NFL odds"),
                      ("NFL_ML.py", "Running the NFL model")],
            "updated": [("nfl_players.csv", "NFL stats", "query_stats"), ("nfl_odds.csv", "Odds", "request_quote"),
                        ("nfl_bets_today.csv", "Bets", "insights")],
            "sub": "NFL player props, moneylines and parlays", "note": "Check injury reports and inactives"},
    "NCAAF": {"bets": "cfb_bets_today.csv", "parlays": "cfb_parlays_today.csv", "lines": "cfb_lines_today.csv",
              "faceoffs": "cfb_faceoffs_today.csv", "days": 6, "when": "this week", "default_type": "receiving_yards",
              "steps": [("CFB_Scraper.py", "Updating college stats"), ("CFB_Odds_Scraper.py", "Getting college odds"),
                        ("CFB_ML.py", "Running the college model")],
              "updated": [("cfb_players.csv", "College stats", "query_stats"), ("cfb_odds.csv", "Odds", "request_quote"),
                          ("cfb_bets_today.csv", "Bets", "insights")],
              "sub": "College football player props, moneylines and parlays",
              "note": "Check injury reports and depth charts"},
}[SPORT]
SPORT_KEY = {"NHL": "nhl", "NBA": "nba", "WNBA": "wnba", "MLB": "mlb", "NFL": "nfl", "NCAAF": "cfb"}[SPORT]

# sports with a "This week" option (football already gets several days of odds) -> Week_Run.py's sport name
WEEK_SPORTS = {"NHL": "nhl", "NBA": "nba", "WNBA": "wnba", "MLB": "mlb"}
ODDS_SETUP = {"NHL": ("icehockey_nhl", "Odds_Scraper.py", "NHL_ODDS_API_KEY"),
              "NBA": ("basketball_nba", "NBA_Odds_Scraper.py", "NBA_ODDS_API_KEY"),
              "WNBA": ("basketball_wnba", "WNBA_Odds_Scraper.py", "WNBA_ODDS_API_KEY"),
              "MLB": ("baseball_mlb", "MLB_Odds_Scraper.py", "MLB_ODDS_API_KEY")}


@st.cache_data(ttl=600, show_spinner=False)
def week_cost(sport):
    """(games in the next 7 days, about how many credits their odds cost) - from the odds service's FREE game list"""
    api_sport, script, key_name = ODDS_SETUP[sport]
    key = os.getenv(key_name) or os.getenv("ODDS_API_KEY")
    events = requests.get(f"https://api.the-odds-api.com/v4/sports/{api_sport}/events", params={"apiKey": key},
                          timeout=15).json()
    now = pd.Timestamp.now(tz="America/Toronto")
    last = (now + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    days = [d for d in (pd.Timestamp(e["commence_time"]).tz_convert("America/Toronto") for e in events)
            if d > now and d.strftime("%Y-%m-%d") <= last]
    if sport == "NHL":
        return len(days), 5 * len(days)   # every game, ~5 credits each
    # the others: props for the MAX_PROP_GAMES closest games of each day (~4 credits each) + 1 for all moneylines
    cap = int(re.search(r"^MAX_PROP_GAMES = (\d+)", open(script, encoding="utf-8").read(), re.M).group(1))
    per_day = pd.Series([d.strftime("%Y-%m-%d") for d in days]).value_counts()
    return len(days), 1 + 4 * int(per_day.clip(upper=cap).sum())

live_ticker(SPORT)   # the scrolling score strip, always at the top
badge = ('<span style="font-size:12px;font-weight:700;color:#000;background:#22C55E;border-radius:999px;'
         'padding:3px 9px;margin-left:10px;vertical-align:middle;letter-spacing:0">2.0</span>') if V2 else ""
# the header: same font and white-to-green title in both views - "Parlay Quant" for bets, "Card Room" for card games
title, sub = ("Card Room", "Casino games · fake chips, just for fun") if BLACKJACK \
    else ("Parlay Quant", f'{CFG["sub"]} · Ontario')
st.html(f'<div class="pq-header"><div><div><span class="pq-title">{title}</span>{badge}</div>'
        f'<div class="pq-sub">{sub}</div></div>'
        f'<div class="pq-date">{pd.Timestamp.today().strftime("%A, %B %d")}</div></div>')


def undo_last():
    delete_bet(st.session_state.last_placed["id"])
    st.toast("Removed from your bets.", icon=":material/undo:")
    st.session_state.last_placed = None


def dismiss_last():
    st.session_state.last_placed = None


# "Undo" banner right after you mark a bet as placed - for when you clicked the wrong one.
# The spot is reserved here at the top, but filled in at the very END of the page (see bottom of file),
# because the "I placed this" buttons further down run after this line.
undo_slot = st.container()


def show_undo_banner():
    if not st.session_state.get("last_placed"):
        return
    with undo_slot.container(border=True):
        text_col, undo_col, ok_col = st.columns([6, 1, 1], vertical_alignment="center")
        text_col.markdown(f":material/check_circle: **Saved:** {st.session_state.last_placed['text']}"
                          .replace("$", "\\$"))
        undo_col.button("Undo", icon=":material/undo:", on_click=undo_last, width="stretch")
        ok_col.button("OK", on_click=dismiss_last, width="stretch")

@st.fragment(run_every=60)
def schedule(date, title, sport="NHL"):
    """one day's (NHL) or one week's (NFL) games for the sidebar: logos, abbreviations, score or start time"""
    st.markdown(f"**{title}**")
    try:
        games = games_for(sport, date)
    except Exception:
        st.caption("Schedule unavailable right now.")
        return
    if not games:
        st.caption("No games.")
        return
    rows = ""
    for g in games:
        started = g["gameState"] not in ("FUT", "PRE")
        live = g["gameState"] in ("LIVE", "CRIT")

        def team(t, other):
            score = ""
            if started:   # show the score once the game is on, green for whoever's leading
                up = t.get("score", 0) > other.get("score", 0)
                score = f'<span class="sc {"up" if up else ""}">{t.get("score", 0)}</span>'
            return f'<span class="sb-team"><img src="{t.get("logo_url") or logo(t["abbrev"])}">{t["abbrev"]}{score}</span>'

        rows += (f'<div class="sb-game">{team(g["awayTeam"], g["homeTeam"])}<span class="sb-at">@</span>'
                 f'{team(g["homeTeam"], g["awayTeam"])}'
                 f'<span class="sb-time {"live" if live else ""}">{g.get("status_text") or game_status(g)}</span></div>')
    st.html(rows)


with st.sidebar:
    if SPORT in ("NFL", "NCAAF"):
        schedule(None, "This week", SPORT)
    else:
        schedule(TODAY, "Tonight", SPORT)
        tomorrow = (pd.Timestamp.today() + pd.Timedelta(days=1))
        schedule(tomorrow.strftime("%Y-%m-%d"), f"Tomorrow · {tomorrow.strftime('%a %b %d')}", SPORT)
    st.divider()
    st.caption("Data last updated")
    for path, name, icon in CFG["updated"]:
        when = updated_at(path)
        st.markdown(f":material/{icon}: **{name}** · {when.strftime('%b %d, %I:%M %p') if when else 'never'}")
    st.divider()
    if SPORT == "NCAAF":
        st.caption("College odds: moneylines for every game cost ~1 credit; player props only for the biggest "
                   "games (MAX_PROP_GAMES in CFB_Odds_Scraper.py, ~4 credits each). Run it on Saturday morning.")
    elif SPORT == "NBA":
        st.caption("NBA odds: moneylines for every game tonight cost ~1 credit; player props only for the closest "
                   "games (MAX_PROP_GAMES in NBA_Odds_Scraper.py, ~4 credits each). Run it in the late afternoon.")
    elif SPORT == "WNBA":
        st.caption("WNBA odds: moneylines for every game tonight cost ~1 credit; player props only for the closest "
                   "games (MAX_PROP_GAMES in WNBA_Odds_Scraper.py, ~4 credits each). Run it in the late afternoon.")
    elif SPORT == "MLB":
        st.caption("MLB odds: moneylines for every game today cost ~1 credit; player props only for the closest "
                   "games (MAX_PROP_GAMES in MLB_Odds_Scraper.py, ~4 credits each). Run it after the lineups "
                   "are posted (a few hours before first pitch).")
    elif SPORT == "NFL":
        st.caption("NFL odds use their own key (NFL_ODDS_API_KEY), about 5 credits per game. Get them once, "
                   "on the morning of game day (it grabs the next 4 days of games).")
    else:
        st.caption("Odds cost about 5 API credits per game (500 free a month). Get them once a day, "
                   "a couple of hours before puck drop.")

NHL_TYPES = [("shots", "Shots on goal", "sports_hockey"), ("points", "Points", "scoreboard"),
             ("goals", "Goals", "sports_score"), ("saves", "Goalie saves", "shield"),
             ("moneyline", "Moneylines", "emoji_events")]
NFL_TYPES = [("passing_yards", "Passing yards", "sports_football"), ("rushing_yards", "Rushing yards", "directions_run"),
             ("receiving_yards", "Receiving yards", "trending_up"), ("receptions", "Receptions", "back_hand"),
             ("moneyline", "Moneylines", "emoji_events")]
NBA_TYPES = [("points", "Points", "sports_basketball"), ("rebounds", "Rebounds", "replay"),
             ("assists", "Assists", "handshake"), ("threes", "3-pointers", "target"),
             ("moneyline", "Moneylines", "emoji_events")]
MLB_TYPES = [("strikeouts", "Pitcher strikeouts", "sports_baseball"), ("hits", "Hits", "sports_cricket"),
             ("home_runs", "Home runs", "rocket_launch"), ("rbis", "RBIs", "directions_run"),
             ("moneyline", "Moneylines", "emoji_events")]
# college uses the same bet types as the NFL
BET_TYPES = {"NHL": NHL_TYPES, "NBA": NBA_TYPES, "WNBA": NBA_TYPES, "MLB": MLB_TYPES}.get(SPORT, NFL_TYPES)
TYPE_LABEL = {stat: label for stat, label, _ in NHL_TYPES + NFL_TYPES + NBA_TYPES + MLB_TYPES}


def bankroll():
    """the BANKROLL setting from ML.py, so suggested stakes match the model's"""
    match = re.search(r"^BANKROLL\s*=\s*([\d.]+)", open("ML.py", encoding="utf-8").read(), re.M)
    return float(match.group(1)) if match else 100.0


def suggested_stake(chance, price, cap):
    """a quarter of the Kelly formula, capped at `cap` of your bankroll (0 if the bet isn't worth it)"""
    kelly = (chance * price - 1) / (price - 1) / 4
    return round(max(0.0, min(kelly, cap)) * bankroll(), 2)


def single_card(b, key):
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        when = f' · {pd.Timestamp(b["game_date"]).strftime("%a %b %d")}' if "game_date" in b.index else ""
        left.html(bet_card_html(
            b["team"], f'{html.escape(b["player"])} <span class="muted">· {html.escape(b["bet"])}</span>',
            f'{b["game"]}{when}',
            [("blue", f"{b['book'].upper()} {american(b['price'])}"),
             ("", (f"Model {b['raw_chance']:.0%} · Blend {b['model_chance']:.0%} · Book {b['book_chance']:.0%}"
                   if "raw_chance" in b.index else f"Model {b['model_chance']:.0%} · Book {b['book_chance']:.0%}")),
             ("green", f"{b['ev']:+.0%} expected profit"),
             ("", f"Still bet it down to {american(b['min_price'])}")], b.get("sport", "nhl"),
            player=None if b["stat"] == "moneyline" else b["player"]))
        # NBA: the opponent's elite defenders playing tonight (already priced into the model's chance)
        if isinstance(b.get("tough_d"), str) and b["tough_d"]:
            left.caption(f":material/shield: **Tough D tonight:** {b['tough_d']} - opponents score less with them "
                         f"on the court. The model has already taken this into account.")
        leg = leg_dict(b)
        place_bet_box(right, key, [leg], b["book"], b["price"], b["stake"])


def legs_html(legs):
    """one line per parlay leg: the player's photo (or team logo for moneylines), the bet, the odds
    (and each leg's day, when the legs are on different days - a parlay for the week)"""
    days = {str(leg.get("game_date")) for leg in legs} - {"None", "nan"}
    day = lambda leg: (f'{pd.Timestamp(leg["game_date"]).strftime("%a %b %d")} · '
                       if len(days) > 1 and str(leg.get("game_date")) in days else "")
    return "".join(f'<div class="leg">{face(None if leg.get("stat") == "moneyline" else leg["player"], leg["team"], leg.get("sport", "nhl"), 30)}'
                   f'<div><b>{html.escape(str(leg["player"]))}</b> '
                   f'<span class="muted">· {html.escape(leg["bet"])} · {american(leg["price"])} · '
                   f'{day(leg)}{html.escape(str(leg["game"]))}</span></div></div>' for leg in legs)


# ---- parlay slip (kept in st.session_state so it survives every click) ----
if "slip" not in st.session_state:
    st.session_state.slip = []         # the legs you've added
    st.session_state.slip_book = None  # a parlay has to be at ONE book


def add_leg(leg, book):
    slip = st.session_state.slip
    if slip and st.session_state.slip_book != book:
        st.session_state.slip_msg = f"A parlay has to be at one book - this slip is at {st.session_state.slip_book}."
    elif any(l["player"] == leg["player"] and l["stat"] == leg["stat"] for l in slip):
        st.session_state.slip_msg = f"{leg['player']} already has a {TYPE_LABEL[leg['stat']].lower()} leg in this parlay."
    else:
        slip.append(leg)
        st.session_state.slip_book = book
        st.session_state.slip_msg = None


def remove_leg(i):
    st.session_state.slip.pop(i)
    if not st.session_state.slip:
        st.session_state.slip_book = None


def clear_slip():
    st.session_state.slip = []
    st.session_state.slip_book = None
    st.session_state.slip_msg = None


# ---- mega parlay: 6+ legs mixing every sport (for FanDuel's 6+ leg promos) ----
# every sport's model saves the prices it checked in a "lines" file - this pools them all
MEGA_SOURCES = [("nhl", "lines_today.csv", 0), ("nba", "nba_lines_today.csv", 0), ("wnba", "wnba_lines_today.csv", 0),
                ("mlb", "mlb_lines_today.csv", 0), ("nfl", "nfl_lines_today.csv", 6), ("cfb", "cfb_lines_today.csv", 6)]
SPORT_NAME = {"nhl": "NHL", "nba": "NBA", "wnba": "WNBA", "mlb": "MLB", "nfl": "NFL", "cfb": "NCAAF"}
if "mega_skip" not in st.session_state:
    st.session_state.mega_skip = set()   # legs you swapped out


def all_sport_lines():
    """every sport's lines from its latest model run (skips sports whose model hasn't run recently)"""
    frames = []
    for sport, path, days in MEGA_SOURCES:
        if made_today(path, days):
            lines = pd.read_csv(path)
            lines["sport"] = sport
            if "game_date" not in lines.columns:   # NHL lines are always for the day they were made
                lines["game_date"] = updated_at(path).strftime("%Y-%m-%d")
            frames.append(lines)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def started_games(sport):
    """today's games in a sport that have already started ("PIT @ UTA" labels) - too late to bet those"""
    try:
        games = games_for(SPORT_NAME[sport], TODAY)
    except Exception:
        return set()   # scoreboard down -> don't hide anything
    return {f'{g["awayTeam"]["abbrev"]} @ {g["homeTeam"]["abbrev"]}' for g in games
            if g["gameState"] not in ("FUT", "PRE")}


def leg_key(row):
    return f'{row["sport"]}|{row["player"]}|{row["bet"]}|{row["game"]}'


def mega_parlay():
    st.subheader("Mega parlay (6+ legs, every sport)")
    st.caption("Builds the best big parlay for every sport on its own AND every mix of sports (NHL + NFL, "
               "NBA + MLB + NFL, ...) - made for FanDuel's 6+ leg promos. Each one takes the most likely leg from "
               "each game (one per game, so the legs don't depend on each other), only from legs the model doesn't "
               "think are bad bets. Each sport's model has to have run recently (Tonight tab with that sport "
               "picked) for its legs to show up.")
    lines = all_sport_lines()
    if lines.empty:
        st.info("No model has run recently - run one on the Tonight tab first.", icon=":material/info:")
        return
    lines = lines[lines["game_date"] >= TODAY]   # nothing from past days
    # nothing from games that have already started today either - you can't bet those any more
    is_today = lines["game_date"] == TODAY
    started = {s: started_games(s) for s in lines.loc[is_today, "sport"].unique()}
    already_on = pd.Series([g in started.get(s, set()) for s, g in zip(lines["sport"], lines["game"])],
                           index=lines.index)
    lines = lines[~(is_today & already_on)]
    if lines.empty:
        st.info("Every game the models have lines for has already started. Run a model for upcoming games "
                "(Tonight tab) to build a new one.", icon=":material/info:")
        return

    c1, c2, c3 = st.columns(3)
    books = sorted(lines["book"].unique())
    book = c1.segmented_control("Book", books, key="mega_book",
                                default="fanduel" if "fanduel" in books else books[0])
    n_legs = c2.slider("Legs", 6, 15, 6, key="mega_legs")
    window = c3.segmented_control("Games", ["Today", "Next 3 days", "Next 7 days"], default="Next 3 days",
                                  key="mega_window", help="Today = only today's games. NFL / college games are "
                                  "usually on the weekend, so widen this to include them.")
    c4, c5, c6 = st.columns(3)
    available = [SPORT_NAME[s] for s in SPORT_NAME if s in set(lines["sport"])]
    sports = c4.pills("Sports", available, selection_mode="multi", default=available, key="mega_sports")
    min_chance = c5.slider("Each leg at least", 0.50, 0.85, 0.55, 0.05, key="mega_min", format="%.2f",
                           help="The lowest chance (after blending with the books) any single leg can have")
    shortest = c6.select_slider("Shortest leg allowed", ["No limit", "-500", "-300", "-200", "-150"],
                                value="No limit", key="mega_short",
                                help="Some promos only count legs at certain odds - e.g. pick -300 if every leg "
                                     "has to be -300 or longer. Check the promo's rules in the FanDuel app.")
    days = {"Today": 0, "Next 3 days": 2, "Next 7 days": 6}[window or "Next 3 days"]
    last_day = (pd.Timestamp(TODAY) + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    min_price = 1.0 if shortest == "No limit" else american_to_decimal(float(shortest))

    pool = lines[(lines["book"] == book) & (lines["model_chance"] >= min_chance) & (lines["ev"] >= 0)
                 & (lines["game_date"] <= last_day) & (lines["price"] >= min_price)
                 & lines["sport"].map(SPORT_NAME).isin(sports or [])]
    pool = pool[~pool.apply(leg_key, axis=1).isin(st.session_state.mega_skip)] if len(pool) else pool
    # each game's most likely leg - one leg per game, so the legs don't depend on each other
    # (sport + game: an NHL "BOS @ PHI" and an NBA "BOS @ PHI" are different games)
    best_per_game = pool.sort_values("model_chance", ascending=False).drop_duplicates(["sport", "game"])

    def build(mix):
        """the most likely parlay using ONLY these sports - with at least one leg from each of them"""
        candidates = best_per_game[best_per_game["sport"].isin(mix)]
        if len(mix) > n_legs or len(candidates) < n_legs:
            return None
        seed = candidates.groupby("sport").head(1)   # every sport in the mix gets its best leg first...
        rest = candidates.drop(seed.index).head(n_legs - len(seed))   # ...then the most likely legs left, any sport
        legs_rows = pd.concat([seed, rest])
        price, chance = math.prod(legs_rows["price"]), math.prod(legs_rows["model_chance"])
        return {"mix": mix, "rows": legs_rows, "price": price, "chance": chance, "ev": chance * price - 1}

    # every sport on its own, and every combination of sports (NHL + NFL, NBA + NFL, NHL + NBA + NFL, ...)
    present = [s for s in SPORT_NAME if s in set(best_per_game["sport"])]
    options = [p for k in range(1, len(present) + 1) for mix in itertools.combinations(present, k)
               if (p := build(mix))]
    if not options:
        st.warning(f"Only {len(best_per_game)} game(s) have a leg at {book} that's at least {min_chance:.0%} likely "
                   f"and not a bad bet - a {n_legs}-leg parlay needs {n_legs} different games. Try: more days, "
                   f"more sports, a lower minimum chance, or run the other sports' models.", icon=":material/warning:")
        if st.session_state.mega_skip:
            st.button("Bring back swapped-out legs", on_click=st.session_state.mega_skip.clear,
                      icon=":material/restart_alt:")
        return

    rank = st.segmented_control("Rank by", ["Most likely to hit", "Best expected profit", "Biggest payout"],
                                default="Most likely to hit", key="mega_rank")
    sort_key = {"Most likely to hit": "chance", "Best expected profit": "ev", "Biggest payout": "price"}[
        rank or "Most likely to hit"]
    options.sort(key=lambda p: p[sort_key], reverse=True)
    name = lambda p: " + ".join(SPORT_NAME[s] for s in p["mix"])

    st.markdown(f"**The best {n_legs}-leg parlay for every sport and every mix of sports** "
                f"({len(options)} found at {book})")
    st.dataframe(pd.DataFrame([{
        "Sports": name(p),
        "Legs": " · ".join(f"{n} {SPORT_NAME[s]}" for s, n in p["rows"]["sport"].value_counts().items()),
        "Odds": american(p["price"]), "Hits": f"{p['chance']:.1%}", "About 1 in": round(1 / p["chance"]),
        "Expected profit": f"{p['ev']:+.0%}"} for p in options]), hide_index=True, width="stretch")
    choice = st.selectbox("Show this parlay", range(len(options)), key="mega_choice",
                          format_func=lambda i: f"{name(options[i])} · {american(options[i]['price'])} · "
                                                f"hits {options[i]['chance']:.1%}")
    chosen = options[choice if choice is not None and choice < len(options) else 0]

    picked = chosen["rows"].sort_values(["game_date", "sport"])
    legs = [leg_dict(row, ["model_chance"]) for _, row in picked.iterrows()]
    price = math.prod(float(l["price"]) for l in legs)            # payouts multiply
    chance = math.prod(float(l["model_chance"]) for l in legs)   # every leg has to hit
    ev = chance * price - 1
    mix = " + ".join(f"{n} {SPORT_NAME[s]}" for s, n in picked["sport"].value_counts().items())
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.html(f'<div class="bet-title">{n_legs}-leg parlay at {book.upper()}</div>'
                    f'<div class="chips" style="margin-bottom:10px">'
                    f'<span class="chip blue">{american(price)}</span>'
                    f'<span class="chip">{mix}</span>'
                    f'<span class="chip">Hits {chance:.1%} - about 1 in {1 / chance:.0f}</span>'
                    f'<span class="chip">Fair odds {american(1 / chance)}</span>'
                    f'<span class="chip {"green" if ev > 0 else ""}">{ev:+.0%} expected profit</span></div>')
            for (_, row), leg in zip(picked.iterrows(), legs):
                text_col, swap_col = st.columns([8, 1], vertical_alignment="center")
                day = pd.Timestamp(row["game_date"]).strftime("%a")
                text_col.html(legs_html([leg | {"game": f'{SPORT_NAME[row["sport"]]} · {row["game"]} · {day}'}]))
                swap_col.button("", key=f"mega_swap_{leg_key(row)}", icon=":material/swap_horiz:",
                                help="Swap this leg for the next best one",
                                on_click=st.session_state.mega_skip.add, args=(leg_key(row),))
            if st.session_state.mega_skip:
                st.button("Bring back swapped-out legs", on_click=st.session_state.mega_skip.clear,
                          icon=":material/restart_alt:")
        stake = suggested_stake(chance, price, 0.005)   # big parlays: at most 0.5% of your bankroll
        place_bet_box(right, f"mega_{book}_{n_legs}_{abs(hash(tuple(picked.apply(leg_key, axis=1)))) % 10**8}",
                      legs, book, price, stake)
    st.caption(f"Reality check: this hits about once every {1 / chance:.0f} tries - it's a promo play, not a "
               "plan. Legs on later days keep the parlay open until those games finish. Check injury news before "
               "the weekend games, and confirm the price in your FanDuel app.")


def late_bet():
    """log a bet you placed at the book but forgot to click 'I placed this' on.
    It's graded like any other bet - even if its games are already over."""
    how = st.segmented_control("Where was it from?", ["The model's picks (last 7 days)", "Build it from the lines"],
                               default="The model's picks (last 7 days)", key="late_how")
    if how == "Build it from the lines":
        # for Auto parlay / Mega parlay / your own builder parlays (those aren't saved until you click the button)
        st.caption("Pick the legs from the latest lines of every sport. Works for bets from the latest model run "
                   "of each sport - older lines get replaced when a model runs again.")
        lines = all_sport_lines()
        if lines.empty:
            st.info("No lines to pick from - no model has run recently.", icon=":material/info:")
            return
        book = st.segmented_control("Book", sorted(lines["book"].unique()), key="late_book")
        if not book:
            return
        at_book = lines[lines["book"] == book].drop_duplicates(["sport", "player", "bet", "game"])
        labels = {f'{SPORT_NAME[r["sport"]]} · {r["player"]} · {r["bet"]} · {r["game"]} '
                  f'({pd.Timestamp(r["game_date"]).strftime("%a %b %d")}) · {american(r["price"])}': i
                  for i, r in at_book.iterrows()}
        chosen = st.multiselect("Legs (type a name to search)", list(labels), key="late_legs")
        if not chosen:
            return
        rows = at_book.loc[[labels[c] for c in chosen]]
        legs = [leg_dict(row) | {"sport": row["sport"], "game_date": row["game_date"]} for _, row in rows.iterrows()]
        price = math.prod(float(l["price"]) for l in legs)
        st.html(legs_html(legs))
        st.caption("Type in the odds you actually got - the dashboard's price is only a guess for a parlay.")
        place_bet_box(st, f"late_lines_{abs(hash(tuple(chosen))) % 10**8}", legs, book, price, 1.0)
        return

    # the model's own picks (singles and its value parlays), saved every time a model runs
    found = load_bets(FOUND_FILE)
    week_ago = (pd.Timestamp(TODAY) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    found = found[found["game_date"] >= week_ago]
    mine = load_bets()
    already = set(zip(mine["description"], mine["book"], mine["game_date"])) if len(mine) else set()
    found = found[[(d, b, g) not in already for d, b, g in zip(found["description"], found["book"], found["game_date"])]]
    if found.empty:
        st.info("No model picks from the last 7 days that you haven't logged already.", icon=":material/info:")
        return
    found = found.sort_values(["game_date", "id"], ascending=False)
    labels = {f'{pd.Timestamp(r["game_date"]).strftime("%a %b %d")} · {r["type"]} · {r["book"]} · '
              f'{american(r["price"])} · {r["description"]}': i for i, r in found.iterrows()}
    pick = st.selectbox("The bet", list(labels), index=None, placeholder="Pick the bet you placed", key="late_pick")
    if not pick:
        return
    row = found.loc[labels[pick]]
    legs = json.loads(row["legs"])
    st.html(legs_html(legs))
    place_bet_box(st, f"late_found_{row['id']}", legs, row["book"], row["price"], row["stake"],
                  game_date=row["game_date"])


# ---------------------------------------------------------------- the Card Room (picked at the top of the sidebar)
# A casino-style lobby: one tile per game - click Play to sit down, "Back to lobby" to get up.
# To add a game later: write its table function (like baccarat()), draw its tile art, and add it to CARD_ROOM.
def tile_card(x, y, rank, suit, turn=0):
    """one playing card for the lobby artwork (SVG), at x / y, turned by `turn` degrees"""
    color = "#DC2626" if BJ_SUITS.get(suit) else "#111"
    return (f'<g transform="translate({x} {y}) rotate({turn})">'
            f'<rect x="-32" y="-45" width="64" height="90" rx="8" fill="#FAFAFA" stroke="#D4D4D4"/>'
            f'<text x="-24" y="-22" font-size="20" font-weight="800" fill="{color}" font-family="Inter,sans-serif">{rank}</text>'
            f'<text x="10" y="34" font-size="24" fill="{color}">{suit}</text></g>')


def tile_chip(x, y, color, label=""):
    """a casino chip for the lobby artwork"""
    return (f'<g transform="translate({x} {y})"><circle r="17" fill="{color}" stroke="#FFF" stroke-width="3" '
            f'stroke-dasharray="6 5"/><circle r="11" fill="none" stroke="rgba(255,255,255,.6)"/>'
            f'<text y="4" text-anchor="middle" font-size="10" font-weight="800" fill="#FFF" '
            f'font-family="Inter,sans-serif">{label}</text></g>')


def tile_die(x, y, value, turn=0):
    """a die for the lobby artwork"""
    spots = {0: (-14, -14), 2: (14, -14), 3: (-14, 0), 4: (0, 0), 5: (14, 0), 6: (-14, 14), 8: (14, 14)}
    pips = "".join(f'<circle cx="{spots[i][0]}" cy="{spots[i][1]}" r="5" fill="#111"/>' for i in CR_PIPS[value])
    return (f'<g transform="translate({x} {y}) rotate({turn})"><rect x="-28" y="-28" width="56" height="56" rx="11" '
            f'fill="#FAFAFA" stroke="#D4D4D4"/>{pips}</g>')


def tile_art(felt, glow, pieces, title):
    """a game's lobby picture: a felt table with cards and chips on it, and the game's name across the bottom.
    Drawn as an SVG, then wrapped in an <img> - the page strips raw <svg> tags, but pictures are fine."""
    svg = (f'<svg viewBox="0 0 320 180" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><radialGradient id="g{title}" cx="50%" cy="20%" r="85%"><stop offset="0" stop-color="{glow}"/>'
            f'<stop offset="1" stop-color="{felt}"/></radialGradient></defs>'
            f'<rect width="320" height="180" fill="url(#g{title})"/>'
            f'<ellipse cx="160" cy="210" rx="210" ry="95" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="2"/>'
            f'{pieces}'
            f'<rect y="138" width="320" height="42" fill="rgba(0,0,0,.45)"/>'
            f'<text x="16" y="166" font-size="20" font-weight="800" fill="#FFF" letter-spacing="2" '
            f'font-family="Inter,sans-serif">{title.upper()}</text></svg>')
    return (f'<img class="lobby-art" alt="{title}" '
            f'src="data:image/svg+xml;base64,{base64.b64encode(svg.encode("utf-8")).decode()}">')


# every game in the Card Room: name, one-line pitch, its picture, and the function that draws its table
CARD_ROOM = {
    "blackjack": {"name": "Blackjack", "tag": "Beat the dealer to 21 · counting trainer",
                  "art": tile_art("#04120A", "#166534",
                                  tile_card(120, 72, "A", "♠", -12) + tile_card(165, 66, "K", "♥", 10)
                                  + tile_chip(232, 96, "#16A34A", "21") + tile_chip(258, 76, "#000", "BJ"),
                                  "Blackjack"),
                  "play": blackjack},
    "baccarat": {"name": "Baccarat", "tag": "Player, Banker or Tie · closest to 9 wins",
                 "art": tile_art("#1F0606", "#7F1D1D",
                                 tile_card(118, 70, "9", "♦", -10) + tile_card(162, 66, "8", "♣", 8)
                                 + tile_chip(232, 98, "#475569", "P") + tile_chip(258, 78, "#DC2626", "B"),
                                 "Baccarat"),
                 "play": baccarat},
    "video_poker": {"name": "Video Poker", "tag": "Jacks or Better · hold, draw, hit a royal",
                    "art": tile_art("#0F0A2E", "#4338CA",
                                    "".join(tile_card(92 + 34 * i, 70 + abs(i - 2) * 4, r, "♠", (i - 2) * 9)
                                            for i, r in enumerate(["10", "J", "Q", "K", "A"])),
                                    "Video Poker"),
                    "play": video_poker},
    "roulette": {"name": "Roulette", "tag": "American wheel · 0 and 00, red / black",
                 "art": tile_art("#04120A", "#166534",
                                 f'<svg x="92" y="6" width="136" height="136" viewBox="0 0 300 300">'
                                 f'{rl_wheel_parts(labels=False)}</svg>'
                                 + tile_chip(250, 100, "#B91C1C", "") + tile_chip(272, 78, "#141414", ""),
                                 "Roulette"),
                 "play": roulette},
    "three_card_poker": {"name": "Three Card Poker", "tag": "Ante, Play or Fold · beat the dealer",
                         "art": tile_art("#061A2E", "#0E7490",
                                         tile_card(118, 72, "Q", "♦", -12) + tile_card(160, 66, "K", "♦", 0)
                                         + tile_card(202, 72, "A", "♦", 12) + tile_chip(262, 98, "#16A34A", "ANTE"),
                                         "Three Card Poker"),
                         "play": three_card_poker},
    "let_it_ride": {"name": "Let It Ride", "tag": "Pull back or ride · tens or better",
                    "art": tile_art("#1E0F33", "#7C3AED",
                                    tile_card(112, 70, "10", "♠", -10) + tile_card(156, 66, "J", "♠", 0)
                                    + tile_card(200, 70, "Q", "♠", 10) + tile_chip(262, 98, "#7C3AED", "$"),
                                    "Let It Ride"),
                    "play": let_it_ride},
    "craps": {"name": "Craps", "tag": "The dice game · Pass Line and free odds",
              "art": tile_art("#2A0707", "#B91C1C", tile_die(122, 78, 5, -14) + tile_die(194, 72, 2, 10)
                              + tile_chip(262, 98, "#16A34A", "PASS"), "Craps"),
              "play": craps},
    "poker": {"name": "Texas Hold'em", "tag": "Real poker · heads-up vs Ace the bot",
              "art": tile_art("#0F1F4A", "#2563EB", tile_card(128, 72, "A", "♠", -12) + tile_card(172, 70, "A", "♥", 10)
                              + tile_chip(246, 100, "#1F1F1F", "ALL") + tile_chip(276, 92, "#7C3AED", "IN"),
                              "Hold'em"),
              "play": poker},
}


def open_game(key):
    st.session_state.card_game = key


def card_room():
    """the lobby (a tile per game) - or the table of the game you picked"""
    if st.session_state.get("job"):   # scripts started from the Bets view keep running - show how they're doing
        job_progress()
    picked = st.session_state.get("card_game")
    if picked in CARD_ROOM:
        back, name = st.columns([1, 5], vertical_alignment="center")
        back.button("Back to lobby", icon=":material/arrow_back:", on_click=open_game, args=(None,),
                    key="lobby_back", width="stretch")
        name.markdown(f"#### {CARD_ROOM[picked]['name']}")
        CARD_ROOM[picked]["play"]()
        return

    st.html("""<style>
      .lobby-art { width:100%; display:block; border-radius:12px; transition:transform .2s ease, filter .2s ease; }
      .lobby-art:hover { transform:scale(1.03); filter:brightness(1.12); }
      .lobby-soon { aspect-ratio:320/180; border:2px dashed #2A2A2A; border-radius:12px; display:flex;
                    flex-direction:column; align-items:center; justify-content:center; color:#7A7A7A; gap:6px;
                    font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:13px; }
      .lobby-soon span { font-size:34px; color:#3A3A3A; }
    </style>""")
    chips = bj_game()["chips"]
    st.markdown(f"**Pick a table** · you have **{chips:,} chips** (fake - no real money, just for fun). "
                "Anything you started with Run all keeps running in the background.")
    tiles = list(CARD_ROOM.items()) + [(None, None)]   # the last tile: more games on the way
    for row in range(0, len(tiles), 3):
        for col, (key, game) in zip(st.columns(3, gap="medium"), tiles[row:row + 3]):
            with col.container(border=True):
                if game is None:
                    st.html('<div class="lobby-soon"><span>+</span>More games coming soon</div>')
                    st.caption("New tables get added here.")
                    continue
                st.html(game["art"])
                st.markdown(f"**{game['name']}**")
                st.caption(game["tag"])
                st.button("Play", key=f"play_{key}", on_click=open_game, args=(key,), type="primary",
                          width="stretch", icon=":material/play_arrow:")


if BLACKJACK:
    card_room()
    st.stop()   # the Card Room is just the lobby / a table - no betting tabs underneath

tonight_tab, singles_tab, parlays_tab, mega_tab, live_tab, profit_tab, fumbles_tab = st.tabs(
    [":material/target: Tonight", ":material/confirmation_number: Singles", ":material/layers: Parlays",
     ":material/stacks: Mega parlay", ":material/sensors: Live", ":material/account_balance_wallet: Profit & loss",
     ":material/sports_handball: Fumbles"])

# ---------------------------------------------------------------- tab 1: tonight
with tonight_tab:
    # every step = (script, label, extra options)
    steps = [(script, label, ()) for script, label in CFG["steps"]]
    if SPORT in WEEK_SPORTS:
        # Tonight (the normal way) or This week: odds + bets for every game in the next 7 days
        span = st.segmented_control("Games", ["Tonight", "This week"], default="Tonight", key=f"span_{SPORT_KEY}",
                                    help="This week gets the odds for every game in the next 7 days and runs the "
                                         "model for each day. It costs a lot more API credits than tonight.")
        if span == "This week":
            steps = [steps[0], (steps[1][0], f"Getting {SPORT} odds for the week", ("--week",)),
                     ("Week_Run.py", f"Running the {SPORT} model for each day", (WEEK_SPORTS[SPORT],))]
            try:
                n_games, credits = week_cost(SPORT)
                st.warning(f"**This week: {n_games} {SPORT} games in the next 7 days - about {credits} API credits** "
                           f"for the odds. Books usually post player props only a day or so ahead, so later games "
                           f"may have just moneylines for now.", icon=":material/payments:")
            except Exception:
                st.warning("This week costs a lot more API credits than tonight (props cost ~4-5 credits per game).",
                           icon=":material/payments:")
    c1, c2, c3, c4 = st.columns(4)
    # same day, no game started since the last download? the scrapers reuse it (no credits). Tick this to re-download.
    fresh = st.checkbox("Download fresh (uses API credits even if you already downloaded today)", key="force_fresh")
    busy = job_running()   # one job at a time - the buttons wait until it's finished
    # each button starts a background job: (script, label, extra options, force) for every step it runs
    if c1.button(f"Update {SPORT} stats", icon=":material/download:", width="stretch", disabled=busy):
        start_job([steps[0] + (fresh,)])
    if c2.button("Get odds", icon=":material/request_quote:", width="stretch", disabled=busy,
                 help="Uses API credits - unless you already got today's odds and no game has started since"):
        start_job([steps[1] + (fresh,)])
    if c3.button("Find bets", icon=":material/insights:", width="stretch", disabled=busy):
        start_job([steps[2] + (False,)])
    if c4.button("Run all", icon=":material/play_arrow:", type="primary", width="stretch", disabled=busy):
        start_job([step + (fresh and i < 2,) for i, step in enumerate(steps)])

    # while it runs: live progress (the page stays usable - blackjack is in the sidebar)
    if st.session_state.get("job"):
        job_progress()

    # how many bets of each kind the model found
    if made_today(CFG["bets"], CFG["days"]):
        found = pd.read_csv(CFG["bets"])["stat"].value_counts()
        n_parlays = len(pd.read_csv(CFG["parlays"])) if made_today(CFG["parlays"], CFG["days"]) else 0
        cols = st.columns(len(BET_TYPES) + 1)
        for col, (stat, label, _) in zip(cols, BET_TYPES):
            col.metric(label, int(found.get(stat, 0)), border=True)
        cols[-1].metric("Parlays", n_parlays, border=True)
        st.caption("Open the **Singles** and **Parlays** tabs to see and place them.")

    # face-offs: who wins each game
    st.subheader("Face-offs")
    if made_today(CFG["faceoffs"], CFG["days"]):
        cards = ""
        for _, f in pd.read_csv(CFG["faceoffs"]).iterrows():
            def row(team, chance, fav):
                return (f'<div class="fo-row {"fav" if fav else ""}"><img src="{logo(team, SPORT_KEY)}">{team}'
                        f'<span class="pct">{chance:.0%}</span></div>')
            # under each card: the game's day (when there's more than tonight) + the starting pitchers / goalies
            notes = []
            if "date" in f.index and pd.notna(f["date"]) and (str(f["date"]) != TODAY or CFG["days"]):
                notes.append(pd.Timestamp(f["date"]).strftime("%A %b %d"))
            for kind, label in (("pitcher", "Pitchers"), ("goalie", "Goalies")):
                if f"away_{kind}" in f.index:
                    notes.append(f'{label}: {html.escape(str(f[f"away_{kind}"]))} vs '
                                 f'{html.escape(str(f[f"home_{kind}"]))}')
            cards += (f'<div class="fo-card">{row(f["away"], f["away_win"], f["away_win"] > f["home_win"])}'
                      f'<div class="fo-bar"><div style="width:{f["away_win"]:.0%};background:#475569"></div>'
                      f'<div style="width:{f["home_win"]:.0%};background:{BLUE}"></div></div>'
                      f'{row(f["home"], f["home_win"], f["home_win"] >= f["away_win"])}'
                      + "".join(f'<div class="fo-goalies">{n}</div>' for n in notes) + '</div>')
        st.html(f'<div class="fo-grid">{cards}</div>')
    else:
        st.info("Not run yet today. Click **Find bets** or **Run all**.", icon=":material/info:")

# ---------------------------------------------------------------- tab 2: singles, split by bet type
with singles_tab:
    if not made_today(CFG["bets"], CFG["days"]):
        st.info(f"No {SPORT} bets yet {CFG['when']}. Run the model first (Tonight tab).", icon=":material/info:")
    else:
        bets_today = pd.read_csv(CFG["bets"])
        type_tabs = st.tabs([f":material/{icon}: {label} ({(bets_today['stat'] == stat).sum()})"
                             for stat, label, icon in BET_TYPES])
        for type_tab, (stat, label, _) in zip(type_tabs, BET_TYPES):
            with type_tab:
                these = bets_today[bets_today["stat"] == stat]
                if these.empty:
                    st.caption(f"No value {label.lower()} bets {CFG['when']}.")
                for i, b in these.iterrows():
                    single_card(b, f"single_{SPORT_KEY}_{i}")
    st.caption(f"{CFG['note']}, and confirm every price in your Ontario app before betting.")

# ---------------------------------------------------------------- tab 3: parlays
with parlays_tab:
    model_col, builder_col = st.columns([1, 1], gap="large")

    with model_col:
        st.subheader("Model's parlays")
        st.caption("Built automatically from the value bets: one book, different games. Pick **This week** on the "
                   "Tonight tab and Run all to get parlays for the whole week.")
        if not made_today(CFG["parlays"], CFG["days"]):
            st.info("Run the model first (Tonight tab).", icon=":material/info:")
        else:
            parlays_today = pd.read_csv(CFG["parlays"])
            week_run = "span" in parlays_today.columns   # only a "This week" run (Week_Run.py) adds this column
            if parlays_today.empty:
                # say WHY: which days the odds cover, and how many value bets there were to build from
                n_bets = len(pd.read_csv(CFG["bets"])) if made_today(CFG["bets"], CFG["days"]) else 0
                try:
                    odds_days = sorted(pd.read_csv(CFG["lines"])["game_date"].dropna().unique())
                except Exception:
                    odds_days = []
                covers = ", ".join(pd.Timestamp(d).strftime("%a %b %d") for d in odds_days) or "no games"
                st.info(f"**No value parlays.** The odds cover **{covers}**, and the model found **{n_bets} value "
                        f"bet{'s' if n_bets != 1 else ''}** in them - a parlay needs at least 2, in different games. "
                        + ("Pick **This week** on the Tonight tab and **Run all** to look at every game in the next "
                           "7 days." if SPORT in WEEK_SPORTS and len(odds_days) < 2 else
                           "That's the model being strict - no bet beat the books by enough. ")
                        + " You can still build your own on the right.", icon=":material/info:")
            # after a "This week" run: the best parlays mixed across the whole week first, then each night's own
            span = parlays_today["span"] if week_run else pd.Series("day", parlays_today.index)
            for group, title, note in (("week", "Best parlays for the week" if week_run and not parlays_today.empty
                                        else None,
                                        "Legs from any day this week, mixed - the best value across every game. "
                                        "It isn't settled until its last leg's game is over."),
                                       ("day", "Night by night" if (span == "week").any() else None, None)):
                these = parlays_today[span == group]
                if title:
                    st.markdown(f"**{title}**")
                    if note:
                        st.caption(note)
                    if these.empty:
                        st.caption("No week parlays passed the checks this time.")
                for i, p in these.iterrows():
                    legs = json.loads(p["legs"])
                    days = sorted({str(l.get("game_date")) for l in legs} - {"None", "nan"})
                    when = (f'{pd.Timestamp(days[0]).strftime("%a")}-{pd.Timestamp(days[-1]).strftime("%a")} · '
                            if len(days) > 1 else f'{pd.Timestamp(days[0]).strftime("%a %b %d")} · '
                            if days and (span == "week").any() else "")
                    with st.container(border=True):
                        st.html(f'<div class="bet-title">{when}{len(legs)}-leg parlay</div>'
                                f'<div class="chips" style="margin-bottom:10px">'
                                f'<span class="chip blue">{p["book"].upper()} {american(p["price"])}</span>'
                                f'<span class="chip">Hits {p["chance"]:.0%} · fair {american(1 / p["chance"])}</span>'
                                f'<span class="chip green">{p["ev"]:+.0%} expected profit</span></div>{legs_html(legs)}')
                        place_bet_box(st, f"parlay_{SPORT_KEY}_{i}", legs, p["book"], p["price"], p["stake"])

    with builder_col:
        st.subheader("Build your own")
        if not made_today(CFG["lines"], CFG["days"]):
            st.info("Run the model first (Tonight tab) - the builder uses the latest odds.", icon=":material/info:")
        else:
            lines = pd.read_csv(CFG["lines"])
            books = sorted(lines["book"].unique())
            book = st.segmented_control("Book", books, key=f"builder_book_{SPORT_KEY}",
                                        default=st.session_state.slip_book if st.session_state.slip_book in books
                                        else books[0])
            stat = st.segmented_control("Bet type", [s for s, _, _ in BET_TYPES], key=f"builder_type_{SPORT_KEY}",
                                        format_func=lambda s: TYPE_LABEL[s], default=CFG["default_type"])
            options = lines[(lines["book"] == book) & (lines["stat"] == stat)].sort_values(
                "model_chance", ascending=False)
            if options.empty:
                st.caption(f"{book} has no {TYPE_LABEL.get(stat, 'such').lower()} lines for tonight's games.")
            else:
                pick = st.selectbox(
                    "Leg", options.index, key=f"builder_pick_{SPORT_KEY}",
                    format_func=lambda i: (f"{options.loc[i, 'player']} · {options.loc[i, 'bet']} · "
                                           f"{american(options.loc[i, 'price'])} · model "
                                           f"{options.loc[i, 'model_chance']:.0%}"
                                           f"{'  · VALUE' if options.loc[i, 'ev'] >= 0.05 else ''}"))
                leg = leg_dict(options.loc[pick], ["model_chance"])
                st.button("Add to parlay", icon=":material/add:", on_click=add_leg, args=(leg, book),
                          width="stretch")
            if st.session_state.get("slip_msg"):
                st.error(st.session_state.slip_msg, icon=":material/error:")

            # the slip
            slip = st.session_state.slip
            with st.container(border=True):
                if not slip:
                    st.caption("Your parlay is empty. Pick a book, a bet type and a leg, then **Add to parlay**.")
                else:
                    price = math.prod(float(l["price"]) for l in slip)            # odds multiply
                    chance = math.prod(float(l["model_chance"]) for l in slip)   # every leg has to hit
                    ev = chance * price - 1
                    st.html(f'<div class="bet-title">{len(slip)}-leg parlay at '
                            f'{st.session_state.slip_book.upper()}</div>'
                            f'<div class="chips" style="margin-bottom:6px">'
                            f'<span class="chip blue">{american(price)}</span>'
                            f'<span class="chip">Model: hits {chance:.0%} · fair {american(1 / chance)}</span>'
                            f'<span class="chip {"green" if ev > 0 else ""}">{ev:+.0%} expected profit</span></div>')
                    for i, l in enumerate(slip):
                        a, b = st.columns([6, 1])
                        a.html(legs_html([l]))
                        b.button("", icon=":material/close:", key=f"remove_{i}", on_click=remove_leg, args=(i,),
                                 help="Remove this leg")
                    games = [l["game"] for l in slip]
                    if len(set(games)) < len(games):
                        st.warning("Two legs are from the same game. Those are linked (if a team scores a lot, "
                                   "its players hit together), so the model's chance here isn't reliable.",
                                   icon=":material/warning:")
                    if ev <= 0:
                        st.caption("The model doesn't see value in this combination - you can still place it.")
                    signature = "_".join(f"{l['player']}{l['stat']}" for l in slip)
                    placed = place_bet_box(st, f"builder_{abs(hash(signature))}", [dict(l) for l in slip],
                                           st.session_state.slip_book, price,
                                           suggested_stake(chance, price, 0.01))
                    st.button("Clear parlay", icon=":material/delete_sweep:", on_click=clear_slip)
                    if placed:
                        clear_slip()
                        st.rerun()
    # ---- big parlay: 4 to 10 legs, picked automatically ----
    st.divider()
    st.subheader("Auto parlay (2-10 legs)")
    st.caption("Picks the most likely leg from each game (one per game, so the legs don't depend on each other), "
               "only using legs the model doesn't think are bad bets. Every extra leg multiplies the payout - "
               "and cuts the chance of hitting. Keep the stake small.")
    if not made_today(CFG["lines"], CFG["days"]):
        st.info("Run the model first (Tonight tab).", icon=":material/info:")
    else:
        lines = pd.read_csv(CFG["lines"])
        c1, c2, c3 = st.columns([2, 2, 2])
        big_book = c1.segmented_control("Book", sorted(lines["book"].unique()), key=f"big_book_{SPORT_KEY}",
                                        default=sorted(lines["book"].unique())[0])
        n_legs = c2.slider("Legs", 2, 10, 3, key="big_legs")
        min_chance = c3.slider("Each leg at least", 0.50, 0.85, 0.60, 0.05, key="big_min", format="%.2f",
                               help="The lowest chance (after blending with the books) any single leg can have")
        pool = lines[(lines["book"] == big_book) & (lines["model_chance"] >= min_chance) & (lines["ev"] >= 0)]
        best_per_game = pool.sort_values("model_chance", ascending=False).drop_duplicates("game")
        if len(best_per_game) < n_legs and len(best_per_game) >= 2:
            # not enough games for that many legs (e.g. a 2-game night) -> build the biggest one possible
            st.info(f"Only {len(best_per_game)} game(s) tonight have a leg at {big_book} that's at least "
                    f"{min_chance:.0%} likely and not a bad bet - so this is a {len(best_per_game)}-leg parlay "
                    f"(one leg per game).", icon=":material/info:")
            n_legs = len(best_per_game)
        if len(best_per_game) < 2:
            st.warning(f"Only {len(best_per_game)} game(s) tonight have a leg at {big_book} that's at least "
                       f"{min_chance:.0%} likely and not a bad bet - a parlay needs 2 different games. "
                       f"Try another book or lower the minimum chance.", icon=":material/warning:")
        else:
            picked = best_per_game.head(n_legs)
            legs = [leg_dict(row, ["model_chance"]) for _, row in picked.iterrows()]
            price = math.prod(float(l["price"]) for l in legs)            # payouts multiply
            chance = math.prod(float(l["model_chance"]) for l in legs)   # every leg has to hit
            ev = chance * price - 1
            with st.container(border=True):
                left, right = st.columns([3, 2], gap="large")
                left.html(f'<div class="bet-title">{n_legs}-leg parlay at {big_book.upper()}</div>'
                          f'<div class="chips" style="margin-bottom:10px">'
                          f'<span class="chip blue">{american(price)}</span>'
                          f'<span class="chip">Hits {chance:.1%} - about 1 in {1 / chance:.0f}</span>'
                          f'<span class="chip">Fair odds {american(1 / chance)}</span>'
                          f'<span class="chip {"green" if ev > 0 else ""}">{ev:+.0%} expected profit</span></div>'
                          f'{legs_html(legs)}')
                stake = suggested_stake(chance, price, 0.005)   # big parlays: at most 0.5% of your bankroll
                placed = place_bet_box(right, f"big_{big_book}_{n_legs}_{min_chance}", legs, big_book, price, stake)
                if right.button("Edit in builder", key="big_to_builder", icon=":material/edit:", width="stretch",
                                help="Copy these legs into 'Build your own' above to swap or remove legs"):
                    st.session_state.slip = legs
                    st.session_state.slip_book = big_book
                    st.session_state.slip_msg = None
                    st.rerun()
            if chance < 0.05:
                st.caption(f"Reality check: this hits about once every {1 / chance:.0f} tries. "
                           "Even when the math is in your favour, you'll lose it almost every night.")

    st.caption("Confirm every price in your Ontario app - books sometimes price parlays a little differently.")

# ---------------------------------------------------------------- mega parlay (every sport)
with mega_tab:
    mega_parlay()

# ---------------------------------------------------------------- tab 4: live
with live_tab:
    live_bets()

# ---------------------------------------------------------------- tab 5: profit and loss
with profit_tab:
    grade_col, graph_col, _ = st.columns([1, 1, 2])
    if graph_col.button("Profit / loss graph", key="graph_profit", icon=":material/bar_chart:", width="stretch"):
        pnl_popup([("Your bets", load_bets())])
    if grade_col.button("Grade finished bets", icon=":material/fact_check:", type="primary", width="stretch",
                        help="Checks the final scores and marks your bets (and the model's finds) won or lost"):
        with st.spinner("Checking final scores..."):   # straight from the NHL's live game feed - no download needed
            mine, found_graded = grade_bets(), grade_bets(FOUND_FILE)
        st.success(f"Graded {mine} of your bet(s) and {found_graded} of the model's finds. "
                   "Bets in games that aren't over yet stay open.", icon=":material/check_circle:")

    # ---- placed a bet at the book but forgot to click "I placed this"? add it here ----
    with st.expander("Forgot to log a bet? Add it here", icon=":material/history:"):
        late_bet()

    bets = load_bets()
    if bets.empty:
        st.info("No bets saved yet. Click **I placed this** on a bet in the Tonight tab.", icon=":material/info:")
    else:
        settled = bets[bets["status"].isin(SETTLED)]
        open_bets = bets[bets["status"] == "open"]
        wins, losses, pushes = (settled["status"] == "won").sum(), (settled["status"] == "lost").sum(), \
            (settled["status"] == "push").sum()
        cashed = settled[settled["status"] == "cashed out"]
        profit = settled["profit"].sum()
        staked = settled["stake"].sum()

        # win rate vs the win rate you NEED at the odds you took (at -110 you need 52.4%; at +150, 40%)
        decided = settled[settled["status"].isin(["won", "lost"])].copy()
        win_rate = (decided["status"] == "won").mean() if len(decided) else None
        breakeven = (1 / decided["price"]).mean() if len(decided) else None

        m1, m2, m0, m3, m4 = st.columns(5)
        m1.metric("Profit / loss", cad(profit), delta=f"{profit:+.2f} CAD" if profit else None, border=True)
        m2.metric("ROI", f"{profit / staked:+.1%}" if staked else "-", help="profit ÷ money bet", border=True)
        m0.metric("Win rate", f"{win_rate:.0%}" if win_rate is not None else "-",
                  f"break-even {breakeven:.0%}" if breakeven is not None else None,
                  delta_color="off", delta_arrow="off", border=True,
                  help="How often your bets win, next to how often they NEED to win at the odds you took. "
                       "Beating that number is what makes money - a high win rate on short odds can still lose.")
        m3.metric("Record (W-L-P)", f"{wins}-{losses}-{pushes}",
                  f"{len(cashed)} cashed out" if len(cashed) else None,
                  delta_color="off", delta_arrow="off", border=True)
        m4.metric("Open bets", f"{len(open_bets)}", f"{cad(open_bets['stake'].sum())} at risk",
                  delta_color="off", delta_arrow="off", border=True)

        if len(decided):
            with st.expander("Win rate by sport", icon=":material/percent:"):
                decided["sport"] = [("parlay" if len(json.loads(l)) > 1 else
                                     SPORT_NAME.get(json.loads(l)[0].get("sport", "nhl"), "NHL"))
                                    for l in decided["legs"]]
                by_sport = decided.groupby("sport").agg(bets=("status", "size"),
                                                        win_rate=("status", lambda s: (s == "won").mean()),
                                                        need=("price", lambda p: (1 / p).mean()),
                                                        profit=("profit", "sum"), staked=("stake", "sum"))
                by_sport["roi"] = by_sport["profit"] / by_sport["staked"]
                st.dataframe(by_sport.drop(columns="staked").reset_index(), hide_index=True, width="stretch",
                             column_config={"sport": "Sport", "bets": "Bets",
                                            "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
                                            "need": st.column_config.NumberColumn("Need to break even",
                                                                                  format="percent"),
                                            "profit": st.column_config.NumberColumn("Profit", format="CA$%.2f"),
                                            "roi": st.column_config.NumberColumn("ROI", format="percent")})
                st.caption("A sport is making money when its win rate beats its break-even rate.")

        if len(settled):
            st.subheader("Profit over time")
            # bars + running-total line (same as the pop-up) - unlike an area chart it still shows with just one day
            st.altair_chart(pnl_chart(settled), width="stretch")
        if len(settled) < 100:
            st.caption(f"Only {len(settled)} settled bets - results this small are mostly luck. "
                       "Judge the model after a few hundred.")

        st.subheader("All bets")
        table = bets.sort_values("id", ascending=False).copy()
        table["odds"] = table["price"].map(american)
        table["pays if it hits"] = (table["stake"] * table["price"]).round(2)   # stake + profit
        st.dataframe(table[["game_date", "status", "description", "book", "odds", "stake", "pays if it hits",
                            "profit"]],
                     hide_index=True, width="stretch",
                     # widths in pixels so every column fits; the bet description gets the rest
                     column_config={"game_date": st.column_config.TextColumn("Date", width=95),
                                    "status": st.column_config.TextColumn("Status", width=65),
                                    "description": st.column_config.TextColumn("Bet", width=330),
                                    "book": st.column_config.TextColumn("Book", width=80),
                                    "odds": st.column_config.TextColumn("Odds", width=60),
                                    "stake": st.column_config.NumberColumn("Stake", format="CA$%.2f", width=90),
                                    "pays if it hits": st.column_config.NumberColumn("Pays", format="CA$%.2f",
                                                                                     width=90),
                                    "profit": st.column_config.NumberColumn("Profit (CA$)", format="%+.2f",
                                                                            width=100)})

        if len(open_bets):
            st.subheader("Cashed out early?")
            st.caption("Took the book's cash-out offer? Record what they paid you - the bet closes and its profit "
                       "is what you got back minus what you bet.")
            cash_pick_col, cash_amount_col, cash_button_col = st.columns([3, 1, 1], vertical_alignment="bottom")
            open_labels = {row["id"]: f"{row['game_date']} · {row['description']} · {cad(row['stake'])} at "
                                      f"{american(row['price'])}" for _, row in open_bets.iterrows()}
            cash_pick = cash_pick_col.selectbox("Bet", list(open_labels), key="cash_pick",
                                                format_func=lambda i: open_labels.get(i, str(i)))
            cash_amount = cash_amount_col.number_input("The book paid me (CA$)", min_value=0.0, step=1.0,
                                                       key="cash_amount")
            if cash_button_col.button("I cashed out", key="cash_confirm", icon=":material/payments:",
                                      type="primary", width="stretch", disabled=cash_amount <= 0):
                cash_out_bet(cash_pick, cash_amount)
                st.toast(f"Cashed out for {md_cad(cash_amount)}", icon=":material/payments:")
                st.rerun()

        st.subheader("Remove a bet")
        st.caption("Saved one by mistake? Remove it here. This only changes your tracking - "
                   "it doesn't cancel anything at the sportsbook.")
        pick_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
        # bet number -> readable label, worked out once
        labels = {row["id"]: f"{row['game_date']} · {row['status']} · {row['description']} · {cad(row['stake'])}"
                  for _, row in table.iterrows()}
        pick = pick_col.selectbox("Bet", list(labels), key="remove_pick",
                                  format_func=lambda i: labels.get(i, str(i)))
        with button_col.popover("Remove", icon=":material/delete:", width="stretch"):
            st.markdown("Remove this bet from your dashboard?")
            if st.button("Yes, remove it", key="remove_confirm", type="primary", icon=":material/delete:"):
                delete_bet(pick)
                st.rerun()

# ---------------------------------------------------------------- tab 6: fumbles (found but not taken)
with fumbles_tab:
    st.subheader("Fumbles")
    st.caption("Every bet the model found, split into the ones you took and the ones you skipped. "
               "Skipped bets are scored as if you'd bet the model's suggested stake.")
    grade_col, graph_col, _ = st.columns([1, 1, 2])
    if graph_col.button("Profit / loss graph", key="graph_fumbles", icon=":material/bar_chart:", width="stretch"):
        all_found, mine = load_bets(FOUND_FILE), load_bets()
        took_keys = set(zip(mine["game_date"], mine["description"]))
        not_taken = all_found[[(d, x) not in took_keys
                               for d, x in zip(all_found["game_date"], all_found["description"])]]
        pnl_popup([("Bets you took", mine), ("Bets you skipped (at the model's suggested stakes)", not_taken)])
    if grade_col.button("Grade finished bets", key="grade_fumbles", icon=":material/fact_check:", type="primary",
                        width="stretch",
                        help="Checks the final scores and marks your bets and the model's finds won or lost"):
        with st.spinner("Checking final scores..."):   # straight from the NHL's live game feed - no download needed
            mine, found_graded = grade_bets(), grade_bets(FOUND_FILE)
        st.success(f"Graded {mine} of your bet(s) and {found_graded} of the model's finds. "
                   "Bets in games that aren't over yet stay open.", icon=":material/check_circle:")

    found = load_bets(FOUND_FILE)
    placed = load_bets()
    if found.empty:
        st.info("Nothing recorded yet. Every time the model runs (Find bets / Run all), its picks are saved here.",
                icon=":material/info:")
    else:
        # a found bet counts as "taken" if you placed the same bet (same legs) on the same day
        taken_keys = set(zip(placed["game_date"], placed["description"]))
        found["taken"] = [(d, desc) in taken_keys for d, desc in zip(found["game_date"], found["description"])]
        skipped = found[~found["taken"]]
        took = placed   # your real bets, at the stakes and odds you actually got

        def summary(bets):
            done = bets[bets["status"].isin(SETTLED)]
            return done, done["profit"].sum(), done["stake"].sum()

        took_done, took_profit, took_staked = summary(took)
        skip_done, skip_profit, skip_staked = summary(skipped)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model picks found", len(found), f"{int(found['taken'].sum())} taken", delta_color="off",
                  delta_arrow="off", border=True)
        m2.metric("Skipped (fumbles)", len(skipped), border=True)
        m3.metric("Your bets: profit", cad(took_profit),
                  f"{took_profit / took_staked:+.1%} ROI" if took_staked else None, border=True)
        m4.metric("Skipped bets: profit", cad(skip_profit),
                  f"{skip_profit / skip_staked:+.1%} ROI" if skip_staked else None, border=True,
                  help="What the bets you skipped would have made at the model's suggested stakes")

        # one graph, two lines: running profit of the bets you took vs the ones you skipped
        st.subheader("Profit over time: taken vs skipped")
        lines = pd.DataFrame({
            "Bets you took": took_done.groupby("game_date")["profit"].sum(),
            "Bets you skipped": skip_done.groupby("game_date")["profit"].sum(),
        }).fillna(0).sort_index().cumsum()   # days with no bets in one group count as 0 that day
        if lines.empty:
            st.caption("The graph appears once some of these bets have been graded (after the games finish).")
        else:
            # lines with a dot on every day, so even a single day of results is visible
            long = lines.reset_index(names="date").melt("date", var_name="group", value_name="profit")
            chart = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=2.5).encode(
                x=alt.X("date:N", axis=alt.Axis(labelAngle=0)), y=alt.Y("profit:Q", title="profit (CA$)"),
                color=alt.Color("group:N", scale=alt.Scale(domain=["Bets you took", "Bets you skipped"],
                                                            range=[BLUE, AMBER]), title=None,
                                legend=alt.Legend(orient="bottom")),
                tooltip=["date:N", "group:N", alt.Tooltip("profit:Q", title="running total", format="+.2f")])
            st.altair_chart(chart.properties(height=300), width="stretch")
            if skip_profit > took_profit:
                st.caption("Your skipped bets are doing better than the ones you took so far - worth a look at "
                           "how you're choosing. (Small samples are mostly luck though.)")

        # the fumbles themselves - split into won / lost / push / still open
        head_col, which_col = st.columns([3, 2], vertical_alignment="bottom")
        which = which_col.segmented_control("Show", ["Bets you skipped", "Every model pick"],
                                            default="Bets you skipped", key="fumble_which")
        head_col.subheader("Bets you skipped" if which != "Every model pick" else "Every model pick")
        shown = skipped if which != "Every model pick" else found
        if shown.empty:
            st.caption("None - you've taken every bet the model found.")
        else:
            table = shown.sort_values(["game_date", "id"], ascending=False).copy()
            table["odds"] = table["price"].map(american)
            table["you"] = ["took it" if t else "skipped" for t in table["taken"]]
            columns = ["game_date", "type", "description", "book", "odds", "stake", "profit"] + \
                      (["you"] if which == "Every model pick" else [])
            config = {"game_date": st.column_config.TextColumn("Date", width=95),
                      "type": st.column_config.TextColumn("Type", width=100),
                      "description": st.column_config.TextColumn("Bet", width=300),
                      "book": st.column_config.TextColumn("Book", width=80),
                      "odds": st.column_config.TextColumn("Odds", width=60),
                      "stake": st.column_config.NumberColumn("Stake", format="CA$%.2f", width=90),
                      "profit": st.column_config.NumberColumn("Made (CA$)", format="%+.2f", width=110),
                      "you": st.column_config.TextColumn("You", width=75)}
            groups = [("Won", ["won"], "check_circle"), ("Lost", ["lost"], "cancel"),
                      ("Push / void", ["push", "cashed out"], "undo"), ("Still open", ["open"], "schedule")]
            result_tabs = st.tabs([f":material/{icon}: {label} ({table['status'].isin(statuses).sum()})"
                                   for label, statuses, icon in groups])
            for tab, (label, statuses, _) in zip(result_tabs, groups):
                with tab:
                    part = table[table["status"].isin(statuses)]
                    if part.empty:
                        st.caption(f"None {label.lower()} yet.")
                        continue
                    if label != "Still open":
                        # scored at the model's suggested stakes (your real bets are on the Profit & loss tab)
                        st.caption(f"{len(part)} bet(s) · {'made' if label == 'Won' else 'total'} "
                                   f"{cad(part['profit'].sum())} on {cad(part['stake'].sum())} bet")
                    else:
                        st.caption(f"{len(part)} bet(s) waiting on their games - press **Grade finished bets** "
                                   "after the games end.")
                    st.dataframe(part[columns], hide_index=True, width="stretch", column_config=config)

# ---------------------------------------------------------------- end of page
show_undo_banner()   # last, so it also shows a bet you placed during this page load
