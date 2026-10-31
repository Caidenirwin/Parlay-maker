import html
import itertools
import json
import math
import os
import re
import subprocess
import sys
import altair as alt   # chart library that comes with Streamlit - used for the profit/loss pop-up
import pandas as pd
import streamlit as st
from Bet_Tracker import (FOUND_FILE, SETTLED, add_bet, american_to_decimal, cash_out_bet, delete_bet, grade_bets,
                         load_bets)
from Live_Tracker import game_status, get_boxscore, get_scoreboard, live_bet
from NFL_Live import get_nfl_box, get_nfl_scoreboard, nfl_logo

# Start it with:   streamlit run dashboard.py
# The colours and font live in .streamlit/config.toml

st.set_page_config(page_title="Parlay Quant", page_icon=":material/sports_hockey:", layout="wide")
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # so the scripts and csv files are always found
TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")
LEG_COLUMNS = ["player", "team", "game", "stat", "side", "line", "bet", "price", "mean", "sd"]
EXTRA_LEG_COLUMNS = ["sport", "game_date"]   # NFL legs carry these (NHL legs default to nhl / today)
GREEN, RED, AMBER, BLUE, MUTED = "#22C55E", "#EF4444", "#F59E0B", "#3B82F6", "#94A3B8"


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


def leg_dict(row, extra=()):
    """the columns a saved bet leg needs, from a row of a bets / lines file"""
    return {c: row[c] for c in LEG_COLUMNS + EXTRA_LEG_COLUMNS + list(extra) if c in row.index and pd.notna(row[c])}


def updated_at(path):
    return pd.Timestamp.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None


def made_today(path, days=0):
    """made today - or, for NFL files (a week of games), within the last `days` days"""
    when = updated_at(path)
    return when is not None and (pd.Timestamp.today().normalize() - when.normalize()).days <= days


def run_script(script, label, force=False):
    """run one of your scripts like typing 'python script' in the terminal, and show what it printed.
    force=True adds --force: download fresh even if it already did today"""
    with st.status(label + "...", expanded=False) as box:
        command = [sys.executable, script] + (["--force"] if force else [])
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        st.code((result.stdout + result.stderr)[-5000:] or "(nothing printed)")
        ok = result.returncode == 0
        box.update(label=f"{label}: {'done' if ok else 'stopped - open for details'}",
                   state="complete" if ok else "error")
    return ok


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


def bet_card_html(team, title, subtitle, chips, sport="nhl"):
    """team logo + bet name + small info chips"""
    chip_html = "".join(f'<span class="chip {kind}">{html.escape(text)}</span>' for kind, text in chips)
    return (f'<div class="bet"><img src="{logo(team, sport)}"><div><div class="bet-title">{title}</div>'
            f'<div class="bet-sub">{html.escape(subtitle)}</div><div class="chips">{chip_html}</div></div></div>')


# ---------------------------------------------------------------- page
# ---- which sport the page shows (live tracking, profit/loss and fumbles always cover both) ----
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

live_ticker(SPORT)   # the scrolling score strip, always at the top
st.html(f'<div class="pq-header"><div><div class="pq-title">Parlay Quant</div>'
        f'<div class="pq-sub">{CFG["sub"]} · Ontario</div></div>'
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
             ("", f"Still bet it down to {american(b['min_price'])}")], b.get("sport", "nhl")))
        leg = leg_dict(b)
        place_bet_box(right, key, [leg], b["book"], b["price"], b["stake"])


def legs_html(legs):
    return "".join(f'<div class="leg"><img src="{logo(leg["team"], leg.get("sport", "nhl"))}"><div><b>{html.escape(str(leg["player"]))}</b> '
                   f'<span class="muted">· {html.escape(leg["bet"])} · {american(leg["price"])} · '
                   f'{html.escape(str(leg["game"]))}</span></div></div>' for leg in legs)


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


tonight_tab, singles_tab, parlays_tab, mega_tab, live_tab, profit_tab, fumbles_tab = st.tabs(
    [":material/target: Tonight", ":material/confirmation_number: Singles", ":material/layers: Parlays",
     ":material/stacks: Mega parlay", ":material/sensors: Live", ":material/account_balance_wallet: Profit & loss",
     ":material/sports_handball: Fumbles"])

# ---------------------------------------------------------------- tab 1: tonight
with tonight_tab:
    c1, c2, c3, c4 = st.columns(4)
    steps = CFG["steps"]
    # same day, no game started since the last download? the scrapers reuse it (no credits). Tick this to re-download.
    fresh = st.checkbox("Download fresh (uses API credits even if you already downloaded today)", key="force_fresh")
    if c1.button(f"Update {SPORT} stats", icon=":material/download:", width="stretch"):
        run_script(*steps[0], force=fresh)
    if c2.button("Get odds", icon=":material/request_quote:", width="stretch",
                 help="Uses API credits - unless you already got today's odds and no game has started since"):
        run_script(*steps[1], force=fresh)
    if c3.button("Find bets", icon=":material/insights:", width="stretch"):
        run_script(*steps[2])
    if c4.button("Run all", icon=":material/play_arrow:", type="primary", width="stretch"):
        for script, label in steps:
            if not run_script(script, label, force=fresh and script != steps[2][0]):
                break   # stop if a step failed (for example, no props posted yet)

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
            cards += (f'<div class="fo-card">{row(f["away"], f["away_win"], f["away_win"] > f["home_win"])}'
                      f'<div class="fo-bar"><div style="width:{f["away_win"]:.0%};background:#475569"></div>'
                      f'<div style="width:{f["home_win"]:.0%};background:{BLUE}"></div></div>'
                      f'{row(f["home"], f["home_win"], f["home_win"] >= f["away_win"])}'
                      + (f'<div class="fo-goalies">Pitchers: {html.escape(str(f["away_pitcher"]))} vs '
                         f'{html.escape(str(f["home_pitcher"]))}</div></div>'
                         if "away_pitcher" in f.index else
                         f'<div class="fo-goalies">{pd.Timestamp(f["date"]).strftime("%A %b %d")}</div></div>'
                         if "date" in f.index else
                         f'<div class="fo-goalies">Goalies: {html.escape(str(f["away_goalie"]))} vs '
                         f'{html.escape(str(f["home_goalie"]))}</div></div>'))
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
        st.caption("Built automatically from tonight's value bets: one book, different games.")
        if not made_today(CFG["parlays"], CFG["days"]):
            st.info("Run the model first (Tonight tab).", icon=":material/info:")
        else:
            parlays_today = pd.read_csv(CFG["parlays"])
            if parlays_today.empty:
                st.caption("No value parlays tonight - build your own on the right if you want one.")
            for i, p in parlays_today.iterrows():
                legs = json.loads(p["legs"])
                with st.container(border=True):
                    st.html(f'<div class="bet-title">{len(legs)}-leg parlay</div>'
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
                  f"need {breakeven:.0%} to break even" if breakeven is not None else None,
                  delta_color="off", delta_arrow="off", border=True,
                  help="How often your bets win, next to how often they NEED to win at the odds you took. "
                       "Beating that number is what makes money - a high win rate on short odds can still lose.")
        m3.metric("Record (W-L-P)", f"{wins}-{losses}-{pushes}",
                  f"{len(cashed)} cashed out ({cad(cashed['profit'].sum())})" if len(cashed) else None,
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
