import itertools
import json
import math
import os
import requests
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from Bet_Tracker import save_found
from NFL_Live import clean_name

# MLB model - the same strict recipe as the other sports, with baseball's big extra: WHO IS PITCHING.
#   batters:  hits, home runs, RBIs  - his recent form + the opposing STARTING PITCHER's recent form
#   pitchers: strikeouts (starters)  - his recent starts + how much the other lineup strikes out
#   moneyline: runs scored / allowed + both starting pitchers
# Every stat is a whole-number count, so every prop uses a Poisson curve.

# ---- your settings (same strictness as the other sports) ----
BANKROLL = 100        # your betting money in Canadian dollars (CA$)
MIN_EDGE = 0.05       # only bet when the blended chance shows at least +5% expected profit
MODEL_WEIGHT = 0.35   # final chance = 35% model + 65% the books' consensus
TRUST_LOW, TRUST_HIGH = 0.40, 0.60   # player props: only trust the model's chances in this range
# backtest on 2026 games: hits have honest percentages; home runs are no better than guessing, and RBIs run
# too high (says 36%, happens 29%) - so those two can't become bets until the backtest improves
TRUSTED_PROPS = {"strikeouts", "hits"}
# backtest: the moneyline model barely beats "always pick the home team" (0.2485 vs 0.2493) - baseball games are
# close to coin flips - so its win chances are shown in Face-offs but never become bets
TRUST_MONEYLINE = False
MAX_DISAGREE = 0.15   # model and book more than 15 points apart -> skip
MAX_RATIO = 1.6       # model's chance more than 1.6x the book's -> skip
MIN_PRICE = 1.5       # no favourites shorter than -200: risking $2+ to win $1 means one loss wipes out two wins
MAX_PRICE = 2.5       # no long shots longer than +150 - on your graded picks, +350 and longer won 10% (needed 18%)
OVER_EXTRA_EDGE = 0.05  # Overs need 5% MORE edge than Unders: books shade Overs because casual bettors love them
                        # (your first graded week: Unders won 52%, Overs 36%)
MAX_BETS = 8          # never more than 8 single bets a day
MAX_PER_GAME = 2      # at most 2 bets on the same game
SHOW_BACKTEST = False # True = print each model's backtest

ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
if not os.path.exists("mlb_players.csv") or not os.path.exists("mlb_games.csv"):
    raise SystemExit("No MLB stats yet - run: python MLB_Scraper.py (the first download takes ~30 minutes)")
df = pd.read_csv("mlb_players.csv").sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
df = df[~df["team"].isin(["AL", "NL"])].reset_index(drop=True)   # the All-Star Game isn't a real game
bat = df[df["role"] == "batter"].reset_index(drop=True)
pit = df[(df["role"] == "pitcher") & (df["starter"] == 1)].reset_index(drop=True)   # starting pitchers only


def rolling_means(frame, stats, windows, min_periods=None):
    """each row gets the player's average over his previous n games (shift = no peeking at this game)"""
    for stat in stats:
        previous = frame.groupby("player_id")[stat].shift()
        for n in windows:
            frame[f"{stat}_last{n}"] = previous.groupby(frame["player_id"]).rolling(
                n, min_periods=min_periods or n).mean().reset_index(level=0, drop=True)


def rolling_rate(frame, top, bottom, n=5):
    """his previous n starts: total `top` / total `bottom` (e.g. hits allowed per inning)"""
    prev_top = frame.groupby("player_id")[top].shift()
    prev_bottom = frame.groupby("player_id")[bottom].shift()
    top_sum = prev_top.groupby(frame["player_id"]).rolling(n, min_periods=3).sum().reset_index(level=0, drop=True)
    bottom_sum = prev_bottom.groupby(frame["player_id"]).rolling(n, min_periods=3).sum().reset_index(level=0, drop=True)
    return top_sum / bottom_sum.where(bottom_sum > 0)


# ---- batter features ----
BAT_STATS = ["hits", "home_runs", "rbis"]
BAT_OPPORTUNITY = ["at_bats", "walks", "batter_k", "bat_order"]
rolling_means(bat, BAT_STATS + BAT_OPPORTUNITY, (5, 10, 20))

# the starting pitcher he's facing, and how that pitcher's last 5 starts went (per inning)
SP_RATES = {"sp_hits_ip": "hits_allowed", "sp_hr_ip": "hr_allowed", "sp_k_ip": "strikeouts",
            "sp_walks_ip": "walks_allowed", "sp_runs_ip": "runs_allowed"}
for name, column in SP_RATES.items():
    pit[name] = rolling_rate(pit, column, "innings")
pit["sp_innings_last5"] = pit.groupby("player_id")["innings"].transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
starters = pit[["game_id", "team"] + list(SP_RATES) + ["sp_innings_last5"]].drop_duplicates(["game_id", "team"])
bat = bat.merge(starters.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in SP_RATES},
                                         "sp_innings_last5": "opp_sp_innings_last5"}),
                on=["game_id", "opponent"], how="left")

# the other team's whole pitching staff (bullpen too): hits it allowed per game, last 10 games
team_bat = bat.groupby(["team", "opponent", "date", "game_id"])[["hits", "home_runs", "batter_k"]].sum().reset_index()
allowed = team_bat.rename(columns={"team": "batting_team", "opponent": "team"}).sort_values(["team", "date"])
for stat in ["hits", "home_runs"]:
    allowed[f"staff_{stat}_allowed"] = allowed.groupby("team")[stat].transform(
        lambda s: s.shift().rolling(10, min_periods=5).mean())
bat = bat.merge(allowed[["game_id", "team", "staff_hits_allowed", "staff_home_runs_allowed"]].rename(
    columns={"team": "opponent"}), on=["game_id", "opponent"], how="left")

bat_features = [f"{s}_last{n}" for s in BAT_STATS + BAT_OPPORTUNITY for n in (5, 10, 20)] + \
               [f"opp_{c}" for c in SP_RATES] + ["opp_sp_innings_last5", "staff_hits_allowed",
                                                 "staff_home_runs_allowed", "home"]

# ---- pitcher features (starting pitchers) ----
PIT_STATS = ["strikeouts", "innings", "pitches"]
rolling_means(pit, PIT_STATS, (3, 5, 10), min_periods=None)
# how often the lineup he's facing strikes out: that team's batter strikeouts per game, last 10 games
lineup_k = team_bat.sort_values(["team", "date"]).copy()
lineup_k["lineup_k_last10"] = lineup_k.groupby("team")["batter_k"].transform(
    lambda s: s.shift().rolling(10, min_periods=5).mean())
pit = pit.merge(lineup_k[["game_id", "team", "lineup_k_last10"]].rename(columns={"team": "opponent"}),
                on=["game_id", "opponent"], how="left")
pit_features = [f"{s}_last{n}" for s in PIT_STATS for n in (3, 5, 10)] + ["sp_k_ip", "lineup_k_last10", "home"]

# who each model learns from / bets on
TARGETS = {"hits": (bat, bat_features), "home_runs": (bat, bat_features), "rbis": (bat, bat_features),
           "strikeouts": (pit, pit_features)}


def usable(frame, stat):
    """batters: everyday players in the starting lineup. pitchers: starters with 3+ previous starts"""
    if stat == "strikeouts":
        return frame.dropna(subset=["strikeouts_last3"])
    return frame[(frame["starter"] == 1) & (frame["at_bats_last10"] >= 2.5)].dropna(subset=["hits_last10"])


# ---- one strict model per stat (learn 2025, test 2026, then retrain on everything) ----
models = {}
for stat, (frame, feats) in TARGETS.items():
    data = usable(frame, stat)
    train, test = data[data["season"] <= 2025], data[data["season"] == 2026]

    def new_model():
        return HistGradientBoostingRegressor(loss="poisson", random_state=0, learning_rate=0.05, max_iter=300,
                                             max_depth=4, min_samples_leaf=100, l2_regularization=1.0)

    if SHOW_BACKTEST and len(test):
        model = new_model().fit(train[feats], train[stat])
        # a pretend sportsbook line: his recent average rounded to the nearest .5 (at least 0.5)
        avg_col = "strikeouts_last5" if stat == "strikeouts" else f"{stat}_last10"
        test = test.dropna(subset=[avg_col])
        predicted = model.predict(test[feats])
        avg = test[avg_col]
        line = ((avg * 2).round() / 2 + 0.5 * ((avg * 2).round() % 2 == 0)).clip(lower=0.5)
        chance = 1 - poisson.cdf(line.astype(int), predicted)
        hit = (test[stat] > line).values
        base = (avg - test[stat]).abs().mean()
        err = (predicted - test[stat]).abs().mean()
        print(f"\n########## {stat.upper()} ##########  ({len(train)} to learn from, {len(test)} to test)")
        print(f"model {(1 - err / base) * 100:+.1f}% better than the last-10 average | "
              f"chance-error {((chance - hit) ** 2).mean():.4f} vs always-{hit.mean():.0%} "
              f"{((hit.mean() - hit) ** 2).mean():.4f}")
        check = pd.DataFrame({"predicted": chance, "actual": hit, "bucket": pd.cut(chance, [0, .3, .4, .5, .6, .7, 1])})
        print(check.groupby("bucket", observed=True).agg(games=("actual", "size"), predicted=("predicted", "mean"),
                                                          actually_went_over=("actual", "mean")).round(2).to_string())

    models[stat] = new_model().fit(data[feats], data[stat])

# ---- face-offs: who wins (moneyline model) ----
games = pd.read_csv("mlb_games.csv")
teams_list = pd.read_csv("mlb_teams.csv")
done = games[games["completed"] & games["home_score"].notna()]
rows = []
for side, other in (("home", "away"), ("away", "home")):
    part = done[["game_id", "season", "date", f"{side}_team", f"{other}_team", f"{side}_score", f"{other}_score"]]
    part.columns = ["game_id", "season", "date", "team", "opponent", "runs_for", "runs_against"]
    rows.append(part.assign(home=int(side == "home")))
team_games = pd.concat(rows).sort_values(["team", "date", "game_id"])
team_games["win"] = (team_games["runs_for"] > team_games["runs_against"]).astype(int)
for col in ["runs_for", "runs_against"]:
    team_games[f"{col}_last10"] = team_games.groupby("team")[col].transform(
        lambda s: s.shift().rolling(10, min_periods=5).mean())
team_games["game_id"] = team_games["game_id"].astype(str)
starters["game_id"] = starters["game_id"].astype(str)
team_games = team_games.merge(starters[["game_id", "team", "sp_runs_ip"]], on=["game_id", "team"], how="left")
league_sp = pit["sp_runs_ip"].median()   # unknown starter -> an average one
team_games["sp_runs_ip"] = team_games["sp_runs_ip"].fillna(league_sp)
team_cols = ["runs_for_last10", "runs_against_last10", "sp_runs_ip"]
opp_view = team_games[["game_id", "team"] + team_cols].rename(
    columns={"team": "opponent", **{c: f"opp_{c}" for c in team_cols}})
team_games = team_games.merge(opp_view, on=["game_id", "opponent"]).dropna()
ml_features = team_cols + [f"opp_{c}" for c in team_cols] + ["home"]
ml_model = LogisticRegression(C=0.5, max_iter=1000)
ml_model.fit(team_games[team_games["season"] <= 2025][ml_features], team_games[team_games["season"] <= 2025]["win"])
if SHOW_BACKTEST:
    t = team_games[(team_games["season"] == 2026) & (team_games["home"] == 1)]
    p = ml_model.predict_proba(t[ml_features])[:, 1]
    base = team_games[(team_games["season"] <= 2025) & (team_games["home"] == 1)]["win"].mean()
    print(f"\nMONEYLINE: chance-error model {((p - t['win']) ** 2).mean():.4f} vs always-home {((base - t['win']) ** 2).mean():.4f}")
ml_model.fit(team_games[ml_features], team_games["win"])

# ---- tonight's odds ----
if not os.path.exists("mlb_odds.csv"):
    raise SystemExit("No mlb_odds.csv yet - run: python MLB_Odds_Scraper.py")
today = pd.Timestamp.now(tz="America/Toronto").strftime("%Y-%m-%d")
odds = pd.read_csv("mlb_odds.csv")
odds = odds[odds["game_date"] >= today]
if odds.empty:
    raise SystemExit("mlb_odds.csv has no games today - run: python MLB_Odds_Scraper.py")
# the game day these odds are for: usually today, but with no games left today the odds scraper gets tomorrow's
game_day = os.getenv("PQ_DAY") or odds["game_date"].min()   # PQ_DAY: Week_Run.py asks for one day of the week
odds = odds[odds["game_date"] == game_day]
if odds.empty:
    raise SystemExit(f"No odds for {game_day}.")

TEAMS = {clean_name(n): t for n, t in zip(teams_list["name"], teams_list["team"])}   # "Toronto Blue Jays" -> "TOR"
if clean_name("Athletics") in TEAMS:   # the A's: some books still call them Oakland
    TEAMS[clean_name("Oakland Athletics")] = TEAMS[clean_name("Athletics")]
matchups = {}
for game, date in odds[["game", "game_date"]].drop_duplicates().itertuples(index=False):
    away_name, home_name = game.split(" @ ")
    away, home = TEAMS.get(clean_name(away_name)), TEAMS.get(clean_name(home_name))
    if away and home:
        label = f"{away} @ {home}"
        matchups[home] = (away, 1, label, date)
        matchups[away] = (home, 0, label, date)

# the game day's probable starting pitchers, from ESPN
probable = {}   # team -> pitcher's clean name
for e in requests.get(f"{ESPN}/scoreboard", params={"dates": game_day.replace("-", "")}, timeout=15).json().get("events", []):
    for t in e["competitions"][0]["competitors"]:
        if t.get("probables"):
            probable[t["team"]["abbreviation"]] = clean_name(t["probables"][0]["athlete"]["displayName"])

# every starting pitcher's form going into today (his last 3 / 5 / 10 starts)
pit_now = pit.groupby("player_id").agg(player=("player", "last"), team=("team", "last"))
starts = pit.groupby("player_id").size()
for n in (3, 5, 10):
    last_n = pit.groupby("player_id").tail(n).groupby("player_id")[PIT_STATS].mean()
    for stat in PIT_STATS:
        pit_now[f"{stat}_last{n}"] = last_n[stat].where(starts >= n)
last5 = pit.groupby("player_id").tail(5).groupby("player_id")
for name, column in SP_RATES.items():
    pit_now[name] = last5[column].sum() / last5["innings"].sum().where(last5["innings"].sum() > 0)
pit_now["sp_innings_last5"] = last5["innings"].mean()
pit_now["name_key"] = pit_now["player"].map(clean_name)
pit_by_name = pit_now.drop_duplicates("name_key").set_index("name_key")


def starter_form(team):
    """today's starting pitcher for a team (None if ESPN hasn't named one / we don't know him)"""
    key = probable.get(team)
    return pit_by_name.loc[key] if key in pit_by_name.index else None


team_now = team_games.groupby("team")[["runs_for", "runs_against"]].apply(lambda t: t.tail(10).mean())
team_now.columns = ["runs_for_last10", "runs_against_last10"]


def ml_row(team, opp, home):
    def sp(t):
        form = starter_form(t)
        return league_sp if form is None or pd.isna(form["sp_runs_ip"]) else form["sp_runs_ip"]
    row = {"runs_for_last10": team_now.loc[team, "runs_for_last10"],
           "runs_against_last10": team_now.loc[team, "runs_against_last10"], "sp_runs_ip": sp(team),
           "opp_runs_for_last10": team_now.loc[opp, "runs_for_last10"],
           "opp_runs_against_last10": team_now.loc[opp, "runs_against_last10"], "opp_sp_runs_ip": sp(opp)}
    return pd.DataFrame([row | {"home": home}])[ml_features]


win_prob, faceoffs = {}, []
print("\n########## FACE-OFFS (who wins) ##########")
for team, (opp, home, label, date) in matchups.items():
    if home != 1 or team not in team_now.index or opp not in team_now.index:
        continue
    p = (ml_model.predict_proba(ml_row(team, opp, 1))[0, 1] + 1 - ml_model.predict_proba(ml_row(opp, team, 0))[0, 1]) / 2
    win_prob[team], win_prob[opp] = p, 1 - p
    pitchers = f"{probable.get(opp, '?').title()} vs {probable.get(team, '?').title()}"
    print(f"  {opp} {1 - p:4.0%}  @  {team} {p:4.0%}   ({pitchers})")
    faceoffs.append({"away": opp, "away_win": round(1 - p, 3), "home": team, "home_win": round(p, 3), "date": date,
                     "away_pitcher": probable.get(opp, "?").title(), "home_pitcher": probable.get(team, "?").title()})
pd.DataFrame(faceoffs).to_csv("mlb_faceoffs_today.csv", index=False)

# ---- every batter's form going into today ----
recent = bat.groupby("player_id")
form = pd.DataFrame({"player": recent["player"].last(), "team": recent["team"].last(),
                     "last_game": recent["date"].last(), "started_last": recent["starter"].last()})
played = recent.size()
for n in (5, 10, 20):
    last_n = bat.groupby("player_id").tail(n).groupby("player_id")[BAT_STATS + BAT_OPPORTUNITY].mean()
    for stat in BAT_STATS + BAT_OPPORTUNITY:
        form[f"{stat}_last{n}"] = last_n[stat].where(played >= n)
form = form[form["team"].isin(matchups)]
team_last = bat.groupby("team")["date"].max()
# in the lineup for his team's last game -> probably in it today (check the posted lineup before betting!)
form = form[(form["last_game"] == form["team"].map(team_last)) & (form["started_last"] == 1)
            & (form["at_bats_last10"] >= 2.5)]
form["opponent"] = form["team"].map(lambda t: matchups[t][0])
form["home"] = form["team"].map(lambda t: matchups[t][1])
for name in list(SP_RATES) + ["sp_innings_last5"]:
    form[f"opp_{name}"] = form["opponent"].map(
        lambda t: None if starter_form(t) is None else starter_form(t)[name])
staff_now = allowed.groupby("team")[["hits", "home_runs"]].apply(lambda t: t.tail(10).mean())
form["staff_hits_allowed"] = form["opponent"].map(staff_now["hits"])
form["staff_home_runs_allowed"] = form["opponent"].map(staff_now["home_runs"])
form = form.astype({c: float for c in bat_features})
for stat in BAT_STATS:
    form[f"predicted_{stat}"] = models[stat].predict(form[bat_features]) if len(form) else []
form["name_key"] = form["player"].map(clean_name)
players = form.drop_duplicates("name_key").set_index("name_key")

# today's starting pitchers: strikeouts
pitchers = {}
lineup_now = lineup_k.groupby("team")["batter_k"].apply(lambda s: s.tail(10).mean())
for team, (opp, home, label, date) in matchups.items():
    sp = starter_form(team)
    if sp is None or pd.isna(sp["strikeouts_last3"]):
        continue
    row = {c: sp[c] for c in pit_features if c in sp.index} | {"lineup_k_last10": lineup_now.get(opp), "home": home}
    pitchers[probable[team]] = (models["strikeouts"].predict(pd.DataFrame([row])[pit_features].astype(float))[0], team)

# ---- compare with the books ----
odds["implied"] = 1 / odds["price"]
odds["player"] = odds["player"].fillna("")
pair = ["book", "game", "market", "player", "line"]
odds["book_chance"] = odds["implied"] / odds.groupby(pair)["implied"].transform("sum")   # remove the book's cut
odds["book_chance"] = odds.groupby(["game", "market", "player", "line", "side"])["book_chance"].transform("mean")
STAT_OF = {"pitcher_strikeouts": "strikeouts", "batter_hits": "hits", "batter_home_runs": "home_runs",
           "batter_rbis": "rbis", "h2h": "moneyline"}
odds["stat"] = odds["market"].map(STAT_OF)
odds = odds.dropna(subset=["stat"])
odds["sport"] = "mlb"


def model_view(row):
    if row["stat"] == "moneyline":
        team = TEAMS.get(clean_name(row["side"]))
        if team not in win_prob:
            return None
        return win_prob[team], win_prob[team], team, matchups[team][2]
    key = clean_name(row["player"])
    if row["stat"] == "strikeouts":
        if key not in pitchers:
            return None
        mean, team = pitchers[key]
    else:
        if key not in players.index:
            return None
        mean, team = players.loc[key, f"predicted_{row['stat']}"], players.loc[key, "team"]
    line = row["line"]
    over = 1 - poisson.cdf(math.floor(line), mean)
    push = poisson.pmf(line, mean) if line == int(line) else 0
    return (over if row["side"] in ("Over", "Yes") else 1 - over - push), mean, team, matchups[team][2]


views = [model_view(row) for _, row in odds.iterrows()]
odds["raw_chance"] = [v[0] if v else None for v in views]
odds["mean"] = [v[1] if v else None for v in views]
odds["team"] = [v[2] if v else None for v in views]
odds["game"] = [v[3] if v else None for v in views]
odds = odds.dropna(subset=["raw_chance"])
odds["sd"] = 0.0   # every MLB stat is a count (Poisson) - no bell-curve width needed
odds["model_chance"] = MODEL_WEIGHT * odds["raw_chance"] + (1 - MODEL_WEIGHT) * odds["book_chance"]
odds["ev"] = odds["model_chance"] * odds["price"] - 1
odds["player"] = [p if p else s for p, s in zip(odds["player"], odds["side"])]
LABEL = {"strikeouts": "strikeouts", "hits": "hits", "home_runs": "home runs", "rbis": "RBIs"}
odds["bet"] = ["to win (moneyline)" if s == "moneyline" else f"{side} {line:g} {LABEL[s]}"
               for s, side, line in zip(odds["stat"], odds["side"], odds["line"])]

is_ml = odds["stat"] == "moneyline"
good = odds[
    (odds["ev"] >= MIN_EDGE)
    & ((odds["raw_chance"] - odds["book_chance"]).abs() <= MAX_DISAGREE)
    & (odds["raw_chance"] <= odds["book_chance"] * MAX_RATIO)
    & odds["price"].between(MIN_PRICE, MAX_PRICE)                             # risk : reward between 2:1 and 1:1.5
    & ((odds["ev"] >= MIN_EDGE + OVER_EXTRA_EDGE) | ~odds["side"].isin(["Over", "Yes"]))   # Overs need a bigger edge
    & (is_ml | odds["raw_chance"].between(TRUST_LOW, TRUST_HIGH))
    & (is_ml | odds["stat"].isin(TRUSTED_PROPS))   # only the prop types that passed the backtest
    & (~is_ml | (odds["raw_chance"] <= 0.85))
    & (~is_ml | TRUST_MONEYLINE)
].copy()
good["stake"] = (((good["model_chance"] * good["price"] - 1) / (good["price"] - 1)) / 4).clip(upper=0.02) * BANKROLL
# lowest odds still worth it (Overs need the bigger edge) - if your app shows less than this, skip it
good["min_price"] = ((1 + MIN_EDGE + OVER_EXTRA_EDGE * good["side"].isin(["Over", "Yes"])) / good["model_chance"]).clip(
    lower=MIN_PRICE)

singles = good.sort_values("price", ascending=False).drop_duplicates(["player", "market", "side", "line"])
singles = singles.sort_values("ev", ascending=False).drop_duplicates(["player", "market"])
singles = pd.concat([singles[singles["stat"] != "moneyline"],
                     singles[singles["stat"] == "moneyline"].drop_duplicates("game")]).sort_values("ev", ascending=False)
found_before_caps = len(singles)
singles = singles.drop_duplicates("player").groupby("game").head(MAX_PER_GAME).head(MAX_BETS)
if found_before_caps > len(singles):
    print(f"\n{found_before_caps} bets passed the checks - keeping the best {len(singles)}")

leg_columns = ["sport", "player", "team", "game", "game_date", "stat", "side", "line", "bet", "price", "mean", "sd"]
print(f"\n########## MLB BETS TO PLACE ({len(singles)}) ##########")
for _, b in singles.iterrows():
    print(f"  {b['player']} ({b['team']}) - {b['bet']} at {b['book']} {b['price']:.2f} | "
          f"model {b['raw_chance']:.0%} blend {b['model_chance']:.0%} book {b['book_chance']:.0%} | ev {b['ev']:+.0%}")
singles[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev", "stake", "min_price"]].to_csv(
    "mlb_bets_today.csv", index=False)
odds[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev"]].to_csv("mlb_lines_today.csv", index=False)

# ---- value parlays (one book, different games) ----
parlays = []
for book, legs_at_book in good.groupby("book"):
    legs_at_book = legs_at_book[~((legs_at_book["stat"] == "moneyline") & (legs_at_book["model_chance"] < 0.60))]
    legs_at_book = legs_at_book.sort_values("ev", ascending=False).drop_duplicates("player").head(10)
    for size in (2, 3):
        for combo in itertools.combinations(legs_at_book.to_dict("records"), size):
            if len({leg["game"] for leg in combo}) < size:
                continue
            chance = math.prod(leg["model_chance"] for leg in combo)
            price = math.prod(leg["price"] for leg in combo)
            if chance >= 0.15:
                parlays.append({"book": book, "legs": combo, "chance": chance, "price": price, "ev": chance * price - 1})
# keep the best 2 of EACH size, so 3-leg parlays always get a spot
best = [p for size in (2, 3)
        for p in sorted([q for q in parlays if len(q["legs"]) == size], key=lambda q: q["ev"], reverse=True)[:2]]
for p in best:
    p["stake"] = min(((p["chance"] * p["price"] - 1) / (p["price"] - 1)) / 4, 0.01) * BANKROLL
pd.DataFrame([{"book": p["book"], "price": p["price"], "chance": p["chance"], "ev": p["ev"], "stake": p["stake"],
               "legs": json.dumps([{c: leg[c] for c in leg_columns} for leg in p["legs"]], default=float)}
              for p in best], columns=["book", "price", "chance", "ev", "stake", "legs"]).to_csv(
    "mlb_parlays_today.csv", index=False)
print(f"{len(best)} MLB value parlays")

found = [{"legs": [{c: row[c] for c in leg_columns}], "book": row["book"], "price": row["price"],
          "stake": row["stake"], "game_date": row["game_date"]} for _, row in singles.iterrows()]
found += [{"legs": [{c: leg[c] for c in leg_columns} for leg in p["legs"]], "book": p["book"], "price": p["price"],
           "stake": p["stake"], "game_date": min(leg["game_date"] for leg in p["legs"])} for p in best]
save_found(today, found)
