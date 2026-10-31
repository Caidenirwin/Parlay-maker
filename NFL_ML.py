import itertools
import json
import math
import os
import pandas as pd
from scipy.stats import norm, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from Bet_Tracker import save_found
from NFL_Live import clean_name

# ---- your settings (same idea as ML.py, a bit stricter - the NFL is the sharpest betting market) ----
BANKROLL = 100        # your betting money in Canadian dollars (CA$)
MIN_EDGE = 0.05       # only bet when the blended chance shows at least +5% expected profit
MODEL_WEIGHT = 0.35   # final chance = 35% model + 65% the books' consensus - same as the NHL model
TRUST_LOW, TRUST_HIGH = 0.40, 0.60   # backtest: the model's player-prop chances are only honest in this range so far
MAX_DISAGREE = 0.15   # model and book more than 15 points apart -> the book probably knows something (injury...)
MAX_RATIO = 1.6       # model's chance more than 1.6x the book's -> skip
MIN_PRICE = 1.5       # no favourites shorter than -200: risking $2+ to win $1 means one loss wipes out two wins
MAX_PRICE = 2.5       # no long shots longer than +150 - on your graded picks, +350 and longer won 10% (needed 18%)
OVER_EXTRA_EDGE = 0.05  # Overs need 5% MORE edge than Unders: books shade Overs because casual bettors love them
                        # (your first graded week: Unders won 52%, Overs 36%)
MAX_BETS = 8          # never more than 8 single bets a week
MAX_PER_GAME = 2      # at most 2 bets on the same game
SHOW_BACKTEST = False # True = print each model's backtest tables

df = pd.read_csv("nfl_players.csv")
df = df.sort_values(["player_id", "season", "week"])

# what we'll predict (the stats books offer props on)
targets_to_predict = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
# "opportunity" stats: how much he gets used - like shot attempts in hockey
opportunity = ["attempts", "carries", "targets"]

# ---- Step 3: features ----
# average over the player's PREVIOUS games (shift = skip this game, rolling = last N games)
# last 3 / 5 / 8 games (tested: adding last-8 helped passing and rushing yards)
all_games = df.copy()   # every game, kept for working out each player's form before this week
for stat in targets_to_predict + opportunity:
    for n in (3, 5, 8):
        df[f"{stat}_last{n}"] = df.groupby("player_id")[stat].transform(lambda s: s.shift().rolling(n).mean())

# ---- opponent defence: how much has this week's opponent been ALLOWING lately? ----
# add up everything the players facing each team gained against it, per game, then average its previous 6 games
# (min_periods=3: early in a season use what there is, as long as it's at least 3 games)
allowed = df.groupby(["opponent_team", "season", "week"])[targets_to_predict].sum().reset_index()
allowed = allowed.rename(columns={"opponent_team": "team"}).sort_values(["team", "season", "week"])
for stat in targets_to_predict:
    allowed[f"opp_{stat}_allowed"] = allowed.groupby("team")[stat].transform(
        lambda s: s.shift().rolling(6, min_periods=3).mean())
df = df.merge(allowed[["team", "season", "week"] + [f"opp_{s}_allowed" for s in targets_to_predict]],
              left_on=["opponent_team", "season", "week"], right_on=["team", "season", "week"],
              how="left", suffixes=("", "_drop")).drop(columns="team_drop")

# position as numbers (models need numbers, not "WR"): 1 if he's that position, else 0
for pos in ["QB", "RB", "WR", "TE"]:
    df[f"is_{pos}"] = (df["position"] == pos).astype(int)

features = [f"{s}_last{n}" for s in targets_to_predict + opportunity for n in (3, 5, 8)] + \
           [f"opp_{s}_allowed" for s in targets_to_predict] + ["is_QB", "is_RB", "is_WR", "is_TE"]

# each stat only learns from (and bets on) players who actually get used for it
USAGE = {"passing_yards": ("attempts_last3", 15), "rushing_yards": ("carries_last3", 5),
         "receiving_yards": ("targets_last3", 3), "receptions": ("targets_last3", 3)}

# ---- Step 4: one strict model per stat, backtested, then retrained on everything ----
models, spreads = {}, {}
for stat in targets_to_predict:
    usage_col, minimum = USAGE[stat]
    data = df[df[usage_col] >= minimum].dropna(subset=[f for f in features if not f.startswith("opp_")] + [stat])
    train = data[data["season"] <= 2025]   # learn from 2018-2025
    test = data[data["season"] == 2026]    # test on this season - games it has never seen

    # same cautious ("regularized") settings as the NHL model, so it doesn't get overconfident
    is_count = stat == "receptions"         # receptions are counts (0, 1, 2...) -> Poisson; yards are not

    def new_model():
        return HistGradientBoostingRegressor(loss="poisson" if is_count else "squared_error", random_state=0,
                                             learning_rate=0.05, max_iter=300, max_depth=4,
                                             min_samples_leaf=50, l2_regularization=1.0)

    model = new_model().fit(train[features], train[stat])
    predicted = model.predict(test[features])

    # how spread out yards are around the prediction (the bell curve's width) - measured HONESTLY:
    # train on 2018-2024, see how far off it is on 2025 (a season it didn't learn from)
    inner = new_model().fit(train[train["season"] <= 2024][features], train[train["season"] <= 2024][stat])
    last_season = train[train["season"] == 2025]
    spreads[stat] = (last_season[stat] - inner.predict(last_season[features])).std()

    if SHOW_BACKTEST:
        line = (test[f"{stat}_last5"] * 2).round() / 2 + 0.5 * ((test[f"{stat}_last5"] * 2).round() % 2 == 0)
        chance = 1 - (poisson.cdf(line.astype(int), predicted) if is_count else norm.cdf(line, predicted, spreads[stat]))
        hit = (test[stat] > line).values
        baseline_error = (test[f"{stat}_last3"] - test[stat]).abs().mean()
        model_error = (predicted - test[stat]).abs().mean()
        print(f"\n########## {stat.upper()} ##########  ({len(train)} to learn from, {len(test)} to test)")
        print(f"model {(1 - model_error / baseline_error) * 100:+.1f}% better than the last-3 average | "
              f"chance-error {((chance - hit) ** 2).mean():.4f} (50/50 guessing = 0.25)")
        check = pd.DataFrame({"predicted": chance, "actual": hit, "bucket": pd.cut(chance, [0, .3, .4, .5, .6, .7, 1])})
        print(check.groupby("bucket", observed=True).agg(games=("actual", "size"), predicted=("predicted", "mean"),
                                                          actually_went_over=("actual", "mean")).round(2).to_string())

    models[stat] = new_model().fit(data[features], data[stat])   # now learn from every game, 2018 to now

# ---- Step 5: face-offs - who wins the game (moneyline model) ----
games = pd.read_csv("nfl_games.csv")
done = games.dropna(subset=["home_score"])
# one row per team per game, from each team's point of view
rows = []
for side, other in (("home", "away"), ("away", "home")):
    part = done[["season", "week", "gameday", f"{side}_team", f"{other}_team", f"{side}_score", f"{other}_score"]]
    part.columns = ["season", "week", "gameday", "team", "opponent", "points_for", "points_against"]
    rows.append(part.assign(home=int(side == "home")))
team_games = pd.concat(rows).sort_values(["team", "season", "week"])
team_games = team_games[team_games["points_for"] != team_games["points_against"]]   # ties are too rare to learn from
team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
team_games["margin"] = team_games["points_for"] - team_games["points_against"]
team_cols = ["margin_last5", "points_for_last5", "points_against_last5"]
for col in ["margin", "points_for", "points_against"]:
    team_games[f"{col}_last5"] = team_games.groupby("team")[col].transform(
        lambda s: s.shift().rolling(5, min_periods=3).mean())
opp_view = team_games[["season", "week", "team"] + team_cols].rename(
    columns={"team": "opponent", **{c: f"opp_{c}" for c in team_cols}})
team_games = team_games.merge(opp_view, on=["season", "week", "opponent"]).dropna()
ml_features = team_cols + [f"opp_{c}" for c in team_cols] + ["home"]

ml_model = LogisticRegression(C=0.5, max_iter=1000)
ml_model.fit(team_games[team_games["season"] <= 2025][ml_features], team_games[team_games["season"] <= 2025]["win"])
if SHOW_BACKTEST:
    t = team_games[(team_games["season"] == 2026) & (team_games["home"] == 1)]
    p = ml_model.predict_proba(t[ml_features])[:, 1]
    base = team_games[(team_games["season"] <= 2025) & (team_games["home"] == 1)]["win"].mean()
    print(f"\nMONEYLINE: chance-error model {((p - t['win']) ** 2).mean():.4f} vs always-home {((base - t['win']) ** 2).mean():.4f}")
ml_model.fit(team_games[ml_features], team_games["win"])

# ---- Step 6: this week's odds ----
if not os.path.exists("nfl_odds.csv"):
    raise SystemExit("No nfl_odds.csv yet - run: python NFL_Odds_Scraper.py")
today = pd.Timestamp.now(tz="America/Toronto").strftime("%Y-%m-%d")
odds = pd.read_csv("nfl_odds.csv")
odds = odds[odds["game_date"] >= today]   # only games that haven't happened yet
if odds.empty:
    raise SystemExit("nfl_odds.csv has no upcoming games - run: python NFL_Odds_Scraper.py")

TEAMS = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
         "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
         "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
         "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
         "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC", "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA",
         "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
         "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
         "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
         "Washington Commanders": "WAS"}
# this week's games: "Kansas City Chiefs @ ..." -> (away abbrev, home abbrev, date)
matchups = {}   # team -> (opponent, home 1/0, "LAC @ KC", date)
for game, date in odds[["game", "game_date"]].drop_duplicates().itertuples(index=False):
    away_name, home_name = game.split(" @ ")
    away, home = TEAMS.get(away_name), TEAMS.get(home_name)
    if away and home:
        label = f"{away} @ {home}"
        matchups[home] = (away, 1, label, date)
        matchups[away] = (home, 0, label, date)

# face-offs: each team's last 5 games, compared
team_now = team_games.groupby("team")[["margin", "points_for", "points_against"]].apply(lambda t: t.tail(5).mean())
team_now.columns = team_cols


def ml_row(team, opp, home):
    row = {c: team_now.loc[team, c] for c in team_cols} | {f"opp_{c}": team_now.loc[opp, c] for c in team_cols}
    return pd.DataFrame([row | {"home": home}])[ml_features]


win_prob, faceoffs = {}, []
print("\n########## FACE-OFFS (who wins) ##########")
for team, (opp, home, label, date) in matchups.items():
    if home != 1 or team not in team_now.index or opp not in team_now.index:
        continue
    # ask from both sides and average, so the two chances add up to exactly 100%
    p = (ml_model.predict_proba(ml_row(team, opp, 1))[0, 1] + 1 - ml_model.predict_proba(ml_row(opp, team, 0))[0, 1]) / 2
    win_prob[team], win_prob[opp] = p, 1 - p
    print(f"  {opp} {1 - p:4.0%}  @  {team} {p:4.0%}   ({date})")
    faceoffs.append({"away": opp, "away_win": round(1 - p, 3), "home": team, "home_win": round(p, 3), "date": date})
pd.DataFrame(faceoffs).to_csv("nfl_faceoffs_today.csv", index=False)

# ---- Step 7: every player's form going INTO this week (no shift: all of these games are in the past) ----
recent = all_games.groupby("player_id")
form = pd.DataFrame({"player": recent["player_display_name"].last(), "team": recent["team"].last(),
                     "position": recent["position"].last(),
                     "last_game": recent.apply(lambda g: (g["season"].iloc[-1], g["week"].iloc[-1]))})
for stat in targets_to_predict + opportunity:
    for n in (3, 5, 8):
        form[f"{stat}_last{n}"] = recent[stat].apply(lambda s: s.tail(n).mean() if len(s) >= n else float("nan"))
form = form[form["team"].isin(matchups)]
# skip players who missed their team's most recent game (likely injured)
team_last = all_games.groupby("team").apply(lambda g: max(zip(g["season"], g["week"])))
form = form[form["last_game"] == form["team"].map(team_last)]
allowed_now = allowed.groupby("team")[targets_to_predict].apply(lambda t: t.tail(6).mean())
form["opponent"] = form["team"].map(lambda t: matchups[t][0])
for stat in targets_to_predict:
    form[f"opp_{stat}_allowed"] = form["opponent"].map(allowed_now[stat])
for pos in ["QB", "RB", "WR", "TE"]:
    form[f"is_{pos}"] = (form["position"] == pos).astype(int)
for stat in targets_to_predict:
    usage_col, minimum = USAGE[stat]
    ok = form[usage_col] >= minimum   # only players who get used for this stat
    form[f"predicted_{stat}"] = float("nan")
    if ok.any():
        form.loc[ok, f"predicted_{stat}"] = models[stat].predict(form.loc[ok, features])
form["name_key"] = form["player"].map(clean_name)
players = form.drop_duplicates("name_key").set_index("name_key")

# ---- Step 8: compare with the books ----
odds["implied"] = 1 / odds["price"]
odds["player"] = odds["player"].fillna("")   # moneylines have no player
pair = ["book", "game", "market", "player", "line"]
odds["book_chance"] = odds["implied"] / odds.groupby(pair)["implied"].transform("sum")   # remove the book's cut
# the MARKET's opinion = the average of all three books (cut removed). Comparing each book to this is what lets
# us spot one book paying more than the others for the same bet - shopping for the best line is a real edge.
odds["book_chance"] = odds.groupby(["game", "market", "player", "line", "side"])["book_chance"].transform("mean")
STAT_OF = {"player_pass_yds": "passing_yards", "player_rush_yds": "rushing_yards",
           "player_reception_yds": "receiving_yards", "player_receptions": "receptions", "h2h": "moneyline"}
odds["stat"] = odds["market"].map(STAT_OF)
odds = odds.dropna(subset=["stat"])
odds["sport"] = "nfl"


def model_view(row):
    """(raw model chance, predicted mean, team, game label) for one line - or None if the model can't price it"""
    if row["stat"] == "moneyline":
        team = TEAMS.get(row["side"])
        if team not in win_prob:
            return None
        return win_prob[team], win_prob[team], team, matchups[team][2]
    key = clean_name(row["player"])
    if key not in players.index or pd.isna(players.loc[key, f"predicted_{row['stat']}"]):
        return None
    mean, team = players.loc[key, f"predicted_{row['stat']}"], players.loc[key, "team"]
    line = row["line"]
    if row["stat"] == "receptions":
        over = 1 - poisson.cdf(math.floor(line), mean)
        push = poisson.pmf(line, mean) if line == int(line) else 0
    else:
        over, push = 1 - norm.cdf(line, mean, spreads[row["stat"]]), 0
    return (over if row["side"] == "Over" else 1 - over - push), mean, team, matchups[team][2]


views = [model_view(row) for _, row in odds.iterrows()]
odds["raw_chance"] = [v[0] if v else None for v in views]
odds["mean"] = [v[1] if v else None for v in views]
odds["team"] = [v[2] if v else None for v in views]
odds["game"] = [v[3] if v else None for v in views]
odds = odds.dropna(subset=["raw_chance"])
odds["sd"] = odds["stat"].map(spreads).fillna(0)
odds["model_chance"] = MODEL_WEIGHT * odds["raw_chance"] + (1 - MODEL_WEIGHT) * odds["book_chance"]   # blend
odds["ev"] = odds["model_chance"] * odds["price"] - 1
odds["player"] = [p if p else s for p, s in zip(odds["player"], odds["side"])]   # moneyline -> team name
LABEL = {"passing_yards": "passing yards", "rushing_yards": "rushing yards",
         "receiving_yards": "receiving yards", "receptions": "receptions"}
odds["bet"] = ["to win (moneyline)" if s == "moneyline" else f"{side} {line:g} {LABEL[s]}"
               for s, side, line in zip(odds["stat"], odds["side"], odds["line"])]

is_ml = odds["stat"] == "moneyline"
good = odds[
    (odds["ev"] >= MIN_EDGE)
    & ((odds["raw_chance"] - odds["book_chance"]).abs() <= MAX_DISAGREE)
    & (odds["raw_chance"] <= odds["book_chance"] * MAX_RATIO)
    & odds["price"].between(MIN_PRICE, MAX_PRICE)                             # risk : reward between 2:1 and 1:1.5
    & ((odds["ev"] >= MIN_EDGE + OVER_EXTRA_EDGE) | ~odds["side"].isin(["Over", "Yes"]))   # Overs need a bigger edge
    & (is_ml | odds["raw_chance"].between(TRUST_LOW, TRUST_HIGH))   # props: only where the backtest says it's honest
    & (~is_ml | (odds["raw_chance"] <= 0.85))
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
print(f"\n########## NFL BETS TO PLACE ({len(singles)}) ##########")
for _, b in singles.iterrows():
    print(f"  {b['game_date']}  {b['player']} ({b['team']}) - {b['bet']} at {b['book']} {b['price']:.2f} | "
          f"model {b['raw_chance']:.0%} blend {b['model_chance']:.0%} book {b['book_chance']:.0%} | ev {b['ev']:+.0%}")
singles[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev", "stake", "min_price"]].to_csv(
    "nfl_bets_today.csv", index=False)
odds[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev"]].to_csv("nfl_lines_today.csv", index=False)

# ---- Step 9: value parlays (one book, different games) ----
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
    "nfl_parlays_today.csv", index=False)
print(f"{len(best)} NFL value parlays")

# remember everything it found (the dashboard's Fumbles page grades these too)
found = [{"legs": [{c: row[c] for c in leg_columns}], "book": row["book"], "price": row["price"],
          "stake": row["stake"], "game_date": row["game_date"]} for _, row in singles.iterrows()]
found += [{"legs": [{c: leg[c] for c in leg_columns} for leg in p["legs"]], "book": p["book"], "price": p["price"],
           "stake": p["stake"], "game_date": min(leg["game_date"] for leg in p["legs"])} for p in best]
save_found(today, found)
