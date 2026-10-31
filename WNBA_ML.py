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

# WNBA model - the same strict recipe as the hockey and football models, with basketball stats:
# points (bell curve), rebounds / assists / threes (whole-number counts -> Poisson), plus who wins each game.

# ---- your settings (same strictness as the other sports) ----
BANKROLL = 100        # your betting money in Canadian dollars (CA$)
MIN_EDGE = 0.05       # only bet when the blended chance shows at least +5% expected profit
MODEL_WEIGHT = 0.35   # final chance = 35% model + 65% the books' consensus
TRUST_LOW, TRUST_HIGH = 0.40, 0.60   # player props: only trust the model's chances in this range
TRUSTED_PROPS = {"points", "rebounds", "assists", "threes"}   # prop types allowed to become bets (see backtest)
MAX_DISAGREE = 0.15   # model and book more than 15 points apart -> skip
MAX_RATIO = 1.6       # model's chance more than 1.6x the book's -> skip
MIN_PRICE = 1.5       # no favourites shorter than -200: risking $2+ to win $1 means one loss wipes out two wins
MAX_PRICE = 2.5       # no long shots longer than +150 - on your graded picks, +350 and longer won 10% (needed 18%)
OVER_EXTRA_EDGE = 0.05  # Overs need 5% MORE edge than Unders: books shade Overs because casual bettors love them
                        # (your first graded week: Unders won 52%, Overs 36%)
MAX_BETS = 8          # never more than 8 single bets a night
MAX_PER_GAME = 2      # at most 2 bets on the same game
MIN_MINUTES = 20      # only players averaging 20+ minutes over their last 5 games (bench guys are coin flips)
SHOW_BACKTEST = False # True = print each model's backtest

if not os.path.exists("wnba_players.csv") or not os.path.exists("wnba_games.csv"):
    raise SystemExit("No WNBA stats yet - run: python WNBA_Scraper.py (the first download takes ~5 minutes)")
df = pd.read_csv("wnba_players.csv").sort_values(["player_id", "date"])
targets_to_predict = ["points", "rebounds", "assists", "threes"]
opportunity = ["minutes", "shots_tried", "threes_tried"]
COUNT_STATS = {"rebounds", "assists", "threes"}
WINDOWS = (3, 5, 10)
all_games = df.copy()

# ---- features: his previous 3 / 5 / 10 games (shift = no peeking at this game) ----
df = df.reset_index(drop=True)
for stat in targets_to_predict + opportunity:
    previous = df.groupby("player_id")[stat].shift()   # each game sees only the games before it
    for n in WINDOWS:
        df[f"{stat}_last{n}"] = previous.groupby(df["player_id"]).rolling(n).mean().reset_index(level=0, drop=True)
# rest: days since his last game (1 = second night of a back-to-back; capped at 5 = fully rested)
df["rest"] = pd.to_datetime(df["date"]).groupby(df["player_id"]).diff().dt.days.clip(1, 5).fillna(5)

# opponent defence: what tonight's opponent has been allowing over its previous 10 games
allowed = df.groupby(["opponent", "date"])[targets_to_predict].sum().reset_index()
allowed = allowed.rename(columns={"opponent": "team"}).sort_values(["team", "date"])
for stat in targets_to_predict:
    allowed[f"opp_{stat}_allowed"] = allowed.groupby("team")[stat].transform(
        lambda s: s.shift().rolling(10, min_periods=5).mean())
df = df.merge(allowed[["team", "date"] + [f"opp_{s}_allowed" for s in targets_to_predict]],
              left_on=["opponent", "date"], right_on=["team", "date"], how="left", suffixes=("", "_drop"))
df = df.drop(columns="team_drop")

features = [f"{s}_last{n}" for s in targets_to_predict + opportunity for n in WINDOWS] + \
           [f"opp_{s}_allowed" for s in targets_to_predict] + ["home", "rest"]

# ---- one strict model per stat (learn 2024 + 2025, test 2026, then retrain on everything) ----
models, spreads = {}, {}
for stat in targets_to_predict:
    data = df[df["minutes_last5"] >= MIN_MINUTES].dropna(subset=[f"{s}_last10" for s in targets_to_predict] + [stat])
    train, test = data[data["season"] <= 2025], data[data["season"] == 2026]
    is_count = stat in COUNT_STATS

    def new_model():
        return HistGradientBoostingRegressor(loss="poisson" if is_count else "squared_error", random_state=0,
                                             learning_rate=0.05, max_iter=300, max_depth=4,
                                             min_samples_leaf=50, l2_regularization=1.0)

    model = new_model().fit(train[features], train[stat])
    # bell-curve width measured honestly: how far off the 2024-2025 model was on 2026 games it never saw
    spreads[stat] = (test[stat] - model.predict(test[features])).std()

    if SHOW_BACKTEST and len(test):
        predicted = model.predict(test[features])
        # a pretend sportsbook line: his last-5 average rounded to the nearest .5
        line = (test[f"{stat}_last5"] * 2).round() / 2 + 0.5 * ((test[f"{stat}_last5"] * 2).round() % 2 == 0)
        chance = 1 - (poisson.cdf(line.astype(int), predicted) if is_count else norm.cdf(line, predicted, spreads[stat]))
        hit = (test[stat] > line).values
        base = (test[f"{stat}_last5"] - test[stat]).abs().mean()
        err = (predicted - test[stat]).abs().mean()
        print(f"\n########## {stat.upper()} ##########  ({len(train)} to learn from, {len(test)} to test)")
        print(f"model {(1 - err / base) * 100:+.1f}% better than the last-5 average | "
              f"chance-error {((chance - hit) ** 2).mean():.4f} (50/50 guessing = 0.25)")
        check = pd.DataFrame({"predicted": chance, "actual": hit, "bucket": pd.cut(chance, [0, .3, .4, .5, .6, .7, 1])})
        print(check.groupby("bucket", observed=True).agg(games=("actual", "size"), predicted=("predicted", "mean"),
                                                          actually_went_over=("actual", "mean")).round(2).to_string())

    models[stat] = new_model().fit(data[features], data[stat])

# ---- face-offs: who wins (moneyline model) ----
games = pd.read_csv("wnba_games.csv")
teams_list = pd.read_csv("wnba_teams.csv")
done = games[games["completed"] & games["home_score"].notna()]
rows = []
for side, other in (("home", "away"), ("away", "home")):
    part = done[["season", "date", f"{side}_team", f"{other}_team", f"{side}_score", f"{other}_score"]]
    part.columns = ["season", "date", "team", "opponent", "points_for", "points_against"]
    rows.append(part.assign(home=int(side == "home")))
team_games = pd.concat(rows).sort_values(["team", "date"])
team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
team_games["margin"] = team_games["points_for"] - team_games["points_against"]
team_games["rest"] = pd.to_datetime(team_games["date"]).groupby(team_games["team"]).diff().dt.days.clip(1, 5).fillna(5)
team_cols = ["margin_last10", "points_for_last10", "points_against_last10", "rest"]
for col in ["margin", "points_for", "points_against"]:
    team_games[f"{col}_last10"] = team_games.groupby("team")[col].transform(
        lambda s: s.shift().rolling(10, min_periods=5).mean())
opp_view = team_games[["date", "team"] + team_cols].rename(
    columns={"team": "opponent", **{c: f"opp_{c}" for c in team_cols}})
team_games = team_games.merge(opp_view, on=["date", "opponent"]).dropna()
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
if not os.path.exists("wnba_odds.csv"):
    raise SystemExit("No wnba_odds.csv yet - run: python WNBA_Odds_Scraper.py")
today = pd.Timestamp.now(tz="America/Toronto").strftime("%Y-%m-%d")
odds = pd.read_csv("wnba_odds.csv")
odds = odds[odds["game_date"] >= today]
if odds.empty:
    raise SystemExit("wnba_odds.csv has no games tonight - run: python WNBA_Odds_Scraper.py")
# the game day these odds are for: usually tonight, but with no game tonight the odds scraper gets tomorrow's
game_day = odds["game_date"].min()
odds = odds[odds["game_date"] == game_day]

TEAMS = {clean_name(n): t for n, t in zip(teams_list["name"], teams_list["team"])}   # "Boston Celtics" -> "BOS"
matchups = {}
for game, date in odds[["game", "game_date"]].drop_duplicates().itertuples(index=False):
    away_name, home_name = game.split(" @ ")
    away, home = TEAMS.get(clean_name(away_name)), TEAMS.get(clean_name(home_name))
    if away and home:
        label = f"{away} @ {home}"
        matchups[home] = (away, 1, label, date)
        matchups[away] = (home, 0, label, date)

tonight = pd.Timestamp(game_day)   # rest days count up to the game, not to today
team_now = team_games.groupby("team")[["margin", "points_for", "points_against"]].apply(lambda t: t.tail(10).mean())
team_now.columns = team_cols[:3]
team_now["rest"] = (tonight - pd.to_datetime(team_games.groupby("team")["date"].max())).dt.days.clip(1, 5)


def ml_row(team, opp, home):
    row = {c: team_now.loc[team, c] for c in team_cols} | {f"opp_{c}": team_now.loc[opp, c] for c in team_cols}
    return pd.DataFrame([row | {"home": home}])[ml_features]


win_prob, faceoffs = {}, []
print("\n########## FACE-OFFS (who wins) ##########")
for team, (opp, home, label, date) in matchups.items():
    if home != 1 or team not in team_now.index or opp not in team_now.index:
        continue
    p = (ml_model.predict_proba(ml_row(team, opp, 1))[0, 1] + 1 - ml_model.predict_proba(ml_row(opp, team, 0))[0, 1]) / 2
    win_prob[team], win_prob[opp] = p, 1 - p
    print(f"  {opp} {1 - p:4.0%}  @  {team} {p:4.0%}")
    faceoffs.append({"away": opp, "away_win": round(1 - p, 3), "home": team, "home_win": round(p, 3), "date": date})
pd.DataFrame(faceoffs).to_csv("wnba_faceoffs_today.csv", index=False)

# ---- every player's form going into tonight ----
recent = all_games.groupby("player_id")
form = pd.DataFrame({"player": recent["player"].last(), "team": recent["team"].last(),
                     "last_game": recent["date"].last()})
games_played = recent.size()
for n in WINDOWS:
    last_n = all_games.groupby("player_id").tail(n).groupby("player_id")[targets_to_predict + opportunity].mean()
    for stat in targets_to_predict + opportunity:
        # blank if he hasn't played n games yet, exactly like in training
        form[f"{stat}_last{n}"] = last_n[stat].where(games_played >= n)
form = form[form["team"].isin(matchups)]
team_last = all_games.groupby("team")["date"].max()
form = form[form["last_game"] == form["team"].map(team_last)]   # missed his team's last game -> skip (likely hurt)
form = form[form["minutes_last5"] >= MIN_MINUTES]
allowed_now = allowed.groupby("team")[targets_to_predict].apply(lambda t: t.tail(10).mean())
form["opponent"] = form["team"].map(lambda t: matchups[t][0])
form["home"] = form["team"].map(lambda t: matchups[t][1])
form["rest"] = (tonight - pd.to_datetime(form["last_game"])).dt.days.clip(1, 5)
for stat in targets_to_predict:
    form[f"opp_{stat}_allowed"] = form["opponent"].map(allowed_now[stat])
for stat in targets_to_predict:
    form[f"predicted_{stat}"] = models[stat].predict(form[features]) if len(form) else []
form["name_key"] = form["player"].map(clean_name)
players = form.drop_duplicates("name_key").set_index("name_key")

# ---- compare with the books (same as the football models) ----
odds["implied"] = 1 / odds["price"]
odds["player"] = odds["player"].fillna("")
pair = ["book", "game", "market", "player", "line"]
odds["book_chance"] = odds["implied"] / odds.groupby(pair)["implied"].transform("sum")   # remove the book's cut
odds["book_chance"] = odds.groupby(["game", "market", "player", "line", "side"])["book_chance"].transform("mean")
STAT_OF = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists",
           "player_threes": "threes", "h2h": "moneyline"}
odds["stat"] = odds["market"].map(STAT_OF)
odds = odds.dropna(subset=["stat"])
odds["sport"] = "wnba"


def model_view(row):
    if row["stat"] == "moneyline":
        team = TEAMS.get(clean_name(row["side"]))
        if team not in win_prob:
            return None
        return win_prob[team], win_prob[team], team, matchups[team][2]
    key = clean_name(row["player"])
    if key not in players.index or pd.isna(players.loc[key, f"predicted_{row['stat']}"]):
        return None
    mean, team = players.loc[key, f"predicted_{row['stat']}"], players.loc[key, "team"]
    line = row["line"]
    if row["stat"] in COUNT_STATS:
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
odds["model_chance"] = MODEL_WEIGHT * odds["raw_chance"] + (1 - MODEL_WEIGHT) * odds["book_chance"]
odds["ev"] = odds["model_chance"] * odds["price"] - 1
odds["player"] = [p if p else s for p, s in zip(odds["player"], odds["side"])]
odds["bet"] = ["to win (moneyline)" if s == "moneyline" else f"{side} {line:g} {s}"
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
print(f"\n########## WNBA BETS TO PLACE ({len(singles)}) ##########")
for _, b in singles.iterrows():
    print(f"  {b['player']} ({b['team']}) - {b['bet']} at {b['book']} {b['price']:.2f} | "
          f"model {b['raw_chance']:.0%} blend {b['model_chance']:.0%} book {b['book_chance']:.0%} | ev {b['ev']:+.0%}")
singles[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev", "stake", "min_price"]].to_csv(
    "wnba_bets_today.csv", index=False)
odds[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev"]].to_csv("wnba_lines_today.csv", index=False)

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
    "wnba_parlays_today.csv", index=False)
print(f"{len(best)} WNBA value parlays")

found = [{"legs": [{c: row[c] for c in leg_columns}], "book": row["book"], "price": row["price"],
          "stake": row["stake"], "game_date": row["game_date"]} for _, row in singles.iterrows()]
found += [{"legs": [{c: leg[c] for c in leg_columns} for leg in p["legs"]], "book": p["book"], "price": p["price"],
           "stake": p["stake"], "game_date": min(leg["game_date"] for leg in p["legs"])} for p in best]
save_found(today, found)
