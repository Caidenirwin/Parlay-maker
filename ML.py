import itertools
import json
import math
import os
import re
import unicodedata
import pandas as pd
import requests
from scipy.stats import norm, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from Goalie_Scraper import get_starting_goalies
from Bet_Tracker import save_found

# ---- your settings ----
BANKROLL = 100        # your betting money in Canadian dollars (CA$); stakes are worked out from this
MIN_EDGE = 0.05       # only bet when the model expects at least +5% profit per $1
MAX_DISAGREE = 0.15   # if the model and the book differ by more than 15 points, the book probably knows
                      # something the model doesn't (injury, lineup change) -> skip that bet
MAX_RATIO = 1.6       # ...or if the model's chance is more than 1.6x the book's (catches long shots: 10% vs 5%)
MIN_PRICE = 1.5       # no favourites shorter than -200: risking $2+ to win $1 means one loss wipes out two wins
MAX_PRICE = 2.5       # no long shots longer than +150 - on your graded picks, +350 and longer won 10% (needed 18%)
OVER_EXTRA_EDGE = 0.05  # Overs need 5% MORE edge than Unders: books shade Overs because casual bettors love them
                        # (your first graded week: Unders won 52%, Overs 36%)
MODEL_WEIGHT = 0.35   # final chance = 35% model + 65% the books' (cut removed). On real Oct 10 results the books were
                      # more accurate than the model alone, and a blend leaning on them was nearly as accurate as the
                      # books - so the model only bets when it disagrees enough to still show value after blending
MAX_BETS = 8          # never more than 8 single bets a night - the strongest ones by expected profit
MAX_PER_GAME = 2      # at most 2 bets on the same game (bets in one game win and lose together)
SHOW_DETAILS = False  # True = also print the backtests and every team's player list


def say(*args):
    """print, but only when SHOW_DETAILS is on - keeps the output to just the bets you should place"""
    if SHOW_DETAILS:
        print(*args)


def clean_name(name):
    """'Alexis Lafrenière' / 'J.T. Miller' -> 'alexis lafreniere' / 'jt miller', so names from different sites match"""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


df = pd.read_csv("skaters.csv")
say(df.shape)

# the stats we predict, and the betting lines to check for each
stats = ["shots", "points", "goals"]
lines = {"shots": [1.5, 2.5], "points": [0.5, 1.5], "goals": [0.5]}   # goals over 0.5 = "anytime goal scorer"

df = df.sort_values(["playerId", "gameDate"])

# ---- Step 2: features ----
# average over the player's PREVIOUS games (shift = skip tonight, rolling = last N games)
# shot attempts (shots + misses + blocked) are a feature too - how much he TRIES to shoot is steadier than shots on goal.
# last20 = a longer view of his form; blank for players with under 20 games (the model handles blanks)
feature_stats = stats + ["attempts"]
for stat in feature_stats:
    df[f"{stat}_prev"] = df.groupby("playerId")[stat].shift()
    df[f"{stat}_last5"] = df.groupby("playerId")[stat].transform(lambda s: s.shift().rolling(5).mean())
    df[f"{stat}_last10"] = df.groupby("playerId")[stat].transform(lambda s: s.shift().rolling(10).mean())
    df[f"{stat}_last20"] = df.groupby("playerId")[stat].transform(lambda s: s.shift().rolling(20).mean())
df["toi_last5"] = df.groupby("playerId")["toiMinutes"].transform(lambda s: s.shift().rolling(5).mean())
# power-play minutes over his previous 5 games (PP time = more open ice = more shots and points)
df["pp_last5"] = df.groupby("playerId")["ppMinutes"].transform(lambda s: s.shift().rolling(5).mean())

# ---- opponent: how many shots and goals has the opponent been ALLOWING lately? ----
# add up every shot and goal scored AGAINST each team in each game (by all the players facing them)
allowed = df.groupby(["opponentTeamAbbrev", "gameId", "gameDate"])[["shots", "goals"]].sum().reset_index()
allowed = allowed.rename(columns={"opponentTeamAbbrev": "team", "shots": "shots_allowed", "goals": "goals_allowed"})
allowed = allowed.sort_values(["team", "gameDate"])
# that team's average allowed over its PREVIOUS 10 games (shift again, so no peeking)
for col in ["shots_allowed", "goals_allowed"]:
    allowed[f"opp_{col}_last10"] = allowed.groupby("team")[col].transform(lambda s: s.shift().rolling(10).mean())

# attach it to each player's game: look up his opponent in that same game
df = df.merge(allowed[["team", "gameId", "opp_shots_allowed_last10", "opp_goals_allowed_last10"]],
              left_on=["opponentTeamAbbrev", "gameId"], right_on=["team", "gameId"], how="left")
df = df.drop(columns="team")

all_games = df.copy()   # every game, kept for working out each player's form before tonight

# ---- Step 3: drop rows with no history ----
# a player's first 10 games have blank (NaN) averages, so the model can't learn from them
features = [f"{stat}_{w}" for stat in feature_stats for w in ("prev", "last5", "last10", "last20")] + \
           ["toi_last5", "pp_last5", "home", "opp_shots_allowed_last10", "opp_goals_allowed_last10"]
before = len(df)
# remove rows where a needed column is blank (last20 is allowed to be blank - under 20 games of history)
df = df.dropna(subset=[f for f in features if not f.endswith("_last20")])
say("dropped", before - len(df), "rows with no history, kept", len(df))

# ---- Step 4: split into training games and test games ----
# learn from older games, then test on newer games the model has never seen
cutoff = "2026-02-01"
train = df[df["gameDate"] < cutoff]    # games before Feb 1 -> the model learns from these
test = df[df["gameDate"] >= cutoff]    # Feb 1 and later -> used only to check how good it is
say("train:", len(train), "games | test:", len(test), "games")

# ---- Steps 5-7: one model per stat, each tested against its own baseline ----
models = {}
for stat in stats:
    say(f"\n########## {stat.upper()} ##########")

    # Step 5: baseline - "he'll get his last-5 average tonight"
    baseline_error = (test[f"{stat}_last5"] - test[stat]).abs().mean()
    say("baseline: off by", round(baseline_error, 3), stat, "per game on average")

    # Step 6: train - gradient boosting, loss="poisson" because the answer is a count (0, 1, 2...)
    # the other settings make it more cautious so it doesn't get overconfident (regularization):
    #   learning_rate=0.05     -> each tree only fixes a small part of the error, so it learns slower and steadier
    #   max_depth=4            -> each tree can only ask 4 questions in a row, so no ultra-specific rules
    #   min_samples_leaf=200   -> every rule must be backed by at least 200 games, not a handful of lucky ones
    #   l2_regularization=1.0  -> penalizes extreme predictions, pulling them toward normal values
    model = HistGradientBoostingRegressor(loss="poisson", random_state=0, learning_rate=0.05, max_iter=300,
                                          max_depth=4, min_samples_leaf=200, l2_regularization=1.0)
    model.fit(train[features], train[stat])

    # Step 7: test it on games it has never seen
    predictions = model.predict(test[features])
    model_error = (predictions - test[stat]).abs().mean()
    say("model:    off by", round(model_error, 3), stat, "per game on average")
    say("model is", round((1 - model_error / baseline_error) * 100, 1), "% better than the baseline")

    # Step 7b: backtest the chances - when it says 60%, does it happen ~60% of the time?
    for line in lines[stat]:
        chance = 1 - poisson.cdf(int(line), predictions)   # model's chance of going over the line
        hit = test[stat] > line                             # did he actually go over? (True/False)
        # put every test game in a bucket by its predicted chance: 0-10%, 10-20%, ... 90-100%
        bucket = pd.cut(chance, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        check = pd.DataFrame({"predicted": chance, "actual": hit.values, "bucket": bucket})
        table = check.groupby("bucket", observed=True).agg(games=("actual", "size"),
                                                           predicted=("predicted", "mean"),
                                                           actually_hit=("actual", "mean"))
        say(f"\nBacktest: over {line} {stat} (predicted chance vs how often it really happened)")
        say(table.round(2).to_string())

    # the test is done, so now let the model learn from EVERY game, including the newest ones
    model.fit(df[features], df[stat])
    models[stat] = model

# ---- Goalies: a separate saves model (goalies are in their own file) ----
goalies = pd.read_csv("goalies.csv")
# only games he STARTED - a backup who comes in for the 3rd period would drag his averages down
goalies = goalies[goalies["gamesStarted"] == 1].sort_values(["playerId", "gameDate"])
for col in ["saves", "shotsAgainst"]:
    for n in (5, 10):
        goalies[f"{col}_last{n}"] = goalies.groupby("playerId")[col].transform(lambda s: s.shift().rolling(n).mean())

# how many shots each TEAM takes per game, averaged over its previous 10 games (the shooters he'll face)
shots_for = all_games.groupby(["teamAbbrev", "gameId", "gameDate"])["shots"].sum().reset_index()
shots_for = shots_for.rename(columns={"teamAbbrev": "team", "shots": "shots_for"}).sort_values(["team", "gameDate"])
shots_for["opp_shots_for_last10"] = shots_for.groupby("team")["shots_for"].transform(lambda s: s.shift().rolling(10).mean())

# his own team's defence: shots it allowed over its last 10 (from the skater opponent table, looked up by HIS team)
goalies = goalies.merge(allowed[["team", "gameId", "opp_shots_allowed_last10"]].rename(
                            columns={"opp_shots_allowed_last10": "own_shots_allowed_last10"}),
                        left_on=["teamAbbrev", "gameId"], right_on=["team", "gameId"], how="left").drop(columns="team")
# the opponent's offence: how many shots they've been taking
goalies = goalies.merge(shots_for[["team", "gameId", "opp_shots_for_last10"]],
                        left_on=["opponentTeamAbbrev", "gameId"], right_on=["team", "gameId"], how="left").drop(columns="team")
all_goalie_games = goalies.copy()

goalie_features = ["saves_last5", "saves_last10", "shotsAgainst_last5", "shotsAgainst_last10",
                   "own_shots_allowed_last10", "opp_shots_for_last10", "home"]
goalies = goalies.dropna(subset=goalie_features)
g_train = goalies[goalies["gameDate"] < cutoff]
g_test = goalies[goalies["gameDate"] >= cutoff]

say("\n########## SAVES (goalies) ##########")
baseline_error = (g_test["saves_last5"] - g_test["saves"]).abs().mean()
goalie_model = HistGradientBoostingRegressor(loss="poisson", random_state=0, learning_rate=0.05, max_iter=300,
                                             max_depth=4, min_samples_leaf=100, l2_regularization=1.0)
goalie_model.fit(g_train[goalie_features], g_train["saves"])
g_pred = goalie_model.predict(g_test[goalie_features])
model_error = (g_pred - g_test["saves"]).abs().mean()
say("baseline: off by", round(baseline_error, 2), "| model: off by", round(model_error, 2),
    "|", round((1 - model_error / baseline_error) * 100, 1), "% better")

# saves swing more than a Poisson curve expects (a 24-save average can easily be 15 or 35),
# so use a normal "bell curve" whose width = how far off the model usually is in training
saves_sd = (g_train["saves"] - goalie_model.predict(g_train[goalie_features])).std()
for line in (22.5, 25.5, 28.5):
    chance = 1 - norm.cdf(line, g_pred, saves_sd)
    check = pd.DataFrame({"predicted": chance, "actual": (g_test["saves"] > line).values,
                          "bucket": pd.cut(chance, bins=[0, .2, .35, .5, .65, .8, 1])})
    say(f"\nBacktest: over {line} saves")
    say(check.groupby("bucket", observed=True).agg(games=("actual", "size"), predicted=("predicted", "mean"),
                                                   actually_hit=("actual", "mean")).round(2).to_string())

goalie_model.fit(goalies[goalie_features], goalies["saves"])   # now learn from every start

# ---- Moneyline: who wins the game (the two teams "face off") ----
teams = pd.read_csv("teams.csv").sort_values(["team", "gameDate"])
teams["goalDiff"] = teams["goalsFor"] - teams["goalsAgainst"]   # +2 = won by 2, -1 = lost by 1
for col in ["goalDiff", "shotsFor", "shotsAgainst"]:
    teams[f"{col}_last10"] = teams.groupby("team")[col].transform(lambda s: s.shift().rolling(10).mean())

# the starting goalie's save % over his previous 10 starts (a hot goalie steals games)
starts = all_goalie_games.sort_values(["playerId", "gameDate"]).copy()
saves_10 = starts.groupby("playerId")["saves"].transform(lambda s: s.shift().rolling(10, min_periods=5).sum())
shots_10 = starts.groupby("playerId")["shotsAgainst"].transform(lambda s: s.shift().rolling(10, min_periods=5).sum())
starts["sv_last10"] = saves_10 / shots_10
teams = teams.merge(starts[["gameId", "teamAbbrev", "sv_last10"]].rename(columns={"teamAbbrev": "team"}),
                    on=["gameId", "team"], how="left")

# face off: put the opponent's numbers on the same row, so the model compares the two teams
team_cols = ["goalDiff_last10", "shotsFor_last10", "shotsAgainst_last10", "sv_last10"]
opp_cols = {c: f"opp_{c}" for c in team_cols}
teams = teams.merge(teams[["gameId", "team"] + team_cols].rename(columns={"team": "opponent", **opp_cols}),
                    on=["gameId", "opponent"])
ml_features = team_cols + list(opp_cols.values()) + ["home"]
typical_sv = teams["sv_last10"].median()   # goalie with under 5 starts -> assume an average goalie
teams[["sv_last10", "opp_sv_last10"]] = teams[["sv_last10", "opp_sv_last10"]].fillna(typical_sv)
teams = teams.dropna(subset=ml_features)

# logistic regression: a simple model that turns the features into a win chance (0-100%).
# it beat the boosted trees here - team results depend on a few strong signals, so simple works best
ml_model = LogisticRegression(C=0.5, max_iter=1000)
t_train, t_test = teams[teams["gameDate"] < cutoff], teams[teams["gameDate"] >= cutoff]
ml_model.fit(t_train[ml_features], t_train["win"])


def home_win_chance(home_rows, away_rows):
    """ask the model from BOTH sides and average, so the two teams' chances add up to exactly 100%"""
    home_view = ml_model.predict_proba(home_rows[ml_features])[:, 1]
    away_view = ml_model.predict_proba(away_rows[ml_features])[:, 1]
    return (home_view + (1 - away_view)) / 2


home_rows = t_test[t_test["home"] == 1].sort_values("gameId")
away_rows = t_test[t_test["home"] == 0].set_index("gameId").loc[home_rows["gameId"]]
p_home = home_win_chance(home_rows, away_rows)
won = home_rows["win"].values
baseline = t_train[t_train["home"] == 1]["win"].mean()
say("\n########## MONEYLINE ##########")
say(f"chance-error: model {((p_home - won) ** 2).mean():.4f} vs always-{baseline:.0%}-home {((baseline - won) ** 2).mean():.4f}")
check = pd.DataFrame({"predicted": list(p_home) + list(1 - p_home), "actual": list(won) + list(1 - won)})
check["bucket"] = pd.cut(check["predicted"], bins=[0, .35, .45, .55, .65, 1])
say(check.groupby("bucket", observed=True).agg(games=("actual", "size"), predicted=("predicted", "mean"),
                                               actually_won=("actual", "mean")).round(2).to_string())
ml_model.fit(teams[ml_features], teams["win"])   # now learn from every game

# ---- Step 8: predict tonight's games ----
# 1. tonight's schedule from the NHL (free, no key)
# the game day to predict: tonight - or the day Week_Run.py asks for (PQ_DAY) when you pick "This week"
today = os.getenv("PQ_DAY") or pd.Timestamp.today().strftime("%Y-%m-%d")
schedule =requests.get(f"https://api-web.nhle.com/v1/schedule/{today}").json()
day = schedule["gameWeek"][0]                  # first day of the week = today
games_tonight = day["games"] if day["date"] == today else []
say("\n", today, "-", len(games_tonight), "games tonight")

# team -> (opponent, home 1 or away 0)
matchups = {}
for game in games_tonight:
    home_team = game["homeTeam"]["abbrev"]
    away_team = game["awayTeam"]["abbrev"]
    matchups[home_team] = (away_team, 1)
    matchups[away_team] = (home_team, 0)

# team -> the game it's in, like "SEA @ DET" (the parlay builder uses this to keep legs in different games)
game_of = {}
for game in games_tonight:
    label = f"{game['awayTeam']['abbrev']} @ {game['homeTeam']['abbrev']}"
    game_of[game["homeTeam"]["abbrev"]] = label
    game_of[game["awayTeam"]["abbrev"]] = label

# 2. each player's form going INTO tonight, from his most recent games (no shift: all of them are in the past)
recent = all_games.groupby("playerId")
form = pd.DataFrame({
    "skaterFullName": recent["skaterFullName"].last(),
    "teamAbbrev": recent["teamAbbrev"].last(),      # the team he played for most recently
    "lastGame": recent["gameDate"].last(),
    "gamesPlayed": recent["gameId"].count(),
    "toi_last5": recent["toiMinutes"].apply(lambda s: s.tail(5).mean()),
    "pp_last5": recent["ppMinutes"].apply(lambda s: s.tail(5).mean()),
})
for stat in feature_stats:
    form[f"{stat}_prev"] = recent[stat].last()
    form[f"{stat}_last5"] = recent[stat].apply(lambda s: s.tail(5).mean())
    form[f"{stat}_last10"] = recent[stat].apply(lambda s: s.tail(10).mean())
    # blank under 20 games, exactly like in training
    form[f"{stat}_last20"] = recent[stat].apply(lambda s: s.tail(20).mean() if len(s) >= 20 else float("nan"))

# 3. keep players whose team plays tonight
form = form[form["teamAbbrev"].isin(matchups.keys())]
form = form[form["gamesPlayed"] >= 10]   # same rule as training: needs 10 games of history
# skip players who missed their team's last game (likely injured, scratched or sent down)
team_last_game = all_games.groupby("teamAbbrev")["gameDate"].max()
form = form[form["lastGame"] == form["teamAbbrev"].map(team_last_game)]

form["opponent"] = form["teamAbbrev"].map(lambda t: matchups[t][0])
form["home"] = form["teamAbbrev"].map(lambda t: matchups[t][1])

# opponent's average shots and goals allowed over its last 10 games (no shift: all already played)
opp_now = allowed.groupby("team")[["shots_allowed", "goals_allowed"]].apply(lambda t: t.tail(10).mean())
form["opp_shots_allowed_last10"] = form["opponent"].map(opp_now["shots_allowed"])
form["opp_goals_allowed_last10"] = form["opponent"].map(opp_now["goals_allowed"])

# 4. predict, then turn each average into chances with a Poisson distribution
for stat in stats:
    # (an empty list - e.g. no games today - can't go into the model, so give it an empty column instead)
    form[f"predicted_{stat}"] = models[stat].predict(form[features]).round(2) if len(form) else pd.Series(dtype=float)
form["shots_over_1.5"] = 1 - poisson.cdf(1, form["predicted_shots"])    # chance of 2 or more shots
form["shots_over_2.5"] = 1 - poisson.cdf(2, form["predicted_shots"])    # chance of 3 or more shots
form["points_over_0.5"] = 1 - poisson.cdf(0, form["predicted_points"])  # chance of 1 or more points
form["goals_over_0.5"] = 1 - poisson.cdf(0, form["predicted_goals"])    # chance of scoring (anytime goal)

# goalies: every goalie's form going into tonight (from his recent STARTS)
g_recent = all_goalie_games.groupby("playerId")
goalie_form = pd.DataFrame({
    "skaterFullName": g_recent["goalieFullName"].last(),   # same column name as skaters, so one lookup works for everyone
    "starts": g_recent["gameId"].count(),
})
for col in ["saves", "shotsAgainst"]:
    for n in (5, 10):
        goalie_form[f"{col}_last{n}"] = g_recent[col].apply(lambda s: s.tail(n).mean())
goalie_form = goalie_form[goalie_form["starts"] >= 10]   # needs 10 starts of history, same as training
goalie_form["name_key"] = goalie_form["skaterFullName"].map(clean_name)

# who is actually starting tonight (Daily Faceoff) - only those goalies get saves bets
# team full name ("Detroit Red Wings") -> abbreviation ("DET"), built from tonight's NHL schedule
abbrev_of = {}
for game in games_tonight:
    for side in ("homeTeam", "awayTeam"):
        team = game[side]
        abbrev_of[clean_name(team["placeName"]["default"] + " " + team["commonName"]["default"])] = team["abbrev"]
try:
    starters = get_starting_goalies(today)
    starters = starters[starters["status"].isin(["Confirmed", "Likely"])]   # skip "Unconfirmed" - could be either goalie
    starters["teamAbbrev"] = starters["team_name"].map(clean_name).map(abbrev_of)
    starters["name_key"] = starters["goalie"].map(clean_name)
except Exception as error:
    print("Couldn't get starting goalies (", error, ") - no saves bets tonight")
    starters = pd.DataFrame(columns=["name_key", "teamAbbrev", "status", "goalie"])
# inner merge = keep only goalies who are starting tonight, and use their CURRENT team
# (a goalie traded in the summer still shows his old team in last season's games)
goalie_form = goalie_form.merge(starters[["name_key", "teamAbbrev", "status"]], on="name_key")
goalie_form = goalie_form[goalie_form["teamAbbrev"].isin(matchups.keys())]
goalie_form["opponent"] = goalie_form["teamAbbrev"].map(lambda t: matchups[t][0])
goalie_form["home"] = goalie_form["teamAbbrev"].map(lambda t: matchups[t][1])
goalie_form["own_shots_allowed_last10"] = goalie_form["teamAbbrev"].map(opp_now["shots_allowed"])
shots_for_now = shots_for.groupby("team")["shots_for"].apply(lambda s: s.tail(10).mean())
goalie_form["opp_shots_for_last10"] = goalie_form["opponent"].map(shots_for_now)
if goalie_form.empty:
    # early in the day starters usually aren't confirmed yet - skip saves bets, everything else still runs
    print("No starting goalies confirmed yet - no saves bets this run. Run it again closer to puck drop "
          "(starters are usually confirmed 1-3 hours before) to get them.")
    goalie_form["predicted_saves"] = pd.Series(dtype=float)
else:
    goalie_form["predicted_saves"] = goalie_model.predict(goalie_form[goalie_features]).round(1)

# tonight's face-offs: each team's last 10 games + its starting goalie's last 10 starts
raw_teams = pd.read_csv("teams.csv").sort_values(["team", "gameDate"])
raw_teams["goalDiff"] = raw_teams["goalsFor"] - raw_teams["goalsAgainst"]
team_now = raw_teams.groupby("team")[["goalDiff", "shotsFor", "shotsAgainst"]].apply(lambda t: t.tail(10).mean())
team_now.columns = ["goalDiff_last10", "shotsFor_last10", "shotsAgainst_last10"]
goalie_starts = all_goalie_games.sort_values("gameDate").groupby("goalieFullName").tail(10)
sv_now = goalie_starts.groupby("goalieFullName").apply(lambda g: g["saves"].sum() / g["shotsAgainst"].sum())
sv_now.index = sv_now.index.map(clean_name)
starter_of = dict(zip(starters["teamAbbrev"], starters["goalie"]))   # team -> tonight's starter


def team_row(team, opponent, home):
    row = {**team_now.loc[team].to_dict(), "home": home}
    row["sv_last10"] = sv_now.get(clean_name(starter_of.get(team, "")), typical_sv)   # unknown starter -> average
    for col in team_cols:   # the opponent's numbers, same way
        row[f"opp_{col}"] = team_now.loc[opponent][col] if col != "sv_last10" else \
            sv_now.get(clean_name(starter_of.get(opponent, "")), typical_sv)
    return pd.DataFrame([row])


win_prob = {}   # team -> chance of winning tonight
faceoffs = []   # saved for the dashboard
print("\n########## FACE-OFFS (who wins) ##########")
for game in games_tonight:
    home_team, away_team = game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"]
    p = home_win_chance(team_row(home_team, away_team, 1), team_row(away_team, home_team, 0))[0]
    win_prob[home_team], win_prob[away_team] = p, 1 - p
    print(f"  {away_team} {1 - p:4.0%}  @  {home_team} {p:4.0%}   "
          f"(goalies: {starter_of.get(away_team, '?')} vs {starter_of.get(home_team, '?')})")
    faceoffs.append({"away": away_team, "away_win": round(1 - p, 3), "home": home_team, "home_win": round(p, 3),
                     "away_goalie": starter_of.get(away_team, "?"), "home_goalie": starter_of.get(home_team, "?")})
pd.DataFrame(faceoffs).to_csv("faceoffs_today.csv", index=False)

# 5. print the top 10 players on each team tonight, best chance of a point first
for game in games_tonight:
    home_team = game["homeTeam"]["abbrev"]
    away_team = game["awayTeam"]["abbrev"]
    say(f"\n=== {away_team} @ {home_team} ===")
    for team in (away_team, home_team):
        opponent = matchups[team][0]
        say(f"  -- {team} (vs {opponent}, who allow {opp_now.loc[opponent, 'shots_allowed']:.1f} shots "
              f"and {opp_now.loc[opponent, 'goals_allowed']:.1f} goals a game lately)")
        top = form[form["teamAbbrev"] == team].sort_values("predicted_points", ascending=False).head(10)
        for _, p in top.iterrows():
            say(f"  {team}  {p['skaterFullName']:24} "
                  f"1+ point: {p['points_over_0.5']:4.0%}   goal: {p['goals_over_0.5']:4.0%}   "
                  f"shots {p['predicted_shots']:4.2f} (over 2.5: {p['shots_over_2.5']:4.0%})")

# ---- Step 9: compare with the sportsbooks, keep only the bets worth placing ----
def american_odds(decimal_price):
    """decimal odds (2.50) -> the American odds Ontario apps also show (+150 / -150)"""
    if decimal_price >= 2:
        return f"{(decimal_price - 1) * 100:+.0f}"
    return f"{-100 / (decimal_price - 1):+.0f}"



# 1. load tonight's odds (made by Odds_Scraper.py, so this script never spends API credits)
if not os.path.exists("odds.csv"):
    raise SystemExit("No odds.csv yet - run: python Odds_Scraper.py")
odds_day = pd.Timestamp.fromtimestamp(os.path.getmtime("odds.csv")).strftime("%Y-%m-%d")   # local time, not UTC
if odds_day != pd.Timestamp.today().strftime("%Y-%m-%d"):   # the download itself must be from today
    raise SystemExit(f"odds.csv is from {odds_day}, not today - run: python Odds_Scraper.py")
odds = pd.read_csv("odds.csv")
# only odds for TONIGHT's games - the file could hold another day's games (e.g. scraped late at night),
# and a team that plays tonight AND tomorrow would otherwise get tomorrow's odds matched to tonight's game
tonight_full_names = set()
for game in games_tonight:
    def full(team):
        return clean_name(team["placeName"]["default"] + " " + team["commonName"]["default"])
    tonight_full_names.add(full(game["awayTeam"]) + " @ " + full(game["homeTeam"]))
odds = odds[odds["game"].map(lambda g: " @ ".join(clean_name(t) for t in str(g).split(" @ "))).isin(tonight_full_names)]
if "game_date" in odds.columns:   # a week's download: only this day's games (the same teams play again later)
    odds = odds[odds["game_date"] == today]
odds["game_date"] = today          # every bet remembers its game day (needed when you get the whole week)
if odds.empty:
    raise SystemExit("odds.csv has no odds for tonight's games - run: python Odds_Scraper.py")

# 2. the book's real opinion: remove its cut (the "vig") by scaling Over + Under so they add up to 100%
odds["implied"] = 1 / odds["price"]
odds["player"] = odds["player"].fillna("")   # moneylines have no player - blank, so the two teams pair up
pair = ["book", "game", "market", "player", "line"]
sides = odds.groupby(pair)["implied"].transform("count")
odds["book_chance"] = odds["implied"] / odds.groupby(pair)["implied"].transform("sum")
odds.loc[sides < 2, "book_chance"] = odds["implied"]   # anytime goal only has "Yes", so the cut can't be removed

# moneylines: set them aside and give them the face-off model's win chance
moneylines = odds[odds["market"] == "h2h"].copy()
moneylines["team"] = moneylines["side"].map(clean_name).map(abbrev_of)   # "Detroit Red Wings" -> "DET"
moneylines["model_chance"] = moneylines["team"].map(win_prob)
moneylines["mean"] = moneylines["model_chance"]   # for moneylines the live tracker starts from the pre-game win chance
moneylines["sd"] = 0
moneylines = moneylines.dropna(subset=["model_chance"])                  # only tonight's games
moneylines["game"] = moneylines["team"].map(game_of)
moneylines["stat"] = "moneyline"
moneylines["player"] = moneylines["side"]                                # shows the team's name in the output
moneylines["name_key"] = "moneyline " + moneylines["team"]
odds = odds[odds["market"] != "h2h"]

# 3. the model's chance for the exact same bet (player props)
everyone = pd.concat([form, goalie_form])   # skaters + goalies in one lookup table
everyone["name_key"] = everyone["skaterFullName"].map(clean_name)
players = everyone.drop_duplicates("name_key").set_index("name_key")
odds["name_key"] = odds["player"].map(clean_name)
odds = odds[odds["name_key"].isin(players.index)]   # players the model knows and who are playing tonight
stat_of = {"player_points": "points", "player_goal_scorer_anytime": "goals",
           "player_shots_on_goal": "shots", "player_total_saves": "saves"}


def model_chance(row):
    stat = stat_of[row["market"]]
    mean = players.loc[row["name_key"], f"predicted_{stat}"]
    line = row["line"]
    if stat == "saves":
        # bell curve for saves (see the goalie section); +-0.5 because saves are whole numbers
        over = 1 - norm.cdf(math.floor(line) + 0.5, mean, saves_sd)
        push = norm.cdf(line + 0.5, mean, saves_sd) - norm.cdf(line - 0.5, mean, saves_sd) if line == int(line) else 0
    else:
        over = 1 - poisson.cdf(math.floor(line), mean)                      # chance of going over the line
        push = poisson.pmf(line, mean) if line == int(line) else 0          # whole-number lines can tie
    return over if row["side"] in ("Over", "Yes") else 1 - over - push


odds["model_chance"] = [model_chance(row) for _, row in odds.iterrows()]   # works even when there are no props yet
# the model's predicted average (e.g. 3.1 shots) - the live tracker uses it to update the chance mid-game
odds["mean"] = [players.loc[row["name_key"], f"predicted_{stat_of[row['market']]}"] for _, row in odds.iterrows()]
odds["sd"] = [saves_sd if stat_of[m] == "saves" else 0 for m in odds["market"]]   # spread, only used for saves
odds["team"] = odds["name_key"].map(players["teamAbbrev"])
odds["game"] = odds["team"].map(game_of)
odds["stat"] = odds["market"].map(stat_of)
odds = pd.concat([odds, moneylines])   # player props + moneylines, checked the same way from here on

# blend the model with the market: raw_chance = the model on its own, model_chance = the blended chance we bet on
odds["raw_chance"] = odds["model_chance"]
odds["model_chance"] = MODEL_WEIGHT * odds["raw_chance"] + (1 - MODEL_WEIGHT) * odds["book_chance"]

# 4. expected profit per $1: win -> get price back, lose -> lose the $1
odds["ev"] = odds["model_chance"] * odds["price"] - 1

# 5. keep only the bets worth placing
good = odds[
    (odds["ev"] >= MIN_EDGE)                                                   # enough profit expected
    # the safety checks look at the model ON ITS OWN (raw_chance) - a wild raw disagreement means the book knows something
    & ((odds["raw_chance"] - odds["book_chance"]).abs() <= MAX_DISAGREE)      # not a huge disagreement
    & (odds["raw_chance"] <= odds["book_chance"] * MAX_RATIO)                 # not wildly more confident than the book
    & odds["price"].between(MIN_PRICE, MAX_PRICE)                             # risk : reward between 2:1 and 1:1.5
    & ((odds["ev"] >= MIN_EDGE + OVER_EXTRA_EDGE) | ~odds["side"].isin(["Over", "Yes"]))   # Overs need a bigger edge
    & (odds["raw_chance"] <= 0.85)                                            # extremes aren't trustworthy
    & ~((odds["stat"] == "goals") & (odds["raw_chance"] > 0.40))              # backtest: goal chances over 40% are shaky
    & ~((odds["stat"] == "saves") & ((odds["raw_chance"] < 0.30) | (odds["raw_chance"] > 0.70)))   # saves: only trust 30-70%
].copy()

# stake: a quarter of the Kelly formula (the standard way to size bets), never more than 2% of your bankroll
good["stake"] = (((good["model_chance"] * good["price"] - 1) / (good["price"] - 1)) / 4).clip(upper=0.02) * BANKROLL
# lowest odds still worth it (Overs need the bigger edge) - if your app shows less than this, skip it
good["min_price"] = ((1 + MIN_EDGE + OVER_EXTRA_EDGE * good["side"].isin(["Over", "Yes"])) / good["model_chance"]).clip(
    lower=MIN_PRICE)


def describe(row):
    if row["market"] == "h2h":
        return "to win (moneyline)"
    if row["market"] == "player_goal_scorer_anytime":
        return "anytime goal"
    if row["stat"] == "saves":   # show whether he's a confirmed or only likely starter
        return f"{row['side']} {row['line']:g} saves  (starter: {players.loc[row['name_key'], 'status']})"
    return f"{row['side']} {row['line']:g} {row['stat']}"


good["bet"] = [describe(row) for _, row in good.iterrows()]   # loop works even when there are no good bets

# every line the model priced (good value or not) - the dashboard's parlay builder lets you pick from these
odds["bet"] = [describe(row) for _, row in odds.iterrows()]
lines = odds[(odds["model_chance"] > 0) & (odds["model_chance"] < 1)].copy()
lines[["player", "team", "game", "game_date", "stat", "side", "line", "bet", "price", "mean", "sd",
       "book", "raw_chance", "model_chance", "book_chance", "ev"]].to_csv("lines_today.csv", index=False)

# the same bet at the best-paying book only
singles = good.sort_values("price", ascending=False).drop_duplicates(["name_key", "market", "side", "line"])
singles = singles.sort_values("ev", ascending=False)
# books can post different lines (Under 25.5 at one, 26.5 at another) - keep only his best one per stat
singles = singles.drop_duplicates(["name_key", "market"])
# never bet both teams in the same game - keep only the better moneyline per game
is_ml = singles["stat"] == "moneyline"
singles = pd.concat([singles[~is_ml], singles[is_ml].drop_duplicates("game")]).sort_values("ev", ascending=False)
# tighten: the strongest bets only
found_before_caps = len(singles)
singles = singles.drop_duplicates("player")            # one bet per player (his best one), not shots AND points AND goals
singles = singles.groupby("game").head(MAX_PER_GAME)   # at most MAX_PER_GAME per game
singles = singles.head(MAX_BETS)                       # and at most MAX_BETS for the whole night
if found_before_caps > len(singles):
    print(f"{found_before_caps} bets passed the checks - keeping the best {len(singles)} "
          f"(1 per player, {MAX_PER_GAME} per game, {MAX_BETS} max)")

print(f"\n########## BETS TO PLACE ({len(singles)}) ##########")
if singles.empty:
    print("Nothing worth betting tonight - the books' prices are fair or better than the model. Skip tonight.")
for _, b in singles.iterrows():
    # fair odds = a chance turned into odds (1 / chance), so you can compare odds with odds
    print(f"\n  {b['player']} ({b['team']}, {b['game']})  -  {b['bet']}")
    print(f"     BET AT {b['book']}: {american_odds(b['price'])}   (decimal {b['price']:.2f})")
    print(f"     model's fair odds: {american_odds(1 / b['model_chance'])} ({b['model_chance']:.0%})   |   "
          f"book's fair odds: {american_odds(1 / b['book_chance'])} ({b['book_chance']:.0%})")
    print(f"     expected profit {b['ev']:+.0%}   |   bet CA${b['stake']:.2f}   |   "
          f"still bet it down to: {american_odds(b['min_price'])}")

# save tonight's bets for the dashboard (every run replaces the file)
leg_columns = ["player", "team", "game", "game_date", "stat", "side", "line", "bet", "price", "mean", "sd"]
singles[leg_columns + ["book", "raw_chance", "model_chance", "book_chance", "ev", "stake", "min_price"]].to_csv("bets_today.csv", index=False)

# ---- Step 10: value parlays (every leg must be at the SAME book - you can't mix books on one slip) ----
parlays = []
for book, book_legs in good.groupby("book"):
    # goalie saves only go in a parlay when they're at least 60% likely (saves swing a lot, one bad game sinks the slip)
    book_legs = book_legs[~((book_legs["stat"] == "saves") & (book_legs["model_chance"] < 0.60))]
    # same for moneylines: a team only goes in a parlay when it's at least 60% to win
    book_legs = book_legs[~((book_legs["stat"] == "moneyline") & (book_legs["model_chance"] < 0.60))]
    book_legs = book_legs.sort_values("ev", ascending=False).drop_duplicates("name_key").head(10)
    for size in (2, 3):
        for combo in itertools.combinations(book_legs.to_dict("records"), size):
            if len({leg["game"] for leg in combo}) < size:
                continue   # legs in the same game are linked, so their chances can't just be multiplied
            chance = math.prod(leg["model_chance"] for leg in combo)   # all legs must hit
            price = math.prod(leg["price"] for leg in combo)          # payouts multiply
            if chance >= 0.15:                                         # skip lottery tickets
                parlays.append({"book": book, "legs": combo, "chance": chance, "price": price,
                                "ev": chance * price - 1})

# keep the best 2 of EACH size, so 3-leg parlays always get a spot (not just whichever size scores highest)
best_parlays = [p for size in (2, 3)
                for p in sorted([q for q in parlays if len(q["legs"]) == size], key=lambda q: q["ev"], reverse=True)[:2]]
print(f"\n########## PARLAYS TO PLACE ({len(best_parlays)}) ##########")
if not best_parlays:
    print("No value parlays tonight.")
for i, p in enumerate(best_parlays, start=1):
    stake = min(((p["chance"] * p["price"] - 1) / (p["price"] - 1)) / 4, 0.01) * BANKROLL   # max 1% on parlays
    print(f"\n  #{i} BET AT {p['book']}: {american_odds(p['price'])}   (decimal {p['price']:.2f})")
    print(f"     model's fair odds: {american_odds(1 / p['chance'])} (hits {p['chance']:.0%})   |   "
          f"expected profit {p['ev']:+.0%}   |   bet CA${stake:.2f}")
    for leg in p["legs"]:
        print(f"       {leg['player']:24} {leg['bet']:16} {american_odds(leg['price']):>6}   "
              f"(model's fair odds {american_odds(1 / leg['model_chance'])})   {leg['game']}")
    p["stake"] = stake

# save tonight's parlays for the dashboard - the legs go in as JSON text so a whole parlay fits in one row
pd.DataFrame([{"book": p["book"], "price": p["price"], "chance": p["chance"], "ev": p["ev"], "stake": p["stake"],
               "legs": json.dumps([{c: leg[c] for c in leg_columns} for leg in p["legs"]], default=float)}
              for p in best_parlays], columns=["book", "price", "chance", "ev", "stake", "legs"]).to_csv("parlays_today.csv", index=False)

# keep a permanent record of EVERY bet the model found (taken or not) - the dashboard's Fumbles page
# grades these too, so you can see how the bets you skipped would have done
found = [{"legs": [{c: row[c] for c in leg_columns}], "book": row["book"], "price": row["price"], "stake": row["stake"]}
         for _, row in singles.iterrows()]
found += [{"legs": [{c: leg[c] for c in leg_columns} for leg in p["legs"]], "book": p["book"], "price": p["price"],
           "stake": p["stake"]} for p in best_parlays]
save_found(today, found)

print("\nCheck lineups before betting, and confirm each price in your Ontario app.")
