import pandas as pd

# 2018 to this season. One season was far too little to learn from (only ~500 QB games) - tested:
# with 8 seasons of history every stat's chances got clearly more accurate
seasons = list(range(2018, 2027))   # 2018, 2019, ..., 2026

# already updated today and no game has started since? nothing new to download
from Freshness import already_fresh, football_game_starts
from paths import csv_path
if already_fresh(csv_path("nfl_players.csv"), football_game_starts("nfl")):
    raise SystemExit(0)

tables = []
for season in seasons:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
    # usecols: only download the columns we keep (the files have 150) - quicker and lighter
    season_df = pd.read_csv(url, low_memory=False, usecols=[
        "player_id", "player_display_name", "position", "team", "opponent_team", "season", "week", "season_type",
        "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
        "carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds"])
    print(season, "->", len(season_df), "rows")
    tables.append(season_df)

df = pd.concat(tables, ignore_index=True)

# regular season only (no playoffs), and only the positions books offer props on
df = df[df["season_type"] == "REG"]
df = df[df["position"].isin(["QB", "RB", "WR", "TE"])]

# keep only the columns we need
df = df[["player_id", "player_display_name", "position", "team", "opponent_team", "season", "week",
         "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
         "carries", "rushing_yards", "rushing_tds",
         "targets", "receptions", "receiving_yards", "receiving_tds"]]

# each player's games together, oldest to newest
df = df.sort_values(["player_id", "season", "week"])

df.to_csv(csv_path("nfl_players.csv"), index=False)
print(len(df), "rows saved to data/nfl_players.csv")

# ---- game results: who won, final scores (for the face-off / moneyline model) ----
games = pd.read_csv("https://github.com/nflverse/nfldata/raw/master/data/games.csv")
games = games[(games["season"] >= 2018) & (games["game_type"] == "REG")]
games = games[["game_id", "season", "week", "gameday", "away_team", "away_score", "home_team", "home_score"]]
games.to_csv(csv_path("nfl_games.csv"), index=False)   # includes this season's upcoming games (no score yet)
print(len(games), "games saved to data/nfl_games.csv")
