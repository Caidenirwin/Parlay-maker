# Data and architecture

Research and educational project. Not betting advice.

This note matches the scripts in the repository. Scraped tables and model outputs live in `data/`. Personal bet logs and the largest player-stat downloads are gitignored and are created locally when you run the scripts.

## Pipeline

Each league follows the same three steps. The dashboard runs them for you.

1. **Stats.** A `*_Scraper.py` script downloads games, teams, and (where used) player rows into `data/`.
2. **Odds.** A `*_Odds_Scraper.py` script reads lines from [The Odds API](https://the-odds-api.com). Keys come from environment variables and are never written into the repo. The shared variable is `ODDS_API_KEY`. Optional per-league overrides are `NHL_ODDS_API_KEY`, `NBA_ODDS_API_KEY`, `WNBA_ODDS_API_KEY`, `MLB_ODDS_API_KEY`, `NFL_ODDS_API_KEY`, and `CFB_ODDS_API_KEY`.
3. **Model.** A `*_ML.py` script (NHL uses `ML.py`) trains gradient-boosted trees, blends those probabilities with the market, keeps positive-EV candidates, and sizes stakes with a fractional Kelly cap. It writes today's lines, bets, parlays, and matchup files back to `data/`.

`dashboard.py` is a Streamlit app. It shells out to the scripts above and reads the CSVs they produce. `Bet_Tracker.py` records bets you choose to log and grades them from public box scores. `Live_Tracker.py` and `NFL_Live.py` read live scoreboards. `Freshness.py` skips a download when today's file is already current. `Goalie_Scraper.py` supplies starting goalies to the NHL model.

## Files by league

| League | Stats | Odds | Model | Saved in `data/` | Created locally, not committed |
|---|---|---|---|---|---|
| NHL | `NHL_Scraper.py` | `Odds_Scraper.py` | `ML.py` | `teams.csv`, `odds.csv`, `lines_today.csv` | `skaters.csv`, `goalies.csv`, `bets_today.csv`, `parlays_today.csv`, `faceoffs_today.csv` |
| NBA | `NBA_Scraper.py` | `NBA_Odds_Scraper.py` | `NBA_ML.py` | `nba_games.csv`, `nba_teams.csv`, `nba_odds.csv`, `nba_lines_today.csv` | `nba_players.csv`, `nba_bets_today.csv`, `nba_parlays_today.csv`, `nba_faceoffs_today.csv` |
| WNBA | `WNBA_Scraper.py` | `WNBA_Odds_Scraper.py` | `WNBA_ML.py` | `wnba_games.csv`, `wnba_teams.csv` | `wnba_players.csv`, `wnba_odds.csv`, `wnba_lines_today.csv`, `wnba_bets_today.csv`, `wnba_parlays_today.csv`, `wnba_faceoffs_today.csv` |
| MLB | `MLB_Scraper.py` | `MLB_Odds_Scraper.py` | `MLB_ML.py` | `mlb_games.csv`, `mlb_teams.csv`, `mlb_odds.csv`, `mlb_lines_today.csv` | `mlb_players.csv`, `mlb_bets_today.csv`, `mlb_parlays_today.csv`, `mlb_faceoffs_today.csv` |
| NFL | `NFL_Scraper.py` | `NFL_Odds_Scraper.py` | `NFL_ML.py` | `nfl_games.csv`, `nfl_odds.csv`, `nfl_lines_today.csv` | `nfl_players.csv`, `nfl_bets_today.csv`, `nfl_parlays_today.csv`, `nfl_faceoffs_today.csv` |
| College football | `CFB_Scraper.py` | `CFB_Odds_Scraper.py` | `CFB_ML.py` | `cfb_games.csv`, `cfb_teams.csv`, `cfb_odds.csv`, `cfb_lines_today.csv` | `cfb_players.csv`, `cfb_bets_today.csv`, `cfb_parlays_today.csv`, `cfb_faceoffs_today.csv` |
| PWHL | `PWHL_Scraper.py` | — | — | `pwhl_games.csv`, `pwhl_teams.csv`, `pwhl_goalies.csv` | `pwhl_skaters.csv` |

PWHL stats come from the public feed the league site uses. There is no PWHL odds scraper or model in this repo.

`Bet_Tracker.py` writes `my_bets.csv` and `found_bets.csv` under `data/`. Both are gitignored so a personal betting log is not committed.

## What a row contains

- **Games and teams** are public schedule and score tables (ids, dates, abbreviations, scores, and logo URLs).
- **Odds** rows are a game, sportsbook, market, side, price, and the book’s implied probability.
- **Lines** rows are the model’s priced markets for the current slate: player, stat, line, price, model probability, and expected value.
- **Player stat files** (gitignored) are per-game counting stats used to fit the models. Re-create them by running the scraper for that league.

Run the scripts from the repository root, for example `python NHL_Scraper.py` or `streamlit run dashboard.py`. Paths are resolved from the repo, so the CSVs always land in `data/`.
