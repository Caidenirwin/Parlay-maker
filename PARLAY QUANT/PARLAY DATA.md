# PARLAY DATA

Project folder: [parlay](file:///C:/Users/caide/Downloads/parlay)

## Scripts
| Script | What it does | Run |
|---|---|---|
| [nhl_scraper.py](file:///C:/Users/caide/Downloads/parlay/nhl_scraper.py) | Pulls NHL player/goalie/team stats into `data/nhl.db` + CSVs | `python nhl_scraper.py --update` |
| [model.py](file:///C:/Users/caide/Downloads/parlay/model.py) | Machine-learning model (gradient-boosted trees). Learns each player's shots/points/goals/assists/blocks/PPP, goalie saves and team goals from past games; retrains automatically when new stats arrive | `python model.py --retrain` |
| [odds_scraper.py](file:///C:/Users/caide/Downloads/parlay/odds_scraper.py) | Pulls live NHL odds from The Odds API (needs `ODDS_API_KEY`) | `python odds_scraper.py --books fanduel` |
| [picks.py](file:///C:/Users/caide/Downloads/parlay/picks.py) | Prices lines, keeps +EV bets, sizes with fractional Kelly | `python picks.py --bankroll 200 --books fanduel` |
| [settle.py](file:///C:/Users/caide/Downloads/parlay/settle.py) | Grades logged picks; record, ROI, calibration | `python settle.py` |
| [dashboard.py](file:///C:/Users/caide/Downloads/parlay/dashboard.py) | Runs all of the above, then builds [dashboard.html](file:///C:/Users/caide/Downloads/parlay/dashboard.html): picks, best book (FanDuel / DraftKings / BetMGM), parlay ideas, profit, record, ROI | `python dashboard.py --bankroll 200` |

## Daily flow
One command does everything (stats → retrain model → odds from all 3 books → picks → settle → dashboard):

`python dashboard.py --bankroll 200`

- Rebuild the page without scraping or spending API credits: `python dashboard.py --no-refresh`
- Needs `ODDS_API_KEY` set for odds and picks

## Notes

