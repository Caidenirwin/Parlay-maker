# Parlay Quant

Research and educational project. Not betting advice.

A multi-sport sports-betting research model covering NHL, NBA, NFL, college football, MLB, WNBA and PWHL. It scrapes team, player and game data, pulls live odds from The Odds API, estimates win probabilities with gradient-boosted tree models, and surfaces positive-expected-value (EV) bets and parlays in a Streamlit dashboard.

## How it works

1. **Data collection:** `*_Scraper.py` scripts pull team, player, goalie and game stats for each league.
2. **Odds:** `*_Odds_Scraper.py` scripts pull current lines from The Odds API.
3. **Modelling:** `*_ML.py` scripts train gradient-boosted tree models and blend model probabilities with market (bookmaker) odds.
4. **Bet selection:** Candidate bets are filtered by expected value and sized with a fractional Kelly criterion.
5. **Dashboard and tracking:** `dashboard.py` (Streamlit) shows today's lines, edges and parlays. `Bet_Tracker.py` and `Live_Tracker.py` track results.

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key at [the-odds-api.com](https://the-odds-api.com) and set it as an environment variable (never commit keys):

```powershell
setx ODDS_API_KEY "your-key"
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

CSV tables live in `data/`. Large raw player-stat files are not included; the scraper scripts write them there. [docs/DATA.md](docs/DATA.md) lists which file each script reads and writes.

## Disclaimer

**This project is for educational and research purposes only.**

- Nothing in this repository is financial, investment, or betting advice, and none of it is a recommendation to place any bet.
- The author does not condone or encourage gambling. Sports betting involves real risk of losing money, and past model performance does not guarantee future results.
- The software is provided "as is", without warranty of any kind. The author is not responsible or liable for any losses, damages, or other consequences resulting from the use or misuse of this code, its outputs, or any decisions made based on them. You use it entirely at your own risk.
- You are solely responsible for complying with the laws and age requirements of your jurisdiction. Do not use this project where sports betting is illegal.
- If you choose to gamble, please do so responsibly. If gambling is causing you problems, help is available: in Ontario, call ConnexOntario at 1-866-531-2600. In the US, call 1-800-GAMBLER.
