import os
import subprocess
import sys
import pandas as pd

# "This week": runs a sport's model once for EACH game day in its odds file and merges the results,
# so the dashboard shows the whole week's bets, parlays and face-offs at once.
# Each day gets its own run because each day has its own matchups, rest days and starting goalies / pitchers.
#   python Week_Run.py nhl      (or nba / wnba / mlb)
# Get the week's odds first:  python Odds_Scraper.py --week   (or NBA_Odds_Scraper.py --week, ...)

SPORTS = {"nhl": ("ML.py", "odds.csv", ""), "nba": ("NBA_ML.py", "nba_odds.csv", "nba_"),
          "wnba": ("WNBA_ML.py", "wnba_odds.csv", "wnba_"), "mlb": ("MLB_ML.py", "mlb_odds.csv", "mlb_")}
OUTPUTS = ["bets_today.csv", "lines_today.csv", "parlays_today.csv", "faceoffs_today.csv"]

if len(sys.argv) < 2 or sys.argv[1] not in SPORTS:
    raise SystemExit("Which sport? python Week_Run.py nhl   (or nba / wnba / mlb)")
script, odds_file, prefix = SPORTS[sys.argv[1]]
if not os.path.exists(odds_file):
    raise SystemExit(f"No {odds_file} yet - get the week's odds first (the odds scraper with --week)")

today = pd.Timestamp.now(tz="America/Toronto").strftime("%Y-%m-%d")
odds = pd.read_csv(odds_file)
days = sorted(d for d in odds["game_date"].unique() if d >= today) if "game_date" in odds.columns else [today]
print(f"{len(days)} game day(s) with odds: {', '.join(pd.Timestamp(d).strftime('%a %b %d') for d in days)}")

merged = {name: [] for name in OUTPUTS}
for day in days:
    print(f"\n==================== {pd.Timestamp(day).strftime('%A %b %d')} ====================")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PQ_DAY": day, "PYTHONIOENCODING": "utf-8"})
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print("  skipped -", (output.strip().splitlines() or ["no output"])[-1])
        continue
    # just the useful part: from the face-offs to the end
    start = output.find("FACE-OFFS")
    print(output[output.rfind("\n", 0, start) + 1 if start > 0 else -1500:].strip())
    for name in OUTPUTS:
        path = prefix + name
        if os.path.exists(path) and os.path.getsize(path) > 0:
            part = pd.read_csv(path)
            if name == "faceoffs_today.csv" and "date" not in part.columns:
                part["date"] = day   # NHL face-offs don't carry their day
            merged[name].append(part)

# one file per output for the whole week, earliest games first
for name, parts in merged.items():
    if parts:
        week = pd.concat(parts, ignore_index=True)
        for col in ("game_date", "date"):
            if col in week.columns:
                week = week.sort_values(col, kind="stable")
                break
        week.to_csv(prefix + name, index=False)
total = sum(len(p) for p in merged["bets_today.csv"])
print(f"\nThis week: {total} bet(s) across {len(days)} day(s) - saved for the dashboard")
