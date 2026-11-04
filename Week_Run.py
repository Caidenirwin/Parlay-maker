import itertools
import json
import math
import os
import subprocess
import sys
import pandas as pd
from Bet_Tracker import save_found

# "This week": runs a sport's model once for EACH game day in its odds file and merges the results,
# so the dashboard shows the whole week's bets, parlays and face-offs at once.
# Each day gets its own run because each day has its own matchups, rest days and starting goalies / pitchers.
#   python Week_Run.py nhl      (or nba / wnba / mlb)
# Get the week's odds first:  python Odds_Scraper.py --week   (or NBA_Odds_Scraper.py --week, ...)

SPORTS = {"nhl": ("ML.py", "odds.csv", ""), "nba": ("NBA_ML.py", "nba_odds.csv", "nba_"),
          "wnba": ("WNBA_ML.py", "wnba_odds.csv", "wnba_"), "mlb": ("MLB_ML.py", "mlb_odds.csv", "mlb_")}
OUTPUTS = ["bets_today.csv", "lines_today.csv", "parlays_today.csv", "faceoffs_today.csv"]
BANKROLL = 100   # CA$, the same as the models - parlay stakes are quarter-Kelly, at most 1% of this
NOT_LEG = {"book", "raw_chance", "model_chance", "book_chance", "ev", "stake", "min_price", "tough_d"}   # not leg columns


def week_parlays(bets):
    """the best parlays for the WHOLE week: legs from any day, mixed. Same rules as each night's parlays -
    one book per slip, every leg a different game (and player), 60%+ on moneylines / saves, no lottery tickets
    (15%+ to hit) - keeping the best 3 of each size."""
    leg_cols = [c for c in bets.columns if c not in NOT_LEG]
    parlays = []
    for book, legs in bets.groupby("book"):
        legs = legs[~(legs["stat"].isin(["moneyline", "saves"]) & (legs["model_chance"] < 0.60))]
        legs = legs.sort_values("ev", ascending=False).head(14).to_dict("records")
        for size in (2, 3):
            for combo in itertools.combinations(legs, size):
                if len({l["game"] for l in combo}) < size or len({l["player"] for l in combo}) < size:
                    continue   # legs in the same game are linked - their chances can't just be multiplied
                chance = math.prod(l["model_chance"] for l in combo)
                price = math.prod(l["price"] for l in combo)
                if chance >= 0.15:
                    parlays.append({"book": book, "legs": combo, "chance": chance, "price": price,
                                    "ev": chance * price - 1})
    # best first, but spread out: no single leg in more than 2 of the week's parlays (one miss can't sink them all)
    best, used = [], {}
    for size in (2, 3):
        picked = 0
        for p in sorted([q for q in parlays if len(q["legs"]) == size], key=lambda q: q["ev"], reverse=True):
            keys = [(l["player"], l["bet"], l["game"]) for l in p["legs"]]
            if picked < 3 and all(used.get(k, 0) < 2 for k in keys):
                best.append(p)
                picked += 1
                for k in keys:
                    used[k] = used.get(k, 0) + 1
    for p in best:
        p["stake"] = min(((p["chance"] * p["price"] - 1) / (p["price"] - 1)) / 4, 0.01) * BANKROLL
        p["legs"] = [{c: l[c] for c in leg_cols} for l in sorted(p["legs"], key=lambda l: str(l["game_date"]))]
    return best

if len(sys.argv) < 2 or sys.argv[1] not in SPORTS:
    raise SystemExit("Which sport? python Week_Run.py nhl   (or nba / wnba / mlb)")
script, odds_file, prefix = SPORTS[sys.argv[1]]
if not os.path.exists(odds_file):
    raise SystemExit(f"No {odds_file} yet - get the week's odds first (the odds scraper with --week)")

today = pd.Timestamp.now(tz="America/Toronto").strftime("%Y-%m-%d")
odds = pd.read_csv(odds_file)
days = sorted(d for d in odds["game_date"].unique() if d >= today) if "game_date" in odds.columns else [today]
print(f"{len(days)} game day(s) with odds: {', '.join(pd.Timestamp(d).strftime('%a %b %d') for d in days)}")
if len(days) < 2:
    print(f"  (only one day - that's a tonight download. For the whole week, get the odds with --week first:"
          f" the dashboard's 'This week' + Run all does both)")

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

# the week's parlays - mixed across every day - go on top of each night's own parlays
if merged["bets_today.csv"]:
    best = week_parlays(pd.concat(merged["bets_today.csv"], ignore_index=True))
    print(f"\n########## PARLAYS FOR THE WEEK ({len(best)}) ##########")
    for i, p in enumerate(best, start=1):
        print(f"\n  #{i} at {p['book']}: decimal {p['price']:.2f} - hits {p['chance']:.0%}, expected profit "
              f"{p['ev']:+.0%}, bet CA${p['stake']:.2f}")
        for leg in p["legs"]:
            print(f"       {pd.Timestamp(leg['game_date']).strftime('%a %b %d')}  {leg['player']:24} {leg['bet']:16} "
                  f"{leg['game']}")
    week = pd.DataFrame([{"book": p["book"], "price": p["price"], "chance": p["chance"], "ev": p["ev"],
                          "stake": p["stake"], "legs": json.dumps(p["legs"], default=float), "span": "week"}
                         for p in best], columns=["book", "price", "chance", "ev", "stake", "legs", "span"])
    nightly = pd.read_csv(prefix + "parlays_today.csv") if merged["parlays_today.csv"] else week.iloc[0:0]
    nightly["span"] = "day"
    pd.concat([week, nightly], ignore_index=True).to_csv(prefix + "parlays_today.csv", index=False)
    # the Fumbles page grades these too (each leg on its own day; the parlay sits under its first day)
    save_found(today, [{"legs": p["legs"], "book": p["book"], "price": p["price"], "stake": p["stake"],
                        "game_date": p["legs"][0]["game_date"]} for p in best])
print(f"\nThis week: {total} bet(s) across {len(days)} day(s) - saved for the dashboard")
