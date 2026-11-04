# Self-taught blackjack bot

The bot teaches itself blackjack by playing hundreds of millions of hands at the dashboard's table, using
**Monte Carlo control** (reinforcement learning). It starts knowing nothing ("stand on 20 or 21, hit everything
else"). No basic-strategy chart and no card-counting rules are given to it.

## Quick start

```
python train.py --minutes 60 --workers 22     # train (resumes automatically)
python evaluate.py --hands 5000000             # compare it with the textbook chart
```

The dashboard reads `checkpoints/strategy.json`. In the Card Room's blackjack, pick **Coach: Your trained bot**.
It picks up new checkpoints by itself, even while training is running.

## How it works

| Piece | File | What it does |
|---|---|---|
| Table rules | `bjbot/rules.py` | The dashboard's rules: 6 decks, dealer stands on 17, blackjack pays 3:2, double on two cards (after a split too), split any pair up to 4 hands (split aces get one card each). Hi-Lo running count of every card seen |
| Learning | `bjbot/learner.py` | State = your total, soft / hard, dealer's up card, can double, true-count bucket (3,600 states), plus 500 pair states (the pair's card, up card, count) where **Split** is a 4th move. A split is worth the total result of every hand it makes. For each state and move it keeps the average result. It plays the best move so far, but tries a random move now and then (exploring) |
| Training | `train.py` | 20+ worker processes play batches of hands, the results are merged, and the strategy is updated. **Linear discounting**: epoch t counts t times as much, so results from its early, bad play fade away (the same idea as linear CFR in the poker bot) |
| Strategy file | `bjbot/strategy.py` | Best move + average result of every move in every state, plus the **bet sizes** it learned |
| Benchmark | `evaluate.py` | Agreement with basic strategy, where the count changes its mind, house edge on fresh hands |

## What it found by itself
- **Basic strategy**: it ends up agreeing with the textbook chart almost everywhere, and plays it just as well.
- **Splitting**: it learns the pair-splitting chart by itself: always split aces and 8s, never split 10s or 5s, and
  split 9s against everything except a 7, 10 or ace.
- **A mistake in the dashboard's chart**: with soft 18 against a 3, 4, 5 or 6 and 3 or more cards (so you can't
  double), you should *stand*, not hit. The chart has been fixed.
- **Final check** (6.0 billion hands of training, tested on 10 million fresh hands): it agrees with the textbook in
  531 of 542 common spots, and every one of the 11 differences is a near-tie (at most 0.007 chips apart). Flat
  betting it loses 0.38 per 100 chips, the same as the chart (0.35, +/- 0.07). With its own count betting it
  **wins 0.70 per 100 chips**.
- **Card counting**: it measures the edge at every true count, finds that a hot shoe favours the player, and bets more
  there (1 unit at a neutral count, up to 8 when the count is high). It also learns the famous count-based plays,
  such as doubling 9 against a 2 or standing on 15 against a 10 when the count is high.

For learning only. Online blackjack in Ontario reshuffles every hand, so counting doesn't work there, and casinos can
refuse service to counters.
