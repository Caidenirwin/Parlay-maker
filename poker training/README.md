# Self-play poker bot (Heads-up No-Limit Texas Hold'em)

The bot teaches itself poker by playing millions of hands against itself, using
**Monte Carlo Counterfactual Regret Minimization (MCCFR)**, the same family of
algorithms behind Libratus and Pluribus. No human hand histories are used.

## Quick start

```
python train.py --minutes 60 --workers 20     # train (resumes automatically)
python evaluate.py --deals 20000               # benchmark vs. baseline bots
python play.py                                 # play against it yourself
```

## How it works

| Piece | File | What it does |
|---|---|---|
| Hand evaluator | `pokerbot/cards.py` | Fast 7-card evaluator + Monte-Carlo equity |
| Card abstraction | `pokerbot/abstraction.py` | Preflop: all 169 starting hands. Postflop: 16 hand-strength buckets per street |
| Betting abstraction | `pokerbot/game.py` | Fold / check-call / ½-pot / pot / all-in, with raise caps. 100bb stacks |
| Training | `pokerbot/cfr.py`, `train.py` | External-sampling MCCFR, regret-based pruning, linear-CFR discounting, multi-process |
| Real game + bots | `pokerbot/engine.py`, `pokerbot/bots.py` | Full NLHE rules with any bet size; the CFR bot maps unusual bet sizes onto its tree (pseudo-harmonic action translation) |

Outputs are in `checkpoints/`:
- `blueprint.pkl` is the strategy the bot plays.
- `table.pkl` holds the raw regrets, used to resume training.
- `meta.json` records iteration counts.

## Plugging it into a bot competition

```python
from pokerbot.bots import CFRBot
from pokerbot.cards import card_from_str

bot = CFRBot("checkpoints/blueprint.pkl")
bot.new_hand(seat)                        # seat 0 = button/SB, 1 = big blind
kind, amount = bot.act({
    "seat": 0, "hole": [card_from_str("As"), card_from_str("Kd")],
    "board": [], "street": 0,
    "contrib": [50, 100],                 # total chips each seat has put in this hand
    "to_call": 50, "can_raise": True, "min_raise_to": 200, "max_raise_to": 10000,
    "stack": 10000, "sb": 50, "bb": 100,
    "actions": [],                        # (street, seat, 'fold'|'call'|'raise', amount_to)
})
# kind is 'fold' | 'call' | 'raise'; amount = total chips to have in after raising
```

Write a small adapter that turns your competition's messages into that dict.
The bot is stateless between calls: it rebuilds everything from `actions`.

## Making it stronger
- **Train longer**: `train.py` resumes from the checkpoint, so you can keep adding hours.
- More buckets (`N_POST_BUCKETS`, `QUANTILES` in `abstraction.py`) or more bet sizes
  (`PRE_MENU`/`POST_MENU` in `game.py`) give finer play but need more training.
  Changing either one means deleting `checkpoints/` and retraining from scratch.
- If the competition uses 200bb stacks, set `STACK = 200.0` in `game.py` and retrain.
