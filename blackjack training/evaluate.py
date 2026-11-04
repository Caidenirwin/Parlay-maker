"""How good is the bot? It plays fresh hands (no exploring) and is compared with the textbook basic strategy.

    python evaluate.py --hands 5000000 --workers 22

Prints: how often it agrees with the basic-strategy chart (pairs too), where it plays differently (and whether
the count changes its mind), its house edge with a 95% range, and the bet sizes it learned. Both play with the same
game engine the bot trained on, so it's a fair race.
"""
import argparse
import multiprocessing as mp
import os
import random

from bjbot.learner import MOVES, N_STATES, decode, play_hand
from bjbot.rules import Shoe
from bjbot.strategy import load, state_key

HERE = os.path.dirname(os.path.abspath(__file__))


def chart(t, soft, up, can_double):
    """the textbook basic strategy for this table (6 decks, dealer stands on 17) - up: ace = 11"""
    if soft:
        if t >= 19:
            return "S"
        if t == 18:   # soft 18: double vs 3-6 (stand there when you can't double), stand vs 2 / 7 / 8, hit vs 9-A
            return "D" if can_double and 3 <= up <= 6 else "S" if up <= 8 else "H"
        dbl = {13: (5, 6), 14: (5, 6), 15: (4, 6), 16: (4, 6), 17: (3, 6)}.get(t)
        return "D" if can_double and dbl and dbl[0] <= up <= dbl[1] else "H"
    if t >= 17:
        return "S"
    if t >= 13:
        return "S" if up <= 6 else "H"
    if t == 12:
        return "S" if 4 <= up <= 6 else "H"
    if t == 11:
        return "D" if can_double and up <= 10 else "H"
    if t == 10:
        return "D" if can_double and up <= 9 else "H"
    if t == 9:
        return "D" if can_double and 3 <= up <= 6 else "H"
    return "H"


def chart_split(card, up):
    """the textbook pair-splitting chart (6 decks, double after split allowed) - card: ace = 1, up: ace = 11"""
    if card in (1, 8):
        return True
    if card in (5, 10):
        return False
    if card == 9:
        return up not in (7, 10, 11)
    if card == 7:
        return up <= 7
    if card == 6:
        return up <= 6
    if card == 4:
        return up in (5, 6)
    return up <= 7   # 2s and 3s


def chart_move(s):
    d = decode(s)
    if d[0] == "pair":
        _, card, up, _ = d
        up11 = 11 if up == 1 else up
        if chart_split(card, up11):
            return "P"
        return chart(12, True, up11, True) if card == 1 else chart(card * 2, False, up11, True)
    _, t, soft, up, dbl, _ = d
    return chart(t, soft, 11 if up == 1 else up, dbl)


def _play(args):
    """(profit flat, profit with its own bet sizes, units bet) over `hands` hands - per starting bet"""
    policy, units, hands, seed = args
    rng = random.Random(seed)
    shoe = Shoe(rng)
    flat = counted = staked = 0.0
    for _ in range(hands):
        result, true, _, _ = play_hand(shoe, policy, rng, 0.0)
        size = units.get(min(max(round(true), -6), 10), 1)
        flat, counted, staked = flat + result, counted + result * size, staked + size
    return flat, counted, staked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=5_000_000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()
    bot = load(os.path.join(HERE, "checkpoints", "strategy.json"))
    print(f"the bot has trained on {bot['meta']['hands']:,} hands ({bot['meta'].get('seconds', 0) / 3600:.2f} h)")
    policy, book = [], []
    for s in range(N_STATES):
        st = bot["states"].get(state_key(s))
        book.append(MOVES.index(chart_move(s)))
        policy.append(MOVES.index(st["best"]) if st else book[-1])
    # 1. does it agree with the textbook? (neutral count, spots that really happen)
    agree = n = 0
    differ = []
    for s in range(N_STATES):
        st, d = bot["states"].get(state_key(s)), decode(s)
        if not st or d[-1] != 2 or sum(st["tries"].values()) < 20_000:
            continue
        n += 1
        mine, theirs = st["best"], MOVES[book[s]]
        if mine == theirs:
            agree += 1
            continue
        gap = st["ev"][mine] - st["ev"].get(theirs, 0)
        up_card = d[2] if d[0] == "pair" else d[3]
        up = "A" if up_card == 1 else up_card
        spot = (f"pair of {'A' if d[1] == 1 else d[1]}s vs {up}" if d[0] == "pair" else
                f"{'soft' if d[2] else 'hard'} {d[1]} vs {up}{'' if d[4] else ' (3+ cards)'}")
        differ.append((gap, f"{spot}: bot {mine}, chart {theirs} (+{gap:.3f})"))
    print(f"\nagrees with basic strategy in {agree} of {n} common spots (neutral count)")
    for _, line in sorted(differ, reverse=True)[:12]:
        print("   ", line)
    pairs = [s for s in range(N_STATES) if decode(s)[0] == "pair" and decode(s)[-1] == 2]
    split_agree = sum(policy[s] == book[s] for s in pairs)
    print(f"pairs: it splits / doesn't split like the chart in {split_agree} of {len(pairs)} pair spots")
    # 2. house edge on fresh hands (per starting bet, the usual way it's quoted)
    units = {int(tc): v["units"] for tc, v in bot["betting"].items()}
    per = args.hands // args.workers
    for pol, name in ((policy, "the bot"), (book, "basic-strategy chart")):
        with mp.Pool(args.workers) as pool:
            parts = pool.map(_play, [(pol, units, per, 7_000 + w) for w in range(args.workers)])
        flat, counted, staked = (sum(p[i] for p in parts) for i in range(3))
        hands = per * args.workers
        ci = 1.96 * 1.15 / hands ** 0.5 * 100
        print(f"\n{name:22}: flat bet {flat / hands * 100:+.2f} per 100 chips (+/- {ci:.2f})"
              + (f" | with its own count betting {counted / staked * 100:+.2f}" if name == "the bot" else ""))
    print("\nbet sizes it learned (true count: units):",
          "  ".join(f"{tc}: {v['units']}" for tc, v in sorted(bot["betting"].items(), key=lambda kv: int(kv[0]))
                    if v["hands"] > 100_000))


if __name__ == "__main__":
    main()
