"""Play heads-up against the trained bot in the terminal.

    python play.py
Commands on your turn:  f = fold,  c = check/call,  r 600 = raise to 600 total,
                        a = all-in,  q = quit
"""
import random

from pokerbot.bots import CFRBot
from pokerbot.cards import cards_str, evaluate, category
from pokerbot.engine import play_hand

STACK, SB, BB = 10000, 50, 100


class Human:
    def new_hand(self, seat):
        self.seen = 0

    def act(self, obs):
        me = obs["seat"]
        for st, seat, kind, amt in obs["actions"][self.seen:]:
            if seat != me:
                word = {"fold": "folds", "call": "checks/calls", "raise": f"raises to {amt}"}[kind]
                print(f"  >> bot {word}")
        self.seen = len(obs["actions"]) + 1
        print(f"\n  board: [{cards_str(obs['board'])}]   your hand: {cards_str(obs['hole'])}")
        print(f"  pot {sum(obs['contrib'])}  you in {obs['contrib'][me]}  "
              f"bot in {obs['contrib'][1 - me]}  to call {obs['to_call']}")
        if obs["can_raise"]:
            print(f"  raise range: {obs['min_raise_to']} .. {obs['max_raise_to']}")
        while True:
            s = input("  your action> ").strip().lower().split()
            if not s:
                continue
            if s[0] == "q":
                raise SystemExit
            if s[0] == "f":
                return "fold", 0
            if s[0] == "c":
                return "call", 0
            if s[0] == "a":
                return "raise", obs["max_raise_to"]
            if s[0] == "r" and len(s) > 1 and s[1].isdigit():
                return "raise", int(s[1])
            print("  ? use f / c / r <amount> / a / q")


def main():
    bot = CFRBot(seed=None)
    human = Human()
    total = 0
    rng = random.Random()
    for hand in range(1, 10 ** 9):
        cs = rng.sample(range(52), 9)
        holes, board = (cs[0:2], cs[2:4]), cs[4:9]
        human_seat = hand % 2
        bots = [human, bot] if human_seat == 0 else [bot, human]
        print("\n" + "=" * 60)
        print(f"Hand {hand}: you are the {'button (SB)' if human_seat == 0 else 'big blind'}")
        log = []
        r = play_hand(bots, holes, board, STACK, SB, BB, log=log)
        st, seat, kind, amt = log[-1]
        if seat != human_seat:
            print(f"  >> bot {'folds' if kind == 'fold' else 'calls'}")
        won = r if human_seat == 0 else -r
        total += won
        hb = holes[1 - human_seat]
        print(f"  bot had {cards_str(hb)}   board {cards_str(board)}")
        print(f"  you: {category(evaluate(holes[human_seat] + board))}, "
              f"bot: {category(evaluate(hb + board))}")
        print(f"  result: {won:+d}   session total: {total:+d} ({total / BB:+.1f} bb)")


if __name__ == "__main__":
    main()
