"""Real heads-up no-limit hold'em engine with arbitrary bet sizes (chips).

A bot is any object with  act(obs) -> (kind, amount)
  kind   : 'fold' | 'call' | 'raise'   ('call' also means check)
  amount : for 'raise', the TOTAL chips you will have put in this hand
Optional: bot.new_hand(seat) is called at the start of every hand.

obs (dict) given to the bot:
  seat          0 = button/small blind, 1 = big blind
  hole          [c1, c2]  (ints, see cards.py; use card_str to print)
  board         visible board cards
  street        0 preflop, 1 flop, 2 turn, 3 river
  contrib       [seat0 total chips in, seat1 total chips in]
  to_call       chips needed to call
  can_raise     bool
  min_raise_to, max_raise_to
  stack, sb, bb starting stack and blinds
  actions       list of (street, seat, kind, amount_to) so far this hand
"""
import random

from .cards import evaluate


def play_hand(bots, holes, board, stack=10000, sb=50, bb=100, log=None):
    """Play one hand. bots[0] is the button. Returns chips won by seat 0."""
    c = [sb, bb]
    street = 0
    player = 0
    nact = 0
    last_inc = bb
    actions = []
    for s, b in enumerate(bots):
        if hasattr(b, "new_hand"):
            b.new_hand(s)
    while True:
        p, o = player, 1 - player
        to_call = c[o] - c[p]
        can_raise = c[o] < stack
        min_to = min(c[o] + max(last_inc, bb), stack)
        nb = (0, 3, 4, 5)[street]
        obs = {
            "seat": p, "hole": list(holes[p]), "board": list(board[:nb]),
            "street": street, "contrib": list(c), "to_call": to_call,
            "can_raise": can_raise, "min_raise_to": min_to, "max_raise_to": stack,
            "stack": stack, "sb": sb, "bb": bb, "actions": list(actions),
        }
        kind, amt = bots[p].act(obs)
        if kind == "fold" and to_call <= 0:
            kind = "call"
        if kind == "raise" and not can_raise:
            kind = "call"
        if kind == "fold":
            actions.append((street, p, "fold", c[p]))
            if log is not None:
                log.append((street, p, "fold", c[p]))
            return c[p] if p == 1 else -c[0]
        if kind == "raise":
            amt = int(round(amt))
            amt = max(min_to, min(stack, amt))
            inc = amt - c[o]
            last_inc = max(last_inc, inc)
            c[p] = amt
            nact += 1
            actions.append((street, p, "raise", amt))
            if log is not None:
                log.append((street, p, "raise", amt))
            player = o
            continue
        # call / check
        c[p] = c[o]
        actions.append((street, p, "call", c[p]))
        if log is not None:
            log.append((street, p, "call", c[p]))
        nact += 1
        if nact >= 2:
            if c[p] >= stack or street == 3:
                v0 = evaluate(list(holes[0]) + list(board))
                v1 = evaluate(list(holes[1]) + list(board))
                if v0 > v1:
                    return c[1]
                if v1 > v0:
                    return -c[0]
                return 0
            street += 1
            nact = 0
            last_inc = bb
            player = 1
        else:
            player = o


def duplicate_match(bot_a, bot_b, n_deals, seed=0, stack=10000, sb=50, bb=100):
    """Play each random deal twice with seats swapped. Returns list of
    per-deal chip results for bot_a (sum of both seatings)."""
    rng = random.Random(seed)
    res = []
    for _ in range(n_deals):
        cs = rng.sample(range(52), 9)
        h = (cs[0:2], cs[2:4])
        board = cs[4:9]
        r1 = play_hand([bot_a, bot_b], h, board, stack, sb, bb)
        r2 = -play_hand([bot_b, bot_a], h, board, stack, sb, bb)
        res.append(r1 + r2)
    return res
