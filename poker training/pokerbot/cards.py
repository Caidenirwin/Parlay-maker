"""Card encoding and a fast pure-Python 5-7 card hand evaluator.

Card = int 0..51, rank = c >> 2 (0 = deuce ... 12 = ace), suit = c & 3.
evaluate(cards) returns an int; higher is a better hand.
"""
import random

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_from_str(s):
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def cards_str(cs):
    return " ".join(card_str(c) for c in cs)


# ---------------------------------------------------------------- tables
POP = [bin(m).count("1") for m in range(8192)]
STRAIGHT_HIGH = [-1] * 8192
RANKS_DESC = [None] * 8192
TOP5 = [0] * 8192
for _m in range(8192):
    for _hi in range(12, 3, -1):
        _need = 0b11111 << (_hi - 4)
        if _m & _need == _need:
            STRAIGHT_HIGH[_m] = _hi
            break
    else:
        _wheel = (1 << 12) | 0b1111
        if _m & _wheel == _wheel:
            STRAIGHT_HIGH[_m] = 3
    _rs = tuple(r for r in range(12, -1, -1) if _m >> r & 1)
    RANKS_DESC[_m] = _rs
    _v = 0
    for _i in range(5):
        _v = (_v << 4) | (_rs[_i] if _i < len(_rs) else 0)
    TOP5[_m] = _v

CAT_NAMES = ["High card", "Pair", "Two pair", "Trips", "Straight",
             "Flush", "Full house", "Quads", "Straight flush"]


def evaluate(cards):
    counts = [0] * 13
    s0 = s1 = s2 = s3 = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        s = c & 3
        if s == 0:
            s0 |= 1 << r
        elif s == 1:
            s1 |= 1 << r
        elif s == 2:
            s2 |= 1 << r
        else:
            s3 |= 1 << r
    for sm in (s0, s1, s2, s3):
        if POP[sm] >= 5:
            sh = STRAIGHT_HIGH[sm]
            if sh >= 0:
                return (8 << 20) | (sh << 16)
            # with <= 7 cards a flush cannot coexist with quads / full house
            return (5 << 20) | TOP5[sm]
    rm = s0 | s1 | s2 | s3
    quad = -1
    trips = []
    pairs = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n >= 2:
            if n == 4:
                quad = r
            elif n == 3:
                trips.append(r)
            else:
                pairs.append(r)
    if quad >= 0:
        k = (rm & ~(1 << quad)).bit_length() - 1
        return (7 << 20) | (quad << 16) | (max(k, 0) << 12)
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        pr = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > pr:
            pr = pairs[0]
        return (6 << 20) | (t << 16) | (pr << 12)
    sh = STRAIGHT_HIGH[rm]
    if sh >= 0:
        return (4 << 20) | (sh << 16)
    if trips:
        t = trips[0]
        ks = RANKS_DESC[rm & ~(1 << t)]
        return (3 << 20) | (t << 16) | (ks[0] << 12) | (ks[1] << 8)
    if len(pairs) >= 2:
        a, b = pairs[0], pairs[1]
        k = (rm & ~(1 << a) & ~(1 << b)).bit_length() - 1
        return (2 << 20) | (a << 16) | (b << 12) | (max(k, 0) << 8)
    if pairs:
        a = pairs[0]
        ks = RANKS_DESC[rm & ~(1 << a)] + (0, 0, 0)
        return (1 << 20) | (a << 16) | (ks[0] << 12) | (ks[1] << 8) | (ks[2] << 4)
    return TOP5[rm]


def category(value):
    return CAT_NAMES[value >> 20]


def hand_strength(hole, board, samples, rng=random):
    """Monte-Carlo equity of `hole` vs. one uniformly random opponent hand,
    rolling out the rest of the board. Ties count half."""
    known = set(hole)
    known.update(board)
    deck = [c for c in range(52) if c not in known]
    need = 5 - len(board)
    k = 2 + need
    board = list(board)
    hole = list(hole)
    wins = 0.0
    sample = rng.sample
    for _ in range(samples):
        s = sample(deck, k)
        b = board + s[2:]
        my = evaluate(hole + b)
        op = evaluate(s[:2] + b)
        if my > op:
            wins += 1.0
        elif my == op:
            wins += 0.5
    return wins / samples
