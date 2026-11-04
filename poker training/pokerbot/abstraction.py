"""Card abstraction: maps (hole cards, board) to a small bucket index.

Preflop : the 169 strategically distinct starting hands (lossless).
Postflop: Monte-Carlo hand strength vs. a random hand, cut into
          percentile buckets (finer resolution for strong hands).
"""
import json
import os
import random

from .cards import hand_strength

HS_SAMPLES = 100           # rollouts per bucket computation
N_POST_BUCKETS = 16
# percentile cut points -> 16 buckets, narrower near the top
QUANTILES = [0.12, 0.24, 0.35, 0.45, 0.54, 0.62, 0.69, 0.75, 0.80, 0.85,
             0.89, 0.92, 0.95, 0.97, 0.985]
_THRESH_FILE = os.path.join(os.path.dirname(__file__), "hs_thresholds.json")
_thresholds = None


def preflop_index(hole):
    a, b = hole
    ra, rb = a >> 2, b >> 2
    hi, lo = (ra, rb) if ra >= rb else (rb, ra)
    if hi == lo:
        return hi                                   # 0..12 pairs
    suited = (a & 3) == (b & 3)
    idx = hi * (hi - 1) // 2 + lo                   # 0..77
    return 13 + idx * 2 + (1 if suited else 0)      # 13..168


def preflop_name(idx):
    from .cards import RANKS
    if idx < 13:
        return RANKS[idx] * 2
    j = idx - 13
    suited = j & 1
    j >>= 1
    hi = 1
    while (hi + 1) * hi // 2 <= j:
        hi += 1
    lo = j - hi * (hi - 1) // 2
    return RANKS[hi] + RANKS[lo] + ("s" if suited else "o")


def compute_thresholds(n=3000, seed=1):
    rng = random.Random(seed)
    out = {}
    for street, nb in ((1, 3), (2, 4), (3, 5)):
        vals = []
        for _ in range(n):
            cs = rng.sample(range(52), 2 + nb)
            vals.append(hand_strength(cs[:2], cs[2:], HS_SAMPLES * 2, rng))
        vals.sort()
        out[str(street)] = [vals[int(q * (n - 1))] for q in QUANTILES]
    with open(_THRESH_FILE, "w") as f:
        json.dump(out, f, indent=1)
    return out


def thresholds():
    global _thresholds
    if _thresholds is None:
        if os.path.exists(_THRESH_FILE):
            with open(_THRESH_FILE) as f:
                raw = json.load(f)
        else:
            raw = compute_thresholds()
        _thresholds = {int(k): v for k, v in raw.items()}
    return _thresholds


def hs_to_bucket(street, hs):
    th = thresholds()[street]
    b = 0
    for t in th:
        if hs > t:
            b += 1
        else:
            break
    return b


def bucket(hole, board, rng=random):
    """Bucket for the current street (determined by len(board))."""
    if not board:
        return preflop_index(hole)
    street = {3: 1, 4: 2, 5: 3}[len(board)]
    return hs_to_bucket(street, hand_strength(hole, board, HS_SAMPLES, rng))


def deal_buckets(hole, board5, rng=random):
    """Buckets for all 4 streets for one player given the full 5-card board."""
    return (preflop_index(hole),
            hs_to_bucket(1, hand_strength(hole, board5[:3], HS_SAMPLES, rng)),
            hs_to_bucket(2, hand_strength(hole, board5[:4], HS_SAMPLES, rng)),
            hs_to_bucket(3, hand_strength(hole, board5, HS_SAMPLES, rng)))

