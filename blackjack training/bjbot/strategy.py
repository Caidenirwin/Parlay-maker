"""Turn the bot's memory into a strategy file, and read it back (the dashboard uses this file).

Keys in "states":
  "total,soft,up,can_double,bucket"   e.g. "16,0,10,1,2" = hard 16 vs a 10, can double, neutral count
  "pair,card,up,bucket"               e.g. "pair,8,10,2"  = a pair of 8s vs a 10 (you can split), neutral count
(ace = 1, 10 / J / Q / K = 10)
"""
import json

from .learner import COUNT_BUCKETS, MOVES, N_STATES, TC_RANGE, best_policy, decode, legal_moves


def bet_units(edge):
    """how many betting units at a given edge: 1 when the shoe doesn't favour you, up to 8 as it gets hotter"""
    return 1 if edge <= 0 else max(1, min(8, round(1 + edge * 400)))


def state_key(s):
    d = decode(s)
    return ",".join(map(str, d[1:])) if d[0] == "total" else "pair," + ",".join(map(str, d[1:]))


def export(sums, weights, counts, bet_sums, bet_weights, bet_counts, path, meta):
    policy = best_policy(sums, weights, counts)
    states = {}
    for s in range(N_STATES):
        legal = legal_moves(s)
        evs = {MOVES[a]: round(sums[s * 4 + a] / weights[s * 4 + a], 4) for a in range(legal) if counts[s * 4 + a]}
        if evs:
            states[state_key(s)] = {"best": MOVES[policy[s]], "ev": evs,
                                    "tries": {MOVES[a]: counts[s * 4 + a] for a in range(legal)}}
    betting = {}
    for i, tc in enumerate(TC_RANGE):
        if bet_counts[i]:
            edge = bet_sums[i] / bet_weights[i]
            betting[str(tc)] = {"edge": round(edge, 5), "hands": bet_counts[i], "units": bet_units(edge)}
    with open(path, "w") as f:
        json.dump({"meta": meta, "count_buckets": COUNT_BUCKETS, "states": states, "betting": betting}, f)


def load(path):
    with open(path) as f:
        return json.load(f)
