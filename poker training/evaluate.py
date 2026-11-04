"""Evaluate the trained bot against baseline bots with duplicate poker.

    python evaluate.py --deals 20000 --workers 20

Each deal is played twice with seats swapped (same cards) to cut luck.
Reports big blinds per 100 hands with a 95% confidence interval.
"""
import argparse
import csv
import json
import math
import os
import multiprocessing as mp
import time

from pokerbot.bots import BASELINES, CFRBot
from pokerbot.engine import duplicate_match

_bp = {}


def _job(args):
    opp_name, n, seed, bp_path = args
    if bp_path not in _bp:
        import pickle
        with open(bp_path, "rb") as f:
            _bp[bp_path] = pickle.load(f)
    me = CFRBot(_bp[bp_path], seed=seed)
    if opp_name == "CFRBot(self)":
        opp = CFRBot(_bp[bp_path], seed=seed + 1)
    else:
        cls = BASELINES[opp_name]
        try:
            opp = cls(seed=seed + 1)
        except TypeError:
            opp = cls()
    res = duplicate_match(me, opp, n, seed=seed)
    return opp_name, res, me.on_tree, me.off_tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=20000, help="deals per opponent (x2 hands)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    ap.add_argument("--blueprint", default="checkpoints/blueprint.pkl")
    ap.add_argument("--opponents", default=",".join(BASELINES))
    ap.add_argument("--label", default=None,
                    help="checkpoint name to record in results/eval_history.csv "
                         "(default: total training hands)")
    args = ap.parse_args()

    opps = args.opponents.split(",")
    chunk = max(50, args.deals // args.workers)
    jobs = []
    for o in opps:
        left, i = args.deals, 0
        while left > 0:
            n = min(chunk, left)
            jobs.append((o, n, 1000 + i * 31 + hash(o) % 997, args.blueprint))
            left -= n
            i += 1
    t0 = time.time()
    results = {o: [] for o in opps}
    on = off = 0
    with mp.Pool(args.workers) as pool:
        for o, res, a, b in pool.imap_unordered(_job, jobs):
            results[o].extend(res)
            on += a
            off += b
    bb = 100
    train_hands = 0
    meta_path = os.path.join(os.path.dirname(args.blueprint), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            train_hands = json.load(f).get("iterations", 0)
    label = args.label or f"{train_hands / 1e6:.1f}M hands"
    history = []
    print(f"\n{'opponent':<16}{'hands':>9}{'bb/100':>10}{'  95% CI':>14}")
    print("-" * 49)
    for o in opps:
        r = results[o]
        n = len(r)
        per_hand = [x / 2 / bb for x in r]          # bb per hand (avg of the 2 seatings)
        m = sum(per_hand) / n
        var = sum((x - m) ** 2 for x in per_hand) / max(1, n - 1)
        ci = 1.96 * math.sqrt(var / n)
        print(f"{o:<16}{2 * n:>9,}{100 * m:>10.1f}{'+/- ' + format(100 * ci, '.1f'):>14}")
        history.append([label, train_hands, o, round(100 * m, 1), round(100 * ci, 1), 2 * n])
    os.makedirs("results", exist_ok=True)
    hpath = os.path.join("results", "eval_history.csv")
    new = not os.path.exists(hpath)
    with open(hpath, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["checkpoint", "train_hands", "opponent", "bb_per_100", "ci95", "hands_played"])
        w.writerows(history)
    print(f"appended to {hpath} as '{label}'")
    print(f"\non-tree decisions: {on:,}  fallback decisions: {off:,} "
          f"({100 * off / max(1, on + off):.1f}%)   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
