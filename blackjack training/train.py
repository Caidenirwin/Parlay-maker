"""Train the blackjack bot by self-play (Monte Carlo reinforcement learning). Resumes automatically.

    python train.py --minutes 30 --workers 22

Every epoch, each worker plays a batch of hands with the bot's current strategy (plus some exploring), then all
the results are added to the bot's memory and its strategy is updated.

Linear discounting (the same trick as the poker bot's linear CFR): epoch t's results count t times as much as
epoch 1's. Early on the bot plays badly ("hit everything under 20"), so the results it saw then are unfair to moves
like Hit - weighting newer epochs more lets those early mistakes fade away.

Checkpoints go in checkpoints/:
  memory.json    every state + move it has tried: weighted total result, weight, and raw tries (used to resume)
  strategy.json  what it plays now - this is the file the dashboard reads
  meta.json      how many hands it has played, and for how long
"""
import argparse
import json
import multiprocessing as mp
import os
import time

from bjbot.learner import N_STATES, TC_RANGE, best_policy, run
from bjbot.strategy import export

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "checkpoints")


def load_memory():
    path = os.path.join(CK, "memory.json")
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        if m.get("version") == 3:   # version 3 = with splitting (older checkpoints are a different game)
            return m
    cells, tcs = N_STATES * 4, len(TC_RANGE)
    return {"sums": [0.0] * cells, "weights": [0.0] * cells, "counts": [0] * cells, "bet_sums": [0.0] * tcs,
            "bet_weights": [0.0] * tcs, "bet_counts": [0] * tcs, "hands": 0, "epoch": 0, "seconds": 0.0,
            "version": 3}


def _job(args):
    return run(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--hands", type=int, default=100_000, help="hands per worker per epoch")
    ap.add_argument("--explore", type=float, default=0.3, help="chance of trying a random move (once per hand)")
    args = ap.parse_args()
    os.makedirs(CK, exist_ok=True)
    m = load_memory()
    print(f"{'resumed' if m['hands'] else 'starting from zero'}: {m['hands']:,} hands so far, epoch {m['epoch']}",
          flush=True)
    end = time.time() + args.minutes * 60
    with mp.Pool(args.workers) as pool:
        while time.time() < end:
            t = time.time()
            policy = best_policy(m["sums"], m["weights"], m["counts"])
            weight = m["epoch"] + 1.0   # linear discounting: newer epochs count more
            jobs = [(policy, args.hands, m["epoch"] * 1000 + w, args.explore) for w in range(args.workers)]
            for s, c, bs, bc in pool.imap_unordered(_job, jobs):
                for k in range(len(s)):
                    if c[k]:
                        m["sums"][k] += s[k] * weight
                        m["weights"][k] += c[k] * weight
                        m["counts"][k] += c[k]
                for k in range(len(bs)):
                    if bc[k]:
                        m["bet_sums"][k] += bs[k] * weight
                        m["bet_weights"][k] += bc[k] * weight
                        m["bet_counts"][k] += bc[k]
            m["epoch"] += 1
            m["hands"] += args.hands * args.workers
            m["seconds"] += time.time() - t
            changed = sum(a != b for a, b in zip(policy, best_policy(m["sums"], m["weights"], m["counts"])))
            print(f"epoch {m['epoch']:5d} | {m['hands']:,} hands | {args.hands * args.workers / (time.time() - t):,.0f}"
                  f" hands/s | strategy changed in {changed} states | {m['seconds'] / 3600:.2f} h trained", flush=True)
            if m["epoch"] % 20 == 0:
                save(m)
    save(m)


def save(m):
    meta = {"hands": m["hands"], "epoch": m["epoch"], "seconds": round(m["seconds"]),
            "saved": time.strftime("%Y-%m-%d %H:%M")}
    tmp = os.path.join(CK, "memory.tmp")
    with open(tmp, "w") as f:
        json.dump(m, f)
    os.replace(tmp, os.path.join(CK, "memory.json"))   # never leave a half-written checkpoint
    export(m["sums"], m["weights"], m["counts"], m["bet_sums"], m["bet_weights"], m["bet_counts"],
           os.path.join(CK, "strategy.tmp"), meta)
    os.replace(os.path.join(CK, "strategy.tmp"), os.path.join(CK, "strategy.json"))
    with open(os.path.join(CK, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved checkpoints ({m['hands']:,} hands)", flush=True)


if __name__ == "__main__":
    main()
