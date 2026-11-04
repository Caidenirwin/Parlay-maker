"""Train the poker bot by parallel self-play (external-sampling MCCFR).

    python train.py --minutes 60 --workers 20

Resumes automatically from checkpoints/table.pkl. Writes
checkpoints/blueprint.pkl (the strategy the bot plays) every few epochs.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

from pokerbot import abstraction
from pokerbot.cfr import worker_run, merge, finalize_epoch, average_strategy


def save(table, meta, out):
    tmp = os.path.join(out, "table.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(out, "table.pkl"))
    bp = average_strategy(table)
    tmp = os.path.join(out, "blueprint.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(bp, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(out, "blueprint.pkl"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    ap.add_argument("--epoch-seconds", type=float, default=60)
    ap.add_argument("--discount-epochs", type=int, default=30,
                    help="apply linear-CFR discounting for this many epochs")
    ap.add_argument("--prune-after", type=int, default=5,
                    help="enable regret-based pruning after this many epochs")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    abstraction.thresholds()  # make sure bucket thresholds exist before forking
    tpath = os.path.join(args.out, "table.pkl")
    mpath = os.path.join(args.out, "meta.json")
    table, meta = {}, {"epoch": 0, "iterations": 0, "seconds": 0.0}
    if os.path.exists(tpath):
        with open(tpath, "rb") as f:
            table = pickle.load(f)
        with open(mpath) as f:
            meta = json.load(f)
        print(f"resumed: epoch {meta['epoch']}, {meta['iterations']:,} iterations, "
              f"{len(table):,} infosets")

    snap = os.path.join(args.out, "snapshot.pkl")
    t_stop = time.time() + args.minutes * 60
    with mp.Pool(args.workers) as pool:
        while time.time() < t_stop:
            e = meta["epoch"]
            t0 = time.time()
            if table:
                with open(snap, "wb") as f:
                    pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
                sp = snap
            else:
                sp = None
            secs = min(args.epoch_seconds, 10 + 5 * e,
                       max(5.0, t_stop - time.time()))
            prune = e >= args.prune_after
            jobs = [(sp, e, secs, e * 100003 + i * 7919 + 17, prune)
                    for i in range(args.workers)]
            n_it = 0
            for delta, n in pool.imap_unordered(worker_run, jobs):
                merge(table, delta)
                n_it += n
            disc = (e + 1) / (e + 2) if e < args.discount_epochs else None
            finalize_epoch(table, disc)
            dt = time.time() - t0
            meta["epoch"] = e + 1
            meta["iterations"] += n_it
            meta["seconds"] += dt
            print(f"epoch {e + 1:4d} | +{n_it:,} iters ({n_it / dt:,.0f}/s) | "
                  f"total {meta['iterations']:,} | infosets {len(table):,} | "
                  f"prune={prune} | {dt:.0f}s", flush=True)
            if (e + 1) % args.save_every == 0:
                save(table, meta, args.out)
    save(table, meta, args.out)
    if os.path.exists(snap):
        os.remove(snap)
    print("saved", os.path.join(args.out, "blueprint.pkl"))


if __name__ == "__main__":
    main()
