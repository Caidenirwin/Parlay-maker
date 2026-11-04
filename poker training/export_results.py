"""Turn the training logs into results/training_log.csv for the dashboard.

    python export_results.py

Reads train_log*.txt (in order) and writes one row per training epoch.
Evaluation results are appended to results/eval_history.csv by evaluate.py.
"""
import csv
import glob
import os
import re

LINE = re.compile(r"epoch\s+(\d+) \| \+([\d,]+) iters \(([\d,]+)/s\) \| total ([\d,]+) \| "
                  r"infosets ([\d,]+) \| prune=(\w+) \| (\d+)s")


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-16", "utf-8-sig"):
        try:
            txt = raw.decode(enc)
            if "epoch" in txt:
                return txt
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1")


def main():
    os.makedirs("results", exist_ok=True)
    logs = sorted(glob.glob("train_log*.txt"), key=lambda p: (len(p), p))
    rows, elapsed = [], 0.0
    for run, path in enumerate(logs, 1):
        for m in LINE.finditer(read_text(path)):
            epoch, added, rate, total, infosets, prune, secs = m.groups()
            elapsed += int(secs)
            rows.append({
                "run": run, "epoch": int(epoch),
                "total_iterations": int(total.replace(",", "")),
                "iters_per_sec": int(rate.replace(",", "")),
                "infosets": int(infosets.replace(",", "")),
                "elapsed_minutes": round(elapsed / 60, 2),
            })
    with open("results/training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote results/training_log.csv ({len(rows)} epochs from {len(logs)} logs)")


if __name__ == "__main__":
    main()
