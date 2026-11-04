"""External-sampling Monte-Carlo CFR (with regret-based pruning) over the
abstract game, plus helpers to run it in parallel worker processes.

Table: key -> [regrets(list), strategy_sum(list)]
key  : f"{bucket}:{betting history}"
"""
import pickle
import random
import time

from . import abstraction
from .cards import evaluate
from .game import root

PRUNE_THRESHOLD = -2000.0   # skip exploring actions with regret below this...
PRUNE_PROB = 0.95           # ...this often (after warm-up)
REGRET_FLOOR = -50000.0     # regrets are clipped here so pruned actions can recover

_ROOT = root()
_BSTR = [str(i) + ":" for i in range(200)]


def regret_match(reg):
    s = 0.0
    for r in reg:
        if r > 0:
            s += r
    n = len(reg)
    if s <= 0:
        return [1.0 / n] * n
    return [(r / s if r > 0 else 0.0) for r in reg]


def sample_deal(rng):
    cs = rng.sample(range(52), 9)
    h0, h1, board = cs[0:2], cs[2:4], cs[4:9]
    b0 = abstraction.deal_buckets(h0, board, rng)
    b1 = abstraction.deal_buckets(h1, board, rng)
    v0, v1 = evaluate(h0 + board), evaluate(h1 + board)
    w = 1 if v0 > v1 else (-1 if v1 > v0 else 0)
    return (b0, b1), (w, -w)


class Trainer:
    """Runs MCCFR iterations against a read-only `base` table, writing into a
    copy-on-touch `local` table so the result can be returned as a delta."""

    def __init__(self, base, seed, prune=False):
        self.base = base
        self.local = {}
        self.rng = random.Random(seed)
        self.prune = prune

    def _entry(self, key, n):
        e = self.local.get(key)
        if e is None:
            b = self.base.get(key)
            if b is None:
                e = [[0.0] * n, [0.0] * n, None]
            else:
                e = [list(b[0]), list(b[1]), b]
            self.local[key] = e
        return e

    def traverse(self, node, p, bk, win, prune):
        if node.term:
            if node.term == 1:
                return -node.c[p] if node.folder == p else node.c[1 - p]
            return win[p] * node.c[0]
        cur = node.player
        acts = node.acts
        n = len(acts)
        e = self._entry(_BSTR[bk[cur][node.street]] + node.hist, n)
        reg = e[0]
        sigma = regret_match(reg)
        kids = node.kids
        if cur == p:
            utils = [0.0] * n
            explored = [True] * n
            ev = 0.0
            for i in range(n):
                k = kids[i]
                if prune and reg[i] < PRUNE_THRESHOLD and not k.term:
                    explored[i] = False
                    continue
                u = self.traverse(k, p, bk, win, prune)
                utils[i] = u
                ev += sigma[i] * u
            for i in range(n):
                if explored[i]:
                    reg[i] += utils[i] - ev
            return ev
        strat = e[1]
        for i in range(n):
            strat[i] += sigma[i]
        r = self.rng.random()
        acc = 0.0
        a = n - 1
        for i in range(n):
            acc += sigma[i]
            if r < acc:
                a = i
                break
        return self.traverse(kids[a], p, bk, win, prune)

    def iterate(self):
        bk, win = sample_deal(self.rng)
        rng = self.rng
        for p in (0, 1):
            pr = self.prune and rng.random() < PRUNE_PROB
            self.traverse(_ROOT, p, bk, win, pr)

    def delta(self):
        out = {}
        for k, (reg, st, b) in self.local.items():
            if b is None:
                out[k] = (reg, st)
            else:
                br, bs = b
                out[k] = ([reg[i] - br[i] for i in range(len(reg))],
                          [st[i] - bs[i] for i in range(len(st))])
        return out


# ----------------------------------------------------------------- workers
_cache = {"epoch": None, "base": None}


def worker_run(args):
    snap_path, epoch, seconds, seed, prune = args
    if _cache["epoch"] != epoch:
        if snap_path is None:
            _cache["base"] = {}
        else:
            with open(snap_path, "rb") as f:
                _cache["base"] = pickle.load(f)
        _cache["epoch"] = epoch
    tr = Trainer(_cache["base"], seed, prune)
    t_end = time.time() + seconds
    n = 0
    while True:
        for _ in range(20):
            tr.iterate()
        n += 20
        if time.time() >= t_end:
            break
    return tr.delta(), n


def merge(table, delta):
    for k, (dr, ds) in delta.items():
        e = table.get(k)
        if e is None:
            table[k] = [list(dr), list(ds)]
        else:
            r, s = e
            for i in range(len(dr)):
                r[i] += dr[i]
                s[i] += ds[i]


def finalize_epoch(table, discount=None):
    for r, s in table.values():
        for i in range(len(r)):
            v = r[i]
            if discount is not None:
                v *= discount
                s[i] *= discount
            if v < REGRET_FLOOR:
                v = REGRET_FLOOR
            r[i] = v


def average_strategy(table):
    """Blueprint: key -> tuple of probabilities (average strategy)."""
    out = {}
    for k, (r, s) in table.items():
        tot = sum(s)
        if tot > 0:
            out[k] = tuple(x / tot for x in s)
        else:
            out[k] = tuple(regret_match(r))
    return out
