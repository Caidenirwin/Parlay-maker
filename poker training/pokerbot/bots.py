"""Bots: the trained CFR bot plus baseline opponents for evaluation."""
import pickle
import random

from . import abstraction
from .cards import hand_strength
from .game import root

# ====================================================== helper / fallback


def pot(obs):
    return obs["contrib"][0] + obs["contrib"][1]


def heuristic_action(obs, hs, rng=random):
    """Simple equity-driven policy, used as a fallback when the game
    leaves the abstraction (and as a baseline opponent)."""
    to_call = obs["to_call"]
    p = pot(obs)
    me = obs["seat"]
    opp = obs["contrib"][1 - me]
    if to_call > 0:
        odds = to_call / (p + to_call)
        if hs > 0.82 and obs["can_raise"]:
            return "raise", opp + 2 * opp * 0.75
        if hs > odds + 0.08:
            return "call", 0
        return "fold", 0
    if hs > 0.68 and obs["can_raise"]:
        return "raise", opp + 2 * opp * 0.66
    if hs < 0.25 and obs["can_raise"] and rng.random() < 0.15:     # small bluff
        return "raise", opp + 2 * opp * 0.5
    return "call", 0


# ============================================================ CFR bot


def _translate(x, opts, rng):
    """Pseudo-harmonic action translation (Ganzfried & Sandholm 2013).
    opts: sorted list of (pot_fraction, token)."""
    if x <= opts[0][0]:
        return opts[0][1]
    if x >= opts[-1][0]:
        return opts[-1][1]
    for (a, ta), (b, tb) in zip(opts, opts[1:]):
        if a <= x <= b:
            pa = ((b - x) * (1 + a)) / ((b - a) * (1 + x))
            if rng is None:
                return ta if pa >= 0.5 else tb
            return ta if rng.random() < pa else tb
    return opts[-1][1]


class CFRBot:
    name = "CFRBot"

    def __init__(self, blueprint="checkpoints/blueprint.pkl", purify=0.03,
                 seed=None):
        if isinstance(blueprint, str):
            with open(blueprint, "rb") as f:
                blueprint = pickle.load(f)
        self.bp = blueprint
        self.purify = purify
        self.rng = random.Random(seed)
        self.root = root()
        self.on_tree = 0
        self.off_tree = 0
        self._bc = {}

    def new_hand(self, seat):
        self._bc = {}

    def _bucket(self, hole, board):
        k = (tuple(hole), tuple(board))
        b = self._bc.get(k)
        if b is None:
            b = self._bc[k] = abstraction.bucket(hole, board, self.rng)
        return b

    def _hs(self, hole, board):
        k = ("hs", tuple(hole), tuple(board))
        v = self._bc.get(k)
        if v is None:
            v = self._bc[k] = hand_strength(hole, board, 300, self.rng)
        return v

    def _sync(self, obs):
        """Replay the real action history through the abstract tree."""
        bb = obs["bb"]
        me = obs["seat"]
        node = self.root
        rc = [obs["sb"] / bb, 1.0]
        hole_key = tuple(obs["hole"])
        for idx, (st, seat, kind, amt) in enumerate(obs["actions"]):
            if node.term or node.player != seat or node.street != st:
                return None
            if kind == "fold":
                tok = "f"
            elif kind == "call":
                tok = "c"
            else:
                raises = [t for t in node.acts if t not in ("f", "c")]
                if not raises:
                    return None
                if amt >= obs["stack"]:
                    tok = "a"
                else:
                    opp = rc[1 - seat]
                    x = (amt / bb - opp) / (2.0 * opp)
                    opts = sorted((node.pot_fraction(t), t) for t in raises)
                    r = None if seat == me else random.Random(hash((hole_key, idx, amt)))
                    tok = _translate(x, opts, r)
            if tok not in node.acts:
                return None
            node = node.child(tok)
            rc[seat] = amt / bb
        return node

    def act(self, obs):
        node = self._sync(obs)
        hole, board = obs["hole"], obs["board"]
        if (node is None or node.term or node.player != obs["seat"]
                or node.street != obs["street"]):
            self.off_tree += 1
            return heuristic_action(obs, self._hs(hole, board), self.rng)
        key = f"{self._bucket(hole, board)}:{node.hist}"
        probs = self.bp.get(key)
        if probs is None:
            self.off_tree += 1
            return heuristic_action(obs, self._hs(hole, board), self.rng)
        self.on_tree += 1
        if self.purify:
            q = [x if x >= self.purify else 0.0 for x in probs]
            s = sum(q)
            probs = [x / s for x in q] if s > 0 else probs
        r = self.rng.random()
        acc = 0.0
        tok = node.acts[-1]
        for t, pr in zip(node.acts, probs):
            acc += pr
            if r < acc:
                tok = t
                break
        return self.token_to_action(tok, node, obs)

    @staticmethod
    def token_to_action(tok, node, obs):
        if tok == "f":
            return ("fold", 0) if obs["to_call"] > 0 else ("call", 0)
        if tok == "c" or not obs["can_raise"]:
            return "call", 0
        if tok == "a":
            return "raise", obs["max_raise_to"]
        opp = obs["contrib"][1 - obs["seat"]]
        from .game import SIZES
        tgt = opp + SIZES[tok] * 2 * opp
        tgt = max(obs["min_raise_to"], min(obs["max_raise_to"], tgt))
        return "raise", tgt


# ====================================================== baseline bots


class CallBot:
    name = "CallingStation"

    def act(self, obs):
        return "call", 0


class RaiseBot:
    name = "Maniac"

    def act(self, obs):
        if obs["can_raise"]:
            opp = obs["contrib"][1 - obs["seat"]]
            return "raise", opp + 2 * opp
        return "call", 0


class RandomBot:
    name = "Random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, obs):
        r = self.rng.random()
        if obs["to_call"] > 0 and r < 0.2:
            return "fold", 0
        if r < 0.65 or not obs["can_raise"]:
            return "call", 0
        lo, hi = obs["min_raise_to"], obs["max_raise_to"]
        return "raise", lo + (hi - lo) * self.rng.random() ** 3


class HeuristicBot:
    """'Solid regular' style bot: plays by equity and pot odds."""
    name = "EquityBot"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._c = {}

    def new_hand(self, seat):
        self._c = {}

    def act(self, obs):
        k = (tuple(obs["hole"]), tuple(obs["board"]))
        hs = self._c.get(k)
        if hs is None:
            hs = self._c[k] = hand_strength(obs["hole"], obs["board"], 300, self.rng)
        return heuristic_action(obs, hs, self.rng)


class TightBot:
    """Nit: only plays premium-ish hands, then bets them hard."""
    name = "Nit"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._c = {}

    def new_hand(self, seat):
        self._c = {}

    def act(self, obs):
        k = (tuple(obs["hole"]), tuple(obs["board"]))
        hs = self._c.get(k)
        if hs is None:
            hs = self._c[k] = hand_strength(obs["hole"], obs["board"], 300, self.rng)
        opp = obs["contrib"][1 - obs["seat"]]
        if hs > 0.75 and obs["can_raise"]:
            return "raise", opp + 2 * opp
        if hs > 0.6 or obs["to_call"] == 0:
            return "call", 0
        return "fold", 0


BASELINES = {
    "Random": RandomBot,
    "CallingStation": CallBot,
    "Maniac": RaiseBot,
    "EquityBot": HeuristicBot,
    "Nit": TightBot,
}
