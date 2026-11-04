"""Abstract heads-up no-limit hold'em betting tree (units = big blinds).

Seat 0 = button / small blind (acts first preflop, last postflop).
Seat 1 = big blind.
Contributions c[i] are the TOTAL chips put in this hand.

Action tokens:
  f  fold          c  check / call
  h  raise 1/2 pot p  raise pot      a  all-in
A raise of x pot adds x * (pot after calling) on top of the call.
"""

STACK = 100.0      # starting stack in big blinds (training abstraction)
SB, BB = 0.5, 1.0
SIZES = {"h": 0.5, "p": 1.0}
# raise menu indexed by number of raises already made on this street
PRE_MENU = [("h", "p", "a"), ("p", "a"), ("p", "a"), ("a",)]
POST_MENU = [("h", "p", "a"), ("p", "a"), ("a",)]


def raise_target(opp, last_inc, x):
    inc = max(x * 2.0 * opp, last_inc, BB)
    return opp + inc


class Node:
    __slots__ = ("street", "c", "player", "nraise", "nact", "last_inc",
                 "hist", "term", "folder", "acts", "_kids")

    def __init__(self, street, c, player, nraise, nact, last_inc, hist,
                 term=0, folder=-1):
        self.street = street
        self.c = c                  # (c0, c1)
        self.player = player
        self.nraise = nraise
        self.nact = nact
        self.last_inc = last_inc
        self.hist = hist
        self.term = term            # 0 = decision, 1 = fold, 2 = showdown
        self.folder = folder
        self.acts = None if term else self._legal()
        self._kids = None

    # ------------------------------------------------------------ legality
    def _legal(self):
        p = self.player
        mine, opp = self.c[p], self.c[1 - p]
        acts = ["f"] if opp > mine else []
        acts.append("c")
        if opp < STACK:
            menu = PRE_MENU if self.street == 0 else POST_MENU
            if self.nraise < len(menu):
                seen = set()
                for t in menu[self.nraise]:
                    if t == "a":
                        acts.append("a")
                        continue
                    tgt = raise_target(opp, self.last_inc, SIZES[t])
                    if tgt >= STACK or tgt in seen:
                        continue
                    seen.add(tgt)
                    acts.append(t)
        return tuple(acts)

    def target(self, tok):
        """Total contribution after taking raise token `tok`."""
        if tok == "a":
            return STACK
        opp = self.c[1 - self.player]
        return raise_target(opp, self.last_inc, SIZES[tok])

    def pot_fraction(self, tok):
        """Raise size expressed as fraction of pot-after-call."""
        opp = self.c[1 - self.player]
        return (self.target(tok) - opp) / (2.0 * opp)

    # ------------------------------------------------------------ children
    def _apply(self, tok):
        p = self.player
        c = list(self.c)
        opp = c[1 - p]
        if tok == "f":
            return Node(self.street, self.c, p, 0, 0, 0, self.hist + "f", 1, p)
        if tok == "c":
            c[p] = opp
            h = self.hist + "c"
            if self.nact + 1 >= 2:
                if opp >= STACK or self.street == 3:
                    return Node(self.street, tuple(c), p, 0, 0, 0, h, 2)
                return Node(self.street + 1, tuple(c), 1, 0, 0, BB, h + "/")
            return Node(self.street, tuple(c), 1 - p, self.nraise,
                        self.nact + 1, self.last_inc, h)
        tgt = self.target(tok)
        c[p] = tgt
        inc = tgt - opp
        return Node(self.street, tuple(c), 1 - p, self.nraise + 1,
                    self.nact + 1, max(inc, self.last_inc), self.hist + tok)

    @property
    def kids(self):
        k = self._kids
        if k is None:
            k = self._kids = [self._apply(t) for t in self.acts]
        return k

    def child(self, tok):
        return self.kids[self.acts.index(tok)]

    def __repr__(self):
        return f"Node(st={self.street} c={self.c} p={self.player} h='{self.hist}' acts={self.acts})"


def root():
    return Node(0, (SB, BB), 0, 0, 0, BB, "")


def count_tree(node=None):
    node = node or root()
    if node.term:
        return 0, 1
    d, t = 1, 0
    for k in node.kids:
        dd, tt = count_tree(k)
        d += dd
        t += tt
    return d, t
