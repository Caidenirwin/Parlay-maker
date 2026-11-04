"""The dashboard table's blackjack rules, built for speed: cards are plain numbers (ace = 1, 10 / J / Q / K = 10).

6 decks, reshuffled when fewer than 52 cards are left. Dealer stands on 17 (soft 17 too). Blackjack pays 3 to 2 and
the dealer checks for blackjack first. Double down on your first two cards (after a split too). Split any pair (any
two 10-value cards count as a pair) up to 4 hands; split aces get one card each; 21 after a split isn't a blackjack.
No surrender.
"""
import random

DECKS = 6
CUT = 52                 # shuffle a new shoe when fewer cards than this are left
MAX_HANDS = 4            # split up to 4 hands
HILO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}   # the Hi-Lo card count


def total(cards):
    """(best total, soft?) - an ace counts 11 when that doesn't bust (that's a 'soft' hand)"""
    t = sum(cards)
    if 1 in cards and t + 10 <= 21:
        return t + 10, True
    return t, False


class Shoe:
    """a 6-deck shoe that keeps the Hi-Lo running count of every card that's been SEEN"""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.shuffle()

    def shuffle(self):
        self.cards = [v for v in range(1, 10) for _ in range(4 * DECKS)] + [10] * (16 * DECKS)
        self.rng.shuffle(self.cards)
        self.running = 0

    def draw(self, seen=True):
        card = self.cards.pop()
        if seen:
            self.running += HILO[card]
        return card

    def see(self, card):
        self.running += HILO[card]

    def true_count(self):
        """running count / decks left - the number card counters bet by"""
        return self.running / max(len(self.cards) / 52, 0.5)


def dealer_plays(up, hole, shoe):
    cards = [up, hole]
    while total(cards)[0] < 17:
        cards.append(shoe.draw())
    return total(cards)[0]


def settle(player_total, dealer_total, bet):
    """the result in chips for a hand that wasn't a blackjack on the deal"""
    if player_total > 21:
        return -bet
    if dealer_total > 21 or player_total > dealer_total:
        return bet
    return 0 if player_total == dealer_total else -bet
