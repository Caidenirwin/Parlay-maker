"""How the bot learns: Monte Carlo control (reinforcement learning), by playing hand after hand.

A STATE is what a player can see when deciding:
    your total (4-21), soft or hard, the dealer's up card, whether you can still double, and how hot the shoe is
    (the Hi-Lo true count, in 5 buckets)
  or, holding a PAIR you're allowed to split: the pair's card, the dealer's up card and the count bucket.
For every state and every move (Stand / Hit / Double / sPlit) the bot keeps the total result and the number of times
it tried it - so the average is its estimate of "what this move is worth here". A split is worth the total result of
every hand it ends up making. The bot always plays the move with the best average so far, EXCEPT that it sometimes
tries a random move on purpose (exploring) - otherwise it could never find out that a move it's been avoiding is
actually better.

Only results that are fair are learned from: once a hand has had a random move, the decisions BEFORE it weren't
followed by the bot's real strategy, so they're skipped. It starts knowing nothing: "stand on 20 or 21, hit
everything else, never split" - the rest it has to discover.

It also learns BETTING: the average result of hands that start at each true count. Where that turns positive, the
shoe favours the player and the bot bets more (card counting, discovered from scratch).
"""
import random

from .rules import CUT, MAX_HANDS, Shoe, dealer_plays, settle, total

MOVES = "SHDP"                                  # stand, hit, double down, split
COUNT_BUCKETS = ["-2 or less", "-2 to 0", "0 to +2", "+2 to +4", "+4 or more"]
N_TOTAL_STATES = 18 * 2 * 10 * 2 * 5            # totals 4-21 x soft x up card x can double x count bucket
N_PAIR_STATES = 10 * 10 * 5                     # pair card (ace..10) x up card x count bucket
N_STATES = N_TOTAL_STATES + N_PAIR_STATES
TC_RANGE = range(-6, 11)                        # true counts tracked for betting (clipped to this range)


def count_bucket(true):
    return 0 if true <= -2 else 1 if true < 0 else 2 if true < 2 else 3 if true < 4 else 4


def state_index(player_total, soft, up, can_double, bucket):
    return ((((player_total - 4) * 2 + soft) * 10 + (up - 1)) * 2 + can_double) * 5 + bucket


def pair_index(card, up, bucket):
    return N_TOTAL_STATES + ((card - 1) * 10 + (up - 1)) * 5 + bucket


def decode(s):
    """('total', total, soft, up, can_double, bucket) or ('pair', card, up, bucket)"""
    if s >= N_TOTAL_STATES:
        s, bucket = divmod(s - N_TOTAL_STATES, 5)
        card, up = divmod(s, 10)
        return "pair", card + 1, up + 1, bucket
    s, bucket = divmod(s, 5)
    s, can_double = divmod(s, 2)
    s, up = divmod(s, 10)
    t, soft = divmod(s, 2)
    return "total", t + 4, soft, up + 1, can_double, bucket


def legal_moves(s):
    d = decode(s)
    return 4 if d[0] == "pair" else 3 if d[4] else 2


def starting_policy():
    """no blackjack knowledge at all: stand on 20 or 21, hit everything else, never split"""
    policy = []
    for s in range(N_STATES):
        d = decode(s)
        t = d[1] if d[0] == "total" else (12 if d[1] == 1 else d[1] * 2)   # a pair of aces is soft 12
        policy.append(0 if t >= 20 else 1)
    return policy


def best_policy(sums, weights, counts, min_tries=30):
    """the move with the best average result in every state (the starting rule where it hasn't learned enough).
    sums / weights = a WEIGHTED average: newer results count more (see train.py), counts = raw tries"""
    start = starting_policy()
    policy = []
    for s in range(N_STATES):
        legal = legal_moves(s)
        tried = [(sums[s * 4 + a] / weights[s * 4 + a], a) for a in range(legal) if counts[s * 4 + a] >= min_tries]
        policy.append(max(tried)[1] if len(tried) == legal else start[s])
    return policy


def play_hand(shoe, policy, rng, explore):
    """one round at this table. Returns (total result, true count at the deal, [(state*4+move, result)] to learn
    from, fair_for_betting)"""
    if len(shoe.cards) < CUT:
        shoe.shuffle()
    true_at_deal = shoe.true_count()
    player = [shoe.draw()]
    up = shoe.draw()
    player.append(shoe.draw())
    hole = shoe.draw(seen=False)            # face down: nobody counts it until it's turned over
    if total(player)[0] == 21 or total([up, hole])[0] == 21:   # a blackjack on the deal - no decisions
        shoe.see(hole)
        p21, d21 = total(player)[0] == 21, total([up, hole])[0] == 21
        return (0 if p21 and d21 else 1.5 if p21 else -1), true_at_deal, [], True
    outcomes = []          # (player total, bet) for every hand played this round, in order
    decisions = []         # [state*4+move, first outcome it's responsible for, one past the last]
    explored = [None]      # index into decisions where the random move happened
    n_hands = [1]

    def play(cards, split_aces=False):
        mine = []
        if split_aces:     # split aces: one card each, and that's it
            outcomes.append((total(cards)[0], 1))
            return
        can_double, bet = 1, 1
        while True:
            t, soft = total(cards)
            if t >= 21:
                break
            bucket = count_bucket(shoe.true_count())
            pair = len(cards) == 2 and cards[0] == cards[1] and n_hands[0] < MAX_HANDS
            s = pair_index(cards[0], up, bucket) if pair else state_index(t, soft, up, can_double, bucket)
            legal = 4 if pair else 3 if can_double else 2
            if explored[0] is None and rng.random() < explore:
                move, explored[0] = rng.randrange(legal), len(decisions)   # try something on purpose
            else:
                move = policy[s]
            rec = [s * 4 + move, len(outcomes), None]
            decisions.append(rec)
            mine.append(rec)
            if move == 3:   # split: two hands, each the pair card + a new card (doubling after a split is fine)
                n_hands[0] += 1
                card = cards[0]
                play([card, shoe.draw()], split_aces=card == 1)
                play([card, shoe.draw()], split_aces=card == 1)
                rec[2] = len(outcomes)
                return
            if move == 0:
                break
            cards.append(shoe.draw())
            can_double = 0
            if move == 2:
                bet = 2
                break
        outcomes.append((total(cards)[0], bet))
        for rec in mine:
            rec[2] = len(outcomes)

    play(player)
    shoe.see(hole)
    alive = any(t <= 21 for t, _ in outcomes)
    dealer = dealer_plays(up, hole, shoe) if alive else 0
    results = [settle(t, dealer, bet) for t, bet in outcomes]
    learn_from = decisions if explored[0] is None else decisions[explored[0]:]
    learn = [(k, sum(results[a:b])) for k, a, b in learn_from]
    return sum(results), true_at_deal, learn, explored[0] is None


def run(policy, hands, seed, explore=0.3):
    """a worker's batch of hands. Returns (sums, counts, bet_sums, bet_counts) to add to the bot's memory."""
    rng = random.Random(seed)
    shoe = Shoe(rng)
    sums, counts = [0.0] * (N_STATES * 4), [0] * (N_STATES * 4)
    bet_sums, bet_counts = [0.0] * len(TC_RANGE), [0] * len(TC_RANGE)
    lo, hi = TC_RANGE[0], TC_RANGE[-1]
    for i in range(hands):
        # every 4th hand: no exploring at all. Betting is learned ONLY from those - skipping hands that happened to
        # explore would be biased (hands with no decisions to explore, like a 20, are mostly winners)
        plain = i % 4 == 0
        result, true, learn, _ = play_hand(shoe, policy, rng, 0.0 if plain else explore)
        for k, r in learn:
            sums[k] += r
            counts[k] += 1
        if plain:
            i_tc = min(max(round(true), lo), hi) - lo
            bet_sums[i_tc] += result
            bet_counts[i_tc] += 1
    return sums, counts, bet_sums, bet_counts
