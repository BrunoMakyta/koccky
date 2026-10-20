import math
# Calibrated elimination-risk parameters (change these to retune):
ELIMINATION_BASE = 2.00
ELIMINATION_PER_EXTRA_PLAYER = 0.50
import time
from itertools import product

# Existing limits for reveal. No Monte Carlo is used.
MAX_REVEAL_OPTIONS = 400
MAX_ROLL_HISTOGRAMS = 200000
REVEAL_MARGIN = 0.05
ALLOW_REVEAL_ALL = True

# New: policy for the next player. 'Sharpness' is a behavioral assumption,
# not a mathematical property of the dice.
OPPONENT_DECISION_SHARPNESS = 2.5
# Exact averaging over all complete opponent hands is affordable at this size.
MAX_EXACT_OPPONENT_HANDS = 1300
# Keep below engine's 3-second timeout (includes margin).
TIME_BUDGET_SECONDS = 1.8


def count_shown(s, face):
    return sum(
        die == face or (face != 1 and die == 1)
        for player_dice in s["shown"]
        for die in player_dice
    )


def count_known(s, face):
    amount = count_shown(s, face) + s["your_dice"].count(face)
    if face != 1:
        amount += s["your_dice"].count(1)
    return amount


def player_total(s, i):
    return s["hidden_counts"][i] + len(s["shown"][i])


# Extra loss applies ONLY when our own dice reach zero.
def elimination_penalty(s):
    alive = sum(player_total(s, i) > 0 for i in range(len(s["names"])))
    return ELIMINATION_BASE + ELIMINATION_PER_EXTRA_PLAYER * max(0, alive - 2)


def binomial_probabilities(N, p):
    return [
        math.comb(N, i) * p**i * (1 - p)**(N - i)
        for i in range(N + 1)
    ]


def expected_win_of_bid_with_need(s, need, face, probabilities, opponents):
    # Retained as a fast heuristic for ranking reveal candidates.
    my_dice = player_total(s, s["you"])
    rivals = max(1, len(opponents))
    transferred = (sum(min(2, player_total(s, i)) for i in opponents) / rivals)
    reward = transferred * (1 + 1 / rivals)
    expected_win = 0.0
    for i, probability in enumerate(probabilities[face]):
        if i < need:
            loss = min(need - i, my_dice)
            expected_win -= probability * (loss + (elimination_penalty(s) if loss >= my_dice else 0))
        else:
            expected_win += probability * reward
    return expected_win


def expected_win_of_bid(s, count, face, probabilities, opponents):
    return expected_win_of_bid_with_need(
        s, count - count_known(s, face), face, probabilities, opponents
    )


def expected_win_of_call(s, count, face, probabilities, opponents):
    count -= count_known(s, face)
    my_dice = player_total(s, s["you"])
    bidder_dice = player_total(s, s["bidder"])
    rivals = max(1, len(opponents))
    transferred = min(2, my_dice)
    penalty = transferred * (1 + 1 / rivals)
    expected_win = 0.0
    for i, probability in enumerate(probabilities[face]):
        if i < count:
            expected_win += probability * min(count - i, bidder_dice) / rivals
        else:
            expected_win -= probability * (penalty + (elimination_penalty(s) if transferred >= my_dice else 0))
    return expected_win


def next_alive(s, player):
    for j in range(1, len(s["names"])):
        i = (player + j) % len(s["names"])
        if player_total(s, i) > 0:
            return i
    return None


def chance_to_be_called(s, count, face):
    # One-step horizon boundary. From the future challenger's viewpoint,
    # compare calling now with the zero IMMEDIATE gain of raising a bid.
    # This is only used to estimate our opponent's next-bid alternative.
    next_player = next_alive(s, s["you"])
    if next_player is None:
        return 0.0
    n = s["hidden_counts"][next_player]
    N = sum(s["hidden_counts"]) - n
    p = 1 / 6 if face == 1 else 1 / 3
    shown = count_shown(s, face)
    dist = binomial_probabilities(N, p)
    stats = BinomialStats(dist)
    rivals = max(1, sum(player_total(s, i) > 0 for i in range(len(s["names"]))) - 1)
    wrong_penalty = min(2, player_total(s, next_player)) * (1 + 1 / rivals)
    bidder_cap = player_total(s, s["you"])
    chance = 0.0
    for k, prob_hand in enumerate(binomial_probabilities(n, p)):
        need = count - shown - k
        call_ev = stats.call_value(need, bidder_cap, wrong_penalty, rivals)
        call_prob = 1 / (1 + math.exp(-OPPONENT_DECISION_SHARPNESS * call_ev))
        chance += prob_hand * call_prob
    return chance


def minimum_raise(count, face, new_face):
    # Uses the user's rules: from ones to a normal face 2*count suffices.
    if new_face == 1:
        return count + 1 if face == 1 else count // 2 + 1
    if face == 1:
        return 2 * count
    return count if new_face > face else count + 1


class BinomialStats:
    """O(1) binomial expected loss/reward for every integer bid threshold."""
    def __init__(self, probabilities):
        self.n = len(probabilities) - 1
        self.cdf = [0.0]
        self.moment = [0.0]
        for i, value in enumerate(probabilities):
            self.cdf.append(self.cdf[-1] + value)
            self.moment.append(self.moment[-1] + i * value)

    def _cdf(self, threshold):
        if threshold > self.n:
            return 1.0
        return self.cdf[max(0, threshold)]

    def _moment(self, threshold):
        return self.moment[max(0, min(self.n + 1, threshold))]

    def expected_loss(self, need, cap):
        # E[min(need - J, cap) * 1{J < need}], J ~ binomial.
        a = need - cap
        return (cap * self._cdf(a)
                + need * (self._cdf(need) - self._cdf(a))
                - (self._moment(need) - self._moment(a)))

    def bid_value(self, need, gain, own_loss_cap, elimination_cost=0.0):
        # Elimination occurs when unknown matches J <= need - own_loss_cap.
        p_eliminated = self._cdf(need - own_loss_cap + 1)
        return (gain * (1 - self._cdf(need))
                - self.expected_loss(need, own_loss_cap)
                - elimination_cost * p_eliminated)

    def call_value(self, need, bidder_loss_cap, wrong_penalty, rivals):
        return self.expected_loss(need, bidder_loss_cap) / rivals - wrong_penalty * (1 - self._cdf(need))


def reroll_distributions(number, cache):
    if number in cache:
        return cache[number]
    factorial = [math.factorial(i) for i in range(number + 1)]
    multiplier = factorial[number] / (6 ** number)
    outcomes = []

    def generate(remaining, face, counts, denominator):
        if face == 6:
            outcomes.append((
                tuple(counts + [remaining]),
                multiplier / (denominator * factorial[remaining])
            ))
            return
        for amount in range(remaining + 1):
            generate(remaining - amount, face + 1,
                     counts + [amount], denominator * factorial[amount])
    generate(number, 1, [], 1)
    cache[number] = outcomes
    return outcomes


class BidModel:
    """Expected IMMEDIATE dice-value of our bid, integrating caller behavior and dice."""

    def __init__(self, s, roll_cache, next_call_cache):
        self.s = s
        self.you = s["you"]
        self.next = next_alive(s, self.you)
        self.total_hidden = sum(s["hidden_counts"])
        self.self_hidden = s["hidden_counts"][self.you]
        self.n = s["hidden_counts"][self.next] if self.next is not None else 0
        self.m = self.total_hidden - self.self_hidden - self.n
        self.rivals = max(1, sum(player_total(s, i) > 0 for i in range(len(s["names"]))) - 1)
        self.public = [count_shown(s, f) for f in range(1, 7)]
        self.ones = binomial_probabilities(self.m, 1 / 6)
        self.other = binomial_probabilities(self.m, 1 / 3)
        self.unknown_opp = self.total_hidden - self.n
        self.opp_ones = binomial_probabilities(self.unknown_opp, 1 / 6)
        self.opp_other = binomial_probabilities(self.unknown_opp, 1 / 3)
        self.other_stats = [BinomialStats(self.ones), BinomialStats(self.other)]
        self.opp_stats = [BinomialStats(self.opp_ones), BinomialStats(self.opp_other)]
        self.histograms = reroll_distributions(self.n, roll_cache) if math.comb(self.n+5, 5) <= MAX_EXACT_OPPONENT_HANDS else None
        self.next_call_cache = next_call_cache
        self.weight_cache = {}
        self.utility_cache = {}
        self.opponent_suffix_cache = {}

    def next_opponent_call(self, count, face):
        future_next = next_alive(self.s, self.next)
        if future_next is None:
            return 0.0
        key = (self.total_hidden, self.public[face-1],
               self.s["hidden_counts"][future_next], face == 1, count)
        if key not in self.next_call_cache:
            t = self.s.copy()
            t["you"] = self.next
            self.next_call_cache[key] = chance_to_be_called(t, count, face)
        return self.next_call_cache[key]

    def policy_weights(self, count, face):
        key = (count, face)
        if key in self.weight_cache:
            return self.weight_cache[key]

        s = self.s
        p = 1 / 6 if face == 1 else 1 / 3
        opp_total = player_total(s, self.next)
        our_total = player_total(s, self.you)
        next_next = next_alive(s, self.next)
        transfer = min(2, player_total(s, next_next)) if next_next is not None else 0
        reward_opp = transfer * (1 + 1 / self.rivals)
        penalty_opp = min(2, opp_total) * (1 + 1 / self.rivals)
        public = self.public[face - 1]

        opp_stats = self.opp_stats[0 if face == 1 else 1]
        call_by_k = [
            opp_stats.call_value(count - public - k, our_total,
                                 penalty_opp, self.rivals)
            for k in range(self.n + 1)
        ]

        # What the opponent can bid next, evaluated from THEIR point of view.
        # For each face, find the opponent's BEST quantity, not merely the
        # smallest legal increase. A suffix maximum makes later queries O(1).
        opponent_bid_tables = []
        upper = sum(player_total(s, i) for i in range(len(s["names"])))
        for new_face in range(1, 7):
            minimum = minimum_raise(count, face, new_face)
            if new_face not in self.opponent_suffix_cache:
                alternate_stats = self.opp_stats[0 if new_face == 1 else 1]
                public_new = self.public[new_face - 1]
                suffix = [[-math.inf] * (self.n + 1) for _ in range(upper + 2)]
                for bid_quantity in range(upper, 0, -1):
                    call_probability = self.next_opponent_call(bid_quantity, new_face)
                    values = suffix[bid_quantity]
                    next_values = suffix[bid_quantity + 1]
                    for h in range(self.n + 1):
                        immediate_value = call_probability * alternate_stats.bid_value(
                            bid_quantity - public_new - h, reward_opp, opp_total
                        )
                        values[h] = max(immediate_value, next_values[h])
                self.opponent_suffix_cache[new_face] = suffix
            if minimum <= upper:
                opponent_bid_tables.append(self.opponent_suffix_cache[new_face][minimum])
            else:
                opponent_bid_tables.append([-math.inf] * (self.n + 1))

        weights = [0.0] * (self.n + 1)
        sharpness = OPPONENT_DECISION_SHARPNESS
        if self.histograms is not None:
            # Integrate over ALL opponent private hand compositions, not just
            # the current face. This makes 'best other bid' hand-dependent.
            for hand, hand_probability in self.histograms:
                wild = hand[0]
                hits = wild if face == 1 else wild + hand[face - 1]
                best_alt = opponent_bid_tables[0][wild]
                for g in range(2, 7):
                    alternative = opponent_bid_tables[g-1][wild + hand[g-1]]
                    if alternative > best_alt:
                        best_alt = alternative
                diff = call_by_k[hits] - best_alt
                chance = 1 / (1 + math.exp(-sharpness * diff))
                weights[hits] += hand_probability * chance
        else:
            # With very large hands, avoid a huge full-hand enumeration.
            # Conditional means of the other faces approximate best response.
            distribution = binomial_probabilities(self.n, p)
            for k, probability in enumerate(distribution):
                if face == 1:
                    estimated_wild = k
                    remaining = (self.n - k) / 5
                else:
                    estimated_wild = k / 2
                    remaining = (self.n - k) / 4

                def table_at(tab, match):
                    lo = max(0, min(self.n, int(match)))
                    hi = min(self.n, lo + 1)
                    return tab[lo] * (hi - match) + tab[hi] * (match - lo) if hi != lo else tab[lo]

                best_alt = -math.inf
                for g in range(1, 7):
                    if g == 1:
                        matches = estimated_wild
                    elif g == face:
                        matches = k
                    else:
                        matches = estimated_wild + remaining
                    alt = table_at(opponent_bid_tables[g-1], matches)
                    best_alt = max(best_alt, alt)
                diff = call_by_k[k] - best_alt
                weights[k] = probability / (1 + math.exp(-sharpness * diff))

        self.weight_cache[key] = weights
        return weights

    def bid_value(self, count, face, our_hidden_matches):
        key = (count, face, our_hidden_matches)
        if key in self.utility_cache:
            return self.utility_cache[key]

        if self.next is None:
            return 0.0
        weights = self.policy_weights(count, face)
        need = count - self.public[face - 1] - our_hidden_matches
        reward = min(2, player_total(self.s, self.next)) * (1 + 1 / self.rivals)
        loss = player_total(self.s, self.you)
        other_stats = self.other_stats[0 if face == 1 else 1]
        value = sum(
            weight * other_stats.bid_value(need - k, reward, loss, elimination_penalty(self.s))
            for k, weight in enumerate(weights)
        )
        self.utility_cache[key] = value
        return value


def decide(s):
    legal = s["legal_bids"]
    if not legal:
        return {"action": "call"}

    start = time.monotonic()
    opponents = [i for i in range(len(s["names"]))
                 if i != s["you"] and player_total(s, i) > 0]
    N = sum(s["hidden_counts"]) - len(s["your_dice"])
    ones = binomial_probabilities(N, 1 / 6)
    others = binomial_probabilities(N, 1 / 3)
    probabilities = {1: ones, 2: others, 3: others,
                     4: others, 5: others, 6: others}

    roll_cache = {}
    next_call_cache = {}
    model = BidModel(s, roll_cache, next_call_cache)
    own_hits = [s["your_dice"].count(f) + (s["your_dice"].count(1) if f != 1 else 0)
                for f in range(1, 7)]

    # Our own bid and the caller's decision are evaluated JOINTLY.
    best_bid = max(legal, key=lambda b: model.bid_value(b[0], b[1], own_hits[b[1]-1]))
    best_value = model.bid_value(best_bid[0], best_bid[1], own_hits[best_bid[1]-1])

    call_value = -math.inf
    if s["bid"] is not None and s["can_call"]:
        call_value = expected_win_of_call(
            s, s["bid"][0], s["bid"][1], probabilities, opponents
        )

    if not s["must_raise"] and s["your_dice"]:
        groups = [[i for i, die in enumerate(s["your_dice"]) if die == face]
                  for face in range(1, 7)]
        minimum_to_beat = max(best_value, call_value) + REVEAL_MARGIN

        legal_by_face = {face: [] for face in range(1, 7)}
        for count, face in legal:
            legal_by_face[face].append(count)

        choices = []
        original_shown = model.public
        for amounts in product(*(range(len(group) + 1) for group in groups)):
            revealed_count = sum(amounts)
            if revealed_count == 0:
                continue
            if not ALLOW_REVEAL_ALL and revealed_count == len(s["your_dice"]):
                continue
            rerolled_count = len(s["your_dice"]) - revealed_count
            quick_score = -math.inf
            for face in range(1, 7):
                if not legal_by_face[face]:
                    continue
                public = original_shown[face-1] + amounts[face-1]
                if face != 1:
                    public += amounts[0]
                avg = rerolled_count * (1/6 if face == 1 else 1/3)
                need = min(legal_by_face[face]) - public - round(avg)
                quick_score = max(quick_score, expected_win_of_bid_with_need(
                    s, need, face, probabilities, opponents
                ))
            choices.append((quick_score, amounts))

        choices.sort(reverse=True)
        if len(choices) > MAX_REVEAL_OPTIONS:
            choices = choices[:MAX_REVEAL_OPTIONS]

        used_histograms = 0
        best_reveal = None

        for _, amounts in choices:
            if time.monotonic() - start > TIME_BUDGET_SECONDS:
                break

            revealed_count = sum(amounts)
            rerolled_count = len(s["your_dice"]) - revealed_count
            cost = math.comb(rerolled_count + 5, 5)
            if used_histograms + cost > MAX_ROLL_HISTOGRAMS:
                continue
            used_histograms += cost

            indices = [index for group, amount in zip(groups, amounts)
                       for index in group[:amount]]
            t = s.copy()
            t["shown"] = [dice[:] for dice in s["shown"]]
            t["hidden_counts"] = s["hidden_counts"][:]
            t["shown"][s["you"]].extend(
                face for face, amount in enumerate(amounts, start=1)
                for _ in range(amount)
            )
            t["hidden_counts"][s["you"]] -= revealed_count
            future_model = BidModel(t, roll_cache, next_call_cache)

            grids = [None] * 6
            for face in range(1, 7):
                if not legal_by_face[face]:
                    continue
                grids[face-1] = [
                    max(future_model.bid_value(count, face, matches)
                        for count in legal_by_face[face])
                    for matches in range(rerolled_count + 1)
                ]

            exact_value = 0.0
            for outcomes, probability in reroll_distributions(rerolled_count, roll_cache):
                wild = outcomes[0]
                best_after_roll = max(
                    grids[face - 1][wild if face == 1 else wild + outcomes[face-1]]
                    for face in range(1, 7) if grids[face - 1] is not None
                )
                exact_value += probability * best_after_roll

            if exact_value > minimum_to_beat:
                minimum_to_beat = exact_value
                best_reveal = indices

        if best_reveal is not None:
            return {"action": "reveal", "indices": best_reveal}

    if call_value > best_value:
        return {"action": "call"}
    return {"action": "bid", "count": best_bid[0], "face": best_bid[1]}
