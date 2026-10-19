import math
from itertools import product
import time

# Pri viac nez 400 roznych moznostiach vylozenia predvyberame kandidatov.
MAX_REVEAL_OPTIONS = 400
MAX_ROLL_HISTOGRAMS = 200000
REVEAL_MARGIN = 0.05
ALLOW_REVEAL_ALL = True


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


def binomial_probabilities(N, p):
    return [
        math.comb(N, i) * p**i * (1 - p)**(N - i)
        for i in range(N + 1)
    ]


def expected_win_of_bid_with_need(s, need, face, probabilities, opponents):
    my_dice = player_total(s, s["you"])
    rivals = max(1, len(opponents))
    transferred = (
        sum(min(2, player_total(s, i)) for i in opponents)
        / rivals
    )
    reward = transferred * (1 + 1 / rivals)

    expected_win = 0.0
    for i, probability in enumerate(probabilities[face]):
        if i < need:
            expected_win -= probability * min(need - i, my_dice)
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
            expected_win -= probability * penalty
    return expected_win


def chance_to_be_called(s, count, face):
    next_player = None
    for j in range(1, len(s["names"])):
        i = (s["you"] + j) % len(s["names"])
        if player_total(s, i) > 0:
            next_player = i
            break

    if next_player is None:
        return 0.0

    n = s["hidden_counts"][next_player]
    N = sum(s["hidden_counts"]) - n
    p = 1/6 if face == 1 else 1/3
    shown = count_shown(s, face)

    probabilities = binomial_probabilities(N, p)
    cumulative = [0.0]
    for prob in probabilities:
        cumulative.append(cumulative[-1] + prob)

    chance = 0.0
    for k, prob_hand in enumerate(binomial_probabilities(n, p)):
        needed = count - shown - k
        prob_bluff = cumulative[max(0, min(needed, N + 1))]
        prob_call = 1 / (1 + math.exp(-12 * (prob_bluff - 0.7)))
        chance += prob_hand * prob_call

    return chance


def reroll_distributions(number, cache):
    """Presne rozdelenie poctov 1,2,...,6 po prehodeni 'number' kociek."""
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
            generate(
                remaining - amount,
                face + 1,
                counts + [amount],
                denominator * factorial[amount]
            )

    generate(number, 1, [], 1)
    cache[number] = outcomes
    return outcomes


def decide(s):
    tid = time.time()
    legal = s["legal_bids"]
    if not legal:
        return {"action": "call"}

    opponents = [
        i for i in range(len(s["names"]))
        if i != s["you"] and player_total(s, i) > 0
    ]

    N = sum(s["hidden_counts"]) - len(s["your_dice"])
    ones = binomial_probabilities(N, 1 / 6)
    others = binomial_probabilities(N, 1 / 3)
    probabilities = {1: ones, 2: others, 3: others,
                     4: others, 5: others, 6: others}

    # Vsetky vyhodnotenia pouzivaju rovnaku hodnotu:
    # P(super povie Neverim) * EV(ponuky, ak ju spochybni).
    payoff_cache = {}
    call_chance_cache = {}

    def bid_payoff(need, face):
        key = (face == 1, need)
        if key not in payoff_cache:
            payoff_cache[key] = expected_win_of_bid_with_need(
                s, need, face, probabilities, opponents
            )
        return payoff_cache[key]

    def called_probability(t, count, face, public_count):
        # Po reveal sa meni iba celkovy pocet skrytych a verejna
        # hranica ponuky. Tieto tri udaje urcuju vysledok modelu.
        key = (sum(t["hidden_counts"]), face == 1, count - public_count)
        if key not in call_chance_cache:
            call_chance_cache[key] = chance_to_be_called(t, count, face)
        return call_chance_cache[key]

    legal_by_face = {face: [] for face in range(1, 7)}
    for count, face in legal:
        legal_by_face[face].append(count)

    best_bid = None
    best_value = -math.inf
    for count, face in legal:
        public = count_shown(s, face)
        chance = called_probability(s, count, face, public)
        value = chance * bid_payoff(count - count_known(s, face), face)
        if value > best_value:
            best_value = value
            best_bid = (count, face)

    call_value = -math.inf
    if s["bid"] is not None and s["can_call"]:
        call_value = expected_win_of_call(
            s, s["bid"][0], s["bid"][1], probabilities, opponents
        )

    # Vylozenie skusame len pred povinnym navysenim, ked mame kocky.
    if not s["must_raise"] and s["your_dice"]:
        groups = [
            [i for i, die in enumerate(s["your_dice"]) if die == face]
            for face in range(1, 7)
        ]
        roll_cache = {}
        binom_cache = {}
        candidates = []
        original_shown = [count_shown(s, face) for face in range(1, 7)]
        minimum_to_beat = max(best_value, call_value) + REVEAL_MARGIN

        # Rozne indexy rovnakej hodnoty vytvaraju rovnaky vysledok.
        # Pri vela kockach spravime najprv rychly predvyber bez prehadzovania.
        choices = []
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
                public = original_shown[face - 1] + amounts[face - 1]
                if face != 1:
                    public += amounts[0]
                average_hits = rerolled_count * (1/6 if face == 1 else 1/3)
                need = min(legal_by_face[face]) - public - round(average_hits)
                quick_score = max(quick_score, bid_payoff(need, face))
            choices.append((quick_score, amounts))

        if len(choices) > MAX_REVEAL_OPTIONS:
            choices.sort(reverse=True)
            choices = choices[:MAX_REVEAL_OPTIONS]

        for _, amounts in choices:
            revealed_count = sum(amounts)
            rerolled_count = len(s["your_dice"]) - revealed_count
            revealed_indices = [
                index
                for group, amount in zip(groups, amounts)
                for index in group[:amount]
            ]

            # Zmenime len verejny stav potrebny na model reakcie supera.
            t = s.copy()
            t["shown"] = [dice[:] for dice in s["shown"]]
            t["hidden_counts"] = s["hidden_counts"][:]
            t["shown"][s["you"]].extend(
                face for face, amount in enumerate(amounts, start=1)
                for _ in range(amount)
            )
            t["hidden_counts"][s["you"]] -= revealed_count

            # Pre kazde face a mozny pocet zhodnych prehodenych kociek
            # vypocitame najlepsiu ponuku. Robime to len RAZ.
            grids = [None] * 6
            for face in range(1, 7):
                if not legal_by_face[face]:
                    continue

                public = original_shown[face - 1] + amounts[face - 1]
                if face != 1:
                    public += amounts[0]

                bid_choices = [
                    (count, called_probability(t, count, face, public))
                    for count in legal_by_face[face]
                ]
                grids[face - 1] = [
                    max(
                        chance * bid_payoff(count - public - matches, face)
                        for count, chance in bid_choices
                    )
                    for matches in range(rerolled_count + 1)
                ]

            # Lacny analyticky dolny odhad: vyberieme jednu hodnotu
            # este pred prehodenim a spriemerujeme jej mozne vysledky.
            # Skutocny reveal je aspon taky dobry, lebo face mozeme
            # vybrat az po prehodeni.
            lower_bound = -math.inf
            for face in range(1, 7):
                grid = grids[face - 1]
                if grid is None:
                    continue
                p = 1/6 if face == 1 else 1/3
                key = (rerolled_count, face == 1)
                if key not in binom_cache:
                    binom_cache[key] = binomial_probabilities(rerolled_count, p)
                mean = sum(
                    probability * value
                    for probability, value in zip(binom_cache[key], grid)
                )
                lower_bound = max(lower_bound, mean)

            candidates.append((lower_bound, revealed_indices, grids, rerolled_count))

        # Ked je kandidatov viac, hodnotime najskor vyssie odhady.
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_histograms = 0
        best_reveal = None
        for lower_bound, indices, grids, rerolled_count in candidates:
            cost = math.comb(rerolled_count + 5, 5)
            if used_histograms + cost > MAX_ROLL_HISTOGRAMS:
                continue
            used_histograms += cost
            exact_value = 0.0
            for amounts, probability in reroll_distributions(rerolled_count, roll_cache):
                ones_rolled = amounts[0]
                best_after_roll = max(
                    grids[face - 1][
                        ones_rolled if face == 1 else ones_rolled + amounts[face - 1]
                    ]
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

    timeDelta = time.time() - tid
    #print(timeDelta)
    return {"action": "bid", "count": best_bid[0], "face": best_bid[1]}