import math

# Odhad pravdepodobnosti, ze super okamzite povie "Neverim!"
CHANCE_TO_BE_CALLED = 0.40



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



# Odhad hodnoty nasej ponuky, AK ju super spochybni
def expected_win_of_bid(s, count, face, probabilities, opponents):
    need = count - count_known(s, face)
    my_dice = player_total(s, s["you"])
    rivals = max(1, len(opponents))

    # Priemerna mozna strata vyzyvatela (najviac 2 kocky)
    transferred = (
        sum(min(2, player_total(s, i)) for i in opponents)
        / rivals
    )

    # Nas zisk + relativna strata supera
    reward = transferred * (1 + 1 / rivals)

    expected_win = 0.0

    for i, probability in enumerate(probabilities[face]):
        if i < need:
            expected_win -= probability * min(need - i, my_dice)
        else:
            expected_win += probability * reward

    return expected_win



# Odhad hodnoty toho, ze TERAZ povieme "Neverim!"
def expected_win_of_call(s, count, face, probabilities, opponents):
    need = count - count_known(s, face)
    my_dice = player_total(s, s["you"])
    bidder_dice = player_total(s, s["bidder"])
    rivals = max(1, len(opponents))

    transferred = min(2, my_dice)
    penalty = transferred * (1 + 1 / rivals)

    expected_win = 0.0

    for i, probability in enumerate(probabilities[face]):
        if i < need:
            # Super strati kocky
            expected_win += probability * min(need - i, bidder_dice) / rivals
        else:
            # My odovzdame kocky superovi
            expected_win -= probability * penalty

    return expected_win



def decide(s):
    legal = s["legal_bids"]

    if not legal:
        return {"action": "call"}

    # Pocitame iba superov, ktori este maju kocky
    opponents = [
        i for i in range(len(s["names"]))
        if i != s["you"] and player_total(s, i) > 0
    ]

    # Pocet neznamych kociek
    N = sum(s["hidden_counts"]) - len(s["your_dice"])

    # Binomicke rozdelenia vypocitame iba raz
    ones = binomial_probabilities(N, 1 / 6)
    others = binomial_probabilities(N, 1 / 3)

    probabilities = {
        1: ones,
        2: others,
        3: others,
        4: others,
        5: others,
        6: others
    }

    # Vyberieme najvyhodnejsiu povolenu ponuku
    best_bid = max(
        legal,
        key=lambda bid: (
            expected_win_of_bid(
                s, bid[0], bid[1], probabilities, opponents
            ),
            bid[0]
        )
    )

    best_value = expected_win_of_bid(
        s, best_bid[0], best_bid[1], probabilities, opponents
    )

    # Ak uz existuje ponuka a mozeme povedat "Neverim!"
    if s["bid"] is not None and s["can_call"]:
        current = s["bid"]

        call_value = expected_win_of_call(
            s, current[0], current[1], probabilities, opponents
        )

        # Uprednostnime vyhodne spochybnenie.
        # Pri ponuke zohladnime, ze super nemusi povedat "Neverim!"
        if call_value > 0 or call_value > CHANCE_TO_BE_CALLED * best_value:
            return {"action": "call"}

    return {
        "action": "bid",
        "count": best_bid[0],
        "face": best_bid[1]
    }