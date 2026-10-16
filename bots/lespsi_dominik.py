import random
import math

def count_shown(s, face):
    return sum(
        die == face or (face != 1 and die == 1)
        for player_dice in s["shown"]
        for die in player_dice
    )

    
def expected_win_of_call(s, count, face):
    N = sum(s["hidden_counts"]) - len(s["your_dice"])
    p = 1/6 if face == 1 else 1/3

    expected_win = 0

    for i in range(N + 1):
        probability = math.comb(N, i) * p**i * (1-p)**(N-i)

        if i < count:
            expected_win -= probability * (count-i)
        else:
            expected_win += probability * (2/(len(s["names"]) - 1))

    return expected_win

def expected_win_of_fold(s, count, face):
    N = sum(s["hidden_counts"]) - len(s["your_dice"])
    p = 1/6 if face == 1 else 1/3

    count -= count_shown(s, face) + s["your_dice"].count(face)
    if face != 1:
        count -= s["your_dice"].count(1)

    expected_win = 0

    for i in range(N + 1):
        probability = math.comb(N, i) * p**i * (1-p)**(N-i)

        if i < count:
            expected_win += probability * (count-i)/(len(s["names"]) - 1)
        else:
            expected_win -= probability * (2 + 2/(len(s["names"]) - 1))

    return expected_win

def decide(s):
    ## Do tejto procedúry dávaj, čo vráti ťah.
    ## s je stav hry a jediný parameter tejto procedúry (dict).
    ##
    ## --- Čo je v stave s ---
    ## s["must_raise"]    : True|False - či musíš navýšiť ponuku (po vyložení a prehodení kociek
    ##                      už nemôžeš povedať "Neverím!", musíš dať vyššiu ponuku)
    ## s["can_call"]      : True|False - či môžeš povedať "Neverím!" (ponuka existuje a nie je must_raise)
    ## s["bid"]           : aktuálna ponuka [počet, hodnota], alebo None ak otváraš kolo
    ## s["bidder"]        : index hráča, ktorý dal aktuálnu ponuku (alebo None)
    ## s["legal_bids"]    : zoznam všetkých povolených ponúk [[počet, hodnota], ...] podľa počtu a hodnoty;
    ##                      pravidlá o jednotkách (žolíkoch) sú už zohľadnené, takže stačí vybrať z nich
    ## s["your_dice"]     : tvoje skryté kocky (čísla 1-6); ich poradie = indexy pre "reveal"
    ## s["shown"]         : verejne vyložené kocky každého hráča, napr. [[4], [], [2, 2]]
    ## s["hidden_counts"] : počet skrytých kociek každého hráča (aj tvojich)
    ## s["names"]         : mená hráčov; s["you"] je tvoj index v tomto zozname
    ## s["round"]         : číslo kola
    ## s["rbids"]         : ponuky v tomto kole [[meno, [počet, hodnota]], ...]
    ## s["log"]           : posledných 60 riadkov z logu hry
    ##
    ## --- Čo môžeš vrátiť ---
    ## {"action": "call"}                        - "Neverím!", vyhodnotí sa posledná ponuka
    ## {"action": "bid", "count": 3, "face": 4}  - nová ponuka "tri štvorky" (musí byť v legal_bids)
    ## {"action": "reveal", "indices": [0, 2]}   - vyloží tvoje kocky s týmito indexmi, ostatné prehodí;
    ##                                             potom sa ťa engine opýta znova (must_raise=True)
    ##                                             a ty musíš vrátiť "bid"
    ##
    ## Indexuje sa konzistentne počas celej hry, tvoj index hráča je vždy rovnaký aj kocky
    ##
    ## Pravidlá: jednotky sú žolíky (počítajú sa ako 2-6), okrem ponuky na samotné jednotky.
    ## Na ťah máš 3 sekundy. Pri chybe, timeoute alebo nepovolenom ťahu engine zahrá za teba
    ## náhradný ťah (call, alebo najnižšia povolená ponuka). print() a chyby idú do bot_errors.log.

    # 1) Občas (10 %) vyložíme prvú kocku a prehodíme ostatné. Podmienky: nesmieme byť v must_raise,
    #    musí existovať nejaká povolená ponuka (inak by sme nemali ako navýšiť) a musíme mať skryté kocky.
    
    
    if s["bid"] is not None:
        pocet_co_call = []
        expected_win = []
        for face in range(1, 7):
            if face == 1:
                if s["bid"][1] != 1:
                    pocet = math.floor(s["bid"][0]/2+1)
                else:
                    pocet = s["bid"][0] + 1
                kolko_treba = pocet - count_shown(s, 1) - s["your_dice"].count(1)
                if kolko_treba < 0:
                    return {"action": "bid", "count": pocet, "face": face}
                expected_win.append(expected_win_of_call(s, kolko_treba, face))
                pocet_co_call.append(pocet)

            elif face <= s["bid"][1]:
                if s["bid"][1] != 1:
                    pocet = s["bid"][0] + 1
                else:
                    pocet = s["bid"][0]*2
                kolko_treba = pocet - count_shown(s, face) - s["your_dice"].count(face) - s["your_dice"].count(1)
                if kolko_treba < 0:
                    return {"action": "bid", "count": pocet, "face": face}
                expected_win.append(expected_win_of_call(s, kolko_treba, face))
                pocet_co_call.append(pocet)

            else:
                if s["bid"][1] != 1:
                    pocet = s["bid"][0]
                else:
                    pocet = s["bid"][0]*2
                kolko_treba = pocet - count_shown(s, face) - s["your_dice"].count(face) - s["your_dice"].count(1)
                if kolko_treba < 0:
                    return {"action": "bid", "count": pocet, "face": face}
                expected_win.append(expected_win_of_call(s, kolko_treba, face))
                pocet_co_call.append(pocet)
        if max(expected_win) > expected_win_of_fold(s, s["bid"][0], s["bid"][1]):
            return {"action": "bid", "count": pocet_co_call[expected_win.index(max(expected_win))], "face": expected_win.index(max(expected_win)) + 1}
        else:
            return {"action": "call"}
    else:
        c, f = random.choice(s["legal_bids"][:6])
        return {"action": "bid", "count": c, "face": f}