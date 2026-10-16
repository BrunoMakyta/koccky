import random
import math

def count_shown(s, face):
    return sum(
        die == face or (face != 1 and die == 1)
        for player_dice in s["shown"]
        for die in player_dice
    )

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
    
    if len (s["rbids"]) > 0:
        last_bidder = s["rbids"][-1][1]
        if last_bidder[1] != 1:
            if (last_bidder[0] - s["your_dice"].count(last_bidder[1]) - s["your_dice"].count(1) - s["your_dice"].count(2) - count_shown(s, last_bidder[1])) < math.floor((sum(s["hidden_counts"])-len(s["your_dice"]))/3) + 1:
                return {"action": "bid", "count": last_bidder[0]+1, "face": last_bidder[1]}
            else:
                return {"action": "call"}
        else:
            if (last_bidder[0] - s["your_dice"].count(1) - count_shown(s, 1)) < math.floor((sum(s["hidden_counts"])-len(s["your_dice"]))/6) + 1:
                return {"action": "bid", "count": last_bidder[0]+1, "face": last_bidder[1]}
            else:
                return {"action": "call"}
    else:
        return {"action": "bid", "count": 1, "face": 6}