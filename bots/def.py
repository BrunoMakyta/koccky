import random

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
    if not s["must_raise"] and s["legal_bids"] and s["your_dice"] and random.random() < 0.1:
        return {"action": "reveal", "indices": [0]}
 
    # 2) Ak môžeme zavolať "Neverím!", spravíme to, keď už nemáme čím navýšiť, alebo s 30 % šancou.
    if s["can_call"] and (not s["legal_bids"] or random.random() < 0.3):
        return {"action": "call"}
 
    # 3) Inak navýšime: náhodná ponuka spomedzi 6 najnižších povolených.
    c, f = random.choice(s["legal_bids"][:6])
    return {"action": "bid", "count": c, "face": f}