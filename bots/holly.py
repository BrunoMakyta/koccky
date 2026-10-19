def E(s, b):  # expected number of dice matching bid b (ones are wild unless bidding ones)
    c, f = b
    known = s["your_dice"] + [d for L in s["shown"] for d in L]
    hit = sum(d == f or (f != 1 and d == 1) for d in known)
    unknown = sum(s["hidden_counts"]) - len(s["your_dice"])
    return hit + unknown * (1 / 6 if f == 1 else 1 / 3)

def decide(s):
    if s["can_call"] and (not s["legal_bids"] or s["bid"][0] > E(s, s["bid"]) + 0.7):
        return {"action": "call"}
    c, f = max(s["legal_bids"], key=lambda b: (E(s, b) - b[0], -b[0]))
    if s["must_raise"]:
        return {"action": "bid", "count": c, "face": f}
    else:
        revealIndices = [i for i in range(len(s["your_dice"])) if s["your_dice"][i] == s["bid"][1]]
        if (len(revealIndices) > 0):
            return {"action": "reveal", "indices": revealIndices}
        else:
            return {"action": "bid", "count": c, "face": f}
