import random

def decide(s):
    if not s["must_raise"] and s["legal_bids"] and s["your_dice"] and random.random() < 0.1:
        return {"action": "reveal", "indices": [0]}
    if s["can_call"] and (not s["legal_bids"] or random.random() < 0.3):
        return {"action": "call"}
    c, f = random.choice(s["legal_bids"][:6])
    return {"action": "bid", "count": c, "face": f}
