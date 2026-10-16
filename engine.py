import random

def raise_ok(old, new):
    (c, f), (n, g) = old, new
    if f != 1 and g != 1: return n > c or (n == c and g > f)
    if f != 1: return n > c // 2      # normal -> ones: more than half
    if g != 1: return n >= 2 * c + 1  # ones -> normal: 2*count + 1
    return n > c

class Game:
    def __init__(self, names, dice=5, seed=None):
        self.names, self.rng = list(names), random.Random(seed)
        self.hidden = [[0] * dice for _ in names]
        self.shown = [[] for _ in names]
        self.bid = self.bidder = self.winner = None
        self.turn, self.round, self.must_raise, self.rbids, self.log = 0, 1, False, [], []
        self.reroll()
        self.say(f"Round 1: {self.names[0]} opens.")

    def say(self, s): self.log.append(s)
    def reroll(self): self.hidden = [[self.rng.randint(1, 6) for _ in h] for h in self.hidden]
    def size(self, i): return len(self.hidden[i]) + len(self.shown[i])
    def total(self): return sum(self.size(i) for i in range(len(self.names)))
    def nxt(self, i):
        j = (i + 1) % len(self.names)
        while not self.size(j): j = (j + 1) % len(self.names)
        return j
    def legal_bids(self):
        return [[c, f] for c in range(1, self.total() + 1) for f in range(1, 7)
                if self.bid is None or raise_ok(self.bid, (c, f))]

    def view(self, i, spectate=False):
        v = dict(names=self.names, you=i, turn=self.turn, round=self.round, bid=self.bid, bidder=self.bidder,
                 can_call=self.bid is not None and not self.must_raise, must_raise=self.must_raise,
                 your_dice=self.hidden[i] if i is not None else [], shown=self.shown,
                 hidden_counts=[len(h) for h in self.hidden], rbids=self.rbids,
                 legal_bids=self.legal_bids(), winner=self.winner, log=self.log[-60:])
        if spectate: v["all_hidden"] = self.hidden
        return v

    def act(self, i, a):
        if self.winner is not None or i != self.turn: raise ValueError("not your turn")
        kind, nm = a.get("action"), self.names[i]
        if kind == "call":
            if self.bid is None or self.must_raise: raise ValueError("you can't call now")
            return self.resolve(i)
        if kind == "reveal":
            idx = set(a.get("indices") or [])
            if (self.must_raise or not idx or not self.legal_bids()
                    or any(not isinstance(x, int) or not 0 <= x < len(self.hidden[i]) for x in idx)):
                raise ValueError("bad reveal")
            vals = [d for k, d in enumerate(self.hidden[i]) if k in idx]
            self.shown[i] += vals
            self.hidden[i] = [self.rng.randint(1, 6) for _ in range(len(self.hidden[i]) - len(idx))]
            self.must_raise = True
            self.say(f"{nm} reveals {sorted(vals)} and rerolls the rest.")
            return
        if kind == "bid":
            b = [a.get("count"), a.get("face")]
            if b not in self.legal_bids(): raise ValueError("illegal bid")
            self.bid, self.bidder, self.must_raise = b, i, False
            self.rbids.append([nm, b]); self.say(f"{nm} bids {b[0]}×{b[1]}.")
            self.turn = self.nxt(i); return
        raise ValueError("unknown action")

    def take(self, i, k):  # hidden dice go first, then revealed ones
        out = []
        for _ in range(k):
            if self.hidden[i]: out.append((self.hidden[i].pop(), False))
            elif self.shown[i]: out.append((self.shown[i].pop(), True))
        return out

    def resolve(self, caller):
        (c, f), b, N = self.bid, self.bidder, self.names
        hands = [self.hidden[i] + self.shown[i] for i in range(len(N))]
        actual = sum(d == f or (f != 1 and d == 1) for h in hands for d in h)
        self.say(f"{N[caller]} calls bluff on {c}×{f}. Hands: " + "; ".join(f"{N[i]} {sorted(h)}" for i, h in enumerate(hands) if h))
        if actual >= c:
            moved = self.take(caller, 2); loser = caller
            for d, sh in moved: (self.shown if sh else self.hidden)[b].append(d)
            self.say(f"Found {actual} (needed {c}): bid was true. {N[caller]} gives {len(moved)} dice to {N[b]}.")
        else:
            lost = self.take(b, c - actual); loser = b
            self.say(f"Found {actual} (bid {c}): {N[b]} loses {len(lost)} dice.")
            
        for i in range(len(N)):
            self.hidden[i] += self.shown[i]; self.shown[i] = []
            
        alive = [i for i in range(len(N)) if self.size(i)]
        self.round += 1
        if len(alive) == 1 or self.round > 300:
            self.winner = max(alive, key=self.size); self.say(f"{N[self.winner]} wins!"); return
        self.turn = loser if self.size(loser) else self.nxt(loser)
        self.bid = self.bidder = None; self.must_raise = False; self.rbids = []
        self.reroll(); self.say(f"Round {self.round}: {N[self.turn]} opens.")
