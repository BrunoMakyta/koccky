# Liar's Dice (local)
    python3 server.py [port]      # then open http://127.0.0.1:8000   (stdlib only, no installs)
Drop bots into `bots/` or upload them in the UI. Every `interval` seconds (default 45) 2-4 random bots play a game.
"Join a game" seats you against 1-3 random bots. Settings (interval, dice, bot move delay) live in `cfg` at the top of server.py.

## Bot API: define `decide(state) -> dict`
state: your_dice, shown (public dice per player), hidden_counts, names, you (your index), turn, round,
bid ([count, face] or None), bidder, can_call, must_raise, legal_bids ([[count, face], ...]), rbids (this round's bids).
Return {"action":"call"} | {"action":"bid","count":c,"face":f} | {"action":"reveal","indices":[0,2]}.
After a reveal you're asked again with must_raise=True and must bid. 3 s limit per move; crashes, timeouts and
illegal moves make the engine play a fallback (call, or lowest legal bid). Errors go to bot_errors.log.
NOTE: bots run as normal Python processes with your user's rights - only upload code you trust. Server binds to localhost only.
