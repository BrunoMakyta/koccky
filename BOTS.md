# Liar's Dice (local)
    python3 server.py [port]      # then open http://127.0.0.1:8000   (stdlib only, no installs)
Drop bots into `bots/` or upload them in the UI. Bot games run continuously: the next game starts as soon as the previous one finishes (if at least 2 bots are available).
Live replays a bot game at about 3 seconds per action; while waiting for the very first finished game it can follow a running match. At the end it switches to the newest finished match. This display speed does not slow down the bots.
"Join a game" seats you against random bots. The bot-processing delay (default 0) lives in `cfg` at the top of server.py.

## Bot API: define `decide(state) -> dict`
state: your_dice, shown (public dice per player), hidden_counts, names, you (your index), turn, round,
bid ([count, face] or None), bidder, can_call, must_raise, legal_bids ([[count, face], ...]), rbids (this round's bids).
Return {"action":"call"} | {"action":"bid","count":c,"face":f} | {"action":"reveal","indices":[0,2]}.
After a reveal you're asked again with must_raise=True and must bid. 3 s limit per move; crashes, timeouts and
illegal moves make the engine play a fallback (call, or lowest legal bid). Errors go to bot_errors.log.
Ones -> a normal face requires at least `2 * count` (not `2 * count + 1`).
NOTE: bots run as normal Python processes with your user's rights - only upload code you trust. Server binds to localhost only.
