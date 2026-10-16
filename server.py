import json, os, queue, random, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from engine import Game
import math

ROOT = os.path.dirname(os.path.abspath(__file__))
BOTS, STATS = os.path.join(ROOT, "bots"), os.path.join(ROOT, "stats.json")
DICE_RANGE = (3, 10)  # každá hra dostane náhodný počet kociek na hráča z tohto rozsahu
MAX_KOCAK_V_HRE = 50
cfg = dict(interval=45, delay=0.4)  # interval: seconds between scheduled games (0 = paused)
games, lock = {}, threading.Lock()

HIST = os.path.join(ROOT, "history.json")
history = json.load(open(HIST)) if os.path.exists(HIST) else []  # one record per finished game
bots = lambda: sorted(f[:-3] for f in os.listdir(BOTS) if f.endswith(".py"))

class BotProc:
    def __init__(self, name):
        self.p = subprocess.Popen([sys.executable, os.path.join(ROOT, "bot_runner.py"), os.path.join(BOTS, name + ".py")],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(os.path.join(ROOT, "bot_errors.log"), "a"), text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=lambda: [self.q.put(l) for l in self.p.stdout], daemon=True).start()
    def ask(self, state, t=3.0):
        try:
            self.p.stdin.write(json.dumps(state) + "\n"); self.p.stdin.flush()
            return json.loads(self.q.get(timeout=t))
        except Exception as e:
            self.close(); return {"error": repr(e)}
    def close(self):
        try: self.p.kill()
        except Exception: pass

def fallback(v):
    if v["can_call"]: return {"action": "call"}
    c, f = v["legal_bids"][0]; return {"action": "bid", "count": c, "face": f}

def track(m):  # remember the order in which players ran out of dice
    g = m["game"]; out = m.setdefault("out", [])
    out += [i for i in range(len(g.names)) if not g.size(i) and i not in out]

def finish(m):  # points = place counted from the bottom: first out 0 ... winner N-1
    g, out = m["game"], m.get("out", [])
    rest = sorted((i for i in range(len(g.names)) if i not in out and i != g.winner), key=g.size)
    order = out + rest + [g.winner]
    rec = dict(id=m["id"], t=m["t"], names=g.names, winner=g.names[g.winner],
               points={g.names[i]: k for k, i in enumerate(order)})
    with lock:
        history.append(rec); json.dump(history, open(HIST, "w"))

def leaderboard():
    names = bots(); rows = {n: dict(name=n, points=0, games=0, wins=0, last=[]) for n in names}
    series, avgs = {n: [] for n in names}, {n: [] for n in names}
    for r in history:
        for n, p in r["points"].items():
            if n in rows:
                x = rows[n]; x["points"] += p; x["games"] += 1; x["wins"] += n == r["winner"]; x["last"].append(p)
        for n in names:
            series[n].append(rows[n]["points"])
            l = rows[n]["last"][-50:]; avgs[n].append(sum(l) / len(l) if l else 0)  # rolling average of the bot's last 50 games
    out = sorted(rows.values(), key=lambda x: -x["points"])
    for x in out:
        l = x.pop("last")[-50:]; x["avg50"] = sum(l) / len(l) if l else 0
    return dict(rows=out, series={n: v[-200:] for n, v in series.items()}, avgseries={n: v[-200:] for n, v in avgs.items()}, matches=history[-50:][::-1])

def run_game(gid):
    m = games[gid]; g, h = m["game"], m["human"]
    procs = {i: BotProc(n) for i, n in enumerate(m["names"]) if i != h}
    try:
        while g.winner is None:
            i = g.turn
            if i == h:
                if time.time() - m["last_act"] > 900: m["status"] = "abandoned"; return
                time.sleep(0.2); continue
            if m.get("debug") and not m.get("auto"):  # debug game: wait for a "Next action" click
                if not m["go"].wait(0.3):
                    if time.time() - m["last_act"] > 900: m["status"] = "abandoned"; return
                    continue
                m["go"].clear()
            else: time.sleep(cfg["delay"])
            with m["lock"]: v = g.view(i)
            a = procs[i].ask(v); note = ""
            with m["lock"]:
                try: g.act(i, a)
                except Exception as e:
                    note = f"invalid move ({e}); fallback played"
                    g.say(f"{g.names[i]} sent an invalid move ({e}); fallback played."); g.act(i, fallback(v))
                track(m)
                m["n"] = m.get("n", 0) + 1  # what the bot saw and answered, shown in the debug panel
                m["trace"] = dict(n=m["n"], bot=g.names[i], response=a, note=note,
                                  state={**{k: x for k, x in v.items() if k not in ("log", "legal_bids")},
                                         "legal_bids": f"{len(v['legal_bids'])} options, lowest {v['legal_bids'][:2]}"})
        if not m.get("debug"): finish(m)  # debug games don't count for the leaderboard
        m["status"] = "finished"
    finally:
        for p in procs.values(): p.close()

def start(opp=None, human=False, pick=None, dice=None):
    avail = bots(); lo = 1 if human else 2   # lo = minimálny počet botov (spolu aspoň 2 hráči)
    k = min(len(avail), max(lo, opp) if opp else random.randint(lo, max(lo, min(len(avail), 10))))  # ľubovoľný počet, max. všetci boti no max bde 10
    if k < lo: return None
    names = random.sample(avail, k) + (["You"] if human else []); random.shuffle(names)
    if pick: names = list(pick)
    dice = min(max(3, MAX_KOCAK_V_HRE//k), max(1, int(dice) if dice else random.randint(*DICE_RANGE)))
    gid = str(int(time.time() * 1000))
    games[gid] = dict(id=gid, names=names, human=names.index("You") if human else None, status="running",
                      lock=threading.Lock(), game=Game(names, dice), dice=dice, t=time.time(), last_act=time.time(),
                      debug=bool(pick), go=threading.Event())
    threading.Thread(target=run_game, args=(gid,), daemon=True).start()
    for old in sorted(games, reverse=True)[25:]:
        if games[old]["status"] != "running": del games[old]
    return gid

def scheduler():
    last = 0
    while True:
        time.sleep(1)
        if cfg["interval"] > 0 and time.time() - last >= cfg["interval"] and not any(
                 m["status"] == "running" and m["human"] is None and not m.get("debug") for m in games.values()):
            if start(): last = time.time()

def state():
    gl = sorted(games.values(), key=lambda m: m["t"], reverse=True)
    return dict(cfg=cfg, bots=[dict(name=r["name"], games=r["games"], wins=r["wins"]) for r in leaderboard()["rows"]],
                games=[dict(id=m["id"], names=m["names"], status=m["status"], debug=m.get("debug", False), dice=m["dice"],
                            winner=None if m["game"].winner is None else m["names"][m["game"].winner]) for m in gl])

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def body(self): return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        path = urlparse(self.path).path; p = path.strip("/").split("/")
        if path == "/": return self.reply(open(os.path.join(ROOT, "index.html"), "rb").read(), ctype="text/html; charset=utf-8")
        if path == "/api/state": return self.reply(state())
        if path == "/api/leaderboard": return self.reply(leaderboard())
        if path == "/api/template":  # NOVÉ: stiahnutie default bota ako def.py
            body = open(os.path.join(BOTS, "def.py"), "rb").read()
            self.send_response(200); self.send_header("Content-Type", "text/x-python; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="def.py"')
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            return self.wfile.write(body)
        if p[:2] == ["api", "game"] and len(p) == 3 and p[2] in games:
            m = games[p[2]]; g, h = m["game"], m["human"]
            with m["lock"]:
                v = g.view(h) if h is not None and g.winner is None else g.view(h, spectate=True)
                v.update(id=m["id"], status=m["status"], debug=m.get("debug", False), auto=m.get("auto", False), trace=m.get("trace")); return self.reply(v)
        self.reply({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path); q = parse_qs(u.query); p = u.path.strip("/").split("/")
        try:
            if u.path == "/api/upload":
                name, data = q.get("name", [""])[0], self.body()
                if not re.fullmatch(r"[A-Za-z0-9_\-]{1,30}\.py", name): raise ValueError("Use a simple file name ending in .py")
                if b"def decide" not in data: raise ValueError("The bot must define decide(state)")
                compile(data, name, "exec")
                open(os.path.join(BOTS, name), "wb").write(data)
            elif u.path == "/api/delete":
                name = q.get("name", [""])[0]
                if name in bots(): os.remove(os.path.join(BOTS, name + ".py"))
            elif u.path == "/api/config":
                j = json.loads(self.body())
                cfg["interval"] = max(0, int(j.get("interval", cfg["interval"])))

            elif u.path == "/api/debug":
                j = json.loads(self.body())
                pick = list(dict.fromkeys(n for n in j.get("bots", []) if n in bots()))
                if len(pick) < 2: raise ValueError("Pick at least 2 bots")
                return self.reply({"id": start(pick=pick, dice=j.get("dice"))})
            elif u.path in ("/api/run", "/api/play"):
                j = json.loads(self.body() or "{}")
                gid = start(j.get("opponents"), human=u.path == "/api/play", dice=j.get("dice"))

                if not gid: raise ValueError("Upload more bots first (2 for bot games, 1 to play against)")
                return self.reply({"id": gid})
            elif len(p) == 4 and p[3] in ("step", "auto"):  # debug game: one bot action / run to the end
                m = games[p[2]]; m["last_act"] = time.time()
                if p[3] == "auto": m["auto"] = True
                m["go"].set()
            elif len(p) == 4 and p[3] == "act":
                m = games[p[2]]; a = json.loads(self.body())
                with m["lock"]:
                    if m["human"] is None: raise ValueError("not a human game")
                    m["game"].act(m["human"], a); m["last_act"] = time.time(); track(m)

            else: return self.reply({"error": "not found"}, 404)
            self.reply({"ok": True})
        except Exception as e:
            self.reply({"error": str(e)}, 400)

if __name__ == "__main__":
    os.makedirs(BOTS, exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    threading.Thread(target=scheduler, daemon=True).start()
    print(f"Liar's Dice listening on port {port}; on another device, open http://<this computer's IP>:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()