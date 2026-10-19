# Runs one bot file in its own process; talks JSON lines over stdin/stdout.
import sys, json, importlib.util
out, sys.stdout = sys.stdout, sys.stderr  # bot print() goes to stderr, not the protocol
spec = importlib.util.spec_from_file_location("bot", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for line in sys.stdin:
    try: r = m.decide(json.loads(line))
    except Exception as e: r = {"error": repr(e)}
    out.write(json.dumps(r) + "\n"); out.flush()
