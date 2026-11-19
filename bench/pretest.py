# A long cold prompt is loading; short requests arrive behind it. Reports their time to first token.
# With CANCEL=1 the long request is dropped at a random moment in each round (robustness).
import json, os, random, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
N = int(os.environ.get("CTX", "13500")); ROUNDS = int(os.environ.get("ROUNDS", "1")); CANCEL = os.environ.get("CANCEL") == "1"; NS = int(os.environ.get("NSHORT", "4"))
def ask(msgs, mt, out, key, cancel_after=None):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=mt, temperature=0.0, messages=msgs)
    t0 = time.time(); first = None; text = []
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000) as r:
            if cancel_after is not None:
                time.sleep(cancel_after); r.close(); out[key] = ("cancelled", None); return
            for line in r:
                if not line.startswith(b"data: ") or b"[DONE]" in line: continue
                d = json.loads(line[6:])
                for c in d.get("choices") or []:
                    dl = c.get("delta") or {}
                    if dl.get("content"): text.append(dl["content"])
                    if first is None and (dl.get("content") or dl.get("reasoning")): first = time.time()
        out[key] = ((first or time.time()) - t0, "".join(text))
    except Exception as e: out[key] = ("ERR " + repr(e)[:80], None)
rnd = random.Random(time.time()); bad = 0
for k in range(ROUNDS):
    sysmsg = "Reference notes (ignore them): " + " ".join(rnd.choice(WORDS) for _ in range(N))
    out = {}; tl = threading.Thread(target=ask, args=([{"role": "system", "content": sysmsg}, {"role": "user", "content": "Reply with the single word: done"}], 10, out, "long", rnd.uniform(0.3, 4) if CANCEL else None))
    tl.start(); time.sleep(1.0); ths = []
    for i in range(NS):
        a, b = rnd.randint(10, 99), rnd.randint(10, 99)
        t = threading.Thread(target=ask, args=([{"role": "user", "content": f"What is {a} + {b}? Reply with only the number."}], 20, out, (i, a + b))); t.start(); ths.append(t); time.sleep(0.4)
    [t.join() for t in ths]; tl.join()
    for key, v in sorted((k2, v) for k2, v in out.items() if k2 != "long"):
        ok = v[1] is not None and str(key[1]) in v[1]; bad += not ok
        print(f"round {k} short {key[0]}: first token {v[0] if isinstance(v[0], str) else round(v[0], 2)} s {'ok' if ok else 'WRONG ' + repr(v[1])}")
    v = out["long"]; ok = v[0] == "cancelled" or (v[1] is not None and "done" in v[1].lower()); bad += not ok
    print(f"round {k} long: {v[0] if isinstance(v[0], str) else round(v[0], 1)} {'ok' if ok else 'WRONG ' + repr(v[1])}", flush=True)
print("ALL PASS" if not bad else f"FAILURES {bad}")
