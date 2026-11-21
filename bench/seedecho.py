# Cross-talk check for seeded requests: every request asks for its own code word back, many at once, mixed
# with unseeded ones. Any answer carrying another request's word (or none) is a failure.
import json, os, random, sys, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
N = int(os.environ.get("N", "12")); ROUNDS = int(os.environ.get("ROUNDS", "4"))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
r0 = random.Random(time.time()); bad = []; cross = []; n = [0]
def one(i, rnd):
    code = f"ZX{rnd}Q{i}K{r0.randrange(1000, 9999)}"
    pad = " ".join(r0.choice(WORDS) for _ in range(r0.choice((20, 300, 900))))
    b = dict(model=MODEL, max_tokens=700, temperature=(0.8, 0.25, 0.0)[i % 3], messages=[{"role": "user", "content": f"Notes: {pad}\nIgnore the notes. Reply with exactly this code and nothing else: {code}"}])
    if i % 4 != 3: b["seed"] = r0.randrange(1, 2**31)
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600) as r: d = json.loads(r.read())
        m = d["choices"][0]["message"]; txt = (m.get("content") or "") + " | " + (m.get("reasoning") or "")
    except Exception as e: txt = "ERR " + str(e)[:60]
    n[0] += 1
    import re
    others = [c for c in re.findall(r"ZX\d+Q\d+K\d{4}", txt) if c != code]
    if others: cross.append((code, others[:2]))
    if code not in txt: bad.append((code, txt[:80]))
for rnd in range(ROUNDS):
    ths = [threading.Thread(target=one, args=(i, rnd)) for i in range(N)]
    for t in ths: t.start(); time.sleep(r0.choice((0, 0.05, 0.3)))
    for t in ths: t.join()
print(f"seedecho: {n[0]} requests, {len(cross)} with another request's code {cross[:3]}, {len(bad)} without their own {bad[:2]}: {'PASS' if not bad and not cross else 'FAIL'}")
