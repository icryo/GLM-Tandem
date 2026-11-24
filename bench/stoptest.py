# Requests with a `stop` string that share decode steps with other streams. With two-token steps the stop
# can complete on the first token of a pair; the server used to die there (KeyError in its loop).
#   URL=http://localhost:8010/v1/chat/completions MODEL=default_model python3 stoptest.py
import json, os, random, threading, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
ROUNDS = int(os.environ.get("ROUNDS", "12")); SEEDED = os.environ.get("SEEDED", "0") == "1"
W = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "gamma", "haze", "iris", "jade", "kelp", "lumen", "moss", "nova", "opal", "pine", "quartz", "reed", "slate", "tide"]
def post(body, timeout=120):
    body = dict(body, model=MODEL)
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout) as r:
        return json.loads(r.read())
def filler(out, i):
    try:
        b = dict(messages=[dict(role="user", content=f"Write about 150 words on topic number {i}: rivers.")], max_tokens=220, temperature=0.6, enable_thinking=False)
        out.append(bool(post(b)["choices"][0]["message"].get("content")))
    except Exception as e: out.append(f"ERR {e!r}"[:80])
def stopper(out, r, k):
    rnd = random.Random(1000 * r + k); ws = [rnd.choice(W) + str(rnd.randint(10, 99)) for _ in range(40)]; cut = rnd.randint(5, 35)
    b = dict(messages=[dict(role="user", content="Repeat exactly, nothing else: " + " ".join(ws))], max_tokens=400, temperature=0, stop=[ws[cut]], enable_thinking=False)
    if SEEDED: b.update(seed=rnd.randrange(10**9), temperature=0.25)
    try:
        d = post(b); c = d["choices"][0]; t = c["message"].get("content") or ""
        ok = c.get("finish_reason") == "stop" and ws[cut] not in t and t.strip().split() == ws[:cut]
        out.append(ok if ok else f"BAD finish={c.get('finish_reason')} words={len(t.split())}/{cut}")
    except Exception as e: out.append(f"ERR {e!r}"[:80])
bad = 0; n = 0
for r in range(ROUNDS):
    fo = []; so = []
    th = [threading.Thread(target=filler, args=(fo, r))] + [threading.Thread(target=stopper, args=(so, r, k)) for k in range(3)]
    for t in th: t.start(); time.sleep(0.15)
    for t in th: t.join()
    n += len(so) + len(fo); b = [x for x in so + fo if x is not True]; bad += len(b)
    if b: print(f"round {r}: {b}", flush=True)
print(f"stoptest: {n - bad}/{n} ok:", "PASS" if not bad else "FAIL")
