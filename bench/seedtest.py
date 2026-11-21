# Seeded requests: the same seed and prompt must give the same output, alone and while other requests run;
# different seeds must differ; and a seeded request must not be slower than an unseeded one.
import json, os, random, sys, time, threading, urllib.request, urllib.error
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
N = int(os.environ.get("N", "6")); EXACT = os.environ.get("EXACT", "1") == "1"; MAXT = int(os.environ.get("MAXT", "200"))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
r0 = random.Random(time.time())
def call(body):
    t = time.time()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r: d = json.loads(r.read())
            break
        except urllib.error.URLError as e:                       # the stock HTTP server's accept queue is short
            if attempt == 3 or "reset" not in str(e).lower(): raise
            time.sleep(0.3)
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + "|" + (m.get("reasoning") or ""), d["usage"]["completion_tokens"], time.time() - t
def body(i, seed, temp):
    notes = " ".join(r0.choice(WORDS) for _ in range(300))
    b = dict(model=MODEL, max_tokens=MAXT, temperature=temp, response_format={"type": "json_object"},
             messages=[{"role": "user", "content": f"Notes {i}: {notes}\nReturn a JSON object with keys title, summary and tags describing a short story about {WORDS[(i * 7) % len(WORDS)]}."}])
    if seed is not None: b["seed"] = seed
    return b
bodies = [body(i, r0.randrange(1, 2**31), (0.8, 0.25, 0.0)[i % 3]) for i in range(N)]
ref = [call(b) for b in bodies]                                       # alone, prompt fresh
again = [call(b) for b in bodies]                                     # alone, prompt cached
stop = [False]
def noise(k):
    while not stop[0]:
        try: call(body(100 + k, None if k % 2 else r0.randrange(1, 2**31), 0.7))
        except Exception: pass
def longnoise():
    while not stop[0]:
        try: call(dict(model=MODEL, max_tokens=60, temperature=0.5, messages=[{"role": "user", "content": "Notes: " + " ".join(r0.choice(WORDS) for _ in range(LONG)) + "\nIgnore the notes and name three colours."}]))
        except Exception: pass
LONG = int(os.environ.get("LONG", "9000"))
ths = [threading.Thread(target=noise, args=(k,)) for k in range(3)] + [threading.Thread(target=longnoise)]; [t.start() for t in ths]; time.sleep(1.0)
res = [None] * N
def one(i): res[i] = call(bodies[i])
t2 = [threading.Thread(target=one, args=(i,)) for i in range(N)]; [(t.start(), time.sleep(0.03)) for t in t2]; [t.join() for t in t2]
stop[0] = True; [t.join() for t in ths]
same_alone = sum(a[0] == b[0] for a, b in zip(ref, again)); same_busy = sum(a[0] == b[0] for a, b in zip(ref, res))
b2 = dict(bodies[0]); b2["seed"] = bodies[0]["seed"] + 1; other = call(b2)[0] != ref[0][0]
uns = dict(bodies[0]); uns.pop("seed"); uns["messages"] = body(999, None, 0.8)["messages"]; sd = dict(uns); sd["seed"] = 7; sd["messages"] = body(998, None, 0.8)["messages"]
ru = call(uns); rs = call(sd); su = (ru[1] - 1) / ru[2]; ss = (rs[1] - 1) / rs[2]
ok = same_alone == N and other and ss > 0.8 * su and (same_busy == N or not EXACT)
fails = [(i, bodies[i]["temperature"], next((k for k in range(min(len(ref[i][0]), len(res[i][0]))) if ref[i][0][k] != res[i][0][k]), -1), len(ref[i][0])) for i in range(N) if ref[i][0] != res[i][0]]
if fails: print("  differing (index, temperature, first differing char, length):", fails)
print(f"seedtest: repeat alone {same_alone}/{N}, repeat under load {same_busy}/{N}, another seed differs {other}, seeded {ss:.0f} vs unseeded {su:.0f} tok/s incl. prompt: {'PASS' if ok else 'FAIL'}")
