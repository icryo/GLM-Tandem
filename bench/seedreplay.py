# Replay captured seeded requests against the server. W workers pull bodies from a slice of the set.
# Prints per-class medians and totals; with REPEAT=1 sends each body twice (in order) and compares the outputs.
import json, os, sys, time, threading, urllib.request, hashlib, statistics as st
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
W = int(os.environ.get("W", "1")); LO = int(os.environ.get("LO", "0")); HI = int(os.environ.get("HI", "80"))
SET = os.path.expanduser(os.environ.get("SET", "~/llm-stack/cluster/replayset.jsonl"))
bodies = [json.loads(l) for l in open(SET)][LO:HI]
for b in bodies: b["model"] = MODEL
if os.environ.get("TEMP0") == "1":
    for b in bodies: b["temperature"] = 0
if os.environ.get("NOSEED") == "1":
    for b in bodies: b.pop("seed", None)
lock = threading.Lock(); nxt = [0]; res = [None] * len(bodies)
def call(b):
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        d = json.loads(r.read())
    u = d.get("usage", {}); m = d["choices"][0]["message"]; txt = (m.get("content") or "") + "|" + (m.get("reasoning") or "")
    ok = True
    if (b.get("response_format") or {}).get("type") == "json_object":
        try: json.loads(m.get("content") or "")
        except Exception: ok = d["choices"][0].get("finish_reason") == "length"
    return time.time() - t0, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), hashlib.md5(txt.encode()).hexdigest()[:10], ok, txt
def worker():
    while True:
        with lock:
            i = nxt[0]; nxt[0] += 1
        if i >= len(bodies): return
        try: res[i] = call(bodies[i])
        except Exception as e: res[i] = ("ERR", str(e)[:100])
t = time.time(); ths = [threading.Thread(target=worker) for _ in range(W)]; [x.start() for x in ths]; [x.join() for x in ths]; wall = time.time() - t
err = [r for r in res if r[0] == "ERR"]; ok = [r for r in res if r[0] != "ERR"]
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
out = sum(r[2] for r in ok)
print(f"W={W} n={len(ok)} errors={len(err)} {err[:1]} bad_json={sum(not r[4] for r in ok)} wall {wall:.1f} s | out {out} tok, {out / wall:.1f} tok/s overall | req/min {60 * len(ok) / wall:.1f}", flush=True)
for name, a, b_ in (("in<1200", 0, 1200), ("in 1200-2000", 1200, 2000), ("in 2000-3500", 2000, 3500), ("in>=3500", 3500, 10**9)):
    s = [r for r in ok if a <= r[1] < b_]
    if s: print(f"  {name:13s} n={len(s):3d} total p50 {q([r[0] for r in s], .5):5.2f} p90 {q([r[0] for r in s], .9):5.2f} s | out p50 {q([r[2] for r in s], .5):4.0f} | tok/s incl. prompt p50 {q([r[2] / r[0] for r in s], .5):5.1f}")
if os.environ.get("REPEAT") == "1":
    same = 0; diff = []
    for i, b in enumerate(bodies):
        r2 = call(b); r3 = call(b)
        if r2[3] == r3[3]: same += 1
        else: diff.append(i)
    print(f"repeat (solo twice): {same}/{len(bodies)} identical; differing {diff[:10]}")
    print(f"first run vs solo: {sum(1 for i, b in enumerate(bodies) if res[i][0] != 'ERR' and res[i][3] == call(b)[3])}/{len(bodies)} identical")
if os.environ.get("HASHOUT"):
    json.dump([r[3] if r[0] != "ERR" else None for r in res], open(os.environ["HASHOUT"], "w"))
if os.environ.get("TEXTOUT"):
    json.dump([r[5] if r[0] != "ERR" else None for r in res], open(os.environ["TEXTOUT"], "w"))
