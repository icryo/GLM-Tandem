# Decode speed and prompt time as the context grows, for 1/2/4 concurrent requests.
# Round A: fresh documents (prompt processing measured). Round B: same documents, new question
# (prefix cached), so the decode numbers are not mixed with other streams' prompt processing.
import json, os, random, sys, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8080/v1/chat/completions"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
SIZES = [int(x) for x in os.environ.get("SIZES", "0,6000,24000").split(",")]      # words of context
NS = [int(x) for x in os.environ.get("NS", "1,2,4").split(",")]
MAXT = int(os.environ.get("MAXT", "300"))
TOPICS = ["the history of the bicycle", "how tides work", "the invention of the printing press", "why the sky is dark at night"]
def doc(seed, n):
    r = random.Random(seed); out = []
    while len(out) < n:
        out += [" ".join(r.choice(WORDS) for _ in range(r.randint(6, 14))).capitalize() + "."]
        n -= r.randint(6, 14)
    return " ".join(out)
def one(i, res, text, q):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=MAXT, temperature=0.6,
                messages=[{"role": "user", "content": (f"Reference notes:\n{text}\n\n" if text else "") + q}])
    t0 = time.time(); first = None; usage = {}
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000) as r:
            for line in r:
                if not line.startswith(b"data: ") or b"[DONE]" in line: continue
                d = json.loads(line[6:])
                if d.get("usage"): usage = d["usage"]
                if first is None and d.get("choices") and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d["choices"]): first = time.time()
        end = time.time(); ct = usage.get("completion_tokens", 0)
        res[i] = (first - t0, (ct - 1) / (end - first), ct, end - t0, usage.get("prompt_tokens", 0))
    except Exception as e:
        res[i] = ("ERR", str(e)[:80])
def run(n, texts, tag):
    res = {}; ths = [threading.Thread(target=one, args=(i, res, texts[i], f"Ignore the notes. Write a detailed essay on {TOPICS[i % 4]} ({tag} {time.time()}).")) for i in range(n)]
    t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]; wall = time.time() - t
    ok = [v for v in res.values() if v[0] != "ERR"]
    if len(ok) < n: return f"errors: {[v for v in res.values() if v[0] == 'ERR'][:1]}"
    return (f"prompt {ok[0][4]:6d} tok | first token {min(v[0] for v in ok):6.1f}-{max(v[0] for v in ok):6.1f} s | "
            f"decode each {min(v[1] for v in ok):5.1f}-{max(v[1] for v in ok):5.1f} sum {sum(v[1] for v in ok):6.1f} tok/s | wall {wall:6.1f} s")
for size in SIZES:
    for n in NS:
        texts = [doc(f"{size}-{n}-{i}-{time.time()}", size) if size else "" for i in range(n)]
        print(f"ctx~{size:6d}w n={n} fresh : {run(n, texts, 'a')}", flush=True)
        print(f"ctx~{size:6d}w n={n} cached: {run(n, texts, 'b')}", flush=True)
print("DONE")
