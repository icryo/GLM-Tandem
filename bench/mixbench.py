# Two or more concurrent requests whose contexts have given sizes (words); contexts are cached first,
# one at a time, so the measured round is decode only. CTX="6000,6000" or "300,6000" or "6000,48000".
import json, os, random, sys, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
CTX = [int(x) for x in os.environ.get("CTX", "6000,6000").split(",")]; MAXT = int(os.environ.get("MAXT", "200"))
def doc(seed, n):
    r = random.Random(seed); return " ".join(r.choice(WORDS) for _ in range(n))
EXTRA = json.loads(os.environ.get("EXTRA", "{}")); STAG = float(os.environ.get("STAG", "0"))
def one(i, res, msgs):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=MAXT, temperature=0.6, messages=msgs); body.update(EXTRA); time.sleep(STAG * i)
    t0 = time.time(); first = None; usage = {}; text = []
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if dl.get("content"): text.append(dl["content"])
                if first is None and (dl.get("content") or dl.get("reasoning")): first = time.time()
    end = time.time(); ct = usage.get("completion_tokens", 0)
    res[i] = (usage.get("prompt_tokens", 0), first - t0, (ct - 1) / max(end - first, 1e-6), "".join(text))
docs = [f"Reference notes (ignore them): {doc(f'{i}-{n}-{time.time()}', n)}" for i, n in enumerate(CTX)]
conv = []
for i, d in enumerate(docs):                       # cold prompt, then keep the conversation so later turns are append-only
    m = [{"role": "system", "content": d}, {"role": "user", "content": "Say OK."}]
    r = {}; one(i, r, m); print(f"cold {i}: prompt {r[i][0]} tok, first token {r[i][1]:.1f} s = {r[i][0] / r[i][1]:.0f} tok/s", flush=True)
    conv.append(m + [{"role": "assistant", "content": r[i][3] or "OK."}])
def turn(tag, idx, q):
    q = f"{q} ({time.time()})"
    res = {}; ths = [threading.Thread(target=one, args=(i, res, conv[i] + [{"role": "user", "content": q}])) for i in idx]
    [t.start() for t in ths]; [t.join() for t in ths]
    for i in sorted(res):
        print(f"{tag} {i}: prompt {res[i][0]} tok, first token {res[i][1]:.2f} s, decode {res[i][2]:.1f} tok/s", flush=True)
        conv[i] += [{"role": "user", "content": q}, {"role": "assistant", "content": res[i][3] or "."}]
for i in range(len(docs)): turn("alone", [i], "Write a detailed essay on how tides work.")
if len(docs) > 1:
    for k in range(int(os.environ.get("ROUNDS", "2"))): turn("together", list(range(len(docs))), "Write a detailed essay on the history of the bicycle.")
