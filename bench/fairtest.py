# NDEC streams decode; after 2 s a cold long prompt arrives. Reports the decoders' speed before, during and
# after the prompt loads, and the newcomer's time to first token.
import json, os, random, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
N = int(os.environ.get("CTX", "13500")); NDEC = int(os.environ.get("NDEC", "1")); MT = int(os.environ.get("MT", "4000"))
def ask(msgs, mt, stamps):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=mt, temperature=0.6, messages=msgs)
    t0 = time.time(); usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if dl.get("content") or dl.get("reasoning"): stamps.append(time.time())
    return usage
r = random.Random(time.time()); sysmsg = "Reference notes (ignore them): " + " ".join(r.choice(WORDS) for _ in range(N))
dec = [[] for _ in range(NDEC)]; use = [None] * NDEC
def d(i): use[i] = ask([{"role": "user", "content": f"Write a very long, detailed history of the city number {i} of an imaginary country, at least 3000 words ({time.time()})."}], MT, dec[i])
ths = [threading.Thread(target=d, args=(i,)) for i in range(NDEC)]; [t.start() for t in ths]
time.sleep(3); t1 = time.time(); st = []
u = ask([{"role": "system", "content": sysmsg}, {"role": "user", "content": "Say OK."}], 20, st); t2 = st[0] if st else time.time()
[t.join() for t in ths]
print(f"newcomer: {u.get('prompt_tokens')} tok, first token after {t2 - t1:.1f} s = {u.get('prompt_tokens', 0) / (t2 - t1):.0f} tok/s")
for i in range(NDEC):
    s = dec[i]; n = use[i].get("completion_tokens", 0); k = n / max(len(s), 1)      # tokens per delivered chunk
    pre = [x for x in s if x < t1]; mid = [x for x in s if t1 <= x < t2]; post = [x for x in s if x >= t2 + 0.5]
    gaps = [b - a for a, b in zip(s, s[1:]) if t1 <= a < t2]
    big = sorted(((b - a, a - t1) for a, b in zip(s, s[1:]) if t1 - 1 <= a < t2 + 2), reverse=True)[:4]
    print("   longest pauses (length s, start s after the prompt arrived; prompt done at %.1f): " % (t2 - t1) + ", ".join(f"{g:.2f}@{w:.1f}" for g, w in big))
    print(f"stream {i}: before {k * len(pre) / max(pre[-1] - pre[0], 1e-6):.0f} tok/s | while the prompt loads {k * len(mid) / (t2 - t1):.1f} tok/s (longest pause {max(gaps or [0]):.2f} s) | after {k * len(post) / max(post[-1] - post[0], 1e-6) if len(post) > 1 else 0:.0f} tok/s | {n} tok")
