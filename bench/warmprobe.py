# After a fresh start: time the first and second use of each request shape (compiles show up as a slow first use).
import json, os, time, threading, urllib.request, random
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
rnd = random.Random(time.time())
def req(prompt, mt=24, temp=0.0, stream=False):
    body = dict(model=MODEL, max_tokens=mt, temperature=temp, stream=stream, messages=[{"role": "user", "content": prompt}])
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300) as r: r.read()
    return time.time() - t
def conc(n, mt=24, temp=0.0, words=8):
    out = [0] * n
    def one(i): out[i] = req(f"[{rnd.random()}] " + " ".join(rnd.choice(WORDS) for _ in range(words)) + f" Count from {i} to {i + 30}.", mt, temp)
    ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]; [t.start() for t in ths]; [t.join() for t in ths]; return max(out)
steps = [("1 stream T=0", lambda: conc(1)), ("1 stream T=0.7", lambda: conc(1, temp=0.7)), ("1 stream 200 tok", lambda: conc(1, mt=200))]
steps += [(f"{n} streams", (lambda n=n: conc(n, mt=40))) for n in (2, 3, 4, 5, 6, 7, 8)]
steps += [("2 streams T=0.7", lambda: conc(2, mt=40, temp=0.7)), ("prompt 700 tok", lambda: conc(1, words=450)), ("prompt 3k tok", lambda: conc(1, words=2000)), ("prompt 9k tok", lambda: conc(1, words=6000)), ("2 streams + 3k prompts", lambda: conc(2, words=2000))]
for name, f in steps:
    a = f(); b = f(); print(f"{name:26s} first {a:6.2f} s   again {b:6.2f} s   {'<-- one-time cost %.1f s' % (a - b) if a - b > 1.5 else ''}", flush=True)
