# A streaming agent-style request (unseeded, CTX-word context, ~400 tokens out) every GAP seconds next to WJ workers
# sending seeded non-streaming jobs back to back. Reports the stream's wait and rate, and the jobs' times.
import json, os, random, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WJ = int(os.environ.get("WJ", "2")); DUR = float(os.environ.get("DUR", "70")); GAP = float(os.environ.get("GAP", "14")); CTX = int(os.environ.get("CTX", "3000"))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
rnd = random.Random(time.time()); lock = threading.Lock(); jobs = []; streams = []
def job():
    b = dict(model=MODEL, max_tokens=350, temperature=0.8, seed=rnd.randrange(1, 2**31),
             messages=[{"role": "user", "content": "Notes: " + " ".join(rnd.choice(WORDS) for _ in range(1150)) + "\nIgnore the notes and write four paragraphs about rivers."}])
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600) as r: d = json.loads(r.read())
    with lock: jobs.append((time.time() - t, d["usage"]["completion_tokens"]))
def stream():
    b = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=400, temperature=0.6,
             messages=[{"role": "user", "content": "Notes: " + " ".join(rnd.choice(WORDS) for _ in range(CTX)) + "\nIgnore the notes and write a detailed essay about bridges."}])
    t0 = time.time(); first = None; usage = {}; gaps = []; last = None
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if d.get("choices") and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d["choices"]):
                now = time.time()
                if first is None: first = now
                if last is not None: gaps.append(now - last)
                last = now
    with lock: streams.append((first - t0, (usage.get("completion_tokens", 1) - 1) / max(last - first, 1e-6), max(gaps or [0]), time.time() - t0))
job(); end = time.time() + DUR
def jw():
    while time.time() < end: job()
jobs.clear(); ths = [threading.Thread(target=jw) for _ in range(WJ)]; [t.start() for t in ths]
st = []
while time.time() < end - 8:
    time.sleep(GAP if st else 3); t = threading.Thread(target=stream); t.start(); st.append(t)
[t.join() for t in ths + st]
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
print(f"agentmix WJ={WJ}: jobs {len(jobs)} done, median {med([j[0] for j in jobs]):.1f} s, max {max([j[0] for j in jobs] or [0]):.1f} s, {sum(j[1] for j in jobs) / DUR:.0f} tok/s | "
      f"streams {len(streams)}: first token median {med([s[0] for s in streams]):.1f} s (max {max(s[0] for s in streams):.1f}), rate median {med([s[1] for s in streams]):.0f} tok/s, longest pause {max(s[2] for s in streams):.1f} s, total median {med([s[3] for s in streams]):.1f} s")
