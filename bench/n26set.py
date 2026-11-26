# One fixed request sequence at T=0 against a freshly started server: output hashes (to compare two builds
# byte for byte) and timings (decode at short and long context, cold prompt reads, small reads).
# Run it right after a start, with nothing else in between: the cache state decides how prompts are cut into
# blocks, and with that the rounding.   SECTIONS=text,short,long60,long140,small   LABEL=name
import hashlib, json, os, random, sys, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
SECTIONS = os.environ.get("SECTIONS", "text,short,long60,long140,small").split(","); LABEL = os.environ.get("LABEL", "")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
def words(seed, n):
    r = random.Random(seed); return " ".join(r.choice(WORDS) for _ in range(n))
def sha(t): return hashlib.sha256(t.encode()).hexdigest()[:16]
def ask(prompt, mt, stream=True):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=mt, temperature=0, stream=stream)
    if stream: body["stream_options"] = {"include_usage": True}
    t0 = time.time(); first = None; usage = {}; out = []
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000) as r:
        if not stream:
            d = json.loads(r.read()); m = d["choices"][0]["message"]
            return (m.get("reasoning") or "") + "|" + (m.get("content") or ""), d["usage"], time.time() - t0, None
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta") or {}; piece = (dl.get("reasoning") or "") + (dl.get("content") or "")
                if piece:
                    if first is None: first = time.time()
                    out.append(piece)
    return "".join(out), usage, time.time() - t0, (first or time.time()) - t0
SAVED = {}
def line(name, text, usage, total, ttft, extra=""):
    SAVED[name] = text
    ct = usage.get("completion_tokens", 0); pt = usage.get("prompt_tokens", 0)
    dec = f" decode {ct / max(total - ttft, 1e-9):6.1f} tok/s" if ttft is not None else ""
    ft = f" first {ttft:7.2f} s" if ttft is not None else ""
    print(f"{sha(text)} {name}: prompt={pt} out={ct} total {total:7.2f} s{ft}{dec}{extra}", flush=True)
print(f"== {LABEL} {time.strftime('%H:%M:%S')}", flush=True)
if "text" in SECTIONS:
    for i, p in enumerate(["Write a 300 word essay on the history of the bicycle.",
                           "Write a Python function that merges two sorted lists, with a short docstring and three test cases.",
                           "Reply with a JSON object with keys name, year and summary describing the Suez canal.",
                           "List ten prime numbers greater than 100 and explain briefly how you checked each one.",
                           "Explain in 200 words how a transistor amplifies a signal."]):
        line(f"text {i}", *ask(p, 500, stream=False))
ESSAY = "Write a 700 word essay on the history of the steam engine."
if "short" in SECTIONS:
    for i in range(3): line(f"short {i}", *ask(ESSAY, 900))
for name, nwords in (("long60", 26000), ("long140", 61000)):
    if name not in SECTIONS: continue
    P = "Here are some notes:\n" + words(11 + nwords, nwords) + "\n\n" + ESSAY
    for i in range(4):
        text, usage, total, ttft = ask(P, 900)
        extra = f"  read {usage.get('prompt_tokens', 0) / max(ttft, 1e-9):7.0f} tok/s (cold)" if i == 0 else ""
        line(f"{name} {i}", text, usage, total, ttft, extra)
if "small" in SECTIONS:
    for nwords in (20, 35, 50, 130, 300, 900):
        ts = []
        for i in range(6):
            text, usage, total, _ = ask(words(1000 * nwords + i, nwords) + "\n\nReply with the single word OK.", 12, stream=False)
            ts.append(total); h = sha(text); pt = usage.get("prompt_tokens", 0)
        ts.sort(); print(f"{h} small read {nwords} words (~{pt} tokens) x6: median {1e3 * ts[3]:6.0f} ms  min {1e3 * ts[0]:6.0f} ms", flush=True)
if os.environ.get("SAVE"): json.dump(SAVED, open(os.environ["SAVE"], "w"))
print(f"== end {LABEL} {time.strftime('%H:%M:%S')}", flush=True)
