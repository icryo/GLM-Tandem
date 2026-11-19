# Prompt cache behaviour: does a long conversation survive a burst of short requests, and is a shared
# system prompt reused by a new conversation? Prints first-token times.
import json, os, random, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
N = int(os.environ.get("CTX", "13500")); FLOOD = int(os.environ.get("FLOOD", "40"))
def ask(msgs, mt=40):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=mt, temperature=0.6, messages=msgs)
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
    return (first or time.time()) - t0, usage.get("prompt_tokens", 0), (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), "".join(text)
r = random.Random(time.time()); sysmsg = "Reference notes (ignore them): " + " ".join(r.choice(WORDS) for _ in range(N))
conv = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Say OK."}]
t, p, c, a = ask(conv); print(f"cold long prompt: {p} tok, first token {t:.1f} s", flush=True); conv += [{"role": "assistant", "content": a or "OK."}]
conv += [{"role": "user", "content": "Name three colours."}]
t, p, c, a = ask(conv); print(f"next turn: {p} tok (cached {c}), first token {t:.2f} s", flush=True); conv += [{"role": "assistant", "content": a or "."}]
t0 = time.time()
def small(i): ask([{"role": "system", "content": f"You are helper {i} {time.time()}."}, {"role": "user", "content": f"Give one fact about the number {i}."}], 30)
for k in range(0, FLOOD, 4):
    ths = [threading.Thread(target=small, args=(i,)) for i in range(k, min(k + 4, FLOOD))]; [x.start() for x in ths]; [x.join() for x in ths]
print(f"{FLOOD} short requests in {time.time() - t0:.1f} s", flush=True)
conv += [{"role": "user", "content": "Name three animals."}]
t, p, c, a = ask(conv); print(f"long conversation after the burst: {p} tok (cached {c}), first token {t:.2f} s", flush=True)
t, p, c, a = ask([{"role": "system", "content": sysmsg}, {"role": "user", "content": "A different question: name a river."}])
print(f"new conversation, same system prompt: {p} tok (cached {c}), first token {t:.2f} s", flush=True)
