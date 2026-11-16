# Soak through the gateway: multi-turn conversations, long outputs, random cancels. Reports decode tok/s and failures.
import json, random, sys, time, urllib.request
URL = "http://localhost:8080/v1/chat/completions"; random.seed(7)
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 300
topics = ["tides", "the transistor", "volcanoes", "TCP congestion control", "photosynthesis", "the printing press", "B-trees", "the Roman aqueducts"]
def stream(messages, max_tokens, temperature, cancel_after=None):
    body = dict(model="glm-5.3-flash", messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = []; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {}); s = dl.get("content") or ""; rs = dl.get("reasoning") or dl.get("reasoning_content") or ""
                if s or rs:
                    if first is None: first = time.time()
                    n += 1; text.append(s)
            if cancel_after and n >= cancel_after: return "".join(text), None, True
    ct = (usage or {}).get("completion_tokens", n)
    return "".join(text), (ct / (time.time() - first) if first and ct > 20 else None), False
end = time.time() + DUR; rates = []; fails = 0; reqs = 0; cancels = 0
while time.time() < end:
    conv = [{"role": "user", "content": f"Write about 250 words on {random.choice(topics)}."}]
    for turn in range(random.randint(1, 4)):
        try:
            cancel = random.random() < 0.12
            text, tps, cancelled = stream(conv, random.choice([200, 500, 900]), random.choice([0.0, 0.6, 0.8]), cancel_after=30 if cancel else None)
            reqs += 1
            if cancelled: cancels += 1; break
            if tps: rates.append(tps)
            if len(text.strip()) < 20 and tps: fails += 1; print("SHORT/EMPTY content", repr(text[:60]), flush=True)
            conv += [{"role": "assistant", "content": text}, {"role": "user", "content": random.choice(["Now summarize that in two sentences.", "Give three follow-up questions.", "Rewrite the first paragraph more formally."])}]
        except Exception as e:
            fails += 1; print("ERR", type(e).__name__, e, flush=True); time.sleep(2); break
rates.sort()
print(f"requests={reqs} cancels={cancels} failures={fails} decode tok/s: median {rates[len(rates)//2]:.1f} p10 {rates[len(rates)//10]:.1f} p90 {rates[9*len(rates)//10]:.1f} (n={len(rates)})")
