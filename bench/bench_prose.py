# Fresh vs warm prose decode. Each "fresh" prompt is unique so no prefix cache applies.
import json, os, sys, time, urllib.request, random
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
TEMP = float(os.environ.get("TEMP", "0.6")); N = int(os.environ.get("N", "4")); MT = int(os.environ.get("MT", "700"))
topics = ["the transistor", "the printing press", "plate tectonics", "the steam engine", "penicillin", "the telegraph", "container shipping", "the bicycle", "radio astronomy", "the Suez canal"]
def run(prompt):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=MT, temperature=TEMP, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    first = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if first is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d.get("choices", [])): first = time.time()
    ct = usage["completion_tokens"]; return ct / (time.time() - first), ct
random.seed(int(time.time()))
fresh = []; warm = []
for i in range(N):
    p = f"[{random.randint(0, 10**9)}] Write a 500 word essay on the history of {random.choice(topics)}."
    f, n1 = run(p); w, n2 = run(p)
    fresh.append(f); warm.append(w)
    print(f"fresh {f:6.1f} tok/s ({n1} tok)   same prompt again {w:6.1f} tok/s ({n2} tok)", flush=True)
print(f"PROSE temp={TEMP}: fresh mean {sum(fresh)/len(fresh):.1f} tok/s   warm mean {sum(warm)/len(warm):.1f} tok/s")
