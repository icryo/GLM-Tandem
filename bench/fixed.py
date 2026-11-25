# N identical T=0 requests with a fixed prompt (same token sequence every time): prints tokens and decode time.
import json, os, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MT = int(os.environ.get("MT", "900")); N = int(os.environ.get("N", "4"))
P = os.environ.get("PROMPT", "Write a 700 word essay on the history of the steam engine.")
if os.environ.get("CTXW"):      # synthetic long context: seeded dictionary words in front of the same request
    import random
    _r = random.Random(11); _w = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
    P = "Here are some notes:\n" + " ".join(_r.choice(_w) for _ in range(int(os.environ["CTXW"]))) + "\n\n" + P
for i in range(N):
    body = dict(model="default_model", messages=[{"role": "user", "content": P}], max_tokens=MT, temperature=0, stream=True, stream_options={"include_usage": True})
    first = None; usage = None
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if first is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d.get("choices", [])): first = time.time()
    ct = usage["completion_tokens"]; print(f"{ct} tok  {ct / (time.time() - first):6.1f} tok/s", flush=True)
