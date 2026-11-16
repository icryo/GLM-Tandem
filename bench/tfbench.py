# TensorFold community-benchmark protocol (fixtures, 256 tokens, T=1 top_k 20 top_p 0.95 and T=0, seeds 1234+,
# 1 warm-up + 5 repeats, thinking off) against our server. Rate = (completion_tokens - 1) / (last - first delivery).
import json, os, statistics as st, sys, time, urllib.request
BASE = os.environ.get("BASE", "http://localhost:8080"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
FIX = {"code": (False, "Write a short Python function that computes the Fibonacci sequence and explain it."),
       "chat": (True, "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.")}
def run(fid, temp, seed):
    chat, prompt = FIX[fid]
    body = dict(model=MODEL, max_tokens=256, temperature=temp, stream=True, stream_options={"include_usage": True})
    if temp > 0: body.update(top_k=20, top_p=0.95)
    if os.environ.get("SEED") == "1": body["seed"] = seed
    if chat:
        body.update(messages=[{"role": "user", "content": prompt}], chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}); path = "/v1/chat/completions"
    else:
        body.update(prompt=prompt); path = "/v1/completions"
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    first = last = None; usage = None; n_del = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta") or {}
                txt = (dl.get("content") or "") + (dl.get("reasoning") or "") + (c.get("text") or "")
                if txt:
                    now = time.time(); first = first or now; last = now; n_del += 1
    ct = usage["completion_tokens"]
    return (ct - 1) / (last - first), ct
for fid in ("code", "chat"):
    for temp in (1.0, 0.0):
        run(fid, temp, 1233)                                       # warm-up, excluded
        v = [run(fid, temp, 1234 + i) for i in range(5)]
        r = [x[0] for x in v]
        print(f"{fid:5s} T={temp:.0f}: median {st.median(r):6.1f} tok/s  (min {min(r):6.1f}  max {max(r):6.1f})  tokens {[x[1] for x in v]}", flush=True)
