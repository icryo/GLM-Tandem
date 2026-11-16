# Concurrent requests against the server: per-request decode speed, wait before first token, total throughput.
import json, os, sys, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8080/v1/chat/completions"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
TOPICS = ["the history of the bicycle", "how tides work", "the invention of the printing press", "why the sky is dark at night",
          "the life cycle of a star", "how bread rises", "the origins of chess", "how bridges carry load"]
def one(i, res):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=300, temperature=float(os.environ.get("TEMP", "0.6")),
                messages=[{"role": "user", "content": f"Write a detailed essay on {TOPICS[i % len(TOPICS)]} (variant {time.time()})."}])
    t0 = time.time(); first = None; usage = {}
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
            for line in r:
                if not line.startswith(b"data: ") or b"[DONE]" in line: continue
                d = json.loads(line[6:])
                if d.get("usage"): usage = d["usage"]
                if first is None and d.get("choices") and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d["choices"]): first = time.time()
        end = time.time(); ct = usage.get("completion_tokens", 0)
        res[i] = (first - t0, (ct - 1) / (end - first), ct, end - t0)
    except Exception as e:
        res[i] = ("ERR", str(e)[:80])
for n in (1, 2, 4, 8):
    res = {}; ths = [threading.Thread(target=one, args=(i, res)) for i in range(n)]
    t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]; wall = time.time() - t
    ok = [v for v in res.values() if v[0] != "ERR"]; err = n - len(ok)
    if ok:
        toks = sum(v[2] for v in ok)
        print(f"{n} concurrent: total {toks / wall:6.1f} tok/s over {wall:5.1f} s | per-request decode {min(v[1] for v in ok):.0f}-{max(v[1] for v in ok):.0f} tok/s | first token after {min(v[0] for v in ok):.1f}-{max(v[0] for v in ok):.1f} s | errors {err}", flush=True)
    else:
        print(f"{n} concurrent: all failed: {list(res.values())[:2]}", flush=True)
