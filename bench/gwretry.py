# Gateway hold-and-retry check: short requests keep arriving while the model server is killed once.
# With the patch every request should finish with 200 (those caught by the restart wait for it); before it
# they failed with 502 for ~40 s. Synthetic prompts only.
#   URL=http://localhost:8080/v1/chat/completions DUR=100 KILL_AT=6 python3 gwretry.py
import json, os, subprocess, sys, threading, time, urllib.request, urllib.error
URL = os.environ.get("URL", "http://localhost:8080/v1/chat/completions"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
DUR = float(os.environ.get("DUR", "100")); KILL_AT = float(os.environ.get("KILL_AT", "6")); GAP = float(os.environ.get("GAP", "2"))
res = []; lock = threading.Lock(); t0 = time.time()
def one(i, stream):
    body = dict(model=MODEL, messages=[{"role": "user", "content": f"Reply with only the number {1000 + i}."}], max_tokens=40, temperature=0, stream=stream)
    t = time.time(); status = None; ok = False
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
            raw = r.read().decode(errors="replace"); status = r.status
            if stream:                                   # join the content deltas of the SSE stream
                text = ""
                for line in raw.splitlines():
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            for c in json.loads(line[6:]).get("choices") or []: text += (c.get("delta") or {}).get("content") or ""
                        except Exception: pass
            else:
                text = (json.loads(raw)["choices"][0]["message"].get("content") or "")
            ok = str(1000 + i) in text
    except urllib.error.HTTPError as e: status = e.code
    except Exception as e: status = f"{type(e).__name__}"
    with lock: res.append((round(t - t0, 1), i, stream, status, ok, round(time.time() - t, 1)))
ths = []; i = 0; killed = False
while time.time() - t0 < DUR:
    if not killed and time.time() - t0 >= KILL_AT:
        killed = True
        subprocess.run("pkill -f 'python3 tpserve.py'", shell=True); print(f"[{time.time() - t0:5.1f} s] killed the model server on this Mac", flush=True)
    th = threading.Thread(target=one, args=(i, i % 2 == 1)); th.start(); ths.append(th); i += 1
    time.sleep(GAP)
for th in ths: th.join()
res.sort()
bad = [r for r in res if r[3] != 200 or not r[4]]
for r in res: print(f"  sent {r[0]:6.1f} s  #{r[1]:2d} {'stream' if r[2] else 'json  '} status {r[3]} answer_ok {r[4]} took {r[5]:6.1f} s")
print(f"requests {len(res)}  failed {len(bad)}  slowest {max(r[5] for r in res):.1f} s")
print("PASS" if not bad else "FAIL")
