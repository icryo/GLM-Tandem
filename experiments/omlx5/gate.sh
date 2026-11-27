#!/bin/sh
# gate.sh: exit 0 only when the gateway has nothing in flight and no real (non-probe) request ended in the last 60 s.
ssh -o ConnectTimeout=5 a@192.168.1.252 '/Users/a/.omlx/bin/omlx-cluster-python - <<"PY"
import json, sys, time, urllib.request, subprocess
s = json.load(urllib.request.urlopen("http://localhost:8080/gateway/stats", timeout=5))
busy = len(s.get("inflight") or []) + len((s.get("queue") or {}).get("decoding") or []) + len((s.get("queue") or {}).get("waiting") or [])
last = 0
for l in subprocess.run(["tail", "-n", "200", "/Users/a/llm-stack/logs/requests.jsonl"], capture_output=True, text=True).stdout.splitlines():
    try: r = json.loads(l)
    except Exception: continue
    if r.get("probe"): continue
    last = max(last, float(r.get("ts", 0)) + float(r.get("total_s") or 0))
age = time.time() - last  # noqa
print(f"gate: inflight={busy} last_real_request_age_s={age:.0f}")
sys.exit(0 if busy == 0 and age > 60 else 1)
PY'
