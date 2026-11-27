#!/bin/sh
# gatewait.sh [max_s]: wait (on the gateway Mac) until nothing is in flight and no real request ended in the last 60 s.
ssh -o ConnectTimeout=5 a@192.168.1.252 "/Users/a/.omlx/bin/omlx-cluster-python - ${1:-300}" <<'PY'
import json, sys, time, urllib.request, subprocess
deadline = time.time() + float(sys.argv[1])
while True:
    s = json.load(urllib.request.urlopen("http://localhost:8080/gateway/stats", timeout=5))
    busy = len(s.get("inflight") or []) + len((s.get("queue") or {}).get("decoding") or []) + len((s.get("queue") or {}).get("waiting") or [])
    last = 0
    for l in subprocess.run(["tail", "-n", "200", "/Users/a/llm-stack/logs/requests.jsonl"], capture_output=True, text=True).stdout.splitlines():
        try: r = json.loads(l)
        except Exception: continue
        if r.get("probe"): continue
        last = max(last, float(r.get("ts", 0)) + float(r.get("total_s") or 0))
    age = time.time() - last
    if busy == 0 and age > 60:
        print(f"gate open: inflight=0 last_real_request_age_s={age:.0f}"); sys.exit(0)
    if time.time() > deadline:
        print(f"gate closed: inflight={busy} age={age:.0f}"); sys.exit(1)
    time.sleep(5)
PY
