#!/bin/zsh
# quiet.sh [MAX_WAIT_S] (on .252): wait until the gateway has nothing in flight and no real (non-probe)
# request ended in the last 60 s; exit 0 when quiet, 1 after MAX_WAIT_S (default 600).
P=/Users/a/.omlx/bin/omlx-cluster-python
MAX=${1:-600}; t0=$(date +%s)
while true; do
  $P - <<'PY' && exit 0
import json, sys, time, urllib.request, subprocess
s = json.load(urllib.request.urlopen("http://localhost:8080/gateway/stats", timeout=5))
busy = len(s.get("inflight") or []) + len((s.get("queue") or {}).get("decoding") or []) + len((s.get("queue") or {}).get("waiting") or [])
rows = subprocess.run(["tail", "-n", "100", "/Users/a/llm-stack/logs/requests.jsonl"], capture_output=True, text=True).stdout.splitlines()
last = 0
for l in rows:
    try: r = json.loads(l)
    except Exception: continue
    if r.get("probe"): continue
    last = max(last, float(r.get("ts", 0)) + float(r.get("total_s") or 0))
sys.exit(0 if busy == 0 and time.time() - last > 60 else 1)
PY
  [ $(( $(date +%s) - t0 )) -ge $MAX ] && { echo "not quiet after $MAX s"; exit 1; }
  sleep 5
done
