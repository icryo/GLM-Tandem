# Timings and sizes only (gateway request log + the model server's own counters). Where do job (non-stream) seconds go?
import json, os, re, sys, datetime, collections, bisect, statistics as st
def ts(s): return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()
T0 = ts(os.environ["T0"]); T1 = ts(os.environ["T1"]) if os.environ.get("T1") else 9e18
a, b = (int(x) for x in os.environ["LOGRANGE"].split(","))
allreq = []
for l in open(os.path.expanduser("~/llm-stack/logs/requests.jsonl")):
    try: d = json.loads(l)
    except Exception: continue
    t = d.get("ts") or 0
    if t < T0 - 900 or t >= T1 or d.get("probe"): continue
    allreq.append(d)
allreq.sort(key=lambda d: d["ts"]); starts = [d["ts"] for d in allreq]
def inflight(t):
    lo = bisect.bisect_left(starts, t - 900); hi = bisect.bisect_left(starts, t - 1e-6)
    return [d for d in allreq[lo:hi] if d["ts"] + (d.get("total_s") or 0) > t]
day0 = datetime.datetime.fromtimestamp(T0).replace(hour=0, minute=0, second=0, microsecond=0)
rt = re.compile(r"tokenize\s+([\d.]+) ms prompt=(\d+)"); rw = re.compile(r"tower\+embed ([\d.]+) ms"); rv = re.compile(r"trunk prefill (\d+) tok ([\d.]+) ms")
rm = re.compile(r"^\[tp-mem r0\] (\d\d):(\d\d):(\d\d)")
cur = None; last_h = None; dayoff = 0; reads = []; pend = None
with open(os.path.expanduser("~/llm-stack/logs/tp-cluster.log"), errors="replace") as f:
    for i, l in enumerate(f, 1):
        if i < a: continue
        if i > b: break
        m = rm.match(l)
        if m:
            hh = int(m.group(1))
            if last_h is not None and hh < last_h: dayoff += 1
            last_h = hh; cur = (day0 + datetime.timedelta(days=dayoff, hours=hh, minutes=int(m.group(2)), seconds=int(m.group(3)))).timestamp(); continue
        if cur is None: continue
        m = rt.search(l)
        if m: pend = [cur, int(m.group(2)), None, None, float(m.group(1)), 0.0]; continue
        if pend is None: continue
        m = rw.search(l)
        if m: pend[5] = float(m.group(1)); continue
        m = rv.search(l)
        if m: pend[2] = int(m.group(1)); pend[3] = float(m.group(2)); reads.append(pend); pend = None
byn = collections.defaultdict(list)
for e in reads: byn[e[1]].append(e)
def near(t, n):
    best = None
    for k in (n - 1, n, n + 1):
        for e in byn.get(k, ()):
            if -150 <= e[0] - t <= 400 and (best is None or abs(e[0] - t) < abs(best[0] - t)): best = e
    return best
J = [d for d in allreq if not d.get("stream") and d.get("status") == 200 and T0 <= d["ts"] < T1 and d.get("input_tokens")]
print(f"jobs {len(J)}: total {sum(d['total_s'] for d in J):.0f} s, output {sum(d.get('output_tokens') or 0 for d in J)} tokens, prompt p50 {st.median(d['input_tokens'] for d in J):.0f} tokens; gateway queue wait {sum(d.get('queue_wait_s') or 0 for d in J):.0f} s, gated {sum(1 for d in J if d.get('gated'))}")
A = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0, 0]); un = 0
RB = collections.defaultdict(lambda: [0, 0, 0.0])
for d in J:
    e = near(d["ts"], d["input_tokens"])
    if e is None: un += 1; continue
    fl = inflight(d["ts"]); key = "alone" if not fl else "1 other in flight" if len(fl) == 1 else "2+ others in flight"
    x = A[key]; x[0] += 1; x[1] += d["total_s"]; x[2] += e[4] / 1e3; x[3] += e[5] / 1e3; x[4] += e[3] / 1e3; x[5] += d.get("output_tokens") or 0; x[6] += e[2]
    rb = RB["<64" if e[2] < 64 else "64-256" if e[2] < 256 else "256-700" if e[2] < 700 else "700-1500" if e[2] < 1500 else "1500-2500" if e[2] < 2500 else ">=2500"]; rb[0] += 1; rb[1] += e[2]; rb[2] += e[3]
print(f"matched to a server read line: {sum(x[0] for x in A.values())} (unmatched {un})")
for key in ("alone", "1 other in flight", "2+ others in flight"):
    x = A[key]
    if not x[0]: continue
    rest = x[1] - x[2] - x[3] - x[4]
    print(f"  {key:20s} n={x[0]:5d} total={x[1]:6.0f}s = tokenize {x[2]:4.0f} + image tower {x[3]:5.0f} + reading {x[4]:5.0f} ({x[6] / max(x[0], 1):.0f} tok each) + rest {rest:6.0f}; output {x[5]} tok -> {x[5] / max(rest - 0.1 * x[0], 1e-9):6.1f} tok/s after 0.1 s fixed; mean job {x[1] / x[0]:.2f} s")
print("prompt reads by size (tokens read, server-measured):")
for k in ("<64", "64-256", "256-700", "700-1500", "1500-2500", ">=2500"):
    rb = RB[k]
    if rb[0]: print(f"  {k:9s} n={rb[0]:5d} mean {rb[1] / rb[0]:6.0f} tok in {rb[2] / rb[0]:6.0f} ms  ({1e3 * rb[1] / max(rb[2], 1e-9):5.0f} tok/s), total {rb[2] / 1e3:5.0f} s")
