# Timing and sizes only. Does a prompt read cost more after the GPU has been idle? Joins the model server's
# "[tp-vis] tower+embed / trunk prefill N tok T ms" lines to gateway requests by token count inside the minute bracket.
import json, os, re, datetime, collections, bisect
T0 = 1795624560.0; A = int(os.environ.get("FROM", "752715"))
req = []
for l in open(os.path.expanduser("~/llm-stack/logs/requests.jsonl")):
    try: d = json.loads(l)
    except Exception: continue
    if (d.get("ts") or 0) >= T0: req.append(d)
req.sort(key=lambda d: d["ts"])
ends = []; info = {}
for i, d in enumerate(req):
    t0 = d["ts"]; prev_end = max((e for e in ends[-60:]), default=0.0)
    infl = sum(1 for e in ends[-60:] if e > t0)
    info[id(d)] = (t0 - prev_end, infl); ends.append(t0 + (d.get("total_s") or 0))
jobs = [d for d in req if not d.get("probe") and d.get("status") == 200 and d.get("input_tokens")]
byn = collections.defaultdict(list)
for d in jobs: byn[d["input_tokens"]].append(d)
day0 = datetime.datetime.fromtimestamp(T0).replace(hour=0, minute=0, second=0, microsecond=0)
rm = re.compile(r"^\[tp-mem r0\] (\d\d):(\d\d):(\d\d)"); rw = re.compile(r"tower\+embed ([\d.]+) ms"); rv = re.compile(r"trunk prefill (\d+) tok ([\d.]+) ms")
cur = None; last_h = None; dayoff = 0; tower = None; out = []; used = set(); nread = 0
with open(os.path.expanduser("~/llm-stack/logs/tp-cluster.log"), errors="replace") as f:
    for i, l in enumerate(f, 1):
        if i < A: continue
        m = rm.match(l)
        if m:
            hh = int(m.group(1))
            if last_h is not None and hh < last_h: dayoff += 1
            last_h = hh; cur = (day0 + datetime.timedelta(days=dayoff, hours=hh, minutes=int(m.group(2)), seconds=int(m.group(3)))).timestamp(); continue
        m = rw.search(l)
        if m: tower = float(m.group(1)); continue
        m = rv.search(l)
        if m and cur is not None:
            n, t = int(m.group(1)), float(m.group(2)); nread += 1
            cands = [d for k in (n, n + 1, n + 2) for d in byn.get(k, ()) if cur - 5 <= d["ts"] <= cur + 66 and id(d) not in used]
            if len(cands) == 1:
                d = cands[0]; used.add(id(d)); gap, infl = info[id(d)]
                out.append((gap, infl, n, t, tower, d.get("stream"), d.get("total_s"), d.get("output_tokens") or 0))
            tower = None
print("read lines", nread, "matched uniquely", len(out))
def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
B = ((-1e9, 0.3), (0.3, 1), (1, 2), (2, 4), (4, 10), (10, 60), (60, 1e9))
for lo_n, hi_n in ((16, 128), (300, 900), (900, 3000)):
    print(f"reads of {lo_n}-{hi_n} tokens, request alone at arrival: time minus 0.314 ms/token (ms), and raw")
    for lo, hi in B:
        v = [(t - 0.314 * n, t, n) for gap, infl, n, t, tw, s, tot, ot in out if infl == 0 and lo <= gap < hi and lo_n <= n < hi_n]
        if len(v) >= 6: print(f"   idle {max(lo, 0):>5}-{min(hi, 999):<5} s n={len(v):4d}  residual p25 {q([x[0] for x in v], .25):6.0f} p50 {q([x[0] for x in v], .5):6.0f} p75 {q([x[0] for x in v], .75):6.0f}   raw p50 {q([x[1] for x in v], .5):6.0f}  tokens p50 {q([x[2] for x in v], .5)}")
print("image tower when it ran (> 20 ms), alone at arrival:")
for lo, hi in B:
    v = [tw for gap, infl, n, t, tw, s, tot, ot in out if infl == 0 and lo <= gap < hi and tw and tw > 20]
    if len(v) >= 6: print(f"   idle {max(lo, 0):>5}-{min(hi, 999):<5} s n={len(v):4d}  tower p25 {q(v, .25):6.0f} p50 {q(v, .5):6.0f} p75 {q(v, .75):6.0f} ms")
print("job total minus tower minus read minus output/143.7 (s), small jobs alone:")
for lo, hi in B:
    v = [tot - (tw or 0) / 1e3 - t / 1e3 - ot / 143.7 for gap, infl, n, t, tw, s, tot, ot in out if infl == 0 and not s and lo <= gap < hi and 300 <= n < 900 and ot >= 20]
    if len(v) >= 6: print(f"   idle {max(lo, 0):>5}-{min(hi, 999):<5} s n={len(v):4d}  rest p25 {q(v, .25):6.3f} p50 {q(v, .5):6.3f} p75 {q(v, .75):6.3f}")
