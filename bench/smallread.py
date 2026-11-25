# Server counters only: time of small prompt reads on long conversations, by token count (first occurrence vs repeats).
import re, os, sys, collections, statistics as st
a, b = (int(x) for x in os.environ["LOGRANGE"].split(","))
rt = re.compile(r"tokenize\s+([\d.]+) ms prompt=(\d+)"); rv = re.compile(r"trunk prefill (\d+) tok ([\d.]+) ms"); rs = re.compile(r"split->generate\s+([\d.]+) ms")
rf = re.compile(r"cache fetch\s+([\d.]+) ms"); ri = re.compile(r"cache insert\s+([\d.]+) ms")
pend = None; rows = []; cur = None
with open(os.path.expanduser("~/llm-stack/logs/tp-cluster.log"), errors="replace") as f:
    for i, l in enumerate(f, 1):
        if i < a: continue
        if i > b: break
        m = rt.search(l)
        if m:
            pend = dict(tokms=float(m.group(1)), prompt=int(m.group(2)), fetch=0.0, ins=0.0, split=None, read=None); continue
        if pend is None: continue
        m = rv.search(l)
        if m and pend["read"] is None: pend["read"] = (int(m.group(1)), float(m.group(2))); continue
        m = rf.search(l)
        if m and pend["read"] is None: pend["fetch"] = float(m.group(1)); continue
        m = ri.search(l)
        if m and pend["read"] is not None and pend["split"] is None: pend["ins"] += float(m.group(1)); continue
        m = rs.search(l)
        if m and pend["read"] is not None:
            pend["split"] = float(m.group(1)); rows.append(pend); pend = None
L = [r for r in rows if r["prompt"] >= 60000]
print(f"long-conversation turns with a complete trace: {len(L)}")
seen = set(); first = []; rep = []
for r in L:
    n, ms = r["read"]
    if n > 400: continue
    (rep if n in seen else first).append((n, ms)); seen.add(n)
def show(name, xs):
    for lo, hi in ((1, 24), (24, 48), (48, 96), (96, 200), (200, 400)):
        v = [ms for n, ms in xs if lo <= n < hi]
        if v: print(f"  {name:22s} {lo:3d}-{hi:3d} tok: n={len(v):4d} median {st.median(v):6.1f} ms  p10 {sorted(v)[len(v) // 10]:6.1f}  p90 {sorted(v)[int(.9 * len(v))]:6.1f}")
show("first time this size", first); show("size seen before", rep)
sm = [r for r in L if r["read"][0] <= 400]
print(f"per turn medians (reads <= 400 tok, n={len(sm)}): tokenize {st.median(r['tokms'] for r in sm):.0f} ms, cache fetch {st.median(r['fetch'] for r in sm):.0f}, read {st.median(r['read'][1] for r in sm):.0f} (tokens {st.median(r['read'][0] for r in sm):.0f}), insert {st.median(r['ins'] for r in sm):.0f}, start of generation {st.median(r['split'] for r in sm):.0f}")
print(f"sums over all {len(L)} turns: tokenize {sum(r['tokms'] for r in L) / 1e3:.0f} s, fetch {sum(r['fetch'] for r in L) / 1e3:.0f} s, reads {sum(r['read'][1] for r in L) / 1e3:.0f} s (of which reads <= 400 tok {sum(r['read'][1] for r in sm) / 1e3:.0f} s), insert {sum(r['ins'] for r in L) / 1e3:.0f} s, start of generation {sum(r['split'] for r in L) / 1e3:.0f} s")
by = collections.defaultdict(list)
for r in L: by[r["prompt"] // 40000 * 40].append(r["tokms"])
print("tokenize ms by prompt size: " + "  ".join(f"{k}-{k + 40}k: {st.median(v):.0f} (n={len(v)})" for k, v in sorted(by.items())))
