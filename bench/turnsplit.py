# Shape-only: hashes and lengths of request parts, timings from the gateway log. Prints aggregates only.
# Where does the time of streaming (agent) turns go in a given period? First against an ideal built from
# request prefixes, then (LOGRANGE=first,last line of the model log) joined with the model server's own
# counters: tokens actually read, tokenize / tower / read times, and whole-turn accounting.
# Note: for tool-calling streams the gateway's first-token time includes generation (the server sends a
# delta only once a tool call is complete), so the first part overstates waiting; trust the joined part.
#   T0 / T1: period reported ("YYYY-MM-DD HH:MM"); TRIE0: where prefix history starts (default 2026-11-23 00:00)
#   SKIP: "a,b" pairs of local times to leave out (service windows), separated by ";"
import json, os, sys, datetime, hashlib, collections, statistics as st
def ts(s): return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()
T0 = ts(os.environ["T0"]); T1 = ts(os.environ["T1"]) if os.environ.get("T1") else 9e18
TRIE0 = ts(os.environ.get("TRIE0", "2026-11-23 00:00"))
SKIP = [tuple(ts(x) for x in p.split(",")) for p in os.environ.get("SKIP", "").split(";") if p]
RATE = float(os.environ.get("RATE", "2440")); FLOOR = float(os.environ.get("FLOOR", "0.9"))
LOG = os.path.expanduser("~/llm-stack/logs/requests.jsonl"); CAPROOT = os.path.expanduser("~/llm-stack/captures")
def skipped(t): return any(a <= t < b for a, b in SKIP)
R = {}; allreq = []
for l in open(LOG):
    try: d = json.loads(l)
    except Exception: continue
    t = d.get("ts") or 0
    if t < TRIE0 or t >= T1 or d.get("probe"): continue
    allreq.append(d)
    if d.get("status") == 200:
        try: R[(int(d.get("id")), int(t))] = d
        except Exception: pass
IV = [(d["ts"], d["ts"] + (d.get("total_s") or 0), bool(d.get("stream")), d.get("ttft_s"), d) for d in allreq]
IV.sort(key=lambda x: x[0])
import bisect
starts = [x[0] for x in IV]
def inflight(a):
    """requests in flight at time a (arrived up to 15 min earlier)"""
    lo = bisect.bisect_left(starts, a - 900); hi = bisect.bisect_left(starts, a - 1e-6)
    return [x for x in IV[lo:hi] if x[1] > a]
files = []
for sub in os.listdir(CAPROOT):
    p = os.path.join(CAPROOT, sub)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        try: ns, rid = f[:-5].split("-", 1); ns = int(ns); rid = int(rid)
        except ValueError: continue
        if TRIE0 - 5 <= ns / 1e9 < T1: files.append((ns, rid, os.path.join(p, f)))
files.sort()
def req_for(ns, rid):
    t = int(ns / 1e9)
    for dt in (0, -1, 1):
        d = R.get((rid, t + dt))
        if d is not None: return d
    return None
def h(s): return hashlib.blake2b(s.encode("utf-8", "ignore") if isinstance(s, str) else s, digest_size=8).digest()
IMG = 4000
def parts(m):
    out = []; c = m.get("content"); role = m.get("role", "")
    if isinstance(c, str): out.append((h(role + "\0" + c), len(c), 0))
    elif isinstance(c, list):
        for p in c:
            if p.get("type") == "image_url":
                u = (p.get("image_url") or {}).get("url", ""); out.append((h(role + "\0img\0" + u), IMG, 1))
            else:
                t = p.get("text") or ""; out.append((h(role + "\0" + t), len(t), 0))
    else: out.append((h(role + "\0"), 0, 0))
    if m.get("tool_calls"):
        s = json.dumps(m["tool_calls"], sort_keys=True); out.append((h(role + "\0tc\0" + s), len(s), 0))
    return out
trie = {}; S = []; byreq = {}; nfiles = 0
for ns, rid, f in files:
    d = req_for(ns, rid)
    if not d or not d.get("stream"): continue
    try: b = json.load(open(f))["body"]
    except Exception: continue
    nfiles += 1
    atoms = []
    if b.get("tools"):
        s = json.dumps(b["tools"], sort_keys=True); atoms.append((h("tools\0" + s), len(s), 0))
    for m in b.get("messages") or []: atoms += parts(m)
    tot = sum(a[1] for a in atoms) or 1
    node = trie; shared = 0; k = 0
    for a in atoms:
        nxt = node.get(a[0])
        if nxt is None: break
        shared += a[1]; k += 1; node = nxt
    node = trie
    for a in atoms: node = node.setdefault(a[0], {})
    inp = d.get("input_tokens") or 0
    rec = dict(ts=d["ts"], ttft=d.get("ttft_s"), tot=d.get("total_s") or 0, inp=inp, out=d.get("output_tokens") or 0,
               new=inp * (1 - shared / tot), k=k, n=len(atoms), hs=[a[0] for a in atoms], cs=[a[1] for a in atoms], chars=tot,
               nimg=sum(a[2] for a in atoms))
    byreq[id(d)] = rec
    S.append(rec)
print(f"stream captures parsed {nfiles}; streams with prefix history {len(S)}", flush=True)
def lcp_chars(a, b):
    s = 0
    for x, y, c in zip(a["hs"], b["hs"], a["cs"]):
        if x != y: break
        s += c
    return s
tt = [r for r in S if r["ttft"] is not None and T0 <= r["ts"] < T1 and not skipped(r["ts"])]
if not tt: print("no stream turns in the period"); sys.exit()
span = (min(T1, max(r["ts"] for r in tt)) - T0) / 3600
print(f"period {os.environ['T0']} -> {os.environ.get('T1', 'now')}: {len(tt)} stream turns over {span:.1f} h, stream seconds {sum(r['tot'] for r in tt):.0f}, "
      f"first-token wait {sum(r['ttft'] for r in tt):.0f} s, output tokens {sum(r['out'] for r in tt)}")
print(f"prompt tokens {sum(r['inp'] for r in tt)}, new by part-prefix estimate {sum(r['new'] for r in tt):.0f} ({100 * sum(r['new'] for r in tt) / max(sum(r['inp'] for r in tt), 1):.1f}%)")
cls = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
detail = collections.defaultdict(lambda: [0, 0.0, 0.0])
hist = collections.defaultdict(lambda: [0, 0.0, 0.0])
sizeb = collections.defaultdict(lambda: [0, 0.0, 0.0])
for r in tt:
    ideal = FLOOR + r["new"] / RATE; ex = max(0.0, r["ttft"] - ideal)
    fl = inflight(r["ts"]); a = r["ts"]
    streams = [x for x in fl if x[2]]; jobs = [x for x in fl if not x[2]]
    reading = [x for x in streams if x[3] is None or a < x[0] + x[3]]
    sib = 0
    for x in streams:
        o = byreq.get(id(x[4]))
        if o is not None and lcp_chars(r, o) >= 0.5 * r["chars"]: sib += 1
    if not fl: key = "alone"
    elif not streams: key = "only jobs in flight"
    elif reading: key = "a stream still reading its prompt"
    else: key = "streams decoding only"
    c = cls[key]; c[0] += 1; c[1] += r["ttft"]; c[2] += ex; c[3] += ideal
    if streams:
        d = detail[("sibling of an in-flight stream (shares >= half its prompt)" if sib else "unrelated to the in-flight streams", key)]
        d[0] += 1; d[1] += r["ttft"]; d[2] += ex
    if not fl:
        b = "<0.3" if ex < 0.3 else "0.3-1" if ex < 1 else "1-3" if ex < 3 else "3-10" if ex < 10 else "10-30" if ex < 30 else ">=30"
        hh = hist[b]; hh[0] += 1; hh[1] += ex; hh[2] += min(max(0.0, (r["ttft"] - FLOOR) * RATE - r["new"]), r["inp"])
    sb = sizeb[("new<2k" if r["new"] < 2000 else "new 2-10k" if r["new"] < 10000 else "new 10-40k" if r["new"] < 40000 else "new>=40k")]
    sb[0] += 1; sb[1] += r["ttft"]; sb[2] += ex
tot_ex = sum(c[2] for c in cls.values())
print(f"\nideal = {FLOOR} s + new/{RATE:.0f} tok/s. Excess over ideal: {tot_ex:.0f} s = {100 * tot_ex / max(sum(r['tot'] for r in tt), 1):.0f}% of stream seconds")
for key in sorted(cls, key=lambda k: -cls[k][2]):
    c = cls[key]; print(f"  {key:36s} n={c[0]:5d} wait={c[1]:7.0f}s ideal={c[3]:7.0f}s excess={c[2]:7.0f}s mean wait {c[1] / c[0]:5.1f}s")
print("with streams in flight, by relation:")
for key in sorted(detail, key=lambda k: -detail[k][2]):
    d = detail[key]; print(f"  {key[0]:58s} | {key[1]:34s} n={d[0]:4d} wait={d[1]:6.0f}s excess={d[2]:6.0f}s")
print("alone turns by excess (s): count, excess seconds, implied re-read tokens")
for b in ("<0.3", "0.3-1", "1-3", "3-10", "10-30", ">=30"):
    hh = hist[b]; print(f"  {b:6s} n={hh[0]:5d} excess={hh[1]:7.0f}s reread~{hh[2] / 1e3:8.0f}k tok")
print("by new tokens:")
for b in ("new<2k", "new 2-10k", "new 10-40k", "new>=40k"):
    sb = sizeb[b]; print(f"  {b:11s} n={sb[0]:5d} wait={sb[1]:7.0f}s excess={sb[2]:7.0f}s")
al = sorted(r["ttft"] for r in tt if not inflight(r["ts"]))
if al: print(f"alone first token p50 {st.median(al):.2f} p90 {al[int(.9 * len(al))]:.2f} p99 {al[int(.99 * len(al))]:.2f}; prompt size p50 {st.median(r['inp'] for r in tt):.0f} max {max(r['inp'] for r in tt)}")

# ---- join with the server's own counters (model log; numbers only) ----
LR = os.environ.get("LOGRANGE")
if LR:
    import re
    a, b = (int(x) for x in LR.split(","))
    day0 = datetime.datetime.fromtimestamp(T0).replace(hour=0, minute=0, second=0, microsecond=0)
    cur = None; last_h = None; dayoff = 0; pins = []; fetch = []; toks = []; nstart = 0
    rp = re.compile(r"pin saved at (\d+) of (\d+) \(read from (\d+)\)"); rf = re.compile(r"cache fetch .* prompt=(\d+) reuse=(\d+)")
    rt = re.compile(r"tokenize\s+[\d.]+ ms prompt=(\d+)"); rv = re.compile(r"trunk prefill (\d+) tok ([\d.]+) ms"); rw = re.compile(r"tower\+embed ([\d.]+) ms"); reads = []; pend = None; ptok = 0.0; ptow = 0.0; rm = re.compile(r"^\[tp-mem r0\] (\d\d):(\d\d):(\d\d)")
    with open(os.path.expanduser("~/llm-stack/logs/tp-cluster.log"), errors="replace") as f:
        for i, l in enumerate(f, 1):
            if i < a: continue
            if i > b: break
            m = rm.match(l)
            if m:
                hh = int(m.group(1))
                if last_h is not None and hh < last_h: dayoff += 1
                last_h = hh
                cur = (day0 + datetime.timedelta(days=dayoff, hours=hh, minutes=int(m.group(2)), seconds=int(m.group(3)))).timestamp(); continue
            if "starting mlx_lm.server" in l: nstart += 1; continue
            if cur is None: continue
            m = rp.search(l)
            if m: pins.append((cur, int(m.group(1)), int(m.group(2)), int(m.group(3)))); continue
            m = rf.search(l)
            if m: fetch.append((cur, int(m.group(1)), int(m.group(2)))); continue
            m = rt.search(l)
            if m: toks.append((cur, int(m.group(1)))); pend = int(m.group(1)); ptok = float(m.group(0).split()[1]); ptow = 0.0; continue
            m = rw.search(l)
            if m and pend is not None: ptow = float(m.group(1)); continue
            m = rv.search(l)
            if m and pend is not None: reads.append((cur, pend, int(m.group(1)), float(m.group(2)), ptok, ptow)); pend = None
    print(f"\nserver log lines {a}-{b}: {len(pins)} pin lines, {len(fetch)} slow-fetch lines, {len(toks)} tokenize lines, server starts {nstart}")
    def near(lst, t, n, idx):
        best = None
        for e in lst:
            if abs(e[idx] - n) <= 2 and -150 <= e[0] - t <= 400:
                if best is None or abs(e[0] - t) < abs(best[0] - t): best = e
        return best
    J = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0]); unm = [0, 0.0]; rows = []
    for r in tt:
        if r["inp"] < 8000: continue
        p = near(pins, r["ts"], r["inp"], 2); fq = near(fetch, r["ts"], r["inp"], 1)
        rd_ = near(reads, r["ts"], r["inp"], 1)
        if rd_ is not None: read = rd_[2]; src = "image path"
        elif p is not None: read = p[2] - p[3]; src = "image path"
        elif fq is not None: read = fq[1] - fq[2]; src = "text path"
        else: unm[0] += 1; unm[1] += r["ttft"]; continue
        short = max(0.0, read - r["new"])           # tokens read that an earlier request had already covered
        fl = inflight(r["ts"]); key = "alone" if not fl else "only jobs in flight" if not any(x[2] for x in fl) else "a stream in flight"
        b_ = "re-read <1k" if short < 1000 else "re-read 1-4k" if short < 4000 else "re-read 4-20k" if short < 20000 else "re-read >=20k"
        j = J[(key, b_)]; j[0] += 1; j[1] += r["ttft"]; j[2] += read; j[3] += r["new"]; j[4] += short
        rows.append((r, read, short, key, src))
    print(f"long turns (>= 8k tokens) matched to a server counter: {len(rows)} (unmatched {unm[0]}, their wait {unm[1]:.0f} s)")
    print("tokens the server actually read vs tokens new by the part-prefix estimate:")
    for key in sorted(J):
        j = J[key]; print(f"  {key[0]:20s} {key[1]:14s} n={j[0]:4d} wait={j[1]:6.0f}s read={j[2] / 1e3:8.0f}k new={j[3] / 1e3:7.0f}k avoidable={j[4] / 1e3:7.0f}k (~{j[4] / RATE:5.0f} s at {RATE:.0f} tok/s)")
    tr = sum(x[1] for x in rows); tn = sum(x[0]["new"] for x in rows); tsh = sum(x[2] for x in rows)
    print(f"  total: read {tr / 1e3:.0f}k, new {tn / 1e3:.0f}k, avoidable re-read {tsh / 1e3:.0f}k tokens = ~{tsh / RATE:.0f} s of {sum(x[0]['ttft'] for x in rows):.0f} s first-token wait on these turns")
    z = [x for x in rows if x[1] >= 0.98 * x[0]["inp"] and x[0]["inp"] >= 20000]
    print(f"  full re-reads of prompts >= 20k: {len(z)} turns, {sum(x[0]['inp'] for x in z) / 1e3:.0f}k tokens, wait {sum(x[0]['ttft'] for x in z):.0f} s; of them avoidable (seen before) {sum(1 for x in z if x[2] >= 0.5 * x[0]['inp'])}")
    # wait not explained by reading: per matched turn, wait - (FLOOR + read/RATE)
    q = collections.defaultdict(lambda: [0, 0.0])
    for r, read, short, key, src in rows:
        e = max(0.0, r["ttft"] - (FLOOR + read / RATE)); q[key][0] += 1; q[key][1] += e
    print("wait beyond floor + own reading (queueing, sharing the GPU):")
    for key in q: print(f"  {key:20s} n={q[key][0]:4d} {q[key][1]:6.0f} s")
    rd = sorted(x[1] for x in rows); print(f"tokens read per long turn: p50 {st.median(rd):.0f} p75 {rd[int(.75 * len(rd))]} p90 {rd[int(.9 * len(rd))]} ; new p50 {st.median(x[0]['new'] for x in rows):.0f}")

    # server-side decomposition for turns matched to an image-path read line
    D = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0]); PB = collections.defaultdict(lambda: [0, 0, 0.0]); OB = collections.defaultdict(lambda: [0, 0.0])
    for r in tt:
        if r["inp"] < 8000: continue
        e = near(reads, r["ts"], r["inp"], 1)
        if e is None: continue
        fl = inflight(r["ts"]); key = "alone" if not fl else "only jobs in flight" if not any(x[2] for x in fl) else "a stream in flight"
        d = D[key]; other = r["ttft"] - (e[3] + e[4] + e[5]) / 1e3
        d[0] += 1; d[1] += r["ttft"]; d[2] += e[4] / 1e3; d[3] += e[5] / 1e3; d[4] += e[3] / 1e3; d[5] += other
        if e[2] >= 1500:
            pb = PB[("ctx<60k" if r["inp"] < 60000 else "ctx 60-120k" if r["inp"] < 120000 else "ctx>=120k")]; pb[0] += 1; pb[1] += e[2]; pb[2] += e[3] / 1e3
        ob = OB[(key, "<0.5" if other < 0.5 else "0.5-1" if other < 1 else "1-2" if other < 2 else "2-5" if other < 5 else "5-15" if other < 15 else ">=15")]; ob[0] += 1; ob[1] += other
    print("\nserver-side split of the first-token wait (turns with an image-path read line):")
    for key in D:
        d = D[key]; print(f"  {key:20s} n={d[0]:4d} wait={d[1]:6.0f}s = tokenize {d[2]:5.0f} + image tower {d[3]:5.0f} + reading {d[4]:6.0f} + everything else {d[5]:6.0f} (mean {d[5] / d[0]:.2f} s)")
    print("reading speed in the server for reads >= 1,500 tokens:")
    for key in PB: print(f"  {key:12s} n={PB[key][0]:3d} {PB[key][1] / max(PB[key][2], 1e-9):6.0f} tok/s ({PB[key][1] / 1e3:.0f}k tokens, {PB[key][2]:.0f} s)")
    print("'everything else' per turn (s): count, seconds")
    for key in ("alone", "only jobs in flight", "a stream in flight"):
        print("  " + key + ": " + "  ".join(f"{b}: {OB[(key, b)][0]} / {OB[(key, b)][1]:.0f}s" for b in ("<0.5", "0.5-1", "1-2", "2-5", "5-15", ">=15")))

    # whole-turn accounting: is the "everything else" simply output generated before the first visible delta?
    A = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0, 0.0]); late = collections.defaultdict(lambda: [0, 0])
    for r in tt:
        if r["inp"] < 8000: continue
        e = near(reads, r["ts"], r["inp"], 1)
        if e is None: continue
        fl = inflight(r["ts"]); key = "alone" if not fl else "only jobs in flight" if not any(x[2] for x in fl) else "a stream in flight"
        pre = (e[3] + e[4] + e[5]) / 1e3; a_ = A[key]
        a_[0] += 1; a_[1] += r["tot"]; a_[2] += pre; a_[3] += r["out"]; a_[4] += r["tot"] - r["ttft"]
        after = r["tot"] - r["ttft"]; l_ = late[key]; l_[0] += 1
        if r["out"] >= 8 and after < 0.25 * (r["out"] / 135.0): l_[1] += 1      # most of the output existed before the first delta
        if r["out"] >= 8 and after < 0.25 * (r["out"] / 135.0): a_[5] += r["out"]; a_[6] += r["ttft"] - pre
    print("\nwhole turn = tokenize + tower + reading (server-measured) + the rest; rest per output token:")
    for key in A:
        a_ = A[key]; rest = a_[1] - a_[2]
        print(f"  {key:20s} n={a_[0]:4d} total={a_[1]:6.0f}s measured prompt work={a_[2]:5.0f}s rest={rest:6.0f}s output={a_[3]:7d} tok -> {a_[3] / max(rest - 0.35 * a_[0], 1e-9):6.1f} tok/s after 0.35 s fixed per turn; time after first delta {a_[4]:5.0f}s")
        print(f"  {'':20s} turns whose output was mostly generated before the first visible delta: {late[key][1]} of {late[key][0]}; their output {a_[5]} tok in {a_[6]:.0f} s of 'wait' -> {a_[5] / max(a_[6] - 0.35 * late[key][1], 1e-9):.1f} tok/s")
