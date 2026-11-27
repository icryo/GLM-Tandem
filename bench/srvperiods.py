# Server-side counters per period (counts and timings only), from ~/llm-stack/logs/tp-cluster.log.
# Lines carry no time, so each line gets the time of the latest access-log line before it
# ('127.0.0.1 - - [26/Nov/2026 18:20:30] ...').
#   PERIODS="name=YYYY-mm-dd HH:MM,YYYY-mm-dd HH:MM;name2=...,now" python srvperiods.py
import datetime, os, re, sys, time
def ts(s): return time.time() if s.strip() == "now" else datetime.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M").timestamp()
periods = [(p.split("=")[0], *[ts(x) for x in p.split("=")[1].split(",")]) for p in os.environ["PERIODS"].split(";")]
r_acc = re.compile(r"\[(\d\d/\w\w\w/\d{4} \d\d:\d\d:\d\d)\]")
r_mtp = re.compile(r"\[tp-mtp\] coupled=\w+ tok=(\d+) tps=([\d.]+) cycles=(\d+) .*?tok/cycle=([\d.]+) head (\d+)/(\d+) lookup (\d+)/(\d+)")
r_ms = re.compile(r"\[tp-mspec\] streams=(\d+) steps=(\d+) tokens=(\d+) total ([\d.]+) tok/s ([\d.]+) ms/step drafts (\d+)/(\d+)")
r_vis = re.compile(r"\[tp-vis\] trunk prefill (\d+) tok ([\d.]+) ms")
r_pr = re.compile(r"\[tp-trace\] \+\s+[\d.]+ ms prompt\(\)\s+([\d.]+) ms")
r_tok = re.compile(r"\[tp-trace\] \+\s+[\d.]+ ms tokenize\s+[\d.]+ ms prompt=(\d+)")
S = {p[0]: dict(mtp=[], ms=[], vis=[], pr=[]) for p in periods}
now = 0.0; last_prompt = None
for l in open(os.path.expanduser("~/llm-stack/logs/tp-cluster.log"), errors="replace"):
    m = r_acc.search(l)
    if m:
        now = datetime.datetime.strptime(m.group(1), "%d/%b/%Y %H:%M:%S").timestamp(); continue
    name = next((p[0] for p in periods if p[1] <= now < p[2]), None)
    if name is None: continue
    if (m := r_mtp.search(l)):
        tok, tps, cyc, tpc, ha, ho, la, lo = m.groups()
        if int(tok) >= 20: S[name]["mtp"].append((int(tok), float(tps), int(cyc), int(ha), int(ho), int(la), int(lo)))
    elif (m := r_ms.search(l)):
        S[name]["ms"].append(tuple(float(x) for x in m.groups()))
    elif (m := r_vis.search(l)):
        S[name]["vis"].append((int(m.group(1)), float(m.group(2))))
    elif (m := r_tok.search(l)):
        last_prompt = int(m.group(1))
    elif (m := r_pr.search(l)):
        if last_prompt: S[name]["pr"].append((last_prompt, float(m.group(1))))
        last_prompt = None
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for name, a, b in periods:
    d = S[name]; M = d["mtp"]
    print(f"== {name} ({(b - a) / 3600:.1f} h)")
    if M:
        tok = sum(r[0] for r in M); sec = sum(r[0] / r[1] for r in M); cyc = sum(r[2] for r in M)
        print(f"  single-stream generations n={len(M)} tokens={tok}: decode {tok / sec:.1f} tok/s token-weighted, {tok / cyc:.2f} tok/cycle, "
              f"{1e3 * sec / cyc:.2f} ms/cycle, head {100 * sum(r[3] for r in M) / max(1, sum(r[4] for r in M)):.1f}% accepted, "
              f"copy drafts {100 * sum(r[5] for r in M) / max(1, tok):.1f}% of tokens; per-generation tps p50 {q([r[1] for r in M], .5):.1f}")
    if d["ms"]:
        T = d["ms"]; tok = sum(r[2] for r in T); sec = sum(r[2] / r[3] for r in T if r[3] > 0)
        print(f"  shared (2+ stream) runs n={len(T)} tokens={tok}: {tok / sec:.1f} tok/s total, {sum(r[1] * r[4] for r in T) / sum(r[1] for r in T):.2f} ms/step, drafts {100 * sum(r[5] for r in T) / max(1, sum(r[6] for r in T)):.1f}% accepted")
    for key, label in (("pr", "text prompt reads"), ("vis", "image prompt reads (trunk)")):
        P = d[key]
        if not P: continue
        for lo_, hi_ in ((1, 200), (200, 1000), (1000, 4000), (4000, 30000), (30000, 10**9)):
            sel = [x for x in P if lo_ <= x[0] < hi_]
            if sel:
                print(f"  {label} {lo_:>6}-{hi_ if hi_ < 10**9 else 'max':>6} tokens: n={len(sel):5d} p50 {q([x[1] for x in sel], .5):8.1f} ms  rate {sum(x[0] for x in sel) / max(1e-9, sum(x[1] for x in sel)) * 1e3:7.0f} tok/s")
