# Analyse an ioreport.py sample log: GPU performance-state use, wake-up ramps, per-die DRAM read rate.
import sys, re, collections
FREQ = [0, 338, 486, 636, 796, 888, 988, 1084, 1182, 1278, 1374, 1470, 1578, 1620]   # MHz, device-tree perf-states (index = P-state)
rows = []
for line in open(sys.argv[1]):
    m = re.match(r"(\d+\.\d+) dt=(\d+\.\d+) (.*)", line)
    if not m: continue
    d = {}
    for part in m.group(3).split(" | "):
        k, _, v = part.partition("{")
        d[k.split(":", 1)[-1] if not k.startswith("DCS BW") and not k.startswith("DCS Floor") and not k.startswith("PMC DCS") else k] = \
            {a.rsplit(":", 1)[0]: float(a.rsplit(":", 1)[1]) for a in v.rstrip("}").split(" ") if ":" in a}
    rows.append((float(m.group(1)), float(m.group(2)), d))
print("samples", len(rows), "span %.0f s" % (rows[-1][0] - rows[0][0]), "median dt %.3f" % sorted(r[1] for r in rows)[len(rows) // 2])
def acc(key, cond=lambda d: True):
    c = collections.Counter(); T = 0
    for ts, dt, d in rows:
        if key in d and cond(d):
            for s, v in d[key].items(): c[s] += v * dt
            T += dt
    return c, T
def show(title, c, T, n=12):
    tot = sum(c.values()) or 1
    print(f"{title} ({T:.0f} s): " + "  ".join(f"{s}:{v / tot:.3f}" for s, v in sorted(c.items(), key=lambda x: -x[1])[:n]))
c, T = acc("GPUPH"); show("GPUPH all", c, T)
c, T = acc("PWRCTRL"); show("PWRCTRL all", c, T)
c, T = acc("SOCxDCS"); show("SOCxDCS all", c, T)
busy = lambda d: d.get("GPUPH", {}).get("P13", 0) >= 0.98
def mean_bw(c):
    tot = sum(c.values()) or 1
    return sum((float(s.replace("GB/s", "")) - 32) * v for s, v in c.items()) / tot
for k in ("DCS BW:AMCC0 RD", "DCS BW:AMCC1 RD", "DCS BW:AMCC0 WR", "DCS BW:AMCC1 WR", "DCS BW:AGX RD"):
    c, T = acc(k, busy)
    if c: show(k + " while P13>=0.98", c, T, 16); print("   mean ~%.0f GB/s (bin centres)" % mean_bw(c) if "AMCC" in k else "")
for k in ("DCS Floor:AGX-DCS-BW", "DCS Floor:DCS0-BW", "DCS Floor:DCS1-BW", "PMC DCS Floor:PMP", "PMC DCS Floor:GFX", "PMC DCS Floor:BOOST"):
    c, T = acc(k, busy)
    if c: show(k + " while busy", c, T)
# wake-up ramps: OFF share >= 0.9 in the previous sample, then active until P13 >= 0.98
ramps = []; i = 1
while i < len(rows):
    off_prev = rows[i - 1][2].get("GPUPH", {}).get("OFF", 0)
    g = rows[i][2].get("GPUPH", {})
    if off_prev >= 0.9 and g.get("OFF", 0) < 0.9:
        j = i; low = 0.0; lost = 0.0; act = 0.0
        while j < len(rows):
            ts, dt, d = rows[j]; g = d.get("GPUPH", {})
            if g.get("P13", 0) >= 0.98 or g.get("OFF", 0) >= 0.98: break
            for s, v in g.items():
                if s.startswith("P"):
                    p = int(s[1:]); act += v * dt
                    if p < 13: low += v * dt; lost += v * dt * (1 - FREQ[p] / FREQ[13])
            j += 1
        reached = j < len(rows) and rows[j][2].get("GPUPH", {}).get("P13", 0) >= 0.98
        ramps.append((rows[i][0], rows[min(j, len(rows) - 1)][0] - rows[i - 1][0], low, lost, reached)); i = j + 1
    else: i += 1
full = [r for r in ramps if r[4]]
print(f"wake-ups: {len(ramps)}, reaching P13: {len(full)}")
if full:
    srt = lambda k: sorted(r[k] for r in full)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    for name, k in (("wall time until steady P13 (s)", 1), ("active time below P13 (s)", 2), ("compute time lost vs P13 (s)", 3)):
        v = srt(k); print(f"  {name}: p10 {q(v, .1):.3f} p50 {q(v, .5):.3f} p90 {q(v, .9):.3f} mean {sum(v) / len(v):.3f}")
short = [r for r in ramps if not r[4]]
if short: print(f"  bursts that never reached P13: {len(short)}, mean active-below-P13 {sum(r[2] for r in short) / len(short):.3f} s")
if len(sys.argv) > 2:
    for r in ramps[:int(sys.argv[2])]: print("   ramp at %.3f wall %.3f low %.3f lost %.3f reached %s" % r)
