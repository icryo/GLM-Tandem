# Compare two KPROF dumps per unit (cycle / step): per-kernel GPU time, grouped.
import re, sys, collections
def load(p):
    d = {}
    for l in open(p):
        m = re.match(r"KPROF\s+([\d.]+) ms\s+(\d+) calls\s+([\d.]+) us/call\s+(.*)", l)
        if m: d[m.group(4).strip()] = (float(m.group(1)), int(m.group(2)))
    return d
def group(name):
    n = name.lower()
    for key, g in (("expanddims", "exchange wait (hop)"), ("moe_gate_up", "MoE gate/up"), ("moe_down", "MoE down"), ("router", "MoE router"),
                   ("indexer", "index scores"), ("tp_idx", "index scores"), ("topk", "top-k"), ("select", "top-k"), ("decode_blocks", "sparse attn"), ("sparse", "sparse attn"),
                   ("sdpa", "attention (sdpa)"), ("scaled_dot", "attention (sdpa)"), ("kda", "KDA"), ("gdn", "KDA"), ("hc_", "HC"), ("hcrows", "HC"),
                   ("quantizedmatmul", "projections (qmm)"), ("qmv", "projections (qmm)"), ("gather", "gather"), ("sliceupdate", "slice update"), ("slice", "slice"),
                   ("concatenate", "concatenate"), ("copy", "copy"), ("roll", "roll"), ("where", "select/where"), ("matmul", "matmul"), ("rmsnorm", "norms"),
                   ("layernorm", "norms"), ("softmax", "softmax"), ("argmax", "sampling"), ("arg", "sampling")):
        if key in n: return g
    return "other: " + name.split("_")[0][:28]
A, B = load(sys.argv[1]), load(sys.argv[2]); ua, ub = float(sys.argv[3]), float(sys.argv[4])
ga, gb = collections.defaultdict(float), collections.defaultdict(float)
for k, (ms, c) in A.items(): ga[group(k)] += ms / ua
for k, (ms, c) in B.items(): gb[group(k)] += ms / ub
keys = sorted(set(ga) | set(gb), key=lambda k: -(gb.get(k, 0) - ga.get(k, 0)))
print(f"{'group':34s} {'single ms/cycle':>16s} {'pair ms/step':>13s} {'difference':>11s}")
tot_a = tot_b = 0
for k in keys:
    a, b = ga.get(k, 0), gb.get(k, 0); tot_a += a * (k != "exchange wait (hop)"); tot_b += b * (k != "exchange wait (hop)")
    if max(a, b) >= 0.05: print(f"{k:34s} {a:16.3f} {b:13.3f} {b - a:+11.3f}")
print(f"{'TOTAL without exchange waits':34s} {tot_a:16.3f} {tot_b:13.3f} {tot_b - tot_a:+11.3f}")
print("\ntop kernels by growth (ms per unit):")
names = sorted(set(A) | set(B), key=lambda k: -(B.get(k, (0, 0))[0] / ub - A.get(k, (0, 0))[0] / ua))
for k in names[:18]:
    a = A.get(k, (0, 0)); b = B.get(k, (0, 0))
    print(f"  {b[0] / ub - a[0] / ua:+7.3f}  single {a[0] / ua:6.3f} ({a[1] / ua:5.1f} calls)  pair {b[0] / ub:6.3f} ({b[1] / ub:5.1f} calls)  {k[:110]}")
