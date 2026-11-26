# n26cmp.py A.json B.json : outputs of the same fixed T=0 sequence from two builds (n26set.py SAVE=...).
# Identical, or how far the two texts agree before they part (two correct builds that round differently agree for
# a long stretch and then diverge at a near-tie; a broken one diverges at once or produces rubbish).
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = 0; rows = []
for k in a:
    if k not in b: rows.append((k, "missing in second")); continue
    x, y = a[k], b[k]
    if x == y: same += 1; continue
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]: n += 1
    rows.append((k, f"agree for {n} of {len(x)} / {len(y)} characters ({100 * n / max(1, min(len(x), len(y))):.0f}%)"))
print(f"{same} of {len(a)} outputs identical")
for k, r in rows: print(f"   {k}: {r}")
