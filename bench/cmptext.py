# Compare saved outputs of two replay runs: how many are identical and where the others first differ.
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); same = 0; fr = []
for x, y in zip(a, b):
    if x == y: same += 1; continue
    i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
    fr.append(round(i / max(1, len(x)), 2))
print(f"{same}/{len(a)} identical; first difference at fraction of output: {sorted(fr)}")
