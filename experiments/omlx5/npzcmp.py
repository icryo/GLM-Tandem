# npzcmp.py A.npz B.npz: bit-compare arrays of the same name in two saved runs.
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in sorted(a.files):
    if k not in b.files: print(k, "missing in B"); continue
    x, y = a[k], b[k]
    print(f"{k:22s} identical {bool(np.array_equal(x, y))}  max abs diff {float(np.max(np.abs(x - y))):.3e}")
