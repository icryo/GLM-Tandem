# Scale index feasibility: how many distinct (scale, bias) pairs does each 4-bit expert matrix have?
# A 16-bit index into a per-tensor table of pairs is exact only if a tensor has <= 65,536 distinct pairs.
# CPU only: reads the raw bf16 bytes from the safetensors files with numpy memmap (no MLX, no GPU).
# usage: scalecount.py MODEL_DIR [layers e.g. 3,10,25,44] [experts per layer]
import json, os, sys, struct
import numpy as np
MP = sys.argv[1]
LAYERS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "3,10,25,44").split(",")]
NEXP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
wm = json.load(open(os.path.join(MP, "model.safetensors.index.json")))["weight_map"]
hdr = {}
def header(f):
    if f not in hdr:
        with open(os.path.join(MP, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr[f] = (json.loads(fh.read(n)), 8 + n)
    return hdr[f]
def raw16(name):
    f = wm[name]; h, base = header(f); e = h[name]
    assert e["dtype"] in ("BF16", "F16"), e["dtype"]
    a, b = e["data_offsets"]
    mm = np.memmap(os.path.join(MP, f), dtype=np.uint16, mode="r", offset=base + a, shape=((b - a) // 2,))
    return np.array(mm).reshape(e["shape"])
pre = "model.language_model.layers.{}.mlp.experts.{}.{}.{}"
tot = {"groups": 0, "pairs_le": 0, "mats": 0}
print(f"{'layer':>5} {'proj':>9} {'groups/mat':>10} | distinct pairs per matrix: min  median  max | distinct scales  biases (median) | pairs over all {NEXP} experts")
for L in LAYERS:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        per = []; ns = []; nb = []; allp = []
        for e in range(NEXP):
            s = raw16(pre.format(L, e, proj, "scales")).astype(np.uint32)
            b = raw16(pre.format(L, e, proj, "biases")).astype(np.uint32)
            p = (s << 16) | b
            u = np.unique(p)
            per.append(u.size); ns.append(np.unique(s).size); nb.append(np.unique(b).size); allp.append(u)
            tot["groups"] += p.size; tot["mats"] += 1; tot["pairs_le"] += int(u.size <= 65536)
        un = np.unique(np.concatenate(allp)).size
        print(f"{L:5d} {proj:>9} {s.size:10d} | {min(per):8d} {int(np.median(per)):7d} {max(per):7d} | {int(np.median(ns)):8d} {int(np.median(nb)):7d} | {un}", flush=True)
# Row-block variant: an index per block of R rows (table per block) -> how many pairs per 16 / 64 / 256 rows?
L = LAYERS[0]
s = raw16(pre.format(L, 0, "gate_proj", "scales")).astype(np.uint32); b = raw16(pre.format(L, 0, "gate_proj", "biases")).astype(np.uint32)
p = (s << 16) | b
for R in (16, 64, 256):
    c = [np.unique(p[i:i + R]).size for i in range(0, p.shape[0], R)]
    print(f"layer {L} expert 0 gate_proj: rows per table {R}: groups per block {R * p.shape[1]}, distinct pairs median {int(np.median(c))} max {max(c)}")
print(f"matrices with <= 65,536 distinct pairs: {tot['pairs_le']} of {tot['mats']}")
