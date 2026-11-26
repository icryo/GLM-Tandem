# Is the fused gate/up expert path less accurate than the separate one? One real expert layer (layer L of the
# checkpoint, half of the hidden rows like one Mac holds), the server's own SwitchGLU class, unfused vs fused, against
# an exact float64 reference computed on the CPU from the dequantized weights. Inputs: real-scale synthetic hidden
# states. Shapes: a prompt block (T=2048, 600), a small read (T=26, 100) and verify rows (T=1..3).
import json, os, sys, copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import omlx.patches.glm5_next_mlx_lm.model as RM
RM.glm5_next_vlm_runtime.apply()
from omlx.patches import m5_gather_qmm as G
print("reroute:", G.apply_m5_gather_qmm_workaround())
from omlx.patches.deepseek_v4 import switch_layers as SL
from omlx.patches import moe_gate_up_fusion as F
Lm = RM._language
MP = "/Users/a/models/GLM-5.3-Flash-TF-4bit-MTP"; LAYER = int(os.environ.get("LAYER", "10")); E = 288; D = 4096; H = 2048; HALF = 1024
idx = json.load(open(MP + "/model.safetensors.index.json"))["weight_map"]
files = {}
def get(name):
    f = idx[name]
    if f not in files: files[f] = mx.load(os.path.join(MP, f))
    return files[f][name]
def stack(proj, field, rows=None):
    arrs = []
    for e in range(E):
        a = get(f"model.language_model.layers.{LAYER}.mlp.experts.{e}.{proj}.{field}")
        if rows is not None: a = a[rows[0]:rows[1]] if proj != "down_proj" else a[:, (rows[0] // 64) * (1 if field != "weight" else 8):(rows[1] // 64) * (1 if field != "weight" else 8)]
        arrs.append(a)
    out = mx.stack(arrs); mx.eval(out); return out
# rank 0's half: gate/up rows 0..1023, down columns 0..1023 (weight packs 8 codes per uint32 -> 1024/8 words; scales 1024/64 groups)
sw = SL.SwitchGLU(D, HALF, E, activation=Lm.Glm5NextClampedSwiGLU(10.0))
nn.quantize(sw, group_size=64, bits=4)
for proj in ("gate_proj", "up_proj", "down_proj"):
    m = getattr(sw, proj)
    for field in ("weight", "scales", "biases"):
        setattr(m, field, stack(proj, field, (0, HALF)))
mx.eval(sw.parameters()); files.clear()
print("layer", LAYER, {p: tuple(getattr(sw, p).weight.shape) for p in ("gate_proj", "up_proj", "down_proj")})
def deq(m):
    return np.array(mx.dequantize(m.weight, m.scales, m.biases, group_size=64, bits=4).astype(mx.float32), dtype=np.float64)
Wg, Wu, Wd = deq(sw.gate_proj), deq(sw.up_proj), deq(sw.down_proj)
fused = copy.deepcopy(sw)
F._fuse_one(fused); mx.eval(fused.parameters())
print("fused has gate_up_proj:", "gate_up_proj" in fused)
def ref(x, inds):
    xs = np.array(x.astype(mx.float32), dtype=np.float64).reshape(-1, D); ii = np.array(inds).reshape(xs.shape[0], -1)
    out = np.zeros((xs.shape[0], ii.shape[1], D))
    for t in range(xs.shape[0]):
        for j, e in enumerate(ii[t]):
            g = Wg[e] @ xs[t]; u = Wu[e] @ xs[t]
            g = np.minimum(g, 10.0); u = np.clip(u, -10.0, 10.0)
            h = g / (1 + np.exp(-g)) * u
            out[t, j] = Wd[e] @ h
    return out
rng = np.random.default_rng(LAYER)
print(f"{'tokens':>6} | rel. RMS error vs float64: separate  fused  | max abs diff separate vs fused | identical")
for T in [int(v) for v in os.environ.get("TS", "1,2,3,26,100,600,2048").split(",")]:
    x = mx.array((rng.standard_normal((1, T, D)) * 0.35).astype(np.float32)).astype(mx.bfloat16)
    inds = mx.array(np.stack([rng.choice(E, 8, replace=False) for _ in range(T)])[None].astype(np.uint32))
    a = sw(x, inds).astype(mx.float32); b = fused(x, inds).astype(mx.float32); mx.eval(a, b)
    nsub = min(T, 48); sel = np.sort(rng.choice(T, nsub, replace=False))
    ms = mx.array(sel.astype(np.int32)); r = ref(mx.take(x, ms, axis=1), mx.take(inds, ms, axis=1))
    A = np.array(a, dtype=np.float64)[0, sel]; B = np.array(b, dtype=np.float64)[0, sel]
    rms = lambda z: float(np.sqrt(np.mean(z ** 2)))
    ea, eb = rms(A - r) / rms(r), rms(B - r) / rms(r)
    same = bool(mx.array_equal(a, b).item())
    print(f"{T:6d} | {ea:27.3e} {eb:6.3e} | {float(mx.abs(a - b).max().item()):30.3e} | {same}", flush=True)
