# Whole MoE block of one real layer (router, routed experts, shared expert), the server's own module class, with the
# server's fused decode kernels for 1-8 rows (oMLX decode_kernels and our moerows), unfused vs fused gate/up layout.
# Prints whether the outputs are bit-identical per row count. Full (unsplit) layer, ~1.8 GB.
import json, os, sys, copy
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import omlx.patches.glm5_next_mlx_lm.model as RM
RM.glm5_next_vlm_runtime.apply()
from omlx.patches import m5_gather_qmm as G
G.apply_m5_gather_qmm_workaround()
from omlx.patches import moe_gate_up_fusion as F
Lm = RM._language
MP = "/Users/a/models/GLM-5.3-Flash-TF-4bit-MTP"; LAYER = int(os.environ.get("LAYER", "10")); E = 288
cfg = json.load(open(MP + "/config.json")); tcd = cfg.get("text_config", cfg)
import importlib
C = importlib.import_module(Lm.__name__.rsplit(".", 1)[0] + ".config")
tc = C.TextConfig.from_dict(tcd) if hasattr(C.TextConfig, "from_dict") else C.TextConfig(**{k: v for k, v in tcd.items() if k in C.TextConfig.__dataclass_fields__})
moe = Lm.Glm5NextMoE(tc)
idx = json.load(open(MP + "/model.safetensors.index.json"))["weight_map"]
pre = f"model.language_model.layers.{LAYER}.mlp."
files = {}
def get(name):
    f = idx[name]
    if f not in files: files[f] = mx.load(os.path.join(MP, f))
    return files[f][name]
nn.quantize(moe, group_size=64, bits=4, class_predicate=lambda p, m: hasattr(m, "to_quantized") and p.split(".")[-1] != "gate")
for proj in ("gate_proj", "up_proj", "down_proj"):
    m = getattr(moe.switch_mlp, proj)
    for field in ("weight", "scales", "biases"):
        setattr(m, field, mx.stack([get(f"{pre}experts.{e}.{proj}.{field}") for e in range(E)]))
    sm = getattr(moe.shared_experts, proj)
    for field in ("weight", "scales", "biases"):
        setattr(sm, field, get(f"{pre}shared_experts.{proj}.{field}"))
moe.gate.weight = get(pre + "gate.weight")
if hasattr(moe.gate, "e_score_correction_bias"): moe.gate.e_score_correction_bias = get(pre + "gate.e_score_correction_bias")
mx.eval(moe.parameters()); files.clear()
print("module", type(moe).__name__, "switch", tuple(moe.switch_mlp.gate_proj.weight.shape), "shared", tuple(moe.shared_experts.gate_proj.weight.shape))
import moerows
class _L:   # moerows.install expects decoder layers with .mlp
    def __init__(self, mlp): self.mlp = mlp; self.is_linear = False
inst = os.environ.get("MOEROWS", "1") == "1"
fused = copy.deepcopy(moe)
if inst:
    print("moerows:", moerows.install([type("Layer", (), {})()]) if False else "skipped (module-level hook test below)")
F._fuse_one(fused.switch_mlp); mx.eval(fused.parameters())
rng = np.random.default_rng(1)
for T in (1, 2, 3, 4, 8, 26, 300):
    x = mx.array((rng.standard_normal((1, T, 4096)) * 0.35).astype(np.float32)).astype(mx.bfloat16)
    a = moe(x); b = fused(x); mx.eval(a, b)
    print(f"T={T:4d} identical {bool(mx.array_equal(a, b).item())} max abs diff {float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()):.3e}", flush=True)
