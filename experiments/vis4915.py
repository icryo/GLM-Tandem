# Does MLX #4915 (wrong order after reshape/transpose of a dense array with all flags false, Metal only,
# since 0.32.4; trigger: Conv3d with in_channels % 16 != 0 and out_channels % 16 == 0) affect our
# vision tower? Run the real GLM-5.3-Flash vision tower (vendored oMLX glm5_next) on random patches on
# the GPU and on the CPU stream (no trigger there) and compare. Also the PR's own repro.
import json, os, sys
import numpy as np
import mlx.core as mx
import importlib
import omlx.patches.glm5_next_mlx_lm.model as RM
RM.glm5_next_vlm_runtime.apply()
_pkg = RM._language.__name__.rsplit(".", 1)[0]
V = importlib.import_module(_pkg + ".vision"); C = importlib.import_module(_pkg + ".config")
print("vision module:", V.__file__)

print("mlx", mx.__version__)
# 1) the PR's repro
y = mx.conv_general(mx.random.normal((2, 12, 15, 18, 2), key=mx.random.key(1)), mx.random.normal((240, 4, 5, 6, 2), key=mx.random.key(2)), stride=(4, 5, 6))
g = y.reshape(-1, 240).T.reshape(-1)
with mx.stream(mx.cpu):
    yc = mx.conv_general(mx.random.normal((2, 12, 15, 18, 2), key=mx.random.key(1)), mx.random.normal((240, 4, 5, 6, 2), key=mx.random.key(2)), stride=(4, 5, 6))
    c = yc.reshape(-1, 240).T.reshape(-1)
print("PR repro: max |gpu - cpu| =", float(mx.max(mx.abs(g - c)).item()))

# 2) our tower
MP = "/Users/a/models/GLM-5.3-Flash-TF-4bit-MTP"
cfg = json.load(open(os.path.join(MP, "config.json")))
vc = cfg["vision_config"]
VC = C.VisionConfig.from_dict(vc) if hasattr(C.VisionConfig, "from_dict") else C.VisionConfig(**{k: v for k, v in vc.items() if k in C.VisionConfig.__dataclass_fields__})
model = V.VisionModel(VC)
wm = json.load(open(os.path.join(MP, "model.safetensors.index.json")))["weight_map"]
pref = next(p for p in ("model.visual.", "visual.", "model.vision_tower.", "vision_tower.") if any(k.startswith(p) for k in wm))
files = sorted({f for k, f in wm.items() if k.startswith(pref)})
W = {}
for f in files:
    d = mx.load(os.path.join(MP, f))
    W.update({k[len(pref):]: v for k, v in d.items() if k.startswith(pref)})
W = model.sanitize(W) if hasattr(model, "sanitize") else W
DT = os.environ.get("DT", "bfloat16")
if DT != "bfloat16":
    W = {k: (v.astype(getattr(mx, DT)) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in W.items()}
model.load_weights(list(W.items()), strict=False)
mx.eval(model.parameters())
print("vision weights:", len(W), "tensors from", pref, "dtype", model.patch_embed.proj.weight.dtype)
P = VC.patch_size; T = VC.temporal_patch_size; Cin = VC.in_channels
grid = mx.array([[1, 32, 24]])          # one image, 32 x 24 patches
n = 32 * 24
rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((n, Cin * T * P * P)).astype(np.float32)).astype(model.patch_embed.proj.weight.dtype)
out_g = model(x, grid); mx.eval(out_g)
with mx.stream(mx.cpu):
    out_c = model(x, grid); mx.eval(out_c)
pe_g = model.patch_embed(x)
with mx.stream(mx.cpu):
    pe_c = model.patch_embed(x)
mx.eval(pe_g, pe_c)
f = lambda a: a.astype(mx.float32)
rel = lambda a, b: float((mx.sqrt(mx.mean((f(a) - f(b)) ** 2)) / mx.sqrt(mx.mean(f(b) ** 2))).item())
print(f"patch embed: max |gpu - cpu| {float(mx.max(mx.abs(f(pe_g) - f(pe_c))).item()):.4g}, rel RMS {rel(pe_g, pe_c):.3e}")
print(f"tower output {tuple(out_g.shape)}: max |gpu - cpu| {float(mx.max(mx.abs(f(out_g) - f(out_c))).item()):.4g}, rel RMS {rel(out_g, out_c):.3e}")
# cosine per output row: a reordering bug gives rows near 0
cos = mx.sum(f(out_g) * f(out_c), axis=-1) / (mx.linalg.norm(f(out_g), axis=-1) * mx.linalg.norm(f(out_c), axis=-1) + 1e-9)
print(f"row cosine gpu vs cpu: min {float(mx.min(cos).item()):.5f} mean {float(mx.mean(cos).item()):.5f}")
