# What does importing a module change in modules that were already loaded? Snapshot module attributes and class
# attributes (by identity) of mlx / mlx_lm / mlx_vlm / omlx modules, import the target, print what changed.
# Also lists environment variables and MLX global settings that changed.
import importlib, os, sys, types
TARGET = sys.argv[1] if len(sys.argv) > 1 else "omlx.scheduler"
import mlx.core as mx
import omlx.patches.glm5_next_mlx_lm.model as RM
RM.glm5_next_vlm_runtime.apply()
import mlx_lm.server, mlx_lm.generate, mlx_lm.models.cache
from omlx.cluster import progressive_loading, tensor_strategies
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from omlx.patches import m5_gather_qmm
PREF = ("mlx", "mlx_lm", "mlx_vlm", "omlx")
def snap():
    out = {}
    for name, m in list(sys.modules.items()):
        if not isinstance(m, types.ModuleType) or not name.split(".")[0] in PREF: continue
        for k, v in list(vars(m).items()):
            if k.startswith("__"): continue
            out[(name, k)] = id(v)
            if isinstance(v, type) and getattr(v, "__module__", "") == name:
                for ck, cv in list(vars(v).items()):
                    if not ck.startswith("__") or ck in ("__call__", "__init__"): out[(name, k + "." + ck)] = id(cv)
    return out
env0 = dict(os.environ); a = snap(); mods0 = set(sys.modules)
importlib.import_module(TARGET)
b = snap()
changed = sorted(k for k in a if k in b and a[k] != b[k])
print(f"import {TARGET}: {len(set(sys.modules) - mods0)} new modules; {len(changed)} attributes of earlier modules replaced")
for k in changed: print("   REPLACED", ".".join(k))
for k, v in os.environ.items():
    if env0.get(k) != v: print("   ENV", k, "=", v)
for k in env0:
    if k not in os.environ: print("   ENV removed", k)
