# MLX patch

`mlx_tp_patches.diff` applies to MLX v0.32.4 (commit `36969c0`).

```sh
git clone https://github.com/ml-explore/mlx && cd mlx && git checkout 36969c0
git apply /path/to/mlx_tp_patches.diff
# in a Python 3.11 environment with MLX's build requirements installed:
python setup.py build_ext --inplace
```

Build on one Mac and copy `python/mlx/` to the same path on the other. `scripts/pyx` puts the build first on the path.

Everything is off unless an environment variable turns it on:

| Variable | Effect |
|---|---|
| `MLX_FENCE_MODE=1` | The CPU thread waits for the GPU by spinning inline on the fence instead of sleeping on an event. **This is the one the recipe uses.** |
| `MLX_FENCE_MODE=2` (bit 2) | Skips the coherence kernel. **Corrupts output. Do not use**; kept only to document the result. |
| `MLX_FENCE_MODE=4` (bit 4) | Merged coherence and update kernel. Correct, slower. |
| `MLX_KPROF=1` | Per-kernel GPU time (one command buffer per primitive; for profiling only). |
| `MLX_KGAP=1` | Records GPU idle gaps between command buffers. |
