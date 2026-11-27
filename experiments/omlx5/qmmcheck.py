# Does oMLX 0.8.0rc1's bundled quant_matmul binary (built for MLX 0.32.4) still compute the same thing when
# loaded into MLX 0.32.5? Prompt-path projections of >= 128 rows use its affine_q4_qmm_t (glm5_next linear.py).
# Deterministic inputs; saves outputs to OUT (.npz) so a run under pyx (0.32.4) and one under pyx5 (0.32.5)
# can be compared bit for bit; also times the call and stock mx.quantized_matmul. A few hundred MB at most.
import os, sys, time
import numpy as np
import mlx.core as mx
from omlx.custom_kernels.quant_matmul import fast

OUT = sys.argv[1]
print("mlx", mx.__version__, "| affine_q4_qmm_t:", fast.has_symbol("affine_q4_qmm_t"), flush=True)


def per_call_us(make, n=20):
    mx.eval(make()); mx.synchronize()
    best = 1e9
    for _ in range(3):
        outs = [make() for _ in range(n)]
        t = time.perf_counter(); mx.eval(outs); mx.synchronize()
        best = min(best, (time.perf_counter() - t) / n * 1e6)
    return best


res = {}
for N, K in ((4096, 4096), (12544, 4096)):
    mx.random.seed(N)
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    for R in (128, 512, 2048):
        x = (mx.random.normal((R, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(wq, sc, bi, x)
        ext = lambda: fast.affine_q4_qmm_t(x, wq, sc, bi, 8, 64)
        stock = lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4)
        a, b = ext(), stock(); mx.eval(a, b)
        res[f"ext_{N}_{R}"] = np.array(a.astype(mx.float32))
        res[f"stock_{N}_{R}"] = np.array(b.astype(mx.float32))
        d = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())
        print(f"N={N:5d} K={K} rows={R:4d}: ext {per_call_us(ext):8.1f} us  stock {per_call_us(stock):8.1f} us  "
              f"ext==stock {bool(mx.array_equal(a, b).item())} (max diff {d:.2e})", flush=True)
np.savez(OUT, **res)
print("saved", OUT)
