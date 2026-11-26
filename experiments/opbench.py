# Operator timings for comparing two MLX builds on this model's shapes: 4-bit group-64 quantized matmul of B rows
# against the linear-attention input projection (12576 x 4096 per Mac) and output projection (4096 x 4096).
# Small: ~40 MB of weights, a few seconds of GPU. Prints microseconds per call (best of several chained runs).
import os, sys, time
import mlx.core as mx
print("mlx", mx.__version__, "| build:", os.path.dirname(mx.__file__))
def chain(fn, x, n):
    prev = None
    for _ in range(n): prev = fn(x if prev is None else mx.depends(x, [prev]))
    return prev
def bench(fn, x, n, reps=5):
    mx.eval(chain(fn, x, max(20, n))); best = 1e9
    for _ in range(reps):
        o = chain(fn, x, n); t = time.perf_counter(); mx.eval(o); best = min(best, (time.perf_counter() - t) / n)
    return best * 1e6
mx.random.seed(1)
ROWS = [int(v) for v in os.environ.get("ROWS", "1,2,3,4,6,8,12,16,32,64,128,512,2048").split(",")]
print(f"{'rows':>5} " + " ".join(f"{name:>14}" for name in ("in-proj us", "out-proj us")) + "   checksum")
shapes = {"in": (12576, 4096), "out": (4096, 4096)}
W = {}
for k, (n, kk) in shapes.items():
    w = (mx.random.normal((n, kk)) * 0.02).astype(mx.float32)
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, sc, bi); W[k] = (wq, sc.astype(mx.bfloat16), bi.astype(mx.bfloat16))
for B in ROWS:
    x = mx.random.normal((B, 4096)).astype(mx.bfloat16); mx.eval(x); ts = []; cs = 0.0
    for k in ("in", "out"):
        wq, sc, bi = W[k]
        fn = lambda xx, wq=wq, sc=sc, bi=bi: mx.quantized_matmul(xx, wq, sc, bi, transpose=True, group_size=64, bits=4)
        ts.append(bench(fn, x, 60 if B <= 128 else 12)); cs += float(fn(x).astype(mx.float32).abs().sum().item())
    print(f"{B:5d} " + " ".join(f"{t:14.1f}" for t in ts) + f"   {cs:.6g}", flush=True)
