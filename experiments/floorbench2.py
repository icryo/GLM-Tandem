# Below 144 tokens (fewer than 4 rows per expert) the prompt path runs one small product per (row, expert); from 144
# it runs oMLX's tensor-unit tile kernel over rows sorted by expert. Here: both paths for the same routes at
# T = 16..200, time per call and how far their results differ. Synthetic weights of the real per-Mac shapes.
import os, sys, time
import mlx.core as mx
from omlx.patches import m5_gather_qmm as G
from omlx.patches import m5_gather_qmm_nax as NAX
print("mlx", mx.__version__, "| reroute applied:", G.apply_m5_gather_qmm_workaround())
E, K, NH = 288, 4096, 1024
mx.random.seed(2)
def qstack(n, k):
    ws, ss, bs = [], [], []
    for i in range(0, E, 24):
        w = (mx.random.normal((24, n, k)) * 0.02).astype(mx.float32).reshape(24 * n, k)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, sc, bi)
        ws.append(wq.reshape(24, n, -1)); ss.append(sc.reshape(24, n, -1).astype(mx.bfloat16)); bs.append(bi.reshape(24, n, -1).astype(mx.bfloat16))
    out = [mx.concatenate(a, axis=0) for a in (ws, ss, bs)]; mx.eval(out); mx.clear_cache(); return out
gate = qstack(NH, K); down = qstack(K, NH)
def chain(fn, x, n):
    prev = None
    for _ in range(n): prev = fn(x if prev is None else mx.depends(x, [prev]))
    return prev
def bench(fn, x, n=6, reps=4):
    mx.eval(chain(fn, x, n)); best = 1e9
    for _ in range(reps):
        o = chain(fn, x, n); t = time.perf_counter(); mx.eval(o); best = min(best, (time.perf_counter() - t) / n)
    return best * 1e3
def stock(x, t, idx):      # what the server does today (per-row products below 4 rows per expert)
    return mx.gather_qmm(x, t[0], t[1], t[2], rhs_indices=idx, transpose=True, group_size=64, bits=4, sorted_indices=True)
def tiles(x, t, idx):      # the tile kernel regardless of rows per expert
    return NAX.sorted_gather_qmm(x, t[0], t[1], t[2], idx, group_size=64, bits=4, mode="affine", stream=None)
print(f"{'tokens':>6} {'rows/expert':>11} | {'today: gate':>11} {'down':>7} {'42 layers':>10} | {'tiles: gate':>11} {'down':>7} {'42 layers':>10} | max rel diff gate, down")
for T in [int(v) for v in os.environ.get("TS", "16,24,32,48,64,80,100,128,144,200").split(",")]:
    r = mx.random.uniform(shape=(T, E)); top = mx.argpartition(-r, kth=7, axis=-1)[:, :8].reshape(-1)
    order = mx.argsort(top); idx = top[order].astype(mx.uint32); row_map = order // 8
    x = mx.random.normal((T, K)).astype(mx.bfloat16); xr = mx.contiguous(x[row_map][:, None, :])
    h = mx.random.normal((T * 8, 1, NH)).astype(mx.bfloat16); mx.eval(idx, xr, h)
    res = []
    for fn in (stock, tiles):
        try:
            a = fn(xr, gate, idx); b = fn(h, down, idx); mx.eval(a, b)
            res.append((bench(lambda xx: fn(xx, gate, idx), xr), bench(lambda hh: fn(hh, down, idx), h), a.astype(mx.float32), b.astype(mx.float32)))
        except Exception as e:
            res.append(None); print("   ", fn.__name__, "failed:", repr(e)[:160])
    if res[0] is None or res[1] is None: continue
    (g0, d0, a0, b0), (g1, d1, a1, b1) = res
    rel = lambda p, q: float((mx.abs(p - q).max() / mx.abs(p).max()).item())
    print(f"{T:6d} {T * 8 / E:11.2f} | {g0:11.3f} {d0:7.3f} {42 * (2 * g0 + d0):10.1f} | {g1:11.3f} {d1:7.3f} {42 * (2 * g1 + d1):10.1f} | {rel(a0, a1):.2e} {rel(b0, b1):.2e}", flush=True)
