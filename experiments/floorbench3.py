# Gate and up as two tile-kernel calls against one call over [gate; up] stacked along the output rows
# (what oMLX's gate/up fusion does, without its activation epilogue). Synthetic weights, per-Mac shapes.
import os, time
import mlx.core as mx
from omlx.patches import m5_gather_qmm as G
from omlx.patches import m5_gather_qmm_nax as NAX
G.apply_m5_gather_qmm_workaround()
E, K, NH = 288, 4096, 1024
mx.random.seed(2)
def qstack(n, k):
    ws, ss, bs = [], [], []
    for i in range(0, E, 24):
        w = (mx.random.normal((24, n, k)) * 0.02).astype(mx.float32).reshape(24 * n, k)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, sc, bi)
        ws.append(wq.reshape(24, n, -1)); ss.append(sc.reshape(24, n, -1).astype(mx.bfloat16)); bs.append(bi.reshape(24, n, -1).astype(mx.bfloat16))
    out = [mx.concatenate(a, axis=0) for a in (ws, ss, bs)]; mx.eval(out); mx.clear_cache(); return out
gate = qstack(NH, K); up = qstack(NH, K)
both = [mx.concatenate([a, b], axis=1) for a, b in zip(gate, up)]; mx.eval(both)
def chain(fn, x, n):
    prev = None
    for _ in range(n): prev = fn(x if prev is None else mx.depends(x, [prev]))
    return prev
def bench(fn, x, n=6, reps=4):
    mx.eval(chain(fn, x, n)); best = 1e9
    for _ in range(reps):
        o = chain(fn, x, n); t = time.perf_counter(); mx.eval(o); best = min(best, (time.perf_counter() - t) / n)
    return best * 1e3
def tiles(x, t, idx): return NAX.sorted_gather_qmm(x, t[0], t[1], t[2], idx, group_size=64, bits=4, mode="affine", stream=None)
print(f"{'tokens':>6} | gate + up separately ms | stacked ms | saved per layer ms | x42 ms | same bits")
for T in (64, 144, 288, 600, 1024, 2048):
    r = mx.random.uniform(shape=(T, E)); top = mx.argpartition(-r, kth=7, axis=-1)[:, :8].reshape(-1)
    order = mx.argsort(top); idx = top[order].astype(mx.uint32); row_map = order // 8
    x = mx.random.normal((T, K)).astype(mx.bfloat16); xr = mx.contiguous(x[row_map][:, None, :]); mx.eval(idx, xr)
    def sep(xx): return mx.concatenate([tiles(xx, gate, idx), tiles(xx, up, idx)], axis=-1)
    a = sep(xr); b = tiles(xr, both, idx); mx.eval(a, b)
    same = bool(mx.array_equal(a.astype(mx.float32), b.astype(mx.float32)).item())
    t2 = bench(lambda xx: tiles(mx.depends(xx, [tiles(xx, gate, idx)]), up, idx), xr); t1 = bench(lambda xx: tiles(xx, both, idx), xr)
    print(f"{T:6d} | {t2:23.3f} | {t1:10.3f} | {t2 - t1:18.3f} | {42 * (t2 - t1):6.1f} | {same}", flush=True)
