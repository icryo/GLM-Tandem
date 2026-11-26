# Where does the floor of a prompt read come from? One expert layer on one Mac with synthetic 4-bit group-64 weights
# of the real shapes (288 experts; gate and up 1024 x 4096 each per Mac, down 4096 x 1024), routed like a prompt
# block of T tokens (8 experts per token, rows sorted by expert), through the same gather_qmm call the prompt path
# makes, with oMLX's reroute to its tensor-unit kernel applied. Prints ms per call and the projected cost of 42
# layers. ~2 GB of memory, a few seconds of GPU; run on an idle Mac.
import os, sys, time
import mlx.core as mx
REROUTE = os.environ.get("REROUTE", "1") == "1"
if REROUTE:
    try:
        from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround
        print("oMLX reroute applied:", apply_m5_gather_qmm_workaround())
    except Exception as e:
        print("oMLX reroute not available:", repr(e))
print("mlx", mx.__version__, os.path.dirname(mx.__file__))
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
gb = lambda t: sum(a.nbytes for a in t) / 1e9
print(f"gate (= up) {gb(gate):.2f} GB, down {gb(down):.2f} GB per layer per Mac")
def chain(fn, x, n):
    prev = None
    for _ in range(n): prev = fn(x if prev is None else mx.depends(x, [prev]))
    return prev
def bench(fn, x, n=6, reps=4):
    mx.eval(chain(fn, x, n)); best = 1e9
    for _ in range(reps):
        o = chain(fn, x, n); t = time.perf_counter(); mx.eval(o); best = min(best, (time.perf_counter() - t) / n)
    return best * 1e3
def call(x, t, idx):
    return mx.gather_qmm(x, t[0], t[1], t[2], rhs_indices=idx, transpose=True, group_size=64, bits=4, sorted_indices=True)
print(f"{'tokens':>6} {'rows':>6} {'gate ms':>8} {'down ms':>8} {'layer ms':>9} {'42 layers ms':>12} {'ms/token':>9} {'experts touched':>15}")
for T in [int(v) for v in os.environ.get("TS", "16,32,64,100,128,144,200,288,400,600,1024,2048").split(",")]:
    r = mx.random.uniform(shape=(T, E)); top = mx.argpartition(-r, kth=7, axis=-1)[:, :8].reshape(-1)
    order = mx.argsort(top); idx = top[order].astype(mx.uint32); row_map = order // 8
    x = mx.random.normal((T, K)).astype(mx.bfloat16); xr = mx.contiguous(x[row_map][:, None, :])
    h = mx.random.normal((T * 8, 1, NH)).astype(mx.bfloat16); mx.eval(idx, xr, h)
    touched = len(set(idx.tolist()))
    tg = bench(lambda xx: call(xx, gate, idx), xr); td = bench(lambda hh: call(hh, down, idx), h)
    layer = 2 * tg + td
    print(f"{T:6d} {T * 8:6d} {tg:8.3f} {td:8.3f} {layer:9.3f} {42 * layer:12.1f} {42 * layer / T:9.3f} {touched:15d}", flush=True)
