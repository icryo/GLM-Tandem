# Exact, faster index scoring for decode and verify rows of the sparse-attention layers.
#
# The vendored kernels (small_indexer.py) launch one 32-thread threadgroup per pooled key and query row. They are
# compute-bound: every key reloads and converts the row's query values. Here one simdgroup walks KB keys, scores U
# keys per step from one load of the query values, and handles up to RG query rows together so the key values are
# loaded once for those rows. Per (row, key) the arithmetic is the vendored one, in the same order: one head per
# lane, fp32 dot in groups of four terms, relu * weight, simd_sum. Outputs are bit-identical (python idxrows.py
# checks that against the vendored sources on random inputs).
#
# install(layers) replaces small_scores / small_scores_batched in the vendored modules. It refuses to install when
# the vendored kernel text is not the one this file was derived from.
import hashlib, os, sys
import mlx.core as mx

KB = int(os.environ.get("TP_IDX_KB", "32"))
STATS = {"single": 0, "batched": 0, "fallback": 0}
_REF_MD5 = {"single": "", "batched": ""}          # filled by _ref_md5() below from the embedded reference text

REF_SINGLE = """
    uint p = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;   // heads lane, lane+32, ...
    int P = pool_len[0];
    int qpos = before[0] + (int)s;
    float out = -1e30f;  // same fill as the general path mx.where(valid, scores, -1e30)
    if ((int)p < P && ((int)p + 1) * KPOOL - 1 <= qpos) {
        const device T* kp = k + p * HD;
        float v = 0.0f;
        for (uint h = lane; h < (uint)NH; h += 32) {
            const device T* qh = q + ((s * NH + h) * HD);
            float acc = 0.0f;
            for (int d = 0; d < HD; d += 4) {
                acc += float(qh[d]) * float(kp[d]) + float(qh[d + 1]) * float(kp[d + 1])
                     + float(qh[d + 2]) * float(kp[d + 2]) + float(qh[d + 3]) * float(kp[d + 3]);
            }
            v += metal::max(acc, 0.0f) * float(w[s * NH + h]);
        }
        out = simd_sum(v);
    }
    if (lane == 0) {
        scores[s * (uint)P + p] = static_cast<T>(out);
    }
"""
REF_BATCHED = """
    uint p = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint b = threadgroup_position_in_grid.z;
    uint lane = thread_position_in_threadgroup.x;   // heads lane, lane+32, ...
    int PX = dims[0];                // pooled extent (score row stride)
    int S = dims[1];
    int P = rows[b * 2];             // this row's valid pooled windows
    int qpos = rows[b * 2 + 1] + (int)s;
    uint r = b * (uint)S + s;
    float out = -1e30f;  // same fill as the general path mx.where(valid, scores, -1e30)
    if ((int)p < P && ((int)p + 1) * KPOOL - 1 <= qpos) {
        const device T* kp = k + (size_t)b * (size_t)dims[2] + p * HD;
        float v = 0.0f;
        for (uint h = lane; h < (uint)NH; h += 32) {
            const device T* qh = q + ((r * NH + h) * HD);
            float acc = 0.0f;
            for (int d = 0; d < HD; d += 4) {
                acc += float(qh[d]) * float(kp[d]) + float(qh[d + 1]) * float(kp[d + 1])
                     + float(qh[d + 2]) * float(kp[d + 2]) + float(qh[d + 3]) * float(kp[d + 3]);
            }
            v += metal::max(acc, 0.0f) * float(w[r * NH + h]);
        }
        out = simd_sum(v);
    }
    if (lane == 0) {
        scores[r * (uint)PX + p] = static_cast<T>(out);
    }
"""
def _md5(s): return hashlib.md5(s.encode()).hexdigest()

def _gen(U, RG, batched):
    """Kernel text: U keys per step, RG query rows per simdgroup. Every condition is uniform across a simdgroup."""
    L = []; A = L.append
    A("uint blk = threadgroup_position_in_grid.x;")
    A("int g = (int)threadgroup_position_in_grid.y;")
    A("uint lane = thread_position_in_threadgroup.x;")
    if batched:
        A("uint b = threadgroup_position_in_grid.z;")
        A("int PX = dims[0]; int S = dims[1];")
        A("int P = rows[b * 2]; int q0pos = rows[b * 2 + 1];")
        A("const device T* kb = k + (size_t)b * (size_t)dims[2];")
        A("uint rbase = b * (uint)S;")
    else:
        A("int PX = pool_len[0]; int S = pool_len[1]; int P = PX; int q0pos = before[0];")
        A("const device T* kb = k; uint rbase = 0;")
    for r in range(RG):
        A(f"bool live{r} = g * {RG} + {r} < S;")
        A(f"int s{r} = min(g * {RG} + {r}, S - 1);")
        A(f"const device T* qh{r} = q + (((rbase + (uint)s{r}) * NH + lane) * HD);")
        A(f"const float wl{r} = float(w[(rbase + (uint)s{r}) * NH + lane]);")
        A(f"int qpos{r} = q0pos + s{r};")
    A(f"for (uint j = 0; j < (uint)KB; j += {U}) {{")
    A("  int p0 = (int)(blk * (uint)KB + j);")
    A("  if (p0 >= PX) { break; }")
    A("  if (p0 >= P) {")                              # only reachable in the batched kernel: fill, no arithmetic
    A("    if (lane == 0) {")
    for r in range(RG):
        for u in range(U):
            A(f"      if (live{r} && p0 + {u} < PX) {{ scores[(rbase + (uint)s{r}) * (uint)PX + (uint)(p0 + {u})] = static_cast<T>(-1e30f); }}")
    A("    }")
    A("    continue;")
    A("  }")
    for u in range(U): A(f"  const device T* k{u} = kb + (uint)min(p0 + {u}, PX - 1) * HD;")
    acc = [(r, u) for r in range(RG) for u in range(U)]
    A("  float " + ", ".join(f"a{r}_{u} = 0.0f" for r, u in acc) + ";")
    A("  for (int d = 0; d < HD; d += 4) {")
    for u in range(U):
        A(f"    float x{u}0 = float(k{u}[d]), x{u}1 = float(k{u}[d + 1]), x{u}2 = float(k{u}[d + 2]), x{u}3 = float(k{u}[d + 3]);")
    for r in range(RG):
        A(f"    float q{r}0 = float(qh{r}[d]), q{r}1 = float(qh{r}[d + 1]), q{r}2 = float(qh{r}[d + 2]), q{r}3 = float(qh{r}[d + 3]);")
        for u in range(U):
            A(f"    a{r}_{u} += q{r}0 * x{u}0 + q{r}1 * x{u}1 + q{r}2 * x{u}2 + q{r}3 * x{u}3;")
    A("  }")
    for r, u in acc:
        A(f"  float o{r}_{u} = simd_sum(metal::max(a{r}_{u}, 0.0f) * wl{r});")
    A("  if (lane == 0) {")
    for r, u in acc:
        A(f"    if (live{r} && p0 + {u} < PX) {{")
        A(f"      bool ok = (p0 + {u} < P) && ((p0 + {u} + 1) * KPOOL - 1 <= qpos{r});")
        A(f"      scores[(rbase + (uint)s{r}) * (uint)PX + (uint)(p0 + {u})] = static_cast<T>(ok ? o{r}_{u} : -1e30f);")
        A("    }")
    A("  }")
    A("}")
    return "\n".join("    " + x for x in L)

def _shape(S):
    """(keys per step, rows per simdgroup) for S query rows; measured on the 64-core M5 Ultra."""
    if S == 1: return 8, 1
    if S % 3 == 0: return 4, 3
    return 4, 2

_KERN = {}
def _kernel(U, RG, batched):
    key = (U, RG, batched)
    if key not in _KERN:
        _KERN[key] = mx.fast.metal_kernel(
            name=f"tp_idx_scores_{'b' if batched else 's'}_u{U}r{RG}",
            input_names=["q", "k", "w", "rows", "dims"] if batched else ["q", "k", "w", "before", "pool_len"],
            output_names=["scores"], source=_gen(U, RG, batched))
    return _KERN[key]

def _ok(q, heads, head_dim):
    return heads == 32 and head_dim % 4 == 0 and q.dtype in (mx.float16, mx.bfloat16) and KB % 8 == 0

def scores_single(q, pool_keys, weights, before, kpool):
    """q (1,S,H,D), pool_keys (1,P,D), weights (1,S,H) -> (1,S,P) in q.dtype; None when the shape is not covered."""
    _, S, H, D = q.shape; P = pool_keys.shape[1]
    if not _ok(q, H, D) or q.shape[0] != 1 or P < 1 or S < 1: return None
    U, RG = _shape(S)
    out = _kernel(U, RG, False)(
        inputs=[q.reshape(-1), pool_keys.reshape(-1), weights.reshape(-1), mx.array([before], dtype=mx.int32), mx.array([P, S], dtype=mx.int32)],
        template=[("T", q.dtype), ("NH", H), ("HD", D), ("KPOOL", kpool), ("KB", KB)],
        grid=(((P + KB - 1) // KB) * 32, (S + RG - 1) // RG, 1), threadgroup=(32, 1, 1),
        output_shapes=[(S, P)], output_dtypes=[q.dtype])[0]
    return out[None]

def scores_batched(q, keys, kstride, weights, before, plen, kpool, pool_extent):
    """q (B,S,H,D), keys = backing buffer with kstride elements per sequence, weights (B,S,H) -> (B,S,pool_extent)."""
    B, S, H, D = q.shape
    if not _ok(q, H, D) or pool_extent < 1 or S < 1: return None
    U, RG = _shape(S)
    rows = mx.array([v for pair in zip(plen, before) for v in pair], dtype=mx.int32)
    return _kernel(U, RG, True)(
        inputs=[q.reshape(-1), keys, weights.reshape(-1), rows, mx.array([pool_extent, S, kstride], dtype=mx.int32)],
        template=[("T", q.dtype), ("NH", H), ("HD", D), ("KPOOL", kpool), ("KB", KB)],
        grid=(((pool_extent + KB - 1) // KB) * 32, (S + RG - 1) // RG, B), threadgroup=(32, 1, 1),
        output_shapes=[(B, S, pool_extent)], output_dtypes=[q.dtype])[0]

def install(layers):
    cls = type(layers[0]); L = sys.modules[cls.__module__]
    if not hasattr(L, "small_scores"): return "language module has no small_scores"
    SI = sys.modules[L.small_scores.__module__]
    if getattr(SI, "_tp_idxrows", False): return "already installed"
    if _md5(getattr(SI, "_SRC", "")) != _md5(REF_SINGLE) or _md5(getattr(SI, "_BATCHED_SRC", "")) != _md5(REF_BATCHED):
        return "NOT installed: the vendored scoring kernels differ from the ones this module was derived from"
    orig_single, orig_batched = SI.small_scores, SI.small_scores_batched
    def small_scores(q, pool_keys, weights, before, kpool):
        SI._validate_scores(q, pool_keys, weights, [before], [pool_keys.shape[1]])
        out = scores_single(q, pool_keys, weights, before, kpool)
        if out is None:
            STATS["fallback"] += 1; return orig_single(q, pool_keys, weights, before, kpool)
        STATS["single"] += 1; return out
    def small_scores_batched(q, pool_keys, weights, before, plen, kpool, pool_cache=None):
        SI._validate_scores(q, pool_keys, weights, before, plen)
        keys, kstride = SI._pool_storage(pool_keys, pool_cache)
        out = scores_batched(q, keys, kstride, weights, before, plen, kpool, pool_keys.shape[1])
        if out is None:
            STATS["fallback"] += 1; return orig_batched(q, pool_keys, weights, before, plen, kpool, pool_cache)
        STATS["batched"] += 1; return out
    SI.small_scores = small_scores; SI.small_scores_batched = small_scores_batched
    L.small_scores = small_scores                      # language.py imported the name
    SI._tp_idxrows = True
    return f"index scores: {KB} keys per simdgroup, shapes (keys per step, rows together) 1 row {_shape(1)}, 2 rows {_shape(2)}, 3 rows {_shape(3)}"

if __name__ == "__main__":
    # Bit-identity against the vendored kernel text over random shapes, then timings. Small: a few MB, ~1 s of GPU.
    import random, time
    NH, HD, KPOOL = 32, 128, 4
    ref_s = mx.fast.metal_kernel(name="ref_idx_s", input_names=["q", "k", "w", "before", "pool_len"], output_names=["scores"], source=REF_SINGLE)
    ref_b = mx.fast.metal_kernel(name="ref_idx_b", input_names=["q", "k", "w", "rows", "dims"], output_names=["scores"], source=REF_BATCHED)
    def ref_single(q, k, w, before):
        S, P = q.shape[1], k.shape[1]
        return ref_s(inputs=[q.reshape(-1), k.reshape(-1), w.reshape(-1), mx.array([before], dtype=mx.int32), mx.array([P], dtype=mx.int32)],
                     template=[("T", q.dtype), ("NH", NH), ("HD", HD), ("KPOOL", KPOOL)], grid=(P * 32, S, 1), threadgroup=(32, 1, 1),
                     output_shapes=[(S, P)], output_dtypes=[q.dtype])[0][None]
    def ref_batched(q, keys, kstride, w, before, plen, px):
        B, S = q.shape[:2]
        return ref_b(inputs=[q.reshape(-1), keys, w.reshape(-1), mx.array([v for pair in zip(plen, before) for v in pair], dtype=mx.int32),
                             mx.array([px, S, kstride], dtype=mx.int32)],
                     template=[("T", q.dtype), ("NH", NH), ("HD", HD), ("KPOOL", KPOOL)], grid=(px * 32, S, B), threadgroup=(32, 1, 1),
                     output_shapes=[(B, S, px)], output_dtypes=[q.dtype])[0]
    def same(a, b): return bool(mx.array_equal(a.astype(mx.float32), b.astype(mx.float32)).item())
    rnd = random.Random(5); mx.random.seed(5); bad = 0; n = 0
    for dt in (mx.bfloat16, mx.float16):
        for S in range(1, 9):
            for P in (1, 2, 7, 31, 32, 33, 513, 2049, 4097, 9001):
                q = mx.random.normal((1, S, NH, HD)).astype(dt); k = mx.random.normal((1, P, HD)).astype(dt); w = mx.random.normal((1, S, NH)).astype(dt)
                for before in (0, max(0, P * KPOOL - 9), P * KPOOL - 1, P * KPOOL + 40, rnd.randrange(0, P * KPOOL + 1)):
                    n += 1; ok = same(ref_single(q, k, w, before), scores_single(q, k, w, before, KPOOL)); bad += not ok
                    if not ok: print("MISMATCH single", dt, S, P, before)
    for dt in (mx.bfloat16,):
        for B in (1, 2, 3, 4, 8):
            for S in (1, 2, 3, 4, 5, 8):
                for px, cap in ((33, 40), (700, 1024), (5000, 5000), (8191, 9000)):
                    q = mx.random.normal((B, S, NH, HD)).astype(dt); buf = mx.random.normal((B, cap, HD)).astype(dt); w = mx.random.normal((B, S, NH)).astype(dt)
                    plen = [px if i == 0 else rnd.randrange(0, px + 1) for i in range(B)]
                    before = [max(0, p * KPOOL + rnd.randrange(-6, 8)) for p in plen]
                    n += 1; ok = same(ref_batched(q, buf, cap * HD, w, before, plen, px), scores_batched(q, buf, cap * HD, w, before, plen, KPOOL, px)); bad += not ok
                    if not ok: print("MISMATCH batched", B, S, px, cap, plen, before)
    print(f"bit-identity: {n} cases, {bad} mismatches")
    def bench(fn, q, n=150, reps=5):
        def chain(m):
            prev = None
            for _ in range(m): prev = fn(q if prev is None else mx.depends(q, [prev]))
            return prev
        mx.eval(chain(300)); best = 1e9
        for _ in range(reps):
            o = chain(n); t = time.perf_counter(); mx.eval(o); best = min(best, (time.perf_counter() - t) / n)
        return best * 1e6
    if os.environ.get("BENCH", "1") == "1":
        print(f"{'keys':>6} {'rows':>4} {'vendored us':>11} {'this us':>8}")
        for P in (4900, 14600, 27500, 35200, 50000):
            for S in (1, 2, 3, 4, 8):
                q = mx.random.normal((1, S, NH, HD)).astype(mx.bfloat16); k = mx.random.normal((1, P, HD)).astype(mx.bfloat16); w = mx.random.normal((1, S, NH)).astype(mx.bfloat16)
                mx.eval(q, k, w); before = P * KPOOL
                print(f"{P:6d} {S:4d} {bench(lambda qq: ref_single(qq, k, w, before), q):11.1f} {bench(lambda qq: scores_single(qq, k, w, before, KPOOL), q):8.1f}", flush=True)
        for B, S in ((2, 2), (4, 2)):
            P = 27500; q = mx.random.normal((B, S, NH, HD)).astype(mx.bfloat16); buf = mx.random.normal((B, P, HD)).astype(mx.bfloat16); w = mx.random.normal((B, S, NH)).astype(mx.bfloat16)
            mx.eval(q, buf, w); plen = [P, 3300, 3300, 3300][:B]; before = [p * KPOOL for p in plen]
            print(f"batched {B} streams x {S} rows (one long, others short): vendored {bench(lambda qq: ref_batched(qq, buf, P * HD, w, before, plen, P), q):.1f} us, this {bench(lambda qq: scores_batched(qq, buf, P * HD, w, before, plen, KPOOL, P), q):.1f} us", flush=True)
    sys.exit(1 if bad else 0)
