# Index scoring, exact variants generated for U keys per step and optionally all query rows inside one simdgroup
# (key values loaded once for every row). Same per-key arithmetic and order as the live kernel -> must be bit-identical.
import os, time
import mlx.core as mx
from idxblock import REF, mk, NH, HD, KPOOL, chain, timeit
def gen(U, S=0):
    rows = range(S) if S else [None]
    L = ["uint blk = threadgroup_position_in_grid.x;", "uint lane = thread_position_in_threadgroup.x;", "int P = pool_len[0];"]
    if not S: L.append("uint s = threadgroup_position_in_grid.y;")
    for r in rows:
        sr = "s" if r is None else str(r); t = "" if r is None else str(r)
        L += [f"const device T* qh{t} = q + (({sr} * NH + lane) * HD);", f"const float wl{t} = float(w[{sr} * NH + lane]);", f"int qpos{t} = before[0] + (int){sr};"]
    L += ["for (uint j = 0; j < (uint)KB; j += %d) {" % U, "  uint p0 = blk * (uint)KB + j;", "  if ((int)p0 >= P) { break; }"]
    for u in range(U): L.append(f"  const device T* k{u} = k + (uint)min((int)p0 + {u}, P - 1) * HD;")
    acc = [(r, u) for r in rows for u in range(U)]
    nm = lambda r, u: f"a{'' if r is None else r}_{u}"
    L.append("  float " + ", ".join(f"{nm(r, u)} = 0.0f" for r, u in acc) + ";")
    L.append("  for (int d = 0; d < HD; d += 4) {")
    for u in range(U): L.append(f"    float x{u}0 = float(k{u}[d]), x{u}1 = float(k{u}[d + 1]), x{u}2 = float(k{u}[d + 2]), x{u}3 = float(k{u}[d + 3]);")
    for r in rows:
        t = "" if r is None else str(r)
        L.append(f"    float q{t}0 = float(qh{t}[d]), q{t}1 = float(qh{t}[d + 1]), q{t}2 = float(qh{t}[d + 2]), q{t}3 = float(qh{t}[d + 3]);")
        for u in range(U): L.append(f"    {nm(r, u)} += q{t}0 * x{u}0 + q{t}1 * x{u}1 + q{t}2 * x{u}2 + q{t}3 * x{u}3;")
    L.append("  }")
    for r, u in acc:
        t = "" if r is None else str(r)
        L.append(f"  float r{nm(r, u)} = simd_sum(metal::max({nm(r, u)}, 0.0f) * wl{t});")
    L.append("  if (lane == 0) {")
    for r, u in acc:
        sr = "s" if r is None else str(r); t = "" if r is None else str(r)
        L.append(f"    if ((int)p0 + {u} < P) {{ scores[{sr} * (uint)P + p0 + {u}] = static_cast<T>((((int)p0 + {u} + 1) * KPOOL - 1 <= qpos{t}) ? r{nm(r, u)} : -1e30f); }}")
    L += ["  }", "}"]
    return "\n".join("    " + x for x in L)
KB = int(os.environ.get("KB", "32"))
def call(kern, q, k, w, before, inside):
    S, P = q.shape[1], k.shape[1]
    return kern(inputs=[q.reshape(-1), k.reshape(-1), w.reshape(-1), mx.array([before], dtype=mx.int32), mx.array([P], dtype=mx.int32)],
                template=[("T", q.dtype), ("NH", NH), ("HD", HD), ("KPOOL", KPOOL), ("KB", KB)],
                grid=(((P + KB - 1) // KB) * 32, 1 if inside else S, 1), threadgroup=(32, 1, 1), output_shapes=[(S, P)], output_dtypes=[q.dtype])[0]
def ref_call(q, k, w, before):
    S, P = q.shape[1], k.shape[1]
    return K_REF(inputs=[q.reshape(-1), k.reshape(-1), w.reshape(-1), mx.array([before], dtype=mx.int32), mx.array([P], dtype=mx.int32)],
                 template=[("T", q.dtype), ("NH", NH), ("HD", HD), ("KPOOL", KPOOL)], grid=(P * 32, S, 1), threadgroup=(32, 1, 1), output_shapes=[(S, P)], output_dtypes=[q.dtype])[0]
K_REF = mk("idx_ref", REF)
US = [int(x) for x in os.environ.get("US", "2,4,8").split(",")]; MUS = [int(x) for x in os.environ.get("MUS", "1,2,4").split(",")]
mx.random.seed(3)
print(f"KB={KB}  {'keys':>6} {'rows':>4} {'live':>7} | " + " ".join(f"U{u:<5}" for u in US) + " | rows inside: " + " ".join(f"U{u:<5}" for u in MUS) + " | identical")
for P in [int(x) for x in os.environ.get("PS", "35200").split(",")]:
    for S in (1, 2, 3):
        q = mx.random.normal((1, S, NH, HD)).astype(mx.bfloat16); k = mx.random.normal((1, P, HD)).astype(mx.bfloat16)
        w = mx.random.normal((1, S, NH)).astype(mx.bfloat16); before = (P - 3) * KPOOL; mx.eval(q, k, w)
        ref = ref_call(q, k, w, before); mx.eval(ref); same = True; t1 = []; t2 = []
        for u in US:
            kern = mk(f"idx_u{u}", gen(u)); o = call(kern, q, k, w, before, False); mx.eval(o)
            same = same and bool(mx.array_equal(ref.astype(mx.float32), o.astype(mx.float32)).item())
            t1.append(timeit(lambda qq, kern=kern: call(kern, qq, k, w, before, False), q))
        for u in MUS:
            kern = mk(f"idx_s{S}u{u}", gen(u, S)); o = call(kern, q, k, w, before, True); mx.eval(o)
            same = same and bool(mx.array_equal(ref.astype(mx.float32), o.astype(mx.float32)).item())
            t2.append(timeit(lambda qq, kern=kern: call(kern, qq, k, w, before, True), q))
        t_ref = timeit(lambda qq: ref_call(qq, k, w, before), q)
        print(f"       {P:6d} {S:4d} {t_ref:7.1f} | " + " ".join(f"{t:6.1f}" for t in t1) + " |              " + " ".join(f"{t:6.1f}" for t in t2) + f" | {same}", flush=True)
