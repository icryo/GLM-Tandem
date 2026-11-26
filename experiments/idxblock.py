# Long-context index scoring: the live kernel (one 32-thread threadgroup per pool key and row) against a blocked
# variant where each simdgroup walks KB keys. Same arithmetic per key (one head per lane, fp32 dot in groups of four
# terms, relu * weight, simd_sum), so the outputs must be bit-identical; only the launch structure changes.
# Tiny: at most ~13 MB of data, a few hundred ms of GPU time. Run on an idle Mac.
import os, sys, time
import mlx.core as mx

REF = """
    uint p = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;   // heads lane, lane+32, ...
    int P = pool_len[0];
    int qpos = before[0] + (int)s;
    float out = -1e30f;
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
BLK = """
    uint blk = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    int P = pool_len[0];
    int qpos = before[0] + (int)s;
    for (uint j = 0; j < (uint)KB; j++) {
        uint p = blk * (uint)KB + j;
        if ((int)p >= P) { break; }
        float out = -1e30f;
        if (((int)p + 1) * KPOOL - 1 <= qpos) {
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
    }
"""
# Two keys per step: the head's query values are loaded once and used for both keys; each key keeps its own
# accumulator and the same order of additions.
BLK2 = """
    uint blk = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    int P = pool_len[0];
    int qpos = before[0] + (int)s;
    const device T* qh = q + ((s * NH + lane) * HD);
    const float wl = float(w[s * NH + lane]);
    for (uint j = 0; j < (uint)KB; j += 2) {
        uint p0 = blk * (uint)KB + j;
        if ((int)p0 >= P) { break; }
        uint p1 = p0 + 1;
        bool has1 = (int)p1 < P;
        const device T* k0 = k + p0 * HD;
        const device T* k1 = k + (has1 ? p1 : p0) * HD;
        float a0 = 0.0f, a1 = 0.0f;
        for (int d = 0; d < HD; d += 4) {
            float q0 = float(qh[d]), q1 = float(qh[d + 1]), q2 = float(qh[d + 2]), q3 = float(qh[d + 3]);
            a0 += q0 * float(k0[d]) + q1 * float(k0[d + 1]) + q2 * float(k0[d + 2]) + q3 * float(k0[d + 3]);
            a1 += q0 * float(k1[d]) + q1 * float(k1[d + 1]) + q2 * float(k1[d + 2]) + q3 * float(k1[d + 3]);
        }
        float s0 = simd_sum(metal::max(a0, 0.0f) * wl);
        float s1 = simd_sum(metal::max(a1, 0.0f) * wl);
        if (lane == 0) {
            scores[s * (uint)P + p0] = static_cast<T>((((int)p0 + 1) * KPOOL - 1 <= qpos) ? s0 : -1e30f);
            if (has1) { scores[s * (uint)P + p1] = static_cast<T>((((int)p1 + 1) * KPOOL - 1 <= qpos) ? s1 : -1e30f); }
        }
    }
"""
def mk(name, src): return mx.fast.metal_kernel(name=name, input_names=["q", "k", "w", "before", "pool_len"], output_names=["scores"], source=src)
K_REF, K_BLK, K_BLK2 = mk("idx_ref", REF), mk("idx_blk", BLK), mk("idx_blk2", BLK2)
NH, HD, KPOOL = 32, 128, 4
def call(kern, q, k, w, before, kb=None):
    S, P = q.shape[1], k.shape[1]
    tmpl = [("T", q.dtype), ("NH", NH), ("HD", HD), ("KPOOL", KPOOL)] + ([("KB", kb)] if kb else [])
    nx = P if kb is None else (P + kb - 1) // kb
    return kern(inputs=[q.reshape(-1), k.reshape(-1), w.reshape(-1), mx.array([before], dtype=mx.int32), mx.array([P], dtype=mx.int32)],
                template=tmpl, grid=(nx * 32, S, 1), threadgroup=(32, 1, 1), output_shapes=[(S, P)], output_dtypes=[q.dtype])[0]
def chain(fn, q, n):
    prev = None
    for _ in range(n):
        qq = q if prev is None else mx.depends(q, [prev])
        prev = fn(qq)
    return prev
def timeit(fn, q, n=200, reps=5):
    mx.eval(chain(fn, q, 400))                       # warm-up: lets the GPU reach its top frequency state
    best = 1e9
    for _ in range(reps):
        out = chain(fn, q, n); t = time.perf_counter(); mx.eval(out); best = min(best, (time.perf_counter() - t) / n)
    return best * 1e6
mx.random.seed(3)
KBS = [int(x) for x in os.environ.get("KBS", "4,8,16,32,64,128").split(",")]
print(f"{'keys':>6} {'rows':>4} {'live us':>8} | " + " ".join(f"KB{kb:<4}" for kb in KBS) + " | " + " ".join(f"2x{kb:<4}" for kb in KBS) + " | identical")
for P in [int(x) for x in os.environ.get("PS", "4900,14600,35200,50000").split(",")]:
    for S in (1, 2, 3):
        q = mx.random.normal((1, S, NH, HD)).astype(mx.bfloat16); k = mx.random.normal((1, P, HD)).astype(mx.bfloat16)
        w = mx.random.normal((1, S, NH)).astype(mx.bfloat16); before = (P - 3) * KPOOL    # last keys: window not complete -> fill value
        mx.eval(q, k, w)
        ref = call(K_REF, q, k, w, before); mx.eval(ref)
        same = True
        for kb in KBS:
            for kern in (K_BLK, K_BLK2):
                o = call(kern, q, k, w, before, kb); mx.eval(o)
                same = same and bool(mx.array_equal(ref.astype(mx.float32), o.astype(mx.float32)).item())
        t_ref = timeit(lambda qq: call(K_REF, qq, k, w, before), q)
        t1 = [timeit(lambda qq, kb=kb: call(K_BLK, qq, k, w, before, kb), q) for kb in KBS]
        t2 = [timeit(lambda qq, kb=kb: call(K_BLK2, qq, k, w, before, kb), q) for kb in KBS]
        print(f"{P:6d} {S:4d} {t_ref:8.1f} | " + " ".join(f"{t:6.1f}" for t in t1) + " | " + " ".join(f"{t:6.1f}" for t in t2) + f" | {same}", flush=True)
