# What the GLM DSA indexer pays when oMLX's sparse_index extension is missing (MLX 0.32.5 + oMLX 0.8.0rc1
# binaries): decode/verify rows then leave _fast_short_select and take the general path, i.e. matmul scores
# + relu*weight head sum + validity mask + argpartition top-512, instead of fused scores + dsa_topk_indices.
# Times both halves at real shapes (32 heads x 128, pool keys = tokens / 4) per call; 11 sparse layers/pass.
# Run with pyx (0.32.4, bundled extension present) and pyx5 (0.32.5). GPU work only, a few MB.
import time
import mlx.core as mx
from omlx.custom_kernels.sparse_index import fast

H, D, TOPK = 32, 128, 512
print("mlx", mx.__version__, "| sparse_index native:", fast.is_native_available(), flush=True)


def per_call_us(make, n=40):
    mx.eval(make()); mx.synchronize()
    best = 1e9
    for _ in range(3):
        outs = [make() for _ in range(n)]
        t = time.perf_counter(); mx.eval(outs); mx.synchronize()
        best = min(best, (time.perf_counter() - t) / n * 1e6)
    return best


for P in (14600, 35200, 50000):
    for R in (1, 2, 4):
        mx.random.seed(P + R)
        q = (mx.random.normal((1, R, H, D)) * 0.1).astype(mx.bfloat16)
        keys = (mx.random.normal((1, P, D)) * 0.1).astype(mx.bfloat16)
        w = (mx.random.normal((1, R, H)) * 0.1).astype(mx.bfloat16)
        valid = mx.arange(P)[None, None] < (P - 3)
        mx.eval(q, keys, w, valid)

        def scores_fb():
            hs = q @ keys[:, None].swapaxes(-1, -2)
            s = mx.sum(w[..., None] * mx.maximum(hs, mx.array(0, hs.dtype)), axis=2)
            return mx.where(valid, s, -1e30)

        s = scores_fb(); mx.eval(s)

        def topk_fb():
            return mx.argpartition(-s, kth=TOPK - 1, axis=-1)[..., :TOPK]

        a = per_call_us(scores_fb); b = per_call_us(topk_fb)
        line = f"P={P:6d} rows={R}: fallback scores {a:7.1f} us  argpartition top-512 {b:7.1f} us"
        if fast.is_native_available():
            def topk_nat():
                return fast.dsa_topk_indices(s[:, None], TOPK)[:, 0]
            c = per_call_us(topk_nat)
            same = bool(mx.array_equal(mx.sort(topk_nat(), axis=-1), mx.sort(topk_fb(), axis=-1)).item())
            line += f"  native dsa_topk_indices {c:7.1f} us (same set: {same})"
        print(line, flush=True)
