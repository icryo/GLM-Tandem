# Cost of joining the two halves of the vocabulary logits: (a) as the server does it (transpose copy, all_gather of
# the full rows, argmax over the joined row) against (b) a per-rank max/argmax and a gather of two numbers per row.
import time, os
import mlx.core as mx
g = mx.distributed.init(strict=True, backend="jaccl"); R = g.rank(); n = g.size()
V = 154880; part = V // n
mx.random.seed(7)
for rows in (1, 2, 3, 4, 8):
    base = mx.random.normal((1, rows, part)).astype(mx.bfloat16); mx.eval(base)
    noise = mx.random.gumbel((rows, V)); mx.eval(noise); nl = noise[:, R * part:(R + 1) * part]; mx.eval(nl)
    def full(i, sample):
        y = base * (1.0 + 0.001 * i)
        yt = mx.contiguous(mx.moveaxis(y, -1, 0)); lg = mx.moveaxis(mx.distributed.all_gather(yt), 0, -1)[0]
        z = lg.astype(mx.float32)
        if sample: z = z * 1.25 + noise
        return mx.argmax(z, axis=-1)
    def local(i, sample):
        y = base * (1.0 + 0.001 * i)
        z = y[0].astype(mx.float32)
        if sample: z = z * 1.25 + nl
        m = mx.max(z, axis=-1); a = mx.argmax(z, axis=-1).astype(mx.float32) + R * part
        both = mx.distributed.all_gather(mx.stack([m, a], axis=0)[None])          # [n, 2, rows]
        w = mx.argmax(both[:, 0], axis=0)                                         # first rank wins ties, as argmax over the joined row
        return mx.take_along_axis(both[:, 1], w[None], axis=0)[0].astype(mx.int32)
    for name, f in (("full gather", full), ("local argmax", local)):
        for sample in (False, True):
            for i in range(30): mx.eval(f(i, sample))
            best = 1e9
            for rep in range(5):
                t = time.perf_counter()
                for i in range(200): mx.eval(f(i, sample))
                best = min(best, time.perf_counter() - t)
            if R == 0: print(f"rows {rows} {name:12s} {'sampled' if sample else 'greedy '}: {best / 200 * 1e3:.3f} ms", flush=True)
    a = full(3, True); b = local(3, True); mx.eval(a, b)
    if R == 0: print(f"rows {rows} same tokens: {bool((a == b).all().item())}", flush=True)
