# Minimal model of interleaved prompt reading (no model, one Mac): two chains of GPU work, each step followed by
# a GPU -> CPU -> GPU hop (CPU add of a payload). "seq": one chain, the GPU waits for every hop. "wave": two
# chains alternate, so each hop should be hidden behind the other chain's GPU work.
#   pyx hopwave.py <gpu ms per step> <hop payload MB> [steps]
import os, sys, time, collections
import mlx.core as mx
T_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
P_MB = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 80
DEPTH = int(os.environ.get("DEPTH", "3")); D = 4096; R = 2048
W = (mx.random.normal((D, D)) * 0.0156).astype(mx.bfloat16); mx.eval(W)
RP = max(1, int(P_MB * 1e6 / (D * 2)))                      # payload rows
def work(h, k):
    for _ in range(k): h = h @ W
    return h
h0 = mx.random.normal((R, D)).astype(mx.bfloat16); mx.eval(h0)
def t_of(k, reps=6):
    best = 1e9
    for _ in range(reps):
        x = work(h0, k); t = time.perf_counter(); mx.eval(x); best = min(best, time.perf_counter() - t)
    return best * 1e3
m1 = t_of(4) / 4; K = max(1, round(T_MS / m1)); tk = t_of(K)
pay0 = mx.random.normal((RP, D)).astype(mx.bfloat16); mx.eval(pay0)
def hop_ms(reps=6):
    best = 1e9
    for _ in range(reps):
        y = mx.add(pay0, pay0, stream=mx.cpu); t = time.perf_counter(); mx.eval(y); best = min(best, time.perf_counter() - t)
    return best * 1e3
print(f"gpu work per step: {K} matmuls = {tk:.2f} ms (asked {T_MS}); hop payload {RP} rows = {RP * D * 2 / 1e6:.1f} MB, CPU add alone {hop_ms():.2f} ms", flush=True)
class C: pass
def run(nchains, hop, steps):
    cs = []
    for i in range(nchains):
        c = C(); c.h = h0 + 0.0; c.y = None; cs.append(c)
    mx.eval([c.h for c in cs]); Q = collections.deque(); t = time.perf_counter()
    for s in range(steps):
        for c in cs:
            while len(Q) > DEPTH: mx.eval(Q.popleft())
            h = work(c.h, K)
            n = mx.sqrt(mx.mean(mx.square(h.astype(mx.float32)))) + 1e-6          # keep the scale bounded
            h = (h.astype(mx.float32) / n).astype(mx.bfloat16)
            if hop:
                y = mx.add(h[:RP], h[:RP], stream=mx.cpu)                          # the hop: CPU pass over the payload
                yg = mx.squeeze(mx.expand_dims(y, 0), 0)
                c.h = h + yg[:1] * 0.0                                             # the next step needs the hop result
                tok = y
            else:
                c.h = h; tok = h
            mx.async_eval(tok); Q.append(tok)
    mx.eval([c.h for c in cs]); dt = time.perf_counter() - t
    return 1e3 * dt / (steps * nchains)
for name, nch, hop in (("no hop, one chain ", 1, False), ("hop, one chain    ", 1, True), ("hop, two chains   ", 2, True), ("no hop, two chains", 2, False)):
    r = [run(nch, hop, N if nch == 1 else N // 2) for _ in range(3)]
    print(f"  {name}: {min(r):6.2f} ms per step (runs {' '.join(f'{v:.2f}' for v in r)})", flush=True)
