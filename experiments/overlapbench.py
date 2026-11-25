# Do independent kernels overlap in MLX on the GPU, and does encode order matter?
# A = bus-bound sweep of a 2 GiB buffer, B = ALU-bound loop without memory traffic. All loops are bounded.
#   pyx overlapbench.py            (needs ~2.2 GiB; total GPU time well under 2 s)
import os, time, subprocess
import mlx.core as mx
C = int(os.environ.get("GPU_CORES", "0")) or int(subprocess.run("ioreg -l | awk -F'= ' '/\"gpu-core-count\"/{print $2; exit}'", shell=True, capture_output=True, text=True).stdout.strip() or 64)
NW = 1 << 29                                   # uint32 words = 2 GiB
SGC = int(os.environ.get("SG_PER_CORE", "96")) # SIMD groups per core for the sweep
WORK = int(os.environ.get("WORK", "170000"))
w = mx.zeros((NW,), dtype=mx.uint32) + 1; mx.eval(w)
A_SRC = """
  uint sg = threadgroup_position_in_grid.x * simdgroups_per_threadgroup + simdgroup_index_in_threadgroup;
  uint lane = thread_index_in_simdgroup;
  uint nsg = geo[0], off = geo[1], words = geo[2];
  if (sg >= nsg) { return; }
  uint per = words / nsg; uint cnt = per / 32u;
  const device uint* p = w + off + (size_t)sg * per;
  uint s = 0;
  for (uint i = 0; i < cnt; i++) { s += p[i * 32u + lane]; }
  if (lane == 0) { out[sg] = float(s & 0xffffu); }
"""
B_SRC = """
  uint gid = thread_position_in_grid.x;
  float x = float(gid & 1023u) * 1e-3f + 1.0f;
  uint n = work[0];
  for (uint i = 0; i < n; i++) { x = x * 1.0000001f + 1e-7f; x = (x > 2.0f) ? x - 1.0f : x; }
  out[gid] = x;
"""
kA = mx.fast.metal_kernel(name="ovl_stream", input_names=["w", "geo"], output_names=["out"], source=A_SRC)
kB = mx.fast.metal_kernel(name="ovl_alu", input_names=["work"], output_names=["out"], source=B_SRC)
def A(off=0, words=NW, cores=C):
    nsg = cores * SGC; tg = 384; ntg = (nsg * 32 + tg - 1) // tg
    geo = mx.array([nsg, off, words], dtype=mx.uint32)
    return kA(inputs=[w, geo], grid=(ntg * tg, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[(nsg,)], output_dtypes=[mx.float32])[0]
def B(mult=1, cores=C):
    n = cores * 384
    return kB(inputs=[mx.array([WORK * mult], dtype=mx.uint32)], grid=(n, 1, 1), threadgroup=(384, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
def t(fn, reps=5):
    best = 1e9
    for _ in range(reps):
        xs = fn(); t0 = time.perf_counter(); mx.eval(*xs); best = min(best, time.perf_counter() - t0)
    return best * 1e3
for _ in range(2): mx.eval(A(), B())
print(f"device cores {C}, sweep {SGC} SIMD groups/core, mlx {mx.__version__}")
a = t(lambda: [A()]); b = t(lambda: [B()])
print(f"A alone (2 GiB sweep): {a:.2f} ms = {NW * 4 / 1e9 / (a / 1e3):.0f} GB/s      B alone (ALU): {b:.2f} ms")
print(f"one eval, A then B : {t(lambda: [A(), B()]):.2f} ms   (serial would be {a + b:.2f})")
print(f"one eval, B then A : {t(lambda: [B(), A()]):.2f} ms")
for m in (2, 4):
    bm = t(lambda: [B(m)])
    print(f"B x{m} alone {bm:.2f} | A then B {t(lambda: [A(), B(m)]):.2f} | B then A {t(lambda: [B(m), A()]):.2f}   (serial {a + bm:.2f})")
h = NW // 2
a1 = t(lambda: [A(0, h)])
print(f"two sweeps of 1 GiB: one alone {a1:.2f} ms | both in one eval {t(lambda: [A(0, h), A(h, h)]):.2f} ms | on half the cores each {t(lambda: [A(0, h, C // 2), A(h, h, C // 2)]):.2f} ms")
# dependent pair for reference: B's work count depends on A's output (forces order)
def dep():
    x = A(); wk = (mx.zeros((1,), dtype=mx.uint32) + WORK) + (x[0] * 0).astype(mx.uint32)
    n = C * 384
    return [kB(inputs=[wk], grid=(n, 1, 1), threadgroup=(384, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0]]
print(f"dependent A -> B   : {t(dep):.2f} ms")
try:
    s2 = mx.new_stream(mx.gpu)
    def two():
        x = A()
        with mx.stream(s2): y = B()
        return [x, y]
    print(f"B on a second GPU stream, one eval: {t(two):.2f} ms")
    def two2():
        x = A()
        with mx.stream(s2): y = B(2)
        return [x, y]
    print(f"B x2 on a second GPU stream       : {t(two2):.2f} ms")
except Exception as e:
    print("second stream test failed:", repr(e)[:120])
