# In-encoder exchange prototype, step 1 (one Mac, no network): what does a GPU -> CPU -> GPU handshake
# cost when the GPU signals and waits INSIDE the command stream (TensorFold glm_ep.metal style), against
# MLX's hop through a CPU-stream op (how every all_sum reaches the network today, with our fast fence)?
#   chain: heavy 4-bit matmul (12576x4096, 2 rows) -> [exchange point] -> next matmul ...
#   gpu   : no exchange (floor)
#   hop   : y -> CPU-stream add -> back (MLX path)
#   encx  : signal kernel writes counter i (system-coherent), native CPU thread sees it and writes ack i,
#           wait kernel spins (bounded) on the ack, then passes y on
# usage: python encx_hop.py ITERS   (needs ./libpoller.dylib next to it)
import ctypes, os, sys, time
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "libpoller.dylib"))
lib.poller_start.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]; lib.poller_join.restype = ctypes.c_long
HDR = r"""
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}
#include <metal_atomic>
"""
SIG = r"""
  if (thread_position_in_grid.x == 0) {
    volatile coherent(system) device uint* g = (volatile coherent(system) device uint*)(gflag);
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
    g[0] = val[0];
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
    tok[0] = val[0];
  }
"""
WAIT = r"""
  // one threadgroup: thread 0 spins (bounded) on the CPU's ack, then everyone copies y through
  threadgroup uint ok;
  if (thread_position_in_threadgroup.x == 0) {
    volatile coherent(system) device uint* c = (volatile coherent(system) device uint*)(cflag);
    uint want = tok[0];
    uint n = 0;
    ok = 0;
    while (n < 1000000u) {
      metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
      if (c[0] >= want) { ok = 1; break; }
      n++;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint i = thread_position_in_threadgroup.x; i < uint(NY); i += threads_per_threadgroup.x) {
    out[i] = y[i];
  }
  if (thread_position_in_threadgroup.x == 0) { flag_out[0] = ok; }
"""
sig_k = mx.fast.metal_kernel(name="encx_signal", input_names=["y", "gflag", "val"], output_names=["tok"], header=HDR, source=SIG)
ONE = r"""
  // signal and wait in one dispatch: thread 0 publishes the counter, then spins (bounded) on the ack
  threadgroup uint ok;
  if (thread_position_in_threadgroup.x == 0) {
    volatile coherent(system) device uint* g = (volatile coherent(system) device uint*)(gflag);
    volatile coherent(system) device uint* c = (volatile coherent(system) device uint*)(cflag);
    const uint want = val[0];
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
    g[0] = want;
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
    uint n = 0;
    ok = 0;
    while (n < 1000000u) {
      if (c[0] >= want) { ok = 1; break; }
      n++;
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (uint i = thread_position_in_threadgroup.x; i < uint(NY); i += threads_per_threadgroup.x) {
    out[i] = y[i];
  }
  if (thread_position_in_threadgroup.x == 0) { flag_out[0] = ok; }
"""
one_k = mx.fast.metal_kernel(name="encx_one", input_names=["y", "gflag", "cflag", "val"], output_names=["out", "flag_out"], header=HDR, source=ONE)
wait_k = mx.fast.metal_kernel(name="encx_wait", input_names=["y", "cflag", "tok"], output_names=["out", "flag_out"], header=HDR, source=WAIT)
O, I = 12576, 4096
banks = []
for i in range(6):
    w = (mx.random.normal((O, I)) * 0.02).astype(mx.bfloat16)
    q = mx.quantize(w, group_size=64, bits=4); mx.eval(q); banks.append(q)
flags = mx.zeros((64,), dtype=mx.uint32); cflags = mx.zeros((64,), dtype=mx.uint32); mx.eval(flags, cflags)
gnp = np.asarray(memoryview(flags)); cnp = np.asarray(memoryview(cflags))
gnp[0] = 123; assert mx.array_equal(flags[0], mx.array(123, dtype=mx.uint32)).item(), "buffer view is a copy"
def chain(n, mode, vals=None):
    x = mx.random.normal((1, 2, I)).astype(mx.bfloat16); mx.eval(x)
    oks = []
    for i in range(n):
        y = mx.quantized_matmul(x, *banks[i % 6], transpose=True, group_size=64, bits=4)[..., :I]
        y = y * 0.01
        if mode == "hop":
            y = mx.squeeze(mx.expand_dims(mx.add(y, y, stream=mx.cpu), 0), 0) * 0.5
        elif mode == "encx":
            yc = mx.contiguous(y)
            t = sig_k(inputs=[yc, flags, vals[i]], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.uint32])[0]
            y, ok = wait_k(inputs=[yc, cflags, t], template=[("NY", yc.size)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[yc.shape, (1,)], output_dtypes=[yc.dtype, mx.uint32])
            y = (y + y) * 0.5; oks.append(ok)
        elif mode == "encx1":
            yc = mx.contiguous(y)
            y, ok = one_k(inputs=[yc, flags, cflags, vals[i]], template=[("NY", yc.size)], grid=(256, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[yc.shape, (1,)], output_dtypes=[yc.dtype, mx.uint32])
            y = (y + y) * 0.5; oks.append(ok)
        else:
            y = (y + y) * 0.5
        x = y
    return x, oks
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
vals = [mx.array([i + 1], dtype=mx.uint32) for i in range(N)]; mx.eval(vals)
res = {}
MODES = os.environ.get("MODES", "gpu,hop,encx,encx1").split(",")
allt = {m: [] for m in MODES}
for rep in range(int(os.environ.get("REPS", "3"))):
    for mode in MODES:
        gnp[0] = 0; cnp[0] = 0
        if mode.startswith("encx"):
            lib.poller_start(gnp.ctypes.data, cnp.ctypes.data, N)
        t = time.perf_counter(); out, oks = chain(N, mode, vals); mx.eval(out, *oks); dt = time.perf_counter() - t
        if mode.startswith("encx"):
            to = lib.poller_join(); good = int(sum(int(o.item()) for o in oks))
            if to or good != N: print(f"  encx: poller timeouts {to}, GPU waits satisfied {good}/{N}")
        res[mode] = min(res.get(mode, 1e9), dt / N * 1e6); allt[mode].append(round(dt / N * 1e6, 1))
print("all reps (us/step):", allt)
if "encx1" not in res: res["encx1"] = float("nan")
print(f"per step: gpu-only {res['gpu']:.2f} us | MLX hop {res['hop']:.2f} us (+{res['hop'] - res['gpu']:.2f}) | in-encoder handshake {res['encx']:.2f} us (+{res['encx'] - res['gpu']:.2f}) | one-dispatch handshake {res['encx1']:.2f} us (+{res['encx1'] - res['gpu']:.2f})")
