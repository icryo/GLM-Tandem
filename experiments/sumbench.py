# One prompt block's partial is [1, 2048, 4096]. Cost of adding two of them on the CPU (what all_sum does after the
# receive) against stacking them on the CPU (what all_gather does) and adding on the GPU. Also: are the bits the same?
import time
import mlx.core as mx

def best(fn, n=30):
    b = 1e9
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(fn())
        b = min(b, time.perf_counter() - t)
    return b * 1e3

for dt in (mx.bfloat16, mx.float32, mx.float16):
    a = mx.random.normal((1, 2048, 4096)).astype(dt)
    b = mx.random.normal((1, 2048, 4096)).astype(dt)
    mx.eval(a, b)
    cpu = mx.add(a, b, stream=mx.cpu)
    gpu = a + b
    mx.eval(cpu, gpu)
    same = bool(mx.array_equal(cpu.astype(mx.float32), gpu.astype(mx.float32)).item())
    t_cpu = best(lambda: mx.add(a, b, stream=mx.cpu))
    t_stack = best(lambda: mx.stack([a, b], stream=mx.cpu))
    t_gpu = best(lambda: a + b)
    st = mx.stack([a, b], stream=mx.cpu)
    mx.eval(st)
    t_gpu2 = best(lambda: st[0] + st[1])
    print(f"{str(dt):22s} {a.nbytes / 2**20:6.1f} MiB: CPU add {t_cpu:6.2f} ms | CPU stack {t_stack:6.2f} ms | GPU add {t_gpu:5.2f} ms | GPU add of stacked {t_gpu2:5.2f} ms | CPU sum == GPU sum: {same}")
