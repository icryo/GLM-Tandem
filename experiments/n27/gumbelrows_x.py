# Gumbel noise for several output positions in ONE dispatch, bit-identical to the server's
# _noise(seed, index, V) = mx.random.gumbel(shape=(V,), key=mx.random.key(seed + index)), which MLX
# builds from ~9 kernels per position (RandomBits, uint->float, divide, minimum, multiply, add, log,
# negative, log, negative).
#
# MLX's recipe, reproduced exactly (mlx/random.cpp, kernels/random.metal):
#   key(seed)          = (seed >> 32, seed & 0xffffffff)
#   bits(shape=(V,))   : half = V / 2 (V even); thread y in [0, half) hashes counter (y, y + half) with
#                        threefry2x32 and writes out[y] = x, out[y + half] = y
#   uniform            = min(float(bits) / float(UINT32_MAX), nextafter(1, 0)) * 1 + 0
#   gumbel             = -log(-log(uniform))          (metal::precise::log)
#
# rows(seeds_plus_index: list[int], V) -> [R, V] float32
import mlx.core as mx

_SRC = r"""
  const uint y = thread_position_in_grid.x;
  const uint r = thread_position_in_grid.y;
  if (y >= uint(HALF)) {
    return;
  }
  const uint k0 = keys[2 * r];
  const uint k1 = keys[2 * r + 1];
  const uint ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
  uint vx = y + ks[0];
  uint vy = (y + uint(HALF)) + ks[1];
  const uint rot[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      const uint rr = rot[i % 2][j];
      vx += vy;
      vy = (vy << rr) | (vy >> (32 - rr));
      vy ^= vx;
    }
    vx += ks[(i + 1) % 3];
    vy += ks[(i + 2) % 3] + uint(i) + 1u;
  }
  const float maxv = 4294967296.0f;          // float32(UINT32_MAX)
  const float upper = 0.99999994039535522f;  // nextafter(1.0f, 0.0f)
  float u0 = static_cast<float>(vx) / maxv;
  float u1 = static_cast<float>(vy) / maxv;
  u0 = u0 < upper ? u0 : upper;
  u1 = u1 < upper ? u1 : upper;
  device float* o = out + size_t(r) * size_t(V);
  o[y] = -metal::precise::log(-metal::precise::log(u0));
  o[y + uint(HALF)] = -metal::precise::log(-metal::precise::log(u1));
"""

_kern = {}


def _kernel():
    if "k" not in _kern:
        _kern["k"] = mx.fast.metal_kernel(name="tp_gumbel_rows", input_names=["keys"], output_names=["out"], source=_SRC)
    return _kern["k"]


def keys_of(seeds):
    return mx.array([[(s >> 32) & 0xFFFFFFFF, s & 0xFFFFFFFF] for s in seeds], dtype=mx.uint32)


def rows(seeds, V):
    """Gumbel noise [len(seeds), V] for keys mx.random.key(s) of each s (= seed + output index)."""
    if V % 2:
        return None
    R = len(seeds)
    half = V // 2
    tg = 256
    return _kernel()(inputs=[keys_of(seeds)], template=[("V", V), ("HALF", half)],
                     grid=(((half + tg - 1) // tg) * tg, R, 1), threadgroup=(tg, 1, 1),
                     output_shapes=[(R, V)], output_dtypes=[mx.float32])[0]


def ref(seed, V):
    return mx.random.gumbel(shape=(V,), key=mx.random.key(seed))


if __name__ == "__main__":
    import random, sys, time
    V = int(sys.argv[1]) if len(sys.argv) > 1 else 154880
    rnd = random.Random(1)
    bad = 0; n = 0
    seeds = [0, 1, 2**32 - 1, 2**32, 2**40 + 12345] + [rnd.randrange(0, 2**30) * 4096 + rnd.randrange(0, 4096) for _ in range(59)]
    for i in range(0, len(seeds), 8):
        chunk = seeds[i:i + 8]
        g = rows(chunk, V)
        rf = mx.stack([ref(s, V) for s in chunk])
        mx.eval(g, rf)
        eq = mx.array_equal(g, rf, equal_nan=True).item()
        n += len(chunk)
        if not eq:
            d = mx.sum(g != rf).item()
            bad += 1
            print("MISMATCH chunk", i, "elements differing", d, "max abs", mx.max(mx.abs(g - rf)).item())
    print(f"{n} positions x {V} values: {'all bit-identical' if bad == 0 else f'{bad} chunks differ'}")
    # speed: R new positions per step, current recipe vs one dispatch
    for R in (1, 2, 4, 8):
        base = [rnd.randrange(0, 2**30) * 4096 for _ in range(R)]
        def cur(it): return mx.stack([ref(b + it, V) for b in base])
        def new(it): return rows([b + it for b in base], V)
        best = {}
        for rep in range(4):
            for name, f in (("cur", cur), ("new", new)):
                for it in range(3): mx.eval(f(it))
                t0 = time.perf_counter(); outs = [f(it) for it in range(200)]; mx.eval(outs)
                best[name] = min(best.get(name, 1e9), (time.perf_counter() - t0) / 200 * 1e6)
        print(f"R={R}: current {best['cur']:.1f} us, one dispatch {best['new']:.1f} us per step (throughput, back to back)")
