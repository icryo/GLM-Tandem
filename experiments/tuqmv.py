# 4-bit affine mat-vec for a few rows on Metal 4 tensor units (M5): each simdgroup unpacks a
# 16-row x 16-input weight tile into a cooperative tensor and lets matmul2d apply it to every row.
import mlx.core as mx

_HDR = """
#include <metal_simdgroup>
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""
_SRC = r"""
  const ushort lane = thread_index_in_simdgroup;
  const int row0 = (int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup)) * 16;
  const short qid = lane >> 2;
  const short fm = ((qid & 4) | ((lane >> 1) & 3));
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  constexpr auto desc = matmul2d_descriptor(
      16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<float, float, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<float, float, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  for (short i = 0; i < 16; i++) {
    ct_c[i] = 0.0f;
    ct_b[i] = 0.0f;
  }
  constexpr int WW = K / 8;
  constexpr int G = K / GS;
  const device uint32_t* w0 = w + size_t(row0 + fm) * WW;
  const device uint32_t* w1 = w + size_t(row0 + fm + 8) * WW;
  const device T* s0 = scales + size_t(row0 + fm) * G;
  const device T* s1 = scales + size_t(row0 + fm + 8) * G;
  const device T* b0 = biases + size_t(row0 + fm) * G;
  const device T* b1 = biases + size_t(row0 + fm + 8) * G;
  for (int k = 0; k < K; k += 16) {
    const int g = k / GS;
    const float sa = float(s0[g]), ba = float(b0[g]);
    const float sb = float(s1[g]), bb = float(b1[g]);
    const int pos = k + fn;
    const uint wa = w0[pos >> 3] >> (4 * (pos & 7));
    const uint wb = w1[pos >> 3] >> (4 * (pos & 7));
    for (short i = 0; i < 4; i++) {
      ct_a[i] = sa * float((wa >> (4 * i)) & 0xf) + ba;
      ct_a[4 + i] = sb * float((wb >> (4 * i)) & 0xf) + bb;
    }
    for (short i = 0; i < 8; i++) {
      const short r = fm + (i >> 2) * 8;
      const short c = fn + (i & 3);
      if (c < TT) {
        ct_b[i] = float(x[c * K + k + r]);
      }
    }
    op.run(ct_a, ct_b, ct_c);
  }
  for (short i = 0; i < 8; i++) {
    const short r = fm + (i >> 2) * 8;
    const short c = fn + (i & 3);
    if (c < TT) {
      out[size_t(c) * O + row0 + r] = static_cast<T>(ct_c[i]);
    }
  }
"""
_k = {}
def tu_qmv(x, w, scales, biases, group_size=64, nsg=2):
    T_, K = x.shape
    O = w.shape[0]
    assert O % (16 * nsg) == 0 and K % 64 == 0 and T_ <= 16
    if "k" not in _k:
        _k["k"] = mx.fast.metal_kernel(name="tu_qmv_rows4", input_names=["x", "w", "scales", "biases"], output_names=["out"], header=_HDR, source=_SRC)
    return _k["k"](inputs=[x, w, scales, biases],
                   template=[("T", x.dtype), ("K", K), ("O", O), ("GS", group_size), ("TT", T_), ("NSG", nsg)],
                   grid=(32, O // 16, 1), threadgroup=(32, nsg, 1), output_shapes=[(T_, O)], output_dtypes=[x.dtype])[0]

if __name__ == "__main__":
    import time
    NB = 4
    for (O, K) in ((16384, 4096), (32768, 1024)):
        banks = []
        for i in range(NB):
            wf = (mx.random.normal((O, K)) * 0.02).astype(mx.bfloat16)
            q = mx.quantize(wf, group_size=64, bits=4); mx.eval(q); banks.append(q)
        for T_ in (1, 2, 3, 4):
            x = mx.random.normal((T_, K)).astype(mx.bfloat16); mx.eval(x)
            def bench(fn, n=300):
                mx.eval([fn(i % NB) for i in range(30)]); t = time.perf_counter(); mx.eval([fn(i % NB) for i in range(n)]); return 1e6 * (time.perf_counter() - t) / n
            ref = lambda i: mx.quantized_matmul(x, *banks[i], transpose=True, group_size=64, bits=4)
            mine = lambda i: tu_qmv(x, *banks[i])
            a = ref(0).astype(mx.float32); b = mine(0).astype(mx.float32); mx.eval(a, b)
            d = mx.abs(a - b).max().item(); sc = mx.abs(a).max().item()
            mib = O * K * 4.5 / 8 / 2**20
            print(f"w {O}x{K} ({mib:.0f} MiB) rows={T_}: mlx {bench(ref):6.1f} us   tensor-unit {bench(mine):6.1f} us   max|diff| {d:.2e} (scale {sc:.1f})", flush=True)
