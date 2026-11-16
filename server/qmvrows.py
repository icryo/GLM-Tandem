# Multi-row 4-bit affine quantized mat-vec for verify blocks (2..4 rows).
#
# MLX runs x[R, K] @ W.T for R >= 2 with its qmv_wide kernel, ~30% slower than the one-row
# qmv_fast on the large attention projections. This kernel keeps qmv_fast's layout (two
# simdgroups of four output rows, 16 values per thread per 512-value block) and applies each
# decoded weight block to every row of x, so the weights are still read once and every row
# gets the one-row arithmetic.
import mlx.core as mx

STATS = {"rows": 0}
ENABLED = [True]
MIN_BYTES = 48 << 20      # only the vocabulary head: MLX's wide kernel is as fast or faster below this

_SRC = r"""
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  constexpr int RPS = 4;
  constexpr int NSG = 2;
  constexpr int WB = K / 2;            // bytes per weight row (4 bits per value)
  constexpr int G = K / GS;            // groups per weight row
  const int out_row = int(threadgroup_position_in_grid.y) * (NSG * RPS) + int(simd_gid) * RPS;
  const device uint8_t* ws = (const device uint8_t*)w + size_t(out_row) * WB + simd_lid * 8;
  const device T* sl = scales + size_t(out_row) * G + simd_lid / (GS / 16);
  const device T* bl = biases + size_t(out_row) * G + simd_lid / (GS / 16);
  const device T* xp = x + simd_lid * 16;
  float res[R][RPS];
  for (int r = 0; r < R; r++) {
    for (int row = 0; row < RPS; row++) {
      res[r][row] = 0.0f;
    }
  }
  for (int k = 0; k < K; k += 512) {
    float xt[R][16];
    float sum[R];
    for (int r = 0; r < R; r++) {
      const device T* xr = xp + r * K;
      float s = 0.0f;
      for (int i = 0; i < 16; i += 4) {
        s += float(xr[i]) + float(xr[i + 1]) + float(xr[i + 2]) + float(xr[i + 3]);
        xt[r][i] = float(xr[i]);
        xt[r][i + 1] = float(xr[i + 1]) / 16.0f;
        xt[r][i + 2] = float(xr[i + 2]) / 256.0f;
        xt[r][i + 3] = float(xr[i + 3]) / 4096.0f;
      }
      sum[r] = s;
    }
    for (int row = 0; row < RPS; row++) {
      const device uint16_t* wl = (const device uint16_t*)(ws + row * WB);
      const float s = float(sl[row * G]);
      const float b = float(bl[row * G]);
      float wv[16];
      for (int i = 0; i < 4; i++) {
        const uint16_t v = wl[i];
        wv[4 * i] = float(v & 0x000f);
        wv[4 * i + 1] = float(v & 0x00f0);
        wv[4 * i + 2] = float(v & 0x0f00);
        wv[4 * i + 3] = float(v & 0xf000);
      }
      for (int r = 0; r < R; r++) {
        float accum = 0.0f;
        for (int i = 0; i < 4; i++) {
          accum += (xt[r][4 * i] * wv[4 * i] + xt[r][4 * i + 1] * wv[4 * i + 1] +
                    xt[r][4 * i + 2] * wv[4 * i + 2] + xt[r][4 * i + 3] * wv[4 * i + 3]);
        }
        res[r][row] += s * accum + sum[r] * b;
      }
    }
    ws += 256;
    sl += 512 / GS;
    bl += 512 / GS;
    xp += 512;
  }
  for (int r = 0; r < R; r++) {
    for (int row = 0; row < RPS; row++) {
      float v = simd_sum(res[r][row]);
      if (simd_lid == 0) {
        out[size_t(r) * O + out_row + row] = static_cast<T>(v);
      }
    }
  }
"""

_k = {}


def _kernel():
    if "k" not in _k:
        _k["k"] = mx.fast.metal_kernel(
            name="tp_qmv_rows4",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            header="#include <metal_simdgroup>\n#include <metal_stdlib>\nusing namespace metal;\n",
            source=_SRC,
        )
    return _k["k"]


def rows_qmv(x, w, scales, biases, group_size=64):
    """x [..., R, K] with every leading dim 1; returns [..., R, O] or None when not covered."""
    if x.ndim < 2 or w.ndim != 2:
        return None
    R, K = x.shape[-2], x.shape[-1]
    if not 2 <= R <= 4 or any(d != 1 for d in x.shape[:-2]):
        return None
    O = w.shape[0]
    if w.dtype != mx.uint32 or w.shape[1] * 8 != K or K % 512 or O % 8 or group_size not in (32, 64, 128):
        return None
    if x.dtype not in (mx.bfloat16, mx.float16) or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return None
    if scales.shape != (O, K // group_size) or biases.shape != scales.shape:
        return None
    STATS["rows"] += 1
    out = _kernel()(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("K", K), ("O", O), ("GS", group_size), ("R", R)],
        grid=(32, 2 * (O // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(R, O)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], O)


def install():
    """Route covered mx.quantized_matmul calls (2..4 rows, 4-bit affine) through the row kernel."""
    if getattr(mx, "_tp_qmvrows", False):
        return True
    real = mx.quantized_matmul

    def quantized_matmul(x, w, scales, biases=None, transpose=True, group_size=64, bits=4, mode="affine", stream=None):
        if (ENABLED[0] and bits == 4 and transpose and mode == "affine" and biases is not None and stream is None
                and x.ndim >= 2 and x.shape[-2] == 2 and w.nbytes >= MIN_BYTES):
            y = rows_qmv(x, w, scales, biases, group_size)
            if y is not None:
                return y
        return real(x, w, scales, biases, transpose=transpose, group_size=group_size, bits=bits, mode=mode, stream=stream)

    mx.quantized_matmul = quantized_matmul
    mx._tp_qmvrows = True
    return True
