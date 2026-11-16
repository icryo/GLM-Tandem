# Fused dense latent attention for short verify blocks in the sparse-attention layers.
#
# While the context is short enough that the index selection is bypassed, a 2..8-row verify
# block runs: per-head query projection (quantized) -> scores against every cached latent key
# -> mask -> softmax -> weighted sum of the latents -> per-head output projection (quantized),
# as five or six dependent dispatches. Here one threadgroup per (head, row) does all of it.
import inspect, sys, textwrap
import mlx.core as mx

STATS = {"fused": 0, "declined": 0}
ENABLED = [True]
SCAP = int(__import__('os').environ.get('TP_SA_SCAP', '4096'))          # keys a threadgroup can hold scores for
PH = [15]             # phases to run (debug)

_SRC = r"""
  const uint lid = thread_position_in_threadgroup.x;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const int g = int(threadgroup_position_in_grid.y);
  const int h = g / L;
  const int r = g - h * L;
  const int nk = int(meta[0]) - L + r + 1;       // causal: row r sees the keys up to itself
  threadgroup float ql[RK];
  threadgroup float sc[SCAP];
  threadgroup float ol[RK];
  threadgroup float red[32];
  threadgroup float red2[32];

  // 1. q_latent = embed_q[h] @ q[h, r]  (affine 4-bit, groups of GS)
  if (lid < RK) {
    const device T* xq = q + (size_t(h) * L + r) * QD;
    const size_t row = size_t(h) * RK + lid;
    const device uint32_t* wr = eq_w + row * (QD / 8);
    float acc = 0.0f;
    for (int gi = 0; gi < QD / GS; gi++) {
      const float s = float(eq_s[row * (QD / GS) + gi]);
      const float b = float(eq_b[row * (QD / GS) + gi]);
      float dq = 0.0f, sx = 0.0f;
      for (int j = 0; j < GS; j++) {
        const int i = gi * GS + j;
        const float wq = float((wr[i >> 3] >> (4 * (i & 7))) & 0xf);
        const float xv = float(xq[i]);
        dq += wq * xv;
        sx += xv;
      }
      acc += s * dq + b * sx;
    }
    ql[lid] = float(static_cast<T>(acc));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!(PH & 2)) { if (lid < VD) out[(size_t(r) * H + h) * VD + lid] = static_cast<T>(ql[lid]); return; }
  // 2. scores over the keys, running max
  float lmax = -INFINITY;
  for (int k = int(lid); k < nk; k += 1024) {
    const device T* kr = kv + size_t(k) * RK;
    float d = 0.0f;
    for (int i = 0; i < RK; i++) {
      d += ql[i] * float(kr[i]);
    }
    d *= scale[0];
    sc[k] = d;
    lmax = max(lmax, d);
  }
  lmax = simd_max(lmax);
  if (simd_lid == 0) {
    red[simd_gid] = lmax;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = red[0];
  for (int i = 1; i < 32; i++) {
    m = max(m, red[i]);
  }
  float lsum = 0.0f;
  for (int k = int(lid); k < nk; k += 1024) {
    const float e = exp(sc[k] - m);
    sc[k] = e;
    lsum += e;
  }
  lsum = simd_sum(lsum);
  if (simd_lid == 0) {
    red2[simd_gid] = lsum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < 32; i++) {
    total += red2[i];
  }
  const float inv = 1.0f / total;

  if (!(PH & 4)) { if (lid < VD) out[(size_t(r) * H + h) * VD + lid] = static_cast<T>(inv); return; }
  // 3. attended latent
  if (lid < RK) {
    float acc = 0.0f;
    for (int k = 0; k < nk; k++) {
      acc += sc[k] * float(kv[size_t(k) * RK + lid]);
    }
    ol[lid] = float(static_cast<T>(acc * inv));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!(PH & 8)) { if (lid < VD) out[(size_t(r) * H + h) * VD + lid] = static_cast<T>(ol[lid]); return; }
  // 4. out = unembed_out[h] @ latent
  if (lid < VD) {
    const size_t row = size_t(h) * VD + lid;
    const device uint32_t* wr = ue_w + row * (RK / 8);
    float acc = 0.0f;
    for (int gi = 0; gi < RK / GS; gi++) {
      const float s = float(ue_s[row * (RK / GS) + gi]);
      const float b = float(ue_b[row * (RK / GS) + gi]);
      float dq = 0.0f, sx = 0.0f;
      for (int j = 0; j < GS; j++) {
        const int i = gi * GS + j;
        const float wq = float((wr[i >> 3] >> (4 * (i & 7))) & 0xf);
        dq += wq * ol[i];
        sx += ol[i];
      }
      acc += s * dq + b * sx;
    }
    out[(size_t(r) * H + h) * VD + lid] = static_cast<T>(acc);
  }
"""

_k = {}


def _kernel():
    if "k" not in _k:
        _k["k"] = mx.fast.metal_kernel(
            name="tp_mla_dense_rows",
            input_names=["q", "kv", "meta", "scale", "eq_w", "eq_s", "eq_b", "ue_w", "ue_s", "ue_b"],
            output_names=["out"],
            header="#include <metal_simdgroup>\n#include <metal_stdlib>\nusing namespace metal;\n",
            source=_SRC,
        )
    return _k["k"]


_scales = {}


def dense_rows(self, q, kv_latent, mask, cache):
    """q [1, H, L, QD], kv_latent [1, 1, Kv, RK] -> flat [1, L, H * VD] (pre-o_proj), or None."""
    if not ENABLED[0]:
        return None
    eq, ue = self.embed_q, self.unembed_out
    try:
        ew, es, eb = eq["weight"], eq["scales"], eq["biases"]
        uw, us, ub = ue["weight"], ue["scales"], ue["biases"]
    except Exception:
        return None
    B, H, L, QD = q.shape
    Kv, RK = kv_latent.shape[2], kv_latent.shape[3]
    VD = uw.shape[1]
    gs = getattr(eq, "group_size", 0)
    ok = (
        B == 1 and 2 <= L <= 8 and mask is not None and Kv <= SCAP and Kv >= L
        and kv_latent.shape[:2] == (1, 1)
        and getattr(cache[0], "left_padding", None) is None
        and getattr(eq, "bits", 0) == 4 and getattr(ue, "bits", 0) == 4
        and gs in (32, 64) and getattr(ue, "group_size", 0) == gs
        and getattr(eq, "mode", "affine") == "affine" and getattr(ue, "mode", "affine") == "affine"
        and q.dtype == mx.bfloat16 and kv_latent.dtype == q.dtype and es.dtype == q.dtype and us.dtype == q.dtype
        and ew.shape == (H, RK, QD // 8) and es.shape == (H, RK, QD // gs) and eb.shape == es.shape
        and uw.shape == (H, VD, RK // 8) and us.shape == (H, VD, RK // gs) and ub.shape == us.shape
        and RK <= 1024 and VD <= 1024 and QD % gs == 0 and RK % gs == 0
    )
    if not ok:
        STATS["declined"] += 1
        return None
    sc = _scales.get(self.scale)
    if sc is None:
        sc = _scales[self.scale] = mx.array([self.scale], dtype=mx.float32)
    STATS["fused"] += 1
    return _kernel()(
        inputs=[q, kv_latent, mx.array([Kv], dtype=mx.int32), sc, ew, es, eb, uw, us, ub],
        template=[("T", q.dtype), ("H", H), ("L", L), ("QD", QD), ("RK", RK), ("VD", VD), ("GS", gs), ("SCAP", SCAP), ("PH", PH[0])],
        grid=(1024, H * L, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, L, H * VD)],
        output_dtypes=[q.dtype],
    )[0]


def install(layers):
    cls = None
    for l in layers:
        if l.is_linear:
            continue
        cls = next((c for c in type(l.self_attn).__mro__ if "_forward" in c.__dict__), None)
        if cls is not None:
            break
    if cls is None:
        return False
    if getattr(cls, "_tp_sarows", False):
        return True
    Lm = sys.modules[cls.__module__]
    src = textwrap.dedent(inspect.getsource(cls._forward))
    anchor = (
        "    if L <= 8:\n"
        "        q = q_latent if q_latent is not None else _mla_head_proj(self.embed_q, q)\n"
        "        k = v = kv_latent\n"
    )
    if src.count(anchor) != 1:
        return False
    hook = (
        "    if 2 <= L <= 8 and q_latent is None and attn_mask is mask and out_dense is None and topk_indices is None:\n"
        "        _tp_o = _tp_dense_rows(self, q, kv_latent, mask, cache)\n"
        "        if _tp_o is not None:\n"
        "            return self._finish(_tp_o, out_dense)\n"
    )
    src = src.replace(anchor, hook + anchor, 1)
    Lm._tp_dense_rows = dense_rows
    ns = {}
    exec(compile(src, "<tp sarows _forward>", "exec"), Lm.__dict__, ns)
    cls._forward = ns["_forward"]
    cls._tp_sarows = True
    return True
