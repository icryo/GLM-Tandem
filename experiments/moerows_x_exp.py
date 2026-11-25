# Multi-row MoE with the router's top-k selection inside the gate/up kernel.
#
# oMLX folds the router select into the routed gate/up kernel for one token (one dependent
# dispatch less per layer). Verify blocks (2..8 rows) run router logits -> router select ->
# gate/up -> down. This gives the multi-row gate/up kernel (wide shared expert, slot-major
# grid) the same in-kernel selection, per row.
import sys
import mlx.core as mx

import os
STATS = {"fused": 0, "declined": 0, "pre": 0}
PRE = [None]      # (sig, biased) [T, E] computed by the HC pre kernel for the call that follows (hcrows_x, experiment 3)
DEDUP = os.environ.get("TP_MOE_DEDUP", "0") == "1"   # read each distinct expert once per verify window
ENABLED = [True]


def _rep(src, old, new, count=1):
    n = src.count(old)
    assert n == count, (old, n)
    return src.replace(old, new)


def install(layers):
    cls = None
    for l in layers:
        m = l.mlp
        m = m.inner if "inner" in m else m            # tensor-parallel wrapper around the MoE
        cls = next((c for c in type(m).__mro__ if "_decode_select" in c.__dict__), None)
        if cls is not None:
            break
    if cls is None:
        return False
    if getattr(cls, "_tp_moerows", False):
        return True
    L = sys.modules[cls.__module__]
    dk = L._decode_kernels

    src = dk._GATE_UP_SOURCE
    src = _rep(src, "glm_router_topk<NE, TOPK>(sel_biased, simd_lid, picked);",
               "glm_router_topk<NE, TOPK>(sel_biased + token * NE, simd_lid, picked);")
    src = _rep(src, "if (z == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {",
               "if (r == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {")
    src = _rep(src, "gathered[q] = sel_sig[picked[q]];", "gathered[q] = sel_sig[token * NE + picked[q]];")
    src = _rep(src, "sel_indices[q] = uint(picked[q]);", "sel_indices[token * TOPK + q] = uint(picked[q]);")
    src = _rep(src, "sel_scores[q] = sv;", "sel_scores[token * TOPK + q] = sv;")
    if DEDUP:
        # Read each distinct expert once per window: the first row that routes to an expert owns
        # its weight tiles and also computes the result for every later row that routes to it
        # (same arithmetic per row, so each (row, expert) output is unchanged); the later rows'
        # own slots return without touching the weights.
        a = src.index("    constexpr int WB = K * RBITS / 8;   // bytes per weight row")
        b = src.index("  } else {\n#if HAS_SHARED")
        src = src[:a] + r"""
    // Position of this expert in another row's selection = how many experts beat it there
    // (higher score, ties to the lower index): one counting pass instead of a selection replay.
    int rk[NTOK];
    for (int tt = 0; tt < NTOK; tt++) {
      if (tt == token) {
        rk[tt] = r;
        continue;
      }
      const device float* bz = sel_biased + tt * NE;
      const float v = bz[expert];
      int c = 0;
      for (int j = 0; j < (NE + 31) / 32; j++) {
        const int e2 = j * 32 + int(simd_lid);
        if (e2 < NE) {
          const float o = bz[e2];
          c += (o > v || (o == v && e2 < expert)) ? 1 : 0;
        }
      }
      rk[tt] = simd_sum(c);
    }
    for (int tt = 0; tt < token; tt++) {
      if (rk[tt] < TOPK) {
        return;                         // an earlier row owns this expert
      }
    }
    constexpr int WB = K * RBITS / 8;   // bytes per weight row
    constexpr int G = K / RGS;          // groups per row
    const size_t row0 = size_t(expert) * ESTRIDE + out_row;
    const size_t urow0 = row0 + UP_OFF;
    for (int tt = token; tt < NTOK; tt++) {
      const int rr = rk[tt];
      if (rr >= TOPK) {
        continue;
      }
      const device T* xt = x + tt * K;
      float gr[RPS] = {0};
      float ur[RPS] = {0};
      glm_qmv_rows<T, K, RGS, RBITS, RPS>(
          (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
          gate_b + row0 * G, xt, simd_lid, gr);
      glm_qmv_rows<T, K, RGS, RBITS, RPS>(
          (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
          up_b + urow0 * G, xt, simd_lid, ur);
      device T* od = out + size_t(tt * TOPK + rr) * N + out_row;
      for (int row = 0; row < RPS; row++) {
        float gv = simd_sum(gr[row]);
        float uv = simd_sum(ur[row]);
        if (simd_lid == 0) {
          od[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
        }
      }
    }
    return;
""" + src[b:]

    kern = {}

    def kernel():
        if "k" not in kern:
            kern["k"] = mx.fast.metal_kernel(
                name="tp_moe_gate_up_rows_select" + ("_dedup" if DEDUP else ""),
                input_names=["x", "sel_sig", "sel_biased", "sel_scaling", "limit",
                             "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b",
                             "sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b"],
                output_names=["out", "shared_out", "sel_indices", "sel_scores"],
                header=dk._QMV_HEADER,
                source=dk._source(src, HAS_SHARED=0, SHARED_WIDE=1, SLOT_MAJOR=1, SELECT=1),
            )
        return kern["k"]

    consts = {}

    def const(v):
        if v not in consts:
            consts[v] = mx.array([v], dtype=mx.float32)
        return consts[v]

    def gate_up(x, limit, routed_gate, routed_up, shared_gate, shared_up, sig, biased, top_k, scaling, norm):
        rps, nsg = 4, 2
        fused_gu = routed_up is None
        parts = [dk._affine_parts(m) for m in ((routed_gate,) if fused_gu else (routed_gate, routed_up))]
        if any(p is None for p in parts):
            return None
        if fused_gu:
            parts = parts * 2
        (gw, gs, gb, rbits, rgs), (uw, us, ub, ubits, ugs) = parts
        if (rbits, rgs) != (ubits, ugs) or gw.shape != uw.shape or gw.ndim != 3:
            return None
        T, K = x.shape
        E, N, _ = gw.shape
        estride, up_off = N, 0
        if fused_gu:
            if N % 2:
                return None
            N //= 2
            estride, up_off = 2 * N, N
        E_r = sig.shape[-1]
        if sig.shape != (T, E_r) or biased.shape != (T, E_r) or not 1 <= top_k <= 32 or E_r > 1024:
            return None
        if sig.dtype != mx.float32 or biased.dtype != mx.float32:
            return None
        if x.dtype not in (mx.bfloat16, mx.float16) or gs.dtype != x.dtype or us.dtype != x.dtype:
            return None
        if not dk._qmv_fast_ok(rbits, rgs, N, K) or N % (rps * nsg) or not 2 <= T <= 8:
            return None
        sparts = [dk._affine_parts(m) for m in (shared_gate, shared_up)]
        if any(p is None for p in sparts):
            return None
        (sgw, sgs, sgb, sbits, sgsz), (suw, sus, sub, subits, susz) = sparts
        if (sbits, sgsz) != (subits, susz) or sgw.shape[0] != N or suw.shape[0] != N:
            return None
        if sgs.dtype != x.dtype or sus.dtype != x.dtype:
            return None
        if sbits not in (4, 5, 6, 8) or sgsz % 8 or K % sgsz or (K // sgsz) < 1:
            return None
        tiles = N // (rps * nsg)
        slots = T * top_k + 1
        return kernel()(
            inputs=[x, sig, biased, const(float(scaling)), const(float(limit)),
                    gw, gs, gb, uw, us, ub, sgw, sgs, sgb, suw, sus, sub],
            template=[("T", x.dtype), ("K", K), ("N", N), ("TOPK", top_k), ("RBITS", rbits),
                      ("RGS", rgs), ("RPS", rps), ("NSG", nsg), ("ESTRIDE", estride), ("UP_OFF", up_off),
                      ("NE", E_r), ("SEL_NORM", int(bool(norm) and top_k > 1)),
                      ("SBITS", sbits), ("SGS", sgsz), ("NTOK", T)],
            grid=(32, slots * nsg, tiles),
            threadgroup=(32, nsg, 1),
            output_shapes=[(T, top_k, N), (T, N), (T, top_k), (T, top_k)],
            output_dtypes=[x.dtype, x.dtype, mx.uint32, mx.float32],
        )

    prev = cls._decode_select

    def _decode_select(self, x):
        if not ENABLED[0] or x.ndim != 3 or x.shape[0] != 1 or not 2 <= x.shape[1] <= 8:
            return prev(self, x)
        gate, sw, shared = self.gate, self.switch_mlp, self.shared_experts
        T, D = x.shape[1], x.shape[2]
        if (
            not L._DECODE_FUSION
            or gate.n_group != 1
            or shared is None
            or not isinstance(sw, L.SwitchGLU)
            or T * gate.top_k >= L._sort_threshold(*L._switch_projections(sw))
            or L._row_exact_armed()
            or any(L._takes_verify_route(p, T, x.dtype) for p in (shared.gate_proj, shared.up_proj, shared.down_proj))
        ):
            STATS["declined"] += 1
            return None
        limit = getattr(sw.activation, "limit", None)
        if limit is None or shared.limit != limit:
            return None
        x2 = x.reshape(T, D)
        pre = PRE[0]; PRE[0] = None
        if pre is not None and pre[0].shape == (T, gate.e_score_correction_bias.shape[0]) and pre[1].shape == pre[0].shape:
            logits = pre; STATS["pre"] += 1
        else:
            logits = dk.moe_router_logits(x2, dk.router_weights(gate), gate.e_score_correction_bias)
        if logits is None:
            return None
        if "gate_up_proj" in sw:
            routed_gate, routed_up = sw.gate_up_proj, None
        else:
            routed_gate, routed_up = sw.gate_proj, sw.up_proj
        fused = gate_up(x2, limit, routed_gate, routed_up, shared.gate_proj, shared.up_proj,
                        logits[0], logits[1], gate.top_k, gate.routed_scaling_factor, gate.norm_topk_prob)
        if fused is None:
            STATS["declined"] += 1
            return None
        act, shared_act, routes, weights = fused
        y = dk.moe_down_combine(act, routes, weights, sw.down_proj, shared.down_proj, shared_act=shared_act)
        if y is None:
            return None
        STATS["fused"] += 1
        return y.reshape(x.shape)

    cls._decode_select = _decode_select
    cls._tp_moerows = True
    return True
