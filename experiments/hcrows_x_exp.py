# Row-general fused HC path for short verify blocks ([1, L, ...], L <= 8).
#
# oMLX runs a one-token half-layer's HC chain as one dispatch (hc_pre_fused, with the
# previous expand folded in) plus hc_post_mm beside the branch. Verify blocks (L >= 2, or
# any call that captures recurrent state) fall back to three dependent dispatches per
# half-layer: hc_mix -> exact_hc_norm -> hc_expand. Here the one-token kernels get a row
# axis (grid y = row), so every row runs the one-token arithmetic and a verify block costs
# one dependent HC dispatch per half-layer.
import sys
import mlx.core as mx

STATS = {"layers": 0, "fallback": 0, "hop": 0, "late": 0}
ENABLED = [True]
# EXPERIMENT (window 6): where the HC post kernel is encoded. "pre" = before the branch (live behaviour);
# "hop" = after the branch's last kernel, attached to the partial that goes into the cross-Mac sum, so it can
# overlap that bus-bound kernel instead of preceding the first one. Same values either way.
import os
ORDER = os.environ.get("TP_HC_ORDER", "pre")
# EXPERIMENT 2: side work of a sparse-attention layer (the index pool upkeep: key norm, gate scores, window
# compression, pool append) is otherwise encoded lazily in the NEXT cycle, right in front of that layer's
# attention kernels. TP_SIDE=1 queues it behind the same layer's expert kernels instead (same values).
SIDE = os.environ.get("TP_SIDE", "0") == "1" and ORDER == "hop"
# EXPERIMENT 3: the MoE router's logits kernel (one dependent dispatch per expert layer, between the HC pre
# kernel and the gate/up kernel) folded into the tail of the FFN half's HC pre kernel: threadgroup 0 of each
# row already holds the normalized vector the router reads. Same per-lane products and shuffle ladder as
# glm5_router_logits_sigmoid, so the logits are bit-identical (TP_ROUTER_CHECK=1 compares them).
ROUTER = os.environ.get("TP_HC_ROUTER", "0") == "1"
ROUTER_CHECK = os.environ.get("TP_ROUTER_CHECK", "0") == "1"
CHECKS = []
PEND = []
def _pool_side(cache):
    try:
        pc = cache[1]
        if pc is None or getattr(pc, "pooled", None) is None: return []
        return [v for v in pc.state if isinstance(v, mx.array)]
    except Exception:
        return []
def take_pending(x):
    """Called by the all_sum wrapper: make the partial depend on a pending HC post output."""
    if PEND:
        STATS["hop"] += 1
        x = mx.depends(x, PEND[:]); PEND.clear()
    return x
def _late(r):
    if PEND:
        STATS["late"] += 1
        r = mx.depends(r, PEND[:]); PEND.clear()
    return r


def _rep(src, old, new, count=None):
    n = src.count(old)
    assert n >= 1 and (count is None or n == count), (old, n)
    return src.replace(old, new)


def install(layers, verbose=False):
    """Patch the decoder-layer class of ``layers`` (the trunk, in order)."""
    cls = type(layers[0])
    L = sys.modules[cls.__module__]
    dk = L._decode_kernels
    if getattr(cls, "_tp_hcrows", False):
        return True
    if dk.hc_comb_product() != "nax":
        return False

    ROW = "  const int _vr = int(threadgroup_position_in_grid.y);\n"
    pre = dk._HC_PRE_SOURCE
    pre = _rep(pre, "#define HVAL(e) glm_hc_expand_value<T, D>(y, mm, post, (e))",
               "#define HVAL(e) glm_hc_expand_value<T, D>(y + _vr * D, mm + _vr * (HC * D), post + _vr * HC, (e))", 1)
    pre = _rep(pre, "#define HVAL(e) x[(e)]", "#define HVAL(e) x[_vr * (HC * D) + (e)]", 1)
    pre = _rep(pre, "h_out[", "h_out[_vr * (HC * D) + ", 1)
    pre = _rep(pre, "mixes[row]", "mixes[_vr * ((2 + HC) * HC) + row]", 1)
    pre = _rep(pre, "((device T4*)normalized)[lid]", "((device T4*)(normalized + _vr * D))[lid]", 1)
    pre = ROW + pre

    post = dk._HC_POST_SOURCE
    post = _rep(post, "(mixes, scale[1]", "(mixes + _vr * ((2 + HC) * HC), scale[1]", 1)
    post = _rep(post, "post_out[lane]", "post_out[_vr * HC + lane]", 1)
    post = _rep(post, "(comb_out + lane * HC)", "(comb_out + _vr * (HC * HC) + lane * HC)", 1)
    post = _rep(post, "residual[r * D + ", "residual[_vr * (HC * D) + r * D + ", 2)
    post = _rep(post, "mm[r * D + col]", "mm[_vr * (HC * D) + r * D + col]", 1)
    post = ROW + post

    _qh = dk._QMV_HEADER
    _a = _qh.index("template <typename T>\ninline T glm_sigmoid(T x) {"); _b = _qh.index("}\n", _a) + 2
    pre_router_header = dk._HC_PRE_HEADER + "\n" + _qh[_a:_b]
    pre_router = _rep(pre, "#undef HVAL", r"""
  // --- MoE router logits of this row: glm5_router_logits_sigmoid's arithmetic, one expert per simdgroup pass ---
  threadgroup_barrier(mem_flags::mem_device);
  {
    const device T* xv = normalized + _vr * D;
    for (int e = int(simd_gid); e < E; e += 32) {
      const device auto* mat = rw + size_t(e) * D;
      float result = 0.0f;
      int bn = int(simd_lid) * 4;
      for (int i = 0; i < D / 128; ++i) {
        float v_coeff[4];
        for (int tn = 0; tn < 4; tn++) {
          v_coeff[tn] = static_cast<float>(xv[bn + tn]);
        }
        float inter[4];
        for (int tn = 0; tn < 4; tn++) {
          inter[tn] = static_cast<float>(mat[bn + tn]);
        }
        for (int tn = 0; tn < 4; tn++) {
          result += inter[tn] * v_coeff[tn];
        }
        bn += 128;
      }
      for (ushort sn = 16; sn >= 1; sn >>= 1) {
        result += simd_shuffle_down(result, sn);
      }
      if (simd_lid == 0) {
        float sgm = glm_sigmoid<float>(result);
        float biased = sgm + rbias[e];
        sig[size_t(_vr) * E + e] = sgm;
        biased_out[size_t(_vr) * E + e] = biased;
      }
    }
  }
#undef HVAL""", 1)

    kern = {}

    def k_pre_router():
        if "router" not in kern:
            kern["router"] = mx.fast.metal_kernel(
                name="tp_hc_pre_rows_deferred_router",
                input_names=["y", "mm", "post", "fn", "eps", "scale", "base", "norm_weight", "rw", "rbias"],
                output_names=["normalized", "mixes", "h_out", "sig", "biased_out"],
                header=pre_router_header,
                source=dk._source(pre_router, DEFERRED=1),
            )
        return kern["router"]

    def router_of(layer):
        """(router weights, bias, E) of an expert layer the fused multi-row routing covers, else None."""
        r = layer.__dict__.get("_tp_router")
        if r is None:
            r = False
            try:
                m = layer.mlp
                m = m.inner if "inner" in m else m
                gate = getattr(m, "gate", None)
                mr = sys.modules.get("moerows_x")
                if gate is not None and mr is not None and mr.ENABLED[0] and getattr(type(m), "_tp_moerows", False) and gate.n_group == 1:
                    w = dk.router_weights(gate); b = gate.e_score_correction_bias
                    if (w.ndim == 2 and w.shape[1] == 4096 and w.shape[0] % 32 == 0 and 32 <= w.shape[0] <= 1024
                            and w.dtype in (mx.bfloat16, mx.float32) and b.dtype == mx.float32 and b.shape == (w.shape[0],)):
                        r = (w, b, int(w.shape[0]))
            except Exception as e:
                print(f"[hc-router] layer not covered: {e!r}", flush=True)
            layer.__dict__["_tp_router"] = r
        return r or None

    def k_pre(deferred):
        if deferred not in kern:
            inputs = (["y", "mm", "post"] if deferred else ["x"]) + ["fn", "eps", "scale", "base", "norm_weight"]
            kern[deferred] = mx.fast.metal_kernel(
                name="tp_hc_pre_rows" + ("_deferred" if deferred else ""),
                input_names=inputs,
                output_names=["normalized", "mixes"] + (["h_out"] if deferred else []),
                header=dk._HC_PRE_HEADER,
                source=dk._source(pre, DEFERRED=int(deferred)),
            )
        return kern[deferred]

    def k_post():
        if "post" not in kern:
            kern["post"] = mx.fast.metal_kernel(
                name="tp_hc_post_rows",
                input_names=["mixes", "scale", "base", "residual"],
                output_names=["post_out", "comb_out", "mm"],
                header=dk._HC_POST_HEADER,
                source=post,
            )
        return kern["post"]

    eps_cache = {}

    def hc_pre(connection, norm, x, router=None):
        """x: [1, R, HC, D] or an HCD. Returns (xn [1,R,D], h, post, comb, mm)."""
        deferred = isinstance(x, HCD)
        if deferred:
            y = x.y
            dtype, R, D = y.dtype, y.shape[1], y.shape[-1]
            inputs = [y, x.mm, x.post]
        else:
            dtype, R, D = x.dtype, x.shape[1], x.shape[-1]
            inputs = [x]
        mix = connection.fn.shape[0]
        key = id(connection)
        if key not in eps_cache:
            eps_cache[key] = mx.array([connection.norm_eps], dtype=mx.float32)
        inputs += [dk.mix_weights(connection), eps_cache[key], connection.scale, connection.base, norm.weight]
        use_router = router is not None and deferred and D == 4096 and R >= 2
        if use_router:
            rw, rb, E = router
            outs = k_pre_router()(
                inputs=inputs + [rw, rb],
                template=[("T", dtype), ("HC", 4), ("D", D),
                          ("HC_EPS_INT", round(connection.hc_eps / 1e-9)),
                          ("NORM_EPS_INT", round(norm.eps / 1e-9)), ("E", E)],
                grid=(1024 * (mix // 4), R, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(1, R, D), (1, R, mix), (1, R, 4, D), (R, E), (R, E)],
                output_dtypes=[dtype, mx.float32, dtype, mx.float32, mx.float32],
            )
            STATS["router"] = STATS.get("router", 0) + 1
            mr = sys.modules.get("moerows_x")
            if mr is not None:
                mr.PRE[0] = (outs[3], outs[4])
            if ROUTER_CHECK and len(CHECKS) < 400:
                ref = dk.moe_router_logits(outs[0].reshape(R, D), rw, rb)
                if ref is not None:
                    CHECKS.append((outs[3], outs[4], ref[0], ref[1]))
        else:
            outs = k_pre(deferred)(
                inputs=inputs,
                template=[("T", dtype), ("HC", 4), ("D", D),
                          ("HC_EPS_INT", round(connection.hc_eps / 1e-9)),
                          ("NORM_EPS_INT", round(norm.eps / 1e-9))],
                grid=(1024 * (mix // 4), R, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(1, R, D), (1, R, mix)] + ([(1, R, 4, D)] if deferred else []),
                output_dtypes=[dtype, mx.float32] + ([dtype] if deferred else []),
            )
        xn, mixes = outs[0], outs[1]
        h = outs[2] if deferred else x
        simds = 8
        tiles = D // 32
        po, comb, mm = k_post()(
            inputs=[mixes, connection.scale, connection.base, h],
            template=[("T", dtype), ("HC", 4), ("D", D),
                      ("ITERS", int(connection.sinkhorn_iters)),
                      ("HC_EPS_INT", round(connection.hc_eps / 1e-9)), ("SIMDS", simds)],
            grid=(32 * simds * ((tiles + simds - 1) // simds), R, 1),
            threadgroup=(32 * simds, 1, 1),
            output_shapes=[(1, R, 4), (1, R, 4, 4), (R * 4 * D,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        if ORDER == "hop":
            PEND.append(po)
        else:
            xn = mx.depends(xn, [po])
        return xn, h, po, comb, mm

    class HCD(list):
        """Pending HC expand h = hc_expand(y, residual, post, comb); a list so the model
        loop's mx.async_eval(h) walks it."""

        def __init__(self, y, residual, post, comb, mm):
            super().__init__([y, residual, post, comb, mm])

        y = property(lambda s: s[0])
        residual = property(lambda s: s[1])
        post = property(lambda s: s[2])
        comb = property(lambda s: s[3])
        mm = property(lambda s: s[4])

        def arrays(self):
            return list(self)

        def materialize(self):
            return L._decode_hc_expand(self[0], self[1], self[2], self[3])

    L._HCDeferred = HCD

    def ffn_rows(self, y, residual, post, comb, mm):
        rt = router_of(self) if ROUTER else None
        xn, h, post, comb, mm = hc_pre(self.ffn_hc, self.post_attention_layernorm, HCD(y, residual, post, comb, mm), router=rt)
        try:
            out = self.mlp(xn)
        finally:
            mr = sys.modules.get("moerows_x")
            if mr is not None:
                mr.PRE[0] = None
        return _late(out), h, post, comb, mm

    def covered(self, x):
        if isinstance(x, HCD):
            dtype, width, shape = x.y.dtype, x.y.shape[-1], x.y.shape
        elif isinstance(x, mx.array) and x.ndim == 4:
            dtype, width, shape = x.dtype, x.shape[-1], x.shape
        else:
            return False
        if shape[0] != 1 or not 1 <= shape[1] <= L._DECODE_BLOCK:
            return False
        ok = getattr(self, "_tp_hc_ok", None)
        if ok is None or ok[0] != (dtype, width):
            ok = ((dtype, width),
                  L._hc_defer_ok(self.attn_hc, self.input_layernorm, dtype, width)
                  and L._hc_defer_ok(self.ffn_hc, self.post_attention_layernorm, dtype, width))
            self._tp_hc_ok = ok
        return ok[1]

    prev = cls.__call__

    def __call__(self, x, mask=None, cache=None, gdn_sink=None, defer=False):
        rows = x.y.shape[1] if isinstance(x, HCD) else (x.shape[1] if getattr(x, "ndim", 0) == 4 else 0)
        mine = ENABLED[0] and getattr(self, "_tp_trunk", False) and (gdn_sink is not None or rows > 1 or (isinstance(x, HCD) and not defer and not self._tp_last and rows > 1))
        if not mine or not covered(self, x):
            if isinstance(x, HCD) and (gdn_sink is not None or rows > 1):
                x = x.materialize()
            STATS["fallback"] += 1
            return prev(self, x, mask, cache, gdn_sink=gdn_sink, defer=defer)
        STATS["layers"] += 1
        if ORDER == "hop" and STATS["layers"] % 20000 == 0:
            print(f"[hc-order] {STATS}", flush=True)
        xn, h, post, comb, mm = hc_pre(self.attn_hc, self.input_layernorm, x)
        if gdn_sink is not None and self.is_linear:
            r = self.self_attn(xn, mask, cache, gdn_sink=gdn_sink)
        else:
            r = self.self_attn(xn, mask, cache)
        r = _late(r)
        side_now = False
        if SIDE and not self.is_linear and cache is not None:
            sd = _pool_side(cache)
            if sd:
                PEND.extend(sd); STATS["side"] = STATS.get("side", 0) + 1; side_now = True
        key = (r.shape, r.dtype)
        seen = self.__dict__.setdefault("_tp_ffn_seen", set())
        if not self.compile_ffn or key not in seen or side_now:
            # First call per shape runs eagerly (kernel self-checks cannot run in a trace).
            seen.add(key)
            out = HCD(*ffn_rows(self, r, h, post, comb, mm))
        else:
            fc = self.__dict__.get("_tp_ffn_c")
            if fc is None:
                fc = mx.compile(lambda *a: ffn_rows(self, *a))
                self.__dict__["_tp_ffn_c"] = fc
            out = HCD(*fc(r, h, post, comb, mm))
        return out if not self._tp_last else out.materialize()

    cls.__call__ = __call__
    cls._tp_hcrows = True
    for i, layer in enumerate(layers):
        layer.__dict__["_tp_trunk"] = True
        layer.__dict__["_tp_last"] = i + 1 == len(layers)
    return True
