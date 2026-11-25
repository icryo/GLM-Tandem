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

# Where the HC post kernel (compute-only, ~10 us) is encoded. "pre": before the branch, as before 2026-11-25.
# "hop" (default): after the branch's last kernel, attached to the partial that enters the cross-Mac sum, so it
# runs beside that memory-bound kernel instead of in front of the branch's first one. Same values either way
# (only a dependency is added): +1.75% decode, +2.2% on three-row cycles. The server's all_sum wrapper calls
# take_pending(); anything still pending after a branch is attached to the branch output (_late).
import os
ORDER = os.environ.get("TP_HC_ORDER", "hop")
PEND = []


def take_pending(x):
    """Make the partial entering the cross-Mac sum depend on the pending HC post output."""
    if PEND:
        STATS["hop"] += 1
        x = mx.depends(x, PEND[:])
        PEND.clear()
    return x


def _late(r):
    if PEND:
        STATS["late"] += 1
        r = mx.depends(r, PEND[:])
        PEND.clear()
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

    kern = {}

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

    def hc_pre(connection, norm, x):
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
        xn, h, post, comb, mm = hc_pre(self.ffn_hc, self.post_attention_layernorm, HCD(y, residual, post, comb, mm))
        return _late(self.mlp(xn)), h, post, comb, mm

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
        xn, h, post, comb, mm = hc_pre(self.attn_hc, self.input_layernorm, x)
        if gdn_sink is not None and self.is_linear:
            r = self.self_attn(xn, mask, cache, gdn_sink=gdn_sink)
        else:
            r = self.self_attn(xn, mask, cache)
        r = _late(r)
        key = (r.shape, r.dtype)
        seen = self.__dict__.setdefault("_tp_ffn_seen", set())
        if not self.compile_ffn or key not in seen:
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
    sys.modules["tp_hcrows_active"] = sys.modules[__name__]      # the server's all_sum wrapper looks this up
    for i, layer in enumerate(layers):
        layer.__dict__["_tp_trunk"] = True
        layer.__dict__["_tp_last"] = i + 1 == len(layers)
    return True
