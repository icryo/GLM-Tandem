# Linear-attention decode kernel with a sequence axis: one launch advances every concurrent
# sequence's recurrent state by one token (the stock kernel handles one sequence per launch).
import sys
import mlx.core as mx

_k = {}
_consts = {}


def install(layers):
    lay = next((l for l in layers if l.is_linear), None)
    if lay is None:
        return None
    cls = next(c for c in type(lay.self_attn).__mro__ if "_kda_kernel_step" in c.__dict__)
    Lm = sys.modules[cls.__module__]
    dk = Lm._decode_kernels
    src = dk._KDA_SOURCE
    old = "      y[t * QKV + h * DK + c] = static_cast<T>(o);"
    if src.count(old) != 1:
        return None
    src = src.replace(old, "      y[_vs * (TOK * QKV) + t * QKV + h * DK + c] = static_cast<T>(o);")
    names = ("proj", "conv_state", "state_in", "a_pre", "gate_pre", "conv_state_out", "state_out")
    offs = ("_vs * (TOK * PROJ_W)", "_vs * (9 * QKV)", "size_t(_vs) * (QKV * DK)", "_vs * (TOK * QKV)", "_vs * (TOK * QKV)",
            "_vs * (9 * QKV)", "size_t(_vs) * (QKV * DK)")
    pre = "  const int _vs = int(threadgroup_position_in_grid.y);\n" + "".join(
        f"#define {n} ({n} + {o})\n" for n, o in zip(names, offs))
    post = "".join(f"#undef {n}\n" for n in names)
    src_plain = pre + src + "\n" + post
    # Variant for two rows per sequence that also returns the state after the first row, so a
    # rejected second row can be dropped without running the kernel again.
    m1 = "      conv_state_out[r * NP + gc] = win[TOK + r];\n"
    m2 = "      if (thread_index_in_simdgroup == 0) {\n        ys[t][dv_idx] = static_cast<T>(out);\n      }\n    }\n"
    src_mid = None
    if src.count(m1) == 1 and src.count(m2) == 1:
        sm = src.replace(m1, m1 + "      conv_mid_out[r * NP + gc] = win[1 + r];\n")
        sm = sm.replace(m2, "      if (thread_index_in_simdgroup == 0) {\n        ys[t][dv_idx] = static_cast<T>(out);\n      }\n"
                            "      if (t == 0) {\n        for (int i = 0; i < n_per_t; ++i) {\n          auto s_idx = n_per_t * dk_idx + i;\n"
                            "          state_mid_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);\n        }\n      }\n    }\n")
        mid_names = ("conv_mid_out", "state_mid_out"); mid_offs = ("_vs * (9 * QKV)", "size_t(_vs) * (QKV * DK)")
        src_mid = (pre + "".join(f"#define {n} ({n} + {o})\n" for n, o in zip(mid_names, mid_offs)) + sm + "\n" + post
                   + "".join(f"#undef {n}\n" for n in mid_names))

    def kernel(pre_ag, gate5, mid=False):
        key = (pre_ag, gate5, mid)
        if key not in _k:
            inputs = ["proj", "conv_w", "a_log", "dt_bias", "norm_w", "consts", "conv_state", "state_in"]
            inputs += ["a_pre", "gate_pre"] if pre_ag else ["fb_w", "fb_s", "fb_b", "gb_w", "gb_s", "gb_b"]
            _k[key] = mx.fast.metal_kernel(
                name=f"tp_kda_seq_p{int(pre_ag)}{'_q5' if gate5 else ''}{'_mid' if mid else ''}",
                input_names=inputs,
                output_names=["y", "conv_state_out", "state_out"] + (["conv_mid_out", "state_mid_out"] if mid else []),
                header=dk._QMV_HEADER,
                source=dk._source(src_mid if mid else src_plain, HAS_CONV_STATE=1, HAS_STATE=1, PRE_AG=int(pre_ag), GATE5=int(gate5)))
        return _k[key]

    def step(attn, proj, convs, states, a_pre, gate_pre, tok=1, mid=False):
        """proj [1, B, W]; convs [B, 3, 3*QKV]; states [B, H, DK, DK]. Returns (y [1, B, QKV],
        convs, states) or None when not covered."""
        fg = attn.forget_gate
        heads, hd = attn.num_heads, attn.head_dim
        qkv = heads * hd
        B, width = proj.shape[1] // tok, proj.shape[2]
        _, _, v_end, fa_end, ga_end = attn._split_pts
        conv_w, norm_w = attn.conv1d.weight, attn.o_norm.weight
        if (proj.shape[0] != 1 or proj.dtype not in (mx.bfloat16, mx.float16) or hd != 128
                or conv_w.shape != (3 * qkv, 4, 1) or conv_w.dtype != proj.dtype
                or norm_w.shape != (hd,) or norm_w.dtype != proj.dtype
                or fg.A_log.size != heads or fg.dt_bias.size != qkv
                or convs.shape != (B, 3, 3 * qkv) or convs.dtype != proj.dtype
                or states.shape != (B, heads, hd, hd) or states.dtype != mx.float32):
            return None
        template = [("T", proj.dtype), ("TOK", tok), ("DK", hd), ("QKV", qkv), ("PROJ_W", width),
                    ("OFF_FA", v_end), ("OFF_GA", fa_end), ("OFF_B", ga_end)]
        pre_ag = a_pre is not None
        if pre_ag:
            if gate_pre is None or a_pre.shape != (1, B * tok, qkv) or gate_pre.shape != (1, B * tok, qkv) or a_pre.dtype != proj.dtype:
                return None
            extra = [a_pre, gate_pre]; template += [("BITS", 8), ("GS", 64)]; gate5 = False
        else:
            parts = [dk._affine_parts(m) for m in (fg.f_b_proj, attn.g_b_proj)]
            if any(p is None for p in parts):
                return None
            (fw, fs, fbias, fbits, fgs), (gw, gs_, gbias, gbits, ggs) = parts
            if (fbits, fgs) != (gbits, ggs) or fgs not in (32, 64, 128) or fbits not in (4, 5, 8) or (fbits == 5 and tok != 1):
                return None
            if fw.shape != (qkv, 128 * fbits // 32) or gw.shape != fw.shape or fs.dtype != proj.dtype or gs_.dtype != proj.dtype:
                return None
            extra = [fw, fs, fbias, gw, gs_, gbias]; template += [("BITS", fbits), ("GS", fgs)]; gate5 = fbits == 5
        ck = (hd, attn.o_norm.eps, fg.safe_gate_lower_bound)
        if ck not in _consts:
            _consts[ck] = mx.array([hd ** -0.5, 1e-6, attn.o_norm.eps, fg.safe_gate_lower_bound, 1.0 / hd], dtype=mx.float32)
        if mid and (tok != 2 or src_mid is None):
            return None
        shapes = [(1, B * tok, qkv), (B, 3, 3 * qkv), (B, heads, hd, hd)]
        dtypes = [proj.dtype, proj.dtype, mx.float32]
        if mid:
            shapes += shapes[1:]; dtypes += dtypes[1:]
        return tuple(kernel(pre_ag, gate5, mid)(
            inputs=[proj, conv_w, fg.A_log.reshape(-1), fg.dt_bias.reshape(-1), norm_w, _consts[ck], convs, states] + extra,
            template=template, grid=(1024 * heads, B, 1), threadgroup=(1024, 1, 1),
            output_shapes=shapes, output_dtypes=dtypes))

    return step
