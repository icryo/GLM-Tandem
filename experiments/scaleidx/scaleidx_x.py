# Scale index for the decode expert kernels (exact).
#
# The checkpoint's 4-bit expert matrices have only a few hundred distinct (scale, bias) pairs per matrix
# (kern/scalecount.py: 250-550 of 131,072 groups), so each group's two bf16 values (4 bytes) can be
# replaced by a uint16 index into a per-tensor table of packed pairs (2 bytes + a table of a few KB):
# 36 -> 34 bytes per 64 weights (-5.6% of routed-expert bytes). The values are the same, so every
# output bit is unchanged.
#
# This file: kernel variants of oMLX's decode MoE kernels (routed gate/up + SwiGLU, routed down +
# combine) whose routed loads read (index, table); a builder for (index, table) from scales/biases;
# and a test on one real layer (one Mac's half): bit-identity against the stock kernels and speed
# at 1, 2, 4 and 8 tokens.
#
# usage (on a Mac, oMLX python, NOT -I): python scaleidx_x.py [LAYER] [ITERS]
import os, sys, json, time
import numpy as np
import mlx.core as mx

from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk


def _rep(src, old, new, count=1):
    n = src.count(old)
    assert n == count, (old[:60], n)
    return src.replace(old, new)


# --- header: an indexed copy of glm_qmv_rows ------------------------------------------------------
_ROWS_IX = r"""
// glm_qmv_rows with the group's (scale, bias) read through a uint16 index into a table of packed
// pairs (scale bits << 16 | bias bits): the same values as the plain arrays.
template <typename T, int K, int group_size, int bits, int RPS>
inline void glm_qmv_rows_ix(
    const device uint8_t* ws,
    const device uint16_t* sidx,
    const device uint32_t* tab,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = glm_pack_factor<bits>();
  constexpr int bytes_per_pack = glm_bytes_per_pack<bits>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
  constexpr int in_vec_size_g = K / group_size;

  thread float x_thread[values_per_thread];
  ws += simd_lid * packs_per_thread * bytes_per_pack;
  sidx += simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

#if IX_PREFETCH
  // All of this lane's (scale, bias) pairs up front: independent loads, no dependent hop in the loop.
  constexpr int NB = K / block_size;
  uint pp[RPS * NB];
  for (int row = 0; row < RPS; row++) {
    for (int kb = 0; kb < NB; kb++) {
      pp[row * NB + kb] = tab[sidx[row * in_vec_size_g + kb * (block_size / group_size)]];
    }
  }
  int kb = 0;
#endif
  for (int k = 0; k < K; k += block_size) {
    float sum = glm_load_vector<T, values_per_thread, bits>(x, x_thread);
    for (int row = 0; row < RPS; row++) {
      const device uint8_t* wl = ws + row * in_vec_size_w;
#if IX_PREFETCH
      const uint p = pp[row * NB + kb];
#else
      const uint p = tab[sidx[row * in_vec_size_g]];
#endif
      float s = static_cast<float>(as_type<T>(ushort(p >> 16)));
      float b = static_cast<float>(as_type<T>(ushort(p & 0xffffu)));
      result[row] += glm_qdot<values_per_thread, bits>(wl, x_thread, s, b, sum);
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sidx += block_size / group_size;
    x += block_size;
#if IX_PREFETCH
    kb++;
#endif
  }
}
"""
IX_PREFETCH = int(os.environ.get("IX_PREFETCH", "1"))
HEADER_IX = dk._QMV_HEADER + f"#define IX_PREFETCH {IX_PREFETCH}\n" + _ROWS_IX

# routed gate/up: gate_s/up_s become uint16 index arrays (same layout as scales), gate_b/up_b the tables
GATE_UP_IX = _rep(dk._GATE_UP_SOURCE,
                  """    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
        gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
        up_b + urow0 * G, xr, simd_lid, u_res);""",
                  """    glm_qmv_rows_ix<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
        gate_b, xr, simd_lid, g_res);
    glm_qmv_rows_ix<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
        up_b, xr, simd_lid, u_res);""")
DOWN_IX = _rep(dk._DOWN_SOURCE,
               """    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
        down_b + row0 * G, act + (size_t(token) * RT + r) * K, simd_lid, res);""",
               """    glm_qmv_rows_ix<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
        down_b, act + (size_t(token) * RT + r) * K, simd_lid, res);""")

_K = {}


def gate_up_kernel(ix, has_shared=False, slot_major=False):
    key = ("gu", ix, has_shared, slot_major)
    if key not in _K:
        inputs = ["x", "indices", "limit", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"]
        if has_shared:
            inputs += ["sh_gate_w", "sh_gate_s", "sh_gate_b", "sh_up_w", "sh_up_s", "sh_up_b"]
        _K[key] = mx.fast.metal_kernel(
            name="tp_moe_gate_up" + (f"_ix{IX_PREFETCH}" if ix else "_ref") + ("_sh" if has_shared else "") + ("_sm" if slot_major else ""),
            input_names=inputs, output_names=["out"],
            header=HEADER_IX if ix else dk._QMV_HEADER,
            source=dk._source(GATE_UP_IX if ix else dk._GATE_UP_SOURCE, HAS_SHARED=int(has_shared), SHARED_WIDE=0,
                              SLOT_MAJOR=int(slot_major), SELECT=0))
    return _K[key]


def down_kernel(ix, slot_major=False):
    key = ("dn", ix, slot_major)
    if key not in _K:
        _K[key] = mx.fast.metal_kernel(
            name="tp_moe_down" + (f"_ix{IX_PREFETCH}" if ix else "_ref") + ("_sm" if slot_major else ""),
            input_names=["act", "indices", "scores", "down_w", "down_s", "down_b"], output_names=["out"],
            header=HEADER_IX if ix else dk._QMV_HEADER,
            source=dk._source(DOWN_IX if ix else dk._DOWN_SOURCE, HAS_SHARED=0, ADD_SHARED_Y=0,
                              SLOT_MAJOR=int(slot_major), SHARED_WIDE_DOWN=0))
    return _K[key]


def build_index(scales, biases):
    """(uint16 index with the layout of scales, uint32 table of packed pairs) or None if > 65,536 pairs."""
    s = np.array(scales.view(mx.uint16)).astype(np.uint32)
    b = np.array(biases.view(mx.uint16)).astype(np.uint32)
    tab, inv = np.unique((s << 16) | b, return_inverse=True)
    if tab.size > 65536:
        return None
    return mx.array(inv.reshape(scales.shape).astype(np.uint16)), mx.array(tab.astype(np.uint32))


def gate_up(x, idx, limit, g, u, ix, rps=4, nsg=2):
    """g, u = (w, s, b) with s, b either bf16 scales/biases or (uint16 index, uint32 table)."""
    T, K = x.shape
    E, N, _ = g[0].shape
    topk = idx.shape[1]
    sm = dk._slot_major(T)
    tiles = N // (rps * nsg); slots = T * topk
    return gate_up_kernel(ix, False, sm)(
        inputs=[x, idx, mx.array([limit], dtype=mx.float32), *g, *u],
        template=[("T", x.dtype), ("K", K), ("N", N), ("TOPK", topk), ("RBITS", 4), ("RGS", 64), ("RPS", rps),
                  ("NSG", nsg), ("ESTRIDE", N), ("UP_OFF", 0)],
        grid=(32, slots * nsg, tiles) if sm else (32, tiles * nsg, slots), threadgroup=(32, nsg, 1),
        output_shapes=[(T, topk, N)], output_dtypes=[x.dtype])[0]


def down(act, idx, scores, d, ix, rps=4, nsg=2):
    T, rt, K = act.shape
    E, N, _ = d[0].shape
    topk = idx.shape[1]
    sm = dk._slot_major(T)
    tiles = N // (rps * nsg)
    return down_kernel(ix, sm)(
        inputs=[act, idx, scores, *d],
        template=[("T", act.dtype), ("K", K), ("N", N), ("TOPK", topk), ("RBITS", 4), ("RGS", 64), ("RPS", rps), ("NSG", nsg)],
        grid=(32, T * nsg, tiles) if sm else (32, tiles * nsg, T), threadgroup=(32, nsg, 1),
        output_shapes=[(T, N)], output_dtypes=[act.dtype])[0]


def _load_layer(MP, L, E=288, half=1024):
    """Rank 0's half of layer L's routed experts: gate/up rows [0, half), down input columns [0, half)."""
    wm = json.load(open(os.path.join(MP, "model.safetensors.index.json")))["weight_map"]
    files = {}
    def get(n):
        f = wm[n]
        if f not in files: files[f] = mx.load(os.path.join(MP, f))
        return files[f][n]
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        parts = []
        for field in ("weight", "scales", "biases"):
            arr = []
            for e in range(E):
                a = get(f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}.{field}")
                if proj != "down_proj":
                    a = a[:half]
                else:
                    a = a[:, : (half // 8 if field == "weight" else half // 64)]
                arr.append(a)
            t = mx.stack(arr); mx.eval(t); parts.append(t)
        out[proj] = tuple(parts)
    files.clear()
    return out


def _bench(f, iters):
    """f(i) -> array; i cycles through different expert selections so the weights come from DRAM."""
    for i in range(3): mx.eval(f(i))
    t0 = time.perf_counter()
    outs = [f(i) for i in range(iters)]
    mx.eval(outs)
    return (time.perf_counter() - t0) / iters * 1e6


if __name__ == "__main__":
    MP = "/Users/a/models/GLM-5.3-Flash-TF-4bit-MTP"
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time()
    W = _load_layer(MP, L)
    print(f"layer {L} loaded in {time.time() - t0:.1f} s:", {k: tuple(v[0].shape) for k, v in W.items()}, flush=True)
    IX = {}
    for k, (w, s, b) in W.items():
        t1 = time.time(); r = build_index(s, b)
        assert r is not None
        IX[k] = (w, r[0], r[1])
        mb_old = (s.nbytes + b.nbytes) / 2**20; mb_new = (r[0].nbytes + r[1].nbytes) / 2**20
        print(f"  {k}: table {r[1].size} pairs; scales+biases {mb_old:.1f} MiB -> index+table {mb_new:.1f} MiB; weights {w.nbytes / 2**20:.0f} MiB; built in {time.time() - t1:.1f} s", flush=True)
    rng = np.random.default_rng(L)
    E = W["gate_proj"][0].shape[0]; K = 4096; LIM = 10.0
    print(f"{'tokens':>6} | {'gate/up us ref':>14} {'ix':>7} {'gain':>6} | {'down us ref':>11} {'ix':>7} {'gain':>6} | identical gu / down")
    for T in (1, 2, 4, 8):
        x = mx.array((rng.standard_normal((T, K)) * 0.35).astype(np.float32)).astype(mx.bfloat16)
        idx = mx.array(np.stack([rng.choice(E, 8, replace=False) for _ in range(T)]).astype(np.uint32))
        scores = mx.array(rng.random((T, 8)).astype(np.float32))
        a_ref = gate_up(x, idx, LIM, W["gate_proj"], W["up_proj"], False)
        a_ix = gate_up(x, idx, LIM, IX["gate_proj"], IX["up_proj"], True)
        act = a_ref
        d_ref = down(act, idx, scores, W["down_proj"], False)
        d_ix = down(act, idx, scores, IX["down_proj"], True)
        mx.eval(a_ref, a_ix, d_ref, d_ix)
        same_gu = bool(mx.array_equal(a_ref, a_ix).item()); same_d = bool(mx.array_equal(d_ref, d_ix).item())
        sel = [mx.array(np.stack([rng.choice(E, 8, replace=False) for _ in range(T)]).astype(np.uint32)) for _ in range(64)]
        acts = [gate_up(x, s_, LIM, W["gate_proj"], W["up_proj"], False) for s_ in sel]
        mx.eval(sel, acts)
        res = {}
        for rep in range(int(os.environ.get("REPS", "4"))):   # alternate ref / ix, keep the best of each
            for name, f in (("gr", lambda i: gate_up(x, sel[i % 64], LIM, W["gate_proj"], W["up_proj"], False)),
                            ("gi", lambda i: gate_up(x, sel[i % 64], LIM, IX["gate_proj"], IX["up_proj"], True)),
                            ("dr", lambda i: down(acts[i % 64], sel[i % 64], scores, W["down_proj"], False)),
                            ("di", lambda i: down(acts[i % 64], sel[i % 64], scores, IX["down_proj"], True))):
                res[name] = min(res.get(name, 1e9), _bench(f, ITERS))
        tg_r, tg_i, td_r, td_i = res["gr"], res["gi"], res["dr"], res["di"]
        print(f"{T:6d} | {tg_r:14.1f} {tg_i:7.1f} {100 * (1 - tg_i / tg_r):5.1f}% | {td_r:11.1f} {td_i:7.1f} {100 * (1 - td_i / td_r):5.1f}% | {same_gu} / {same_d}", flush=True)
