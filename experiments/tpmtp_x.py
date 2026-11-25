# Tensor-parallel + Lightning MTP (depth 1) prototype. Greedy. One process per Mac.
# Measurement harness (run on both Macs with scripts/runx.sh; it never touches the server's files):
#   VARIANT=pipecost ROWS=2 [ABLATE=mlp|moe_core|mlp_nosum|kda|kda_nosum|sparse|sparse_nosum|...]   pass time by removal
#   VARIANT=pipecost,nosum | pipecost,cpuhop | ...,allranks      no sums / CPU hop without network / print both ranks
#   VARIANT=kpass MLX_KPROF=1 MLX_KPROF_FILE=...                 per-kernel profile of plain passes only
#   VARIANT=prefillcost [ABLATE=...] PF_BLOCK=2048               prompt-reading budget by removal
#   VARIANT=wf PF_BLOCK=3072                                     interleaved prompt reading (wf_forward) vs plain, with an exactness check
#   PROMPT_WORDS=N synthetic context of N dictionary words; TP_HCX=1 loads hcrows_x.py / moerows_x.py (the *_exp copies)
#   VARIANT=rcheck TP_HCX=1 TP_HC_ROUTER=1 TP_ROUTER_CHECK=1     bit-exactness of the router folded into the HC kernel
import os, sys, time, collections
from types import SimpleNamespace
import mlx.core as mx
VAR = set(filter(None, os.environ.get("VARIANT", "").split(",")))
g = mx.distributed.init(strict=True, backend="jaccl"); R = g.rank()
def log(*a):
    if R == 0 or "allranks" in VAR: print(f"[r{R}]", *a, flush=True)
_all_sum = mx.distributed.all_sum
def all_sum(x, *a, **k):
    _hx = sys.modules.get("hcrows_x")
    if _hx is not None and _hx.PEND: x = _hx.take_pending(x)
    if "nosum" in VAR: return x + x
    if "cpuhop" in VAR:                                # GPU->CPU->GPU hop, no network
        y = mx.add(x, x, stream=mx.cpu); WF_CAP[0] = y
        return mx.squeeze(mx.expand_dims(y, 0), 0)
    y = _all_sum(x, *a, **k)
    WF_CAP[0] = y
    return mx.squeeze(mx.expand_dims(y, 0), 0)      # lands on the GPU stream without dispatching a kernel
WF_CAP = [None]
mx.distributed.all_sum = all_sum
def bcast(x):                       # rank 0's value, on every rank, as a lazy GPU-stream array
    return all_sum(x if R == 0 else x * 0)        # x * 0 keeps the same graph (and collectives) on every rank

def bcast2(x):                      # (host-readable CPU-stream result, GPU-stream array for the graph)
    y = _all_sum(x if R == 0 else x * 0)
    return y, mx.squeeze(mx.expand_dims(y, 0), 0)
FASTREAD = "slowread" not in VAR

path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/models/GLM-5.3-Flash-oQ4e-MTP"))
from omlx.utils.model_loading import maybe_apply_pre_load_patches
maybe_apply_pre_load_patches(path, SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=None, mtp_adaptive_max_depth=None))
from omlx.cluster.progressive_loading import progressive_sharded_load
# oMLX bug workaround: the rank model enables "MTP attach" only while constructing the module and
# restores it before the checkpoint is sanitized, so sanitize drops every mtp.* tensor and the head
# stays at its initial values (0% draft acceptance). Keep attach on for the whole load.
from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled as _keep_attach
_keep_attach(True)
if "shardhead" in VAR:
    # Attach the MTP head on every rank (stock: rank zero only) so it can be sharded like a layer.
    import omlx.patches.glm5_next_mlx_lm.model as _RM
    from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled as _set_attach, is_mtp_attach_enabled as _is_attach
    def _language_model_all_ranks(args):
        _RM.glm5_next_vlm_runtime.apply()
        prev = _is_attach(); _set_attach(True)
        try: lmod = _RM._language.LanguageModel(args.text_config, args)
        finally: _set_attach(prev)
        lmod.model._omlx_output_in_cache = True
        return lmod
    _RM._language_model = _language_model_all_ranks
if os.environ.get("TP_UNSHARD_SPARSE") == "1":
    # Experiment: keep the sparse-attention layers whole on every rank (no split, no exchange for them).
    import inspect, textwrap
    from omlx.cluster import tensor_strategies as _TSx
    _src = textwrap.dedent(inspect.getsource(_TSx._shard_glm5_next))
    _src = _src[_src.index("def _shard_glm5_next"):]
    _a = """        else:
            _require_divisible(attention.num_heads, size, "attention heads")
            heads = attention.num_heads // size
            shard_inplace(attention.q_b_proj, _glm5_rows, group=group)
            shard_inplace(attention.o_proj, _glm5_columns, group=group)
            for projection in (attention.embed_q, attention.unembed_out):
                projection.apply(lambda w: mx.contiguous(mx.split(w, size)[rank]))
            attention.num_heads = heads
        # The latent projections, the indexer and the router stay replicated.
        _sum_outputs(attention, group, mx)
"""
    assert _a in _src
    _src = _src.replace(_a, """        # The latent projections, the indexer and the router stay replicated.
        if layer.is_linear:
            _sum_outputs(attention, group, mx)
""")
    _ns = {}
    exec(compile(_src, "<unshard sparse>", "exec"), _TSx.__dict__, _ns)
    _fn = _ns["_shard_glm5_next"]
    _fn._omlx_tensor_strategy = _TSx._shard_glm5_next._omlx_tensor_strategy
    for _k, _v in list(_TSx._ADAPTERS.items()):
        if _v is _TSx._shard_glm5_next: _TSx._ADAPTERS[_k] = _fn
    _TSx._shard_glm5_next = _fn
    log("sparse-attention layers left unsplit")
t0 = time.time()
model, tok = progressive_sharded_load(path, tensor_group=g)
lm = model.language_model
if os.environ.get("TP_HCROWS") == "1":
    if os.environ.get("TP_HCX") == "1":
        import hcrows_x as hcrows; sys.modules["hcrows"] = hcrows; log("hcrows_x order =", hcrows.ORDER)
    else:
        import hcrows
    log(f"hcrows installed: {hcrows.install(list(model.layers))}")
if os.environ.get("TP_MOEROWS") == "1":
    if os.environ.get("TP_HCX") == "1":
        import moerows_x as moerows; sys.modules["moerows"] = moerows
    else:
        import moerows
    log(f"moerows installed: {moerows.install(list(model.layers))}")
if os.environ.get("ROWEXACT"):
    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed as _arm
    _arm(True, row_exact=os.environ["ROWEXACT"] == "1"); log("verify routes armed, row_exact =", os.environ["ROWEXACT"] == "1")
if os.environ.get("TP_SAROWS") == "1":
    import sarows; log(f"sarows installed: {sarows.install(list(model.layers))}")
if os.environ.get("TP_SMALLQ") is not None:
    # Experiment: sparse-attention decode path. 0 = general path (tensor-unit indexer scores + fused select +
    # gathered attention) instead of the small-query path (small_scores + native top-k + split-K decode).
    _Lq = sys.modules[type(model.layers[0]).__module__]
    _Lq._SMALL_QUERY_INDEXER = os.environ["TP_SMALLQ"] == "1"; log("small-query indexer:", _Lq._SMALL_QUERY_INDEXER)
has_head = getattr(lm, "mtp", None) is not None
def load_head_weights(model, path):
    """oMLX's rank loader leaves language_model.mtp.* at its initial values; load it from the checkpoint."""
    import json
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    keys = [k for k in idx if k.startswith("language_model.mtp.")]
    files = {}
    items = []
    for k in keys:
        f = idx[k]
        if f not in files: files[f] = mx.load(os.path.join(path, f))
        items.append((k, files[f][k]))
    model.load_weights(items, strict=False)
    return len(items)
if has_head:
    n_head = load_head_weights(model, path); log("loaded", n_head, "MTP head tensors from checkpoint")
if has_head and "shardhead" in VAR:
    from omlx.cluster import tensor_strategies as _TS
    for b in lm.mtp:
        b.block.is_linear = False
        _TS._shard_glm5_next(SimpleNamespace(layers=[b.block]), g, mx, None)
if has_head: mx.eval([b.parameters() for b in lm.mtp])
if has_head and "dbg" in VAR:
    from mlx.utils import tree_flatten
    fl = tree_flatten([b.parameters() for b in lm.mtp])
    log("head tensors", len(fl), "bytes", sum(v.nbytes for _, v in fl) / 2**20, "MiB;", [(k, tuple(v.shape), str(v.dtype)) for k, v in fl[:6]])
    w = lm.mtp[0].eh_proj.weight if hasattr(lm.mtp[0].eh_proj, "weight") else None
    if w is not None: log("eh_proj", w.shape, w.dtype, "absmean", mx.abs(w.astype(mx.float32)).mean().item() if w.dtype != mx.uint32 else "quantized")
log(f"loaded in {time.time()-t0:.1f}s active={mx.get_active_memory()/2**30:.1f} GiB head={has_head}")

SHARDLM = "shardlm" in VAR
_draft_ctx = [False]
if SHARDLM:
    # Vocabulary head split by rows: each rank computes half of the logits, one all_gather joins them.
    from omlx.patches.glm5_next_mlx_lm import model as _MH
    _Lg = _MH._language
    _hd = lm.lm_head
    V = _hd.weight.shape[0]; half = V // g.size()
    for name in ("weight", "scales", "biases"):
        setattr(_hd, name, mx.contiguous(getattr(_hd, name)[R * half:(R + 1) * half]))
    mx.eval(_hd.parameters())
    _lf = _Lg.linear_forward
    def _linear_forward(layer, x):
        if layer is not _hd: return _lf(layer, x)
        if _draft_ctx[0]:                                  # x exists on rank 0 only: share it first
            x = bcast(x.astype(mx.bfloat16))
        y = _lf(layer, x)                                  # [..., V/ranks]
        yt = mx.contiguous(mx.moveaxis(y, -1, 0))
        gt = mx.distributed.all_gather(yt)
        gt = mx.squeeze(mx.expand_dims(gt, 0), 0)
        return mx.moveaxis(gt, 0, -1)
    _Lg.linear_forward = _linear_forward
    log("lm_head sharded:", tuple(_hd.weight.shape))

msgs = [{"role": "user", "content": os.environ.get("PROMPT", "Write a detailed essay on the history of the transistor.")}]
if os.environ.get("PROMPT_TOKENS"):
    _txt = open("/Applications/oMLX.app/Contents/Resources/omlx/cluster/routes.py").read()[: int(os.environ["PROMPT_TOKENS"]) * 3]
    msgs = [{"role": "user", "content": "Here is some code:\n" + _txt + "\n\nWrite a detailed essay on what this code does."}]
if os.environ.get("PROMPT_WORDS"):      # synthetic long context of seeded dictionary words
    import random as _rnd
    _rr = _rnd.Random(11); _ww = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
    msgs = [{"role": "user", "content": "Here are some notes:\n" + " ".join(_rr.choice(_ww) for _ in range(int(os.environ["PROMPT_WORDS"]))) + "\n\nWrite a detailed essay about these notes."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
log("prompt tokens:", len(ids))
PRIME = "noprime" not in VAR

DEPTH = int(os.environ.get("DEPTH", "1"))
if os.environ.get("EVALEVERY"):
    from omlx.patches.glm5_next_mlx_lm import model as _MM
    _MM._language._DECODE_EVAL_EVERY = int(os.environ["EVALEVERY"]); log("DECODE_EVAL_EVERY", _MM._language._DECODE_EVAL_EVERY)
TM = collections.Counter()

PREK = int(os.environ.get("PREK", "3"))
if "prefix" in VAR:
    # Split trunk forward: the first PREK layers of the NEXT cycle are built and queued before the
    # host reads this cycle's accept decision (their inputs and recurrent states are selected
    # in-graph), so the GPU has work while the host handles the decision.
    from omlx.patches.glm5_next_mlx_lm import model as _MHp
    _Lp = _MHp._language
    gm = lm.model
    assert all(l.is_linear for l in gm.pipeline_layers[:PREK]), "prefix layers must be linear-attention"
    _FL = {PREK + 1, PREK + 2, PREK + 4, PREK + 8}
    def run_prefix(inp, cache, sink):
        h = gm.embed_tokens(inp)
        ssm_mask = _Lp.create_ssm_mask(h, cache[0])
        h4 = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], gm.hc_mult, h.shape[2])))
        for i in range(PREK):
            h4 = gm.pipeline_layers[i](h4, mask=ssm_mask, cache=cache[i], gdn_sink=sink)
        return h, h4
    _gcls = type(gm); _gorig = _gcls.__call__
    def _gcall(self, inputs, cache=None, inputs_embeds=None, gdn_sink=None, hidden_sink=None, **kw):
        st = self.__dict__.pop("_tp_pre", None)
        if st is None:
            return _gorig(self, inputs, cache=cache, inputs_embeds=inputs_embeds, gdn_sink=gdn_sink, hidden_sink=hidden_sink, **kw)
        h_emb, h, pre_sink = st
        layers = self.pipeline_layers; n = len(layers)
        fa_idx = next(i for i, l in enumerate(layers) if not l.is_linear)
        fa_mask = _Lp.create_attention_mask(h_emb, cache[fa_idx][0], return_array=True)
        ssm_mask = _Lp.create_ssm_mask(h_emb, cache[0])
        gdn_sink.extend(pre_sink)
        for i in range(PREK, n):
            layer = layers[i]
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache[i], gdn_sink=gdn_sink)
            if ((i + 1) in _FL or (i + 1) % 8 == 0) and i + 1 < n:
                mx.async_eval(h.arrays() if isinstance(h, _Lp._HCDeferred) else h)
        h = h.mean(axis=2)
        if hidden_sink is not None: hidden_sink.append(h)
        return self.norm(h)
    _gcls.__call__ = _gcall
from omlx.patches.mlx_lm_mtp.batch_generator import _clone_mtp_head_cache
def generate(n):
    cache = model.make_cache()
    mtp_cache = lm.make_mtp_cache() if has_head else None
    p_all = mx.array(ids)[None]
    cut = max(0, p_all.shape[1] - 64)
    for i in range(0, cut, 1024):                                # long prompts: plain chunked prefill
        model(p_all[:, i:min(i + 1024, cut)], cache=cache); mx.eval([c.state for c in cache]); mx.clear_cache()
    p = p_all[:, cut:]
    out = model(p, cache=cache, return_hidden=True)
    h = lm.model.norm(out.hidden_states[-1])
    x1 = bcast(mx.argmax(out.logits[:, -1], axis=-1).astype(mx.int32))      # [1]
    def drafts_for(rows, next_ids):
        """Fold committed (hidden, next token) rows into the head, then chain DEPTH drafts."""
        if SHARDLM and not has_head:
            # Mirror rank zero's head: receive its normed hidden, compute this rank's half of the logits.
            _draft_ctx[0] = True
            try: lg = _linear_forward(_hd, mx.zeros((1, 1, rows.shape[-1]), dtype=mx.bfloat16) + next_ids[0, -1:].astype(mx.bfloat16) * 0)
            finally: _draft_ctx[0] = False
            return mx.argmax(lg[:, -1], axis=-1).astype(mx.int32)
        if not has_head or "nohead" in VAR: return mx.zeros((DEPTH,), dtype=mx.int32) + next_ids[0, -1:] * 0
        _draft_ctx[0] = SHARDLM
        try: lg, hm = lm.mtp_forward(rows, next_ids, mtp_cache, return_hidden=True, logits_keep=1)
        finally: _draft_ctx[0] = False
        d = mx.argmax(lg[:, -1], axis=-1).astype(mx.int32); ds = [d]
        if DEPTH > 1:
            cc = _clone_mtp_head_cache(mtp_cache); hc = hm[:, -1:]
            for _ in range(DEPTH - 1):
                lg, hh = lm.mtp_forward(hc, ds[-1][None], cc, return_hidden=True)
                ds.append(mx.argmax(lg[:, -1], axis=-1).astype(mx.int32)); hc = hh[:, -1:]
        return mx.concatenate(ds)
    dr = bcast(drafts_for(h, mx.concatenate([p[:, 1:], x1[None].astype(p.dtype)], axis=1)))   # [DEPTH]
    mx.eval(x1, dr, [c.state for c in cache])
    a = x1; toks = [int(x1.item())]
    hist = collections.Counter(); cyc = 0; t = time.perf_counter()
    R_ = DEPTH + 1; pre = None
    while len(toks) < n:
        _t0 = time.perf_counter()
        inp = mx.concatenate([a, dr])[None]                                   # [1, 1+DEPTH]
        if pre is not None: lm.model.__dict__["_tp_pre"] = pre; pre = None
        out = model(inp, cache=cache, return_hidden=True)
        tt = mx.argmax(out.logits, axis=-1).astype(mx.int32)[0]               # [R_]
        ok = (tt[:DEPTH] == dr).astype(mx.int32)
        acc = mx.cumprod(ok).sum(keepdims=True)
        h = lm.model.norm(out.hidden_states[-1]) if has_head else out.hidden_states[-1]
        if "prefix" in VAR and DEPTH == 1:
            if has_head:
                _draft_ctx[0] = SHARDLM
                try: lg = lm.mtp_forward(h, tt[None], mtp_cache)
                finally: _draft_ctx[0] = False
            elif SHARDLM:
                _draft_ctx[0] = True
                try: lg = _linear_forward(_hd, mx.zeros((1, 2, h.shape[-1]), dtype=mx.bfloat16) + tt[None, :, None].astype(mx.bfloat16) * 0)
                finally: _draft_ctx[0] = False
            both = mx.argmax(lg[0], axis=-1).astype(mx.int32) if (has_head or SHARDLM) else mx.zeros((2,), dtype=mx.int32) + tt * 0
            info_raw, info = bcast2(mx.concatenate([tt, acc, both]))
            mx.async_eval(info_raw, info)
            sel = info[2:3]
            a_n = mx.take(info[0:2], sel); d_n = mx.take(info[3:5], sel)        # next token, its draft
            keep = sel > 0
            for i in range(PREK):                                                # recurrent state after the kept rows
                c = cache[i]; s1 = out.gdn_states[i].replay(1)
                c[0] = mx.where(keep, c[0], s1[0]); c[1] = mx.where(keep, c[1], s1[1])
            sink_n = []
            h_emb, h4 = run_prefix(mx.concatenate([a_n, d_n])[None], cache, sink_n)
            mx.async_eval(h4.arrays() if hasattr(h4, "arrays") else h4)
            _t1 = time.perf_counter()
            vals = (info_raw if FASTREAD else info).tolist()
            _t2 = time.perf_counter()
            m = vals[2]; cyc += 1; k = m + 1; hist[m] += 1
            model.rollback_speculative_cache(cache[PREK:], out.gdn_states[PREK:], m, R_)
            if m == 0:
                for i in range(PREK):
                    c = cache[i]
                    if getattr(c, "lengths", None) is not None or getattr(c, "left_padding", None) is not None: c.advance(-1)
                if has_head:
                    for c in mtp_cache: c.trim(1)
            toks += vals[:k]
            a = a_n; dr = d_n; pre = (h_emb, h4, sink_n)
        elif "ingraph" in VAR and DEPTH == 1:
            # Draft for both outcomes before the decision is known: one head fold over both rows.
            if has_head:
                lg = lm.mtp_forward(h, tt[None], mtp_cache)
                both = mx.argmax(lg[0], axis=-1).astype(mx.int32)            # [2]: after t1 (reject), after t2 (accept)
            else:
                both = mx.zeros((2,), dtype=mx.int32) + tt * 0
            info = bcast(mx.concatenate([tt, acc, both]))
            mx.async_eval(info)
            _t1 = time.perf_counter()
            vals = info.tolist()
            _t2 = time.perf_counter()
            accv = vals[2]; cyc += 1
            m = model.mtp_clamp_accept(cache, accv, DEPTH)
            model.rollback_speculative_cache(cache, out.gdn_states, m, R_)
            k = m + 1; hist[m] += 1
            toks += vals[:k]
            if k == 1 and has_head:
                for c in mtp_cache: c.trim(1)
            dr = info[3 + m : 4 + m]
            a = info[k - 1 : k]
        else:
            info_raw, info = bcast2(mx.concatenate([tt, acc]))
            mx.async_eval(info_raw, info, h)
            _t1 = time.perf_counter()
            vals = (info_raw if FASTREAD else info).tolist()                      # the one sync per cycle
            _t2 = time.perf_counter()
            accv = vals[-1]; cyc += 1
            m = model.mtp_clamp_accept(cache, accv, DEPTH)
            model.rollback_speculative_cache(cache, out.gdn_states, m, R_)
            k = m + 1; hist[m] += 1
            toks += vals[:k]
            dr = bcast(drafts_for(h[:, :k], info[:k][None]))
            a = info[k - 1 : k]
            mx.async_eval(dr)
        _t3 = time.perf_counter()
        TM["build+encode verify"] += _t1 - _t0; TM["blocked on result"] += _t2 - _t1; TM["rollback+draft build"] += _t3 - _t2; TM["n"] += 1
    mx.eval(dr)
    dt = time.perf_counter() - t
    return toks, (len(toks) - 1) / dt, hist, cyc, dt
n = int(os.environ.get("NTOK", "200"))
generate(16)
if "kprof" in VAR:
    _kr = mx.fast.metal_kernel(name="kprof_reset", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    _kd = mx.fast.metal_kernel(name="kprof_dump", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    _one = mx.zeros((1,)); mx.eval(_one)
    def _mark(k): mx.eval(k(inputs=[_one], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])[0])
    generate(12); _mark(_kr)
    _t, _tps, _h, _c, _dt = generate(n); _mark(_kd)
    log(f"profiled: {len(_t)} tokens in {_c} cycles")
    sys.exit(0)
for rep in range(int(os.environ.get("REPS", "2"))):
    TM.clear(); toks, tps, hist, cyc, dt = generate(n)
    if TM["n"]: log("  per cycle ms: " + ", ".join(f"{k} {1e3*v/TM['n']:.2f}" for k, v in TM.items() if k != "n"))
    log(f"MTP depth {DEPTH}: {tps:6.1f} tok/s  {len(toks)/max(cyc,1):.2f} tok/cycle  {1000*dt/max(cyc,1):.2f} ms/cycle  accepted-drafts histogram {dict(sorted(hist.items()))}")
if R == 0 and "toks" in dir(): print(tok.decode(toks)[:300].replace("\n", " "), flush=True)

if "pathcost" in VAR:
    # Timing only: cost of the forward by code path and row count (tokens are arbitrary but valid ids).
    def loop(rows, verify, steps=150):
        cache = model.make_cache()
        p = mx.array(ids)[None]
        mx.eval(model(p, cache=cache)); mx.eval([c.state for c in cache])
        base = mx.array(ids[-8:], dtype=mx.int32)
        def step(i):
            inp = base[(i % 4):(i % 4) + rows][None]
            if verify:
                out = model(inp, cache=cache, return_hidden=True)
                y = mx.argmax(out.logits[:, -1], axis=-1)
                mx.async_eval(y, out.hidden_states[-1]); y.item()
                model.rollback_speculative_cache(cache, out.gdn_states, rows - 1, rows)
            else:
                y = mx.argmax(model(inp, cache=cache)[:, -1], axis=-1)
                mx.async_eval(y); y.item()
        for i in range(10): step(i)
        t = time.perf_counter()
        for i in range(steps): step(i)
        return 1e3 * (time.perf_counter() - t) / steps
    for name, rows, verify in (("plain 1 row (fused decode, synced each step)", 1, False), ("verify path 1 row", 1, True), ("verify path 2 rows", 2, True), ("plain path 2 rows", 2, False), ("verify path 3 rows", 3, True)):
        ms = loop(rows, verify); log(f"  {name:48s} {ms:6.2f} ms/step")

if "hcx" in VAR:
    # Exactness: the row-general fused HC path against the stock verify path, same inputs.
    import hcrows
    def run(enabled, rows, steps=6):
        hcrows.ENABLED[0] = enabled
        cache = model.make_cache()
        mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
        base = mx.array(ids[-8:], dtype=mx.int32); outs = []
        for i in range(steps):
            inp = base[(i % 4):(i % 4) + rows][None]
            out = model(inp, cache=cache, return_hidden=True)
            lg = out.logits.astype(mx.float32); mx.eval(lg, out.hidden_states[-1]); outs.append(lg)
            model.rollback_speculative_cache(cache, out.gdn_states, rows - 1, rows)
        return outs
    for rows in (1, 2, 3, 5):
        a = run(False, rows); b = run(True, rows)
        d = max(mx.abs(x - y).max().item() for x, y in zip(a, b))
        same = all((mx.argmax(x, -1) == mx.argmax(y, -1)).all().item() for x, y in zip(a, b))
        log(f"  hcx rows={rows}: max |dlogit| = {d:.3e}  argmax equal = {same}  scale {mx.abs(a[0]).max().item():.1f}")
    log(f"  hcx stats {hcrows.STATS}")

if "hcy" in VAR:
    # Ground truth: each verify row against one-token decode of the same tokens.
    import hcrows
    def fresh():
        cache = model.make_cache()
        mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache]); return cache
    base = mx.array(ids[-8:], dtype=mx.int32)
    for rows in (2, 4):
        inp = base[:rows][None]
        hcrows.ENABLED[0] = False
        c = fresh(); seq = []
        for j in range(rows):
            lg = model(inp[:, j:j + 1], cache=c).astype(mx.float32); mx.eval(lg); seq.append(lg[0, 0])
        c = fresh(); st = model(inp, cache=c, return_hidden=True).logits.astype(mx.float32); mx.eval(st)
        hcrows.ENABLED[0] = True
        c = fresh(); mi = model(inp, cache=c, return_hidden=True).logits.astype(mx.float32); mx.eval(mi)
        for j in range(rows):
            log(f"  hcy rows={rows} row {j}: stock-vs-1tok {mx.abs(st[0, j] - seq[j]).max().item():.3e}   rows-vs-1tok {mx.abs(mi[0, j] - seq[j]).max().item():.3e}   rows-vs-stock {mx.abs(mi[0, j] - st[0, j]).max().item():.3e}")

if "qcensus" in VAR:
    # What quantized matmuls does one 2-row verify forward issue, from where, and how big?
    import collections
    seen = collections.Counter(); real = mx.quantized_matmul
    def qm(x, w, *a, **k):
        f = sys._getframe(1); chain = []
        while f is not None and len(chain) < 4:
            chain.append(f.f_code.co_name); f = f.f_back
        seen[(tuple(x.shape), tuple(w.shape), k.get("bits"), w.nbytes >> 20, ">".join(chain))] += 1
        return real(x, w, *a, **k)
    cache = model.make_cache()
    mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
    mx.quantized_matmul = qm
    out = model(mx.array(ids[-2:], dtype=mx.int32)[None], cache=cache, return_hidden=True); mx.eval(out.logits)
    mx.quantized_matmul = real
    tot = 0
    for (xs, ws, bits, mb, ch), n in sorted(seen.items(), key=lambda kv: -kv[0][3] * kv[1]):
        tot += mb * n; log(f"  qc {n:3d}x  x{xs} w{ws} bits={bits} {mb} MiB  {ch}")
    log(f"  qc total {tot} MiB through mx.quantized_matmul per 2-row forward")

if os.environ.get("ABLATE"):
    # In-situ cost by removal: replace a module with a trivial op of the same shape (timing only).
    _ab = set(os.environ["ABLATE"].split(","))
    for _l in model.layers:
        if ("kda" in _ab and _l.is_linear) or ("sparse" in _ab and not _l.is_linear):
            def _fa(self, x, mask=None, cache=None, **k): return x * 0.5
            type(_l.self_attn).__call__ = _fa
        if "kda_nosum" in _ab and _l.is_linear or "sparse_nosum" in _ab and not _l.is_linear:
            _b = type(_l.self_attn).__mro__[1]
            type(_l.self_attn).__call__ = (lambda b: lambda self, *a, **k: (lambda o: o + o)(b.__call__(self, *a, **k)))(_b)
        if "mlp" in _ab:
            def _fm(self, x, *a, **k): return x * 0.5
            type(_l.mlp).__call__ = _fm
        if "mlp_nosum" in _ab:
            _b = type(_l.mlp).__mro__[1]
            type(_l.mlp).__call__ = (lambda b: lambda self, *a, **k: (lambda o: o + o)(b.__call__(self, *a, **k)))(_b)
    if "moe_core" in _ab:                      # expert computation only; its collective stays
        _mb = next(c for l in model.layers for c in type(l.mlp).__mro__ if "_decode_select" in c.__dict__)
        _abn = [0]
        def _abcall(self, x): _abn[0] += 1; return x * 0.5
        _mb.__call__ = _abcall
        import atexit; atexit.register(lambda: log('moe_core ablation calls:', _abn[0], 'moerows', __import__('moerows').STATS, 'hcrows', __import__('hcrows').STATS))
        log("moe base class", _mb.__name__, "layers with it:", sum(isinstance(l.mlp, _mb) for l in model.layers), "of", len(model.layers))
    if "indexer" in _ab:                       # timing only: skip the index pool upkeep while selection is bypassed
        _sa = next(l.self_attn for l in model.layers if not l.is_linear)
        _ic = type(_sa.indexer)
        _ic.__call__ = lambda self, *a, **k: None
    # hcrows runs the FFN half inside mx.compile: drop the traced graphs so the replaced calls take effect
    for _l in model.layers:
        _l.__dict__.pop("_tp_ffn_c", None); _l.__dict__.pop("_tp_ffn_seen", None)
    log("ablated:", sorted(_ab))
if "rcheck" in VAR:
    # Bit-exactness of the router logits computed inside the HC pre kernel against the stock router kernel.
    import hcrows_x as _hx
    _cache = model.make_cache()
    mx.eval(model(mx.array(ids)[None], cache=_cache)); mx.eval([c.state for c in _cache])
    for _rows in (2, 3, 2):
        _hx.CHECKS.clear()
        for _l in model.layers:
            _l.__dict__.pop("_tp_ffn_c", None); _l.__dict__.pop("_tp_ffn_seen", None)      # eager pass: checks hold real arrays
        _out = model(mx.array(ids[-_rows:], dtype=mx.int32)[None], cache=_cache, return_hidden=True); mx.eval(_out.logits)
        _n = len(_hx.CHECKS); _bad = 0; _mx = 0.0
        for (_s, _b, _rs, _rb) in _hx.CHECKS:
            mx.eval(_s, _b, _rs, _rb)
            if not (mx.array_equal(_s, _rs).item() and mx.array_equal(_b, _rb).item()):
                _bad += 1; _mx = max(_mx, mx.abs(_s - _rs).max().item(), mx.abs(_b - _rb).max().item())
        log(f"rcheck rows={_rows}: {_n} layers compared, {_bad} differ, max |diff| {_mx:.3e}; hcrows {_hx.STATS}; moerows {sys.modules['moerows'].STATS}")
    sys.exit(0)
if "dumpsrc" in VAR and R == 0:
    import inspect
    _gm = lm.model; _ly = _gm.layers[0]
    with open("/tmp/runtime_src.txt", "w") as _f:
        for _nm, _ob in (("layer class", type(_ly)), ("layer.__call__", type(_ly).__call__), ("prev layer call (closure of hcrows)", None),
                         ("model class", type(_gm)), ("model.__call__", type(_gm).__call__), ("lm class", type(lm)), ("lm.__call__", type(lm).__call__),
                         ("top model class", type(model)), ("attn wrapper", type(_ly.self_attn)), ("attn wrapper call", type(_ly.self_attn).__call__),
                         ("mlp wrapper", type(_gm.layers[5].mlp)), ("mlp wrapper call", type(_gm.layers[5].mlp).__call__), ("attn_hc", type(_ly.attn_hc)), ("attn_hc call", type(_ly.attn_hc).__call__)):
            if _ob is None: continue
            try: _s = inspect.getsource(_ob) if not isinstance(_ob, type) else f"{_ob.__module__}.{_ob.__qualname__} mro={[c.__module__ + '.' + c.__qualname__ for c in _ob.__mro__]}"
            except Exception as _e: _s = f"<no source: {_e!r}> {_ob!r}"
            _f.write(f"\n===== {_nm} =====\n{_s}\n")
        _c = type(_ly).__call__
        if getattr(_c, "__closure__", None):
            for _cell, _name in zip(_c.__closure__, _c.__code__.co_freevars):
                if _name == "prev":
                    try: _f.write(f"\n===== hcrows prev =====\n{inspect.getsource(_cell.cell_contents)}\n")
                    except Exception as _e: _f.write(f"\n===== hcrows prev: {_e!r} {_cell.cell_contents!r}\n")
    log("dumped runtime sources"); sys.exit(0)
WF_TM = collections.Counter(); WF_DEPTH = int(os.environ.get("WF_DEPTH", "3")); WF_CLEAR = int(os.environ.get("WF_CLEAR", "0"))
def wf_forward(gm, inputs, cache, inputs_embeds=None):
    """Same result as gm(inputs, cache=cache) for one sequence, with the rows split in two halves that walk the
    layers as a wavefront: half A's cross-Mac exchange is in flight while the GPU computes half B's half-layer and
    vice versa. Per layer the order stays A then B, so every cache sees the rows in order. Mirrors the vendored
    Glm5NextDecoderLayer.__call__ (prefill branch) and PipelinedTextModel.__call__."""
    PM = sys.modules[type(gm).__module__]; Lg = PM._language
    h_all = gm.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
    n = h_all.shape[1]; cut = (n // 2 + 3) // 4 * 4
    layers = gm.pipeline_layers
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), None)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), None)
    _parts = (h_all,) if os.environ.get("WF_PARTS", "2") == "1" else (h_all[:, :cut], h_all[:, cut:])
    T = [dict(e=e, h=None, y=None, fa=None, ssm=None, have_fa=False, have_ssm=False) for e in _parts]
    Q = collections.deque()
    def step(t, s):
        _t0 = time.perf_counter()
        # No wait on the own previous exchange: its result is consumed through a GPU-side wait that sits BEHIND the
        # other half's half-step in the queue. The host only waits for the half-step queued WF_DEPTH steps earlier
        # (bounds memory; a blocking wait right before encoding left the GPU idle ~8 ms each time).
        while len(Q) > WF_DEPTH:
            mx.eval(Q.popleft())
            if WF_CLEAR and s % (2 * WF_CLEAR) == 0: mx.clear_cache()
        i = s // 2; layer = layers[i]; c = cache[i]
        _t1 = time.perf_counter(); _k = ("kda" if layer.is_linear else "sparse") if s % 2 == 0 else "mlp"
        WF_TM["wait " + _k] += _t1 - _t0
        if s == 0:
            e = t["e"]
            t["h"] = mx.contiguous(mx.broadcast_to(e[:, :, None, :], (e.shape[0], e.shape[1], gm.hc_mult, e.shape[2])))
        WF_CAP[0] = None
        if s % 2 == 0:
            if layer.is_linear:
                if not t["have_ssm"]:
                    t["ssm"] = Lg.create_ssm_mask(t["e"], cache[ssm_idx] if ssm_idx is not None else None); t["have_ssm"] = True
                mask = t["ssm"]
            else:
                if not t["have_fa"]:
                    fc = cache[fa_idx] if fa_idx is not None else None
                    t["fa"] = Lg.create_attention_mask(t["e"], fc[0] if fc else None, return_array=True); t["have_fa"] = True
                mask = t["fa"]
            x = t["h"]; residual = x
            fused = Lg._decode_hc_pre(layer.attn_hc, layer.input_layernorm, x)
            if fused is None:
                xc, post, comb = layer.attn_hc(x); xn = layer.input_layernorm(xc)
            else:
                xn, post, comb = fused
            r = layer.self_attn(xn, mask, c)
            t["h"] = Lg._decode_hc_expand(r, residual, post, comb)
        else:
            t["h"] = layer._ffn_block(t["h"])
        t["y"] = WF_CAP[0]
        _t2 = time.perf_counter()
        _tok = t["y"] if t["y"] is not None else t["h"]
        mx.async_eval(_tok); Q.append(_tok)
        _t3 = time.perf_counter(); WF_TM["build " + _k] += _t2 - _t1; WF_TM["encode " + _k] += _t3 - _t2; WF_TM["n " + _k] += 1
    for s in range(2 * len(layers)):
        for t in T: step(t, s)
    out = gm.norm(mx.concatenate([t["h"] for t in T], axis=1).mean(axis=2))
    if cache and getattr(gm, "_omlx_output_in_cache", False): PM._keep_send_in_graph(cache[-1], out)
    return out

if "wf" in VAR:
    from mlx.utils import tree_flatten
    _gm = lm.model; _blk = int(os.environ.get("PF_BLOCK", "2048")); _arr = mx.array(ids)[None]; _n = len(ids)
    def _states(cache): return [v for _, v in tree_flatten([c.state for c in cache]) if isinstance(v, mx.array)]
    _real_clear = mx.clear_cache; _noclear = os.environ.get("STOCK_NOCLEAR", "0") == "1"; _mem = [0.0]
    def _stock(blk):
        cache = model.make_cache(); mx.eval([c.state for c in cache]); t = time.perf_counter(); out = None
        for o in range(0, _n, blk):
            if _noclear: mx.clear_cache = lambda: None
            try: out = _gm(_arr[:, o:o + blk], cache=cache); mx.eval(out); mx.eval([c.state for c in cache])
            finally: mx.clear_cache = _real_clear
            _mem[0] = max(_mem[0], (mx.get_active_memory() + mx.get_cache_memory()) / 2**30)
            if _noclear: _real_clear()
        return cache, out, time.perf_counter() - t
    def _wf(blk):
        cache = model.make_cache(); mx.eval([c.state for c in cache]); t = time.perf_counter(); out = None
        for o in range(0, _n, blk):
            a = _arr[:, o:o + blk]
            out = wf_forward(_gm, a, cache) if a.shape[1] >= 512 else _gm(a, cache=cache)
            mx.eval(out); mx.eval([c.state for c in cache])
            _mem[0] = max(_mem[0], (mx.get_active_memory() + mx.get_cache_memory()) / 2**30)
            if not WF_CLEAR and os.environ.get("WF_NEVERCLEAR", "0") != "1": _real_clear()
        return cache, out, time.perf_counter() - t
    for _rep in range(2):
        _mem[0] = 0.0; c1, o1, d1 = _stock(_blk); log(f"wf: stock block {_blk}: {_n / d1:.0f} tok/s (active+cache at block ends up to {_mem[0]:.1f} GiB)")
        _mem[0] = 0.0; c2, o2, d2 = _stock(_blk // 2); log(f"wf: stock block {_blk // 2}: {_n / d2:.0f} tok/s (up to {_mem[0]:.1f} GiB)")
        if os.environ.get("WF_KGAP") == "1" and _rep == 1:
            _kr = mx.fast.metal_kernel(name="kprof_reset", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
            _kd = mx.fast.metal_kernel(name="kprof_dump", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
            _one = mx.zeros((1,)); mx.eval(_one)
            def _mark(k): mx.eval(k(inputs=[_one], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])[0])
            _mark(_kr); _c, _o, _d = (_stock(_blk) if os.environ.get("WF_KGAP_STOCK") == "1" else _wf(_blk)); _mark(_kd); log(f"wf kgap run: {_n / _d:.0f} tok/s"); del _c
        _mem[0] = 0.0; WF_TM.clear(); c3, o3, d3 = _wf(_blk); log(f"wf: wavefront memory at block ends up to {_mem[0]:.1f} GiB"); log(f"wf: wavefront block {_blk} (2 x {_blk // 2}): {_n / d3:.0f} tok/s  ({100 * (d1 / d3 - 1):+.1f}% vs stock {_blk})")
        log("wf host time, ms per half-step: " + "; ".join(f"{k} " + " ".join(f"{p} {1e3 * WF_TM[p + ' ' + k] / max(WF_TM['n ' + k], 1):.2f}" for p in ("wait", "build", "encode")) + f" (n={WF_TM['n ' + k]})" for k in ("kda", "sparse", "mlp")) + f" | total {d3:.2f} s")
        log(f"wf memory: active {mx.get_active_memory() / 2**30:.1f} GiB, cache {mx.get_cache_memory() / 2**30:.1f} GiB, peak {mx.get_peak_memory() / 2**30:.1f} GiB")
        s2, s3 = _states(c2), _states(c3)
        same = len(s2) == len(s3) and all(a.shape == b.shape and mx.array_equal(a, b).item() for a, b in zip(s2, s3))
        nd = sum(1 for a, b in zip(s2, s3) if a.shape != b.shape or not mx.array_equal(a, b).item())
        tail = min(o2.shape[1], o3.shape[1])
        osame = mx.array_equal(o2[:, -1], o3[:, -1]).item()
        log(f"wf check vs stock block {_blk // 2}: {len(s2)}/{len(s3)} cache arrays, all equal = {same} ({nd} differ); last hidden equal = {osame}")
        del c1, c2, c3; mx.clear_cache()
    sys.exit(0)
if "prefillcost" in VAR:
    # Prompt reading by removal: time to read the prompt in blocks of PF_BLOCK tokens (after any ABLATE).
    _blk = int(os.environ.get("PF_BLOCK", "2048")); _arr = mx.array(ids)[None]
    for _rep in range(2):
        _cache = model.make_cache(); mx.eval([c.state for c in _cache]); _t = time.perf_counter()
        for _o in range(0, len(ids), _blk):
            mx.eval(model(_arr[:, _o:_o + _blk], cache=_cache)); mx.eval([c.state for c in _cache])
        _dt = time.perf_counter() - _t
        log(f"prefillcost: {len(ids)} tokens in {_dt:.2f} s = {len(ids) / _dt:.0f} tok/s (block {_blk})")
    sys.exit(0)
if "kpass" in VAR:
    # Per-kernel profile (MLX_KPROF=1) of plain multi-row passes only, after any ABLATE: reset, KN passes, dump.
    _kr = mx.fast.metal_kernel(name="kprof_reset", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    _kd = mx.fast.metal_kernel(name="kprof_dump", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    _one = mx.zeros((1,)); mx.eval(_one)
    def _mark(k): mx.eval(k(inputs=[_one], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])[0])
    _rows = int(os.environ.get("KROWS", "2")); _kn = int(os.environ.get("KN", "40"))
    _cache = model.make_cache()
    mx.eval(model(mx.array(ids)[None], cache=_cache)); mx.eval([c.state for c in _cache])
    _inp = mx.array(ids[-_rows:], dtype=mx.int32)[None]
    def _kstep(inp):
        out = model(inp, cache=_cache, return_hidden=True)
        y = mx.argmax(out.logits[0], axis=-1).astype(mx.int32)[None]
        mx.async_eval(y); return mx.array(y.tolist(), dtype=mx.int32)
    for _i in range(8): _inp = _kstep(_inp)
    _mark(_kr); _t = time.perf_counter()
    for _i in range(_kn): _inp = _kstep(_inp)
    _mark(_kd)
    log(f"kpass: {_kn} passes of {_rows} rows, {1e3 * (time.perf_counter() - _t) / _kn:.2f} ms per pass (profiler on)")
    sys.exit(0)
if "pipecost" in VAR:
    # Upper bound for an in-graph speculative loop: the same forwards with the next step's
    # tokens fed lazily (no host read between steps) against one host read per step.
    def loop(rows, lazy, steps=150, verify=True):
        cache = model.make_cache()
        mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
        inp = mx.array(ids[-rows:], dtype=mx.int32)[None]
        def step(inp):
            if verify:
                out = model(inp, cache=cache, return_hidden=True); lg = out.logits
            else:
                lg = model(inp, cache=cache)
            y = mx.argmax(lg[0], axis=-1).astype(mx.int32)[None]      # [1, rows]
            mx.async_eval(y)
            if not lazy: y = mx.array(y.tolist(), dtype=mx.int32)
            return y
        for i in range(10): inp = step(inp)
        mx.eval(inp); t = time.perf_counter()
        for i in range(steps): inp = step(inp)
        mx.eval(inp)
        return 1e3 * (time.perf_counter() - t) / steps
    for rows in [int(v) for v in os.environ.get("ROWS", "1,2,3").split(",")]:
        a = loop(rows, False); b = loop(rows, True)
        log(f"  pipecost rows={rows}: host read per step {a:6.2f} ms   lazy feedback {b:6.2f} ms   saved {a - b:5.2f} ms")

if "layout" in VAR and R == 0:
    Ls = list(model.layers)
    log("  layout " + "".join("K" if l.is_linear else "S" for l in Ls))
    c = model.make_cache()
    log("  cache " + " ".join(f"{i}:{type(x).__name__}" for i, x in enumerate(c))[:400])
    k = next(l for l in Ls if l.is_linear).self_attn
    log(f"  kda heads={k.num_heads} head_dim={k.head_dim} conv={k.conv_kernel_size} split={getattr(k, '_split_pts', None)}")
    mx.eval(model(mx.array(ids)[None], cache=c)); mx.eval([x.state for x in c])
    i = next(i for i, l in enumerate(Ls) if l.is_linear)
    log(f"  kda cache state shapes {[tuple(s.shape) for s in c[i].state if s is not None]} dtypes {[str(s.dtype) for s in c[i].state if s is not None]}")

if "kgaplazy" in VAR:
    # GPU idle gaps with lazy feedback (no host read between steps), 2 rows.
    _kr = mx.fast.metal_kernel(name="kprof_reset", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    _kd = mx.fast.metal_kernel(name="kprof_dump", input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    _one = mx.zeros((1,)); mx.eval(_one)
    def _mark(k): mx.eval(k(inputs=[_one], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])[0])
    cache = model.make_cache()
    mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
    inp = mx.array(ids[-2:], dtype=mx.int32)[None]
    def step(inp):
        out = model(inp, cache=cache, return_hidden=True)
        y = mx.argmax(out.logits[0], axis=-1).astype(mx.int32)[None]
        mx.async_eval(y); return y
    for i in range(10): inp = step(inp)
    mx.eval(inp); _mark(_kr); t = time.perf_counter()
    for i in range(150): inp = step(inp)
    mx.eval(inp); dt = time.perf_counter() - t; _mark(_kd)
    log(f"  kgaplazy: {1e3 * dt / 150:.2f} ms/step")

if "headcost" in VAR:
    # What the draft head adds to a fully pipelined 2-row cycle, by variant (lazy feedback, no host read).
    def loop(mode, steps=150):
        cache = model.make_cache(); hc = lm.make_mtp_cache() if has_head else None
        mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
        inp = mx.array(ids[-2:], dtype=mx.int32)[None]
        def step(inp):
            out = model(inp, cache=cache, return_hidden=True)
            tt = mx.argmax(out.logits[0], axis=-1).astype(mx.int32)              # [2]
            if mode == "none":
                y = tt[None]
            else:
                h = lm.model.norm(out.hidden_states[-1])
                rows = 1 if mode == "head1" else 2
                _draft_ctx[0] = SHARDLM
                try:
                    if has_head:
                        lg = lm.mtp_forward(h[:, -rows:], tt[None, -rows:], hc, logits_keep=(1 if mode in ("head2keep1", "head1") else 0))
                    else:
                        k = 1 if mode in ("head2keep1", "head1") else 2
                        lg = _linear_forward(_hd, mx.zeros((1, k, h.shape[-1]), dtype=mx.bfloat16) + tt[None, -k:, None].astype(mx.bfloat16) * 0)
                finally: _draft_ctx[0] = False
                d = mx.argmax(lg[0], axis=-1).astype(mx.int32)
                y = mx.stack([tt[-1], d[-1]])[None]
            mx.async_eval(y); return y
        for i in range(10): inp = step(inp)
        mx.eval(inp); t = time.perf_counter()
        for i in range(steps): inp = step(inp)
        mx.eval(inp)
        return 1e3 * (time.perf_counter() - t) / steps
    base = None
    for mode in ("none", "head2", "head2keep1", "head1"):
        ms = loop(mode); base = base or ms
        log(f"  headcost {mode:11s} {ms:6.2f} ms/cycle  (+{ms - base:.2f})")

if "moex" in VAR:
    # Fused multi-row routing against the stock router-select path: same routes, same outputs?
    import moerows
    def run(enabled, rows, steps=5):
        moerows.ENABLED[0] = enabled
        cache = model.make_cache()
        mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
        base = mx.array(ids[-8:], dtype=mx.int32); outs = []
        for i in range(steps):
            inp = base[(i % 4):(i % 4) + rows][None]
            out = model(inp, cache=cache, return_hidden=True)
            lg = out.logits.astype(mx.float32); mx.eval(lg, out.hidden_states[-1]); outs.append(lg)
            model.rollback_speculative_cache(cache, out.gdn_states, rows - 1, rows)
        return outs
    for rows in (2, 3):
        a = run(False, rows); b = run(True, rows)
        d = max(mx.abs(x - y).max().item() for x, y in zip(a, b))
        log(f"  moex rows={rows}: max |dlogit| = {d:.3e}   stats {moerows.STATS}")

if "headsplit" in VAR:
    # Where the draft head's time goes on a pipelined 2-row cycle: its block vs the vocabulary pass.
    from omlx.patches.glm5_next_mlx_lm import model as _MHs
    _cam = _MHs._language.create_attention_mask
    def loop(mode, steps=150):
        cache = model.make_cache(); hc = lm.make_mtp_cache() if has_head else None
        mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
        inp = mx.array(ids[-2:], dtype=mx.int32)[None]
        def step(inp):
            out = model(inp, cache=cache, return_hidden=True)
            tt = mx.argmax(out.logits[0], axis=-1).astype(mx.int32)
            d = tt[-1:]
            if mode != "none":
                h = lm.model.norm(out.hidden_states[-1])
                if has_head and mode in ("block", "full"):
                    hh = h; mask = _cam(hh, hc[0], return_array=True)
                    for i, blk in enumerate(lm.mtp):
                        hh = blk(hh, lm.model.embed_tokens, tt[None], mask, hc[2 * i: 2 * i + 2]); last = blk
                    hk = last.norm(hh[:, -1:])
                else:
                    hk = h[:, -1:]
                if mode in ("lm", "full"):
                    _draft_ctx[0] = SHARDLM
                    try: lg = _linear_forward(_hd, hk if has_head else mx.zeros((1, 1, h.shape[-1]), dtype=mx.bfloat16) + tt[None, -1:, None].astype(mx.bfloat16) * 0)
                    finally: _draft_ctx[0] = False
                    d = mx.argmax(lg[0], axis=-1).astype(mx.int32)
                elif has_head:
                    d = d + (hk.sum() * 0).astype(mx.int32)
            y = mx.stack([tt[-1], d[-1]])[None]
            mx.async_eval(y); return y
        for i in range(10): inp = step(inp)
        mx.eval(inp); t = time.perf_counter()
        for i in range(steps): inp = step(inp)
        mx.eval(inp)
        return 1e3 * (time.perf_counter() - t) / steps
    base = None
    for mode in ("none", "block", "lm", "full"):
        ms = loop(mode); base = base or ms
        log(f"  headsplit {mode:6s} {ms:6.2f} ms/cycle  (+{ms - base:.2f})")

if "satrace" in VAR:
    # Which code path does a sparse-attention layer take for 1, 2 and 3 rows?
    import collections
    sa = next(l.self_attn for l in model.layers if not l.is_linear)
    log('  satrace wrapper', type(sa).__module__, type(sa).__name__, [k for k in sa.keys()][:6])
    sa = sa.inner if 'inner' in sa else sa
    cls_sa = next(c for c in type(sa).__mro__ if '_forward' in c.__dict__)
    Lm = sys.modules[cls_sa.__module__]
    cnt = collections.Counter()
    for name in ("_decode_fused", "_decode_fused_batched", "_rows_attention", "_gathered_attention", "_selected_decode_attention",
                 "_selected_rows_attention", "_dense_flat", "_decode_projections", "_decode_fused_ok", "_batched_fused_ok"):
        if hasattr(cls_sa, name):
            def mk(name, f):
                def w(self, *a, **k):
                    r = f(self, *a, **k); cnt[name + ("" if not isinstance(r, bool) else f"={r}") + (":None" if r is None else "")] += 1; return r
                return w
            setattr(cls_sa, name, mk(name, getattr(cls_sa, name)))
    for name in ("small_windows", "small_windows_batched", "__call__"):
        icl = type(sa.indexer)
        def mk2(name, f):
            def w(self, *a, **k):
                r = f(self, *a, **k); cnt["indexer." + name + (":None" if r is None else "")] += 1; return r
            return w
        setattr(icl, name, mk2(name, getattr(icl, name)))
    osm = Lm.supports_sparse_mla_mask
    def ssm(mask, B, Lr, K):
        r = osm(mask, B, Lr, K); cnt[f"supports_mask({None if mask is None else tuple(mask.shape)})={r}"] += 1; return r
    Lm.supports_sparse_mla_mask = ssm
    for plen in (len(ids), 3000):
        pid = (ids * (plen // len(ids) + 1))[:plen]
        for rows in (1, 2, 3):
            cache = model.make_cache()
            mx.eval(model(mx.array(pid)[None], cache=cache)); mx.eval([c.state for c in cache])
            cnt.clear()
            out = model(mx.array(pid[-rows:], dtype=mx.int32)[None], cache=cache, return_hidden=True); mx.eval(out.logits)
            log(f"  satrace prompt={plen} rows={rows}: " + ", ".join(f"{k} x{v}" for k, v in sorted(cnt.items())))

if "abdbg" in VAR and R == 0:
    l0 = model.layers[0]
    log("  abdbg mlp type", type(l0.mlp).__name__, [c.__name__ for c in type(l0.mlp).__mro__[:4]], "call is patched:", type(l0.mlp).__call__.__name__)
    log("  abdbg layer type", type(l0).__name__, [c.__name__ for c in type(l0).__mro__[:3]], "mlp attr via dict:", type(l0.__dict__.get("mlp", l0.get("mlp") if hasattr(l0, "get") else None)).__name__)
    import inspect
    src = inspect.getsource(type(l0).__mro__[-3]._ffn_block) if hasattr(type(l0).__mro__[-3], "_ffn_block") else ""
    log("  abdbg stats", __import__("hcrows").STATS, __import__("moerows").STATS)

if "sax" in VAR:
    # Fused dense sparse-layer attention against the stock path, same inputs.
    import sarows
    def run(enabled, rows, steps=5, plen=None):
        sarows.ENABLED[0] = enabled
        pid = ids if plen is None else (ids * (plen // len(ids) + 1))[:plen]
        cache = model.make_cache()
        mx.eval(model(mx.array(pid)[None], cache=cache)); mx.eval([c.state for c in cache])
        base = mx.array(pid[-8:], dtype=mx.int32); outs = []
        for i in range(steps):
            inp = base[(i % 4):(i % 4) + rows][None]
            out = model(inp, cache=cache, return_hidden=True)
            lg = out.logits.astype(mx.float32); mx.eval(lg, out.hidden_states[-1]); outs.append(lg)
            model.rollback_speculative_cache(cache, out.gdn_states, rows - 1, rows)
        return outs
    for plen in (None, 900):
        for rows in (2, 3):
            a = run(False, rows, plen=plen); b = run(True, rows, plen=plen)
            d = max(mx.abs(x - y).max().item() for x, y in zip(a, b))
            same = all((mx.argmax(x, -1) == mx.argmax(y, -1)).all().item() for x, y in zip(a, b))
            log(f"  sax prompt={plen or len(ids)} rows={rows}: max |dlogit| = {d:.3e}  argmax equal = {same}   stats {sarows.STATS}")

if "overlap" in VAR:
    # How many routed experts do adjacent verify rows share, per layer, on real text?
    import moerows, collections
    Lm = None
    for l in model.layers:
        m = l.mlp
        c = next((c for c in type(m).__mro__ if "_decode_select" in c.__dict__), None)
        if c is not None: Lm = sys.modules[c.__module__]; break
    dk = Lm._decode_kernels
    logs = []
    orig = dk.moe_down_combine
    def spy(act, routes, weights, *a, **k):
        logs.append(routes); return orig(act, routes, weights, *a, **k)
    dk.moe_down_combine = spy
    for l in model.layers: l.compile_ffn = False        # routes must be real arrays, not trace placeholders
    text = "The transistor was invented at Bell Labs in 1947 by John Bardeen, Walter Brattain and William Shockley. It replaced the vacuum tube and made modern electronics possible, leading first to the integrated circuit and then to the microprocessor. Today billions of transistors fit on a single chip, and the scaling of their size has driven the growth of computing for decades."
    tids = tok.encode(text)
    cache = model.make_cache()
    mx.eval(model(mx.array(ids)[None], cache=cache)); mx.eval([c.state for c in cache])
    share2 = []; share3 = []
    for i in range(0, min(len(tids) - 3, 60), 3):
        logs.clear()
        out = model(mx.array(tids[i:i + 3], dtype=mx.int32)[None], cache=cache, return_hidden=True); mx.eval(out.logits)
        for r in logs:
            if r.shape[0] != 3: continue
            rr = [set(x) for x in r.tolist()]
            share2.append(len(rr[0] & rr[1])); share3.append(len(rr[2] & (rr[0] | rr[1])))
    if R == 0 and share2:
        log(f"  overlap: {len(share2)} layer-windows; row 2 shares {sum(share2)/len(share2):.2f} of 8 experts with row 1; row 3 shares {sum(share3)/len(share3):.2f} of 8 with rows 1-2")
        log("  overlap histogram (row2 with row1): " + str(dict(sorted(collections.Counter(share2).items()))))

if "ceiling" in VAR:
    # Draft head accuracy with exact context: plain one-row greedy decode; after every token the head
    # folds (hidden of that position, the token that followed) and drafts the token after it.
    PROMPTS = [
        "Write a detailed essay on the history of the transistor.",
        "Explain how vaccines train the immune system, for a curious teenager.",
        "Describe the causes and consequences of the fall of the Western Roman Empire.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    ]
    tot = hit = 0
    for pi, ptxt in enumerate(PROMPTS):
        pids = tok.apply_chat_template([{"role": "user", "content": ptxt}], add_generation_prompt=True)
        cache = model.make_cache(); hc = lm.make_mtp_cache() if has_head else None
        p = mx.array(pids)[None]
        out = model(p, cache=cache, return_hidden=True)
        h = lm.model.norm(out.hidden_states[-1])
        t = bcast(mx.argmax(out.logits[:, -1], axis=-1).astype(mx.int32))        # first generated token
        def head(rows, next_ids):
            if has_head:
                _draft_ctx[0] = SHARDLM
                try: lg = lm.mtp_forward(rows, next_ids, hc, logits_keep=1)
                finally: _draft_ctx[0] = False
            else:
                _draft_ctx[0] = True
                try: lg = _linear_forward(_hd, mx.zeros((1, 1, rows.shape[-1]), dtype=mx.bfloat16) + next_ids[:, -1:, None].astype(mx.bfloat16) * 0)
                finally: _draft_ctx[0] = False
            return mx.argmax(lg[:, -1], axis=-1).astype(mx.int32)
        d = bcast(head(h, mx.concatenate([p[:, 1:], t[None].astype(p.dtype)], axis=1)))   # draft of the token after t
        mx.eval(t, d, [c.state for c in cache])
        n = hits = 0; seq = [int(t.item())]
        for step in range(int(os.environ.get("NTOK", "400"))):
            out = model(t[None], cache=cache, return_hidden=True)
            h = lm.model.norm(out.hidden_states[-1])
            t2 = bcast(mx.argmax(out.logits[:, -1], axis=-1).astype(mx.int32))
            d2 = bcast(head(h, t2[None]))
            mx.eval(t2, d2)
            tv = int(t2.item()); n += 1; hits += int(int(d.item()) == tv)
            t, d = t2, d2; seq.append(tv)
            if tv == tok.eos_token_id: break
        tot += n; hit += hits
        log(f"  ceiling prompt {pi}: head draft correct {hits}/{n} = {hits / max(n, 1):.3f}")
    log(f"  ceiling overall: {hit}/{tot} = {hit / max(tot, 1):.3f}")
