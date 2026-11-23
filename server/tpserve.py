# Lean tensor-parallel server: stock mlx_lm.server on every rank + oMLX's GLM loader + GPU-stream collectives.
# Rank 0 serves the OpenAI-style HTTP API on $TP_PORT (default 8010).
import os, sys
import mlx.core as mx
# The generation thread shares the interpreter with the HTTP threads; a 5 ms switch interval lets one
# of them hold it for a whole GPU-idle stretch.
sys.setswitchinterval(float(os.environ.get("TP_SWITCH", "0.0002")))
g = mx.distributed.init(strict=True, backend=os.environ.get("TP_BACKEND", "jaccl"))
_all_sum = mx.distributed.all_sum
def all_sum(x, *args, **kwargs):
    y = _all_sum(x, *args, **kwargs)
    if kwargs.get("stream") is not None or (len(args) > 1 and args[1] is not None):
        return y
    return mx.squeeze(mx.expand_dims(y, 0), 0)      # lands on the GPU stream without dispatching a kernel
if os.environ.get("TP_GPUHOP", "1") == "1":
    mx.distributed.all_sum = all_sum
path = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/GLM-5.3-Flash-oQ4e-MTP"))
from types import SimpleNamespace
R = g.rank()
MTP = os.environ.get("TP_MTP", "1") == "1"
def bcast(x):          # rank 0's value on every rank, lazily, with the same graph on every rank
    return mx.distributed.all_sum(x if R == 0 else x * 0)
def bcast2(x):         # (host-readable CPU-stream result, GPU-stream array for the graph)
    y = _all_sum(x if R == 0 else x * 0)
    return y, mx.squeeze(mx.expand_dims(y, 0), 0)
PREK = int(os.environ.get("TP_PREFIX", "3"))          # trunk layers of the next cycle queued before the host read (0 = off)
HCROWS = os.environ.get("TP_HCROWS", "1") == "1"       # fused HC kernels for verify blocks
_pre = {}
def _install_prefix(model):
    """Split trunk forward for speculative cycles.

    A cycle's host read (which rows were accepted) used to leave the GPU idle until the host had
    woken, rolled back and built the next graph (~1 ms). The first PREK layers are linear-attention
    layers whose state can be chosen in-graph, so they are built for the NEXT cycle before the read;
    the remaining layers continue from that stashed result once the host knows the outcome.
    """
    from omlx.patches.glm5_next_mlx_lm import model as _MH
    Lp = _MH._language; gm = model.language_model.model
    layers = gm.pipeline_layers
    if PREK <= 0 or len(layers) <= PREK + 1 or not all(l.is_linear for l in layers[:PREK]):
        return
    FL = frozenset({PREK + 1, PREK + 2, PREK + 4, PREK + 8, 16, 24, 32, 40})
    def run_prefix(inp, cache, sink):
        h = gm.embed_tokens(inp)
        ssm_mask = Lp.create_ssm_mask(h, cache[0])
        h4 = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], gm.hc_mult, h.shape[2])))
        for i in range(PREK):
            h4 = layers[i](h4, mask=ssm_mask, cache=cache[i], gdn_sink=sink)
        return h, h4
    cls = type(gm); orig = cls.__call__
    def call(self, inputs, cache=None, inputs_embeds=None, gdn_sink=None, hidden_sink=None, **kw):
        stash = self.__dict__.pop("_tp_pre", None)
        if stash is None:
            return orig(self, inputs, cache=cache, inputs_embeds=inputs_embeds, gdn_sink=gdn_sink, hidden_sink=hidden_sink, **kw)
        h_emb, h, pre_sink = stash
        lys = self.pipeline_layers; n = len(lys)
        fa_idx = next(i for i, l in enumerate(lys) if not l.is_linear)
        fa_mask = Lp.create_attention_mask(h_emb, cache[fa_idx][0], return_array=True)
        ssm_mask = Lp.create_ssm_mask(h_emb, cache[0])
        gdn_sink.extend(pre_sink)
        for i in range(PREK, n):
            layer = lys[i]
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=cache[i], gdn_sink=gdn_sink)
            if (i + 1) in FL and i + 1 < n:
                mx.async_eval(h.arrays() if isinstance(h, Lp._HCDeferred) else h)
        h = h.mean(axis=2)
        if hidden_sink is not None: hidden_sink.append(h)
        return self.norm(h)
    cls.__call__ = call
    _pre.update(gm=gm, run=run_prefix, layers=layers, cam=Lp.create_attention_mask, lf=lambda: Lp.linear_forward)
    print(f"[r{R}] next-cycle prefix: {PREK} layers queued before the host read", flush=True)
from omlx.utils.model_loading import maybe_apply_pre_load_patches
maybe_apply_pre_load_patches(path, SimpleNamespace(mtp_enabled=True, mtp_fixed_depth=None, mtp_adaptive_max_depth=None) if MTP else None)
import mlx_lm.server as S
from omlx.cluster.progressive_loading import progressive_sharded_load
# The draft head on both Macs, split like a trunk layer (stock: rank zero only, whole). Its attention
# and expert layer then cost each rank half, and no rank waits idle while the other drafts.
SHARD_HEAD = MTP and os.environ.get("TP_SHARD_HEAD", "1") == "1"
if SHARD_HEAD:
    import omlx.patches.glm5_next_mlx_lm.model as _RM
    from omlx.patches.mlx_vlm_mtp import set_mtp_attach_enabled as _set_attach, is_mtp_attach_enabled as _is_attach
    _set_attach(True)
    def _language_model_all_ranks(args):
        _RM.glm5_next_vlm_runtime.apply()
        prev = _is_attach(); _set_attach(True)
        try: lmod = _RM._language.LanguageModel(args.text_config, args)
        finally: _set_attach(prev)
        lmod.model._omlx_output_in_cache = True
        return lmod
    _RM._language_model = _language_model_all_ranks

def _load_head_weights(model, model_path):
    """oMLX's rank loader leaves language_model.mtp.* at its initial values; load it from the checkpoint."""
    import json
    idx = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
    files, items = {}, []
    for k in (k for k in idx if k.startswith("language_model.mtp.")):
        f = idx[k]
        if f not in files: files[f] = mx.load(os.path.join(model_path, f))
        items.append((k, files[f][k]))
    model.load_weights(items, strict=False)
    return len(items)

def _sharded_load(repo, *args, **kwargs):
    res = progressive_sharded_load(repo, *args, **kwargs)
    model = res[0]
    lm = getattr(model, "language_model", None)
    if MTP and lm is not None:
        head = getattr(lm, "mtp", None)
        if head is not None:
            n = _load_head_weights(model, str(repo))
            if SHARD_HEAD:
                from omlx.cluster import tensor_strategies as _TS
                for _b in head:
                    _b.block.is_linear = False
                    _TS._shard_glm5_next(SimpleNamespace(layers=[_b.block]), g, mx, None)
                print(f"[r{R}] draft head split across ranks", flush=True)
            mx.eval([b.parameters() for b in head])
            print(f"[r{R}] MTP head: loaded {n} tensors", flush=True)
        lm._omlx_mtp_decode_enabled = False        # oMLX's own cluster MTP stays off; ours drives the head
        model._tpm_enabled = True
        model._tp_nlayers = len(model.layers)
        if head is not None:
            # The draft head's attention context lives at the end of the sequence's cache list, so the
            # server's prompt cache saves, restores and trims it together with the trunk caches.
            cls = type(model); base_make = cls.make_cache
            def make_cache(self):
                return base_make(self) + self.language_model.make_mtp_cache()
            cls.make_cache = make_cache
            _shared["lm"] = lm; _shared["nl"] = model._tp_nlayers
    if MTP and lm is not None and HCROWS:
        import hcrows
        print(f"[r{R}] fused HC rows: {hcrows.install(list(model.layers))}", flush=True)
    if MTP and lm is not None and os.environ.get("TP_MOEROWS", "1") == "1":
        import moerows
        print(f"[r{R}] fused multi-row routing: {moerows.install(list(model.layers))}", flush=True)
    if MTP and lm is not None and os.environ.get("TP_QMVROWS", "1") == "1":
        import qmvrows
        print(f"[r{R}] two-row vocabulary head kernel: {qmvrows.install()}", flush=True)
    if MTP and lm is not None: _install_prefix(model)
    if MTP and lm is not None: _install_multiseq(model)
    if lm is not None and os.environ.get("TP_VISION", "1") == "1":
        try:
            import tpvision
            tpvision.install(S, model, str(repo), R)
        except Exception as e:
            print(f"[r{R}] vision not enabled: {type(e).__name__}: {e}", flush=True)
    if lm is not None:
        if SHARDLM: _shard_lm_head(lm)
        if MTP and PRIME and (R == 0 or SHARD_HEAD): _install_prompt_capture()
    return res

SHARDLM = os.environ.get("TP_SHARD_LM_HEAD", "1") == "1"
PRIME = os.environ.get("TP_MTP_PRIME", "1") == "1"
_draft_ctx = [False]
_shared = {}
def _shard_lm_head(lm):
    """Split the vocabulary head by rows: each rank computes its share of the logits and one
    all_gather joins them (the head is 605 MiB, read in full for every token otherwise)."""
    from omlx.patches.glm5_next_mlx_lm import model as _MH
    Lg = _MH._language
    hd = lm.lm_head
    V = hd.weight.shape[0]; n = g.size()
    if V % n: return
    part = V // n
    for name in ("weight", "scales", "biases"):
        setattr(hd, name, mx.contiguous(getattr(hd, name)[R * part:(R + 1) * part]))
    mx.eval(hd.parameters())
    lf = Lg.linear_forward
    def linear_forward(layer, x):
        if layer is not hd: return lf(layer, x)
        if _draft_ctx[0]:                      # x exists on rank 0 only (draft head): share it first
            x = bcast(x.astype(mx.bfloat16))
        yt = mx.contiguous(mx.moveaxis(lf(layer, x), -1, 0))
        return mx.moveaxis(mx.distributed.all_gather(yt), 0, -1)
    Lg.linear_forward = linear_forward
    _shared["head"] = hd; _shared["lf"] = linear_forward
    print(f"[r{R}] lm_head sharded to {tuple(hd.weight.shape)}", flush=True)

def _install_prompt_capture():
    """Fold every non-speculative forward into the draft head's context (rank 0).

    The head caches sit after the trunk caches in the sequence's cache list. Each processed position
    gets exactly one head entry, so the server's cache save/restore/trim keeps both aligned. The pair
    for a position is (normed hidden, next token); the last position of a call does not know its next
    token yet and is folded with its own token (a small draft-quality cost, never a correctness one).
    """
    from omlx.patches.glm5_next_mlx_lm import model as _MH
    cls = _MH.PipelinedTextModel
    cam = _MH._language.create_attention_mask
    orig = cls.__call__
    def call(self, inputs, cache=None, inputs_embeds=None, gdn_sink=None, **kw):
        out = orig(self, inputs, cache=cache, inputs_embeds=inputs_embeds, gdn_sink=gdn_sink, **kw)
        lm = _shared.get("lm")
        if (gdn_sink is None and lm is not None and cache is not None and inputs is not None
                and inputs.ndim == 2 and inputs.shape[0] == 1 and len(cache) > _shared["nl"]):
            hc = cache[_shared["nl"]:]
            ids = mx.concatenate([inputs[:, 1:], inputs[:, -1:]], axis=1).astype(mx.int32)
            h = out
            mask = cam(h, hc[0], return_array=True)
            for j, blk in enumerate(lm.mtp):
                h = blk(h, lm.model.embed_tokens, ids, mask, hc[2 * j: 2 * j + 2])
        return out
    cls.__call__ = call

S.sharded_load = _sharded_load

class _FlushSchedule(int):
    """Stands in for the decode loop's "flush every N layers" constant: `(layer + 1) % N == 0` becomes
    true at chosen layers, so the GPU gets its first work right after a cycle's sync and larger blocks
    later."""
    points = frozenset()
    def __rmod__(self, other): return 0 if other in self.points else 1
def _install_flush_schedule(spec):
    from omlx.patches.glm5_next_mlx_lm import model as _MH
    sched = _FlushSchedule(8); sched.points = frozenset(int(v) for v in spec.split(",") if v)
    _MH._language._DECODE_EVAL_EVERY = sched
    print(f"[r{R}] flush schedule {sorted(sched.points)}", flush=True)
if os.environ.get("TP_FLUSH", "1,2,4,8,16,24,32,40"):
    _install_flush_schedule(os.environ.get("TP_FLUSH", "1,2,4,8,16,24,32,40"))

# --- per-token sync removal -------------------------------------------------------------
# The stock server calls r.logprobs[r.token].item() for every token. That builds a new GPU op
# and blocks on it; on the generation stream it queues behind the *next* token's work, which
# is already in flight, so the Python side can no longer prepare token t+2 while t+1 computes.
# Compute the chosen token's logprob inside the step graph instead and hand it back evaluated.
import importlib; G = importlib.import_module("mlx_lm.generate")
class _Val:
    __slots__ = ("v",)
    def __init__(self, v): self.v = v
    def item(self): return self.v
class _LP:
    __slots__ = ("_mx", "_val")
    def __init__(self, arr, val): self._mx = arr; self._val = val
    def __getitem__(self, i):
        if self._val is not None and isinstance(i, int): return _Val(self._val)
        return self._mx[i]
    def __getattr__(self, name): return getattr(self._mx, name)
_orig_fmt = S._format_top_logprobs
def _fmt(logprobs, top_n, tokenizer):
    return _orig_fmt(getattr(logprobs, "_mx", logprobs), top_n, tokenizer)
S._format_top_logprobs = _fmt

def _seeded_pick(sm, lp, index):
    tp = sm.tp
    for f in tp["chain"]: lp = f(lp)
    return mx.argmax(lp.astype(mx.float32) * (1.0 / tp["temp"]) + _noise(sm.tp_seed, index, lp.shape[-1])[None], axis=-1).astype(mx.uint32)
def _step(self, _logits_fn=None):
    if not getattr(self, '_tp_stripped', False) and len(self.uids) == 1:
        _strip_padding(self.prompt_cache); self._tp_stripped = True
    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    cur_side = getattr(self, "_lp_side", {})
    inputs = self._current_tokens
    if _logits_fn is None:
        logits = self.model(inputs[:, None], cache=self.prompt_cache)[:, -1, :]
    else:
        logits = _logits_fn(inputs)
    token_context = []
    if any(self.logits_processors):
        processed_logits = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            processors = self.logits_processors[e] or ()
            if processors:
                context = self._token_context[e].update_and_fetch(inputs[e : e + 1])
                token_context.append(context)
                for processor in processors:
                    sample_logits = processor(context, sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    samplers = [self.fallback_sampler if s is None else s for s in (self.samplers or [None] * len(self.uids))]
    if any(getattr(s, "tp_seed", None) is not None for s in samplers):
        sampled = mx.concatenate([_seeded_pick(s, logprobs[e : e + 1], len(self.tokens[e]) + 1) if getattr(s, "tp_seed", None) is not None
                                  else s(logprobs[e : e + 1]) for e, s in enumerate(samplers)], axis=0)
    elif not samplers or all(s is samplers[0] for s in samplers):
        sampled = (samplers[0] if samplers else self.fallback_sampler)(logprobs)
    else:
        sampled = mx.concatenate([s(logprobs[e : e + 1]) for e, s in enumerate(samplers)], axis=0)
    tok_lp = mx.take_along_axis(logprobs, sampled[:, None], axis=-1)[:, 0].astype(mx.float32)
    self._next_tokens = sampled
    self._next_logprobs = list(logprobs)
    side = {id(a): tok_lp[e] for e, a in enumerate(self._next_logprobs)}
    self._lp_side = side
    mx.async_eval(self._next_tokens, self._next_logprobs, list(side.values()), token_context)
    cur = [cur_side.get(id(a)) for a in self._current_logprobs]
    mx.eval(inputs, self._current_logprobs, [c for c in cur if c is not None])
    inputs = inputs.tolist()
    for sti, ti in zip(self.tokens, inputs):
        sti.append(ti)
    return inputs, [_LP(a, None if c is None else c.item()) for a, c in zip(self._current_logprobs, cur)]
_plain_step = _step
_DUMMY_LP = None
COUPLED = os.environ.get("TP_COUPLED", "1") == "1"
# A request with a `seed` used to leave the batch path entirely (the stock server runs it alone, one
# token per pass, no drafts). Every token on the fast paths is argmax(logits / T + noise(seed, output
# position)), so the client's seed can simply take the place of the random per-request seed: the
# same seed and prompt then give the same tokens whichever path or batch the request runs in.
# TP_SEEDED: 0 = stock behaviour (seeded requests leave the batch path), 1 = seeded requests use the
# fast path and DECODE alone (default; the output for a seed is then exactly repeatable), 2 = seeded
# requests also share decode steps with other requests (more throughput; under overlap the logits
# round differently with the batch shape, so near-ties can fall the other way and outputs are not
# exactly repeatable).
# Mode 1 only needs the decode steps to be single-stream. Prompts of other requests can still be read
# in between (a separate forward on separate caches), so a seeded request is admitted like any other
# and the rule is applied where a request moves from prompt reading to generation (see _seed_gate):
# it waits there while the generation batch holds, or it is, a request that must decode alone.
# With TP_PROMPT_CONC > 1 that hand-over is not one request at a time, so mode 1 falls back to
# running the seeded request with nothing else admitted (the stock "drain" mechanism).
SEED_MODE = int(os.environ.get("TP_SEEDED", "1"))
SEEDED = SEED_MODE > 0
SEED_GATE = SEED_MODE == 1 and int(os.environ.get("TP_PROMPT_CONC", "1")) == 1 and os.environ.get("TP_FAIR", "1") == "1" and os.environ.get("TP_SEED_GATE", "1") == "1"
def _client_seed(v):
    return (abs(int(v)) % (1 << 40)) * 4096
def _smp_seed(sm, default):
    v = getattr(sm, "tp_seed", None)
    return default if v is None else v
_EXCL = {"bg": None, "on": False}
def _bg_busy():
    bg = _EXCL["bg"]
    if bg is None: return False
    try:
        return bool(len(bg._unprocessed_sequences) or len(bg._currently_processing) or len(bg._prompt_batch.uids) or len(bg._generation_batch.uids)
                    or getattr(bg, "_stash", None) is not None or getattr(bg, "_held", None))
    except Exception:
        return True
_orig_insert_segments = G.BatchGenerator.insert_segments
def _insert_segments_tp(self, *a, **k):
    _EXCL["bg"] = self
    return _orig_insert_segments(self, *a, **k)
G.BatchGenerator.insert_segments = _insert_segments_tp
def _seed_fast(self, args):
    """A seeded request the fast path can serve with the client's seed."""
    if getattr(args, "seed", None) is None or not (SEEDED and COUPLED): return False
    if not (self.model_provider.is_batchable and self.cli_args.kv_bits is None): return False
    sp = args.sampling; lg = args.logits
    # Greedy with a seed: the seed changes nothing, but argmax over 16-bit logits has exact ties that the
    # slightest rounding difference resolves the other way, and under load the hand-over variant did not
    # always repeat. It keeps the stock path there (alone, as before); sampled requests are not sensitive.
    if (sp.temperature or 0.0) == 0: return not SEED_GATE
    if (getattr(sp, "xtc_probability", 0) or 0) > 0: return False
    if lg.logit_bias or any((getattr(lg, k, 0) or 0) != 0 for k in ("repetition_penalty", "presence_penalty", "frequency_penalty")): return False
    return True
_orig_is_batchable = S.ResponseGenerator._is_batchable
def _is_batchable_tp(self, args):
    fast = _seed_fast(self, args)
    if SEED_MODE == 2 or SEED_GATE: return True if fast else _orig_is_batchable(self, args)
    if SEED_MODE == 1:
        busy = _bg_busy()
        if not busy: _EXCL["on"] = False
        if _EXCL["on"]: return False                    # a seeded request is running alone: wait for it
        if fast:                                        # start only on an idle generator, and hold it
            if not busy: _EXCL["on"] = True
            return not busy
    return _orig_is_batchable(self, args)
S.ResponseGenerator._is_batchable = _is_batchable_tp
_orig_make_sampler = S._make_sampler
def _make_sampler_tp(args, tokenizer):
    """Same sampler, plus its parameters so the speculative step can sample with shared noise."""
    smp = _orig_make_sampler(args, tokenizer)
    sp = args.sampling
    t = sp.temperature or 0.0
    if COUPLED and t > 0 and not ((getattr(sp, "xtc_probability", 0) or 0) > 0):
        from mlx_lm.sample_utils import apply_top_p, apply_min_p, apply_top_k
        chain = []
        if 0 < (sp.top_p or 0) < 1.0: chain.append(lambda x, v=sp.top_p: apply_top_p(x, v))
        if (sp.min_p or 0) != 0.0: chain.append(lambda x, v=sp.min_p: apply_min_p(x, v, 1))
        if (sp.top_k or 0) > 0: chain.append(lambda x, v=sp.top_k: apply_top_k(x, v))
        try:
            smp.tp = dict(temp=float(t), chain=chain)
            if SEEDED and getattr(args, "seed", None) is not None:
                smp.tp_seed = _client_seed(args.seed); smp.tp_alone = SEED_MODE == 1
        except AttributeError: pass
    elif t == 0:
        if SEED_MODE == 1 and getattr(args, "seed", None) is not None:
            smp = (lambda f: (lambda lp: f(lp)))(smp)           # its own object: this request decodes alone
            smp.tp_alone = True
        try: smp.greedy = True
        except AttributeError: pass
    return smp
S._make_sampler = _make_sampler_tp

def _noise(seed, index, V):
    """Gumbel noise for one output position. Sampling token = argmax(logprobs / T + noise) is exactly
    categorical sampling at temperature T; drafting with the same noise makes drafts agree with it."""
    return mx.random.gumbel(shape=(V,), key=mx.random.key(seed + index))

LOOKUP = os.environ.get("TP_LOOKUP", "1") == "1"
LOOK_MAX = min(7, int(os.environ.get("TP_LOOKUP_MAX", "7")))
LOOK_NGRAMS = (4, 3)
CHAIN_MAX = min(3, int(os.environ.get("TP_CHAIN_MAX", "3")))
from omlx.patches.mlx_lm_mtp.batch_generator import _clone_mtp_head_cache as _clone_head

def _mtp_undo(self):
    """Accepted draft rows that were never handed out are in the caches; make the token list say so."""
    st = getattr(self, "_tpm", None)
    if st is not None and st.pending:
        self.tokens[0].extend(st.pending)
        st.pending = []
    if st is not None:
        if getattr(st, "fl", None) is not None:
            fl = st.fl
            st.pre = fl.pre; st.pre_saved = fl.saved; st.pre_rows = fl.D + 1
            _pre_undo(self, st)                     # the prefix queued for the cycle after it
            st.fl = fl
            _fast_cancel(self, st, self.model)
        st.drafts = None
        _pre_undo(self, st)

def _pre_undo(self, st):
    """A prefix queued for a cycle that will not run: put the first layers' state back."""
    if getattr(st, "pre", None) is None: return
    for i, (c0, c1) in enumerate(st.pre_saved):
        c = self.prompt_cache[i]; c[0], c[1] = c0, c1
        if getattr(c, "lengths", None) is not None or getattr(c, "left_padding", None) is not None: c.advance(-st.pre_rows)
    st.pre = None; st.pre_saved = None; st.a_val = None
    _pre["gm"].__dict__.pop("_tp_pre", None)

_LK_M = 1000003; _LK_64 = (1 << 64) - 1
_LK_REUSE = os.environ.get("TP_LOOK_REUSE", "1") == "1"; _LK_POOL = []
def _lk_key(h, i, n, tok=None):
    """Hash of the n-gram ending at index i of h (or at a virtual token `tok` appended to h)."""
    if tok is None: k = h[i] & _LK_64; j0 = 1
    else: k = tok & _LK_64; j0 = 0; i = len(h)
    for j in range(1 if j0 else 1, n):
        k = (k * _LK_M + h[i - j]) & _LK_64
    return k

class _Lookup:
    """Prompt-lookup drafting: continue the most recent earlier occurrence of the last few tokens.

    The index maps a 64-bit hash of each n-gram to the position after its latest occurrence; a hit is
    checked against the tokens, so a collision is a miss. The first build over a long prompt is one
    numpy pass (18 ms at 190k tokens; the per-token Python loop took 44 ms on every request)."""
    def __init__(self, hist):
        self.hist = hist; self.idx = {n: {} for n in LOOK_NGRAMS}; self.upto = 0
        self.length = 2; self.ema = 0.6; self.cool = 0
        # The next turn of a conversation starts with almost the same tokens: take over the index of the
        # earlier request (TP_LOOK_REUSE=1) and re-index only from where the two histories part. Entries
        # left over from the old tail are harmless: every hit is checked against the tokens.
        if _LK_REUSE and len(hist) >= 8192:
            best = None; bl = 0
            for o in _LK_POOL:
                oh = o.hist; n = min(len(oh), len(hist))
                if n < 4096 or oh[:64] != hist[:64]: continue
                lo, hi = 0, n
                if oh[:n] == hist[:n]: lo = n
                else:
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if oh[lo:mid] == hist[lo:mid]: lo = mid
                        else: hi = mid - 1
                if lo > bl: best, bl = o, lo
            if best is not None and bl >= len(hist) // 2:
                _LK_POOL.remove(best); self.idx = best.idx; self.upto = max(0, min(best.upto, bl - 1)); best.idx = {n: {} for n in LOOK_NGRAMS}; best.upto = 0
            _LK_POOL.append(self)
            while len(_LK_POOL) > 6: _LK_POOL.pop(0)
    def _index(self):
        h = self.hist; end = len(h) - 1                      # never index the n-gram that ends at the last token
        if end - self.upto > 2048 and self.upto == 0:
            import numpy as _np
            a = _np.asarray(h[:end], dtype=_np.int64).astype(_np.uint64); M = _np.uint64(_LK_M)
            with _np.errstate(over="ignore"):
                for n in LOOK_NGRAMS:
                    if end < n: continue
                    k = a[n - 1:].copy()
                    for j in range(1, n): k = k * M + a[n - 1 - j:end - j]
                    self.idx[n] = dict(zip(k.tolist(), range(n, end + 1)))
            self.upto = end; return
        for i in range(self.upto, end):
            for n in LOOK_NGRAMS:
                if i + 1 >= n: self.idx[n][_lk_key(h, i, n)] = i + 1
        self.upto = max(self.upto, end)
    def draft(self, limit):
        self._index()
        if self.cool > 0: self.cool -= 1; return None
        h = self.hist; L = min(self.length, limit)
        if L < 1: return None
        for n in LOOK_NGRAMS:
            if len(h) < n: continue
            pos = self.idx[n].get(_lk_key(h, len(h) - 1, n))
            if pos is not None and h[pos - n:pos] == h[-n:]:
                cont = h[pos:pos + L]
                if cont: return cont
        return None
    def after(self, tok, limit):
        """Continuation of the history extended by `tok` (not recorded)."""
        h = self.hist; L = min(self.length, limit)
        if L < 1: return None
        for n in LOOK_NGRAMS:
            if len(h) + 1 < n: continue
            pos = self.idx[n].get(_lk_key(h, 0, n, tok))
            if pos is not None and h[pos - n:pos] == h[len(h) - n + 1:] + [tok]:
                cont = h[pos:pos + L]
                if cont: return cont
        return None
    def feedback(self, offered, accepted):
        self.ema = 0.8 * self.ema + 0.2 * (accepted / offered)
        self.length = min(LOOK_MAX, self.length + 2) if accepted == offered else max(2, accepted + 1)
        if self.ema < 0.3: self.cool = 24; self.ema = 0.5; self.length = 2

from time import perf_counter as _pc
def _strip_padding(caches):
    """A sequence prefilled on its own keeps left_padding = [0 - tokens] on its recurrent caches (the
    batch merge creates it and only padded batches clear it). It masks nothing, but its presence makes
    every linear-attention layer build and apply a mask on each step. Drop it when it is a no-op."""
    n = 0
    def visit(c):
        nonlocal n
        subs = getattr(c, "caches", None)
        if subs is not None:
            for x in subs: visit(x)
            return
        if type(c).__name__ == "ArraysCache" and getattr(c, "lengths", None) is None:   # recurrent caches only
            lp = getattr(c, "left_padding", None)
            if lp is not None and bool((lp <= 0).all().item()):
                c.left_padding = None; n += 1
    for c in caches: visit(c)
    return n
def _dump_cache(self):
    import mlx.core as _mx
    def d(c, depth=0):
        subs = getattr(c, "caches", None)
        if subs is not None: return type(c).__name__ + "[" + ", ".join(d(x, depth + 1) for x in subs) + "]"
        items = []
        for k, v in sorted(vars(c).items()):
            if isinstance(v, _mx.array): items.append(f"{k}:{tuple(v.shape)}{'' if v.size > 4 else v.tolist()}")
            elif isinstance(v, (int, float, bool, type(None))): items.append(f"{k}={v}")
            elif isinstance(v, (list, tuple)): items.append(f"{k}=[{', '.join((str(tuple(x.shape)) if isinstance(x, _mx.array) else type(x).__name__) for x in v[:4])}]")
            else: items.append(f"{k}:{type(v).__name__}")
        return type(c).__name__ + "{" + " ".join(items) + "}"
    seen = set()
    for i, c in enumerate(self.prompt_cache):
        key = type(c).__name__
        if key in seen: continue
        seen.add(key); print(f"[tp-cache] layer {i}: {d(c)[:700]}", flush=True)
CHAIN_GATE = float(os.environ.get("TP_CHAIN_GATE", "0.8" if PREK > 0 else "0.55"))
PROBE_EVERY = int(os.environ.get("TP_PROBE_EVERY", "48"))
_KG = os.environ.get("MLX_KGAP") is not None or os.environ.get("MLX_KPROF") is not None
def _kgap_mark(name):
    """Profiling build only: reset or dump the GPU idle-gap recorder."""
    k = mx.fast.metal_kernel(name=name, input_names=["a"], output_names=["out"], source="out[0] = a[0];")
    mx.eval(k(inputs=[mx.zeros((1,))], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])[0])
CONF_LOG = os.environ.get("TP_CONF_LOG", "0") == "1"
_conf_rows = []
AHEAD = os.environ.get("TP_AHEAD", "1") == "1"         # launch the next cycle before handing tokens back
def _want_depth(st):
    want = 1
    if CHAIN_MAX > 1:
        while want < CHAIN_MAX and st.ema[want] > CHAIN_GATE: want += 1
        if want < CHAIN_MAX and (st.probe + 1) % PROBE_EVERY == 0 and st.ema[want - 1] > 0.7: want += 1
    return want

FAST2_GATE = float(os.environ.get("TP_FAST2_GATE", "0.60"))   # P(both drafts accepted) above which a queued cycle carries two drafts
FAST2_PROBE = int(os.environ.get("TP_FAST2_PROBE", "32"))
CONF_GATE = float(os.environ.get("TP_CONF_GATE", "0.9"))      # keep the second draft when p(draft 1) * p(draft 2) reaches this; < 0 = off
_SLOG = os.environ.get("TP_SEEDLOG") == "1"
def _fast_depth(st):
    """Drafts for the next queued cycle: 0 = not a queued cycle, else 1 or 2."""
    if CONF_GATE >= 0 and CHAIN_MAX >= 2: return 2           # always draft two; the gate drops the second
    if CHAIN_MAX > 2 and st.ema[2] > CHAIN_GATE: return 0          # three drafts: the unqueued path
    if CHAIN_MAX < 2 or FAST2_GATE >= 1: return 1
    if st.ema[1] > FAST2_GATE: return 2
    return 2 if (st.probe + 1) % FAST2_PROBE == 0 and st.ema[0] > 0.6 else 1

def _nz(st, idx, V):
    """Noise for one output position, generated once: the draft pick and the later verify row
    of the same position share the array."""
    nz = st.noise.get(idx)
    if nz is None:
        nz = st.noise[idx] = _noise(st.seed, idx, V)
        if len(st.noise) > 24:
            for k in sorted(st.noise)[:-12]: del st.noise[k]
    return nz

def _fast_phase_a(self, st, model, trunk, out, a, dr, s_tok, tp, j0, V, _ta, D):
    """Everything of a queued cycle that can be issued before its outcome is known: the outcome
    itself (to the host), the drafts for whichever outcome it is, the next cycle's inputs, and the
    next cycle's first layers. This cycle verified L drafts; the next one will carry D."""
    L = int(dr.shape[0]); R_ = L + 1
    acc = mx.cumprod((s_tok[:L] == dr).astype(mx.int32)).sum(keepdims=True)      # [1], 0..L
    # Keep only as many drafts as every sparse cache can roll back to (known before the outcome).
    table = [int(model.mtp_clamp_accept(trunk, m, L)) for m in range(L + 1)]
    if table != list(range(L + 1)):
        acc = mx.take(mx.array(table, dtype=mx.int32), acc)
    def pick(row, off, sel):
        """The draft for the next output position after the kept rows: one argmax, with that
        position's noise chosen by the outcome."""
        if tp is not None:
            nz = mx.take(mx.stack([_nz(st, j0 + off + i, V) for i in range(R_)]), sel, axis=0)
            row = row.astype(mx.float32) * (1.0 / tp["temp"]) + nz
        return mx.argmax(row, axis=-1).astype(mx.int32)
    # The outcome goes to the host first: its read returns as soon as the verify rows are sampled,
    # while the GPU still has the draft head and the next cycle's first layers queued.
    info_raw, info = bcast2(mx.concatenate([s_tok, acc]))    # [t1 .. tR, accepted]
    mx.async_eval(info_raw, info)
    sel = info[R_:R_ + 1]
    a_n = mx.take(info[0:R_], sel)
    # Draft head over every row (its context needs each kept row); only the last kept row's
    # output goes through the vocabulary head.
    hidden = st.lm.args.hidden_size
    if st.head:
        lmod = st.lm
        hh = lmod.model.norm(out.hidden_states[-1])          # [1, R_, H]
        hc = st.head_cache
        mask = _pre["cam"](hh, hc[0], return_array=True)
        ids2 = info[0:R_][None]
        last = None
        for i, blk in enumerate(lmod.mtp):
            hh = blk(hh, lmod.model.embed_tokens, ids2, mask, hc[2 * i: 2 * i + 2]); last = blk
        hk = mx.take(hh, sel, axis=1)                        # [1, 1, H]
        _draft_ctx[0] = ("head" in _shared) and not SHARD_HEAD
        try: lg = _pre["lf"]()(lmod.lm_head, last.norm(hk))
        finally: _draft_ctx[0] = False
    else:
        _draft_ctx[0] = True
        try: lg = _shared["lf"](_shared["head"], mx.zeros((1, 1, hidden), dtype=mx.bfloat16) + sel[None, :, None].astype(mx.bfloat16) * 0)
        finally: _draft_ctx[0] = False
    ds = [pick(lg[:, 0], 1, sel)]
    def prob(row, tok):                                      # head's probability of its own draft
        z = row.astype(mx.float32) * ((1.0 / tp["temp"]) if tp is not None else 1.0)
        return mx.exp(mx.take(z[0], tok) - mx.logsumexp(z[0], axis=-1, keepdims=True))
    cf = [prob(lg[:, 0], ds[0])]
    if D == 2:
        # Second draft: one chained head step from the first. Its context copy still holds any
        # rejected rows of this cycle, which can only cost draft quality.
        if st.head:
            cc = _clone_head(st.head_cache)
            _draft_ctx[0] = ("head" in _shared) and not SHARD_HEAD
            try: lg2 = st.lm.mtp_forward(hk, ds[0][None], cc)
            finally: _draft_ctx[0] = False
        else:
            _draft_ctx[0] = True
            try: lg2 = _shared["lf"](_shared["head"], mx.zeros((1, 1, hidden), dtype=mx.bfloat16) + ds[0][None, :, None].astype(mx.bfloat16) * 0)
            finally: _draft_ctx[0] = False
        ds.append(pick(lg2[:, -1], 2, sel))
        cf.append(prob(lg2[:, -1], ds[1]))
    d_n = ds[0] if D == 1 else mx.concatenate(ds)
    conf = mx.concatenate(cf)
    if "head" not in _shared: d_n = bcast(d_n); conf = bcast(conf)
    # Its own evaluation: a read of the confidence must not wait for the prefix queued below
    # (arrays of one async_eval share one completion event).
    mx.async_eval(conf, d_n)
    caps = out.gdn_states
    saved = []
    for i in range(PREK):                                    # recurrent state after the rows that are kept
        c = trunk[i]; c0, c1 = c[0], c[1]
        for keep_rows in range(L, 0, -1):                    # kept rows = accepted + 1
            sk = caps[i].replay(keep_rows)
            c0 = mx.where(sel >= keep_rows, c0, sk[0]); c1 = mx.where(sel >= keep_rows, c1, sk[1])
        c[0], c[1] = c0, c1
        saved.append((c0, c1))
    sink_n = []
    h_emb, h4 = _pre["run"](mx.concatenate([a_n, d_n])[None], trunk, sink_n)
    mx.async_eval(h4.arrays() if hasattr(h4, "arrays") else h4)
    # Rollback states for the other recurrent layers, built (not evaluated) while the GPU works.
    lys = _pre["layers"]; nl = len(trunk)
    rec = [i for i in range(PREK, nl) if lys[i].is_linear]
    rb = [[caps[PREK + j].replay(m + 1) for j in range(len(rec))] for m in range(L)]
    return SimpleNamespace(out=out, info_raw=info_raw, a=a, a_n=a_n, d_n=d_n, saved=saved, rec=rec, rb=rb, L=L, D=D,
                           pre=(h_emb, h4, sink_n), ta=_ta, tb=_pc(), j0=j0, a_val=None, conf=conf, conf_in=None)

def _fast_launch(self, st, model, D):
    """Build and queue a whole cycle (rows: next token, its head drafts) without reading it."""
    nl = model._tp_nlayers
    trunk = self.prompt_cache[:nl]
    st.head_cache = self.prompt_cache[nl:] if st.head else None
    _ta = _pc()
    a = self._next_tokens; dr = st.drafts[1]
    R_ = 1 + int(dr.shape[0])
    inp = mx.concatenate([a.astype(mx.int32), dr])[None]
    _pre["gm"].__dict__["_tp_pre"] = st.pre; st.pre = None; st.pre_saved = None
    out = model(inp, cache=self.prompt_cache, return_hidden=True)
    logits = out.logits[0]
    sampler = (self.samplers[0] if self.samplers and self.samplers[0] is not None else self.fallback_sampler)
    tp = getattr(sampler, "tp", None)
    V = logits.shape[-1]
    j0 = st.pos + 1
    if tp is not None:
        if tp["chain"]:                                      # top-p / top-k / min-p need normalized scores
            lpf = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            for f in tp["chain"]: lpf = f(lpf)
        else:
            lpf = logits.astype(mx.float32)                  # argmax is unchanged by the normalizer
        g = mx.stack([_nz(st, j0 + i, V) for i in range(R_)])
        if _SLOG: st.__dict__.setdefault("slog", []).append(f"f{j0}x{R_}")
        s_tok = mx.argmax(lpf * (1.0 / tp["temp"]) + g, axis=-1).astype(mx.int32)
        st.coupled = True
    else:
        s_tok = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.int32)
    st.fl = _fast_phase_a(self, st, model, trunk, out, a, dr, s_tok, tp, j0, V, _ta, D)
    st.fl.a_val = st.a_val; st.a_val = None
    st.fl.conf_in = st.conf_next; st.conf_next = None

def _fast_finish(self, st, model, lp_proxy):
    """Read a queued cycle's outcome, settle the caches, and (when the next cycle can be queued
    too) queue it before returning, so the GPU works while the server handles tokens."""
    fl = st.fl; st.fl = None
    nl = model._tp_nlayers
    trunk = self.prompt_cache[:nl]; lys = _pre["layers"]
    L = fl.L; R_ = L + 1
    _tb = _pc()
    a_val = fl.a_val if fl.a_val is not None else fl.a.item()
    vals = fl.info_raw.tolist()                              # the one blocking read per cycle
    _tc = _pc()
    m = vals[R_]; st.cycles += 1; st.fast += 1
    drop = L - m
    if drop > 0:
        for i, sk in zip(fl.rec, fl.rb[m]):
            c = trunk[i]; c[0], c[1] = sk
        for i in range(nl):
            c = trunk[i]
            if lys[i].is_linear:
                if getattr(c, "lengths", None) is not None or getattr(c, "left_padding", None) is not None: c.advance(-drop)
            elif i >= PREK:
                c.trim(drop)
        if st.head:
            for c in st.head_cache: c.trim(drop)
    st.head_off += L; st.head_acc += m
    for i in range(L):                                       # P(first i+1 head drafts all accepted)
        st.ema[i] = 0.9 * st.ema[i] + 0.1 * (1.0 if m > i else 0.0)
    if L >= 2:
        st.h2[0] += 1; st.h2[1] += m >= 1; st.h2[2] += m >= 2
    k = m + 1
    st.pending = vals[:m]; st.emitted += k; st.probe += 1
    if st.look is None:
        st.look = _Lookup(list(self.tokens[0]) + [a_val]) if LOOKUP else False
    remaining = self.max_tokens[0] - self._num_tokens[0] - 1 - len(st.pending)
    if st.look:
        st.look.hist.extend(vals[:k])
        st.cand_prev = bool(st.look.draft(remaining - 2))
    if CONF_LOG and fl.conf_in is not None and R == 0:
        _conf_rows.append([round(float(v), 4) for v in fl.conf_in.tolist()] + [m])
    st.drafts = ("head", fl.d_n); st.conf_next = fl.conf
    st.pre = fl.pre; st.pre_saved = fl.saved; st.pre_rows = fl.D + 1; st.a_val = vals[m]
    st.pos = fl.j0 + m                                       # output index of the next cycle's first row
    self._next_tokens = fl.a_n
    self._next_logprobs = [_DUMMY_LP]
    self._current_tokens = fl.a
    self.tokens[0].append(a_val)
    _td = _pc(); st.tm[0] += fl.tb - fl.ta; st.tm[1] += _tc - _tb; st.tm[2] += _td - _tc
    if fl.D == 2 and CONF_GATE >= 0:
        # Confidence gate (per cycle, as in adaptive-verify engines): the second draft's row only
        # stays when the head was confident in both drafts. The prefix already ran it through the
        # first layers; dropping it here saves it from all the others.
        p = fl.conf.tolist()
        if p[0] * p[1] < CONF_GATE: _truncate_pre(self, st)
    if AHEAD and not st.cand_prev and remaining >= 10:
        D = _fast_depth(st)
        if D: _fast_launch(self, st, model, D)
    return [a_val], lp_proxy

def _truncate_pre(self, st):
    """Drop the last row of a queued three-row prefix: its recurrent states go back to the state
    after two rows and every stashed array loses its third row."""
    h_emb, h4, sink = st.pre
    for i in range(PREK):
        c = self.prompt_cache[i]
        c[0], c[1] = sink[i].replay(2)
        if getattr(c, "lengths", None) is not None or getattr(c, "left_padding", None) is not None: c.advance(-1)
    def cut(cap):
        return type(cap)(cap.layer, cap.proj[:, :2], cap.conv_state, cap.state,
                         None if cap.a_pre is None else cap.a_pre[:, :2],
                         None if cap.gate_pre is None else cap.gate_pre[:, :2])
    if hasattr(h4, "arrays"):
        y, res, post, comb, mm = h4
        h4 = type(h4)(y[:, :2], res[:, :2], post[:, :2], comb[:, :2], mm[: 2 * res.shape[2] * res.shape[3]])
    else:
        h4 = h4[:, :2]
    st.pre = (h_emb[:, :2], h4, [cut(c) for c in sink]); st.pre_rows = 2
    st.drafts = ("head", st.drafts[1][:1])
    if st.conf_next is not None: st.conf_next = st.conf_next[:1]
    st.cut += 1

def _fast_cancel(self, st, model):
    if _SLOG: st.__dict__.setdefault("slog", []).append("CANCEL")
    """A queued cycle whose tokens will never be used (generation stopped): take its rows back out
    of every cache. Its first row was sampled but, as far as the server knows, never fed."""
    fl = st.fl; st.fl = None
    nl = model._tp_nlayers; lys = _pre["layers"]
    rows = fl.L + 1
    caps = fl.out.gdn_states; j = 0
    for i in range(nl):
        c = self.prompt_cache[i]
        if lys[i].is_linear:
            cap = caps[j]; j += 1
            c[0], c[1] = cap.conv_state, cap.state
            if getattr(c, "lengths", None) is not None or getattr(c, "left_padding", None) is not None: c.advance(-rows)
        else:
            c.trim(rows)
    if st.head:
        for c in self.prompt_cache[nl:]: c.trim(rows)
    self._next_tokens = fl.a
    st.pre = None; st.pre_saved = None; st.a_val = None; st.drafts = None

# --- several requests at once ---------------------------------------------------------------------
# The server's batch path (one batch dimension, merged caches) cannot use the fused decode kernels
# and is slower than serving the requests one after another. Here concurrent sequences are rows of
# one fused forward instead, exactly like the rows of a verify block, except that every sequence
# keeps its own caches: the attention layers run per sequence (sharing each weight read where the
# kernel allows) and everything else (hyper-connections, expert layers, vocabulary head) runs once
# for all rows. One request alone keeps the speculative path.
class _BatchRows:
    """A layer's merged batch cache, used by a forward whose rows are the batch's sequences."""
    __slots__ = ("c", "tok", "rec")
    def __init__(self, c, tok=1, rec=None): self.c = c; self.tok = tok; self.rec = rec
    def __bool__(self): return True
    def __getitem__(self, i): return None
class _MultiSeq:
    """Per-sequence caches of one layer."""
    __slots__ = ("seqs",)
    def __init__(self, seqs): self.seqs = seqs
    def __bool__(self): return True
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return None
_MFL = frozenset({1, 2, 4, 8, 16, 24, 32, 40})
SHARE_SA = os.environ.get("TP_SHARE_SA", "1") == "1"
_MABL = os.environ.get("TP_MULTI_ABL", "")
_SPECTEST = os.environ.get("TP_MULTI_SPECTEST", "0") == "1"
KDA_COMPILE = os.environ.get("TP_KDA_COMPILE", "1") == "1"
_multi = {}
def _install_multiseq(model):
    import mlx.nn as nn
    from omlx.patches.glm5_next_mlx_lm import model as _MH
    Lp = _MH._language
    KDA, SA = Lp.Glm5NextLinearAttention, Lp.Glm5NextSparseAttention
    k_orig, s_orig = KDA.__call__, SA.__call__
    def k_rows(self, inputs, c, tok=1, rec=None):
        """inputs [1, B, D], one row per sequence; c is the layer's batch cache (stacked states)."""
        B = inputs.shape[1]
        fg = self.forget_gate
        kseq = _multi.get("kda_seq")
        if (kseq is not None and Lp._DECODE_FUSION and self.fuse_in and fg.safe_gate_lower_bound is not None
                and self.conv_kernel_size == 4 and c[0] is not None and c[1] is not None):
            if not self._fused_ready and self._decode_groups is None:
                self._fused_in_proj(inputs)
                if not self._fused_ready:
                    self._decode_groups = self._build_decode_groups()
            if self._fused_ready:
                proj = (Lp.fused_quantized_matmul(inputs, self._fw, self._fs, self._fb, bits=self._bits, group_size=self._gs)
                        if self._fq else inputs @ self._fw.T)
            else:
                proj = self._grouped_in_proj(inputs)
            if proj is not None:
                _, _, v_end, fa_end, ga_end = self._split_pts
                f_b, g_b = fg.f_b_proj, self.g_b_proj
                bits = {getattr(m, "bits", None) for m in (f_b, g_b)}
                a_pre = gate_pre = None
                if not (isinstance(f_b, nn.QuantizedLinear) and bits <= {4, 5, 8} and len(bits) == 1):
                    a_pre = Lp.linear_forward(f_b, proj[..., v_end:fa_end])
                    gate_pre = Lp.linear_forward(g_b, proj[..., fa_end:ga_end])
                want_mid = rec is not None and tok == 2
                r = kseq(self, proj, c[0], c[1], a_pre, gate_pre, tok, want_mid)
                _mkc[0 if r is not None else 1] += 1
                if r is not None:
                    if rec is not None:                   # entry and first-row states, for rollback
                        rec.append((c, c[0], c[1], r[3] if want_mid else None, r[4] if want_mid else None))
                    c[0] = r[1]; c[1] = r[2]; c.advance(tok)
                    return Lp.linear_forward(self.o_proj, r[0])
        if rec is not None:
            raise RuntimeError("speculative rows need the sequence-axis linear-attention kernel")
        _mkc[2] += 1
        xb = inputs[0].reshape(-1, tok, inputs.shape[-1])    # [B, tok, D]: the stock batched layer
        y = k_orig(self, xb, Lp.create_ssm_mask(xb, c), c)
        return y.reshape(1, -1, y.shape[-1])
    def k_call(self, inputs, mask=None, cache=None, **kw):
        if isinstance(cache, _BatchRows):
            if "kda" in _MABL: return inputs * 0.5         # timing experiment only
            return k_rows(self, inputs, cache.c, cache.tok, cache.rec)
        if not isinstance(cache, _MultiSeq):
            return k_orig(self, inputs, mask, cache, **kw)
        seqs = cache.seqs
        if _MABL == "kda": return inputs * 0.5            # timing experiment only
        fg = self.forget_gate
        ok = (Lp._DECODE_FUSION and self.fuse_in and fg.safe_gate_lower_bound is not None and self.conv_kernel_size == 4
              and all(getattr(c, "lengths", None) is None for c in seqs))
        proj = None
        if ok:
            if not self._fused_ready and self._decode_groups is None:
                self._fused_in_proj(inputs)
                if not self._fused_ready:
                    self._decode_groups = self._build_decode_groups()
            if self._fused_ready:
                proj = (Lp.fused_quantized_matmul(inputs, self._fw, self._fs, self._fb, bits=self._bits, group_size=self._gs)
                        if self._fq else inputs @ self._fw.T)
            else:
                proj = self._grouped_in_proj(inputs)
        if proj is not None:
            # The per-sequence kernels are a fixed graph for a given number of sequences: trace it
            # once (after one eager call has validated the kernels) so each step costs one call.
            B = len(seqs)
            _, _, v_end, fa_end, ga_end = self._split_pts
            f_b, g_b = fg.f_b_proj, self.g_b_proj
            bits = {getattr(m, "bits", None) for m in (f_b, g_b)}
            outside = not (isinstance(f_b, nn.QuantizedLinear) and bits <= {4, 5, 8} and len(bits) == 1)
            def core(proj, convs, states):
                a_pre = gate_pre = None
                if outside:
                    a_pre = Lp.linear_forward(f_b, proj[..., v_end:fa_end])
                    gate_pre = Lp.linear_forward(g_b, proj[..., fa_end:ga_end])
                ys, nc, ns = [], [], []
                for b in range(len(convs)):              # one small kernel per sequence, own state
                    r = self._kda_kernel_step(proj[:, b:b + 1], convs[b], states[b],
                                              None if a_pre is None else a_pre[:, b:b + 1],
                                              None if gate_pre is None else gate_pre[:, b:b + 1])
                    if r is None: return None
                    ys.append(r[0]); nc.append(r[1]); ns.append(r[2])
                return Lp.linear_forward(self.o_proj, mx.concatenate(ys, axis=1)), nc, ns
            convs = [c[0] for c in seqs]; states = [c[1] for c in seqs]
            res = None
            kseq = _multi.get("kda_seq")
            if kseq is not None and all(v is not None for v in convs) and all(v is not None for v in states):
                # One kernel launch for all sequences. Their states live stacked; each sequence's
                # cache holds a slice, and while nobody replaced those slices the stack is reused.
                memo = self.__dict__.get("_tp_stk")
                if memo is not None and len(memo[0]) == B and all(x is y for x, y in zip(memo[0], states)):
                    cs, ss = memo[1], memo[2]
                else:
                    cs, ss = mx.concatenate(convs, axis=0), mx.concatenate(states, axis=0)
                a_pre = gate_pre = None
                if outside:
                    a_pre = Lp.linear_forward(f_b, proj[..., v_end:fa_end])
                    gate_pre = Lp.linear_forward(g_b, proj[..., fa_end:ga_end])
                r = kseq(self, proj, cs, ss, a_pre, gate_pre)
                _mkc[0 if r is not None else 1] += 1
                if r is not None:
                    y, conv_o, st_o = r
                    nc = [conv_o[b:b + 1] for b in range(B)]; ns = [st_o[b:b + 1] for b in range(B)]
                    self.__dict__["_tp_stk"] = (ns, conv_o, st_o)
                    res = (Lp.linear_forward(self.o_proj, y), nc, ns)
            if res is None:
                _mkc[2] += 1
                res = core(proj, convs, states)
            if res is not None:
                out, nc, ns = res
                for c, conv, state in zip(seqs, nc, ns):
                    c[0] = conv; c[1] = state; c.advance(1)
                return out
        return mx.concatenate([k_orig(self, inputs[:, b:b + 1], None, c) for b, c in enumerate(seqs)], axis=1)
    # Sparse-attention layers: the input projections and the output projection read the same
    # weights for every sequence, so they run once for all rows; only the cache update, the index
    # upkeep and the attention itself run per sequence.
    dp_orig, fin_orig = SA._decode_projections, SA._finish
    def dp(self, x, cache):
        pre = self.__dict__.pop("_tp_proj", None)
        return pre if pre is not None else dp_orig(self, x, cache)
    def fin(self, flat, out_dense):
        if self.__dict__.get("_tp_hold_o") and out_dense is None:
            return flat                                  # o_proj is applied once to all sequences
        return fin_orig(self, flat, out_dense)
    SA._decode_projections = dp; SA._finish = fin
    def s_call(self, x, mask=None, cache=None, **kw):
        if isinstance(cache, _BatchRows):
            if "sparse" in _MABL: return x * 0.5          # timing experiment only
            # Sparse layers: the stock batched path (its own fused batch kernels and batch caches).
            c = cache.c
            xb = x[0].reshape(-1, cache.tok, x.shape[-1])    # [B, tok, D]
            m = Lp.create_attention_mask(xb, c[0], return_array=True)
            y = s_orig(self, xb, m, c)
            return y.reshape(1, -1, y.shape[-1])
        if not isinstance(cache, _MultiSeq):
            return s_orig(self, x, mask, cache, **kw)
        seqs = cache.seqs
        if _MABL == "sparse": return x * 0.5              # timing experiment only
        if not SHARE_SA or not Lp._DECODE_FUSION:
            return mx.concatenate([s_orig(self, x[:, b:b + 1], None, c) for b, c in enumerate(seqs)], axis=1)
        idx = self.indexer
        short = getattr(idx, "bypass_short", True)
        def selecting(c):
            done = idx._processed(c[1])
            done = max(done) if isinstance(done, list) else done
            return not (short and done + 1 <= idx.index_topk)
        sel = [selecting(c) for c in seqs]
        x_layers = [self.q_a_proj, self.kv_a_proj_with_mqa, idx.wk] + ([idx.weights_proj] if any(sel) else [])
        outs = Lp._multi_linear(x, x_layers)
        if outs is None:
            outs = [Lp.linear_forward(l, x) for l in x_layers]
        qr = self.q_a_layernorm(outs[0])
        qr_layers = [self.q_b_proj] + ([idx.wq_b] if any(sel) else [])
        qr_outs = Lp._multi_linear(qr, qr_layers) if len(qr_layers) > 1 else None
        if qr_outs is None:
            qr_outs = [Lp.linear_forward(l, qr) for l in qr_layers]
        flats = []
        self.__dict__["_tp_hold_o"] = True
        try:
            for b, c in enumerate(seqs):
                r = slice(b, b + 1)
                self.__dict__["_tp_proj"] = (
                    qr[:, r], qr_outs[0][:, r], outs[1][:, r],
                    (qr_outs[1][:, r] if sel[b] else None, outs[2][:, r], outs[3][:, r] if sel[b] else None))
                flats.append(s_orig(self, x[:, r], None, c))
        finally:
            self.__dict__.pop("_tp_hold_o", None); self.__dict__.pop("_tp_proj", None)
        return Lp.linear_forward(self.o_proj, mx.concatenate(flats, axis=1))
    KDA.__call__ = k_call; SA.__call__ = s_call
    _multi.update(Lp=Lp)
    # Experiment: route count from which the expert layers take the sorted gather path (stock: 64).
    if os.environ.get("TP_SORT_ROUTES"):
        _n = int(os.environ["TP_SORT_ROUTES"]); _st = Lp._sort_threshold
        Lp._sort_threshold = lambda *a: min(_n, _st(*a))
        print(f"[r{R}] sorted expert path from {_n} routes", flush=True)
    if os.environ.get("TP_KDA_SEQ", "1") == "1":
        import kdaseq
        _multi["kda_seq"] = kdaseq.install(list(model.layers))
        print(f"[r{R}] linear-attention kernel with a sequence axis: {_multi['kda_seq'] is not None}", flush=True)
    print(f"[r{R}] concurrent sequences run as rows of one fused forward", flush=True)

from time import perf_counter as _pc
_mstat = {}
_mev = [0.0]
_mkg = [0.0]
_mkc = [0, 0, 0]
def _multi_logits(self, inputs):
    _t0 = _pc()
    try:
        return _multi_logits_inner(self, inputs)
    finally:
        B = len(self.uids); e = _mstat.setdefault(B, [0, 0.0, _pc()]); e[0] += 1; e[1] += _pc() - _t0
        if _KG and B == int(os.environ.get("TP_KG_B", "8")):
            if e[0] == 30: _kgap_mark("kprof_reset"); _mkg[0] = _pc()
            if e[0] == 130: _kgap_mark("kprof_dump"); print(f"[r{R}] kgap multi window B={B}: 100 steps {1e3 * (_pc() - _mkg[0]):.1f} ms", flush=True)
        if e[0] % 150 == 0 and R == 0 and os.environ.get("TP_MULTI_LOG", "0") == "1":
            now = _pc(); print(f"[tp-multi] B={B}: forward call {1e3 * e[1] / 150:.2f} ms/step of which inside async_eval {1e3 * _mev[0] / 150:.2f}, step period {1e3 * (now - e[2]) / 150:.2f} ms | kda seq kernel used/declined/per-seq {_mkc}", flush=True)
            e[1] = 0.0; e[2] = now; _mev[0] = 0.0
def _rows_forward(self, rows, tok=1, rec=None):
    """All active sequences as rows of one fused forward; `tok` consecutive rows per sequence
    (sequence-major). Returns (logits [B * tok, V], final normed hidden [1, B * tok, H])."""
    lm = self.model.language_model; gm = lm.model; layers = gm.pipeline_layers; nl = len(layers)
    pc = self.prompt_cache
    h = gm.embed_tokens(rows.astype(mx.int32)[None])
    h = mx.contiguous(mx.broadcast_to(h[:, :, None, :], (1, h.shape[1], gm.hc_mult, h.shape[2])))
    for i, layer in enumerate(layers):
        h = layer(h, mask=None, cache=_BatchRows(pc[i], tok, rec))
        if (i + 1) in _MFL and i + 1 < nl:
            _te = _pc(); mx.async_eval(h.arrays() if hasattr(h, "arrays") else h); _mev[0] += _pc() - _te
    hn = gm.norm(h.mean(axis=2))
    return _multi["Lp"].linear_forward(lm.lm_head, hn)[0], hn

def _multi_logits_inner(self, inputs):
    B = len(self.uids)
    tok = 2 if (_SPECTEST and 2 * B <= 8) else 1         # timing experiment: two rows per stream
    lg, _ = _rows_forward(self, mx.repeat(inputs, 2) if tok == 2 else inputs, tok)
    return lg[::2] if tok == 2 else lg                   # [B, V]

LEAN_TAIL = os.environ.get("TP_LEAN_TAIL", "1") == "1"
def _multi_step(self):
    """One token for every active sequence. The stock step normalizes the whole vocabulary, keeps a
    log-probability array per sequence and runs one sampler per sequence, which costs milliseconds
    once several sequences are active. When every request is greedy or plain temperature sampling
    (optionally with shared noise), the next tokens are one batched argmax instead."""
    global _DUMMY_LP
    B = len(self.uids)
    samplers = [self.fallback_sampler if sm is None else sm for sm in (self.samplers or [None] * B)]
    kinds = []
    for sm in samplers:
        tp = getattr(sm, "tp", None)
        if getattr(sm, "greedy", False): kinds.append(0.0)
        elif tp is not None and not tp["chain"]: kinds.append(tp["temp"])
        else: kinds.append(None)
    if not LEAN_TAIL or any(self.logits_processors) or any(k is None for k in kinds):
        return _step(self, lambda inp: _multi_logits(self, inp))
    if _DUMMY_LP is None: _DUMMY_LP = mx.zeros((1,))
    inputs = self._next_tokens
    logits = _multi_logits(self, inputs)                     # [B, V]
    if all(k == 0.0 for k in kinds):
        nxt = mx.argmax(logits, axis=-1)
    else:
        scale = mx.array([1.0 if k == 0.0 else 1.0 / k for k in kinds], dtype=mx.float32)[:, None]
        w = mx.array([0.0 if k == 0.0 else 1.0 for k in kinds], dtype=mx.float32)[:, None]
        g = mx.random.gumbel(logits.shape)
        sd = [getattr(sm, "tp_seed", None) for sm in samplers]
        if any(v is not None for v in sd):
            V = logits.shape[-1]
            g = mx.stack([g[b] if sd[b] is None else _noise(sd[b], len(self.tokens[b]) + 1, V) for b in range(B)])
        nxt = mx.argmax(logits.astype(mx.float32) * scale + g * w, axis=-1)
    nxt = nxt.astype(inputs.dtype)
    mx.async_eval(nxt)
    self._current_tokens = inputs
    self._current_logprobs = self._next_logprobs
    self._next_tokens = nxt
    self._next_logprobs = [_DUMMY_LP] * B
    mx.eval(inputs)
    toks = inputs.tolist()
    for sti, ti in zip(self.tokens, toks):
        sti.append(ti)
    lp = _LP(_DUMMY_LP, 0.0)
    return toks, [lp] * B

def _seq_lists(b):
    """Per-sequence cache lists of a generation batch."""
    sq = getattr(b, "_seqs", None)
    if sq is not None: return sq
    return [[c.extract(i) for c in b.prompt_cache] for i in range(len(b.uids))]

def _set_seqs(self, seqs):
    for sq in seqs: _strip_padding(sq)
    if len(seqs) >= 2:
        self._seqs = seqs; self.prompt_cache = [c for sq in seqs for c in sq]
    elif len(seqs) == 1:
        self._seqs = None; self.prompt_cache = G._merge_caches(seqs); self._tp_stripped = False
    else:
        self._seqs = None; self.prompt_cache = []

def _absorb(self, batch):
    """Take another batch's sequences in (the server's own cache merge). Only called when no
    accepted-but-unsent tokens exist, so dropping the speculative state loses nothing: a queued
    cycle is taken back out first."""
    _ms_drop(self)
    for b in (self, batch):
        if getattr(b, "_tpm", None) is not None:
            _mtp_undo(b); b._tpm = None
    _orig_extend(self, batch)
    self._tp_stripped = False

_orig_extend = None
def _extend(self, batch):
    if not batch.uids:
        return
    st = getattr(self, "_tpm", None)
    if self.uids and st is not None and st.pending:      # tokens still owed to the client: join next step
        self._deferred = getattr(self, "_deferred", []) + [batch]
        return
    if not self.uids and len(batch.uids) == 1 and getattr(batch, "_seqs", None) is None:
        if getattr(batch, "_tpm", None) is not None:
            _mtp_undo(batch); batch._tpm = None
        return _orig_extend(self, batch)
    _absorb(self, batch)

# --- speculation with several requests active ----------------------------------------------------
# Two to four streams: every stream carries its own draft row, so one fused forward verifies
# [next token, draft] for each of them (2 * B rows). Rank zero samples; the outcome reaches the host
# early, while the GPU still runs the draft head for all streams. Rows of rejected drafts are taken
# out per stream afterwards (the kernel returned the recurrent state after the first row; the
# sparse and head caches trim per row). A stream emits one or two tokens per step.
MSPEC = os.environ.get("TP_MSPEC", "1") == "1"
MSPEC_MAX = int(os.environ.get("TP_MSPEC_MAX", "4"))
HEAD_ROWS = os.environ.get("TP_HEAD_ROWS", "1") == "1"
def _ms_kinds(self):
    B = len(self.uids)
    kinds = []
    for sm in (self.samplers or [None] * B):
        sm = self.fallback_sampler if sm is None else sm
        tp = getattr(sm, "tp", None)
        if getattr(sm, "greedy", False): kinds.append(0.0)
        elif tp is not None and not tp["chain"]: kinds.append(float(tp["temp"]))
        else: return None
    return kinds

def _ms_eligible(self):
    if not (MSPEC and _multi.get("kda_seq") is not None and "head" in _shared): return False
    if not 2 <= len(self.uids) <= MSPEC_MAX: return False
    if getattr(self, "_deferred", None) or any(self.logits_processors): return False
    return _ms_kinds(self) is not None

def _ms_noise(ms, b, idx, V):
    key = (ms.uids[b], idx)
    nz = ms.noise.get(key)
    if nz is None:
        nz = ms.noise[key] = _noise(ms.seeds[b], idx, V)
        if len(ms.noise) > 96:
            for k in sorted(ms.noise, key=lambda kk: kk[1])[:48]: del ms.noise[k]
    return nz

def _trim_rows(pair, trims):
    """Drop the last trims[b] rows of sequence b from a (key cache, pooling cache) pair."""
    kv, pool = pair[0], pair[1]
    u = min(trims); extra = [t - u for t in trims]
    if u: kv.trim(u)
    if any(extra):
        kv.prepare(right_padding=extra); kv.finalize()
    pool.trim_rows(trims)

def _ms_launch(self, ms):
    """Queue one step for every stream: verify forward, outcome to the host, draft head."""
    lm = self.model.language_model; gm = lm.model; nl = self.model._tp_nlayers
    Lp = _multi["Lp"]
    B = len(self.uids)
    a = self._next_tokens.astype(mx.int32)
    tok = 2 if ms.d is not None else 1
    rows = mx.stack([a, ms.d], axis=1).reshape(-1) if tok == 2 else a
    rec = []
    lg, hn = _rows_forward(self, rows, tok, rec)
    V = lg.shape[-1]; kinds = ms.kinds
    sampled = any(k > 0 for k in kinds)
    pos = [len(t) for t in self.tokens]                      # output index of each stream's first row
    z = lg.astype(mx.float32)
    if sampled:
        if ms.zero is None: ms.zero = mx.zeros((V,), dtype=mx.float32)
        scale = mx.array([1.0 if kinds[b] == 0 else 1.0 / kinds[b] for b in range(B) for _ in range(tok)], dtype=mx.float32)[:, None]
        nz = mx.stack([_ms_noise(ms, b, pos[b] + 1 + t, V) if kinds[b] > 0 else ms.zero for b in range(B) for t in range(tok)])
        z = z * scale + nz
    s_tok = mx.argmax(z, axis=-1).astype(mx.int32)           # [B * tok]
    if tok == 2:
        acc = (s_tok.reshape(B, 2)[:, 0] == ms.d).astype(mx.int32)
    else:
        acc = s_tok * 0
    info_raw, info = bcast2(mx.concatenate([s_tok, acc]))    # the host reads this; everything below stays queued
    mx.async_eval(info_raw, info)
    st2 = info[: B * tok].reshape(B, tok); accg = info[B * tok:]
    a_n = mx.take_along_axis(st2, accg[:, None], axis=1)[:, 0]
    H = hn.shape[-1]
    if getattr(lm, "mtp", None) is not None:
        hh = hn.reshape(B, tok, H); hc = self.prompt_cache[nl:]
        mask = Lp.create_attention_mask(hh, hc[0], return_array=True)
        last = None
        for i, blk in enumerate(lm.mtp):
            lay = getattr(blk, "block", None)
            if HEAD_ROWS and lay is not None:
                # The block's expert layer on [1, B * tok] rows (fused multi-row kernels); its
                # attention keeps the batched path, which needs each stream's own cache.
                x = blk.eh_proj(mx.concatenate([blk.enorm(gm.embed_tokens(st2)), blk.hnorm(hh)], axis=-1))
                x = x + lay.self_attn(lay.input_layernorm(x), mask, hc[2 * i: 2 * i + 2])
                xr = lay.post_attention_layernorm(x).reshape(1, B * tok, H)
                hh = x + lay.mlp(xr).reshape(B, tok, H)
            else:
                hh = blk(hh, gm.embed_tokens, st2, mask, hc[2 * i: 2 * i + 2])
            last = blk
        hk = mx.take_along_axis(hh, mx.broadcast_to(accg[:, None, None], (B, 1, H)), axis=1)
        _draft_ctx[0] = not SHARD_HEAD
        try: lgd = Lp.linear_forward(lm.lm_head, last.norm(hk))
        finally: _draft_ctx[0] = False
    else:
        _draft_ctx[0] = True
        try: lgd = _shared["lf"](_shared["head"], mx.zeros((B, 1, H), dtype=mx.bfloat16) + accg[:, None, None].astype(mx.bfloat16) * 0)
        finally: _draft_ctx[0] = False
    zd = lgd[:, 0].astype(mx.float32)                        # [B, V]
    if sampled:
        sc = mx.array([1.0 if k == 0 else 1.0 / k for k in kinds], dtype=mx.float32)[:, None]
        n0 = mx.stack([_ms_noise(ms, b, pos[b] + 2, V) if kinds[b] > 0 else ms.zero for b in range(B)])
        if tok == 2:
            n1 = mx.stack([_ms_noise(ms, b, pos[b] + 3, V) if kinds[b] > 0 else ms.zero for b in range(B)])
            n0 = mx.where(accg[:, None] > 0, n1, n0)
        zd = zd * sc + n0
    d_n = mx.argmax(zd, axis=-1).astype(mx.int32)
    mx.async_eval(d_n, a_n)
    ms.fl = SimpleNamespace(tok=tok, rec=rec, info_raw=info_raw, a=a, a_n=a_n, d_n=d_n, B=B)

def _ms_finish(self, ms):
    """Read the queued step's outcome, emit each stream's tokens, drop rejected rows."""
    global _DUMMY_LP
    if _DUMMY_LP is None: _DUMMY_LP = mx.zeros((1,))
    fl = ms.fl; ms.fl = None
    B, tok = fl.B, fl.tok
    nl = self.model._tp_nlayers; layers = self.model.language_model.model.pipeline_layers
    a_vals = ms.a_vals if ms.a_vals is not None else fl.a.tolist()
    _tw = _pc(); vals = fl.info_raw.tolist(); ms.tw = _pc() - _tw   # the one blocking read per step
    s = [vals[b * tok:(b + 1) * tok] for b in range(B)]; acc = vals[B * tok:]
    emitted, fin = [], [None] * B
    for b in range(B):
        out = []
        for tkn in [a_vals[b]] + ([s[b][0]] if tok == 2 and acc[b] else []):
            self._num_tokens[b] += 1; out.append(tkn)
            r = "length" if self._num_tokens[b] >= self.max_tokens[b] else None
            if self._matchers[b].advance(tkn): r = "stop"
            if r is not None:
                fin[b] = r; break
        emitted.append(out)
    trims = [tok - len(e) for e in emitted]
    if any(trims):
        m = mx.array([t == 0 for t in trims]); rew = mx.array(trims, dtype=mx.int32)
        for c, _, _, conv_mid, st_mid in fl.rec:
            c[0] = mx.where(m[:, None, None], c[0], conv_mid)
            c[1] = mx.where(m[:, None, None, None], c[1], st_mid)
            if getattr(c, "lengths", None) is not None: c.lengths = c.lengths + rew
            if getattr(c, "left_padding", None) is not None: c.left_padding = c.left_padding + rew
        for i in range(nl):
            if not layers[i].is_linear: _trim_rows(self.prompt_cache[i].caches, trims)
        hc = self.prompt_cache[nl:]
        for j in range(0, len(hc), 2): _trim_rows(hc[j:j + 2], trims)
        ms.rejects += sum(1 for t in trims if t)
    for b in range(B): self.tokens[b].extend(emitted[b])
    nxt = [s[b][acc[b] if tok == 2 else 0] for b in range(B)]
    self._current_tokens = fl.a; self._current_logprobs = self._next_logprobs
    self._next_tokens = fl.a_n; self._next_logprobs = [_DUMMY_LP] * B
    ms.d = fl.d_n; ms.a_vals = nxt
    if tok == 2: ms.offered += B; ms.accepted += sum(acc)
    lp = _LP(_DUMMY_LP, 0.0)
    resp, keep = [], []
    for b in range(B):
        for j, tkn in enumerate(emitted[b]):
            r = fin[b] if j == len(emitted[b]) - 1 else None
            resp.append(self.Response(uid=self.uids[b], token=tkn, logprobs=lp, finish_reason=r,
                                      prompt_cache=self.extract_cache(b) if r is not None else None,
                                      all_tokens=self.tokens[b] if r is not None else None))
        if fin[b] is None: keep.append(b)
    if len(keep) < B:
        ms.d = ms.d[mx.array(keep)] if keep else None
        ms.a_vals = [nxt[b] for b in keep]; ms.seeds = [ms.seeds[b] for b in keep]; ms.kinds = [ms.kinds[b] for b in keep]
        ms.busy = True
        try: self.filter(keep)
        finally: ms.busy = False
        ms.uids = list(self.uids)
    return resp

def _ms_cancel(self):
    """A queued step that will not be read (a stream joined or was removed): take its rows back out."""
    ms = getattr(self, "_ms", None)
    if ms is None or ms.fl is None: return
    fl = ms.fl; ms.fl = None
    nl = self.model._tp_nlayers; layers = self.model.language_model.model.pipeline_layers
    for c, conv_in, st_in, _, _ in fl.rec:
        c[0], c[1] = conv_in, st_in
        if getattr(c, "lengths", None) is not None or getattr(c, "left_padding", None) is not None: c.advance(-fl.tok)
    for i in range(nl):
        if not layers[i].is_linear: self.prompt_cache[i].trim(fl.tok)
    for c in self.prompt_cache[nl:]: c.trim(fl.tok)
    self._next_tokens = fl.a

def _ms_drop(self):
    ms = getattr(self, "_ms", None)
    if ms is None: return
    _ms_cancel(self)
    if R == 0 and ms.steps and os.environ.get("TP_MTP_LOG", "1") == "1":
        dt = max(_pc() - ms.t0, 1e-6)
        print(f"[tp-mspec] streams={len(ms.kinds)} steps={ms.steps} tokens={ms.emitted} total {ms.emitted / dt:.1f} tok/s "
              f"{1e3 * dt / ms.steps:.2f} ms/step drafts {ms.accepted}/{ms.offered} rows dropped {ms.rejects} | per step: build {1e3 * ms.tm[0] / ms.steps:.2f} "
              f"read wait {1e3 * ms.tm[2] / ms.steps:.2f} finish {1e3 * ms.tm[1] / ms.steps:.2f} ms | step after clean {1e3 * ms.per[0][0] / max(ms.per[0][1], 1):.2f} ms (n={ms.per[0][1]}) "
              f"after a rejection {1e3 * ms.per[1][0] / max(ms.per[1][1], 1):.2f} ms (n={ms.per[1][1]})", flush=True)
    self._ms = None

def _ms_next(self):
    ms = getattr(self, "_ms", None)
    if ms is not None and ms.uids != list(self.uids):
        _ms_drop(self); ms = None
    if ms is None:
        if getattr(self, "_tpm", None) is not None:
            _mtp_undo(self); self._tpm = None
        if not getattr(self, "_tp_stripped", False):
            _strip_padding(self.prompt_cache); self._tp_stripped = True
        B = len(self.uids)
        seeds = bcast(mx.random.randint(0, 2**30, (B,))).tolist()
        ms = self._ms = SimpleNamespace(uids=list(self.uids), d=None, a_vals=None, fl=None, kinds=_ms_kinds(self),
                                        seeds=[_smp_seed(sm, int(x) * 4096) for sm, x in zip((self.samplers or [None] * B), seeds)], noise={}, zero=None, busy=False,
                                        steps=0, emitted=0, offered=0, accepted=0, rejects=0, t0=_pc(), tm=[0.0, 0.0, 0.0], tw=0.0, per=[[0.0, 0], [0.0, 0]], last=None, lastk=0)
    if ms.fl is None: _ms_launch(self, ms)
    _t0 = _pc(); _r0 = ms.rejects
    resp = _ms_finish(self, ms)
    _t1 = _pc()
    ms.steps += 1; ms.emitted += len(resp)
    ms.tm[1] += _t1 - _t0 - ms.tw; ms.tm[2] += ms.tw
    k = 1 if ms.rejects > _r0 else 0
    if ms.last is not None: ms.per[ms.lastk][0] += _t0 - ms.last; ms.per[ms.lastk][1] += 1
    ms.last = _t0; ms.lastk = k
    if self.uids and ms.uids == list(self.uids) and _ms_eligible(self):
        _t2 = _pc(); _ms_launch(self, ms); ms.tm[0] += _pc() - _t2   # the next step runs while the server handles these tokens
    else:
        _ms_drop(self)
    return resp

_orig_gb_next = None
def _gb_next(self):
    if not self.uids:
        return []
    if _ms_eligible(self):
        return _ms_next(self)
    if getattr(self, "_ms", None) is not None: _ms_drop(self)
    return _orig_gb_next(self)

_qos_done = [False]
def _raise_qos():
    """Ask for the highest scheduling class for the generation thread (performance cores, no throttling)."""
    _qos_done[0] = True
    if os.environ.get("TP_QOS", "0") != "1": return
    try:
        import ctypes
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        r = lib.pthread_set_qos_class_self_np(0x21, 0)          # QOS_CLASS_USER_INTERACTIVE
        print(f"[r{R}] generation thread QoS user-interactive: rc={r}", flush=True)
    except Exception as e:
        print(f"[r{R}] QoS not set: {e}", flush=True)

def _mtp_step(self):
    """Speculative decoding for one sequence on a tensor-parallel group.

    Drafts come from prompt lookup (free, several tokens) when the recent tokens repeat earlier
    context, otherwise from the model's MTP head (one token). Every rank runs the same verify forward
    over [next token, drafts...]; rank zero samples, and its decisions reach the other ranks through
    one small in-graph collective, so there is one blocking read per cycle. Every emitted token is
    sampled from the full model's logits; drafts only decide how many rows a cycle verifies.
    """
    global _DUMMY_LP
    if not _qos_done[0]: _raise_qos()
    dq = getattr(self, "_deferred", None)
    if dq:
        st0 = getattr(self, "_tpm", None)
        if not (st0 is not None and st0.pending):
            self._deferred = []
            for b in dq: _absorb(self, b)
    if len(self.uids) >= 2:
        if getattr(self, "_tpm", None) is not None:
            _mtp_undo(self); self._tpm = None
        if not getattr(self, "_tp_stripped", False):
            _strip_padding(self.prompt_cache); self._tp_stripped = True
        return _multi_step(self)
    model = self.model
    usable = (
        getattr(model, "_tpm_enabled", False)
        and len(self.uids) == 1
        and not any(self.logits_processors)
    )
    st = getattr(self, "_tpm", None)
    if not usable:
        if st is not None: _mtp_undo(self)
        return _plain_step(self)
    if st is not None and st.uid != self.uids[0]:          # a new sequence: fresh draft state
        st = None
    if st is None:
        _strip_padding(self.prompt_cache)
        lm = model.language_model
        _seed = int(bcast(mx.random.randint(0, 2**30, (1,))).item()) * 4096
        _seed = _smp_seed(self.samplers[0] if self.samplers else None, _seed)
        st = self._tpm = SimpleNamespace(pending=[], drafts=None, lm=lm, head=getattr(lm, "mtp", None) is not None,
                                         head_cache=None, uid=self.uids[0], look=None, ema=[0.8, 0.5, 0.35], probe=0, seed=_seed,
                                         pre=None, pre_saved=None, pre_rows=2, a_val=None, conf_next=None, cut=0, noise={}, cand_prev=False, fast=0, fl=None, pos=0, h2=[0, 0, 0], cycles=0, emitted=0, head_off=0, head_acc=0, look_off=0, look_acc=0, look_cycles=0, t0=__import__('time').perf_counter(), tm=[0.0, 0.0, 0.0])
    if _KG and st.cycles == 20 and not st.pending: _kgap_mark("kprof_reset"); st.kg = _pc(); st.kgc = st.cycles; st.kge = st.emitted
    if _KG and st.cycles == 320 and not st.pending and getattr(st, 'kg', None):
        _kgap_mark("kprof_dump"); print(f"[r{R}] kgap window: {st.cycles - st.kgc} cycles {st.emitted - st.kge} tokens {1e3 * (_pc() - st.kg):.1f} ms", flush=True); st.kg = None
    if _DUMMY_LP is None: _DUMMY_LP = mx.zeros((1,))
    lp_proxy = [_LP(_DUMMY_LP, 0.0)]
    # Tokens accepted last cycle are already in the caches: hand them out without any model work.
    if st.pending:
        t = st.pending.pop(0)
        self.tokens[0].append(t)
        return [t], lp_proxy

    if st.fl is not None:
        return _fast_finish(self, st, model, lp_proxy)
    nl = model._tp_nlayers
    trunk = self.prompt_cache[:nl]
    st.head_cache = self.prompt_cache[nl:] if st.head else None
    a = self._next_tokens                                   # [1], sampled earlier, not fed yet
    src, d = st.drafts if st.drafts is not None else (None, None)
    if src == "look":
        dr = mx.array(d, dtype=mx.int32); L = len(d)
    elif src == "head":
        dr = d; L = int(d.shape[0])
    else:
        dr = None; L = 0
    R_ = 1 + L
    inp = a[:, None] if L == 0 else mx.concatenate([a.astype(mx.int32), dr])[None]
    _ta = _pc()
    if st.pre is not None:
        if src != "head" or R_ != st.pre_rows: raise RuntimeError("queued prefix does not match this cycle's rows")
        _pre["gm"].__dict__["_tp_pre"] = st.pre; st.pre = None; st.pre_saved = None
    out = model(inp, cache=self.prompt_cache, return_hidden=True)
    logits = out.logits[0]                                  # [rows, V]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    sampler = (self.samplers[0] if self.samplers and self.samplers[0] is not None else self.fallback_sampler)
    tp = getattr(sampler, "tp", None)
    V = logits.shape[-1]
    j0 = len(self.tokens[0]) + 1                            # output index of the token row 0 predicts
    if tp is not None:
        lpf = logprobs
        for f in tp["chain"]: lpf = f(lpf)
        g = mx.stack([_noise(st.seed, j0 + i, V) for i in range(R_)])
        if _SLOG: st.__dict__.setdefault("slog", []).append(f"s{j0}x{R_}")
        s_tok = mx.argmax(lpf * (1.0 / tp["temp"]) + g, axis=-1).astype(mx.int32)
        st.coupled = True
    else:
        s_tok = sampler(logprobs).astype(mx.int32)          # [rows]
    # Fast cycle: one head draft now and one next cycle; the draft for either outcome, the next
    # cycle's inputs and its first layers are all queued before the host reads the outcome.
    D_next = _fast_depth(st) if (_pre and src == "head" and L in (1, 2) and not st.cand_prev
                                 and (st.head or "head" in _shared)
                                 and self.max_tokens[0] - self._num_tokens[0] - 5 >= 4) else 0
    if D_next:
        st.fl = _fast_phase_a(self, st, model, trunk, out, a, dr, s_tok, tp, j0, V, _ta, D_next)
        st.fl.a_val = st.a_val; st.a_val = None
        return _fast_finish(self, st, model, lp_proxy)
    if L == 0:
        acc = s_tok[:1] * 0
    else:
        acc = mx.cumprod((s_tok[:L] == dr).astype(mx.int32)).sum(keepdims=True)
    info = bcast(mx.concatenate([s_tok, acc]))
    h = st.lm.model.norm(out.hidden_states[-1]) if st.head else None
    mx.async_eval(info, h) if h is not None else mx.async_eval(info)
    _tb = _pc()
    a_val = st.a_val if st.a_val is not None else a.item(); st.a_val = None
    vals = info.tolist()                                    # the one blocking read per cycle
    _tc = _pc()
    m = vals[-1]; st.cycles += 1
    if st.cycles == 3 and R == 0 and os.environ.get('TP_CACHE_DUMP', '0') == '1': _dump_cache(self)
    if L:
        m = model.mtp_clamp_accept(trunk, m, L)
        model.rollback_speculative_cache(trunk, out.gdn_states, m, R_)
        st.head_off += 1; st.head_acc += min(m, 1)
        if src == "head" and L >= 2:
            st.h2[0] += 1; st.h2[1] += m >= 1; st.h2[2] += m >= 2
        if src == "head":
            for i in range(L):                               # P(first i+1 head drafts all accepted)
                st.ema[i] = 0.9 * st.ema[i] + 0.1 * (1.0 if m > i else 0.0)
            st.head_off += L - 1; st.head_acc += max(m - 1, 0)
        if src == "look":
            st.look_off += L - 1; st.look_acc += max(m - 1, 0); st.look_cycles += 1
            if m >= 1: st.look.feedback(L - 1, m - 1)
    k = m + 1
    st.pending = vals[:m]
    st.emitted += k
    nxt = info[:k][None]
    # --- next cycle's drafts ---------------------------------------------------------------------
    # Row 1 is always the head's draft (it is right ~80-100% of the time). When the recent tokens
    # repeat earlier context, the rows after it continue that earlier occurrence; the head's token
    # must agree with the lookup's first token, otherwise the lookup restarts from the head's token.
    if st.look is None:
        st.look = _Lookup(list(self.tokens[0]) + [a_val]) if LOOKUP else False
    budget = self.max_tokens[0] - self._num_tokens[0] - len(st.pending) - 3
    cand = None
    if st.look:
        st.look.hist.extend(vals[:k])
        cand = st.look.draft(budget)
    def _pick(lg, q):
        """Draft token q steps ahead of the next output position: greedy, or with that position's noise."""
        row = lg[:, -1]
        if tp is not None:
            row = row.astype(mx.float32) * (1.0 / tp["temp"]) + _noise(st.seed, j0 + k + q, V)[None]
        return mx.argmax(row, axis=-1).astype(mx.int32)
    # How many head drafts to chain: the deepest row whose measured chance of being reached and
    # accepted still pays for its ~4.5 ms (a row is worth it above ~55%); probe one deeper now and then.
    depth = 1
    st.cand_prev = bool(cand)
    if CHAIN_MAX > 1 and not cand:
        while depth < CHAIN_MAX and st.ema[depth] > CHAIN_GATE: depth += 1
        st.probe += 1
        if depth < CHAIN_MAX and st.probe % 12 == 0 and st.ema[depth - 1] > 0.7: depth += 1
        depth = max(1, min(depth, budget))
    if st.head:
        _draft_ctx[0] = ("head" in _shared) and not SHARD_HEAD
        try:
            lg, hm = st.lm.mtp_forward(h[:, :k], nxt, st.head_cache, return_hidden=True, logits_keep=1)
            ds = [_pick(lg, 0)]
            if depth > 1:
                cc = _clone_head(st.head_cache); hc = hm[:, -1:]
                for q in range(1, depth):
                    lg, hh = st.lm.mtp_forward(hc, ds[-1][None], cc, return_hidden=True)
                    ds.append(_pick(lg, q)); hc = hh[:, -1:]
        finally: _draft_ctx[0] = False
        dn = ds[0] if depth == 1 else mx.concatenate(ds)
    elif "head" in _shared:
        # Mirror rank zero's drafts: receive each head output, compute this rank's share of the logits.
        ds = []
        _draft_ctx[0] = True
        try:
            dep = nxt[:, -1:, None]
            for q in range(depth):
                # Each mirrored call must depend on the previous draft, exactly as on rank zero, or the
                # two ranks could run their collectives in different orders.
                lg = _shared["lf"](_shared["head"], mx.zeros((1, 1, st.lm.args.hidden_size), dtype=mx.bfloat16) + dep.astype(mx.bfloat16) * 0)
                ds.append(_pick(lg, q))
                dep = ds[-1][:, None, None]
        finally: _draft_ctx[0] = False
        dn = ds[0] if depth == 1 else mx.concatenate(ds)
    else:
        dn = mx.zeros((depth,), dtype=mx.int32) + nxt[:, -1] * 0
    dn = dn if "head" in _shared else bcast(dn)
    mx.async_eval(dn)
    cont = None
    if cand:
        d1 = dn.item() if dn.size == 1 else int(dn[0].item())   # wait for the head only when a lookup is on offer
        if d1 == cand[0]:
            cont = cand[:LOOK_MAX]
        else:
            ext = st.look.after(d1, budget - 1)
            cont = ([d1] + ext)[:LOOK_MAX] if ext else None
    if cont and len(cont) > 1:
        st.drafts = ("look", cont)
    else:
        st.drafts = ("head", dn)
    st.pos = j0 + m
    _td = _pc(); st.tm[0] += _tb - _ta; st.tm[1] += _tc - _tb; st.tm[2] += _td - _tc
    self._next_tokens = info[m : m + 1]
    self._next_logprobs = [_DUMMY_LP]
    self._current_tokens = a
    self.tokens[0].append(a_val)
    return [a_val], lp_proxy

_orig_extract = G.GenerationBatch.extract_cache
def _extract_cache(self, idx):
    _mtp_undo(self)
    if getattr(self, "_seqs", None) is not None:
        return self._seqs[idx]
    return _orig_extract(self, idx)
_orig_filter = G.GenerationBatch.filter
def _filter(self, keep):
    _ms0 = getattr(self, "_ms", None)
    if _ms0 is not None and not _ms0.busy: _ms_drop(self)
    _mtp_undo(self)
    st = getattr(self, "_tpm", None)
    if st is not None and not keep:
        if _SLOG and R == 0 and self.uids and st.uid == self.uids[0]: print(f"[tp-seedlog] seed={st.seed} tokens={self.tokens[0][-st.emitted:] if st.emitted else []} cycles={' '.join(getattr(st, 'slog', []))}", flush=True)
        if st.cycles and os.environ.get("TP_MTP_LOG", "1") == "1" and R == 0:
            print(f"[tp-mtp] coupled={getattr(st, 'coupled', False)} tok={st.emitted} tps={st.emitted/max(__import__('time').perf_counter()-st.t0,1e-6):.1f} cycles={st.cycles} fast={st.fast} chain2={st.h2} cut={st.cut} tok/cycle={st.emitted/max(st.cycles,1):.2f} head {st.head_acc}/{st.head_off} lookup {st.look_acc}/{st.look_off} in {st.look_cycles} cycles | ms/cycle build {1e3*st.tm[0]/st.cycles:.2f} wait {1e3*st.tm[1]/st.cycles:.2f} post {1e3*st.tm[2]/st.cycles:.2f} | cache types {sorted({type(c).__name__ for c in self.prompt_cache})}", flush=True)
        if CONF_LOG and _conf_rows and R == 0:
            import json as _json
            with open("/tmp/tp_conf.jsonl", "a") as _f: _f.write(_json.dumps(_conf_rows) + "\n")
            _conf_rows.clear()
        self._tpm = None
    if getattr(self, "_seqs", None) is not None:
        seqs = [self._seqs[i] for i in keep]
        self.prompt_cache = []
        r = _orig_filter(self, keep)
        _set_seqs(self, seqs)
    else:
        r = _orig_filter(self, keep)
    dq = getattr(self, "_deferred", None)
    if dq:
        self._deferred = []
        for bb in dq: _absorb(self, bb)
    return r

if os.environ.get("TP_FASTSTEP", "1") == "1":
    G.GenerationBatch._step = _mtp_step if MTP else _step
    if MTP:
        G.GenerationBatch.extract_cache = _extract_cache
        G.GenerationBatch.filter = _filter
        _orig_extend = G.GenerationBatch.extend
        G.GenerationBatch.extend = _extend
        _orig_gb_next = G.GenerationBatch.next
        G.GenerationBatch.next = _gb_next
        G.GenerationBatch.__len__ = lambda self: len(self.uids) + sum(len(b.uids) for b in getattr(self, "_deferred", ()))


# ---- Prompt cache lookup without the quadratic search (TP_TRIE_FIX=1) --------------------------------
# Stock PromptTrie.search also looks for a cached entry LONGER than the request (useful only for caches
# that can be trimmed; this model's recurrent state cannot). It walks the tree building `extra + [tok]`
# per node, which is quadratic in the entry length: 11-12 s of pure Python on the generation thread
# whenever a new prompt shares its first tokens with a cached 100k-token conversation (every chat prompt
# shares the opening template tokens), during which every active stream is frozen.
if os.environ.get("TP_TRIE_FIX", "1") == "1":
    from mlx_lm.models import cache as _MC
    def _trie_search(self, model, tokens):
        R = _MC.PromptTrieResult
        if model not in self._trie: return R(model, None, None, None, 0)
        cur = self._trie[model]
        if not tokens and "__value__" in cur: return R(model, [], None, None, 0)
        last = -1; i = 0; n = len(tokens)
        while i < n and tokens[i] in cur:
            cur = cur[tokens[i]]
            if "__value__" in cur: last = i
            i += 1
        if last == n - 1 >= 0: return R(model, tokens, None, None, 0)
        return R(model, None, tokens[:last + 1] if last > 0 else None, None, i)
    _MC.PromptTrie.search = _trie_search

# ---- Prompt cache policy (TP_CACHE_POLICY=1) -------------------------------------------------------
# Stock: at most 10 entries, evicted oldest-first by insertion. A burst of short requests therefore
# evicts a long conversation, whose next turn re-reads everything after the system prompt (seen in
# real traffic: 205 turns at 60k+ tokens in four days, 30-60 s each). Here: short entries (< TP_CACHE_SMALL
# tokens) live in their own tier capped by count; long entries are limited by bytes only (TP_CACHE_GB);
# both tiers evict least-recently-USED (a fetch refreshes the entry). Same decisions on every rank.
if os.environ.get("TP_CACHE_POLICY", "1") == "1":
    _PC = S.LRUPromptCache
    _C_SMALL = int(os.environ.get("TP_CACHE_SMALL", "8000")); _C_NSMALL = int(os.environ.get("TP_CACHE_NSMALL", "48"))
    _C_BYTES = int(float(os.environ.get("TP_CACHE_GB", "40")) * 2**30)
    class _Order:
        def __init__(self, ordering=None):
            self._ordering = ["assistant", "user", "system"]; self._items = []
        def __len__(self): return len(self._items)
        @property
        def _lrus(self): return {k: [it for it in self._items if it[2] == k] for k in self._ordering}
        def push(self, model, tokens, cache_type="assistant"): self._items.append((model, tokens, cache_type))
        def _find(self, model, tokens):
            n = len(tokens)
            for i, it in enumerate(self._items):
                if len(it[1]) == n and it[0] == model and it[1] == tokens: return i
            return -1
        def remove(self, model, tokens):
            i = self._find(model, tokens)
            if i >= 0: del self._items[i]
        def touch(self, model, tokens):
            i = self._find(model, tokens)
            if i >= 0: self._items.append(self._items.pop(i))
        def n_small(self): return sum(len(it[1]) < _C_SMALL for it in self._items)
        def oldest(self, small):
            for i, it in enumerate(self._items):
                if small is None or (len(it[1]) < _C_SMALL) == small: return i
            return -1
        def pop(self, i=0):
            it = self._items.pop(i); return it[0], it[1]
    _PC.CacheOrder = _Order
    _pc_init = _PC.__init__
    def _pc_init2(self, max_size=10, max_bytes=1 << 63): _pc_init(self, 1 << 40, 1 << 63)
    _PC.__init__ = _pc_init2
    # Pinned entries (S._tp_pins, set by the image path): the state just before the oldest image of a long
    # conversation. Thinning keeps the newest one per conversation and byte eviction takes them last.
    def _pinned(it):
        p = getattr(S, "_tp_pins", None)
        return bool(p) and (len(it[1]), tuple(it[1][-8:])) in p
    def _pc_drop(self, i):
        model, tokens = self._lru.pop(i); e = self._trie.pop(model, tokens)
        p = getattr(S, "_tp_pins", None)
        if p: p.discard((len(tokens), tuple(tokens[-8:])))
        self._n_bytes -= e.nbytes; self._n_bytes_by_type[e.cache_type] -= e.nbytes
    # Which long entries to keep (TP_CACHE_THIN=1). One turn of a 170k-token conversation stores two
    # entries of ~2.3 GB each (end of the user message, end of the answer) and nothing removed the earlier
    # ones, so one conversation filled the cache in about eight turns and plain least-recently-used
    # eviction then dropped the shared system-prompt checkpoint (never "used" by a long conversation) and
    # other conversations before the dead copies. On inserting a long entry:
    #  - earlier checkpoints of the same conversation (entries that are a prefix of the new one) are
    #    thinned: the longest stays (the next turn starts from it), then one per TP_CACHE_SPACING of the
    #    conversation's length (default 12%), so a client that edits its history re-reads only from the
    #    nearest older checkpoint instead of from the start;
    #  - answer-end entries that branch off one of those checkpoints but are not a prefix of the new entry
    #    are dropped: the client sent the answer back in a different form, so they can never match again;
    #  - system-prompt entries are evicted last.
    # Prefix tests sample the boundary instead of comparing every token: a wrong guess only changes what
    # is kept, never what is served (lookups stay exact).
    _C_THIN = os.environ.get("TP_CACHE_THIN", "1") == "1"; _C_SPACING = float(os.environ.get("TP_CACHE_SPACING", "0.12"))
    def _pc_is_prefix(short, long_):
        n = len(short)
        if n >= len(long_): return False
        return short[-16:] == long_[n - 16:n] and short[:16] == long_[:16] and short[n // 2:n // 2 + 16] == long_[n // 2:n // 2 + 16]
    def _pc_thin(self, model, tokens):
        o = self._lru; n = len(tokens); items = o._items
        anc = [it for it in items[:-1] if it[0] == model and it[2] != "system" and _C_SMALL <= len(it[1]) < n and _pc_is_prefix(it[1], tokens)]
        if not anc: return
        anc.sort(key=lambda it: -len(it[1])); drop = []
        space = max(_C_SMALL, int(_C_SPACING * n)); mark = n
        seen_pin = False
        for j, it in enumerate(anc):
            if _pinned(it):
                if seen_pin: drop.append(it)                 # an older pin of the same conversation
                seen_pin = True; continue
            if j == 0: continue                              # the checkpoint the next turn starts from
            if len(it[1]) <= mark - space: mark = len(it[1])
            else: drop.append(it)
        parent = anc[0][1]; ppin = _pinned(anc[0])
        for it in items[:-1]:
            # Entries that continue from the parent but not into the new entry can never match again: answers
            # sent back in another form, and (after a pin) the whole old tail whose image was removed.
            if (it[0] == model and (it[2] == "assistant" or (ppin and it[2] != "system")) and len(it[1]) > len(parent) and it not in anc
                    and not _pc_is_prefix(tokens, it[1]) and _pc_is_prefix(parent, it[1])):
                drop.append(it)
        for it in drop:
            i = o._find(it[0], it[1])
            if i >= 0 and i != len(o) - 1: _pc_drop(self, i)
    _pc_insert = _PC.insert_cache
    def _pc_insert2(self, model, tokens, prompt_cache, *, cache_type="assistant"):
        _pc_insert(self, model, tokens, prompt_cache, cache_type=cache_type)
        o = self._lru
        while o.n_small() > _C_NSMALL: _pc_drop(self, o.oldest(True))
        if _C_THIN and len(tokens) >= _C_SMALL:
            try: _pc_thin(self, model, tokens)
            except Exception as e: print(f"[tp-cache] thinning skipped: {e!r}", flush=True)
        while self._n_bytes > _C_BYTES and len(o) > 1:
            i = next((j for j, it in enumerate(o._items[:-1]) if len(it[1]) >= _C_SMALL and it[2] != "system" and not _pinned(it)), -1)
            if i < 0: i = next((j for j, it in enumerate(o._items[:-1]) if len(it[1]) >= _C_SMALL and it[2] != "system"), -1)
            if i < 0: i = o.oldest(False)                # only system prompts and the new entry are long
            if i < 0 or i == len(o) - 1: i = 0           # no other long entry left: oldest of any kind
            if i == len(o) - 1: break                    # never the entry just inserted
            _pc_drop(self, i)
    _PC.insert_cache = _pc_insert2
    if os.environ.get("TP_CACHELOG") == "1":
        def _pc_insert3(self, model, tokens, prompt_cache, *, cache_type="assistant"):
            _pc_insert2(self, model, tokens, prompt_cache, cache_type=cache_type)
            print(f"[tp-cachelog] insert {cache_type} {len(tokens)} | bytes {self._n_bytes / 2**30:.1f} GiB | entries "
                  + " ".join(f"{it[2][0]}{len(it[1])}{'*' if _pinned(it) else ''}" for it in self._lru._items if len(it[1]) >= _C_SMALL), flush=True)
        _PC.insert_cache = _pc_insert3
    _pc_fetch = _PC.fetch_nearest_cache
    def _pc_fetch2(self, model, tokens):
        cache, rest = _pc_fetch(self, model, tokens)
        if cache is not None: self._lru.touch(model, tokens[:len(tokens) - len(rest)])
        if os.environ.get("TP_CACHELOG") == "1" and len(tokens) >= _C_SMALL: print(f"[tp-cachelog] fetch {len(tokens)} -> reuse {len(tokens) - len(rest)}", flush=True)
        return cache, rest
    _PC.fetch_nearest_cache = _pc_fetch2
    def _pc_trim_to(self, *, n_sequences=None, n_bytes=None): pass
    _PC.trim_to = _pc_trim_to

    # Stock _tokenize finds the end of the system prompt by the first token where "system + empty user
    # message" differs from the real prompt. With this template there is often no difference (the short
    # form is a clean prefix), so no system checkpoint is saved and a new conversation with the same
    # system prompt re-reads all of it. Use the common prefix in that case.
    _RG = next(c for c in vars(S).values() if isinstance(c, type) and "_tokenize" in vars(c) and "_generate" in vars(c))
    _rg_tok = _RG._tokenize
    _SYS_MIN = int(os.environ.get("TP_SYS_CKPT_MIN", "512"))
    def _rg_tok2(self, tokenizer, request, args):
        out = _rg_tok(self, tokenizer, request, args)
        try:
            prompt, segments, types, init, vision = out
            msgs = getattr(request, "messages", None)
            if (request.request_type == "chat" and vision is None and types and types[0] == "user" and msgs
                    and msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"):
                n = 0
                while msgs[n]["role"] == "system": n += 1
                cta = dict(self.model_provider.cli_args.chat_template_args)
                if args.chat_template_kwargs: cta.update(args.chat_template_kwargs)
                st = tokenizer.apply_chat_template(msgs[:n] + [{"role": "user", "content": ""}], add_generation_prompt=False,
                                                   tools=request.tools, tokenize=True, **cta)
                k = 0; m = min(len(st), len(segments[0]) - 1)
                while k < m and st[k] == prompt[k]: k += 1
                if k >= _SYS_MIN:
                    segments = [segments[0][:k], segments[0][k:]] + segments[1:]; types = ["system"] + types
                    return prompt, segments, types, init, vision
        except Exception as e:
            print(f"[tp-cache] system checkpoint split skipped: {e!r}", flush=True)
        return out
    _RG._tokenize = _rg_tok2


# ---- Prompt processing shares the GPU with active streams (TP_FAIR=1) -------------------------------
# Stock scheduling runs ONE decode step per prompt block, so every active stream stalls (about one token
# per second at 2,048-token blocks) while a long prompt loads. Here, while streams are decoding, a long
# prompt is read in smaller blocks (TP_FAIR_CHUNK tokens) with enough decode steps in between to take
# about TP_FAIR_MS of GPU time, i.e. roughly half the machine each. Counted in steps, not wall time, so
# every rank makes the same choice. With no active stream the full block size is used.
if os.environ.get("TP_FAIR", "1") == "1":
    _F_CHUNK = int(os.environ.get("TP_FAIR_CHUNK", "1024")); _F_MS = float(os.environ.get("TP_FAIR_MS", "450"))
    _F_FULL = int(os.environ.get("TP_PREFILL_STEP", "4096")); _F_STEP_MS = {1: 13.0, 2: 20.0, 3: 26.0, 4: 30.0}
    # Short prompts jump the queue (TP_PREEMPT=1): prompts are read one at a time, so without this a
    # follow-up turn with a few new tokens waits for a whole 100k-token prompt ahead of it. When a long
    # prompt is in progress and a queued request has at most TP_PREEMPT_MAX tokens left to read, the long
    # prompt's batch is set aside (self._stash), the short one is read and starts generating, then the
    # long one resumes. remove()/extract_cache() look in the set-aside batch too.
    _PRE = os.environ.get("TP_PREEMPT", "1") == "1"; _PRE_MAX = int(os.environ.get("TP_PREEMPT_MAX", "4096"))
    def _left(seq): return sum(len(x) for x in seq)
    def _promote_small(self):
        if len(self._generation_batch) >= self.completion_batch_size: return False
        for i, q in enumerate(self._unprocessed_sequences):
            if _left(q[1]) <= _PRE_MAX:
                if i:
                    del self._unprocessed_sequences[i]; self._unprocessed_sequences.appendleft(q)
                return True
        return False
    def _swap(self):
        st = self._stash; self._stash = (self._prompt_batch, self._currently_processing)
        self._prompt_batch, self._currently_processing = st
    _F_DBG = os.environ.get("TP_FAIR_DEBUG") == "1"
    _bg_next0 = G.BatchGenerator._next
    # Requests that must decode alone (seeded, TP_SEEDED=1). A request whose prompt is read (one token
    # left, which starts generation) is set aside in self._held, in arrival order, while it may not start:
    # the generation batch holds a decode-alone request, or it is one itself and the batch is not empty,
    # or somebody is already waiting. The head of the line starts as soon as the rule allows; with an
    # empty generation batch it always does, so nothing can wait forever.
    def _alone(sm): return getattr(sm, "tp_alone", False)
    def _may_start(self, sm):
        g = self._generation_batch
        gs = list(g.samplers or [])
        for d in (getattr(g, "_deferred", None) or []): gs += list(d.samplers or [])     # joining at the next step
        if not gs and len(g) == 0: return True
        if len(gs) >= self.completion_batch_size: return False
        return not _alone(sm) and not any(_alone(x) for x in gs)
    # A long prompt takes turns with decode-alone requests. Short prompts jump ahead of a long one and no
    # prompt is read while a decode-alone request generates, so a steady flow of seeded jobs would leave a
    # long prompt (an agent request) unread for as long as the flow lasts. Each decode-alone generation
    # therefore owes the long prompt one block (self._long_due) before the next such request starts.
    def _long_pending(self):
        if not getattr(self, "_long_due", False): return False
        st = getattr(self, "_stash", None)
        return (st is not None and any(_left(q[0]) > _PRE_MAX for q in st[1])) or any(_left(q[0]) > _PRE_MAX for q in self._currently_processing)
    def _seed_gate(self):
        held = getattr(self, "_held", None)
        if held is None: held = self._held = []
        out = []
        cp = self._currently_processing
        if len(cp) == 1 and len(cp[0][0]) == 1 and len(cp[0][0][0]) == 1 and len(self._prompt_batch) == 1:
            sm = self._prompt_batch.samplers[0] if self._prompt_batch.samplers else None
            if held or not _may_start(self, sm) or (_alone(sm) and _long_pending(self)):
                held.append((self._prompt_batch, cp))
                self._prompt_batch = G.PromptProcessingBatch.empty(self.model, self.sampler, prefill_step_size=self.prefill_step_size)
                self._currently_processing = []
        # A queued request with one token left (its prompt is fully cached) would be made into a batch and
        # start generating inside one stock call, past the check above: take those out of the queue here.
        if self._unprocessed_sequences and any(len(q[1]) == 1 and len(q[1][0]) == 1 for q in self._unprocessed_sequences):
            rest = []
            for q in list(self._unprocessed_sequences):
                if len(q[1]) == 1 and len(q[1][0]) == 1 and (held or not _may_start(self, q[5]) or (_alone(q[5]) and _long_pending(self))):
                    cur = self._currently_processing; self._currently_processing = []
                    self._unprocessed_sequences = type(self._unprocessed_sequences)([q])
                    pb = self._make_batch(1)
                    held.append((pb, self._currently_processing)); self._currently_processing = cur
                else:
                    rest.append(q)
            self._unprocessed_sequences = type(self._unprocessed_sequences)(rest)
        # Arrival order. While a decode-alone request generates no prompts are read, so requests that arrive
        # with their prompt already cached, or that were read earlier, could keep passing one whose prompt
        # still has to be read. The head of the line therefore also waits for any earlier request (lower uid)
        # with a short prompt left to read; long prompts are exempt, as with TP_PREEMPT.
        if len(held) > 1: held.sort(key=lambda h: h[0].uids[0])
        while held:
            pb, cp = held[0]
            if not _may_start(self, pb.samplers[0] if pb.samplers else None): break
            if _alone(pb.samplers[0] if pb.samplers else None) and _long_pending(self): break
            u0 = pb.uids[0]
            if any(u < u0 and 1 < _left(q[0]) <= _PRE_MAX for u, q in zip(self._prompt_batch.uids, self._currently_processing)) \
               or any(q[0] < u0 and 1 < _left(q[1]) <= _PRE_MAX for q in self._unprocessed_sequences): break
            held.pop(0)
            last = [cp[0][0][0]]; prog = (cp[0][2],) * 2
            gen = pb.split([0]).generate(last)
            out.append(G.PromptProcessingBatch.Response(gen.uids[0], prog, True, True))
            self._generation_batch.extend(gen)
        return out
    def _bg_next(self):
        if SEED_GATE:
            pre = _seed_gate(self)
            if pre:
                r = _bg_next1(self)
                return pre + list(r[0]), r[1]
        return _bg_next1(self)
    def _bg_next1(self):
        if SEED_GATE and len(self._generation_batch) > 0 and any(_alone(x) for x in (self._generation_batch.samplers or [])):
            # A decode-alone request is generating: no prompt blocks in between either (they shift its
            # cycles, and with them the rounding). It is short work; the prompts resume right after.
            self._long_due = True
            tic = _pc(); r = self._generation_batch.next()
            self._counters.decode_time += _pc() - tic; self._counters.generation_tokens += len(r); self._counters.generation_steps += 1
            return [], r
        st = getattr(self, "_stash", None)
        if _F_DBG: print(f"[tp-fair] gen={len(self._generation_batch)} cur={[_left(q[0]) for q in self._currently_processing]} wait={[_left(q[1]) for q in self._unprocessed_sequences]} stash={st is not None}", flush=True)
        if st is not None:
            if len(self._prompt_batch) == 0 and not self._currently_processing and (_long_pending(self) or not _promote_small(self)):
                self._prompt_batch, self._currently_processing = st; self._stash = None
        elif (_PRE and self._unprocessed_sequences and self._currently_processing and not _long_pending(self)
              and any(_left(seq[0]) > _PRE_MAX for seq in self._currently_processing) and _promote_small(self)):
            self._stash = (self._prompt_batch, self._currently_processing)
            self._prompt_batch = G.PromptProcessingBatch.empty(self.model, self.sampler, prefill_step_size=self.prefill_step_size)
            self._currently_processing = []
        nb = len(self._generation_batch)
        if nb > 0:
            if any(_left(seq[0]) > _F_CHUNK for seq in self._currently_processing):
                self._fair_i = getattr(self, "_fair_i", 0) + 1
                if self._fair_i % max(1, round(_F_MS / _F_STEP_MS.get(nb, 29.0))) != 0:
                    tic = _pc(); r = self._generation_batch.next()
                    self._counters.decode_time += _pc() - tic; self._counters.generation_tokens += len(r); self._counters.generation_steps += 1
                    return [], r
            step = _F_CHUNK
        else:
            step = _F_FULL
        self.prefill_step_size = step; self._prompt_batch.prefill_step_size = step
        was_long = any(_left(seq[0]) > _PRE_MAX for seq in self._currently_processing)
        p0 = self._counters.prompt_tokens; r = _bg_next0(self)
        if was_long and self._counters.prompt_tokens > p0: self._long_due = False
        if self._counters.prompt_tokens - p0 >= 256: _F_BREAK[0] = True
        return r
    # In distributed mode the server calls next() a fixed number of times (tuned so decode steps fill
    # ~0.5 s) before it looks at the request queue again. With prompt blocks in the loop that is many
    # seconds to a whole prompt, during which new requests are not even queued. Leave the loop after
    # every sizeable prompt block (same decision on every rank).
    _F_BREAK = [False]; _tb_next0 = S.TimeBudget.__next__
    def _tb_next(self):
        if _F_BREAK[0]:
            _F_BREAK[0] = False; raise StopIteration()
        return _tb_next0(self)
    S.TimeBudget.__next__ = _tb_next
    _bg_remove0 = G.BatchGenerator.remove; _bg_extract0 = G.BatchGenerator.extract_cache
    def _held_each(self, fn):
        """Run fn with each set-aside (held) prompt batch swapped in as the current one."""
        held = getattr(self, "_held", None) or []
        for i in range(len(held)):
            cur = (self._prompt_batch, self._currently_processing)
            self._prompt_batch, self._currently_processing = held[i]
            try: fn()
            finally:
                held[i] = (self._prompt_batch, self._currently_processing)
                self._prompt_batch, self._currently_processing = cur
        if held: self._held = [h for h in held if len(h[0]) > 0]
    def _bg_remove(self, uids, return_prompt_caches=False):
        r = _bg_remove0(self, uids, return_prompt_caches)
        if getattr(self, "_held", None):
            acc = {}
            if return_prompt_caches: _held_each(self, lambda: acc.update(_bg_extract0(self, uids)))
            _held_each(self, lambda: _bg_remove0(self, uids, False))
            if acc: r = {**acc, **(r or {})}
        if getattr(self, "_stash", None) is not None:
            _swap(self)
            try: r2 = _bg_remove0(self, uids, return_prompt_caches)
            finally: _swap(self)
            if r2: r = {**r2, **r}
        return r
    def _bg_extract(self, uids):
        r = _bg_extract0(self, uids)
        if getattr(self, "_held", None):
            rest = [u for u in uids if u not in r]
            if rest: _held_each(self, lambda: r.update(_bg_extract0(self, rest)))
        if getattr(self, "_stash", None) is not None:
            rest = [u for u in uids if u not in r]
            if rest:
                _swap(self)
                try: r.update(_bg_extract0(self, rest))
                finally: _swap(self)
        return r
    G.BatchGenerator.remove = _bg_remove; G.BatchGenerator.extract_cache = _bg_extract
    G.BatchGenerator._next = _bg_next


# ---- Tokenize only what is new (TP_TOKMEMO=1) -------------------------------------------------------
# Every turn of a conversation re-tokenizes the whole rendered prompt: 75 ms at 190k tokens, and the
# server renders the system prompt twice more to find its end (20 ms each), so about 115 ms per turn
# before any GPU work. The rendered text of turn N+1 starts with almost all of turn N's text. A role
# marker (<|system|>, <|user|>, <|assistant|>, <|observation|>) is an added special token: the tokenizer
# splits the text there before anything else, so the tokens before a marker do not depend on what
# follows it. So: find the longest common prefix with a remembered text, cut at the last role marker
# inside it, reuse the remembered tokens up to there and encode only the rest.
# TP_TOKMEMO_CHECK=1 compares every result with a full encode (testing).
if os.environ.get("TP_TOKMEMO", "1") == "1":
    import bisect as _bis
    _TM_MAX = int(os.environ.get("TP_TOKMEMO_N", "8")); _TM_MIN = 20000; _TM_CHECK = os.environ.get("TP_TOKMEMO_CHECK") == "1"
    _TM = []                                              # entries [text, ids, cut_chars, cut_toks], most recently used last
    _TM_ROLES = ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"); _TM_IDS = {}
    def _tm_lcp(a, b):
        n = min(len(a), len(b))
        if a[:n] == b[:n]: return n
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[lo:mid] == b[lo:mid]: lo = mid
            else: hi = mid - 1
        return lo
    def _tm_encode(hf, text):
        if len(text) < _TM_MIN: return hf.encode(text, add_special_tokens=False)
        bk = hf.backend_tokenizer
        if id(hf) not in _TM_IDS:
            ok = all((t := hf.added_tokens_decoder.get(hf.convert_tokens_to_ids(r))) is not None and t.special and not t.lstrip and not t.rstrip for r in _TM_ROLES)
            _TM_IDS[id(hf)] = frozenset(hf.convert_tokens_to_ids(r) for r in _TM_ROLES) if ok else None
        roles = _TM_IDS[id(hf)]
        if roles is None: return hf.encode(text, add_special_tokens=False)
        best = None; bl = 0; head = text[:64]
        for e in _TM:
            if e[0][:64] != head: continue
            if len(e[0]) == len(text) and e[0] == text:
                _TM.remove(e); _TM.append(e); return list(e[1])
            l = _tm_lcp(e[0], text)
            if l > bl: best, bl = e, l
        off = 0; ti = 0
        if best is not None:
            k = _bis.bisect_right(best[2], bl) - 1         # last role marker starting at or before the common prefix end
            if k >= 0 and best[2][k] > 0: off = best[2][k]; ti = best[3][k]
        enc = bk.encode(text[off:] if off else text, add_special_tokens=False)
        tail = enc.ids; offs = enc.offsets
        if off:
            ids = best[1][:ti] + tail; cc = best[2][:k]; ct = best[3][:k]
        else:
            ids = list(tail); cc = []; ct = []
        for j, t in enumerate(tail):
            if t in roles: cc.append(off + offs[j][0]); ct.append(ti + j)
        if _TM_CHECK:
            full = hf.encode(text, add_special_tokens=False)
            print(f"[tp-tokmemo] {len(text)} chars, reused {ti} of {len(ids)} tokens, {'OK' if list(full) == ids else 'MISMATCH'}", flush=True)
            if list(full) != ids: return full
        if best is not None and bl >= len(best[0]) - 8192 and len(text) >= bl: _TM.remove(best)      # the conversation grew: replace its entry
        _TM.append([text, ids, cc, ct])
        while len(_TM) > _TM_MAX: _TM.pop(0)
        return list(ids)
    from mlx_lm.tokenizer_utils import TokenizerWrapper as _TWc
    _tw_act = _TWc.apply_chat_template
    def _tw_act2(self, *args, tokenize=True, **kwargs):
        if not tokenize: return _tw_act(self, *args, tokenize=False, **kwargs)
        text = _tw_act(self, *args, tokenize=False, **kwargs)
        if not isinstance(text, str): return _tw_act(self, *args, tokenize=True, **kwargs)
        return _tm_encode(self._tokenizer, text)
    _TWc.apply_chat_template = _tw_act2

# ---- Never hang (TP_WATCHDOG_S, default 150; 0 = off) -----------------------------------------------
# Seen live: one rank failed a command buffer, the other waited for its sum forever; /health and
# /v1/models kept answering, so the supervisor did not restart anything (19 minutes of silence).
# Two rules: a generation thread that dies takes the process with it, and a generation loop that has
# not turned for TP_WATCHDOG_S seconds does the same. The loop turns every 0.1 s when idle and at least
# once per prompt block or decode step when busy; the longest legitimate pauses are first-use kernel
# compiles (~15 s) and a vision tower pass (~5 s). The supervisor restarts both ranks on exit.
_WD = float(os.environ.get("TP_WATCHDOG_S", "150"))
if _WD > 0:
    import threading as _th, time as _tm
    _wd_last = [None]
    def _tp_beat(): _wd_last[0] = _tm.monotonic()
    S._tp_beat = _tp_beat
    _RGc = next(c for c in vars(S).values() if isinstance(c, type) and "_generate" in vars(c) and "_next_request" in vars(c))
    _wd_nr = _RGc._next_request
    def _wd_next_request(self, timeout=None):      # timeout None = do not wait; 0.1 = idle poll
        _tp_beat(); r = _wd_nr(self, timeout); _tp_beat(); return r
    _RGc._next_request = _wd_next_request
    _wd_bn = G.BatchGenerator._next
    def _wd_bg_next(self, *a, **k):
        _tp_beat(); r = _wd_bn(self, *a, **k); _tp_beat(); return r
    G.BatchGenerator._next = _wd_bg_next
    _wd_run = _RGc._run_generate
    def _wd_run_generate(self):
        _wd_run(self)
        if getattr(self, "_generation_failed", False):
            print("[tp] fatal: generation thread died; exiting so the supervisor restarts the server", flush=True)
            os._exit(71)
    _RGc._run_generate = _wd_run_generate
    def _wd_loop():
        while True:
            _tm.sleep(5)
            t = _wd_last[0]
            if t is not None and _tm.monotonic() - t > _WD:
                print(f"[tp] fatal: generation loop silent for {_tm.monotonic() - t:.0f} s; exiting so the supervisor restarts the server", flush=True)
                os._exit(72)
    _th.Thread(target=_wd_loop, daemon=True).start()

# ---- Request-path trace (TP_TRACE=1): where the time before the first token goes --------------------
if os.environ.get("TP_TRACE") == "1" and g.rank() == 0:
    import time as _tt
    _T0 = [_tt.perf_counter()]
    def _tr(name, cls, attr, extra=None):
        f0 = getattr(cls, attr)
        def w(*a, **k):
            t = _tt.perf_counter(); r = f0(*a, **k); d = _tt.perf_counter() - t
            if d > 0.002 or extra: print(f"[tp-trace] +{1e3 * (t - _T0[0]):8.1f} ms {name:22s} {1e3 * d:7.1f} ms {extra(a, k, r) if extra else ''}", flush=True)
            return r
        setattr(cls, attr, w)
    _RGc = next(c for c in vars(S).values() if isinstance(c, type) and "_tokenize" in vars(c) and "_generate" in vars(c))
    def _mark(a, k, r):
        if r is not None: _T0[0] = _tt.perf_counter()
        return "REQUEST" if r is not None else ""
    _nr0 = _RGc._next_request
    def _nr(self, timeout=None):
        t = _tt.perf_counter(); r = _nr0(self, timeout)
        if r is not None:
            print(f"[tp-trace] ---- request shared in {1e3 * (_tt.perf_counter() - t):.1f} ms (includes idle wait up to 100 ms)", flush=True); _T0[0] = _tt.perf_counter()
        return r
    _RGc._next_request = _nr
    _tr("tokenize", _RGc, "_tokenize")
    _tr("cache fetch", S.LRUPromptCache, "fetch_nearest_cache")
    _tr("cache insert", S.LRUPromptCache, "insert_cache")
    _tr("batch.next", G.BatchGenerator, "_next", lambda a, k, r: f"prompt_resp={len(r[0])} gen_resp={len(r[1])} cur={[sum(len(x) for x in q[0]) for q in a[0]._currently_processing]}")
    _tr("extract_cache", G.BatchGenerator, "extract_cache")
    _tr("remove", G.BatchGenerator, "remove")
    _tr("prompt()", G.PromptProcessingBatch, "prompt")
    _tr("split->generate", G.PromptProcessingBatch, "generate")
    if os.environ.get("TP_PROF") == "1":
        import cProfile, pstats, io as _io
        def _prof(cls, attr, minms=40):
            f0 = getattr(cls, attr)
            def w(*a, **k):
                pr = cProfile.Profile(); t = _tt.perf_counter(); pr.enable(); r = f0(*a, **k); pr.disable()
                if 1e3 * (_tt.perf_counter() - t) >= minms:
                    b = _io.StringIO(); pstats.Stats(pr, stream=b).sort_stats("cumulative").print_stats(22); print(f"[tp-prof] {attr}\n" + "\n".join(l[:170] for l in b.getvalue().splitlines()[4:34]), flush=True)
                return r
            setattr(cls, attr, w)
        _prof(G.PromptProcessingBatch, "generate"); _prof(_RGc, "_tokenize")
    if os.environ.get("TP_PROF") == "2":                  # profile of all decode steps, printed every TP_PROF_N calls
        import cProfile, pstats, io as _io
        _pp = [cProfile.Profile(), 0]; _pn0 = G.BatchGenerator._next; _PN = int(os.environ.get("TP_PROF_N", "150"))
        def _pnext(self, *a, **k):
            _pp[0].enable(); r = _pn0(self, *a, **k); _pp[0].disable()
            if r[1]: _pp[1] += 1
            if _pp[1] >= _PN:
                b = _io.StringIO(); pstats.Stats(_pp[0], stream=b).sort_stats("tottime").print_stats(28); print(f"[tp-prof] {_PN} decode steps\n" + "\n".join(l[:150] for l in b.getvalue().splitlines()[4:40]), flush=True)
                _pp[0] = cProfile.Profile(); _pp[1] = 0
            return r
        G.BatchGenerator._next = _pnext
    _tr("gen.extend", G.GenerationBatch, "extend")
    _tr("insert_segments", G.BatchGenerator, "insert_segments")


# ---- No buffer-cache flush after small prompt blocks (TP_CLEAR_MIN) ---------------------------------
# mlx_lm frees MLX's buffer cache after every prompt block. For a block of a few hundred tokens that
# costs more than it saves: the next forward (and the vision tower of the next image) re-allocates
# everything (~50-100 ms per small request). Keep the flush for blocks of TP_CLEAR_MIN tokens or more.
_CLEAR_MIN = int(os.environ.get("TP_CLEAR_MIN", "1024"))
if _CLEAR_MIN > 0:
    _pp_prompt0 = G.PromptProcessingBatch.prompt
    def _pp_prompt(self, tokens):
        if tokens and max(len(t) for t in tokens) >= _CLEAR_MIN: return _pp_prompt0(self, tokens)
        keep = mx.clear_cache; mx.clear_cache = lambda: None
        try: return _pp_prompt0(self, tokens)
        finally: mx.clear_cache = keep
    G.PromptProcessingBatch.prompt = _pp_prompt

try:
    info = mx.device_info(); mx.set_cache_limit(int(info["memory_size"]))
except Exception: pass
sys.argv = ["mlx_lm.server", "--model", path, "--host", "0.0.0.0", "--port", os.environ.get("TP_PORT", "8010"),
            "--max-tokens", "32768", "--decode-concurrency", os.environ.get("TP_CONC", "1"), "--prompt-concurrency", os.environ.get("TP_PROMPT_CONC", os.environ.get("TP_CONC", "1")),
            # Same template settings the single-box oMLX deployment used (model_settings.json).
            "--chat-template-args", os.environ.get("TP_CHAT_TEMPLATE_ARGS", '{"reasoning_effort": "low", "clear_thinking": false}')] \
           + ["--prefill-step-size", os.environ.get("TP_PREFILL_STEP", "4096")] + sys.argv[1:]
print(f"[r{g.rank()}] starting mlx_lm.server", flush=True)
# First request after a start pays ~14 s once (kernels compile, weights are touched for the first time).
# Send it ourselves as soon as the port answers (TP_WARMUP=1), so the first real request does not.
if os.environ.get("TP_WARMUP", "1") == "1" and g.rank() == 0:
    def _warm():
        import json as _j, time as _t, urllib.request as _u
        url = f"http://127.0.0.1:{os.environ.get('TP_PORT', '8010')}/v1/chat/completions"
        body = _j.dumps(dict(model="default_model", max_tokens=8, temperature=0, messages=[{"role": "user", "content": "Say OK."}])).encode()
        for _ in range(600):
            _t.sleep(1)
            try:
                t = _t.time(); _u.urlopen(_u.Request(url, body, {"Content-Type": "application/json"}), timeout=300).read()
                print(f"[tp] warm-up request done in {_t.time() - t:.1f} s", flush=True); return
            except Exception:
                pass
    import threading as _wth; _wth.Thread(target=_warm, daemon=True).start()
S.main()
