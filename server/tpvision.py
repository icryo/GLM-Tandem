# Image input for the lean tensor-parallel server.
#
# The bundled mlx_lm.server already routes requests with images (data URLs, shared with every
# rank) through `processor(text, images)` and `model.embed_inputs(tokens, pixel_values=...)`.
# For GLM-5.3 the cluster model class is text-only, so this adds the three missing pieces on
# every rank: the vision tower (1.05 GiB), the image processor, and the embedding splice.
# The tower runs once per image at prompt time; decoding is unchanged.
import copy, json, os
import numpy as np
import mlx.core as mx

_PLACEHOLDER = "<|tp_image_placeholder|>"


def install(S, model, model_path, rank):
    from mlx_vlm.models.glm5_next import VisionConfig, VisionModel
    from mlx_vlm.models.glm5_next.processing import Glm5NextImageProcessor, _image_processor_kwargs

    cfg = json.load(open(os.path.join(model_path, "config.json")))
    vcfg = cfg.get("vision_config")
    if not vcfg:
        return False
    vc = VisionConfig.from_dict(vcfg) if hasattr(VisionConfig, "from_dict") else VisionConfig(**vcfg)
    tower = VisionModel(vc)
    idx = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
    files, weights = {}, {}
    for k, f in idx.items():
        for prefix in ("model.visual.", "visual.", "vision_tower.", "vision_model."):
            if k.startswith(prefix):
                if f not in files:
                    files[f] = mx.load(os.path.join(model_path, f))
                weights[k[len(prefix):]] = files[f][k]
                break
    if not weights:
        return False
    tower.load_weights(list(VisionModel.sanitize(weights).items()), strict=True)
    mx.eval(tower.parameters())
    dtype = tower.patch_embed.proj.weight.dtype
    image_token_id = int(cfg["image_token_id"])
    ip = Glm5NextImageProcessor(**_image_processor_kwargs(model_path))

    class Processor:
        image_token = "<|image|>"

        def __call__(self, text, images):
            pixel_values = []
            for image in images:
                patches, grid = ip._process_one(image)
                count = int(np.prod(grid) // ip.merge_size ** 2)
                if self.image_token not in text:
                    raise ValueError("More images were provided than image tokens.")
                text = text.replace(self.image_token, _PLACEHOLDER * count, 1)
                pixel_values.append((mx.array(patches), [int(g) for g in grid]))
            return text.replace(_PLACEHOLDER, self.image_token), pixel_values

    processor = Processor()
    S.load_processor = lambda model, config: processor

    cls = type(model)

    # Memory of the tower. Its attention is global over one image's patches: at the processor's maximum
    # (about 32k patches) the stock call materialises a 16 x 32k x 32k score tensor per layer (~32 GB), and
    # several images in one request were evaluated in one graph. Three large images reached the 230 GB
    # wired limit: Metal failed the command buffer on one rank only, the other rank went on into the trunk
    # and waited for a sum that never came (server hung while health checks passed). Here: queries go
    # through attention in blocks of TP_VIS_QCHUNK rows (same result), and each image is evaluated on its own.
    _QC = int(os.environ.get("TP_VIS_QCHUNK", "4096"))
    if _QC > 0:
        import mlx_vlm.models.glm5_next.vision as _V
        _Att = type(tower.blocks[0].attn)
        def _att_call(self, hidden_states, cu_seqlens, position_embeddings):
            n = hidden_states.shape[0]
            qkv = _V.linear_forward(self.qkv, hidden_states).reshape(n, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
            q, k, v = (t.squeeze(0) for t in mx.split(qkv, 3, axis=0))
            q = self.q_norm(q); k = self.k_norm(k)
            cos, sin = position_embeddings
            q, k = _V.apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.transpose(1, 0, 2)[None]; k = k.transpose(1, 0, 2)[None]; v = v.transpose(1, 0, 2)[None]
            bounds = [int(x) for x in cu_seqlens.tolist()]
            outs = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                ks, vs = k[:, :, a:b], v[:, :, a:b]
                for c in range(a, b, _QC):
                    outs.append(mx.fast.scaled_dot_product_attention(q[:, :, c:min(c + _QC, b)], ks, vs, scale=self.scale, mask=None))
            o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
            return _V.linear_forward(self.proj, o.transpose(0, 2, 1, 3).reshape(n, -1))
        _Att.__call__ = _att_call

    def embed_inputs(self, tokens, pixel_values=None):
        """Token embeddings [T, H] with every image token replaced by its vision feature."""
        emb = self.language_model.model.embed_tokens(tokens)
        if not pixel_values:
            return emb
        feats = []
        for p, g in pixel_values:
            f = tower(p.astype(dtype), mx.array([g])).astype(emb.dtype)
            mx.eval(f); feats.append(f)
        feats = mx.concatenate(feats, axis=0) if len(feats) > 1 else feats[0]
        mask = tokens == image_token_id
        n = int(mask.sum().item())
        # Image rows already in the prompt cache (an image split by the cached prefix) keep the
        # features that follow them aligned from the end.
        lead = n - feats.shape[0]
        if lead < 0:
            feats = feats[-n:] if n else feats[:0]
            lead = 0
        if n == 0:
            return emb
        order = mx.cumsum(mask.astype(mx.int32)) - 1 - lead
        use = mask & (order >= 0)
        return mx.where(use[:, None], feats[mx.maximum(order, 0)], emb)

    orig_call = cls.__call__

    def __call__(self, inputs, cache=None, skip_logits=False, input_embeddings=None, **kwargs):
        if input_embeddings is None:
            return orig_call(self, inputs, cache=cache, skip_logits=skip_logits, **kwargs)
        return self.language_model(inputs, inputs_embeds=input_embeddings, cache=cache).logits

    # --- image requests on the batched (speculative) path ---------------------------------------
    # The server serves image requests one at a time through its plain decode loop. Instead: run the
    # image part of the prompt here (with the spliced embeddings), store that prefix in the prompt
    # cache under the image-aware key, and let the request continue on the batched path as a cache
    # hit followed by ordinary text tokens.
    if os.environ.get("TP_VISION_FAST", "1") == "1":
        import sys
        from mlx_lm.models.cache import make_prompt_cache
        RG = next(c for c in vars(S).values() if isinstance(c, type) and "_tokenize" in vars(c) and "_generate" in vars(c))
        orig_tok = RG._tokenize
        orig_parts = S._image_parts

        def _image_parts(messages):
            if sys._getframe(1).f_code.co_name == "_generate":
                return iter(())                       # "has images" no longer means "not batchable"
            return orig_parts(messages)

        _beat = getattr(S, '_tp_beat', lambda: None)
        _FT = os.environ.get('TP_VIS_FAILTEST', '')
        _VBLK = int(os.environ.get('TP_VIS_BLOCK', os.environ.get('TP_PREFILL_STEP', '4096')))
        _VT = os.environ.get('TP_TRACE') in ('1', '2'); _MERGE = os.environ.get('TP_VISION_MERGE', '1') == '1'
        _PIN = os.environ.get('TP_IMG_PIN', '1') == '1'; _PIN_MIN = int(os.environ.get('TP_IMG_PIN_MIN', '16000'))
        _PIN_BACK = int(os.environ.get('TP_IMG_PIN_BACK', '2048')); _PINLOG = os.environ.get('TP_PINLOG') == '1'
        if not hasattr(S, '_tp_pins'): S._tp_pins = set()
        _ut = []
        def _user_tok(tokenizer):
            if not _ut:
                try:
                    t = tokenizer.convert_tokens_to_ids('<|user|>'); u = getattr(tokenizer, 'unk_token_id', None)
                    _ut.append(int(t) if isinstance(t, int) and t >= 0 and t != u else None)
                except Exception:
                    _ut.append(None)
            return _ut[0]
        # Long image prompts take turns with everything else (TP_VIS_ASYNC=1). The image part of a prompt is
        # read here, outside the batch scheduler; done in one go on the generation thread, a first read of
        # a 150k-token conversation stopped every other request for about a minute and an image swap for
        # 6 s. A read of more than TP_VIS_ASYNC_MIN tokens is therefore held back as a job: the request
        # loop asks for its next request between rounds of decode steps, and each time it does one block
        # of the job is read instead (TP_VIS_ASYNC_GAP full blocks while nothing decodes, one block of
        # TP_VIS_ASYNC_BLOCK tokens when streams are active). When the job is done its request goes to the loop as if it had just arrived,
        # and finds its prefix in the prompt cache. Every rank sees the same requests and the same batch
        # state, so every rank takes the same turns.
        _ASYNC = os.environ.get('TP_VIS_ASYNC', '1') == '1'; _ASYNC_MIN = int(os.environ.get('TP_VIS_ASYNC_MIN', '40000'))
        _ABLK = int(os.environ.get('TP_VIS_ASYNC_BLOCK', '2048')); _AGAP = int(os.environ.get('TP_VIS_ASYNC_GAP', '2')); _ALOG = os.environ.get('TP_VIS_ASYNC_LOG') == '1'
        _jobs = []; _memo = {}
        from types import SimpleNamespace as _NS

        def _prep(self, tokenizer, request, out):
            prompt, segments, types, state, vision = out
            # Image ids as int32-safe negatives: the batch generator keeps tokens in int32 buffers.
            key = [t if t >= 0 else -((-t) % 2147483000) - 1 for t in vision.cache_key]
            mkey = self.model_provider.model_key
            end = max(i for i, t in enumerate(prompt) if t == image_token_id) + 1
            # One forward for the image AND the text after it (all but the last token): a second small
            # forward for the text tail costs ~70 ms on its own (it sweeps most expert weights again).
            if _MERGE: end = max(end, len(prompt) - 1)
            cache, rest = self.prompt_cache.fetch_nearest_cache(mkey, key[:end])
            done = end - len(rest)
            # Checkpoint before the oldest image (TP_IMG_PIN=1). A client that keeps only its last few
            # images removes the oldest image message whenever it adds one; every saved end-of-prompt state
            # then contains a removed image, so the whole conversation was read again (55-70 s at 150k+
            # tokens). The state at the start of the oldest image's message stays valid after that removal:
            # save it while passing.
            pin = None
            if _PIN and rest:
                try:
                    s0 = min(int(i) for i in vision.starts); c = None; ut = _user_tok(tokenizer)
                    if ut is not None:
                        for j in range(s0 - 1, max(s0 - _PIN_BACK, 0) - 1, -1):
                            if prompt[j] == ut: c = j; break
                    if c is None: c = s0 - 256
                    if c >= _PIN_MIN and done < c < end: pin = c
                except Exception as e:
                    print(f'[tp-vis] pin skipped: {e!r}', flush=True)
            if rest and cache is None: cache = make_prompt_cache(self.model_provider.model)
            return _NS(request=request, out=out, key=key, mkey=mkey, end=end, cache=cache if rest else None, done=done, pin=pin,
                       finished=not rest, gen=None, blk=_VBLK, req=None, t0=__import__('time').perf_counter(), steps=0)

        def _blocks(self, J, _lap):
            """The read of J's prompt up to J.end, one trunk block per step (the first step also runs the tower)."""
            prompt, vision, cache, done, end, pin, key = J.out[0], J.out[4], J.cache, J.done, J.end, J.pin, J.key
            mdl = self.model_provider.model
            toks = mx.array(prompt[done:end])
            n = sum(i < done for i in vision.starts)
            # The tower is local to each rank, so one rank can fail (memory) while the other does
            # not. Agree on the outcome before the trunk: all ranks raise, or all go on.
            err = None
            try:
                if _FT and len(toks) > 3000:        # test hooks: "<rank>" fails there, "hang<rank>" stalls there
                    if _FT == str(rank): raise RuntimeError("TP_VIS_FAILTEST")
                    if _FT == f"hang{rank}": __import__('time').sleep(100000)
                _beat()
                emb = mdl.embed_inputs(toks, pixel_values=vision.pixel_values[n:])
                mx.eval(emb)
                _beat()
            except Exception as e:
                err = e
            world = mx.distributed.init()
            if world.size() > 1:
                bad = int(mx.distributed.all_sum(mx.array(0 if err is None else 1)).item())
            else:
                bad = 0 if err is None else 1
            if bad:
                emb = None; mx.clear_cache()
                raise RuntimeError(f"image could not be processed ({err if err is not None else 'failed on the other machine'})")
            _lap('tower+embed')
            trunk = mdl.language_model.model
            try:
                # Blocks of up to TP_PREFILL_STEP tokens: every forward of 130+ rows costs a fixed
                # ~200 ms (one pass over all expert weights), so 1,024-token blocks paid it per block.
                a = 0; total = end - done; cut = pin - done if pin is not None else -1
                while a < total:
                    b = min(a + J.blk, total)
                    if a < cut < b: b = cut
                    trunk(toks[None, a:b], cache=cache, inputs_embeds=emb[None, a:b])
                    mx.eval([c.state for c in cache])
                    _beat()
                    if b == cut:
                        cp = copy.deepcopy(cache); mx.eval([c.state for c in cp])
                        S._tp_pins.add((pin, tuple(key[pin - 8:pin])))
                        self.prompt_cache.insert_cache(J.mkey, key[:pin], cp, cache_type="user")
                        if _VT or _PINLOG: print(f'[tp-vis] pin saved at {pin} of {len(prompt)} (read from {done})', flush=True)
                    a = b
                    if a < total: yield
            except Exception as e:
                # Mid-trunk the ranks are no longer in step; nothing can be recovered. Exit so the
                # supervisor restarts both ranks instead of leaving a server that never answers.
                print(f"[tp-vis] fatal: image prefill failed mid-trunk: {e!r}; exiting", flush=True)
                os._exit(70)

        def _finish(self, J):
            if _PINLOG: print(f'[tp-vis] read {J.end - J.done} of {len(J.out[0])} tokens (cached {J.done}), pin {J.pin}', flush=True)
            self.prompt_cache.insert_cache(J.mkey, J.key[:J.end], J.cache, cache_type="user")
            J.cache = None; J.gen = None; J.finished = True

        def _tokenize(self, tokenizer, request, args):
            _t = [__import__('time').perf_counter()]
            def _lap(nm):
                if _VT: n = __import__('time').perf_counter(); print(f'[tp-vis] {nm} {1e3 * (n - _t[0]):.1f} ms', flush=True); _t[0] = n
            J = _memo.pop(id(request), None)
            if J is not None and J.request is not request: J = None
            prompt, segments, types, state, vision = J.out if J is not None else orig_tok(self, tokenizer, request, args)
            _lap('template+processor')
            frame = sys._getframe(1)
            if vision is None or frame.f_code.co_name != "_generate":
                return prompt, segments, types, state, vision
            if J is None: J = _prep(self, tokenizer, request, (prompt, segments, types, state, vision))
            if not J.finished:
                stream = frame.f_locals.get("generation_stream")
                def run():
                    for _ in _blocks(self, J, _lap): pass
                if stream is not None:
                    with mx.stream(stream): run()
                else:
                    run()
                _lap('trunk prefill %d tok' % (J.end - J.done))
                _finish(self, J)
            segs, o = [], 0
            for sg in segments:
                segs.append(J.key[o:o + len(sg)]); o += len(sg)
            return J.key, segs, types, state, vision

        def _wants_job(self, r):
            """True when request r was taken as a job (a long image-prompt read); it comes back from _step later."""
            try:
                rqueue, request, args = r
                if not any(orig_parts(request.messages)): return False
                tokenizer = self.model_provider.tokenizer
                out = orig_tok(self, tokenizer, request, args)
                if out[4] is None: return False
                J = _prep(self, tokenizer, request, out)
            except Exception:
                return False                              # the ordinary path reports whatever is wrong with it
            _memo.clear(); J.req = r
            if J.finished or J.end - J.done <= _ASYNC_MIN:
                _memo[id(request)] = J; return False      # short read: in one go, as before
            _jobs.append(J)
            if _ALOG: print(f'[tp-vis] job: {J.end - J.done} of {len(out[0])} tokens to read, {len(_jobs)} waiting', flush=True)
            return True

        def _step(self):
            """One block of the first job. Returns its request once the job is done, else None."""
            J = _jobs[0]
            f = sys._getframe(1)
            while f is not None and f.f_code.co_name != "_generate": f = f.f_back
            bg = f.f_locals.get("batch_generator") if f is not None else None
            gb = getattr(bg, "_generation_batch", None)
            ngen = len(gb) if gb is not None else 0
            if ngen and any(getattr(x, "tp_alone", False) for x in (gb.samplers or [])):
                S._tp_vjob_due = True; return None        # a decode-alone request is generating: no blocks in between
            S._tp_vjob_due = False
            # Nothing decoding (idle, or the gap a decode-alone request leaves before the next one starts):
            # TP_VIS_ASYNC_GAP full blocks. Streams decoding: one short block between their rounds.
            J.blk = _VBLK if ngen == 0 else _ABLK
            if J.gen is None: J.gen = _blocks(self, J, lambda nm: None)
            try:
                with mx.stream(mx.default_stream(mx.default_device())):
                    for _ in range(_AGAP if ngen == 0 else 1): next(J.gen); J.steps += 1
                return None
            except StopIteration:
                _finish(self, J); _jobs.pop(0); _memo.clear(); _memo[id(J.request)] = J
                if _ALOG: print(f'[tp-vis] job done: {J.end - J.done} tokens in {J.steps + 1} blocks, {__import__("time").perf_counter() - J.t0:.1f} s', flush=True)
                return J.req
            except RuntimeError as e:                     # the images could not be processed (every rank agreed)
                _jobs.pop(0); S._tp_vjob_due = False; J.req[0].put(e); return None

        if _ASYNC:
            _nr0 = RG._next_request
            def _next_request(self, timeout=None):
                if not _jobs:
                    r = _nr0(self, timeout)
                    return None if r is not None and _wants_job(self, r) else r
                r = _nr0(self, None)                      # a job is waiting: do not sit on an empty queue
                if r is not None:
                    return None if _wants_job(self, r) else r
                return _step(self)
            RG._next_request = _next_request

        S._image_parts = _image_parts
        RG._tokenize = _tokenize
    cls.embed_inputs = embed_inputs
    cls.__call__ = __call__
    nb = sum(v.nbytes for v in weights.values())
    print(f"[r{rank}] vision tower: {len(weights)} tensors, {nb / 2**30:.2f} GiB", flush=True)
    return True
