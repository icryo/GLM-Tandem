# Image input for the lean tensor-parallel server.
#
# The bundled mlx_lm.server already routes requests with images (data URLs, shared with every
# rank) through `processor(text, images)` and `model.embed_inputs(tokens, pixel_values=...)`.
# For GLM-5.3 the cluster model class is text-only, so this adds the three missing pieces on
# every rank: the vision tower (1.05 GiB), the image processor, and the embedding splice.
# The tower runs once per image at prompt time; decoding is unchanged.
import json, os
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
        _VT = os.environ.get('TP_TRACE') == '1'; _MERGE = os.environ.get('TP_VISION_MERGE', '1') == '1'
        def _tokenize(self, tokenizer, request, args):
            _t = [__import__('time').perf_counter()]
            def _lap(nm):
                if _VT: n = __import__('time').perf_counter(); print(f'[tp-vis] {nm} {1e3 * (n - _t[0]):.1f} ms', flush=True); _t[0] = n
            prompt, segments, types, state, vision = orig_tok(self, tokenizer, request, args)
            _lap('template+processor')
            frame = sys._getframe(1)
            if vision is None or frame.f_code.co_name != "_generate":
                return prompt, segments, types, state, vision
            # Image ids as int32-safe negatives: the batch generator keeps tokens in int32 buffers.
            key = [t if t >= 0 else -((-t) % 2147483000) - 1 for t in vision.cache_key]
            mkey = self.model_provider.model_key
            mdl = self.model_provider.model
            end = max(i for i, t in enumerate(prompt) if t == image_token_id) + 1
            # One forward for the image AND the text after it (all but the last token): a second small
            # forward for the text tail costs ~70 ms on its own (it sweeps most expert weights again).
            if _MERGE: end = max(end, len(prompt) - 1)
            cache, rest = self.prompt_cache.fetch_nearest_cache(mkey, key[:end])
            done = end - len(rest)
            if rest:
                if cache is None:
                    cache = make_prompt_cache(mdl)
                stream = frame.f_locals.get("generation_stream")
                def prefill():
                    toks = mx.array(prompt[done:end])
                    n = sum(i < done for i in vision.starts)
                    # The tower is local to each rank, so one rank can fail (memory) while the other does
                    # not. Agree on the outcome before the trunk: all ranks raise, or all go on.
                    err = None
                    try:
                        if _FT and len(toks) > 3000:        # test hooks: "<rank>" fails there, "hang<rank>" stalls there
                            if _FT == str(rank): raise RuntimeError("TP_VIS_FAILTEST")
                            if _FT == f"hang{rank}": __import__('time').sleep(100000)
                        emb = mdl.embed_inputs(toks, pixel_values=vision.pixel_values[n:])
                        mx.eval(emb)
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
                        # One block of up to TP_PREFILL_STEP tokens: every forward of 130+ rows costs a fixed
                        # ~200 ms (one pass over all expert weights), so 1,024-token blocks paid it per block.
                        for a in range(0, end - done, _VBLK):
                            b = min(a + _VBLK, end - done)
                            trunk(toks[None, a:b], cache=cache, inputs_embeds=emb[None, a:b])
                            mx.eval([c.state for c in cache])
                            _beat()
                    except Exception as e:
                        # Mid-trunk the ranks are no longer in step; nothing can be recovered. Exit so the
                        # supervisor restarts both ranks instead of leaving a server that never answers.
                        print(f"[tp-vis] fatal: image prefill failed mid-trunk: {e!r}; exiting", flush=True)
                        os._exit(70)
                if stream is not None:
                    with mx.stream(stream):
                        prefill()
                else:
                    prefill()
                _lap('trunk prefill %d tok' % (end - done))
                self.prompt_cache.insert_cache(mkey, key[:end], cache, cache_type="user")
            segs, o = [], 0
            for sg in segments:
                segs.append(key[o:o + len(sg)]); o += len(sg)
            return key, segs, types, state, vision

        S._image_parts = _image_parts
        RG._tokenize = _tokenize
    cls.embed_inputs = embed_inputs
    cls.__call__ = __call__
    nb = sum(v.nbytes for v in weights.values())
    print(f"[r{rank}] vision tower: {len(weights)} tensors, {nb / 2**30:.2f} GiB", flush=True)
    return True
