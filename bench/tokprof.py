# CPU only, synthetic content: where the per-turn tokenize step of an image-bearing conversation spends its time.
import os, sys, time, io, base64, json, random, copy, statistics as st
import mlx.core as mx
mx.set_default_device(mx.cpu)
from pathlib import Path
import numpy as np
from PIL import Image
from mlx_lm.utils import load_tokenizer
import mlx_lm.server as S
MP = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/GLM-5.3-Flash-oQ4e-MTP"))
from omlx.utils.model_loading import maybe_apply_pre_load_patches
maybe_apply_pre_load_patches(MP)
from mlx_vlm.models.glm5_next.processing import Glm5NextImageProcessor, _image_processor_kwargs
tok = load_tokenizer(Path(MP)); hf = tok._tokenizer
ip = Glm5NextImageProcessor(**_image_processor_kwargs(MP))
_PLACEHOLDER = "<|tp_image_placeholder|>"
class Processor:
    image_token = "<|image|>"
    def __call__(self, text, images):
        pixel_values = []
        for image in images:
            patches, grid = ip._process_one(image)
            count = int(np.prod(grid) // ip.merge_size ** 2)
            text = text.replace(self.image_token, _PLACEHOLDER * count, 1)
            pixel_values.append((mx.array(patches), [int(g) for g in grid]))
        return text.replace(_PLACEHOLDER, self.image_token), pixel_values
processor = Processor()
rnd = random.Random(5)
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
def words(n): return " ".join(rnd.choice(WORDS) for _ in range(n))
def image(w, h, fmt):
    a = np.linspace(0, 255, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), dtype=np.float32)
    a = (a * 0.6 + np.random.default_rng(rnd.randrange(1 << 30)).integers(0, 100, (h, w, 3))).clip(0, 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(a).save(b, fmt, **({"quality": 85} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode(), len(b.getvalue())
W, H, FMT = int(os.environ.get("IMG_W", "1280")), int(os.environ.get("IMG_H", "880")), os.environ.get("IMG_FMT", "JPEG")
TURNS = int(os.environ.get("TURNS", "230")); NIMG = int(os.environ.get("NIMG", "5"))
tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": words(30), "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path"]}}} for i in range(12)]
msgs = [{"role": "system", "content": words(3000)}]
for t in range(TURNS):
    msgs.append({"role": "user", "content": words(rnd.randrange(20, 400))})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{t}", "type": "function", "function": {"name": "tool_1", "arguments": json.dumps({"path": words(2), "n": t})}}]})
    msgs.append({"role": "tool", "tool_call_id": f"c{t}", "content": words(rnd.randrange(50, 900))})
sizes = []
for k in range(NIMG):
    u, nb = image(W, H, FMT); sizes.append(nb)
    msgs.insert(len(msgs) - 3 * (NIMG - k) * 2, {"role": "user", "content": [{"type": "text", "text": words(20)}, {"type": "image_url", "image_url": {"url": u}}]})
msgs.append({"role": "user", "content": words(60)})
cta = json.loads(os.environ.get("TP_CHAT_TEMPLATE_ARGS", '{"reasoning_effort": "low", "clear_thinking": false}'))
body = json.dumps({"messages": msgs, "tools": tools})
print(f"messages {len(msgs)}, images {NIMG} x {W}x{H} {FMT} ({sum(sizes) // NIMG // 1024} KiB each), request body {len(body) / 1e6:.2f} MB", flush=True)
T = {}
def lap(name, f):
    t = time.perf_counter(); r = f(); T.setdefault(name, []).append(1e3 * (time.perf_counter() - t)); return r
token_id = tok.convert_tokens_to_ids("<|image|>")
for rep in range(int(os.environ.get("REPS", "5"))):
    m = lap("json.loads of the body", lambda: json.loads(body))["messages"]
    urls = lap("process_message_content", lambda: S.process_message_content(m, keep_images=True))
    text = lap("chat template render", lambda: tok.apply_chat_template(m, add_generation_prompt=True, tools=tools, tokenize=False, **cta))
    imgs = lap("decode images", lambda: [S._decode_image(u) for u in urls])
    text2, pv = lap("image preprocessing + placeholder expansion", lambda: processor(text, images=imgs))
    prompt = lap("tokenizer.encode (full)", lambda: tok.encode(text2, add_special_tokens=False))
    vis = lap("_make_vision (ids, starts, cache key)", lambda: S._make_vision(prompt, urls=urls, pixel_values=pv, token_id=token_id))
    key = lap("int32-safe key + last image position", lambda: ([t if t >= 0 else -((-t) % 2147483000) - 1 for t in vis.cache_key], max(i for i, t in enumerate(prompt) if t == token_id) + 1))
    lap("think-state scan", lambda: (tok.rfind_think_start(prompt), tok.rfind_think_end(prompt)) if tok.has_thinking else None)
print(f"prompt {len(prompt)} tokens, text {len(text2) / 1e6:.2f} M chars, image tokens {sum(1 for t in prompt if t == token_id)}, patches per image {[int(p[0].shape[0]) for p in pv]}")
tot = 0
for k, v in T.items():
    print(f"  {k:46s} {st.median(v):7.1f} ms  (min {min(v):.1f})"); tot += st.median(v)
print(f"  {'sum':46s} {tot:7.1f} ms")
