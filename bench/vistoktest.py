# CPU only, synthetic content. The memoized image path (tpvision_x.make_processor + the encode memo in
# tpserve_x) must give exactly what the stock tokenize step gives: prompt tokens, image starts, cache key,
# pixel values. Simulates agent conversations that keep their last KEEP images, with edits, two
# conversations interleaved, images of several sizes and formats, repeated images, a small memo.
import os, sys, time, io, base64, json, random, copy
os.environ.setdefault("TP_TOKMEMO_CHECK", "0")
import mlx.core as mx
mx.set_default_device(mx.cpu)
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
MP = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/GLM-5.3-Flash-oQ4e-MTP"))
from omlx.utils.model_loading import maybe_apply_pre_load_patches
maybe_apply_pre_load_patches(MP)
from mlx_lm.utils import load_tokenizer
import mlx_lm.server as S
from mlx_vlm.models.glm5_next.processing import Glm5NextImageProcessor, _image_processor_kwargs
TV = __import__(os.environ.get("VISION_MOD", "tpvision_x"))       # run from the directory that holds it (SERVER_SRC, VISION_MOD)
SRC = os.environ.get("SERVER_SRC", "tpserve_x.py")
src = open(SRC).read(); a = src.index("if os.environ.get(\"TP_TOKMEMO\", \"1\") == \"1\":"); b = src.index("# ---- Never hang")
exec(src[a:b])                                             # installs the memo on TokenizerWrapper (apply_chat_template, encode)
assert "encode" in vars(_TWc), "encode memo not installed"
tok = load_tokenizer(Path(MP)); hf = tok._tokenizer
ip = Glm5NextImageProcessor(**_image_processor_kwargs(MP))
decode0 = S._decode_image; new_encode = vars(_TWc)["encode"]
class StockProcessor:                                      # as in the live tpvision.py
    image_token = "<|image|>"
    def __call__(self, text, images):
        pixel_values = []
        for image in images:
            patches, grid = ip._process_one(image)
            count = int(np.prod(grid) // ip.merge_size ** 2)
            if self.image_token not in text: raise ValueError("More images were provided than image tokens.")
            text = text.replace(self.image_token, TV._PLACEHOLDER * count, 1)
            pixel_values.append((mx.array(patches), [int(g) for g in grid]))
        return text.replace(TV._PLACEHOLDER, self.image_token), pixel_values
stock_proc = StockProcessor()
new_proc, memo = TV.make_processor(ip, decode0, memo_n=int(os.environ.get("MEMO_N", "64")))
cta = json.loads(os.environ.get("TP_CHAT_TEMPLATE_ARGS", '{"reasoning_effort": "low", "clear_thinking": false}'))
def run(proc, msgs, tools, new):
    if new: _TWc.encode = new_encode; S._decode_image = TV._LazyImage
    else:
        if "encode" in vars(_TWc): del _TWc.encode
        S._decode_image = decode0
    me = NS(model_provider=NS(processor=proc, cli_args=NS(chat_template_args=cta)))
    req = NS(request_type="chat", messages=copy.deepcopy(msgs), tools=tools, role_mapping=None)
    t = time.perf_counter()
    out = S.ResponseGenerator._tokenize(me, tok, req, NS(chat_template_kwargs=None))
    return out, 1e3 * (time.perf_counter() - t)
# Embedding splice with a stand-in tower (the real one needs the GPU): features are a fixed function of the
# pixel values, so a remembered feature must equal a recomputed one, and the splice must not depend on the memo.
HID = 8; calls = [0]
def fake_tower(p, g):
    calls[0] += 1
    gg = [int(v) for v in np.array(g).reshape(-1)]
    count = int(np.prod(gg) // ip.merge_size ** 2)
    v = p.astype(mx.float32).reshape(count, -1)
    return v.mean(axis=1, keepdims=True) + v[:, :1] * mx.arange(HID)[None].astype(mx.float32) * 0.01
table = mx.random.normal((hf.vocab_size + 4096, HID)) if hasattr(hf, "vocab_size") else None
img_id = tok.convert_tokens_to_ids("<|image|>")
fake_model = NS(language_model=NS(model=NS(embed_tokens=lambda t: table[t])))
embed_memo, fmemo = TV.make_embed(fake_tower, mx.float32, img_id, cap=int(os.environ.get("FEAT_CAP", str(64 << 20))))
embed_plain, _ = TV.make_embed(fake_tower, mx.float32, img_id, cap=0)
emb_bad = 0; emb_n = 0; tower_plain = tower_memo = 0
rnd = random.Random(int(os.environ.get("SEED", "1")))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
EXTRA = ["<|user|>", "<think>", "</think>", "naïve café 日本語 🙂", "\n\n", "<tool_call>", "{\"a\": [1, 2]}", "<|observation|>"]
def words(n): return " ".join(rnd.choice(WORDS) if rnd.random() > 0.01 else rnd.choice(EXTRA) for _ in range(n))
def image():
    w, h = rnd.choice([(1280, 880), (640, 480), (1920, 1080), (300, 900), (56, 56), (28, 28), (1024, 1024), (811, 613)])
    mode, fmt = rnd.choice([("RGB", "JPEG"), ("RGB", "PNG"), ("RGBA", "PNG"), ("L", "PNG"), ("RGB", "WEBP")])
    ch = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    a = np.random.default_rng(rnd.randrange(1 << 30)).integers(0, 255, (h // 8 + 1, w // 8 + 1, ch), dtype=np.uint8).repeat(8, 0).repeat(8, 1)[:h, :w]
    b = io.BytesIO(); Image.fromarray(a[..., 0] if ch == 1 else a, mode).save(b, fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b.getvalue()).decode()
tools = [{"type": "function", "function": {"name": f"tool_{i}", "description": words(20), "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}} for i in range(6)]
KEEP = int(os.environ.get("KEEP", "5")); STEPS = int(os.environ.get("STEPS", "50")); GROW = int(os.environ.get("GROW", "700"))
convs = [[{"role": "system", "content": "Sys " + words(rnd.choice([200, 3000]))}] for _ in range(2)]
pool = []
def img_msg(u): return {"role": "user", "content": [{"type": "text", "text": words(12)}, {"type": "image_url", "image_url": {"url": u}}]}
bad = 0; n = 0; t_old = t_new = 0.0; hits = 0; imgs_total = 0; maxtok = 0; t_new_warm = t_old_warm = 0.0; nwarm = 0
for step in range(STEPS):
    msgs = rnd.choice(convs)
    op = rnd.random()
    if op < 0.08 and len(msgs) > 6:                          # client edits history
        k = rnd.randrange(1, len(msgs))
        if isinstance(msgs[k].get("content"), str): msgs[k] = dict(msgs[k], content=words(rnd.randrange(1, 200)))
    newimg = rnd.random() < 0.5
    if newimg:
        u = rnd.choice(pool) if pool and rnd.random() < 0.15 else image(); pool.append(u)   # sometimes an image seen before
        msgs.append(img_msg(u))
        ii = [i for i, m in enumerate(msgs) if isinstance(m.get("content"), list)]
        while len(ii) > KEEP: del msgs[ii.pop(0)]; ii = [i for i, m in enumerate(msgs) if isinstance(m.get("content"), list)]
    else:
        msgs.append({"role": "user", "content": words(rnd.randrange(5, 80))})
    if not any(isinstance(m.get("content"), list) for m in msgs): msgs.append(img_msg(image()))
    (o, ms_o) = run(stock_proc, msgs, tools, False)
    before = set(memo)
    (w, ms_w) = run(new_proc, msgs, tools, True)
    t_old += ms_o; t_new += ms_w; n += 1
    if not newimg: t_old_warm += ms_o; t_new_warm += ms_w; nwarm += 1
    ok = list(o[0]) == list(w[0]) and [list(x) for x in o[1]] == [list(x) for x in w[1]] and o[2] == w[2] and o[3] == w[3]
    vo, vw = o[4], w[4]
    ok = ok and (vo is None) == (vw is None)
    if vo is not None and vw is not None:
        ok = ok and vo.starts == vw.starts and vo.cache_key == vw.cache_key and len(vo.pixel_values) == len(vw.pixel_values)
        if ok:
            imgs_total += len(vw.pixel_values); hits += sum(1 for p in vw.pixel_values if p.val is None)
            got = TV._pixels(vw.pixel_values)
            for (pa, ga), (pb, gb) in zip(vo.pixel_values, got):
                if ga != gb or pa.shape != pb.shape or pa.dtype != pb.dtype or not np.array_equal(np.array(pa), np.array(pb)): ok = False
    if ok and vo is not None and vo.starts:
        # as the server does it: read from a cached prefix that ends before one of the images, or from the start
        prompt = list(o[0]); cuts = [0] + [st_ for st_ in vo.starts] + [len(prompt) - 1]
        for done in {0, rnd.choice(cuts), vo.starts[-1]}:
            end = len(prompt) - 1
            if done >= end: continue
            nb = sum(i < done for i in vo.starts)
            toks = mx.array(prompt[done:end])
            c0 = calls[0]; e_old = embed_plain(fake_model, toks, pixel_values=vo.pixel_values[nb:]); tower_plain += calls[0] - c0
            c0 = calls[0]; e_new = embed_memo(fake_model, toks, pixel_values=vw.pixel_values[nb:]); tower_memo += calls[0] - c0
            emb_n += 1
            if e_old.shape != e_new.shape or not np.array_equal(np.array(e_old), np.array(e_new)): emb_bad += 1; ok = False
    bad += not ok; maxtok = max(maxtok, len(o[0]))
    if not ok: print(f"step {step}: MISMATCH (prompt {len(o[0])} vs {len(w[0])})", flush=True)
    # the model answers: a tool call and its result, like an agent turn
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{step}", "type": "function", "function": {"name": "tool_1", "arguments": {"path": words(2)}}}]})
    msgs.append({"role": "tool", "tool_call_id": f"c{step}", "content": words(rnd.randrange(20, GROW))})
print(f"{n} turns, mismatches {bad}, longest prompt {maxtok} tokens, images per prompt avg {imgs_total / max(n, 1):.1f}, remembered {hits} of {imgs_total} ({len(memo)} in memo)")
print(f"tokenize step per turn: stock {t_old / n:.1f} ms, memoized {t_new / n:.1f} ms; turns without a new image ({nwarm}): stock {t_old_warm / max(nwarm, 1):.1f} ms, memoized {t_new_warm / max(nwarm, 1):.1f} ms")
print(f"embedding splice: {emb_n} reads compared, mismatches {emb_bad}; tower calls without memo {tower_plain}, with memo {tower_memo} ({len(fmemo)} images remembered)")
print("PASS" if bad == 0 else "FAIL")
