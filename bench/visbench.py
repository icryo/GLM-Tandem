import base64, io, json, os, time, urllib.request
from PIL import Image, ImageDraw
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
im = Image.new("RGB", (896, 896), "white"); d = ImageDraw.Draw(im)
for i in range(12): d.rectangle((40 + i * 60, 800 - i * 55, 80 + i * 60, 860), fill=(20 * i, 80, 255 - 20 * i))
b = io.BytesIO(); im.save(b, "PNG"); url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
for rep in range(3):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=400, temperature=0.6,
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": f"Describe this chart in detail and speculate what it could represent (take {rep + 3} angles)."}]}])
    t0 = time.time(); first = None; n = 0; usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            dd = json.loads(line[6:])
            if dd.get("usage"): usage = dd["usage"]
            if dd.get("choices") and first is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in dd["choices"]): first = time.time()
    end = time.time(); ct = usage.get("completion_tokens", 0)
    print(f"image request {rep}: prompt {usage.get('prompt_tokens')} tokens, first token {first - t0:.2f}s, {ct} tokens at {(ct - 1) / (end - first):.1f} tok/s")
