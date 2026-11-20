# Image requests of growing size: prompt tokens, time, errors. SIZES="644x644,2000x1500,..."
import base64, io, json, os, random, sys, time, urllib.request, subprocess
from PIL import Image, ImageDraw
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
SIZES = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("SIZES", "644x644,1288x1288,2000x1500,3000x2000,4000x3000,6000x4000").split(",")]
NIMG = int(os.environ.get("NIMG", "1")); FMT = os.environ.get("FMT", "JPEG")
rnd = random.Random(1)
def image(w, h):
    im = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(im)
    for i in range(0, h - 20, 40): d.text((20, 10 + i), " ".join(str(rnd.randint(0, 99999)) for _ in range(max(2, w // 60))), fill=(0, 0, 0))
    b = io.BytesIO(); im.save(b, FMT, quality=85); return f"data:image/{FMT.lower()};base64," + base64.b64encode(b.getvalue()).decode()
def wired(): return int(subprocess.run("vm_stat | awk '/wired down/ {print $4}'", shell=True, capture_output=True, text=True).stdout.strip().rstrip(".")) * 16384 / 2**30
for w, h in SIZES:
    content = [{"type": "image_url", "image_url": {"url": image(w, h)}} for _ in range(NIMG)] + [{"type": "text", "text": "What is the first number in the image? Answer with the number only."}]
    body = dict(model=MODEL, max_tokens=24, temperature=0, messages=[{"role": "user", "content": content}])
    raw = json.dumps(body).encode(); t0 = time.time(); w0 = wired()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, raw, {"Content-Type": "application/json"}), timeout=float(os.environ.get("TMO", "240"))) as r: d = json.loads(r.read())
        u = d["usage"]; msg = d["choices"][0]["message"]; out = (msg.get("content") or "").strip()[:30]
        print(f"{w}x{h} x{NIMG} ({len(raw) // 1024} KB): prompt {u['prompt_tokens']} tok, {time.time() - t0:.2f} s, wired {w0:.0f} -> {wired():.0f} GiB, answer {out!r}", flush=True)
    except Exception as e:
        body = e.read().decode()[:200] if hasattr(e, "read") else ""
        print(f"{w}x{h} x{NIMG} ({len(raw) // 1024} KB): FAILED after {time.time() - t0:.1f} s: {e} {body}", flush=True); break
