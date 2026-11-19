# The dominant real workload: small requests (one fresh image or ~600 fresh text tokens in, ~150 tokens out),
# non-streaming, 1 or 2 at a time. Reports time to first token, decode rate, and requests' wall time.
import base64, io, json, os, random, time, threading, urllib.request
from PIL import Image, ImageDraw
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
N = int(os.environ.get("N", "8")); W = int(os.environ.get("W", "1")); OUT = int(os.environ.get("OUT", "150")); KIND = os.environ.get("KIND", "image"); PX = int(os.environ.get("PX", "644"))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
rnd = random.Random(time.time())
def image():
    im = Image.new("RGB", (PX, PX), "white"); d = ImageDraw.Draw(im)
    for i in range(14): d.text((20, 20 + i * 42), " ".join(rnd.choice(WORDS) for _ in range(6)), fill=(0, 0, 0))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def msgs():
    if KIND == "image": return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image()}}, {"type": "text", "text": "Transcribe the text in this image, then describe the layout in a paragraph."}]}]
    return [{"role": "user", "content": "Notes: " + " ".join(rnd.choice(WORDS) for _ in range(260)) + "\nIgnore the notes and write a paragraph about rivers."}]
res = []
def one():
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=OUT, temperature=0.2, messages=msgs())
    t0 = time.time(); first = None; usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if first is None and d.get("choices") and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d["choices"]): first = time.time()
    end = time.time(); res.append((usage.get("prompt_tokens", 0), first - t0, (usage.get("completion_tokens", 0) - 1) / max(end - first, 1e-6), end - t0, usage.get("completion_tokens", 0)))
def worker():
    for _ in range(N): one()
one(); res.clear()
t = time.time(); ths = [threading.Thread(target=worker) for _ in range(W)]; [x.start() for x in ths]; [x.join() for x in ths]; wall = time.time() - t
med = lambda xs: sorted(xs)[len(xs) // 2]
print(f"{KIND} x{W}: prompt {med([r[0] for r in res])} tok | first token {med([r[1] for r in res]):.2f} s | decode {med([r[2] for r in res]):.0f} tok/s | per request {med([r[3] for r in res]):.2f} s for {med([r[4] for r in res])} tok | {len(res) / wall:.2f} requests/s, {sum(r[4] for r in res) / wall:.0f} tok/s overall")
