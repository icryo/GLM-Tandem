# Replay shaped like the real traffic: image workers (one ~875-token image in, ~120 tokens out) next to text workers
# (~2.7k fresh prompt tokens in, ~350 out), all non-streaming, back to back for DUR seconds. Optional agent turns.
import base64, io, json, os, random, time, threading, urllib.request
from PIL import Image, ImageDraw
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WI = int(os.environ.get("WI", "1")); WT = int(os.environ.get("WT", "1")); DUR = float(os.environ.get("DUR", "60")); PX = int(os.environ.get("PX", "812"))
SEED = os.environ.get("SEED", "1") == "1"      # the real jobs send a seed with every request; that changes the server's path
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
rnd = random.Random(time.time()); lock = threading.Lock(); res = {"image": [], "text": []}
def image():
    im = Image.new("RGB", (PX, PX), "white"); d = ImageDraw.Draw(im)
    for i in range(16): d.text((20, 20 + i * 46), " ".join(rnd.choice(WORDS) for _ in range(7)), fill=(0, 0, 0))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def req(kind):
    if kind == "image": msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image()}}, {"type": "text", "text": "Describe this image in three sentences."}]}]; mt = 120
    else: msgs = [{"role": "user", "content": "Notes: " + " ".join(rnd.choice(WORDS) for _ in range(1150)) + "\nIgnore the notes and write four paragraphs about rivers."}]; mt = 350
    body = dict(model=MODEL, max_tokens=mt, temperature=0.3, messages=msgs); t = time.time()
    if SEED: body["seed"] = rnd.randrange(1, 2**31)
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r: d = json.loads(r.read())
    return time.time() - t, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]
end = time.time() + DUR
def worker(kind):
    while time.time() < end:
        r = req(kind)
        with lock: res[kind].append(r)
req("image"); req("text"); res = {"image": [], "text": []}; t0 = time.time(); end = t0 + DUR
ths = [threading.Thread(target=worker, args=("image",)) for _ in range(WI)] + [threading.Thread(target=worker, args=("text",)) for _ in range(WT)]
[t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
out = []
for k in ("image", "text"):
    v = res[k]
    if v: out.append(f"{k}: {len(v) / wall:.2f} req/s, median {sorted(x[0] for x in v)[len(v) // 2]:.2f} s, p90 {sorted(x[0] for x in v)[int(.9 * (len(v) - 1))]:.2f} s ({v[0][2]} in, {sum(x[1] for x in v) // len(v)} out)")
tok = sum(x[1] for v in res.values() for x in v)
print(f"WI={WI} WT={WT}: " + " | ".join(out) + f" | TOTAL {tok / wall:.0f} output tok/s, {sum(len(v) for v in res.values()) / wall:.2f} req/s")
