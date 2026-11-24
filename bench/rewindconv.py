# A chat client that appends two short side messages after each answer and takes them out again on the
# next real turn (shape seen in real traffic). Each "rewind" turn diverges a few hundred tokens before the
# newest saved state. Prints time to first token per turn. Synthetic content only.
#   URL=http://localhost:8080/v1/chat/completions TARGET=24000 CYCLES=4 python3 rewindconv.py
import json, os, random, sys, time, urllib.request
sys.argv = sys.argv[:1]
os.environ.setdefault("TURNS", "0")
URL = os.environ.get("URL", "http://localhost:8080/v1/chat/completions"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
TARGET = int(os.environ.get("TARGET", "24000")); CYCLES = int(os.environ.get("CYCLES", "4")); IMAGES = os.environ.get("IMAGES", "1") == "1"
R = random.Random(int(os.environ.get("RSEED", "11")))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
def words(n): return " ".join(R.choice(WORDS) for _ in range(n))
def image():
    import base64, io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (448, 448), (R.randrange(256), R.randrange(256), R.randrange(256))); d = ImageDraw.Draw(im)
    for _ in range(10):
        x = sorted(R.randrange(448) for _ in range(2)); y = sorted(R.randrange(448) for _ in range(2))
        d.rectangle([x[0], y[0], x[1], y[1]], fill=(R.randrange(256), R.randrange(256), R.randrange(256)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def call(msgs, max_tokens=60):
    body = dict(model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=0, stream=True, stream_options={"include_usage": True})
    t0 = time.time(); first = None; usage = {}; text = ""
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if first is None and (dl.get("content") or dl.get("reasoning") or dl.get("tool_calls")): first = time.time()
                text += dl.get("content") or ""
    return (first or time.time()) - t0, usage.get("prompt_tokens", 0), text
CODE = R.randrange(100000, 999999)
msgs = [dict(role="system", content=words(3000) + f" The record code is {CODE}."), dict(role="user", content=words(40))]
n = 0
while n * 1.9 < TARGET - 6000:
    k = R.randint(150, 500); n += k + 40
    msgs += [dict(role="assistant", content=words(k)), dict(role="user", content=words(40))]
    if IMAGES and len(msgs) in (14, 28): msgs[-1] = dict(role="user", content=[dict(type="text", text=words(20)), dict(type="image_url", image_url=dict(url=image()))])
Q = " Reply with only the record code."
msgs[-1] = dict(role="user", content=words(20) + Q)
print("turn                      msgs  prompt_tok  first_token_s  code", flush=True)
def turn(kind):
    t, p, text = call(msgs); ok = str(CODE) in text
    print(f"{kind:24s} {len(msgs):5d} {p:10d} {t:13.2f}   {'ok' if ok else 'WRONG'}", flush=True)
    return t, text, ok
bad = 0; rew = []
t, text, ok = turn("first read"); bad += not ok
for c in range(CYCLES):
    msgs += [dict(role="assistant", content=text or "ok"), dict(role="user", content=words(60)), dict(role="user", content=words(65) + Q)]
    t, side, ok = turn("side messages added"); bad += not ok
    del msgs[-2:]; msgs.append(dict(role="user", content=words(25) + Q))
    t, text, ok = turn("side messages removed"); bad += not ok; rew.append(t)
    msgs += [dict(role="assistant", content=text or "ok"), dict(role="user", content=words(30) + Q)]
    t, text, ok = turn("plain next turn"); bad += not ok
print(f"DONE rewind turns first-token median {sorted(rew)[len(rew) // 2]:.2f} s max {max(rew):.2f} s; wrong answers {bad}")
