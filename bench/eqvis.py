# T=0 outputs for a fixed sequence of image requests: the same image behind different text, several images,
# and a long conversation that keeps its last KEEP images (the oldest one is replaced by a text stub on every
# turn, as agent clients do). Prints one sha256 per request, to compare two builds from fresh starts.
import base64, hashlib, io, json, os, random, time, urllib.request
from PIL import Image, ImageDraw, ImageFont
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
MT = int(os.environ.get("MT", "100")); CONV_TOK = int(os.environ.get("CONV_TOK", "100000")); KEEP = int(os.environ.get("KEEP", "5"))
TURNS = int(os.environ.get("TURNS", "6"))
def font(sz):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()
def img(code, size=(448, 448), kind="PNG"):
    im = Image.new("RGB", size, "white"); d = ImageDraw.Draw(im)
    d.rectangle((8, 8, size[0] - 8, size[1] - 8), outline=(30, 60, 220), width=6)
    d.text((size[0] // 10, size[1] // 3), code, fill="black", font=font(size[1] // 5))
    b = io.BytesIO(); im.save(b, kind, **({"quality": 90} if kind == "JPEG" else {}))
    return f"data:image/{kind.lower()};base64," + base64.b64encode(b.getvalue()).decode()
def T(s): return {"type": "text", "text": s}
def I(u): return {"type": "image_url", "image_url": {"url": u}}
def ask(messages):
    body = dict(model=MODEL, messages=messages, max_tokens=MT, temperature=0)
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]; out = (m.get("reasoning") or "") + "|" + (m.get("content") or "")
    return out, d["usage"], time.time() - t
def run(name, messages, want=None):
    out, u, dt = ask(messages)
    ok = "" if want is None else (" recall " + ("ok" if all(w in out for w in want) else "MISSING"))
    print(f"{hashlib.sha256(out.encode()).hexdigest()[:16]} {name}: prompt={u['prompt_tokens']} out={u['completion_tokens']} {dt:.2f}s{ok}", flush=True)

SIZES = [((448, 448), "PNG"), ((896, 672), "PNG"), ((1280, 880), "JPEG"), ((672, 448), "PNG")]
codes = [f"{c}{n}" for c, n in zip("KRTWXZBDFGHJ", (4172, 9035, 6618, 2759, 8341, 1906, 5527, 3084, 7493, 6250, 1378, 9461))]
ims = [img(c, *SIZES[i % len(SIZES)]) for i, c in enumerate(codes)]
A, B, C = ims[0], ims[1], ims[2]
Q = "What code is written in the image? Answer with the code only."
run("text", [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}])
run("text then image", [{"role": "user", "content": [T(Q), I(A)]}], [codes[0]])
run("same image, text after", [{"role": "user", "content": [I(A), T("Read the code in this picture and reply with it.")]}], [codes[0]])
run("two images, one seen", [{"role": "user", "content": [T("Two pictures follow. List both codes."), I(B), I(A)]}], codes[:2])
run("jpeg", [{"role": "user", "content": [I(C), T(Q)]}], [codes[2]])
run("three seen images, new order", [{"role": "user", "content": [T("List the three codes in order."), I(C), I(A), I(B)]}], codes[:3])
run("repeat of request 2", [{"role": "user", "content": [T(Q), I(A)]}], [codes[0]])

# Conversation: long text history, image messages near the end, last KEEP images kept as images.
rnd = random.Random(11)
WORDS = ("signal buffer window frame render layout parse commit branch merge kernel thread packet router socket cache index "
         "vector matrix tensor column record schema query cursor handle stream filter sample output review status").split()
def filler(n_tok):
    return " ".join(rnd.choice(WORDS) for _ in range(int(n_tok * 0.95)))
hist = [{"role": "system", "content": "You are a careful assistant. Codes shown in pictures must be reported exactly."}]
per = 1500; n = max(2, CONV_TOK // (2 * per))
for i in range(n):
    hist.append({"role": "user", "content": f"Note {i}: " + filler(per)})
    hist.append({"role": "assistant", "content": f"Recorded note {i}. " + filler(per // 6)})
shots = []                                   # (index in hist, code)
def add_shot(k):
    hist.append({"role": "user", "content": [T(f"Screenshot {k}. " + filler(60)), I(ims[3 + k])]})
    shots.append((len(hist) - 1, codes[3 + k]))
    for pos, code in shots[:-KEEP]:          # older screenshots become a text stub
        m = hist[pos]
        if isinstance(m["content"], list):
            m["content"] = [m["content"][0], T("[screenshot removed]")]
for k in range(KEEP):
    add_shot(k); hist.append({"role": "assistant", "content": f"Seen screenshot {k}. " + filler(40)})
for t in range(TURNS):
    if t: add_shot(KEEP + t - 1) if 3 + KEEP + t - 1 < len(ims) else None
    live = [c for _, c in shots[-KEEP:]]
    msgs = hist + [{"role": "user", "content": "List every code visible in the screenshots that are still attached, oldest first."}]
    run(f"conversation turn {t} ({len(live)} images)", msgs, live)
    hist.append({"role": "user", "content": f"Step {t}: " + filler(120)})
    hist.append({"role": "assistant", "content": f"Done with step {t}. " + filler(80)})
