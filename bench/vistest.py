# Image requests against the lean server: synthetic images with known content.
import base64, io, json, os, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFont
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
def png(draw_fn, size=(448, 448), bg="white"):
    im = Image.new("RGB", size, bg); draw_fn(ImageDraw.Draw(im)); b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
def font(sz):
    for f in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()
def ask(parts, max_tokens=120, temperature=0):
    body = dict(model=MODEL, messages=[{"role": "user", "content": parts}], max_tokens=max_tokens, temperature=temperature)
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
    m = r["choices"][0]["message"]; u = r.get("usage", {})
    return (m.get("content") or "").strip(), u, time.time() - t
ok = 0; tests = []
img1 = png(lambda d: d.text((60, 150), "7391", fill="black", font=font(130)))
img2 = png(lambda d: (d.rectangle((0, 0, 448, 448), fill=(220, 30, 30)), d.ellipse((120, 120, 330, 330), fill=(30, 60, 220))))
img3 = png(lambda d: d.text((40, 300), "HELLO MAC", fill="darkgreen", font=font(110)), size=(896, 672))
tests.append(("read digits", [{"type": "text", "text": "What number is written in this image? Answer with the number only."}, {"type": "image_url", "image_url": {"url": img1}}], lambda a: "7391" in a))
tests.append(("colors and shape", [{"type": "image_url", "image_url": {"url": img2}}, {"type": "text", "text": "What shape is in the middle, what color is it, and what color is the background? One sentence."}], lambda a: "blue" in a.lower() and "red" in a.lower() and ("circle" in a.lower() or "round" in a.lower() or "disc" in a.lower() or "disk" in a.lower())))
tests.append(("larger image text", [{"type": "image_url", "image_url": {"url": img3}}, {"type": "text", "text": "Transcribe the text in the image exactly."}], lambda a: "HELLO" in a.upper() and "MAC" in a.upper()))
tests.append(("two images", [{"type": "image_url", "image_url": {"url": img1}}, {"type": "image_url", "image_url": {"url": img2}}, {"type": "text", "text": "Which image contains a number, the first or the second? And what is the number?"}], lambda a: "7391" in a and "first" in a.lower()))
for name, parts, check in tests:
    try:
        a, u, dt = ask(parts); good = check(a); ok += good
        print(f"{'PASS' if good else 'FAIL'} {name}: prompt_tokens={u.get('prompt_tokens')} {dt:.2f}s  {a[:110]!r}")
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__} {str(e)[:150]}")
a, u, dt = ask("What is 17 * 23? Answer with the number only.", 30)
print(f"{'PASS' if '391' in a else 'FAIL'} text after images: {a[:40]!r}"); ok += "391" in a
print("ALL PASS" if ok == len(tests) + 1 else "SOME FAILED")
