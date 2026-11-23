# Long agent conversation that keeps only its last KEEP images: each time an image message is added the
# oldest one is removed (some 10-16k tokens before the end). Measures time to first token per turn, and
# optionally the total time of a short seeded job fired while the turn's prompt is being read.
# Synthetic content only. Shape taken from real traffic: system + tools, then user / assistant(tool_calls) /
# tool messages, ~150k tokens.
#   URL=http://localhost:8010/v1/chat/completions TARGET=150000 TURNS=12 python3 editconv.py
import base64, io, json, os, random, sys, threading, time, urllib.request
from PIL import Image, ImageDraw
URL = os.environ.get("URL", "http://localhost:8080/v1/chat/completions"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
TARGET = int(os.environ.get("TARGET", "150000")); TURNS = int(os.environ.get("TURNS", "12")); KEEP = int(os.environ.get("KEEP", "5"))
IMG_EVERY = int(os.environ.get("IMG_EVERY", "2"))        # every n-th measured turn adds an image (and drops the oldest)
GAP_MSGS = int(os.environ.get("GAP_MSGS", "8"))          # plain messages between image messages in the built history
JOB = os.environ.get("JOB", "0") == "1"; MAXT = int(os.environ.get("MAXT", "120")); SEED = int(os.environ.get("RSEED", "7"))
TOOLS_ON = os.environ.get("TOOLS", "1") == "1"; TEMP = float(os.environ.get("TEMP", "0"))
R = random.Random(SEED); BAD = [0]
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
def words(n, r=R): return " ".join(r.choice(WORDS) for _ in range(n))
def image(r=R, size=(448, 448)):
    im = Image.new("RGB", size, (r.randrange(256), r.randrange(256), r.randrange(256))); d = ImageDraw.Draw(im)
    for _ in range(12):
        x = sorted(r.randrange(size[0]) for _ in range(2)); y = sorted(r.randrange(size[1]) for _ in range(2))
        d.rectangle([x[0], y[0], x[1], y[1]], fill=(r.randrange(256), r.randrange(256), r.randrange(256)))
    d.text((20, 20), str(r.randrange(10**6)), fill="black")
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
TOOLS = [dict(type="function", function=dict(name=f"tool_{i}", description=words(60, random.Random(100 + i)),
              parameters=dict(type="object", properties=dict(arg=dict(type="string", description=words(12, random.Random(200 + i)))), required=["arg"]))) for i in range(12)]
_id = [0]; CODES = {}
def step():          # one agent step: assistant tool call + tool result
    _id[0] += 1; cid = f"call_{_id[0]:06d}"
    return [dict(role="assistant", content=words(R.randint(5, 25)), tool_calls=[dict(id=cid, type="function", function=dict(name=f"tool_{R.randrange(12)}", arguments=json.dumps(dict(arg=words(R.randint(3, 12))))))]),
            dict(role="tool", tool_call_id=cid, content=words(R.randint(120, 420)) + f" Record {cid}: code {CODES.setdefault(cid, R.randrange(100000, 999999))}.")]
def image_msg(): return dict(role="user", content=[dict(type="text", text=words(8)), dict(type="image_url", image_url=dict(url=image()))])
def has_img(m): return isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"])
def call(msgs, stream=True, max_tokens=MAXT, seed=None, timeout=3000):
    body = dict(model=MODEL, messages=msgs, max_tokens=max_tokens, temperature=TEMP, stream=stream)
    if stream: body["stream_options"] = {"include_usage": True}
    if TOOLS_ON and stream: body["tools"] = TOOLS
    if seed is not None: body["seed"] = seed; body["temperature"] = 0.25
    t0 = time.time(); first = None; usage = {}; text = ""
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=timeout) as r:
        if not stream:
            d = json.loads(r.read()); return dict(total=time.time() - t0, usage=d.get("usage", {}), text=d["choices"][0]["message"].get("content") or "")
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices") or []:
                dl = c.get("delta") or {}
                if first is None and (dl.get("content") or dl.get("reasoning") or dl.get("tool_calls")): first = time.time()
                text += dl.get("content") or ""
    return dict(ttft=(first or time.time()) - t0, total=time.time() - t0, usage=usage, text=text)
def job(delay, out):
    time.sleep(delay); r = random.Random(time.time())
    msgs = [dict(role="system", content=words(80, r)), dict(role="user", content=[dict(type="text", text=words(1500, r) + " Reply with a short JSON object."), dict(type="image_url", image_url=dict(url=image(r)))])]
    try: out.append(call(msgs, stream=False, max_tokens=120, seed=r.randrange(10**9))["total"])
    except Exception as e: out.append(f"ERR {e!r}"[:80])

# tokens per word, from a small probe
p = call([dict(role="user", content=words(2000))], max_tokens=1)["usage"].get("prompt_tokens", 6000); tpw = p / 2000.0
print(f"probe: {tpw:.2f} tokens per word", flush=True)
msgs = [dict(role="system", content=words(int(6000 / tpw))), dict(role="user", content=words(60))]
est = 6000 + 2500; n_since = 0; imgs = 0
while est < TARGET:
    s = step(); msgs += s; est += int((len(s[0]["content"].split()) + len(s[1]["content"].split()) + 30) * tpw); n_since += 2
    # images only in the last part of the history, like a client that keeps the last KEEP
    if n_since >= GAP_MSGS and est > TARGET - (KEEP + 1) * GAP_MSGS / 2 * 300 * tpw and imgs < KEEP:
        msgs.append(image_msg()); imgs += 1; est += 300; n_since = 0
while imgs < KEEP: msgs.append(image_msg()); imgs += 1; msgs += step()
print(f"built {len(msgs)} messages, about {est} tokens, {imgs} images at {[i for i, m in enumerate(msgs) if has_img(m)]}", flush=True)
print("turn  kind                 msgs  prompt_tok  cached  first_token_s  total_s   job_s", flush=True)
for t in range(TURNS + 1):
    kind = "cold start" if t == 0 else "append"
    if t and t % IMG_EVERY == 0:
        old = next(i for i, m in enumerate(msgs) if has_img(m)); dist = len(msgs) - old
        del msgs[old]; msgs.append(image_msg()); kind = f"image swap (-{dist} msgs)"
    elif t:
        msgs += step()
    # Recall check: one record from the start, the last one before the oldest image, and the newest one.
    oi = next(i for i, m in enumerate(msgs) if has_img(m))
    ids = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]; before = [m["tool_call_id"] for m in msgs[:oi] if m["role"] == "tool"]
    want = [ids[3], before[-1], ids[-1]]
    q = "Reply with only the three codes of records " + ", ".join(want) + ", separated by spaces."
    if msgs[-1]["role"] != "user": msgs.append(dict(role="user", content=q))
    elif has_img(msgs[-1]): msgs[-1]["content"][0]["text"] = q
    else: msgs[-1]["content"] = q
    jo = []; th = None
    if JOB and t: th = threading.Thread(target=job, args=(1.0, jo)); th.start()
    try: r = call(msgs)
    except Exception as e: print(f"{t:3d}  {kind:20s} ERROR {e!r}"[:200], flush=True); break
    if th: th.join()
    u = r["usage"]; cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", "")
    ok = sum(str(CODES[w]) in r["text"] for w in want); BAD[0] += ok < 3
    print(f"{t:3d}  {kind:20s} {len(msgs):5d} {u.get('prompt_tokens', 0):9d} {str(cached):>7s} {r['ttft']:13.1f} {r['total']:8.1f}   recall {ok}/3   {('job %.1f s' % jo[0] if jo and not isinstance(jo[0], str) else jo[0]) if jo else ''}", flush=True)
    msgs.append(dict(role="assistant", content=r["text"] or words(10)))
    msgs.append(dict(role="user", content=words(15)))
print("DONE", "recall failures:", BAD[0])
