# One stream with a long context (CTX words of filler, text only) generating GEN tokens while N short streams
# generate next to it: decode speed of each. Synthetic content.
import json, os, random, sys, threading, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8080/v1/chat/completions"); MODEL = os.environ.get("MODEL", "glm-5.3-flash")
CTX = int(os.environ.get("CTX", "16000")); GEN = int(os.environ.get("GEN", "300")); N = int(os.environ.get("N", "1")); TEMP = float(os.environ.get("TEMP", "0.6"))
R = random.Random(int(os.environ.get("RSEED", "3")))
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
filler = " ".join(R.choice(WORDS) for _ in range(CTX))
def one(msgs, res, k, mt):
    body = dict(model=MODEL, stream=True, stream_options={"include_usage": True}, max_tokens=mt, temperature=TEMP, messages=msgs)
    t0 = time.time(); first = None; usage = {}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3000) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if first is None and d.get("choices") and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning") for c in d["choices"]): first = time.time()
    end = time.time(); ct = usage.get("completion_tokens", 0)
    res[k] = (first - t0, (ct - 1) / max(end - first, 1e-6), ct, usage.get("prompt_tokens", 0))
Q = "Ignore the word list above. Write a detailed essay on how bridges carry load."
if os.environ.get("IMG") == "1":
    import base64, io
    from PIL import Image
    im = Image.new("RGB", (448, 448), (40, 90, 160)); b = io.BytesIO(); im.save(b, "JPEG")
    long_msgs = [{"role": "system", "content": filler}, {"role": "user", "content": [{"type": "text", "text": Q}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}]}]
else:
    long_msgs = [{"role": "system", "content": filler}, {"role": "user", "content": Q}]
res = {}; one(long_msgs, res, "warm", 8); print(f"long prompt {res['warm'][3]} tokens, read in {res['warm'][0]:.1f} s", flush=True)
for rnd in range(int(os.environ.get("ROUNDS", "2"))):
    res = {}; ths = [threading.Thread(target=one, args=(long_msgs, res, "long", GEN))]
    ths += [threading.Thread(target=one, args=([{"role": "user", "content": f"Write a detailed essay on the history of the bicycle (variant {time.time()} {i})."}], res, f"s{i}", GEN)) for i in range(N)]
    t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]; wall = time.time() - t
    print(f"round {rnd}: long {res['long'][1]:.0f} tok/s (first token {res['long'][0]:.1f} s) | short " + " ".join(f"{res[f's{i}'][1]:.0f}" for i in range(N)) + f" tok/s | total {sum(v[2] for v in res.values()) / wall:.0f} tok/s over {wall:.1f} s", flush=True)
