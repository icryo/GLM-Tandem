# How long a short seeded job waits when it arrives while a long unseeded request is being read and answered.
import json, os, random, sys, time, threading, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
r0 = random.Random(time.time()); LONG = int(os.environ.get("LONG", "40000")); OUT = int(os.environ.get("OUT", "150"))
def call(body):
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r: d = json.loads(r.read())
    return time.time() - t, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
def short(seed):
    b = dict(model=MODEL, max_tokens=120, temperature=0.8, messages=[{"role": "user", "content": "Notes: " + " ".join(r0.choice(WORDS) for _ in range(1100)) + "\nIgnore the notes and write three sentences about rain."}])
    if seed: b["seed"] = r0.randrange(1, 2**31)
    return b
alone = call(short(True)); res = {}
def big(): res["long"] = call(dict(model=MODEL, max_tokens=OUT, temperature=0.5, messages=[{"role": "user", "content": "Notes: " + " ".join(r0.choice(WORDS) for _ in range(LONG)) + "\nIgnore the notes and write a paragraph about the sea."}]))
t = threading.Thread(target=big); t.start(); time.sleep(3.0)
s1 = call(short(True)); s2 = call(short(False)); t.join()
print(f"seedwait: short seeded job alone {alone[0]:.1f} s ({alone[1]} in, {alone[2]} out) | arriving 3 s into a {res['long'][1]}-token prompt: seeded {s1[0]:.1f} s, then unseeded {s2[0]:.1f} s | long request {res['long'][0]:.1f} s")
