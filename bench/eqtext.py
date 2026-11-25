# T=0 outputs for fixed synthetic prompts: prints one sha256 per prompt (to compare two builds).
import hashlib, json, os, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions")
PROMPTS = ["Write a 300 word essay on the history of the bicycle.",
           "Write a Python function that merges two sorted lists, with a short docstring and three test cases.",
           "Reply with a JSON object with keys name, year and summary describing the Suez canal.",
           "List ten prime numbers greater than 100 and explain briefly how you checked each one.",
           "Explain in 200 words how a transistor amplifies a signal."]
for p in PROMPTS:
    body = dict(model="default_model", messages=[{"role": "user", "content": p}], max_tokens=int(os.environ.get("MT", "500")), temperature=0)
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]; t = (m.get("reasoning") or "") + "|" + (m.get("content") or "")
    print(hashlib.sha256(t.encode()).hexdigest()[:16], d["usage"]["completion_tokens"], flush=True)
