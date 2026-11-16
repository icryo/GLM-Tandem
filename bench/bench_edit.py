# Copy-heavy workloads: decode speed when most output tokens repeat earlier context.
import json, os, sys, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
def stream(prompt, max_tokens, temperature=0.0):
    body = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    first = None; usage = None; out = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {}); s = (dl.get("content") or "") + (dl.get("reasoning") or "")
                if s:
                    if first is None: first = time.time()
                    out.append(dl.get("content") or "")
    ct = usage["completion_tokens"]; return "".join(out), ct / (time.time() - first), ct
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate.py")).read()
tasks = {
  "code edit (rename a function, return whole file)": ("Here is a Python file:\n```python\n" + src + "\n```\nRename the function `chat` to `send_chat` everywhere it is defined or called. Output the complete updated file in one code block and nothing else.", 2500),
  "add docstrings (whole file back)": ("Here is a Python file:\n```python\n" + src[:3500] + "\n```\nAdd a one-line docstring to each function. Output the complete updated file in one code block and nothing else.", 2000),
  "JSON extraction from text": ("Text: " + " ".join(f"Order {1000+i} was placed by customer C{i*7%50:03d} for {i%9+1} units of SKU-{(i*13)%97:04d} on 2026-0{i%9+1}-1{i%9}." for i in range(40)) + "\nReturn a JSON array with one object per order: order, customer, units, sku, date. JSON only.", 2500),
  "prose essay (little repetition)": ("Write a 500 word essay on the history of the transistor.", 900),
}
for name, (prompt, mt) in tasks.items():
    for temp in ((0.0, 0.6) if "essay" not in name else (0.6,)):
        text, tps, ct = stream(prompt, mt, temp)
        ok = ""
        if "rename" in name: ok = "  correct=" + str("def send_chat(" in text and "def chat(" not in text and text.count("send_chat(") >= 8)
        print(f"{name:50s} temp={temp}: {tps:6.1f} tok/s  ({ct} tokens){ok}", flush=True)
