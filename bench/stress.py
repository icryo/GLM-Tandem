import json, sys, time, threading, urllib.request, socket
URL = "http://localhost:8010/v1/chat/completions"
def post(body, stream=False, cancel_after=None, timeout=600):
    body = dict(model="default_model", **body)
    if stream: body["stream"] = True
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        d = json.load(r); return d["choices"][0]["message"].get("content") or ""
    n = 0; out = []
    for line in r:
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            d = json.loads(line[6:])
            for c in d.get("choices", []):
                s = (c.get("delta", {}).get("content") or "") + (c.get("delta", {}).get("reasoning") or "")
                if s: n += 1; out.append(s)
            if cancel_after and n >= cancel_after:
                r.close(); return "".join(out)
    return "".join(out)
res = {}
def run(name, *a, **k):
    try: res[name] = post(*a, **k)
    except Exception as e: res[name] = f"ERR {type(e).__name__}: {e}"
q = lambda t, **k: dict(messages=[{"role": "user", "content": t}], max_tokens=k.pop("max_tokens", 400), temperature=k.pop("temperature", 0.0), **k)
# 1. three concurrent requests
ths = [threading.Thread(target=run, args=(f"conc{i}", q(f"What is {i+2}*{i+11}? Just the number."))) for i in range(3)]
[t.start() for t in ths]; [t.join() for t in ths]
for i in range(3): print(f"conc{i}:", "PASS" if str((i+2)*(i+11)) in res[f"conc{i}"] else "FAIL", repr(res[f"conc{i}"][-30:]))
# 2. cancel a stream mid-generation, then a normal request must work
run("cancel", q("Write a 600 word story about a lighthouse.", max_tokens=900), stream=True, cancel_after=40)
print("cancelled after", len(res["cancel"]), "chars")
time.sleep(1.0)
run("after_cancel", q("What is 12*12? Just the number."))
print("after cancel:", "PASS" if "144" in res["after_cancel"] else "FAIL", repr(res["after_cancel"][-30:]))
# 3. repetition penalty (logits processor -> plain path)
run("rep", q("List the first 12 prime numbers, comma separated, nothing else.", repetition_penalty=1.05))
print("repetition_penalty path:", "PASS" if "37" in res["rep"] and "2, 3, 5" in res["rep"] else "FAIL", repr(res["rep"][-50:]))
# 4. back to MTP path right after
run("after_rep", q("What is 9*8? Just the number."))
print("after processor path:", "PASS" if "72" in res["after_rep"] else "FAIL", repr(res["after_rep"][-30:]))
# 5. cancel while another request is queued
t1 = threading.Thread(target=run, args=("c2", q("Write a 500 word essay on tides.", max_tokens=900)), kwargs=dict(stream=True, cancel_after=25))
t2 = threading.Thread(target=run, args=("queued", q("What is 7*6? Just the number.")))
t1.start(); time.sleep(0.3); t2.start(); t1.join(); t2.join()
print("queued behind cancelled:", "PASS" if "42" in res["queued"] else "FAIL", repr(res["queued"][-30:]))
# 6. streaming long generation, sampled
t = time.time(); run("long", q("Write a 700 word essay on the history of the transistor.", max_tokens=1200, temperature=0.6), stream=True)
print("long sampled stream:", "PASS" if len(res["long"]) > 2500 and "transistor" in res["long"].lower() else "FAIL", len(res["long"]), "chars in", round(time.time() - t, 1), "s")
