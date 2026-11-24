# Run the gateway with its supervisor command set to fake_upstream.py (port 8011) and its own port 8081, then this.
# Sends jobs and one stream through the test gateway while its upstream is killed.
import json, subprocess, sys, threading, time, urllib.request
URL = "http://127.0.0.1:8081/v1/chat/completions"; res = []
def job(name, secs=2, stream=False):
    b = dict(model="x", messages=[dict(role="user", content="hi")], fake_seconds=secs, stream=stream)
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=120) as r:
            body = r.read().decode(); st = r.status
        ok = ("w4" in body and "[DONE]" in body and "error" not in body) if stream else json.loads(body)["choices"][0]["message"]["content"] == "ok"
        res.append((name, st, "ok" if ok else "BAD: " + body[-120:].replace("\n", " "), round(time.time() - t, 1), body.count(": ping")))
    except Exception as e:
        res.append((name, getattr(e, "code", "ERR"), repr(e)[:70], round(time.time() - t, 1), 0))
for i in range(60):
    try:
        if json.loads(urllib.request.urlopen("http://127.0.0.1:8081/gateway/health", timeout=2).read())["upstream"] == "healthy": break
    except Exception: pass
    time.sleep(1)
job("before", 1)
th = [threading.Thread(target=job, args=("in flight when the upstream dies", 6))]; th[0].start(); time.sleep(1.5)
subprocess.run(["pkill", "-9", "-f", "fake_upstream.py"]); tk = time.time()
for i in range(6):
    time.sleep(1.5); t = threading.Thread(target=job, args=(f"sent {1.5 * (i + 1):.1f} s after the kill", 1)); t.start(); th.append(t)
t = threading.Thread(target=job, args=("stream sent 9.5 s after the kill", 1, True)); t.start(); th.append(t)
for t in th: t.join()
job("after", 1)
for r in res: print("  %-34s status %-4s %-40s %5.1f s  pings %d" % r)
print("failed:", sum(1 for r in res if r[2] != "ok"), "of", len(res))
