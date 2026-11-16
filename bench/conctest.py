# Correctness under concurrency: known-answer tasks with staggered starts and different lengths, so
# sequences join and leave while others are mid-generation (and mid-speculation).
import json, os, random, sys, threading, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
def chat(messages, max_tokens=700, temperature=0.0):
    body = dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return (r["choices"][0]["message"].get("content") or "").strip()
def count_task(n):
    out = chat([{"role": "user", "content": f"Count from 1 to {n}, numbers separated by single spaces, nothing else."}])
    nums = [int(x) for x in out.replace(",", " ").split() if x.isdigit()]
    return nums == list(range(1, n + 1)), out[-40:]
def copy_task(seed):
    rnd = random.Random(seed); words = " ".join(rnd.choice(["alpha", "bravo", "cedar", "delta", "ember", "flint", "gamma", "haze"]) + str(rnd.randint(10, 99)) for _ in range(40))
    out = chat([{"role": "user", "content": f"Repeat this exactly, nothing else:\n{words}"}])
    return out == words, out[-40:]
def sum_task(a, b):
    out = chat([{"role": "user", "content": f"What is {a} + {b}? Answer with the number only."}], max_tokens=30)
    return str(a + b) in out, out[-40:]
def followup_task(n):
    m1 = [{"role": "user", "content": f"Count from 1 to {n}, numbers separated by single spaces, nothing else."}]
    a1 = chat(m1)
    a2 = chat(m1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What was the last number you wrote? Answer with the number only."}], max_tokens=30)
    return str(n) in a2, a2[-40:]
bad = 0; total = 0; lock = threading.Lock()
def run(fn, args, delay, tag):
    global bad, total
    time.sleep(delay)
    try: ok, tail = fn(*args)
    except Exception as e: ok, tail = False, f"{type(e).__name__}: {str(e)[:60]}"
    with lock:
        total += 1; bad += (not ok)
        if not ok: print(f"  FAIL {tag}{args}: ...{tail!r}", flush=True)
rnd = random.Random(7)
for rnd_i in range(int(os.environ.get("ROUNDS", "6"))):
    jobs = []
    for k in range(rnd.randint(2, 7)):
        kind = rnd.choice(["count", "count", "copy", "sum", "follow"])
        if kind == "count": jobs.append((count_task, (rnd.randint(20, 160),), rnd.random() * 2.5, "count"))
        elif kind == "copy": jobs.append((copy_task, (rnd.randint(0, 10**6),), rnd.random() * 2.5, "copy"))
        elif kind == "sum": jobs.append((sum_task, (rnd.randint(100, 999), rnd.randint(100, 999)), rnd.random() * 2.5, "sum"))
        else: jobs.append((followup_task, (rnd.randint(20, 90),), rnd.random() * 2.5, "follow"))
    ths = [threading.Thread(target=run, args=j) for j in jobs]
    t = time.time(); [x.start() for x in ths]; [x.join() for x in ths]
    print(f"round {rnd_i}: {len(jobs)} overlapping requests in {time.time() - t:.1f} s, failures so far {bad}/{total}", flush=True)
print("ALL PASS" if bad == 0 else f"SOME FAILED ({bad}/{total})")
