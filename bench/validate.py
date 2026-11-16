# Correctness checks against the lean TP+MTP server (OpenAI API on :8010).
import json, sys, time, urllib.request
URL = "http://localhost:8010/v1/chat/completions"
def chat(messages, **kw):
    body = dict(model="default_model", messages=messages, max_tokens=kw.pop("max_tokens", 300), temperature=kw.pop("temperature", 0.0), **kw)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r: d = json.load(r)
    m = d["choices"][0]; u = d.get("usage", {})
    return m["message"], m["finish_reason"], u, time.time() - t
ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, detail, flush=True)

# 1. arithmetic + factual, greedy
m, fr, u, dt = chat([{"role": "user", "content": "What is 17*23? Answer with just the number."}], max_tokens=400)
check("arithmetic", "391" in (m.get("content") or ""), repr((m.get("content") or "")[-60:]))
# 2. exact-copy (catches any token-order / dropped-token bug in the two-token emit path)
src = "alpha-7 bravo_12 charlie.33 delta/4 echo:55 foxtrot;6 golf=77 hotel+8 india*99 juliet^10 kilo%11 lima#12 mike@13 november!14 oscar?15"
m, fr, u, dt = chat([{"role": "user", "content": "Repeat the following line exactly, three times, one per line, nothing else:\n" + src}], max_tokens=600)
lines = [l.strip() for l in (m.get("content") or "").strip().splitlines() if l.strip()]
check("exact copy x3", lines.count(src) == 3, f"{len(lines)} lines, matches={lines.count(src)}")
# 3. counting sequence (long, order-sensitive)
m, fr, u, dt = chat([{"role": "user", "content": "Count from 1 to 120, comma separated, nothing else."}], max_tokens=900)
nums = [x.strip() for x in (m.get("content") or "").replace("\n", " ").split(",") if x.strip()]
check("count 1..120", nums == [str(i) for i in range(1, 121)], f"got {len(nums)} items, tail {nums[-3:]}")
# 4. max_tokens cut mid-cycle, then continue the same conversation (cache reuse after undo), many lengths
base = [{"role": "user", "content": "Write the numbers from 1 to 60 in English words, comma separated."}]
full, _, _, _ = chat(base, max_tokens=700)
ref = full.get("content") or ""
bad = 0
for n in range(20, 40):
    part, fr, _, _ = chat(base, max_tokens=n + 200 if False else n)
    # follow-up turn on the truncated transcript must still be coherent: ask for a fixed token
    conv = base + [{"role": "assistant", "content": part.get("content") or ""}, {"role": "user", "content": "Reply with exactly the word: pineapple"}]
    m2, fr2, _, _ = chat(conv, max_tokens=300)
    if "pineapple" not in (m2.get("content") or "").lower(): bad += 1
check("truncate at 20..39 tokens then follow-up", bad == 0, f"{bad} bad of 20")
# 5. determinism: greedy twice gives the same text
a, *_ = chat(base, max_tokens=200); b, *_ = chat(base, max_tokens=200)
check("greedy determinism", a.get("content") == b.get("content"))
# 6. tool call
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
m, fr, u, dt = chat([{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], tools=tools, max_tokens=500)
tc = m.get("tool_calls") or []
check("tool call", bool(tc) and tc[0]["function"]["name"] == "get_weather" and "Paris" in tc[0]["function"]["arguments"], f"finish={fr} {json.dumps(tc)[:160]}")
# 7. stop sequence
m, fr, u, dt = chat([{"role": "user", "content": "Say: one two three four five six seven eight nine ten"}], stop=["six"], max_tokens=300)
check("stop sequence", "six" not in (m.get("content") or "") and "five" in (m.get("content") or ""), repr((m.get("content") or "")[-40:]))
# 8. sampled run sanity
m, fr, u, dt = chat([{"role": "user", "content": "In two sentences, why is the sky blue?"}], temperature=0.7, max_tokens=400)
check("sampled answer", "scatter" in (m.get("content") or "").lower(), repr((m.get("content") or "")[:80]))
print("ALL PASS" if ok else "SOME FAILED")
