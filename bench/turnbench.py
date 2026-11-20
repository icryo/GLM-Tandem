# Agent-shaped turns on a long cached conversation: each turn appends ~NEW words and asks for ~OUT tokens.
# Reports per turn: first byte, first content, total. CTX = words in the system prompt (about 2.3 tokens/word).
import json, os, random, sys, time, urllib.request
URL = os.environ.get("URL", "http://localhost:8010/v1/chat/completions"); MODEL = os.environ.get("MODEL", "default_model")
CTX = int(os.environ.get("CTX", "82000")); NEW = int(os.environ.get("NEW", "220")); OUT = int(os.environ.get("OUT", "55")); TURNS = int(os.environ.get("TURNS", "12"))
STREAM = os.environ.get("STREAM", "1") == "1"; EDIT = os.environ.get("EDIT", "")      # EDIT="0.5": before the last turn, change one message at that fraction of the history
WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 < len(w.strip()) < 10][::37]
rnd = random.Random(time.time())
def words(n): return " ".join(rnd.choice(WORDS) for _ in range(n))
def call(msgs, mt):
    body = dict(model=MODEL, stream=STREAM, max_tokens=mt, temperature=0.2, messages=msgs)
    if STREAM: body["stream_options"] = {"include_usage": True}
    raw = json.dumps(body).encode(); t0 = time.time(); fb = first = None; usage = {}; text = []
    with urllib.request.urlopen(urllib.request.Request(URL, raw, {"Content-Type": "application/json"}), timeout=3000) as r:
        if not STREAM:
            d = json.loads(r.read()); usage = d["usage"]; text = [d["choices"][0]["message"].get("content") or ""]; fb = first = time.time()
        else:
            for line in r:
                if fb is None: fb = time.time()
                if not line.startswith(b"data: ") or b"[DONE]" in line: continue
                d = json.loads(line[6:])
                if d.get("usage"): usage = d["usage"]
                for c in d.get("choices") or []:
                    dl = c.get("delta") or {}
                    if dl.get("content"): text.append(dl["content"])
                    if first is None and (dl.get("content") or dl.get("reasoning")): first = time.time()
    end = time.time()
    return dict(p=usage.get("prompt_tokens", 0), c=usage.get("completion_tokens", 0), fb=fb - t0, first=(first or end) - t0, total=end - t0, text="".join(text), kb=len(raw) // 1024)
NCH = int(os.environ.get("CHUNKS", "40"))             # the history is NCH user/assistant pairs so an edit has somewhere to land
msgs = [{"role": "system", "content": "You are a terse assistant. Reference notes (ignore them): " + words(CTX // 4)}]
for i in range(NCH):
    msgs += [{"role": "user", "content": f"Note {i}: " + words(CTX * 3 // 4 // NCH)}, {"role": "assistant", "content": f"Noted {i}."}]
GROW = int(os.environ.get("GROW", "1"))               # build the history in GROW requests so older checkpoints exist
full = msgs; t0 = time.time()
for gstep in range(GROW):
    msgs = full[:1 + 2 * (NCH * (gstep + 1) // GROW)]
    r = call(msgs + [{"role": "user", "content": "Say OK."}], 8)
    if gstep < GROW - 1: full = msgs + [{"role": "user", "content": "Say OK."}, {"role": "assistant", "content": r["text"] or "OK."}] + full[len(msgs):]
print(f"cold: prompt {r['p']} tok ({r['kb']} KB body), {time.time() - t0:.1f} s in {GROW} request(s) = {r['p'] / (time.time() - t0):.0f} tok/s", flush=True)
msgs = msgs + [{"role": "user", "content": "Say OK."}, {"role": "assistant", "content": r["text"] or "OK."}]
rows = []
for t in range(TURNS):
    if EDIT and t == TURNS - 1:
        k = 1 + 2 * int(float(EDIT) * NCH); k += (msgs[k]["role"] != "user"); msgs[k] = dict(msgs[k], content=msgs[k]["content"][: len(msgs[k]["content"]) // 2] + " (edited)")
    q = (f"Repeat exactly this text and nothing else: {words(NEW)}" if os.environ.get("ECHO") == "1" else f"Tool result {t}: {words(NEW)}\nReply with one sentence about rivers. ({time.time()})")
    r = call(msgs + [{"role": "user", "content": q}], OUT); rows.append(r)
    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": r["text"] or "."}]
    print(f"turn {t}: prompt {r['p']} tok, out {r['c']}, first byte {r['fb']:.3f} s, first content {r['first']:.3f} s, total {r['total']:.3f} s" + ("   <-- after edit at %s of history" % EDIT if EDIT and t == TURNS - 1 else ""), flush=True)
med = lambda k: sorted(x[k] for x in rows[1:TURNS - (1 if EDIT else 0)])[(len(rows) - 1 - (1 if EDIT else 0)) // 2]
print(f"MEDIAN: first byte {med('fb'):.3f} s, first content {med('first'):.3f} s, total {med('total'):.3f} s, out {med('c')} tok")
