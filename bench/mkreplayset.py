# Build a replay set from captured request bodies (stays on the Mac). Seeded JSON-mode jobs only, in arrival order.
import json, os, glob, time, sys
T0 = time.mktime((2026, 11, 20, 4, 2, 30, 0, 0, -1)); T1 = T0 + 86400
N = int(os.environ.get("N", "240")); out = os.path.expanduser("~/llm-stack/cluster/replayset.jsonl")
fs = sorted((os.path.getmtime(f), f) for f in glob.glob(os.path.expanduser("~/llm-stack/captures/*/*")))
rows = []
for m, f in fs:
    if not T0 <= m < T1: continue
    try: b = json.load(open(f)).get("body", {})
    except Exception: continue
    if b.get("seed") is None or b.get("stream"): continue
    rows.append(b)
step = max(1, len(rows) // N); rows = rows[::step][:N]
with open(out, "w") as fh:
    for b in rows: fh.write(json.dumps(b) + "\n")
img = sum('"image_url"' in json.dumps(b)[:200000] for b in rows)
print(f"{len(rows)} bodies, {img} with images, temps {sorted(set(b.get('temperature') for b in rows))}, {os.path.getsize(out)/1e6:.1f} MB")
