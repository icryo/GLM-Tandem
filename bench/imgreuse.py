# Shape-only: hashes of image URLs in request bodies; how often an image comes back, and after how many other images.
import json, os, sys, datetime, hashlib, collections
def ts(s): return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()
T0 = ts(os.environ["T0"]); T1 = ts(os.environ["T1"])
CAPROOT = os.path.expanduser("~/llm-stack/captures"); files = []
for sub in os.listdir(CAPROOT):
    p = os.path.join(CAPROOT, sub)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        try: ns = int(f[:-5].split("-", 1)[0])
        except ValueError: continue
        if T0 <= ns / 1e9 < T1: files.append((ns, os.path.join(p, f)))
files.sort()
order = collections.OrderedDict(); dist = collections.Counter(); nimg = 0; nreq = 0; kinds = collections.Counter(); sizes = []
hits = {n: 0 for n in (8, 32, 64, 256, 1024)}; hits_job = dict(hits); nimg_job = 0; newimg_job = 0
seen_in_stream = collections.Counter()
for ns, f in files:
    try: b = json.load(open(f))["body"]
    except Exception: continue
    stream = bool(b.get("stream")); urls = []
    for m in b.get("messages") or []:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if p.get("type") == "image_url": urls.append((p.get("image_url") or {}).get("url", ""))
    if not urls: continue
    nreq += 1
    for u in urls:
        h = hashlib.blake2b(u.encode(), digest_size=8).digest(); nimg += 1
        if not stream: nimg_job += 1
        if len(sizes) < 4000: sizes.append(len(u))
        if h in order:
            d = len(order) - 1 - list(order).index(h) if len(order) < 4096 else None
            order.move_to_end(h)
            for n in hits:
                if d is not None and d < n:
                    hits[n] += 1
                    if not stream: hits_job[n] += 1
        else:
            order[h] = 1
            if not stream: newimg_job += 1
            while len(order) > 4096: order.popitem(last=False)
print(f"requests with images {nreq}, image parts {nimg} (in jobs {nimg_job}); data-URL size median {sorted(sizes)[len(sizes) // 2] / 1024:.0f} KiB, p90 {sorted(sizes)[int(.9 * len(sizes))] / 1024:.0f} KiB")
for n in sorted(hits): print(f"  memo of {n:4d} images: {100 * hits[n] / max(nimg, 1):5.1f}% of all image parts already in it; jobs only {100 * hits_job[n] / max(nimg_job, 1):5.1f}%")
