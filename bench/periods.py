# Sizes and timings only (gateway request log). Compares two periods with the same code:
#   PERIODS="name=YYYY-mm-dd HH:MM,YYYY-mm-dd HH:MM;name2=...,now"
# Jobs (non-streaming) and streams, split by what else was in flight when the request arrived.
import bisect, datetime, json, os, sys, time
def ts(s): return time.time() if s.strip() == "now" else datetime.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M").timestamp()
periods = [(p.split("=")[0], *[ts(x) for x in p.split("=")[1].split(",")]) for p in os.environ["PERIODS"].split(";")]
lo = min(p[1] for p in periods) - 1200
allreq = []
for l in open(os.path.expanduser("~/llm-stack/logs/requests.jsonl")):
    try: d = json.loads(l)
    except Exception: continue
    if (d.get("ts") or 0) >= lo and not d.get("probe"): allreq.append(d)
allreq.sort(key=lambda d: d["ts"]); starts = [d["ts"] for d in allreq]
def inflight(t):
    i0 = bisect.bisect_left(starts, t - 1200); i1 = bisect.bisect_left(starts, t - 1e-6)
    return [d for d in allreq[i0:i1] if d["ts"] + (d.get("total_s") or 0) > t]
def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for name, a, b in periods:
    R = [d for d in allreq if a <= d["ts"] < b and d.get("status") == 200 and (d.get("output_tokens") or 0) > 0 and d.get("total_s")]
    hours = (b - a) / 3600
    print(f"== {name}: {hours:.1f} h, {len(R)} requests, {sum(d['total_s'] for d in R):.0f} request-seconds, {sum(d['output_tokens'] for d in R)} output tokens")
    for kind, sel in (("jobs", lambda d: not d.get("stream")), ("streams", lambda d: d.get("stream"))):
        K = [d for d in R if sel(d)]
        if not K: continue
        print(f"  {kind}: n={len(K)} seconds={sum(d['total_s'] for d in K):.0f} output={sum(d['output_tokens'] for d in K)} prompt p50={q([d.get('input_tokens') or 0 for d in K], .5)}")
        groups = {}
        for d in K:
            fl = inflight(d["ts"])
            key = "alone" if not fl else ("with a stream in flight" if any(x.get("stream") for x in fl) else "with 1 job in flight" if len(fl) == 1 else "with 2+ jobs in flight")
            groups.setdefault(key, []).append(d)
        for key in ("alone", "with 1 job in flight", "with 2+ jobs in flight", "with a stream in flight"):
            G = groups.get(key)
            if not G: continue
            tot = sum(d["total_s"] for d in G); out = sum(d["output_tokens"] for d in G)
            extra = ""
            if kind == "streams":
                extra = f" first visible p50 {q([d.get('ttft_s') or 0 for d in G], .5):.2f} p90 {q([d.get('ttft_s') or 0 for d in G], .9):.2f} s"
            print(f"    {key:24s} n={len(G):5d} ({100 * tot / sum(d['total_s'] for d in K):4.1f}% of seconds) mean {tot / len(G):6.2f} s p50 {q([d['total_s'] for d in G], .5):6.2f} p90 {q([d['total_s'] for d in G], .9):6.2f}  output/second {out / tot:6.1f}{extra}")
    # share of busy time by number of requests in flight
    ev = sorted([(d["ts"], 1) for d in R] + [(d["ts"] + d["total_s"], -1) for d in R]); n = 0; last = None; busy = {}
    for t, dn in ev:
        if last is not None and n > 0: busy[min(n, 4)] = busy.get(min(n, 4), 0) + t - last
        n += dn; last = t
    tb = sum(busy.values()) or 1
    print(f"  busy {100 * tb / (b - a):.1f}% of the period; in flight 1: {100 * busy.get(1, 0) / tb:.1f}%  2: {100 * busy.get(2, 0) / tb:.1f}%  3: {100 * busy.get(3, 0) / tb:.1f}%  4+: {100 * busy.get(4, 0) / tb:.1f}%")
