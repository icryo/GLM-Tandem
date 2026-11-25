# Server counters only: single-stream cycles vs multi-stream steps in a log range.
import re, os, collections
a, b = (int(x) for x in os.environ["LOGRANGE"].split(","))
rm = re.compile(r"\[tp-mspec\] streams=(\d+) steps=(\d+) tokens=(\d+) total ([\d.]+) tok/s ([\d.]+) ms/step drafts (\d+)/(\d+) rows dropped (\d+) \| per step: build ([\d.]+) read wait ([\d.]+) finish ([\d.]+) ms \| step after clean ([\d.]+) ms \(n=(\d+)\) after a rejection ([\d.]+) ms \(n=(\d+)\)")
rt = re.compile(r"\[tp-mtp\] coupled=\w+ tok=(\d+) tps=([\d.]+) cycles=(\d+) fast=(\d+) .*?tok/cycle=([\d.]+) head (\d+)/(\d+) lookup (\d+)/(\d+) in (\d+) cycles \| ms/cycle build ([\d.]+) wait ([\d.]+) post ([\d.]+)")
M = dict(n=0, steps=0, tok=0, t=0.0, build=0.0, wait=0.0, fin=0.0, acc=0, off=0, drop=0, clean_t=0.0, clean_n=0, rej_t=0.0, rej_n=0)
S = dict(n=0, tok=0, cyc=0, t=0.0, build=0.0, wait=0.0, post=0.0, wall=0.0)
hist = collections.Counter()
with open(os.path.expanduser("~/llm-stack/logs/tp-cluster.log"), errors="replace") as f:
    for i, l in enumerate(f, 1):
        if i < a: continue
        if i > b: break
        m = rm.search(l)
        if m:
            g = m.groups(); st = int(g[1])
            if st < 3: continue
            M["n"] += 1; M["steps"] += st; M["tok"] += int(g[2]); M["t"] += st * float(g[4]) / 1e3
            M["build"] += st * float(g[8]) / 1e3; M["wait"] += st * float(g[9]) / 1e3; M["fin"] += st * float(g[10]) / 1e3
            M["acc"] += int(g[5]); M["off"] += int(g[6]); M["drop"] += int(g[7])
            M["clean_t"] += float(g[11]) * int(g[12]) / 1e3; M["clean_n"] += int(g[12]); M["rej_t"] += float(g[13]) * int(g[14]) / 1e3; M["rej_n"] += int(g[14])
            hist[min(int(float(g[4]) // 4) * 4, 48)] += st
            continue
        m = rt.search(l)
        if m:
            g = m.groups(); cyc = int(g[2])
            if cyc < 3: continue
            S["n"] += 1; S["tok"] += int(g[0]); S["cyc"] += cyc; S["wall"] += int(g[0]) / max(float(g[1]), 1e-9)
            S["build"] += cyc * float(g[10]) / 1e3; S["wait"] += cyc * float(g[11]) / 1e3; S["post"] += cyc * float(g[12]) / 1e3
print(f"single-stream runs {S['n']}: {S['tok']} tokens in {S['cyc']} cycles; wall {S['wall']:.0f} s -> {S['tok'] / S['wall']:.1f} tok/s, {1e3 * S['wall'] / S['cyc']:.2f} ms/cycle "
      f"(build {1e3 * S['build'] / S['cyc']:.2f} wait {1e3 * S['wait'] / S['cyc']:.2f} post {1e3 * S['post'] / S['cyc']:.2f}), {S['tok'] / S['cyc']:.2f} tok/cycle, {1e3 * S['wall'] / S['tok']:.2f} ms per token")
print(f"multi-stream runs {M['n']}: {M['tok']} tokens in {M['steps']} steps; {M['t']:.0f} s -> {M['tok'] / M['t']:.1f} tok/s total, {1e3 * M['t'] / M['steps']:.2f} ms/step "
      f"(build {1e3 * M['build'] / M['steps']:.2f} read wait {1e3 * M['wait'] / M['steps']:.2f} finish {1e3 * M['fin'] / M['steps']:.2f}), {M['tok'] / M['steps']:.2f} tok/step, {1e3 * M['t'] / M['tok']:.2f} ms per token")
print(f"  drafts accepted {M['acc']}/{M['off']} = {M['acc'] / max(M['off'], 1):.2f}; step after a clean step {1e3 * M['clean_t'] / max(M['clean_n'], 1):.2f} ms (n={M['clean_n']}), after a rejection {1e3 * M['rej_t'] / max(M['rej_n'], 1):.2f} ms (n={M['rej_n']})")
print("  steps by ms/step of their run: " + "  ".join(f"{k}-{k + 4}: {v}" for k, v in sorted(hist.items())))
tot = S["tok"] + M["tok"]
print(f"share of output tokens decoded in multi-stream steps: {100 * M['tok'] / tot:.0f}%; share of decode seconds {100 * M['t'] / (M['t'] + S['wall']):.0f}%")
