# Offline check of the prompt-cache policy in tpserve.py with stand-in caches (no model, no GPU).
import os, sys, random, re
import mlx.core as mx, mlx_lm.server as S
src = open(sys.argv[1]).read()
a = src.index("# ---- Prompt cache lookup without the quadratic search"); b = src.index("    # Stock _tokenize finds the end of the system prompt")
exec(compile(src[a:b], "policy", "exec"), dict(os=os, mx=mx, S=S, __name__="policy"))
class C:
    def __init__(self, n): self.nbytes = n * 12600; self.state = []
    def is_trimmable(self): return False
R = random.Random(1); new = lambda n: [R.randrange(1000, 150000) for _ in range(n)]
pc = S.LRUPromptCache(); M = "m"
def ins(t, kind): pc.insert_cache(M, list(t), [C(len(t))], cache_type=kind)
def fetch(t):
    c, rest = pc.fetch_nearest_cache(M, list(t)); return len(t) - len(rest)
def run(n0, cycles, label):
    sysm = new(5000); ins(sysm, "system")
    T = sysm + new(n0 - 5000); reread = []
    ins(T, "user")
    for i in range(cycles):
        ans = new(300); think = new(200)
        ins(T + think + ans, "assistant")                    # answer as generated (thinking included): not a prefix later
        side = T + ans + new(170)                            # turn with two side messages appended
        d = fetch(side); ins(side, "user"); ins(side + new(250), "assistant")
        main = T + ans + new(60)                             # next real turn: side messages removed
        d = fetch(main); reread.append(len(main) - d)
        ins(main, "user"); T = main
    big = [len(it[1]) for it in pc._lru._items if len(it[1]) >= 8000]
    print(f"{label}: re-read on rewind turns: first {reread[0]}, last {reread[-1]}, max {max(reread)} tokens; long entries held {len(big)}, bytes {pc._n_bytes / 2**30:.2f} GiB, spare {len(getattr(S, '_tp_spare', ()))} stale {len(getattr(S, '_tp_stale', ()))}")
run(22000, 12, "22k conversation")
run(190000, 30, "190k conversation")
print("total long entries", sum(len(it[1]) >= 8000 for it in pc._lru._items), "bytes %.1f GiB" % (pc._n_bytes / 2**30))
