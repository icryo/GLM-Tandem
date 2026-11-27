# ragtest_x.py - offline test of TP_RAGGED (ragkv_x.py) against the stock rollback (prepare/finalize roll).
#
#   python ragtest_x.py test [SEED] [STEPS]    equality test: one real GLM sparse-attention layer (per-rank shape:
#                                              32 heads, latent 512, indexer 32 x 128, top-k 2048; random bf16
#                                              weights), two copies of the same batch caches driven by the same
#                                              random two-token steps; copy A uses the stock _trim_rows, copy B the
#                                              ragged one. Asserts bit-identical attention outputs every step and
#                                              identical logical cache contents, through joins, leaves (extract +
#                                              filter), cancels, one-token steps, the dense (< top-k) and the selected
#                                              (>= top-k) attention paths.
#   python ragtest_x.py bench L1,L2 [REPS]     per-layer cost of one rejection: stock trim + roll vs ragged
#                                              bookkeeping (+ the next step's write and mask), two streams.
# Run on a Mac with the oMLX python, from this directory (needs the oMLX environment, so not -I).
import importlib, json, os, random, sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omlx.patches.glm5_next_mlx_lm.model as RM
RM.glm5_next_vlm_runtime.apply()
Lp = RM._language
import mlx_vlm.models.cache as VC
import mlx_lm.models.cache as LC
import ragkv_x

MP = os.environ.get("MP", "/Users/a/models/GLM-5.3-Flash-TF-4bit-MTP")
D = 4096


def stock_trim_rows(pair, trims):
    kv, pool = pair[0], pair[1]
    u = min(trims); extra = [t - u for t in trims]
    if u: kv.trim(u)
    if any(extra):
        kv.prepare(right_padding=extra); kv.finalize()
    pool.trim_rows(trims)


def rag_trim_rows(pair, trims):
    ragkv_x.rag_trim_rows(pair[0], pair[1], trims)
    pair[1].trim_rows(trims)


def make_layer(seed):
    C = importlib.import_module(Lp.__name__.rsplit(".", 1)[0] + ".config")
    tcd = json.load(open(MP + "/config.json"))
    tcd = dict(tcd.get("text_config", tcd))
    tcd["num_attention_heads"] = tcd["num_attention_heads"] // 2          # one rank's share
    if hasattr(C.TextConfig, "from_dict"):
        tc = C.TextConfig.from_dict(tcd)
    else:
        tc = C.TextConfig(**{k: v for k, v in tcd.items() if k in C.TextConfig.__dataclass_fields__})
    attn = Lp.Glm5NextSparseAttention(tc)
    mx.random.seed(seed)
    new = []
    for k, v in tree_flatten(attn.parameters()):
        if "norm" in k and k.endswith("weight"):
            nv = mx.ones(v.shape)
        elif k.endswith("bias") and "norm" in k:
            nv = mx.zeros(v.shape)
        elif v.ndim >= 2:
            nv = mx.random.normal(v.shape) * (1.0 / (v.shape[-1] ** 0.5))
        else:
            nv = mx.random.normal(v.shape) * 0.02
        new.append((k, nv.astype(mx.bfloat16)))
    attn.update(tree_unflatten(new))
    mx.eval(attn.parameters())
    return attn, tc


def single_cache(attn, n, rng):
    c = VC.CacheList(VC.KVCache(), LC.PoolingCache(attn.indexer.index_kpool))
    x = mx.array(rng.standard_normal((1, n, D)).astype("float32") * 0.5).astype(mx.bfloat16)
    pos = 0
    while pos < n:                                     # prompt in blocks, like the server's prefill steps
        b = min(1024, n - pos)
        xb = x[:, pos:pos + b]
        m = Lp.create_attention_mask(xb, c[0], return_array=True)
        mx.eval(attn(xb, m, c))
        pos += b
    return c


def merged(singles):
    return VC.CacheList.merge(singles)


def step(attn, cl, x):
    m = Lp.create_attention_mask(x, cl[0], return_array=True)
    return attn(x, m, cl)


def logical(cl, b):
    kv, pool = cl[0], cl[1]
    lp = int(kv.left_padding[b].item())
    end = kv._tp_end[b] if getattr(kv, "_tp_end", None) is not None else kv._idx
    return kv.keys[b, :, lp:end], int(kv.offset[b].item())


def same_pool(pa, pb):
    if pa._pool_lengths != pb._pool_lengths or pa.remainder != pb.remainder or pa._processed != pb._processed:
        return False
    if (pa.pooled is None) != (pb.pooled is None) or (pa.buf_kv is None) != (pb.buf_kv is None):
        return False
    for b, pl in enumerate(pa._pool_lengths):
        if pl and not mx.array_equal(pa.pooled[b, :pl], pb.pooled[b, :pl]).item():
            return False
        r = pa.remainder[b]
        if r and not mx.array_equal(pa.buf_kv[b, :r], pb.buf_kv[b, :r]).item():
            return False
    return True


def check_state(cA, cB, where):
    kA, kB = cA[0], cB[0]
    assert kA._idx == kB._idx, (where, "extent", kA._idx, kB._idx)
    B = kA.offset.shape[0]
    for b in range(B):
        a, oa = logical(cA, b); bb, ob = logical(cB, b)
        assert oa == ob == a.shape[1] == bb.shape[1], (where, b, oa, ob, a.shape, bb.shape)
        assert mx.array_equal(a, bb).item(), (where, "keys differ", b)
    assert same_pool(cA[1], cB[1]), (where, "pool differs")


def test(seed=1, steps=500):
    import numpy as np
    rng = np.random.default_rng(seed); pyr = random.Random(seed)
    print("install:", ragkv_x.install(topk=2048), flush=True)
    print("BatchKVCache methods:", {k: getattr(VC.BatchKVCache.__dict__[k], "__qualname__", "?") for k in ("trim", "make_mask", "update_and_fetch")})
    attn, tc = make_layer(seed)
    topk = attn.indexer.index_topk
    # Which attention kernels the steps reach (both copies counted together).
    calls = {"fused_batched": 0, "gathered": 0, "sdpa": 0}
    SA = Lp.Glm5NextSparseAttention
    for name, key in (("_decode_fused_batched", "fused_batched"), ("_gathered_attention", "gathered")):
        orig = getattr(SA, name)
        def wrapped(self, *a, _o=orig, _k=key, **k):
            calls[_k] += 1
            return _o(self, *a, **k)
        setattr(SA, name, wrapped)
    _sdpa = Lp.scaled_dot_product_attention
    def sdpa(*a, **k):
        calls["sdpa"] += 1
        return _sdpa(*a, **k)
    Lp.scaled_dot_product_attention = sdpa
    if os.environ.get("RAG_NOFUSED") == "1":         # force the general indexer + gathered-attention path
        SA._batched_fused_ok = lambda self, *a, **k: False
    if os.environ.get("RAG_BREAK") == "1":            # negative control: ragged ends without per-stream writes
        ragkv_x._ragged_write = lambda self, k, v: (setattr(self, "_tp_end", None), type(self).__dict__["update_and_fetch"].__wrapped__(self, k, v))[1]
    print(f"layer: heads {attn.num_heads} latent {attn.kv_lora_rank} indexer {attn.indexer.n_heads}x{attn.indexer.head_dim} topk {topk} kpool {attn.indexer.index_kpool}", flush=True)
    lens = [int(x) for x in os.environ.get("LENS", "1990,300").split(",")]
    singles = [single_cache(attn, n, rng) for n in lens]
    cA, cB = merged(singles), merged(singles)
    del singles
    mx.eval(cA[0].keys, cB[0].keys)
    check_state(cA, cB, "start")
    t0 = time.time()
    n_cmp = n_rej = n_cancel = n_join = n_leave = n_one = 0
    paths = {"dense": 0, "selected": 0}
    for it in range(steps):
        B = cA[0].offset.shape[0]
        tok = 1 if pyr.random() < 0.1 else 2
        before = max(cA[1]._processed)
        paths["dense" if before + tok <= topk else "selected"] += 1
        x = mx.array(rng.standard_normal((B, tok, D)).astype("float32") * 0.5).astype(mx.bfloat16)
        yA = step(attn, cA, x); yB = step(attn, cB, x)
        mx.eval(yA, yB)
        assert mx.array_equal(yA, yB).item(), f"step {it}: attention outputs differ (B={B}, tok={tok}, paths={paths})"
        n_cmp += 1
        if tok == 2:
            if pyr.random() < 0.06:
                cA.trim(2); cB.trim(2); n_cancel += 1               # _ms_cancel: the queued step is taken back out
            else:
                trims = [0 if pyr.random() < 0.7 else 1 for _ in range(B)]
                stock_trim_rows(cA.caches, trims); rag_trim_rows(cB.caches, trims)
                n_rej += sum(trims) > 0
        else:
            n_one += 1
        check_state(cA, cB, f"step {it}")
        r = pyr.random()
        if r < 0.03 and B < 4:                                     # a stream joins
            n = pyr.choice([150, 700, 2300, 3100])
            s1 = single_cache(attn, n, rng)
            oA, oB = merged([s1]), merged([s1])
            cA.extend(oA); cB.extend(oB); n_join += 1
            check_state(cA, cB, f"join {it}")
        elif r < 0.06 and B > 2:                                   # a stream finishes: extract, then filter
            b = pyr.randrange(B)
            eA, eB = cA.extract(b), cB.extract(b)
            assert mx.array_equal(eA[0].keys, eB[0].keys).item() and eA[0].offset == eB[0].offset, f"extract {it}"
            keep = mx.array([i for i in range(B) if i != b], dtype=mx.int32)
            cA.filter(keep); cB.filter(keep); n_leave += 1
            check_state(cA, cB, f"leave {it}")
    print(f"PASS seed {seed}: {n_cmp} steps bit-identical (paths {paths}), rejection steps {n_rej}, cancels {n_cancel}, "
          f"one-token steps {n_one}, joins {n_join}, leaves {n_leave}; final lengths {cA[0].offset.tolist()}; "
          f"{ragkv_x.stats_line()}; kernels {calls}; {time.time() - t0:.1f} s", flush=True)


def bench(lens, reps=10, layers=6):
    """`layers` sparse layers' key caches for streams of the given lengths (right-aligned, shorter ones left-padded),
    latent 512 bf16, each with its own buffer. One timed graph per rep (one eval, so the host round trip is paid
    once, as in a server step): the first stream's draft is rejected in every layer (stock: trim + roll; ragged:
    bookkeeping), then the next step's mask and two-row write. Also the same graph without the rejection (clean
    step) to isolate what the rejection costs."""
    ragkv_x.install(topk=2048)
    L = max(lens); B = len(lens)
    cap = ((L + 2 + 255) // 256) * 256

    class _Pool:
        _processed = [L] * B

    def fresh():
        out = []
        for _ in range(layers):
            kv = VC.BatchKVCache([L - n for n in lens])
            kv.keys = (mx.random.normal((B, 1, cap, 512)) * 0.5).astype(mx.bfloat16)
            kv.values = mx.zeros((B, 1, cap, 0), dtype=mx.bfloat16)
            kv._idx = L; kv.offset = mx.array(lens)
            out.append(kv)
        mx.eval([(kv.keys, kv.values, kv.offset) for kv in out])
        return out

    new = (mx.random.normal((B, 1, 2, 512)) * 0.5).astype(mx.bfloat16)
    nv = mx.zeros((B, 1, 2, 0), dtype=mx.bfloat16)
    mx.eval(new, nv)
    trims = [1] + [0] * (B - 1)
    res = {}
    for name in ("clean", "stock", "ragged"):
        tt = []
        for r in range(reps + 1):
            kvs = fresh()
            mx.synchronize()
            t0 = time.perf_counter()
            outs = []
            for kv in kvs:
                if name == "stock":
                    kv.prepare(right_padding=trims); kv.finalize()
                elif name == "ragged":
                    ragkv_x.rag_trim_rows(kv, _Pool(), trims)
                m = kv.make_mask(2, return_array=True)
                k, _ = kv.update_and_fetch(new, nv)
                outs += [m, k, kv.offset]
            mx.eval(outs)
            t1 = time.perf_counter()
            if r: tt.append(1e3 * (t1 - t0))
            del kvs, outs
        tt.sort()
        res[name] = tt[len(tt) // 2]
        print(f"{name:6s}: lengths {lens}, {layers} layers: {res[name]:.3f} ms median ({tt[0]:.3f}-{tt[-1]:.3f})", flush=True)
    per_s = (res["stock"] - res["clean"]) / layers
    per_r = (res["ragged"] - res["clean"]) / layers
    print(f"rejection cost per layer: stock {per_s:.3f} ms, ragged {per_r:.3f} ms; per rejecting step (12 sparse layers) "
          f"stock {12 * per_s:.2f} ms, ragged {12 * per_r:.2f} ms", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        test(int(sys.argv[2]) if len(sys.argv) > 2 else 1, int(sys.argv[3]) if len(sys.argv) > 3 else 500)
    else:
        bench([int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
