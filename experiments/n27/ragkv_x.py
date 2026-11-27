# Ragged right ends for the batch key caches of the sparse-attention layers (TP_RAGGED=1, experimental).
#
# A two-token multi-stream step (tpserve _ms_*) that rejects some streams' drafts must take those streams' last
# row out of the batch key cache. The stock way, BatchKVCache.prepare(right_padding) + finalize(), rolls the WHOLE
# bank of every stream one place to the right (dynamic_roll = a gather over the entire buffer): ~0.4 ms per sparse
# layer at 2 x 70k tokens, 12 layers per step that has a rejection, growing with context.
#
# Here the rejected row is only forgotten. The cache keeps a per-stream physical end `_tp_end` (host ints, <= the
# batch extent `_idx`, which evolves exactly as in the stock cache), writes each stream's next rows at its own end,
# and builds each stream's mask from its own end. The sparse layers address keys as left_padding + logical position
# (selected pool windows, the partial-window tail and the block's own rows: sparse_mla_decode_batched, the general
# indexer + gathered attention), so with ragged ends they read the same values in the same order as after a roll:
# outputs are bit-identical. The dense path (no selection while every stream is below index_topk tokens) reads the
# whole bank under a mask, where the physical layout sets the summation order; there, and before any structural
# operation (filter, extend, state, prepare/finalize), the pending shifts are applied in ONE roll ("flush"), which
# puts every valid key exactly where the stock sequence of rolls would have put it.
#
# Only instances that went through rag_trim_rows() carry `_tp_end`; every other cache behaves exactly as before.
import sys
import mlx.core as mx

STATS = {"trim_rows": 0, "ragged": 0, "flush": 0, "flush_dense": 0, "writes": 0, "masks": 0, "extract": 0}
_TOPK = [2048]
_DONE = set()
_ROLL = {}
_TRIMS = {}


def _dense(self, n):
    """True when the next forward of n rows will take the dense (whole-bank, masked) attention path, or when that
    cannot be proven otherwise (then a flush is merely a copy, always correct)."""
    pool = getattr(self, "_tp_pool", None)
    proc = getattr(pool, "_processed", None)
    if not isinstance(proc, list) or len(proc) != len(self._tp_end):
        return True
    return max(proc) + n <= _TOPK[0]


def flush(self, why="flush"):
    """Apply the pending per-stream shifts in one roll: the stock layout (rows right-aligned at _idx)."""
    end = getattr(self, "_tp_end", None)
    if end is None:
        return
    self._tp_end = None
    rp = [self._idx - e for e in end]
    if max(rp) > 0 and self.keys is not None:
        pad = mx.array(rp)
        roll = _ROLL[type(self)]
        self.keys = roll(self.keys, pad[:, None], axis=2)
        self.values = roll(self.values, pad[:, None], axis=2)
        self.left_padding = self.left_padding + pad
    STATS[why] += 1


def rag_trim_rows(kv, pool, trims):
    """Drop the last trims[b] rows of stream b from a batch key cache without moving any key. Same offsets and the
    same extent (_idx) as kv.trim(min) + prepare(right_padding) + finalize; the pool is trimmed by the caller."""
    STATS["trim_rows"] += 1
    u = min(trims)
    if kv.keys is None:                     # a side cache never written: the stock calls are free and exact here
        if u: kv.trim(u)
        extra = [x - u for x in trims]
        if any(extra):
            kv.prepare(right_padding=extra); kv.finalize()
        return
    end = kv._tp_end if getattr(kv, "_tp_end", None) is not None else [kv._idx] * len(trims)
    if len(end) != len(trims):
        flush(kv); end = [kv._idx] * len(trims)
    t = tuple(trims)
    ta = _TRIMS.get(t)
    if ta is None:
        ta = _TRIMS[t] = mx.array(list(t), dtype=kv.offset.dtype)
        if len(_TRIMS) > 64: _TRIMS.clear()
    kv._idx -= u
    kv.offset = kv.offset - ta
    end = [e - x for e, x in zip(end, trims)]
    kv._tp_end = None if all(e == kv._idx for e in end) else end
    kv._tp_pool = pool
    if kv._tp_end is not None:
        STATS["ragged"] += 1


def _ragged_mask(self, n, window_size=None):
    STATS["masks"] += 1
    K = self._idx + n
    rinds = mx.arange(K)[None, None, None, :]
    q = (mx.array(self._tp_end, dtype=mx.int32)[:, None] + mx.arange(n, dtype=mx.int32)[None])[:, None, :, None]
    mask = q >= rinds
    if window_size is not None:
        mask = mask & (q < rinds + window_size)
    return mask & (self.left_padding[:, None, None, None] <= rinds)


def _ragged_write(self, keys, values):
    STATS["writes"] += 1
    prev = self._idx
    N = keys.shape[2]
    if self.keys is None or (prev + N) > self.keys.shape[2]:      # growth exactly as the stock method
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        n_steps = (self.step + N - 1) // self.step
        new_k = mx.zeros((B, n_kv_heads, n_steps * self.step, k_head_dim), keys.dtype)
        new_v = mx.zeros((B, n_kv_heads, n_steps * self.step, v_head_dim), values.dtype)
        if self.keys is not None:
            if prev % self.step != 0:
                self.keys = self.keys[..., :prev, :]
                self.values = self.values[..., :prev, :]
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
        else:
            self.keys, self.values = new_k, new_v
    end = self._tp_end
    B = len(end)
    i = 0
    while i < B:                                                  # one write per run of streams with the same end
        j = i
        while j + 1 < B and end[j + 1] == end[i]:
            j += 1
        e = end[i]
        self.keys[i:j + 1, :, e:e + N, :] = keys[i:j + 1]
        if values.shape[-1]:
            self.values[i:j + 1, :, e:e + N, :] = values[i:j + 1]
        i = j + 1
    self.offset = self.offset + N
    self._idx = prev + N
    self._tp_end = [e + N for e in end]
    return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]


def _ragged_extract(self, idx):
    STATS["extract"] += 1
    mod = sys.modules[type(self).__module__]
    cache = mod.KVCache()
    padding = self.left_padding[idx].item()
    e = self._tp_end[idx]
    cache.keys = mx.contiguous(self.keys[idx: idx + 1, :, padding:e])
    cache.values = mx.contiguous(self.values[idx: idx + 1, :, padding:e])
    cache.offset = cache.keys.shape[2]
    return cache


def _wrap(cls):
    d = cls.__dict__
    o_mask, o_uaf, o_trim, o_extract = d["make_mask"], d["update_and_fetch"], d["trim"], d["extract"]
    o_filter, o_extend, o_prepare, o_finalize = d["filter"], d["extend"], d["prepare"], d["finalize"]
    o_state = d["state"]

    def make_mask(self, N, return_array=False, **kwargs):
        if getattr(self, "_tp_end", None) is not None:
            if _dense(self, N):
                flush(self, "flush_dense")
            else:
                return _ragged_mask(self, N, kwargs.get("window_size"))
        return o_mask(self, N, return_array=return_array, **kwargs)

    def update_and_fetch(self, keys, values):
        if getattr(self, "_tp_end", None) is not None:
            if _dense(self, keys.shape[2]):
                flush(self, "flush_dense")
            else:
                return _ragged_write(self, keys, values)
        return o_uaf(self, keys, values)

    def trim(self, n):
        end = getattr(self, "_tp_end", None)
        if end is None:
            return o_trim(self, n)
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        self._tp_end = [e - n for e in end]
        return n

    def extract(self, idx):
        if getattr(self, "_tp_end", None) is not None:
            return _ragged_extract(self, idx)
        return o_extract(self, idx)

    def filter(self, batch_indices):
        flush(self)
        return o_filter(self, batch_indices)

    def extend(self, other):
        flush(self)
        if getattr(other, "_tp_end", None) is not None:
            flush(other)
        return o_extend(self, other)

    def prepare(self, **kw):
        flush(self)
        return o_prepare(self, **kw)

    def finalize(self):
        flush(self)
        return o_finalize(self)

    def _state_get(self):
        flush(self)
        return o_state.fget(self)

    def _state_set(self, v):
        self._tp_end = None
        o_state.fset(self, v)

    for name, fn, orig in (("make_mask", make_mask, o_mask), ("update_and_fetch", update_and_fetch, o_uaf),
                           ("trim", trim, o_trim), ("extract", extract, o_extract), ("filter", filter, o_filter),
                           ("extend", extend, o_extend), ("prepare", prepare, o_prepare), ("finalize", finalize, o_finalize)):
        fn.__wrapped__ = orig
        fn.__qualname__ = cls.__name__ + "." + name + " [ragged]"
        setattr(cls, name, fn)
    cls.state = property(_state_get, _state_set)
    cls._tp_end = None
    cls._tp_pool = None


def install(topk=2048, modules=("mlx_vlm.models.cache", "mlx_lm.models.cache")):
    """Wrap BatchKVCache of the given cache modules (idempotent). Returns the list of wrapped classes."""
    import importlib
    _TOPK[0] = int(topk)
    out = []
    for m in modules:
        try:
            mod = importlib.import_module(m)
        except ImportError:
            continue
        cls = getattr(mod, "BatchKVCache", None)
        if cls is None or cls in _DONE:
            continue
        need = ("make_mask", "update_and_fetch", "trim", "extract", "filter", "extend", "prepare", "finalize", "state")
        if not all(k in cls.__dict__ for k in need) or not hasattr(mod, "dynamic_roll") or not hasattr(mod, "KVCache"):
            continue
        _ROLL[cls] = mod.dynamic_roll
        _wrap(cls)
        _DONE.add(cls)
        out.append(m + ".BatchKVCache")
    return out


def stats_line():
    s = STATS
    return (f"ragged trims {s['ragged']}/{s['trim_rows']} flushes {s['flush']} (+{s['flush_dense']} dense) "
            f"writes {s['writes']} masks {s['masks']} extracts {s['extract']}")
