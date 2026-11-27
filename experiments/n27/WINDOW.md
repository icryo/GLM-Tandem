# Next window: candidate `tpserve_n27_x.py` (built Nov 26 evening, not live)

Staged on both Macs in `~/llm-stack/cluster/tp/` next to the live files (live `tpserve.py` untouched):
`tpserve_n27_x.py` (md5 a824a199...), `gumbelrows_x.py`, `ragkv_x.py`, `ragtest_x.py`.

`tpserve_n27_x.py` = live `tpserve.py` (n26, d0fa074d) plus:

| Switch | Default | What | Offline result |
|---|---|---|---|
| `TP_GUMBEL_ROWS` | 1 | sampling noise of a position from one dispatch instead of ~9 (`gumbelrows_x.py`) | bit-identical over 64 positions x 154,880; 18.7 -> 4.1 us per position (~0.25% of a cycle) |
| `TP_RAGGED` | 0 | ragged key-cache ends: a rejected draft row in a shared two-token step is forgotten per stream instead of rolling every stream's bank (`ragkv_x.py`) | attention output + logical keys + pools bit-identical over 6 seeds x 2,400 steps (fused, gathered and dense paths, joins/leaves/cancel/extract); rejecting step 5.9 -> ~0.5 ms at 70k+2k, 10.4 -> ~1.1 ms at 120k+2k, 5.9 -> ~1.2 ms at 70k+70k |

## Order (each from a fresh start, as in window 7)

1. Reference: live build (`n26.sh` / `w7install.sh tpserve_n26.py ...`), `n26set.py` + `eqvis.py`, pairs
   70k+short and 120k+short (`n26perf.sh`), concbench 1/2/4/8, seeded concurrent outputs saved.
2. `tpserve_n27_x.py` with `TP_GUMBEL_ROWS=1 TP_RAGGED=0`: all hashes must equal the reference
   (sampling noise is bit-identical); expect single-stream and job decode +0-0.5%.
3. `TP_RAGGED=1`: seeded concurrent requests byte-identical to step 2; `[tp-mspec]` "after a
   rejection" ms should fall to the clean-step figure; expect 70k+short ~148-154 -> ~160-167 tok/s.
4. `TP_RAGGED=1 TP_MSPEC_SLOTS=0` (two-token shared steps at any context): expect 120k+short
   133.5 -> ~145-150.
5. Full suite (`n26check.sh suite`: validate, stress, vis, conc x2 + trio, stop 48/48 plain + seeded
   with >= 2 streams, seedtest, preempt + cancel, cachetest, rewindconv, editconv 72k with jobs),
   hang hunt (joins/leaves under ragged state), 15-min soak; keep only what is faster and passes.

Revert at any point: `/tmp/n26.sh old` (or reinstall `tpserve_n26.py` with the window-7 ENVS).

## Not in this candidate

- MLX 0.32.5: oMLX's `sparse_index` extension must be rebuilt against 0.32.5 first; the build
  command was refused by the permission check in this session (see the report). With it, a separate
  A/B in the window.
- Tile-fill prompt kernel: see `experiments/tilefill/` (offline results).
- Scale index: exact but not faster (see `experiments/scaleidx/`).
- In-stream exchange: see `experiments/encx/DESIGN.md`.
