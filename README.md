# GLM-Tandem

GLM-5.3-Flash on two Mac Studios, pedalling together: about 140 tokens/s on fresh prose.

A recipe for serving one GLM-5.3-Flash request across two Mac Studios (M5 Ultra, 256 GB each) joined by
Thunderbolt 5 RDMA, split tensor-parallel. It runs on top of oMLX's cluster loader and stock `mlx_lm.server`,
with an OpenAI-style API.

## Outcomes

Model: uniform 4-bit (group 64) GLM-5.3-Flash with its MTP draft head, 169 GiB, half on each Mac.
All rates are decode tokens/s as seen by a streaming client. "Prose" means fresh essays with little repetition.

**One request**

| Workload | tok/s |
|---|---|
| Fresh prose, temperature 0.6 | 141–146 |
| Fresh prose, temperature 0 | 141 |
| Chat fixture (256 tokens), T=0 / T=1 | 143 / 140 |
| Code fixture (256 tokens), T=0 / T=1 | 157 / 146 |
| JSON extraction | 148 |
| Add docstrings, whole file back | 210 |
| Code edit, whole file back | 219 |
| Request with an image (decode) | 142–146 |

**Several requests at once** (total tokens/s including prompt time, 300-token prose, short prompts)

| Requests | Total | Each |
|---|---|---|
| 2 | 164–169 | 93–99 |
| 4 | 200 | 54–61 |
| 8 | 223–230 | 30–34 |

**One request at long context** (from 146 real agent requests)

| Context | Decode (median) | First token on a follow-up turn (median) |
|---|---|---|
| 20–50k tokens | 137 | 0.9 s |
| 50–80k | 126 | 0.7 s |
| 80–200k | 122 | 0.9 s |

Each pass costs about 1.2 ms more at 31k tokens and 2.1 ms more at 108k than at short context: every sparse
layer scores all pooled keys of the context (compute-bound). Splitting that across the Macs would save about 0.2 ms.

**Prompts and scheduling**

| | Before | Now |
|---|---|---|
| Cold prompt, 108k tokens (tokens/s) | 2,000 | 2,440 |
| Next turn of a 31k conversation after 40 unrelated short requests | 12.5 s | 0.25 s |
| New conversation with the same 31k system prompt | 12.5 s | 0.20 s |
| New 13k prompt while a 108k conversation is cached | 17–20 s | 5.8 s |
| A stream's rate while another request's 31k prompt loads | 4 tok/s, 3–4 s pauses | 40–46 tok/s, 0.5 s pauses |
| Short request arriving behind a 108k prompt: first token | after the whole prompt (45 s+) | 1.3–4 s |
| Small image request (559 tokens in, 150 out), one at a time | 1.62 s | 1.50 s |

Vision costs 1.05 GiB per Mac and nothing during decoding.

## Known problems

- **A long prompt and active streams share the machine about half and half.** The prompt loads at roughly half
  speed while streams decode at 35–45% of theirs (`TP_FAIR_MS`, `TP_FAIR_CHUNK`).
- **Prompts are still read one at a time.** A short one jumps ahead of a long one (`TP_PREEMPT`), two long ones queue.
- **The first prompt after a start is about 15 s slower** (kernels compile on first use).
- **No prompt reuse when two prompts diverge mid-message.** The linear-attention state cannot be rewound, so
  reuse only happens at the end of a system or user message.
- **Tied to oMLX 0.8.0rc1.** The server patches oMLX's GLM runtime at load time by rewriting its source, with
  assertions on every anchor. A different oMLX version will most likely fail those assertions at start-up.
- The fresh-prose rate is limited by the draft head: about 77–79% of its drafts are accepted, and that matches
  its measured ceiling.

## What makes it fast

1. **Speculative decoding with the model's own MTP head.** Each pass verifies the next token and one draft.
   Sampling is coupled (shared Gumbel noise), so sampled output keeps its distribution.
2. **The next pass is queued before the host knows the outcome.** The accept decision is broadcast early; the
   next inputs, the kept draft-head row, and the linear-attention state are all selected inside the graph, and
   the first layers of the next pass are already running when the host reads the result.
3. **Verify rows cost little.** Row-general fused kernels for the hyper-connection chain (`hcrows.py`), the
   expert layers with in-kernel routing (`moerows.py`), and a two-row 4-bit kernel for the vocabulary head
   (`qmvrows.py`).
4. **A cheaper wait between CPU and GPU.** A small MLX patch makes the CPU thread spin on the fence inline
   (`MLX_FENCE_MODE=1`). With about 90 cross-machine sums per pass, this matters.
5. **A confidence gate.** The second row is dropped when the draft's probability is low.
6. **The draft head is split across both Macs** like a trunk layer, so neither waits while the other drafts.
7. **Concurrent requests are rows of one fused pass**, not a batch (`kdaseq.py` gives the linear-attention
   kernel a sequence axis). With 2–4 requests each one also carries its own draft and emits one or two tokens
   per step.
8. **Drafts copied from the context.** When the last few tokens also appear earlier in the context, the tokens
   that followed them there are offered as a draft, up to 7 at a time (`TP_LOOKUP`, on by default). Every
   drafted token is still verified, so the output does not change. This is where the 210–219 tok/s on code
   edits and docstrings comes from; fresh prose gets almost nothing from it.
9. **Images** go through the same fast path: the vision tower runs once at prompt time (`tpvision.py`), and the
   image and the text after it are read in one pass.
10. **A prompt cache that keeps long conversations.** Stock mlx_lm keeps 10 entries and drops the oldest, so a
    burst of short requests evicts a 100k-token conversation. Here short entries have their own small tier and
    long ones are limited by bytes (`TP_CACHE_GB`, default 40), least recently used first. Two stock defects are
    patched as well: the cache lookup's search for a longer entry is quadratic in its length (11–12 s on the
    generation thread next to a cached 100k conversation), and with this chat template the system-prompt
    checkpoint was usually not saved.
11. **Prompts and decoding take turns.** While streams are decoding, a long prompt is read in 1,024-token blocks
    with about 450 ms of decode steps in between; the server looks at its request queue after every block
    (stock: after a fixed number of steps, which during a prompt means after the whole prompt); short prompts
    jump ahead of a long one.
12. **Three Thunderbolt links and 4,096-token prompt blocks.** Decode speed is the same as on one link; prompt
    processing is 13% faster with three links and 4–8% faster with larger blocks. Two links gave +11%.

## What did not help

All measured on this setup; the code for the first two is in `experiments/`.

- Fusing sparse attention into one kernel (`sarows.py`): correct, slower. One threadgroup runs on about one GPU core.
- A tensor-unit kernel for the expert layers (`tuqmv.py`): slower.
- Skipping the fence's coherence kernel: faster, but corrupts output.
- A merged fence kernel; reading each distinct expert once per pass; leaving sparse attention unsplit;
  raising thread priority; skipping index upkeep: no gain.
- Moving the host loop out of Python: estimated at 1–3%, not built.
- The "mixed context lengths are slow" problem listed here earlier was a misdiagnosis: 13k and 108k requests decode
  together at 66–86 tok/s each. The slow case was a cache eviction followed by a full re-read that starved the other stream.
- Prompt blocks of 16,384 tokens: no faster than 8,192.
- Speculation with more than four concurrent requests: extra rows cost memory bandwidth that is already used up.

## Layout

| Path | Contents |
|---|---|
| `server/` | `tpserve.py` (the server) and the kernel modules it loads |
| `scripts/` | `run-cluster.sh` (launcher), `pyx` (oMLX's Python with the patched MLX first on the path) |
| `mlx/` | The MLX patch and how to build it |
| `bench/` | Benchmarks and correctness checks |
| `experiments/` | Negative results kept for reference |

## Running it

You need two Apple-silicon Macs with RDMA over Thunderbolt working in oMLX's cluster mode, oMLX 0.8.0rc1
installed on both, the checkpoint at the same path on both, and this repository at the same path on both.

1. Build the patched MLX (`mlx/README.md`). Without it, remove `MLX_FENCE_MODE=1` from `run-cluster.sh`.
2. On the first Mac:

   ```sh
   PEER=user@<second-mac-thunderbolt-address> COORD_ADDR=<first-mac-thunderbolt-address> \
   IBV_DEVICES=/path/to/devices.json MODEL_PATH=/path/to/GLM-5.3-Flash-4bit-MTP \
   scripts/run-cluster.sh
   ```

3. Check it, from the first Mac (use oMLX's bundled Python; `vistest.py` needs Pillow):

   ```sh
   python bench/validate.py      # known-answer checks: greedy, sampled, stop sequences
   python bench/stress.py        # cancellations and overlapping requests
   python bench/conctest.py      # staggered concurrent known-answer tasks
   python bench/vistest.py       # image requests
   python bench/bench_prose.py   # fresh-prose rate
   python bench/concbench.py     # 1/2/4/8 concurrent requests
   python bench/longbench.py     # long contexts
   ```

   The benchmarks default to `localhost:8010` or `localhost:8080`; set `URL` (or `BASE`) and `MODEL`.

`run-cluster.sh` here is a generalised copy of the launcher used for the measurements (hosts and paths turned
into variables). The generalised copy itself has not been run.

Useful switches (environment variables read by `tpserve.py`): `TP_CONC` (requests decoded together),
`TP_MSPEC=0` (no speculation with several requests), `TP_SHARD_HEAD=0` (draft head on the first Mac only),
`TP_CONF_GATE` (confidence threshold, default 0.9), `TP_VISION=0`.

## Built on

[MLX](https://github.com/ml-explore/mlx) and mlx-lm, oMLX (cluster loader, GLM runtime and decode kernels),
and TensorFold's uniform 4-bit GLM build. The kernels here are derived at load time from oMLX's kernel sources;
none of oMLX's code is copied into this repository.
