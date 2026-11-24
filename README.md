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
| Agent turn at 195k context (560 new tokens in, about 20 out) | 0.92 s | 0.73 s |
| Client edits a message halfway through a 173k conversation | about 75 s (full re-read) | 31 s |
| Prompt cache held after 18 turns of a 172k conversation | about 35 GB | 8 GB |
| Request with one 2,145-token image (24 tokens out) | 1.63 s | 1.34 s |
| Request with three maximum-size images (23,826 tokens) | out of memory, then a hung server | 14 s |
| First request after a start | 15 s | 0.4 s (the one-time cost is paid at start-up) |

Vision costs 1.05 GiB per Mac and nothing during decoding.

**Requests that send a `seed`** (replay of real non-streaming JSON-mode jobs, temperature 0.25 and 0.8, about
2.7k tokens in and 370 out, or one image in and 100–150 out; output tokens/s over the whole run, prompt time included)

| Workers | Stock path | `TP_SEEDED=1` (default) | `TP_SEEDED=2` |
|---|---|---|---|
| 1 | 73 | 95 | 95 |
| 2 | 73 | 95 | 108 |
| 4 | 76 | 97 | 129 |
| Median time of the 2.7k-token job, one worker | 5.2 s | 3.9 s | 3.9 s |
| Two image workers and one text worker, all seeded (`bench/mixreal.py`) | 71 | 85 | 99 |
| Same seed and prompt give the same output | yes | yes, alone and under load | alone yes; under overlap only about 1 output in 10 |
| Short seeded job arriving 3 s into a 94k-token prompt | 39 s | 2.5 s | not measured |
| Streaming request (about 12k-token prompt) next to two seeded job workers: first token, median / worst (`bench/agentmix.py`) | 12.5 / 15.3 s | 9.1 / 9.2 s | 5.7 / 7.5 s |
| Same test: seeded jobs finished in 70 s, median time | 11, 15.5 s | 12, 14.3 s | 14, 10.3 s |

The stock server takes any request with a `seed` out of the batch path: it runs alone, one token per pass,
without the draft head, and nothing else is admitted until it ends. Clients that send a seed with every
request (many do by default) never saw the numbers above this table.

## Known problems

- **A long prompt and active streams share the machine about half and half.** The prompt loads at roughly half
  speed while streams decode at 35–45% of theirs (`TP_FAIR_MS`, `TP_FAIR_CHUNK`).
- **Prompts are still read one at a time.** A short one jumps ahead of a long one (`TP_PREEMPT`), two long ones queue.
- **Reading 130 or more tokens costs a fixed 200 ms** (one pass over all expert weights) plus 0.3 ms per token;
  below that, about 2 ms per token. Every turn and every image request pays it once. Not fixable in software here.
- **A prompt that contains an image is read outside the scheduler.** Everything up to the last image is read
  before the request enters it. Reads of up to 40k tokens (`TP_VIS_ASYNC_MIN`) happen in one go and nothing else
  runs meanwhile: an image swap on a 200k-token conversation holds other requests for 7–8 s. Longer reads take
  turns with everything else (`TP_VIS_ASYNC`, default on): one 2,048-token block between rounds of decode steps
  while streams are active, two full blocks in the gap after each seeded job. A first read of a 205k-token
  conversation takes 78 s alone, as before, and 130–150 s next to continuous streams or a steady flow of seeded
  jobs; in exchange the other requests keep moving (new streams start in 2–4 s instead of waiting up to 80 s,
  the longest job wait fell from about 84 s to 17 s). Job throughput during the read is unchanged.
- **Streams that decode together share padded caches, so a long conversation decodes almost alone.** Every
  stream in a batch gets an attention cache as long as the longest context in it. One 207k-token conversation
  decoding next to eight short requests took active memory from 96 to 212 GiB and hung both ranks; this
  happened five times in one evening of real traffic before it was understood. A request now joins the batch
  only while streams × longest context stays within `TP_BATCH_SLOTS` (250,000 tokens); otherwise it waits in
  line, in arrival order. With that limit the same load peaks at 121 GiB. In the traffic this was built for,
  96 of 14,590 requests overlapped the generation of a conversation that long, so few requests ever wait.
- **A client that keeps only its last few images** removes the oldest image message each time it adds one,
  10–16k tokens before the end. Every saved end-of-prompt state then contains a removed image, so the server
  read the whole conversation again on each such turn (57 s at 148k tokens). `TP_IMG_PIN` (default on) saves the
  state at the start of the oldest image's message while passing it: that turn now takes 5–7 s alone and 5–16 s
  under mixed load, and a job arriving during it waits about 7 s instead of the whole read. Costs one extra
  cache entry per long conversation. Other mid-history edits still re-read from the last saved point before them.
- **The server port serves one model name, `default_model`.** Any other name is refused with an error (the
  stock server would try to load it, which took both ranks down). The gateway maps model names.
- **Two prompts cannot be read as one batch** (`TP_PROMPT_CONC` must stay 1): the padded batch path calls
  `finalize()` on a cache slot this model never fills, and padding through the linear-attention state is unverified.
- **Seeded requests do not share decode steps by default** (`TP_SEEDED=1`), so several seeded jobs at once are
  no faster in total than one after another. `TP_SEEDED=2` lets them share (129 instead of 97 tok/s at four
  workers) but then the output for a seed is not exactly repeatable under overlap: logits are 16-bit, they round
  differently with the batch shape, and near-ties fall the other way. Greedy requests show the same effect, so
  `temperature=0` output also depends on what else is running. Greedy requests with a seed keep the stock path.
  What mode 1 costs in waiting, from one day of real traffic where every job carries a seed: a job that arrives
  while one other request is in flight takes 1.4–2.4 s longer than one that arrives alone, 3.4–5.3 s longer with
  two or more in flight (a quarter of all job seconds), and a stream's first token comes after 4.5 s instead of
  1.0 s (medians). Two defects that only showed in mode 2 are fixed: the `TP_BATCH_SLOTS` memory rule was
  enforced by the mode-1 hand-over line and was silently off in mode 2, and a `stop` string could end the
  server (see "Staying up").
- **Long context costs a flat 7 ms per pass at 195k tokens** (12 ms per pass at short context), whatever the number of rows.
- **No prompt reuse when two prompts diverge mid-message.** The linear-attention state cannot be rewound, so
  reuse only happens at a saved state: the end of a system or user message of an earlier request, or a pin.
  How a client's history changes between turns therefore decides its waiting time. In 1,336 streamed turns of
  real agent traffic, 89% only appended. 38 replaced the last one to three messages (a chat client that adds
  two short messages after each answer and takes them out again on the next turn): each of those re-read the
  whole conversation, 8–13 s at 22–34k tokens and up to 185 s at 194k, because cache thinning kept only the
  newest saved state and that one lay after the change. The two newest are kept now (`TP_CACHE_KEEP_LAST`, see
  "What makes it fast" 15) and such a turn starts in 0.3–0.5 s. 59 changed earlier history. Of those, large
  rewrites (12–60k new tokens in a 70–196k prompt) already start within about 6 s of what reading the new part
  alone would take; small sub-conversations that branch off a shared 11–20k history still re-read all of it
  (6–13 s each, and eight at once queue behind each other). Saving a state where a new prompt leaves a stored
  one would fix that; not built.
- **A restart loses every saved state.** The next turn of each long conversation reads it again from the start
  (100 s at 194k tokens, longer if two such reads overlap). Nothing is written to disk.
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
   per step, while streams × longest context is at most 200,000 tokens (`TP_MSPEC_SLOTS`): past that, taking a
   rejected draft row out of the padded caches costs more than the second token gains (6% faster at 2 × 70k tokens,
   3–6% slower at 2 × 117k and 2 × 207k).
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

13. **Tokenizing only what is new.** The stock server tokenizes the whole rendered prompt on every turn and the
    system prompt twice more (125 ms at 195k tokens). A role marker is a special token, so the tokens before one
    do not depend on what follows: the common prefix with the previous turn's text is cut at the last role marker,
    its tokens are reused and only the rest is encoded (4 ms). `bench/tokmemotest.py` checks it against a full
    encode on randomised conversations; `TP_TOKMEMO_CHECK=1` checks every live request.
14. **The lookup index is kept between turns** and its first build is one numpy pass (44 ms per turn before).
15. **Cache entries are chosen, not just aged.** Earlier checkpoints of the same conversation are thinned to one per
    12% of its length, answer-end entries that can no longer match are dropped, and system-prompt checkpoints are
    evicted last (`TP_CACHE_THIN`, `TP_CACHE_SPACING`). The two newest checkpoints of a conversation always stay
    (`TP_CACHE_KEEP_LAST`): a client that takes its last messages out again diverges just before the newest one.
    The extra copy, and checkpoints on a branch the conversation has left, are the first to go when the cache is
    over its byte limit, so they never cost another conversation its place. `bench/cachesim.py` runs the policy
    on stand-in caches without a model; `bench/rewindconv.py` measures the turn on a live server.
16. **Image prompts in blocks of up to 4,096 tokens** instead of 1,024, because of the fixed cost per block.
17. **Seeded requests on the fast path.** Every token on the fast paths is already
    `argmax(logits / T + noise(seed, output position))` with a per-request seed, so the client's seed replaces
    the random one and a seeded request gets drafts like any other. To keep its output repeatable it decodes
    alone: it is admitted and its prompt is read like any other request's, and the rule applies where a request
    moves from prompt reading to generation. It waits there while the generation batch is not empty; while it
    generates, other requests wait there too and no prompt blocks are read (reading another prompt between its
    decode steps was enough to change its output). The line at that hand-over is in arrival order, and its head
    also waits for an earlier request whose short prompt is still unread; without that, requests arriving with a
    cached prompt kept passing one that needed reading. A long prompt gets one block after every decode-alone
    generation; without that a steady flow of seeded jobs kept a streaming request's prompt unread (first token
    after 16 s median, 34 s worst, in the test above). `bench/seedtest.py` checks repeats alone and under mixed load,
    `bench/seedecho.py` checks that no request gets another's answer, `bench/seedwait.py` measures the wait behind
    a long prompt.

## Staying up

- **Vision memory is bounded.** The tower's attention is global over an image's patches; at the processor's
  maximum (about 32k patches) the stock call builds a 32 GB score tensor per layer, and several images were
  evaluated together. Queries now go through attention in blocks of 4,096 rows (same result) and images are
  evaluated one at a time (`TP_VIS_QCHUNK`).
- **Both Macs agree on a failure.** The tower runs separately on each Mac. If it fails on one, both raise and the
  request gets an error; before, the other Mac went on and waited for a sum that never came.
- **The server exits instead of hanging.** If the generation thread dies, or its loop has not turned for 150 s
  (`TP_WATCHDOG_S`), the process exits; run the launcher under something that restarts it. Health checks alone
  did not catch this: `/health` kept answering while nothing was generated.
- **The launcher stops rank 0 when rank 1's session ends**, so recovery does not wait for the watchdog. Measured
  by killing rank 1 under a supervisor that restarts the launcher on exit: answering again after 39 s (183 s with
  the watchdog alone).
- **Warm-up at start** (`TP_WARMUP`): the server sends itself one request as soon as its port answers, then two
  at once and one with a small image, so those kernels are compiled before real traffic.
- **A start that hangs is caught too.** The loop watchdog only arms on the loop's first turn; a separate deadline
  (`TP_WATCHDOG_START_S`, 900 s) covers the link to the other rank and loading.
- **Memory is logged** (`TP_MEMLOG=<seconds>`): active, pooled and peak MLX memory on each rank.
- **Connection bursts** are queued (`TP_ACCEPT_QUEUE`, 128) instead of reset by the stock accept queue of 5.
- If you put a queueing proxy in front of the server, do not estimate prompt size from the request body: an
  inline image is a few hundred to a few thousand tokens, not its base64 length.
- **A `stop` string cannot end the server any more.** With two-token steps a request can return two responses
  from one step. The stock loop ends a request on the first response whose text completes a stop string,
  forgets it, and then looks it up again for the second response: `KeyError`, the generation thread dies and
  the server exits. Second responses are now handed out on the next call and dropped if their request is gone
  (`TP_ONE_RESP`, default on). `bench/stoptest.py` sends stop-string requests next to another stream: the
  second-response case came up in 16 of 36 of them.
- **The memory rule does not depend on the seeded mode.** The line that enforces `TP_BATCH_SLOTS` runs whenever
  that limit is set.
- **Pauses are logged.** A generation loop that does not turn for more than 8 s prints `[tp] loop pause N s`
  (`TP_WATCHDOG_GAPLOG`); the longest ones seen in normal operation say how far `TP_WATCHDOG_S` can come down.
  `TP_TRACE=2` prints a few timing lines per request (tokenizing, cache lookup and insert, image tower, prompt
  read) without the per-step lines of `TP_TRACE=1`.
- **A proxy in front should hold requests while the server restarts.** Ours restarted the server five seconds
  after an exit but kept sending requests to the closed port meanwhile: 26 jobs failed with 502 in one
  35-second restart. `gateway/hold-and-retry.diff` (against our aiohttp gateway, not part of this repo) makes it
  hold new requests until the server answers again and send a request once more when its connection failed
  before any answer. Against a stand-in upstream that is killed mid-run (`gateway/fake_upstream.py`,
  `gateway/client.py`): 9 of 10 requests fail before, none after.

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
- Different shares between a loading prompt and active streams (blocks of 1,024 to 4,096 tokens, 250 or 450 ms
  of decoding in between): within 1% on a replay of image requests next to 2.7k-token text requests.
- Scoring the long-context index with a plain bf16 matrix product: 1.7 ms per pass faster at 195k tokens, but at
  lower precision than the current kernel. Not used.
- Speculation with more than four concurrent requests: extra rows cost memory bandwidth that is already used up.
- Joining the two halves of the vocabulary logits with a per-rank argmax and a two-number gather instead of
  gathering the full rows (`bench/gatherbench.py`): 0.47–0.67 ms against 0.39–0.61 ms for 1–8 rows. The cost is
  the round trip, not the megabyte of payload.
- The two-row vocabulary-head kernel for three and four rows: prose 140 instead of 142 tok/s, two streams 161
  instead of 169.
- Letting the hyper-connection post kernel run without its ordering dependency: 140 against 142 tok/s.
- Capping MLX's buffer pool: the pool stayed under 35 GiB through every failure; the memory that ran out was live
  (see the padded batch caches under Known problems).

## Layout

| Path | Contents |
|---|---|
| `server/` | `tpserve.py` (the server) and the kernel modules it loads |
| `scripts/` | `run-cluster.sh` (launcher), `pyx` (oMLX's Python with the patched MLX first on the path) |
| `mlx/` | The MLX patch and how to build it |
| `bench/` | Benchmarks and correctness checks |
| `experiments/` | Negative results kept for reference |
| `gateway/` | A patch for a proxy in front of the server (hold and retry while the server restarts) and its stand-in test |

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
   python bench/turnbench.py     # agent-shaped turns on a long cached conversation (CTX, NEW, GROW, EDIT)
   python bench/mixreal.py       # image workers next to text workers (sends a seed per request; SEED=0 for none)
   python bench/seedtest.py      # same seed and prompt repeat exactly, alone and under mixed load
   python bench/seedecho.py      # seeded and unseeded requests at once: nobody gets another's answer
   python bench/seedwait.py      # a short seeded job arriving while a long prompt is read
   python bench/agentmix.py      # a streaming request every few seconds next to seeded job workers
   python bench/seedreplay.py    # replay saved request bodies (SET=file.jsonl, W workers, TEXTOUT to compare runs with cmptext.py)
   python bench/imgsize.py       # image requests of growing size and count
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
