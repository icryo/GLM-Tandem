# Exchange inside the GPU command stream (design, Nov 26)

Status: design + one-Mac handshake prototype (`encx_hop.py`, `poller.c`). Not wired into the server.
The network half needs the Thunderbolt RDMA link, which the live service uses, so it can only be
tested in a window.

## Today

Every tensor-parallel exchange (about 90 per decode pass) is `mx.distributed.all_sum`. MLX runs it
on the CPU stream (`JACCLGroup::communication_stream` is the CPU): the GPU command buffer up to the
partial is committed, the patched fence (MLX_FENCE_MODE=1, MLX_METAL_FAST_SYNCH=1) signals the CPU
stream, a CPU task calls `jaccl::Group::all_sum` (UC queue pairs over three TB5 links, mesh), and
the GPU side waits on the fence before the next kernel. Measured earlier: raw 2-rank all_sum
8.2 us at 16 KB (4.9 us at 2 bytes); a GPU -> CPU -> GPU hop after a heavy kernel costs 6-8 us; the
exchanges take ~1.3 ms of a 12.6 ms 2-row cycle, of which ~0.47 ms is the network.

## Proposed

A custom MLX primitive `EncxAllSum` (C++ extension, eval_gpu only) that never leaves the GPU stream:

1. **signal kernel** (one thread): system-coherent store of a per-exchange counter into a
   host-visible word after the partial is complete (same fence/coherence code as MLX's
   `fence_update`).
2. **native poller thread** per rank (user-interactive QoS, created at load): spins on the counter,
   copies the partial into a registered send buffer (or sends it in place if the MLX buffer can be
   registered), posts the RDMA send, polls the receive completion, then stores the "peer arrived"
   counter.
3. **wait + add kernel**: one threadgroup spins (bounded) on the arrived counter, then all
   threadgroups add the peer's partial from the receive window into the output.

The command buffer is not split, the CPU stream is not used, and MLX schedules nothing on the host
between partial and sum. What is left per exchange is the network time plus the poller's two
coherent round trips.

Risks: a missed counter means a spinning kernel, so every spin is bounded and a timeout must turn
into an error, not a hang. Ranks must issue exchanges in the same order (already a requirement:
identical graphs). Registering MLX buffers for RDMA is undocumented, so the safe first version
copies through registered windows (16 KB partials: ~1 us). The extension needs its own queue pairs
(max 10 per device) beside jaccl's.

## Prototype step 1 (one Mac, no network)

`encx_hop.py` chains a heavy 4-bit matmul (12576x4096, 2 rows) with an exchange point and
compares: no exchange; MLX's CPU-stream hop; the in-stream handshake with the native poller
(`libpoller.dylib`, bounded spins). The difference between the last two is what the design saves
per exchange before the network. Results: see the bottom of this file.

## Steps after that (window needed for 3-4)

1. C++ extension skeleton: primitive, signal/wait kernels, poller thread; build against our patched
   MLX 0.32.4 on mac2.
2. Loopback test on one Mac with two processes is not possible (no shared GPU buffers), so test the
   protocol with the poller doing a local memcpy as the "peer".
3. Window: own queue pairs over one TB5 link, exchange-only benchmark between the two Macs
   (sizes 2 bytes to 16 MB), compare with jaccl all_sum.
4. Window: replace the decode-pass all_sums (hcrows/moerows/attention outputs) behind a switch;
   equality must be bit-exact (same sum order for 2 ranks); hang hunt.

## Results (.252, beside live traffic behind an idle gate, MLX 0.32.4 patched, live MLX env)

Per chain step (heavy matmul + exchange point), best of 6 runs of 300 steps:

| | us/step | over no exchange |
|---|---|---|
| no exchange | 55.5 | - |
| MLX CPU-stream hop (today's path, fast fence) | 59.4 | +3.9 |
| in-stream handshake: signal kernel + native poller + bounded spin-wait kernel | 74.7 | +19.3 |
| same with signal and wait in one dispatch, no fence inside the spin | broken: the GPU missed the CPU's ack until the spin bound in 68-87 of 200 steps | - |

An earlier run gave +5.5 / +19.0 us. So on this chip the handshake through system-coherent memory
costs MORE than MLX's hop with our fast fence (which already removes most of the hop), and a spin
without a system-scope fence per iteration does not see host writes reliably. Conclusion: no gain
to expect from moving the exchange into the command stream; the remaining exchange cost is the
network itself (~5 us floor). Parked; steps 1-4 above not started.
