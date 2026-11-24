# Colocated and P/D serving results

These experiments use Qwen3-4B BF16. The WHU RTX 3090 and RTX 4090 pairs did
not expose CUDA peer access. Their cross-device KV transport used CPU staging.
The native runs used vLLM 0.31.0 and NIXL 1.4.1. Each comparison below has
its own workload and measurement scope.

## Full agent chain on two RTX 3090s

Both methods processed the same eight GSM8K questions in two batches of four.
LatentMAS used 40 rollout steps in each of its three intermediate roles. Within
each method, full output sequences, stopping states and grading matched across
placements. Both layouts held two model replicas. The colocated layout ran
the chain on one active replica, so this experiment measures the serial critical
path.

| Method | Colocated total for 8 (s) | P/D total for 8 (s) | P→D / D→P crossings per batch | Synchronized KV copy per batch (s) | Correct / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LatentMAS | 98.31 | 107.39 | 4 / 3 | 2.52 | 6 |
| TextMAS | 310.12 | 324.39 | 4 / 0 | 3.98 | 7 |

The additional D→P transfers in LatentMAS carried the accumulated state to
the next agent. TextMAS had larger prompts and transferred more total KV in
this workload. [Aggregate data](hf-agent-chain.json) · [Figure](hf-agent-chain.png)

## Native LatentMAS Judge on three RTX 4090s

Frozen inputs came from 24 actual LatentMAS Judge prompts of 743–1,071
positions, with upstream rollouts outside the timed region. Each layout ran
the same arrival traces at 1, 2 and 4 requests/s and generated 256 tokens per
request. The length-matched control used token IDs. The split layout used one
P and two D workers, with the first token emitted by P.

| Input at 4 requests/s | TTFT P90 colocated → split (ms) | TPOT P90 colocated → split (ms) | First-to-second interval P90 colocated → split (ms) |
| --- | ---: | ---: | ---: |
| Latent embeddings | 194 → 608 | 11.54 → 14.60 | 11.36 → 532.17 |
| Matched token IDs | 83 → 254 | 11.50 → 12.83 | 10.20 → 152.92 |

Each tested cell reached 24/24 by the TTFT ≤ 2 s and TPOT ≤ 30 ms thresholds.
Each input type passed 8/8 exact serial 32-token comparisons. Concurrent
256-token continuations can differ across layouts. At 4 requests/s the latent
split matched 7/24 full sequences. The timing comparison used a fixed output
length. [Aggregate data](native-latent-judge.json) · [Figure](native-latent-judge.png)

## Native long-text serving on three RTX 4090s

Each cell contained 40 requests with 8,192 input tokens and 256 output tokens.
At 2 requests/s, 1P+2D lowered TPOT P90 while delaying first tokens. The
joint SLO used TTFT ≤ 2 s and TPOT ≤ 30 ms.

| Placement at 2 requests/s | TTFT P90 (s) | TPOT P90 (ms) | SLO passes / 40 |
| --- | ---: | ---: | ---: |
| Colocated | 0.67 | 23.36 | 40 |
| 1P+2D | 10.24 | 19.35 | 2 |
| 2P+1D | 17.39 | 40.04 | 0 |

[Aggregate data](native-text-serving.json) · [Figure](native-text-serving.png)

## Prefill interference on RTX 3090

The P99 interval between tokens of an active decode was 72.0 ms without a
concurrent prefill, 1,433.4 ms with full prefill on the same GPU, 241.9 ms with
512-token chunks, and 85.8 ms with prefill on a separate GPU.
[Aggregate data](prefill-interference.json) · [Figure](prefill-interference.png)

The dual-GPU concurrent text benchmark records burst and spread arrivals,
cross-GPU KV copies, TTFT, ITL and an exact output-equivalence check.
[Aggregate data](dual-gpu-text-serving.json) · [Figure](dual-gpu-text-serving.png)
