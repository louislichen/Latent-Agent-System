# System architecture

The project serves two evaluation paths. The native path runs colocated or
prefill/decode (P/D) workers on a single node and measures each request at the
HTTP boundary. The HF path invokes LatentMAS and TextMAS through the pinned
upstream implementation and observes the complete agent chain.

## Native serving

| Component | Responsibility |
| --- | --- |
| serving/topology.py | Assign GPUs, worker roles and ports. Route requests to the planned workers. |
| serving/vllm_runtime.py | Launch vLLM workers, wait for readiness, check health and stop owned processes. |
| serving/vllm_client.py | Send requests, carry KV transfer metadata across P/D, and measure streamed token arrivals. |
| serving/vllm_metrics.py | Capture Prometheus counters and validate NIXL transfers. |
| benchmarks/native_workload.py | Replay serial or timed arrivals against the shared client. |

The text and latent Judge benchmarks build a placement, enter the worker
context, and supply their own inputs to the workload runner. Worker roles,
`CUDA_VISIBLE_DEVICES`, and the endpoint used by each request come from the
same placement. Leaving the context tears down the workers it started,
including on a partial startup failure.

For split text requests, P publishes KV metadata and D emits the full measured
output. The latent Judge benchmark counts P's first token as output, appends
that token to D's input, and streams the remaining tokens from D. The client
checks the reported KV positions, total output tokens, and stream termination
at that boundary. A colocated request goes directly to its chosen worker.

## HF agent workflows

The workflow adapter in `workflows/latentmas.py` loads the pinned LatentMAS and
TextMAS methods and prepares the same samples for both full-chain and
data-center runs. `profiling/hf_workflow.py` observes stage transitions,
intermediate state and final outputs. The workload drivers select the layout,
batching and arrival schedule.

## Reproducibility and checks

Benchmark outputs retain source snapshots, workload settings, raw measurements
and exact-output checks. Offline analyzers consume these artifacts. Published
figures and aggregated metrics are in the [dated report](../reports/2026-11-22/README.md).

Run the native serving boundary tests on CPU.

    python -m pip install -e '.[native]'
    python -m unittest discover -s tests -v

The tests exercise placement, request and stream handling, worker launch
configuration, and cleanup after a partial launch. The [experiment guide](experiments.md)
covers the CUDA environments for the end-to-end benchmarks.
