# Latent Agent System

Research code for serving latent multi-agent workflows. The current benchmarks
compare colocated and prefill/decode (P/D) deployments across full agent chains
and individual serving stages.

- [System architecture](docs/architecture.md)
- [Experiment setup and commands](docs/experiments.md)
- [Stage results and figures](reports/2026-11-22/README.md)

## Project layout

| Path | Responsibility |
| --- | --- |
| src/latent_agent_system/serving | Placement, worker lifecycle, request protocol, and KV transport |
| src/latent_agent_system/workflows | Upstream multi-agent workflow adapters |
| src/latent_agent_system/benchmarks | Workloads, layout comparisons, and launchers |
| src/latent_agent_system/profiling | Agent, GPU, and transport measurements |
| src/latent_agent_system/analysis | Offline aggregation and plots |
| src/latent_agent_system/data | Sample preparation and answer parsing |
| experiments/latent_serving | Compatibility entry points for existing experiment commands |
| tests | CPU checks for placement, request protocol, and worker cleanup |
| reports | Curated measurements and figures tied to dated runs |

## Quick start

Install the package with analysis dependencies and list the commands.

    python -m pip install -e '.[analysis]'
    latent-agent --help

The GPU experiments require Linux, CUDA, and the backend versions documented
in the experiment guide. The full-chain HF and native Judge experiments measure
different parts of the workflow. Their workload and correctness checks are
documented with the results.
