"""Command-line entry points for benchmarks, profiling and analysis."""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "prepare-samples": ("data.prepare_samples", "Freeze GSM8K inputs"),
    "stage-checkpoint": ("tools.stage_qwen3_checkpoint", "Verify the Qwen3-4B model shards"),
    "bench-hf-chain": ("benchmarks.benchmark_pd_matrix", "Compare HF agent chains across layouts"),
    "bench-hf-reference": ("serving.hf_semantic_pd", "Check HF Judge output equivalence"),
    "bench-native-text": ("benchmarks.native_pd", "Compare native text serving layouts"),
    "capture-latent-judge": ("benchmarks.capture_native_pd_latent", "Freeze latent Judge embeddings"),
    "bench-latent-judge": ("benchmarks.benchmark_native_pd_latent", "Compare native latent Judge layouts"),
    "bench-dual-text": ("benchmarks.benchmark_dual_gpu_text_serving", "Run two-GPU concurrent text serving"),
    "bench-dc": ("benchmarks.run_dc_workloads", "Run the data-center workload matrix"),
    "bench-dc-worker": ("benchmarks.benchmark_dc_workloads", "Run one data-center workload cell"),
    "profile-hf": ("profiling.profile_hf_existing", "Profile the HF correctness reference"),
    "profile-agent-routes": ("profiling.profile_agent_routes", "Profile agent placement and handoffs"),
    "probe-kv-copy": ("profiling.probe_kv_copy_dual_gpu", "Measure cross-GPU KV copies"),
    "probe-prefill-interference": ("profiling.probe_text_pd_interference", "Measure decode stalls from prefill"),
    "analyze-hf-chain": ("analysis.analyze_pd_matrix", "Summarize paired HF agent runs"),
    "analyze-native-text": ("analysis.analyze_native_pd", "Summarize native text serving"),
    "analyze-latent-judge": ("analysis.analyze_native_pd_latent", "Summarize native latent Judge runs"),
    "analyze-dual-text": ("analysis.analyze_dual_gpu_text_serving", "Summarize two-GPU text serving"),
    "analyze-dc": ("analysis.analyze_dc_workloads", "Summarize data-center workloads"),
    "analyze-prefill-interference": ("analysis.analyze_text_pd_interference", "Summarize prefill interference"),
    "analyze-profiles": ("analysis.analyze_profiles", "Summarize agent stage and kernel profiles"),
    "plot-profile-breakdown": ("analysis.plot_profile_breakdown", "Plot stage-level latency breakdowns"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="\n".join(f"  {name:<29} {description}" for name, (_, description) in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="ARGS")
    args = parser.parse_args()
    module, _ = COMMANDS[args.command]
    original = sys.argv
    try:
        sys.argv = [f"{parser.prog} {args.command}", *args.arguments]
        import_module(f"latent_agent_system.{module}").main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
