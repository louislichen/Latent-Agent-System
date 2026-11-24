# Latent Agent System

[Stage results and figures](reports/2026-11-22/README.md)

Benchmarks for colocated and prefill/decode (P/D) serving of multi-agent latent
workflows. The repository contains the existing comparison harnesses and
profilers. The proposed placement policy across agent roles is a research design.

| Workload | Entry point in experiments/latent_serving |
| --- | --- |
| Full-chain HF LatentMAS and TextMAS with paired output checks | benchmark_pd_matrix.py |
| HF LatentMAS reference and P/D Judge correctness | hf_semantic_pd.py |
| Native vLLM/NIXL text serving | native_pd.py |
| Native vLLM/NIXL Judge with LatentMAS embeddings and a length-matched token control | capture_native_pd_latent.py and benchmark_native_pd_latent.py |
| Prefill interference and concurrent text serving | probe_text_pd_interference.py and benchmark_dual_gpu_text_serving.py |
| Agent stages and KV transfer profiling | profile_agent_routes.py, profile_hf_existing.py and probe_kv_copy_dual_gpu.py |

## Inputs

Use Linux with NVIDIA CUDA GPUs and a local Qwen3-4B BF16 checkpoint. Install
the packages for the chosen backend. The original full-chain runs used PyTorch
2.5.1+cu121 and Transformers 4.57.3. The native runs used vLLM 0.31.0, NIXL
1.4.1 and a CUDA-compatible PyTorch build. The analysis scripts use NumPy and
Matplotlib. Set LATENT_PLOT_FONT to a local CJK font file if needed for\nChinese chart labels. Use free GPUs and distinct output directories for each run.

Set MODEL_DIR to the checkpoint, RUN_ROOT to a writable data directory, and
GSM8K_TEST to the GSM8K main test.jsonl. From the repository root, run these
commands to fetch the upstream method and prepare the sample list.

    CODE="$PWD/experiments/latent_serving"
    mkdir -p "$RUN_ROOT/sources"
    git clone https://github.com/Gen-Verse/LatentMAS.git "$RUN_ROOT/sources/LatentMAS"
    git -C "$RUN_ROOT/sources/LatentMAS" checkout --detach 9a9e4d331eb11430bd9e64754c6b252b06d73031
    python "$CODE/prepare_samples.py" "$GSM8K_TEST" "$RUN_ROOT/samples.json"

## Baselines

Set GPU_P, GPU_D and GPU_D2 to physical GPU UUIDs from nvidia-smi -L.
For the two-GPU full-chain HF comparison, run:

    CUDA_VISIBLE_DEVICES="$GPU_P,$GPU_D" \
      python "$CODE/benchmark_pd_matrix.py" --root "$RUN_ROOT" --model "$MODEL_DIR" \
      --out "$RUN_ROOT/results/hf-chain" --gpu-uuid "$GPU_P" --decode-uuid "$GPU_D"
    python "$CODE/analyze_pd_matrix.py" "$RUN_ROOT/results/hf-chain"

For the three-GPU native Judge comparison, capture upstream inputs in the HF
environment and serve them in the vLLM/NIXL environment:

    CUDA_VISIBLE_DEVICES="$GPU_P" \
      python "$CODE/capture_native_pd_latent.py" --source "$RUN_ROOT/sources/LatentMAS" \
      --model "$MODEL_DIR" --samples "$RUN_ROOT/samples.json" --out "$RUN_ROOT/judge-inputs"
    VLLM_USE_FLASHINFER_SAMPLER=0 \
      python "$CODE/benchmark_native_pd_latent.py" --model "$MODEL_DIR" \
      --inputs "$RUN_ROOT/judge-inputs" --out "$RUN_ROOT/results/native-latent" \
      --gpu "$GPU_P" "$GPU_D" "$GPU_D2"
    python "$CODE/analyze_native_pd_latent.py" "$RUN_ROOT/results/native-latent"

The native Judge benchmark starts after the three upstream latent rollouts.
The HF full-chain benchmark includes those rollouts and checks token sequences
and answer scores across placements. Its colocated control uses one active
GPU with a second model replica resident, so the run measures serial chain
latency. Use the online serving drivers for concurrent workloads. Some
profilers read existing run directories via LATENT_SERVING_ROOT and use
LATENT_SERVING_MODEL to load the same checkpoint.

For the native synthetic text baseline, use native_pd.py with the GPU UUIDs
and --buffer cpu or --buffer cuda according to the interconnect. The script
expects a weights_verified.json marker created by stage_qwen3_checkpoint.py
from the pinned Qwen3-4B shards. Use --help for the remaining runtime options.
