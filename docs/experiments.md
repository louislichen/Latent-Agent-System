# Reproducing the serving experiments

Use Linux with NVIDIA CUDA GPUs and a local Qwen3-4B BF16 checkpoint. Install
the benchmark package in each backend environment from the repository root.

    python -m pip install -e '.[analysis,native]'

The native extra provides the HTTP client. Install the CUDA backend and its
matching PyTorch separately. The original full-chain runs used PyTorch
2.5.1+cu121 and Transformers 4.57.3. The native runs used vLLM 0.31.0, NIXL
1.4.1 and a CUDA-compatible PyTorch build. The analysis scripts use NumPy and
Matplotlib. Set LATENT_PLOT_FONT to a local CJK font file if needed for
Chinese chart labels. Use free GPUs and distinct output directories for each run.

Set MODEL_DIR to the checkpoint, RUN_ROOT to a writable data directory, and
GSM8K_TEST to the GSM8K main test.jsonl. From the repository root, run these
commands to fetch the upstream method and prepare the sample list.

    mkdir -p "$RUN_ROOT/sources"
    git clone https://github.com/Gen-Verse/LatentMAS.git "$RUN_ROOT/sources/LatentMAS"
    git -C "$RUN_ROOT/sources/LatentMAS" checkout --detach 9a9e4d331eb11430bd9e64754c6b252b06d73031
    latent-agent prepare-samples "$GSM8K_TEST" "$RUN_ROOT/samples.json"

## Baselines

Set GPU_P, GPU_D and GPU_D2 to physical GPU UUIDs from nvidia-smi -L.
For the two-GPU full-chain HF comparison, run:

    CUDA_VISIBLE_DEVICES="$GPU_P,$GPU_D" \
      latent-agent bench-hf-chain --root "$RUN_ROOT" --model "$MODEL_DIR" \
      --out "$RUN_ROOT/results/hf-chain" --gpu-uuid "$GPU_P" --decode-uuid "$GPU_D"
    latent-agent analyze-hf-chain "$RUN_ROOT/results/hf-chain"

For the three-GPU native Judge comparison, capture upstream inputs in the HF
environment and serve them in the vLLM/NIXL environment:

    CUDA_VISIBLE_DEVICES="$GPU_P" \
      latent-agent capture-latent-judge --source "$RUN_ROOT/sources/LatentMAS" \
      --model "$MODEL_DIR" --samples "$RUN_ROOT/samples.json" --out "$RUN_ROOT/judge-inputs"
    VLLM_USE_FLASHINFER_SAMPLER=0 \
      latent-agent bench-latent-judge --model "$MODEL_DIR" \
      --inputs "$RUN_ROOT/judge-inputs" --out "$RUN_ROOT/results/native-latent" \
      --gpu "$GPU_P" "$GPU_D" "$GPU_D2"
    latent-agent analyze-latent-judge "$RUN_ROOT/results/native-latent"

The native Judge benchmark starts after the three upstream latent rollouts.
The HF full-chain benchmark includes those rollouts and checks token sequences
and answer scores across placements. Its colocated control uses one active
GPU with a second model replica resident, so the run measures serial chain
latency. Use the online serving drivers for concurrent workloads. Some
profilers read existing run directories via LATENT_SERVING_ROOT and use
LATENT_SERVING_MODEL to load the same checkpoint.

For the native synthetic text baseline, use latent-agent bench-native-text with the GPU
UUIDs and --buffer cpu or --buffer cuda according to the interconnect. The script
expects a weights_verified.json marker created by latent-agent stage-checkpoint
from the pinned Qwen3-4B shards. Use --help for the remaining runtime options.

## Replaying archived results

New runs snapshot the Python package with relative source paths and SHA-256
hashes. Data-center workers run from that snapshot. The analysis commands
also accept the earlier flat source manifests. Existing file commands under
experiments/latent_serving delegate to the package and accept the same flags.
