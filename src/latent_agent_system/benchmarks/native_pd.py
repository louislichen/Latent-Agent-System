"""Benchmark native vLLM colocated and split text serving."""

import argparse
import asyncio
import json
from pathlib import Path
import random
import shutil

from latent_agent_system.benchmarks.native_workload import replay, run_serial
from latent_agent_system.reproducibility import snapshot_sources
from latent_agent_system.serving.topology import plan_placement
from latent_agent_system.serving.vllm_client import completion
from latent_agent_system.serving.vllm_metrics import capture_after, transfer_counters
from latent_agent_system.serving.vllm_runtime import VllmConfig, check_gpu_processes, serving


def text_sender(placement, output_tokens, *, include_case=False):
    async def send(client, prompt, index):
        row = await completion(client, prompt, output_tokens, placement.route(index))
        return {**row, 'index': index, 'input_tokens': len(prompt)} if include_case else row

    return send


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu', required=True, nargs='+', help='Physical GPU UUIDs')
    parser.add_argument('--buffer', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--prefillers', type=int, nargs='+', default=[1])
    parser.add_argument('--port-base', type=int, default=18520)
    parser.add_argument('--requests', type=int, default=40)
    parser.add_argument('--rates', type=float, nargs='+', default=[0.5, 1, 2, 4])
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--chunk-tokens', type=int, default=2048)
    parser.add_argument('--prefill-chunk-tokens', type=int,
                        help='Dedicated prefiller token budget. Colocated and D keep --chunk-tokens.')
    parser.add_argument('--max-seqs', type=int, default=32)
    parser.add_argument('--memory', type=float, default=0.83)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if (len(set(args.gpu)) != len(args.gpu) or len(args.gpu) < 2 or
            any(not 0 < n < len(args.gpu) for n in args.prefillers) or
            any(r <= 0 for r in args.rates) or args.requests <= 1 or args.output_tokens < 2):
        parser.error('Use distinct GPUs, a valid P/D partition, positive rates and multiple requests.')
    if not (args.model / 'weights_verified.json').is_file():
        parser.error('The experiment requires the verified Qwen3-4B checkpoint.')
    try:
        placements = {
            n: plan_placement(args.gpu, n, args.port_base) for n in (0, *args.prefillers)
        }
    except ValueError as exc:
        parser.error(str(exc))
    config = VllmConfig(
        model=args.model, max_model_len=args.prompt_tokens + args.output_tokens + 128,
        chunk_tokens=args.chunk_tokens, prefill_chunk_tokens=args.prefill_chunk_tokens,
        max_seqs=args.max_seqs, memory=args.memory, buffer=args.buffer, eager=args.eager,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {
        f'source/{name}': digest
        for name, digest in snapshot_sources(args.out / 'source').items()
    }
    shutil.copy2(args.model / 'weights_verified.json', args.out / 'weights_verified.json')
    from importlib.metadata import version
    settings = {
        **vars(args), 'model': str(args.model.resolve()), 'out': str(args.out.resolve()),
        'torch': version('torch'), 'vllm': version('vllm'), 'nixl': version('nixl'),
        'source_sha256': hashes, 'seed': 73,
        'ttft_slo_s': 2.0, 'tpot_slo_s': 0.03, 'attainment_target': 0.9,
    }
    (args.out / 'config.json').write_text(json.dumps(settings, indent=2))
    rng = random.Random(settings['seed'])
    trace = [[rng.randrange(500, 14000) for _ in range(args.prompt_tokens)] for _ in range(args.requests)]
    offsets = [0.0]
    for _ in range(len(trace) - 1):
        offsets.append(offsets[-1] + rng.expovariate(1.0))
    probe = [trace[0][:512], trace[1][:4096], trace[2 % len(trace)]]
    first_gate = None
    with (args.out / 'runs.jsonl').open('x') as file:
        for num_prefill in (0, *args.prefillers):
            placement = placements[num_prefill]
            name = f'{num_prefill}p{len(args.gpu) - num_prefill}d' if num_prefill else 'colocated'
            logdir = args.out / name
            with serving(config, placement, logdir) as deployment:
                urls = deployment.placement.urls
                print(f'{name} ready', flush=True)
                before = transfer_counters(urls, logdir, 'start')
                gate = asyncio.run(run_serial(probe, text_sender(placement, 32), connections=4))
                actual = [case['token_ids'] for case in gate]
                if first_gate is None:
                    first_gate = actual
                stats = capture_after(urls, logdir, 'gate', before, len(gate) if num_prefill else 0)
                (logdir / 'gate.json').write_text(json.dumps({
                    'exact_token_match': actual == first_gate, 'cases': gate, 'transfer': stats,
                }, indent=2))
                print(f'{name} short greedy gate exact={actual == first_gate}; NIXL {stats}', flush=True)
                if actual != first_gate:
                    raise RuntimeError(f'{name} changed serial greedy output.')
                send = text_sender(placement, args.output_tokens, include_case=True)
                if not args.smoke:
                    warmup = trace[:8]
                    before = transfer_counters(urls, logdir, 'warmup_before')
                    asyncio.run(replay(warmup, [0.0] * len(warmup), 1.0, send))
                    capture_after(urls, logdir, 'warmup_after', before, len(warmup) if num_prefill else 0)
                for rate in (() if args.smoke else args.rates):
                    deployment.check_health()
                    before = transfer_counters(urls, logdir, f'{rate}_before')
                    raw = asyncio.run(replay(trace, offsets, rate, send))
                    stats = capture_after(urls, logdir, f'{rate}_after', before, len(raw) if num_prefill else 0)
                    deployment.check_health()
                    file.write(json.dumps({'layout': name, 'rate': rate, 'cases': raw, 'transfer': stats}) + '\n')
                    file.flush()
                    print(f'{name} rate={rate} completed={len(raw)}', flush=True)
    check_gpu_processes(args.gpu)
    (args.out / 'complete.json').write_text(json.dumps({'smoke': args.smoke, 'layouts': 1 + len(args.prefillers)}))


if __name__ == '__main__':
    main()
