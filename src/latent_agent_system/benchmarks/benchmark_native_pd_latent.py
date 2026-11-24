"""Compare the frozen Judge requests through the same native vLLM layout."""

import argparse
import asyncio
import base64
from functools import partial
import hashlib
import io
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


def encode_tensor(tensor):
    import torch

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def load_cases(tensors, manifest):
    import torch

    embeddings = torch.load(tensors, map_location='cpu', weights_only=True)
    rng = random.Random(73)
    groups = {'latent': [], 'matched_tokens': []}
    for x, row in zip(embeddings, manifest, strict=True):
        assert list(x.shape) == [row['positions'], row['hidden_size']] and x.dtype == torch.bfloat16
        assert hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest() == row['sha256']
        groups['latent'].append((row['positions'], None, encode_tensor(x), x))
        groups['matched_tokens'].append(
            (row['positions'], [rng.randrange(500, 14000) for _ in range(row['positions'])], None, None))
    return groups


async def send(client, case, index, *, placement, output_tokens, embeddings_file):
    length, ids, embeds, tensor = case
    route = placement.route(index)

    def continuation(token):
        if tensor is None:
            return {'prompt': ids + [token]}
        import torch
        from safetensors import safe_open

        with safe_open(embeddings_file, framework='pt', device='cpu') as f:
            next_embedding = f.get_slice('model.embed_tokens.weight')[token:token + 1]
        assert next_embedding.shape == (1, tensor.shape[1])
        return {'prompt_embeds': encode_tensor(torch.cat((tensor, next_embedding.to(tensor.dtype)), dim=0))}

    row = await completion(
        client, ids, output_tokens, route, prompt_embeds=embeds,
        continuation=continuation if route.prefill else None, prompt_positions=length,
    )
    return {**row, 'index': index, 'input_tokens': length, 'base64_bytes': len(embeds) if embeds else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu', nargs='+', required=True)
    parser.add_argument('--rates', nargs='+', type=float, default=[1, 2, 4])
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--port-base', type=int, default=18560)
    parser.add_argument('--eager', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpu)) != len(args.gpu) or len(args.gpu) < 2:
        parser.error('Use two or more distinct GPU UUIDs.')
    if args.output_tokens < 2 or any(rate <= 0 for rate in args.rates):
        parser.error('Use positive rates and at least two output tokens.')
    try:
        placements = {
            layout: plan_placement(args.gpu, prefiller_count, args.port_base)
            for layout, prefiller_count in (('colocated', 0), ('split', 1))
        }
    except ValueError as exc:
        parser.error(str(exc))
    check_gpu_processes(args.gpu)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    cases = load_cases(args.inputs / 'judge_embeds.pt', manifest['cases'])
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())
    embeddings_file = args.model / index['weight_map']['model.embed_tokens.weight']
    count = len(manifest['cases'])
    config = VllmConfig(
        model=args.model, max_model_len=max(x['positions'] for x in manifest['cases']) + args.output_tokens + 128,
        eager=args.eager, enable_embeds=True,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {
        f'source/{name}': digest
        for name, digest in snapshot_sources(args.out / 'source').items()
    }
    shutil.copy2(args.inputs / 'manifest.json', args.out / 'manifest.json')
    from importlib.metadata import version

    (args.out / 'config.json').write_text(json.dumps({
        'gpus': args.gpu, 'count': count, 'rates': args.rates, 'output_tokens': args.output_tokens,
        'vllm': version('vllm'), 'nixl': version('nixl'), 'torch': version('torch'),
        'source_sha256': hashes, 'ttft_slo_s': 2.0, 'tpot_slo_s': .03,
        'buffer': 'cpu', 'eager': args.eager, 'max_num_batched_tokens': config.chunk_tokens,
        'split_first_token': 'emit P token, append it to the D request, reuse P KV',
        'stage': 'requests to the final Judge with upstream LatentMAS vLLM inputs frozen off path',
    }, indent=2))
    rng = random.Random(73)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(1.0))
    references = {}
    with (args.out / 'runs.jsonl').open('x') as f:
        for layout, placement in placements.items():
            split = bool(placement.prefiller_count)
            serial_send = partial(send, placement=placement, output_tokens=32, embeddings_file=embeddings_file)
            online_send = partial(send, placement=placement, output_tokens=args.output_tokens, embeddings_file=embeddings_file)
            with serving(config, placement, args.out / layout) as deployment:
                print(layout, 'ready', flush=True)
                urls = deployment.placement.urls
                for kind, inputs in cases.items():
                    metrics = args.out / layout / kind
                    metrics.mkdir()
                    before = transfer_counters(urls, metrics, 'serial_before')
                    probe = inputs[:8]
                    gate = asyncio.run(run_serial(probe, serial_send, connections=len(probe)))
                    delta = capture_after(urls, metrics, 'serial_after', before, len(gate) if split else 0)
                    ids = [c['token_ids'] for c in gate]
                    if not split:
                        references[kind] = ids
                    matched = sum(a == b for a, b in zip(ids, references[kind], strict=True))
                    (metrics / 'serial.json').write_text(json.dumps({
                        'cases': gate, 'exact_match': matched, 'transfer': delta,
                    }, indent=2))
                    print(layout, kind, 'serial', matched, len(gate), delta, flush=True)
                    if matched != len(gate):
                        raise RuntimeError(f'Serial output gate failed for {kind}.')
                    for rate in (() if args.smoke else args.rates):
                        deployment.check_health()
                        before = transfer_counters(urls, metrics, f'{rate}_before')
                        actual = asyncio.run(replay(inputs, offsets, rate, online_send))
                        delta = capture_after(urls, metrics, f'{rate}_after', before, len(actual) if split else 0)
                        deployment.check_health()
                        f.write(json.dumps({'layout': layout, 'input_kind': kind, 'rate': rate,
                                            'cases': actual, 'transfer': delta}) + '\n')
                        f.flush()
                        print(layout, kind, rate, len(actual), flush=True)
    check_gpu_processes(args.gpu)
    (args.out / 'complete.json').write_text(json.dumps({'smoke': args.smoke, 'count': count}))


if __name__ == '__main__':
    main()
