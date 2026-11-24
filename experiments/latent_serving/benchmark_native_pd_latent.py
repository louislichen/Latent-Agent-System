"""Compare the frozen Judge requests through the same native vLLM layout."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace

from native_pd import (capture_after, check_gpu_processes, client_for, completion,
                       serving, transfer_counters)


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


async def send(client, case, output_tokens, prefill, decode, index, embeddings_file):
    length, ids, embeds, tensor = case

    def continuation(token):
        if tensor is None:
            return {'prompt': ids + [token]}
        import torch
        from safetensors import safe_open

        with safe_open(embeddings_file, framework='pt', device='cpu') as f:
            next_embedding = f.get_slice('model.embed_tokens.weight')[token:token + 1]
        assert next_embedding.shape == (1, tensor.shape[1])
        return {'prompt_embeds': encode_tensor(torch.cat((tensor, next_embedding.to(tensor.dtype)), dim=0))}

    row = await completion(client, ids, output_tokens, decode[index % len(decode)],
                           prefill[index % len(prefill)] if prefill else None, prompt_embeds=embeds,
                           continuation=continuation if prefill else None, prompt_positions=length)
    return {**row, 'index': index, 'input_tokens': length, 'base64_bytes': len(embeds) if embeds else 0}


async def serial(cases, prefill, decode, embeddings_file):
    async with client_for(len(cases)) as client:
        return [await send(client, c, 32, prefill, decode, i, embeddings_file) for i, c in enumerate(cases)]


async def online(cases, offsets, rate, output_tokens, prefill, decode, embeddings_file):
    async with client_for(len(cases)) as client:
        origin = time.perf_counter() + .05

        async def arrival(i):
            scheduled = origin + offsets[i] / rate
            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
            row = await send(client, cases[i], output_tokens, prefill, decode, i, embeddings_file)
            return {**row, 'scheduled': scheduled}

        return await asyncio.gather(*(arrival(i) for i in range(len(cases))))


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
    check_gpu_processes(args.gpu)
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    cases = load_cases(args.inputs / 'judge_embeds.pt', manifest['cases'])
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())
    embeddings_file = args.model / index['weight_map']['model.embed_tokens.weight']
    count = len(manifest['cases'])
    settings = SimpleNamespace(**vars(args), prompt_tokens=max(x['positions'] for x in manifest['cases']),
                               chunk_tokens=2048, prefill_chunk_tokens=None, memory=.83,
                               max_seqs=32, buffer='cpu', enable_embeds=True)
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ('benchmark_native_pd_latent.py', 'native_pd.py', 'latent_nixl_v031.py'):
        path = args.out / name
        shutil.copy2(Path(__file__).with_name(name), path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(args.inputs / 'manifest.json', args.out / 'manifest.json')
    from importlib.metadata import version

    (args.out / 'config.json').write_text(json.dumps({
        'gpus': args.gpu, 'count': count, 'rates': args.rates, 'output_tokens': args.output_tokens,
        'vllm': version('vllm'), 'nixl': version('nixl'), 'torch': version('torch'),
        'source_sha256': hashes, 'ttft_slo_s': 2.0, 'tpot_slo_s': .03,
        'buffer': 'cpu', 'eager': args.eager, 'max_num_batched_tokens': settings.chunk_tokens,
        'split_first_token': 'emit P token, append it to the D request, reuse P KV',
        'stage': 'requests to the final Judge with upstream LatentMAS vLLM inputs frozen off path',
    }, indent=2))
    rng = random.Random(73)
    offsets = [0.0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + rng.expovariate(1.0))
    references = {}
    with (args.out / 'runs.jsonl').open('x') as f:
        for split, layout in ((False, 'colocated'), (True, 'split')):
            roles = (['kv_producer'] + ['kv_consumer'] * (len(args.gpu) - 1)
                     if split else ['colocated'] * len(args.gpu))
            with serving(settings, roles, args.out / layout) as (urls, servers):
                print(layout, 'ready', flush=True)
                prefill, decode = (urls[:1], urls[1:]) if split else ([], urls)
                for kind, inputs in cases.items():
                    metrics = args.out / layout / kind
                    metrics.mkdir()
                    before = transfer_counters(urls, metrics, 'serial_before')
                    gate = asyncio.run(serial(inputs[:8], prefill, decode, embeddings_file))
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
                        check_gpu_processes(args.gpu, servers)
                        before = transfer_counters(urls, metrics, f'{rate}_before')
                        actual = asyncio.run(online(inputs, offsets, rate, args.output_tokens,
                                                    prefill, decode, embeddings_file))
                        delta = capture_after(urls, metrics, f'{rate}_after', before, len(actual) if split else 0)
                        check_gpu_processes(args.gpu, servers)
                        f.write(json.dumps({'layout': layout, 'input_kind': kind, 'rate': rate,
                                            'cases': actual, 'transfer': delta}) + '\n')
                        f.flush()
                        print(layout, kind, rate, len(actual), flush=True)
    check_gpu_processes(args.gpu)
    (args.out / 'complete.json').write_text(json.dumps({'smoke': args.smoke, 'count': count}))


if __name__ == '__main__':
    main()
