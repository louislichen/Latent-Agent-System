"""Serve and benchmark native vLLM P/D on a single node with explicit GPU placement."""

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4

import httpx


MODEL_NAME = 'qwen3-pd-control'
COMPLETIONS = '/v1/completions'


def gpu_processes(gpus):
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        text=True,
    ).splitlines()
    return {int(pid) for gpu, pid in (r.split(',') for r in rows)
            if gpu.strip() in gpus}


def check_gpu_processes(gpus, servers=()):
    own_groups = {server.pid for server in servers}
    for pid in gpu_processes(gpus):
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group not in own_groups:
            raise RuntimeError(f'A selected GPU is in use by process {pid}.')


def connector_config(role, buffer, enable_embeds=False):
    config = {
        'kv_connector': 'NixlConnector', 'kv_role': role, 'kv_buffer_device': buffer,
        'kv_load_failure_policy': 'fail',
        'kv_connector_extra_config': {'kv_lease_duration': 120},
    }
    if enable_embeds:
        config.update(kv_connector='LatentNixlPullConnector', kv_connector_module_path='latent_nixl_v031')
    return json.dumps(config)


@contextmanager
def serving(args, roles, logs):
    servers = []
    urls = []
    logs.mkdir(parents=True, exist_ok=False)
    try:
        check_gpu_processes(args.gpu)
        for index, role in enumerate(roles):
            port = args.port_base + index
            endpoint = f'http://127.0.0.1:{port}'
            chunk_tokens = args.chunk_tokens
            if role == 'kv_producer' and args.prefill_chunk_tokens is not None:
                chunk_tokens = args.prefill_chunk_tokens
            env = {
                **os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu[index],
                'VLLM_NIXL_SIDE_CHANNEL_HOST': '127.0.0.1',
                'VLLM_NIXL_SIDE_CHANNEL_PORT': str(args.port_base + 16 + index),
                'UCX_TLS': 'sm,self,tcp' if args.buffer == 'cpu' else 'sm,self,tcp,cuda_copy,cuda_ipc',
            }
            command = [
                sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                '--model', str(args.model), '--served-model-name', MODEL_NAME,
                '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                '--max-model-len', str(args.prompt_tokens + args.output_tokens + 128),
                '--gpu-memory-utilization', str(args.memory),
                '--max-num-batched-tokens', str(chunk_tokens), '--max-num-seqs', str(args.max_seqs),
                '--generation-config', 'vllm', '--enable-chunked-prefill', '--no-enable-prefix-caching',
            ]
            if args.eager:
                command.append('--enforce-eager')
            enable_embeds = getattr(args, 'enable_embeds', False)
            if enable_embeds:
                command.append('--enable-prompt-embeds')
                env['PYTHONPATH'] = str(Path(__file__).resolve().parent) + os.pathsep + env.get('PYTHONPATH', '')
                env['VLLM_KV_CACHE_LAYOUT'] = 'HND'
            if role != 'colocated':
                command.extend(['--kv-transfer-config', connector_config(role, args.buffer, enable_embeds)])
            (logs / f'worker{index}_command.json').write_text(json.dumps({
                'argv': command, 'gpu': args.gpu[index], 'role': role,
                'side_channel_port': env['VLLM_NIXL_SIDE_CHANNEL_PORT'], 'UCX_TLS': env['UCX_TLS'],
            }, indent=2))
            with (logs / f'worker{index}.log').open('x') as log:
                servers.append(subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                ))
            urls.append(endpoint)
        deadline = time.monotonic() + 360
        with httpx.Client(timeout=2.0, trust_env=False) as client:
            pending = set(range(len(servers)))
            while pending:
                for index in list(pending):
                    if servers[index].poll() is not None:
                        raise RuntimeError(f'Worker {index} exited. See {logs}/worker{index}.log.')
                    try:
                        response = client.get(urls[index] + '/v1/models')
                        if response.is_success and any(x['id'] == MODEL_NAME for x in response.json()['data']):
                            pending.remove(index)
                    except httpx.RequestError:
                        pass
                if pending and time.monotonic() >= deadline:
                    raise TimeoutError(f'Workers {pending} did not start. See {logs}.')
                if pending:
                    time.sleep(1)
        check_gpu_processes(args.gpu, servers)
        yield urls, servers
    finally:
        for server in servers:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for server in servers:
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=5)


def transfer_counters(urls, out, label):
    keys = ('nixl_xfer_time_seconds_count', 'nixl_xfer_time_seconds_sum',
            'nixl_bytes_transferred_sum', 'nixl_num_failed_transfers_total',
            'nixl_num_kv_expired_reqs_total', 'nixl_num_failed_notifications_total')
    totals = {}
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        for index, url in enumerate(urls):
            response = client.get(url + '/metrics')
            response.raise_for_status()
            (out / f'{label}_worker{index}.prom').write_text(response.text)
            for row in response.text.splitlines():
                if not row.startswith('vllm:nixl_'):
                    continue
                metric, value = row.split()[:2]
                key = metric.split('{')[0].removeprefix('vllm:')
                if key in keys:
                    totals[key] = totals.get(key, 0.0) + float(value)
    return totals


async def completion(client, prompt, output_tokens, decode, prefill=None, *,
                     prompt_embeds=None, continuation=None, prompt_positions=None):
    start = time.perf_counter()
    headers = {'X-Request-Id': uuid4().hex}
    payload = {
        'model': MODEL_NAME, 'max_tokens': output_tokens, 'temperature': 0,
        'ignore_eos': True, 'return_token_ids': True, 'stream': True,
    }
    if prompt_embeds is None:
        payload['prompt'] = prompt
    else:
        payload['prompt_embeds'] = prompt_embeds
    p_s = None
    blocks = None
    ids, stamps = [], []
    if prefill is not None:
        prefill_payload = {
            **payload, 'max_tokens': 1, 'stream': False,
            'kv_transfer_params': {'do_remote_decode': True, 'do_remote_prefill': False},
        }
        p_response = await client.post(prefill + COMPLETIONS, json=prefill_payload, headers=headers)
        if not p_response.is_success:
            raise RuntimeError(f'Prefill HTTP {p_response.status_code}: {p_response.text[:500]}')
        produced = p_response.json()
        metadata = produced.get('kv_transfer_params')
        if not metadata or not metadata.get('do_remote_prefill') or not metadata.get('remote_block_ids'):
            raise ValueError(f'Prefill failed to publish usable transfer metadata: {metadata}')
        block_ids = metadata['remote_block_ids']
        blocks = sum(len(group) for group in block_ids) if isinstance(block_ids[0], list) else len(block_ids)
        payload['kv_transfer_params'] = metadata
        p_s = time.perf_counter() - start
        if continuation is not None:
            assert metadata.get('remote_num_tokens') == prompt_positions
            first = produced['choices'][0]['token_ids']
            assert len(first) == 1
            ids.extend(first)
            stamps.append(p_s)
            payload.update(continuation(first[0]))
            payload['max_tokens'] = output_tokens - 1

    done = False
    async with client.stream('POST', decode + COMPLETIONS, json=payload, headers=headers) as response:
        if not response.is_success:
            raise RuntimeError(f'Serving HTTP {response.status_code}: {(await response.aread())[:500]!r}')
        async for line in response.aiter_lines():
            if not line.startswith('data: '):
                continue
            if line == 'data: [DONE]':
                done = True
                break
            arrived = time.perf_counter() - start
            objects = json.loads(line[6:])
            if 'error' in objects:
                raise RuntimeError(objects['error'])
            for choice in objects.get('choices', []):
                tokens = choice.get('token_ids') or []
                ids.extend(tokens)
                stamps.extend([arrived] * len(tokens))
    if not done or len(ids) != output_tokens:
        raise ValueError(f'Expected {output_tokens} streamed token IDs, got {len(ids)} (done={done}).')
    return {'token_ids': ids, 'token_arrivals_s': stamps, 'start': start,
            'wall_s': time.perf_counter() - start, 'prefill_response_s': p_s, 'kv_blocks': blocks}


def client_for(requests):
    return httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=5.0),
        limits=httpx.Limits(max_connections=2 * requests, max_keepalive_connections=requests),
        trust_env=False,
    )


async def calibrate(urls, num_prefill, trace):
    prefill = urls[:num_prefill]
    decode = urls[num_prefill:]
    async with client_for(4) as client:
        results = []
        for i, prompt in enumerate(trace):
            p = prefill[i % len(prefill)] if prefill else None
            d = decode[i % len(decode)]
            results.append(await completion(client, prompt, 32, d, p))
        return results


async def replay(rate, prompts, offsets, urls, num_prefill, output_tokens):
    prefill = urls[:num_prefill]
    decode = urls[num_prefill:]
    async with client_for(len(prompts)) as client:
        origin = time.perf_counter() + 0.05

        async def run(index, prompt, offset):
            deadline = origin + offset / rate
            await asyncio.sleep(max(0, deadline - time.perf_counter()))
            p = prefill[index % len(prefill)] if prefill else None
            d = decode[index % len(decode)]
            row = await completion(client, prompt, output_tokens, d, p)
            return {**row, 'index': index, 'scheduled': deadline, 'input_tokens': len(prompt)}

        return await asyncio.gather(*(run(i, p, o) for i, (p, o) in enumerate(zip(prompts, offsets))))


def read_delta(before, after):
    return {key: value - before.get(key, 0.0) for key, value in after.items()}


def check_transfer(stats, minimum):
    if (stats.get('nixl_xfer_time_seconds_count', 0) < minimum
            or stats.get('nixl_bytes_transferred_sum', 0) <= 0
            or any(value for key, value in stats.items() if 'failed' in key or 'expired' in key)):
        raise RuntimeError(f'NIXL did not deliver the expected KV without errors: {stats}')


def capture_after(urls, out, label, baseline, count):
    deadline = time.monotonic() + 18.0
    while True:
        delta = read_delta(baseline, transfer_counters(urls, out, label))
        if not count or delta.get('nixl_xfer_time_seconds_count', 0) >= count:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(1)
    if count:
        check_transfer(delta, count)
    return delta


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
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / Path(__file__).name
    shutil.copy2(__file__, source)
    shutil.copy2(args.model / 'weights_verified.json', args.out / 'weights_verified.json')
    from importlib.metadata import version
    settings = {
        **vars(args), 'model': str(args.model.resolve()), 'out': str(args.out.resolve()),
        'torch': version('torch'), 'vllm': version('vllm'), 'nixl': version('nixl'),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'seed': 73,
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
            roles = (['kv_producer'] * num_prefill + ['kv_consumer'] * (len(args.gpu) - num_prefill)
                     if num_prefill else ['colocated'] * len(args.gpu))
            name = f'{num_prefill}p{len(args.gpu) - num_prefill}d' if num_prefill else 'colocated'
            logdir = args.out / name
            with serving(args, roles, logdir) as (urls, servers):
                print(f'{name} ready', flush=True)
                before = transfer_counters(urls, logdir, 'start')
                gate = asyncio.run(calibrate(urls, num_prefill, probe))
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
                if not args.smoke:
                    warmup = trace[:8]
                    before = transfer_counters(urls, logdir, 'warmup_before')
                    asyncio.run(replay(1.0, warmup, [0.0] * len(warmup), urls,
                                       num_prefill, args.output_tokens))
                    capture_after(urls, logdir, 'warmup_after', before, len(warmup) if num_prefill else 0)
                for rate in (() if args.smoke else args.rates):
                    check_gpu_processes(args.gpu, servers)
                    before = transfer_counters(urls, logdir, f'{rate}_before')
                    raw = asyncio.run(replay(rate, trace, offsets, urls, num_prefill, args.output_tokens))
                    stats = capture_after(urls, logdir, f'{rate}_after', before, len(raw) if num_prefill else 0)
                    check_gpu_processes(args.gpu, servers)
                    file.write(json.dumps({'layout': name, 'rate': rate, 'cases': raw, 'transfer': stats}) + '\n')
                    file.flush()
                    print(f'{name} rate={rate} completed={len(raw)}', flush=True)
    check_gpu_processes(args.gpu)
    (args.out / 'complete.json').write_text(json.dumps({'smoke': args.smoke, 'layouts': 1 + len(args.prefillers)}))


if __name__ == '__main__':
    main()
