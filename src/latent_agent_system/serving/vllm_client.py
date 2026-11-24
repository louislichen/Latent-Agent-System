"""Request and stream protocol shared by the native serving workloads."""

import json
import time
from uuid import uuid4

import httpx

from latent_agent_system.serving.topology import Route


MODEL_NAME = 'qwen3-pd-control'
COMPLETIONS = '/v1/completions'


async def completion(client, prompt, output_tokens, route: Route, *,
                     prompt_embeds=None, continuation=None, prompt_positions=None):
    if (prompt is None) == (prompt_embeds is None):
        raise ValueError('Send token IDs or prompt embeddings.')
    if output_tokens < 1 or (route.prefill and continuation is not None and output_tokens < 2):
        raise ValueError('The requested output length is too short for this route.')
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
    if route.prefill is not None:
        prefill_payload = {
            **payload, 'max_tokens': 1, 'stream': False,
            'kv_transfer_params': {'do_remote_decode': True, 'do_remote_prefill': False},
        }
        p_response = await client.post(route.prefill + COMPLETIONS, json=prefill_payload, headers=headers)
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
            actual_positions = metadata.get('remote_num_tokens')
            if actual_positions != prompt_positions:
                raise ValueError(f'Prefill published {actual_positions} positions, expected {prompt_positions}.')
            first = produced['choices'][0]['token_ids']
            if len(first) != 1:
                raise ValueError('Prefill must return exactly one continuation token.')
            ids.extend(first)
            stamps.append(p_s)
            payload.update(continuation(first[0]))
            payload['max_tokens'] = output_tokens - 1

    done = False
    async with client.stream('POST', route.decode + COMPLETIONS, json=payload, headers=headers) as response:
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
