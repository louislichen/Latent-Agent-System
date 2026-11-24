"""Compare identical agent chains with colocated and split P/D."""

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from benchmark_dc_workloads import BLOCKS, Observer, load_methods
from hf_semantic_pd import cache_digest


def state_size(cache):
    return 0 if cache is None else sum(
        tensor.numel() * tensor.element_size()
        for layer in cache.to_legacy_cache() for tensor in layer
    )


def compute_pids(uuid):
    query = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']
    rows = subprocess.check_output(query, text=True).splitlines()
    return {int(row.split(',')[1]) for row in rows if row.split(',')[0].strip() == uuid}


def receive(conn):
    if not conn.poll(240):
        raise TimeoutError('Timed out waiting for the P/D peer')
    response = conn.recv()
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response


def relocate_tensor(tensor, target, transport):
    if tensor.device == target:
        return tensor
    if transport == 'host_staged':
        return tensor.cpu().to(target)
    return tensor.to(target)


def relocate_cache(cache, target, transport):
    if cache is None or transport == 'same_gpu_ipc':
        return cache
    legacy = cache.to_legacy_cache()
    return type(cache).from_legacy_cache(tuple(
        tuple(relocate_tensor(t, target, transport) for t in layer) for layer in legacy
    ))


def decoder(conn, root, checkpoint, device_index, transport):
    import torch

    try:
        torch.cuda.set_device(device_index)
        wrapper, _, _, _, _ = load_methods(root, checkpoint, True, f'cuda:{device_index}')
        observer = Observer(wrapper)
        original_forward = wrapper.model.forward
        cached = None
        first = False

        @wraps(original_forward)
        def forward(*args, **kwargs):
            nonlocal first
            if not first:
                return original_forward(*args, **kwargs)
            assert not args
            first = False
            # The P process owns the inherited cache handle. Send the exact
            # prepared inputs but do not relay the cache back at this boundary.
            forward_args = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                            for k, v in kwargs.items() if k != 'past_key_values'}
            conn.send({'prefill': forward_args})
            result = receive(conn)['prefill_result']
            if transport == 'same_gpu_ipc':
                return result
            start = time.perf_counter()
            result.past_key_values = relocate_cache(result.past_key_values, wrapper.device, transport)
            elapsed = time.perf_counter() - start
            observer.roles[-1]['p_to_d_kv_copy_s'] = elapsed
            if result.hidden_states is not None:
                result.hidden_states = tuple(relocate_tensor(t, wrapper.device, transport) for t in result.hidden_states)
            if result.logits is not None:
                result.logits = relocate_tensor(result.logits, wrapper.device, transport)
            return result

        wrapper.model.forward = forward
        torch.cuda.synchronize()
        conn.send({'ready': True, 'pid': os.getpid(), 'vocab': len(wrapper.tokenizer)})
        with torch.inference_mode():
            while True:
                request = receive(conn)
                if request.get('stop'):
                    return
                if request.get('reset'):
                    cached = None
                    observer.roles.clear()
                    conn.send({'reset': True})
                    continue
                stage = {'role': request['role']}
                observer.roles[:] = [stage]
                call = {k: v.to(wrapper.device) if isinstance(v, torch.Tensor) else v
                        for k, v in request['kwargs'].items()}
                first = True
                if request['kind'] == 'latent':
                    cached = wrapper.generate_latent_batch(**call, past_key_values=cached)
                    torch.cuda.synchronize()
                    conn.send({'cache': cached, 'stage': stage})
                else:
                    torch.set_rng_state(request['rng_cpu'])
                    torch.cuda.set_rng_state(request['rng_cuda'])
                    texts, _ = wrapper.generate_text_batch(**call, past_key_values=cached)
                    del _  # The two upstream methods ignore final text KV.
                    cached = None
                    seqs = stage.pop('_ids_gpu').cpu()
                    conn.send({'texts': texts, 'sequences': seqs, 'stage': stage,
                               'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state()})
    except BaseException:
        conn.send({'error': traceback.format_exc()})
        raise
    finally:
        conn.close()


@contextmanager
def instrument(wrapper, conn, layout, audit, transport):
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast

    observer = Observer(wrapper)
    local_latent = wrapper.generate_latent_batch
    local_text = wrapper.generate_text_batch

    def dispatch(kind, input_ids, attention_mask, inherited, kwargs):
        stage = observer.roles[-1]
        request = {'kind': kind, 'role': stage['role'], 'kwargs': {
            'input_ids': input_ids.cpu(),
            'attention_mask': None if attention_mask is None else attention_mask.cpu(), **kwargs}}
        if kind == 'text':
            request.update(rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())
        conn.send(request)
        callback = receive(conn)['prefill']
        call = {k: v.to(wrapper.device) if isinstance(v, torch.Tensor) else v for k, v in callback.items()}
        call['past_key_values'] = inherited
        prefill_start = time.perf_counter()
        result = wrapper.model(**call)
        if kind == 'latent':
            # These are the only hidden states read by the upstream continuation.
            hidden = (result.hidden_states[0][:, -1:, :].clone(), result.hidden_states[-1][:, -1:, :].clone())
            returned = CausalLMOutputWithPast(past_key_values=result.past_key_values, hidden_states=hidden)
        else:
            returned = CausalLMOutputWithPast(past_key_values=result.past_key_values,
                                            logits=result.logits[:, -1:, :].clone())
        torch.cuda.synchronize()
        stage['prefill_wall_s'] = time.perf_counter() - prefill_start
        stage['p_to_d_kv_bytes'] = state_size(returned.past_key_values)
        stage['kv_handoffs'] = 1
        conn.send({'prefill_result': returned})
        response = receive(conn)
        stage['p_to_d_kv_copy_s'] = response['stage'].get('p_to_d_kv_copy_s', 0)
        if kind == 'latent':
            stage['d_to_p_kv_bytes'] = state_size(response['cache'])
            stage['kv_handoffs'] += 1
            start_copy = time.perf_counter()
            result = relocate_cache(response['cache'], wrapper.device, transport)
            stage['d_to_p_kv_copy_s'] = time.perf_counter() - start_copy if transport != 'same_gpu_ipc' else 0
            return result
        torch.set_rng_state(response['rng_cpu'])
        torch.cuda.set_rng_state(response['rng_cuda'])
        entry = response['stage']
        # Preserve the P-role start timestamp. First Judge token is timed in D
        # on the same host monotonic clock.
        if 'first_token' in entry:
            stage['first_token'] = entry['first_token']
        stage['_ids_gpu'] = response['sequences']
        return response['texts'], None

    def latent(input_ids, attention_mask=None, *, past_key_values=None, **kwargs):
        if layout == 'split':
            result = dispatch('latent', input_ids, attention_mask, past_key_values, kwargs)
            observer.roles[-1]['latent_steps'] = kwargs['latent_steps']
        else:
            result = local_latent(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)
        if audit:
            observer.roles[-1]['kv_sha'] = cache_digest(result)['sha256']
        return result

    def text(input_ids, attention_mask=None, *, past_key_values=None, **kwargs):
        if layout == 'split':
            return dispatch('text', input_ids, attention_mask, past_key_values, kwargs)
        return local_text(input_ids, attention_mask, past_key_values=past_key_values, **kwargs)

    wrapper.generate_latent_batch = latent
    wrapper.generate_text_batch = text
    try:
        yield observer
    finally:
        observer.restore()
        wrapper.generate_text_batch = local_text


def execute(method, wrapper, conn, items, layout, transport, audit=False):
    with instrument(wrapper, conn, layout, audit, transport) as observer:
        upstream, cases, stages, start, end = observer.observe(method, items)
    # Decode retains the preceding latent KV between agents, never between
    # requests. Reset outside the measured interval in both policies.
    conn.send({'reset': True})
    assert receive(conn)['reset']
    return upstream, {
        'service_s': end - start, 'ttft_s': stages[-1]['first_token'] - start,
        'stages': stages, 'cases': cases,
        'kv_handoffs': sum(s.get('kv_handoffs', 0) for s in stages),
        'logical_kv_bytes': sum(s.get('p_to_d_kv_bytes', 0) + s.get('d_to_p_kv_bytes', 0) for s in stages),
    }


def main():
    import torch
    import torch.multiprocessing as mp
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--decode-uuid', help='Second physical GPU. Omit for same-GPU CUDA IPC.')
    args = parser.parse_args()
    uuids = [args.gpu_uuid] + ([args.decode_uuid] if args.decode_uuid else [])
    assert len(uuids) == len(set(uuids))
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(uuids)

    def active_pids():
        return set().union(*(compute_pids(uuid) for uuid in uuids))

    assert not active_pids(), 'GPU has an existing compute process'
    torch.cuda.set_device(0)
    transport = 'same_gpu_ipc'
    if args.decode_uuid:
        transport = 'device_peer' if all(torch.cuda.can_device_access_peer(a, b) for a, b in ((0, 1), (1, 0))) else 'host_staged'
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    filenames = ('benchmark_pd_matrix.py', 'benchmark_dc_workloads.py', 'hf_semantic_pd.py', 'answer_parsing.py')
    for name in (*filenames, *(p.name for p in Path(__file__).parent.glob('env_*.sh'))):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    (args.out / 'source_sha256.json').write_text(json.dumps(manifest, indent=2))
    ctx = mp.get_context('spawn')
    conn, endpoint = ctx.Pipe()
    child = ctx.Process(target=decoder, args=(endpoint, args.root, args.model, len(uuids) - 1, transport))

    def status(stage, **details):
        record = {'stage': stage, 'parent': os.getpid(), 'decoder': child.pid, **details}
        temp = args.out / 'pipeline.tmp'
        temp.write_text(json.dumps(record, indent=2))
        temp.replace(args.out / 'pipeline.json')

    child.start()
    endpoint.close()
    completed = 0
    try:
        status('initializing', completed=completed)
        ready = receive(conn)
        assert ready['ready']
        wrapper, methods, samples, seed, shim = load_methods(args.root, args.model, True)
        assert ready['vocab'] == len(wrapper.tokenizer)
        (args.out / 'config.json').write_text(json.dumps({
            'gpu': torch.cuda.get_device_name(), 'gpu_uuid': args.gpu_uuid, 'decode_gpu_uuid': args.decode_uuid,
            'physical_gpus': len(uuids), 'resident_replicas': 2, 'transport': transport,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'checkpoint': str(args.model.resolve()),
            'sample_sha256': hashlib.sha256((args.root / 'samples.json').read_bytes()).hexdigest(),
            'sample_indices': BLOCKS[0], 'batch_size': 4, 'seed': 17, 'cap': 1024,
            'latent_steps_per_agent': 40, 'hf_unused_import_shim': shim,
            'scope': 'serialized batches with matched GPU and resident model count',
        }, indent=2))
        assert not active_pids() - {os.getpid(), child.pid}
        with torch.inference_mode():
            for name, method in methods.items():
                cap = 'judger_max_new_tokens' if name == 'latent' else 'max_new_tokens_each'
                setattr(method, cap, 16)
                gated = {}
                for layout in ('colocated', 'split'):
                    status('gate', method=name, layout=layout, completed=completed)
                    seed(901)
                    gated[layout] = execute(method, wrapper, conn, samples[:2], layout, transport, audit=True)
                a, b = (gated[x] for x in ('colocated', 'split'))
                assert a[0] == b[0] and a[1]['cases'] == b[1]['cases'], f'{name} gate answer mismatch'
                assert [s.get('kv_sha') for s in a[1]['stages']] == [s.get('kv_sha') for s in b[1]['stages']]
                (args.out / f'{name}_gate.json').write_text(json.dumps({'case_fields_equal': True, 'latent_kv_sha_equal': True}))
                setattr(method, cap, 1024)
                results = {}
                layouts = ('colocated', 'split') if name == 'latent' else ('split', 'colocated')
                for layout in layouts:
                    seed(17)
                    results[layout] = []
                    with (args.out / f'{name}_{layout}.jsonl').open('x') as f:
                        for i in range(0, len(BLOCKS[0]), 4):
                            status('measuring', method=name, layout=layout, batch=i // 4, completed=completed)
                            assert not active_pids() - {os.getpid(), child.pid}
                            upstream, row = execute(method, wrapper, conn, [samples[j] for j in BLOCKS[0][i:i+4]], layout, transport)
                            assert not active_pids() - {os.getpid(), child.pid}
                            results[layout].append((upstream, row['cases']))
                            f.write(json.dumps(row) + '\n')
                            f.flush()
                            completed += len(row['cases'])
                assert results['colocated'] == results['split'], f'{name} formal output mismatch'
                (args.out / f'{name}_equivalence.json').write_text(json.dumps({'cases': len(BLOCKS[0]), 'equal': True}))
        conn.send({'stop': True})
        child.join(15)
        assert child.exitcode == 0
        status('complete', completed=completed)
    except BaseException:
        status('failed', completed=completed, error=traceback.format_exc())
        raise
    finally:
        if child.is_alive():
            child.terminate()
            child.join(15)
        conn.close()


if __name__ == '__main__':
    main()
