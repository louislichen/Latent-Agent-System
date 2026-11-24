"""Equal-card HF serving comparison with streaming and an open-loop trace.

Each worker advances at most four active streams in round-robin order. The HF
forwards are per request. Both layouts use the same prefill chunk size.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import shutil
import sys
import time

from benchmark_pd_matrix import compute_pids, relocate_cache, state_size
from hf_semantic_pd import upstream_config


CHUNK_SIZE = 512
ACTIVE_LIMIT = 4
TRACE = (
    (0.00, 64), (0.00, 64), (.18, 8192), (.19, 8192),
    (.40, 64), (.41, 64), (.80, 8192), (.81, 8192),
    (1.20, 64), (1.21, 64), (1.55, 64), (1.56, 64),
)
NEW_TOKENS = 32
REPLICATES = 3


@dataclass
class Request:
    index: int
    prompt: list[int]
    scheduled: float
    arrived: float
    tokens: list[int] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)
    offset: int = 0
    prefill_started: float | None = None
    prefill_finished: float | None = None
    copy_started: float | None = None
    copy_finished: float | None = None
    kv_bytes: int = 0
    first_token: int | None = None
    cache: object = None

    def emit(self, token):
        self.tokens.append(token)
        self.stamps.append(time.perf_counter())

    def result(self):
        return {'index': self.index, 'prompt_tokens': len(self.prompt), 'scheduled': self.scheduled,
                'arrival': self.arrived, 'prefill_started': self.prefill_started,
                'prefill_finished': self.prefill_finished, 'copy_started': self.copy_started,
                'copy_finished': self.copy_finished, 'kv_bytes': self.kv_bytes,
                'tokens': self.tokens, 'stamps': self.stamps}


class Worker:
    def __init__(self, wrapper):
        self.model = wrapper.model
        self.device = wrapper.device

    def forward(self, ids, cache):
        import torch
        ids = torch.tensor([ids], device=self.device)
        return self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                          logits_to_keep=1, return_dict=True)

    def prefill_chunk(self, job):
        if job.prefill_started is None:
            job.prefill_started = time.perf_counter()
        chunk = job.prompt[job.offset:job.offset + CHUNK_SIZE]
        out = self.forward(chunk, job.cache)
        job.cache = out.past_key_values
        job.offset += len(chunk)
        if job.offset == len(job.prompt):
            job.first_token = out.logits[0, -1].argmax().item()
            job.prefill_finished = time.perf_counter()
            return True
        return False

    def decode_one(self, job):
        out = self.forward([job.tokens[-1]], job.cache)
        job.cache = out.past_key_values
        job.emit(out.logits[0, -1].argmax().item())

    def serve(self, incoming, decode_only=False):
        import torch
        pending, active, complete = deque(), [], []
        ended = False
        with torch.inference_mode(), torch.cuda.device(self.device):
            decode_stream = torch.cuda.current_stream(self.device)
            while not ended or pending or active:
                if not ended and not pending and not active:
                    item = incoming.get()
                    if item is None:
                        ended = True
                    else:
                        pending.append(item)
                while not ended and not incoming.empty():
                    item = incoming.get_nowait()
                    if item is None:
                        ended = True
                    else:
                        pending.append(item)
                if decode_only:
                    while pending and len(active) < ACTIVE_LIMIT:
                        job = pending.popleft()
                        for layer in job.cache.to_legacy_cache():
                            for tensor in layer:
                                tensor.record_stream(decode_stream)
                        job.emit(job.first_token)
                        active.append(job)
                elif pending and len(active) < ACTIVE_LIMIT:
                    if self.prefill_chunk(pending[0]):
                        job = pending.popleft()
                        job.emit(job.first_token)
                        active.append(job)
                for job in active:
                    self.decode_one(job)
                    if len(job.tokens) == NEW_TOKENS:
                        complete.append(job.result())
                        job.cache = None
                active = [job for job in active if len(job.tokens) < NEW_TOKENS]
        return complete

    def prefill_pool(self, incoming, output, destination, route):
        import torch
        try:
            copy_stream = torch.cuda.Stream(device=destination)
            with torch.inference_mode(), torch.cuda.device(self.device):
                while (job := incoming.get()) is not None:
                    while not self.prefill_chunk(job):
                        pass
                    job.kv_bytes = state_size(job.cache)
                    job.copy_started = time.perf_counter()
                    with torch.cuda.stream(copy_stream):
                        job.cache = relocate_cache(job.cache, destination, route)
                    copy_stream.synchronize()
                    job.copy_finished = time.perf_counter()
                    output.put(job)
        finally:
            output.put(None)


def run(workers, prompts, mode, arrival_scale, route):
    incoming = [Queue(), Queue()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        if mode == 'unified':
            active = [pool.submit(w.serve, q) for w, q in zip(workers, incoming)]
        else:
            prefill = pool.submit(workers[0].prefill_pool, incoming[0], incoming[1], workers[1].device, route)
            active = [pool.submit(workers[1].serve, incoming[1], True)]
        origin = time.perf_counter() + .05
        for i, ((delay, _), prompt) in enumerate(zip(TRACE, prompts, strict=True)):
            scheduled = origin + delay * arrival_scale
            time.sleep(max(0, scheduled - time.perf_counter()))
            job = Request(i, prompt, scheduled, time.perf_counter())
            incoming[i % 2 if mode == 'unified' else 0].put(job)
        incoming[0].put(None)
        if mode == 'unified':
            incoming[1].put(None)
        else:
            prefill.result(timeout=180)
        results = [job for future in active for job in future.result(timeout=180)]
    results.sort(key=lambda job: job['index'])
    assert [job['index'] for job in results] == list(range(len(TRACE)))
    return results


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(set(uuids)) == 2

    def assert_idle_of_others():
        assert not set().union(*(compute_pids(gpu) for gpu in uuids)) - {os.getpid()}

    assert_idle_of_others()
    sys.path.insert(0, str(args.root / 'sources/LatentMAS'))
    from models import ModelWrapper
    from utils import set_seed

    args.out.mkdir(parents=True, exist_ok=False)
    frozen = args.out / 'source'
    frozen.mkdir()
    scripts = ('benchmark_dual_gpu_text_serving.py', 'benchmark_pd_matrix.py', 'benchmark_dc_workloads.py',
               'hf_semantic_pd.py', 'answer_parsing.py')
    for name in scripts:
        shutil.copy2(Path(__file__).with_name(name), frozen / name)
    (args.out / 'source_sha256.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                          for p in frozen.iterdir()}, indent=2))
    cfg = upstream_config(NEW_TOKENS)
    cfg.model_name = str(args.model)
    cfg.latent_space_realign = False
    torch.set_num_threads(4)
    set_seed(17)
    wrappers = [ModelWrapper(cfg.model_name, torch.device('cuda', i), use_vllm=False, args=cfg) for i in range(2)]
    workers = [Worker(w) for w in wrappers]
    tokenizer = wrappers[0].tokenizer
    prompts = []
    for i, (_, length) in enumerate(TRACE):
        tokens = tokenizer.encode(f'Request {i}. Explain why a rainbow forms after rain. '
                                  'Start with a short explanation and give examples. ', add_special_tokens=False)
        prompts.append((tokens * ((length + len(tokens) - 1) // len(tokens)))[:length])
    route = 'device_peer' if torch.cuda.can_device_access_peer(0, 1) else 'host_staged'
    config = {'devices': uuids, 'names': [torch.cuda.get_device_name(i) for i in range(2)], 'route': route,
              'checkpoint': str(args.model.resolve()), 'torch': torch.__version__, 'transformers': transformers.__version__,
              'schedule': TRACE, 'output_tokens': NEW_TOKENS, 'prefill_chunk_size': CHUNK_SIZE,
              'active_limit_per_decode_worker': ACTIVE_LIMIT,
              'layout': 'two whole replicas or one prefill pool and one decode pool',
              'engine': 'HF eager, one forward per request, chunked prefill, round-robin active decode',
              'ttft_slo_s': 2.0, 'max_itl_slo_s': .2, 'replicates': REPLICATES}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    with torch.inference_mode():
        for w in workers:
            state = w.forward(prompts[2][:CHUNK_SIZE], None)
            w.forward([int(state.logits[0, -1].argmax().item())], state.past_key_values)
            torch.cuda.synchronize(w.device)
            del state
    with (args.out / 'runs.jsonl').open('x') as f:
        for replicate in range(REPLICATES):
            for trace, (scale, name) in enumerate(((1.0, 'burst'), (4.0, 'spread'))):
                modes = ('unified', 'split')
                if (trace + replicate) % 2:
                    modes = modes[::-1]
                prior = None
                for mode in modes:
                    assert_idle_of_others()
                    cases = run(workers, prompts, mode, scale, route)
                    assert_idle_of_others()
                    tokens = [case['tokens'] for case in cases]
                    if prior is not None:
                        assert tokens == prior, f'Decode outputs changed after placement in {name}'
                    prior = tokens
                    f.write(json.dumps({'trace': name, 'layout': mode, 'replicate': replicate, 'cases': cases}) + '\n')
                    f.flush()
                    print(name, mode, replicate, len(cases), flush=True)
    (args.out / 'complete.json').write_text(json.dumps({'requests': len(TRACE) * 4 * REPLICATES, 'tokens_equal': True}))


if __name__ == '__main__':
    main()
