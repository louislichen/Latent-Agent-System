"""Run official HF LatentMAS and TextMAS with identical GPU and workload budgets."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

from latent_agent_system.profiling.hf_workflow import Observer
from latent_agent_system.workflows.latentmas import load_methods


SCENARIOS = ('batch4', 'burst4', 'serial1')
BLOCKS = (tuple(range(2, 10)), tuple(range(10, 18)))


def wait_until(deadline):
    while (remaining := deadline - time.perf_counter()) > 0:
        time.sleep(remaining)


def burst_work(items, relative_arrivals):
    """An independent producer enqueues requests even while CUDA is busy."""
    ready = queue.Queue()
    origin = time.perf_counter() + .075

    def produce():
        for item, offset in zip(items, relative_arrivals, strict=True):
            wait_until(origin + offset)
            ready.put((item, time.perf_counter(), origin + offset))

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    remaining = len(items)
    try:
        while remaining:
            entry = ready.get(timeout=60)
            group = [entry]
            deadline = entry[1] + .025
            while len(group) < min(remaining, 4):
                try:
                    group.append(ready.get(timeout=max(0, deadline - time.perf_counter())))
                except queue.Empty:
                    break
            remaining -= len(group)
            yield group
    finally:
        producer.join(timeout=1)


def workload(items, scenario):
    if scenario == 'burst4':
        assert len(items) == 8
        yield from burst_work(items, [0] * 6 + [8] * 2)
    else:
        size = 4 if scenario == 'batch4' else 1
        for start in range(0, len(items), size):
            arrival = time.perf_counter()
            yield [(item, arrival, arrival) for item in items[start:start + size]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--method', choices=('latent', 'text'), required=True)
    parser.add_argument('--block', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hf-import-shim', action='store_true')
    args = parser.parse_args()
    import torch
    import transformers

    args.output.mkdir(exist_ok=False, parents=True)
    wrapper, methods, samples, set_seed, shim = load_methods(args.root, args.model, args.hf_import_shim)
    method = methods[args.method]
    # One independent, short case checks observation without changing sampling.
    cap_attr = 'judger_max_new_tokens' if args.method == 'latent' else 'max_new_tokens_each'
    setattr(method, cap_attr, 16)
    with torch.inference_mode():
        set_seed(901)
        raw = method.run_batch(samples[:2])
        observer = Observer(wrapper)
        set_seed(901)
        observed, _, _, _, _ = observer.observe(method, samples[:2])
        assert raw == observed, 'TTFT observer changed the upstream result'
        setattr(method, cap_attr, 1024)
        torch.cuda.synchronize()
        config = {
            'checkpoint': str(args.model), 'config_sha256': hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(),
            'visible_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu_name': torch.cuda.get_device_name(),
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'model_dtype': str(wrapper.model.dtype), 'padding_side': wrapper.tokenizer.padding_side,
            'vocab': len(wrapper.tokenizer), 'top_k': wrapper.model.generation_config.top_k,
            'temperature': .6, 'top_p': .95, 'seed_per_scenario': 17,
            'judge_max_tokens': 1024, 'text_intermediate_cap': 1024, 'latent_steps_per_agent': 40,
            'text_mas_context_length': -1,
            'method': args.method, 'block': args.block, 'observed_vs_raw_gate_equal': True,
            'hf_unused_vllm_sampling_params_import_shim': shim,
            'batching': 'FIFO, up to four requests per whole chain, no replacement during decoding',
            'ttft': 'first sampled Judge token, server host clock from independent enqueue or closed-loop submit',
        }
        (args.output / 'config.json').write_text(json.dumps(config, indent=2))
        items = [samples[i] for i in BLOCKS[args.block]]
        for scenario in SCENARIOS:
            set_seed(17)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            print(json.dumps({'start': scenario, 'method': args.method, 'block': args.block}), flush=True)
            with (args.output / (scenario + '.jsonl')).open('x') as stream:
                for batch, group in enumerate(workload(items, scenario)):
                    batch_items, arrivals, scheduled = zip(*group)
                    _, cases, stages, start, finish = observer.observe(method, list(batch_items))
                    first = stages[-1]['first_token']
                    for case, t, planned in zip(cases, arrivals, scheduled, strict=True):
                        case.update(arrival=t, scheduled_arrival=planned, queue_s=start-t,
                                    ttft_s=first-t, service_ttft_s=first-start,
                                    e2e_s=finish-t, finish=finish)
                    record = {'scenario': scenario, 'batch': batch, 'size': len(group), 'start': start,
                              'finish': finish, 'service_s': finish-start, 'stages': stages,
                              'cases': cases, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(json.dumps({'scenario': scenario, 'batch': batch, 'size': len(group),
                                      'service_s': finish-start}), flush=True)
        observer.restore()
    (args.output / 'complete.json').write_text(json.dumps({'scenarios': list(SCENARIOS), 'requests_each': len(items)}))


if __name__ == '__main__':
    main()
