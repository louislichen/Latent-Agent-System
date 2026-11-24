"""Run official HF LatentMAS and TextMAS with identical GPU and workload budgets."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

from answer_parsing import final_number, numeric
from hf_semantic_pd import upstream_config


ROLES = ('planner', 'critic', 'refiner', 'judger')
SCENARIOS = ('batch4', 'burst4', 'serial1')
BLOCKS = (tuple(range(2, 10)), tuple(range(10, 18)))


def load_methods(root, model_path, allow_hf_import_shim, device='cuda:0'):
    sys.path.insert(0, str(root / 'sources/LatentMAS'))
    from models import ModelWrapper
    from methods.text_mas import TextMASMethod
    from utils import extract_gold, normalize_answer, set_seed
    try:
        from methods.latent_mas import LatentMASMethod
        shim = False
    except ModuleNotFoundError as exc:
        if exc.name != 'vllm' or not allow_hf_import_shim:
            raise
        # The upstream constructor also creates vLLM parameters on its HF path.
        # run_batch never reads that object. Keep the shim scoped to this import.
        shim_module = ModuleType('vllm')
        shim_module.SamplingParams = SimpleNamespace
        sys.modules['vllm'] = shim_module
        try:
            from methods.latent_mas import LatentMASMethod
        finally:
            del sys.modules['vllm']
        shim = True
    import torch

    set_seed(17)
    torch.set_num_threads(4)
    cfg = upstream_config(1024)
    cfg.model_name = str(model_path)
    cfg.text_mas_context_length = -1  # Official run.py default for TextMAS.
    wrapper = ModelWrapper(cfg.model_name, torch.device(device), use_vllm=False, args=cfg)
    methods = {
        'latent': LatentMASMethod(wrapper, latent_steps=40, judger_max_new_tokens=1024,
                                  temperature=.6, top_p=.95, generate_bs=4, args=cfg),
        'text': TextMASMethod(wrapper, max_new_tokens_each=1024, temperature=.6, top_p=.95,
                              generate_bs=4, args=cfg),
    }
    items = json.loads((root / 'samples.json').read_text())
    for item in items:
        item['numeric_gold'] = item['gold']
        item['gold'] = normalize_answer(extract_gold(item['solution']))
        item['question'] = item['question'].strip()
    return wrapper, methods, items, set_seed, shim


class Observer:
    def __init__(self, wrapper):
        import torch
        from transformers import StoppingCriteriaList

        self.roles = []
        model = wrapper.model
        prepare, generate, latent = wrapper.prepare_chat_batch, model.generate, wrapper.generate_latent_batch
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

        def observed_prepare(*args, **kwargs):
            entry = {'role': ROLES[len(self.roles)], 'start': time.perf_counter()}
            self.roles.append(entry)
            return prepare(*args, **kwargs)

        def observed_latent(*args, **kwargs):
            stage = self.roles[-1]
            result = latent(*args, **kwargs)
            layers = result.to_legacy_cache()
            stage['logical_kv_bytes'] = sum(t.numel() * t.element_size() for layer in layers for t in layer)
            stage['latent_steps'] = kwargs['latent_steps']
            return result

        def observed_generate(*args, **kwargs):
            stage = self.roles[-1]
            prompt_width = kwargs['input_ids'].shape[1]

            def first_token(ids, scores, **_):
                if 'first_token' not in stage:
                    torch.cuda.synchronize()
                    stage['first_token'] = time.perf_counter()
                return False

            if stage['role'] == 'judger':
                assert 'stopping_criteria' not in kwargs
                kwargs['stopping_criteria'] = StoppingCriteriaList([first_token])
            output = generate(*args, **kwargs)
            stage['_ids_gpu'] = output.sequences[:, prompt_width:].detach()
            return output

        wrapper.prepare_chat_batch = observed_prepare
        wrapper.generate_latent_batch = observed_latent
        model.generate = observed_generate
        self.restore = lambda: self.restore_wrappers(wrapper, model, prepare, latent, generate)

    @staticmethod
    def restore_wrappers(wrapper, model, prepare, latent, generate):
        wrapper.prepare_chat_batch = prepare
        wrapper.generate_latent_batch = latent
        model.generate = generate

    def observe(self, method, items):
        import torch

        self.roles.clear()
        start = time.perf_counter()
        result = method.run_batch(items)
        torch.cuda.synchronize()
        finish = time.perf_counter()
        assert [stage['role'] for stage in self.roles] == list(ROLES)
        assert start <= self.roles[-1]['first_token'] <= finish
        cases = []
        for stage_idx, stage in enumerate(self.roles):
            next_start = self.roles[stage_idx + 1]['start'] if stage_idx + 1 < len(ROLES) else finish
            stage['wall_s'] = next_start - stage['start']
            if '_ids_gpu' not in stage:
                continue
            padded = stage.pop('_ids_gpu').cpu().tolist()
            actual = [[*ids[:next((i for i, t in enumerate(ids) if t in self.eos), len(ids))]] for ids in padded]
            stage['output_tokens'] = [len(ids) for ids in actual]
            stage['executed_batch_decode_steps'] = len(padded[0])
            if stage['role'] != 'judger':
                continue
            for item, row, ids, active in zip(items, result, padded, actual, strict=True):
                pred = final_number(row['raw_prediction'])
                stopped = len(active) < len(ids)
                cases.append({
                    'dataset_index': item['dataset_index'], 'token_ids': active,
                    'final_output_tokens': len(active), 'finish_reason': 'stop' if stopped else 'length',
                    'numeric_match': pred is not None and pred == numeric(item['numeric_gold']),
                    'official_correct': bool(row['correct']), 'numeric_prediction': pred,
                    'text': row['raw_prediction'],
                    'role_input_tokens': [len(agent['input_ids']) for agent in row['agents']],
                })
        for i, case in enumerate(cases):
            case['intermediate_text_tokens'] = sum(stage['output_tokens'][i] for stage in self.roles[:3]
                                                    if 'output_tokens' in stage)
            case['latent_rollout_steps'] = sum(stage.get('latent_steps', 0) for stage in self.roles[:3])
        return result, cases, list(self.roles), start, finish


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
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--method', choices=('latent', 'text'), required=True)
    parser.add_argument('--block', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hf-import-shim', action='store_true')
    args = parser.parse_args()
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
