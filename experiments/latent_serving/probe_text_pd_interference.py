"""Measure decode stalls caused by a controlled long-prompt arrival trace.

The optional second device can be another physical GPU or an independent MIG
GPU instance. The probe isolates prefill interference. Long requests stop at
prefill completion, so this does not benchmark KV transfer or serving goodput.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from hf_semantic_pd import upstream_config


ARRIVALS = (12, 44, 76)
DECODE_STEPS = 100


class TextProbe:
    def __init__(self, wrapper, short_tokens, long_tokens, chunk_tokens):
        self.wrapper = wrapper
        self.device = wrapper.device
        self.short = short_tokens
        self.long = long_tokens
        self.chunk = chunk_tokens

    def forward(self, ids, cache=None):
        return self.wrapper.model(input_ids=ids, past_key_values=cache, use_cache=True,
                                  logits_to_keep=1, return_dict=True)

    def long_prefill(self, event_step):
        import torch

        with torch.inference_mode(), torch.cuda.device(self.device):
            start = time.perf_counter()
            state = self.forward(self.long)
            torch.cuda.synchronize(self.device)
            assert state.past_key_values.get_seq_length() == self.long.shape[1]
            return {'arrival_decode_step': event_step, 'start': start, 'end': time.perf_counter()}

    def run(self, mode, replicate, peer):
        import torch

        with torch.inference_mode(), torch.cuda.device(self.device), ThreadPoolExecutor(max_workers=1) as pool:
            answer = self.forward(self.short)
            cache = answer.past_key_values
            token = answer.logits[:, -1:, :].argmax(dim=-1)
            output = [token.item()]
            previous = time.perf_counter()
            samples, events, pending = [], [], []
            chunk_start = None
            chunk_offset = 0
            chunk_cache = None
            chunk_at = None
            for step in range(1, DECODE_STEPS + 1):
                if step in ARRIVALS:
                    if mode == 'shared_full':
                        events.append(self.long_prefill(step))
                    elif mode == 'shared_chunked':
                        assert chunk_cache is None
                        chunk_start = time.perf_counter()
                        chunk_at = step
                        chunk_offset = 0
                    elif mode == 'isolated':
                        pending.append(pool.submit(peer.long_prefill, step))
                if chunk_start is not None:
                    state = self.forward(self.long[:, chunk_offset:chunk_offset + self.chunk], chunk_cache)
                    chunk_cache = state.past_key_values
                    chunk_offset += self.chunk
                    if chunk_offset >= self.long.shape[1]:
                        torch.cuda.synchronize(self.device)
                        assert chunk_cache.get_seq_length() == self.long.shape[1]
                        events.append({'arrival_decode_step': chunk_at, 'start': chunk_start,
                                       'end': time.perf_counter()})
                        chunk_start, chunk_cache = None, None
                        del state
                answer = self.forward(token, cache)
                cache = answer.past_key_values
                token = answer.logits[:, -1:, :].argmax(dim=-1)
                output.append(token.item())
                now = time.perf_counter()
                samples.append({'step': step, 'start': previous, 'end': now, 'itl_s': now - previous})
                previous = now
            events.extend(future.result(timeout=60) for future in pending)
            return {'mode': mode, 'replicate': replicate, 'token_ids': output,
                    'token_intervals': samples, 'prefill_events': events}


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prefill-device', type=int, help='Second visible CUDA device, normally 1')
    parser.add_argument('--long-tokens', type=int, default=8192)
    parser.add_argument('--chunk-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.prefill_device == 0:
        parser.error('The isolated condition requires a distinct visible CUDA device.')
    if not 0 < args.chunk_tokens <= args.long_tokens <= 16384:
        parser.error('Require 0 < chunk length <= prompt length <= 16384')
    if (args.long_tokens + args.chunk_tokens - 1) // args.chunk_tokens > 25:
        parser.error('Use at most 25 chunks so the long prompts have time to finish.')
    sys.path.insert(0, str(args.root / 'sources/LatentMAS'))
    from models import ModelWrapper
    from utils import set_seed

    cfg = upstream_config(128)
    cfg.latent_space_realign = False
    cfg.model_name = str(args.model)
    torch.set_num_threads(4)
    set_seed(17)
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out / 'source'
    source.mkdir()
    for filename in (Path(__file__).name, 'hf_semantic_pd.py'):
        shutil.copy2(Path(__file__).with_name(filename), source / filename)
    (args.out / 'source_sha256.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                          for p in source.iterdir()}, indent=2))
    wrapper = ModelWrapper(cfg.model_name, torch.device('cuda:0'), use_vllm=False, args=cfg)
    short = wrapper.tokenizer.encode('Explain why rainbows appear after rain. Use clear steps.', add_special_tokens=False)
    pattern = wrapper.tokenizer.encode('An ordinary text request may arrive while an earlier answer is being generated. '
                                       'Its input is processed before streaming its own answer. ', add_special_tokens=False)
    long = (pattern * ((args.long_tokens + len(pattern) - 1) // len(pattern)))[:args.long_tokens]

    def probe_on(model):
        return TextProbe(model, torch.tensor([short], device=model.device), torch.tensor([long], device=model.device),
                         args.chunk_tokens)

    probe = probe_on(wrapper)
    modes = ['decode_only', 'shared_full', 'shared_chunked']
    peer = None
    if args.prefill_device is not None:
        target = torch.device('cuda', args.prefill_device)
        peer = probe_on(ModelWrapper(cfg.model_name, target, use_vllm=False, args=cfg))
        modes.append('isolated')
    config = {'model': str(args.model.resolve()), 'torch': torch.__version__, 'transformers': transformers.__version__,
              'decode_gpu': torch.cuda.get_device_name(0), 'dtype': str(wrapper.model.dtype),
              'prefill_device': args.prefill_device, 'short_tokens': len(short), 'long_tokens': len(long),
              'chunk_tokens': args.chunk_tokens, 'decode_steps': DECODE_STEPS, 'arrival_decode_steps': ARRIVALS,
              'replicates': 3, 'modes': modes}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    with torch.inference_mode():
        for model in (probe, peer):
            if model is not None:
                model.long_prefill(-1)
                for _ in range(4):
                    model.forward(model.short)
                torch.cuda.synchronize(model.device)
    expected = None
    with (args.out / 'events.jsonl').open('x') as f:
        for replicate in range(3):
            order = modes[replicate:] + modes[:replicate]
            for mode in order:
                record = probe.run(mode, replicate, peer)
                if expected is None:
                    expected = record['token_ids']
                assert record['token_ids'] == expected, f'Decode tokens changed in {mode}'
                f.write(json.dumps(record) + '\n')
                f.flush()
                print(replicate, mode, 'measurements', len(record['token_intervals']), flush=True)
    (args.out / 'complete.json').write_text(json.dumps({'records': len(modes) * 3, 'decode_tokens_equal': True}))


if __name__ == '__main__':
    main()
