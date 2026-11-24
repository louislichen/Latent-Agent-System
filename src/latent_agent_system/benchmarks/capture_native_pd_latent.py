"""Freeze real Judge inputs from upstream LatentMAS's vLLM branch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=24)
    args = parser.parse_args()
    import torch

    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.samples.read_text())
    items = source[2:2 + args.count]
    assert len(items) == args.count
    sys.path.insert(0, str(args.source.resolve()))
    config = SimpleNamespace(model_name=str(args.model), device='cuda:0', device2='cuda:0',
                             task='gsm8k', latent_space_realign=True, latent_only=False,
                             sequential_info_only=False, prompt='sequential', think=False,
                             max_new_tokens=256, tensor_parallel_size=1, gpu_memory_utilization=.8,
                             enable_prefix_caching=True, method='latent_mas', use_second_HF_model=True)
    frozen = []

    class CaptureEngine:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, sampling_params):
            for prompt in prompts:
                raw = prompt['prompt_embeds'].detach().contiguous().cpu()
                compact = raw.to(torch.bfloat16)
                assert raw.ndim == 2 and torch.isfinite(raw).all()
                assert torch.equal(raw, compact.to(raw.dtype)), 'BF16 transport would change the prompt values'
                frozen.append((compact, str(raw.dtype)))
            return [SimpleNamespace(outputs=[SimpleNamespace(text='')]) for _ in prompts]

    # Upstream's normal vLLM constructor selects its separate, unresized HF
    # model. Only replace the final serving engine with the input recorder.
    vllm = ModuleType('vllm')
    vllm.LLM, vllm.SamplingParams = CaptureEngine, SimpleNamespace
    sys.modules['vllm'] = vllm
    try:
        from models import ModelWrapper
        from methods.latent_mas import LatentMASMethod
    finally:
        del sys.modules['vllm']
    torch.manual_seed(17)
    torch.set_num_threads(4)
    model = ModelWrapper(config.model_name, torch.device(config.device), use_vllm=True, args=config)
    method = LatentMASMethod(model, latent_steps=40, judger_max_new_tokens=256,
                             temperature=0, top_p=1, generate_bs=2, args=config)
    with torch.inference_mode():
        for i in range(0, len(items), 2):
            method.run_batch_vllm(items[i:i + 2])
            print(f'Captured {len(frozen)}/{len(items)}', flush=True)
    manifest = []
    for item, (tensor, dtype) in zip(items, frozen, strict=True):
        manifest.append({
            'dataset_index': item['dataset_index'], 'gold': item['gold'],
            'positions': tensor.shape[0], 'hidden_size': tensor.shape[1],
            'upstream_dtype': dtype, 'wire_dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
        })
    torch.save([tensor for tensor, _ in frozen], args.out / 'judge_embeds.pt')
    (args.out / 'manifest.json').write_text(json.dumps({
        'samples_sha256': hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        'model': str(args.model.resolve()), 'batch_size': 2, 'latent_steps_per_agent': 40,
        'upstream_commit': '9a9e4d331eb11430bd9e64754c6b252b06d73031', 'cases': manifest,
    }, indent=2))


if __name__ == '__main__':
    main()
