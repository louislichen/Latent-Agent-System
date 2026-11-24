"""Load official HF LatentMAS and TextMAS methods with shared experiment settings."""

import json
import sys
from types import ModuleType, SimpleNamespace

from latent_agent_system.serving.hf_semantic_pd import upstream_config


ROLES = ('planner', 'critic', 'refiner', 'judger')


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
