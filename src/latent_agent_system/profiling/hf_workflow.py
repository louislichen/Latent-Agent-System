"""Observe the agent stages and final output of an HF workflow."""

import time

from latent_agent_system.data.answer_parsing import final_number, numeric
from latent_agent_system.workflows.latentmas import ROLES


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
