"""CPU regression tests for the serving boundaries shared by the native workloads."""

import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx

from latent_agent_system.serving import vllm_runtime
from latent_agent_system.serving.topology import Route, plan_placement
from latent_agent_system.serving.vllm_client import completion


def stream(tokens, done=True):
    chunk = json.dumps({'choices': [{'token_ids': tokens}]})
    return httpx.Response(200, text=f'data: {chunk}\n\n' + ('data: [DONE]\n\n' if done else ''))


class PlacementTests(unittest.TestCase):
    def test_colocated_and_split_routes(self):
        gpus = ('first', 'second', 'third')
        colocated = plan_placement(gpus, 0, 18560)
        self.assertEqual(colocated.route(4), Route('http://127.0.0.1:18561'))
        split = plan_placement(gpus, 1, 18560)
        self.assertEqual(
            [worker.role for worker in split.workers], ['kv_producer', 'kv_consumer', 'kv_consumer'],
        )
        self.assertEqual(split.route(1), Route('http://127.0.0.1:18562', 'http://127.0.0.1:18560'))
        self.assertEqual(split.workers[0].side_channel_port, 18576)

    def test_invalid_placement_is_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'distinct GPU'):
            plan_placement(('first', 'first'), 1, 18560)
        with self.assertRaisesRegex(ValueError, 'decode worker'):
            plan_placement(('first', 'second'), 2, 18560)


class WorkerTests(unittest.TestCase):
    def test_latent_split_worker_launches_the_packaged_nixl_connector(self):
        placement = plan_placement(('first', 'second'), 1, 18560)
        config = vllm_runtime.VllmConfig(
            model=Path('/model'), max_model_len=512, prefill_chunk_tokens=256, enable_embeds=True,
        )
        command, env = vllm_runtime.worker_launch(config, placement.workers[0])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'first')
        self.assertEqual(env['VLLM_KV_CACHE_LAYOUT'], 'HND')
        self.assertEqual(command[command.index('--max-num-batched-tokens') + 1], '256')
        connector = json.loads(command[command.index('--kv-transfer-config') + 1])
        self.assertEqual(connector['kv_role'], 'kv_producer')
        self.assertEqual(connector['kv_connector_module_path'], 'latent_agent_system.serving.latent_nixl_v031')

    def test_partial_launch_stops_workers_started_by_this_deployment(self):
        placement = plan_placement(('first', 'second'), 1, 18560)
        config = vllm_runtime.VllmConfig(model=Path('/model'), max_model_len=512)
        first = Mock(pid=12345)
        with TemporaryDirectory() as directory:
            logs = Path(directory) / 'workers'
            with (
                patch.object(vllm_runtime, 'check_gpu_processes'),
                patch.object(vllm_runtime.subprocess, 'Popen', side_effect=[first, OSError('launch failed')]),
                patch.object(vllm_runtime.os, 'killpg') as kill,
            ):
                with self.assertRaisesRegex(OSError, 'launch failed'):
                    with vllm_runtime.serving(config, placement, logs):
                        self.fail('An incomplete deployment was exposed.')
                kill.assert_called_once_with(first.pid, signal.SIGTERM)
                first.wait.assert_called_once_with(timeout=20)


class RequestProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_colocated_request_goes_directly_to_decode(self):
        def respond(request):
            payload = json.loads(request.content)
            self.assertEqual(request.url.host, 'decode')
            self.assertEqual(payload['prompt'], [9, 10])
            self.assertNotIn('kv_transfer_params', payload)
            return stream([7, 8])

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            row = await completion(client, [9, 10], 2, Route('http://decode'))
        self.assertEqual(row['token_ids'], [7, 8])
        self.assertIsNone(row['prefill_response_s'])

    async def test_split_without_continuation_counts_the_decode_stream(self):
        hosts = []
        metadata = {'do_remote_prefill': True, 'remote_block_ids': [3], 'remote_num_tokens': 2}

        def respond(request):
            payload = json.loads(request.content)
            hosts.append(request.url.host)
            self.assertEqual(payload['prompt'], [9, 10])
            if request.url.host == 'prefill':
                self.assertFalse(payload['stream'])
                return httpx.Response(200, json={'kv_transfer_params': metadata, 'choices': [{'token_ids': [6]}]})
            self.assertEqual(payload['max_tokens'], 2)
            self.assertEqual(payload['kv_transfer_params'], metadata)
            return stream([7, 8])

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            row = await completion(client, [9, 10], 2, Route('http://decode', 'http://prefill'))
        self.assertEqual(hosts, ['prefill', 'decode'])
        self.assertEqual(row['token_ids'], [7, 8])

    async def test_split_latent_request_counts_prefill_token_once(self):
        metadata = {'do_remote_prefill': True, 'remote_block_ids': [[3, 4]], 'remote_num_tokens': 2}

        def respond(request):
            payload = json.loads(request.content)
            if request.url.host == 'prefill':
                self.assertEqual(payload['prompt_embeds'], 'initial')
                self.assertNotIn('prompt', payload)
                return httpx.Response(200, json={'kv_transfer_params': metadata, 'choices': [{'token_ids': [7]}]})
            self.assertEqual(payload['prompt_embeds'], 'initial+7')
            self.assertEqual(payload['max_tokens'], 1)
            self.assertEqual(payload['kv_transfer_params'], metadata)
            return stream([8])

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            row = await completion(
                client, None, 2, Route('http://decode', 'http://prefill'), prompt_embeds='initial',
                continuation=lambda token: {'prompt_embeds': f'initial+{token}'}, prompt_positions=2,
            )
        self.assertEqual(row['token_ids'], [7, 8])
        self.assertEqual(row['kv_blocks'], 2)
        self.assertEqual(row['token_arrivals_s'][0], row['prefill_response_s'])

    async def test_incomplete_stream_fails(self):
        transport = httpx.MockTransport(lambda request: stream([7, 8], done=False))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaisesRegex(ValueError, 'done=False'):
                await completion(client, [9, 10], 2, Route('http://decode'))


if __name__ == '__main__':
    unittest.main()
