"""Launch, health-check, and shut down an owned deployment of vLLM workers."""

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx

from latent_agent_system.serving.topology import Placement, Worker
from latent_agent_system.serving.vllm_client import MODEL_NAME


@dataclass(frozen=True)
class VllmConfig:
    model: Path
    max_model_len: int
    chunk_tokens: int = 2048
    prefill_chunk_tokens: int | None = None
    max_seqs: int = 32
    memory: float = 0.83
    buffer: str = 'cpu'
    eager: bool = False
    enable_embeds: bool = False


@dataclass
class RunningDeployment:
    placement: Placement
    processes: list[subprocess.Popen]
    logs: Path

    def check_health(self):
        for index, process in enumerate(self.processes):
            if process.poll() is not None:
                raise RuntimeError(f'Worker {index} exited. See {self.logs}/worker{index}.log.')
        check_gpu_processes(self.placement.gpus, self.processes)


def gpu_processes(gpus):
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        text=True,
    ).splitlines()
    return {int(pid) for gpu, pid in (r.split(',') for r in rows)
            if gpu.strip() in gpus}


def check_gpu_processes(gpus, servers=()):
    own_groups = {server.pid for server in servers}
    for pid in gpu_processes(gpus):
        try:
            group = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if group not in own_groups:
            raise RuntimeError(f'A selected GPU is in use by process {pid}.')


def connector_config(role, buffer, enable_embeds=False):
    config = {
        'kv_connector': 'NixlConnector', 'kv_role': role, 'kv_buffer_device': buffer,
        'kv_load_failure_policy': 'fail',
        'kv_connector_extra_config': {'kv_lease_duration': 120},
    }
    if enable_embeds:
        config.update(
            kv_connector='LatentNixlPullConnector',
            kv_connector_module_path='latent_agent_system.serving.latent_nixl_v031',
        )
    return json.dumps(config)


def worker_launch(config: VllmConfig, worker: Worker):
    chunk_tokens = config.chunk_tokens
    if worker.role == 'kv_producer' and config.prefill_chunk_tokens is not None:
        chunk_tokens = config.prefill_chunk_tokens
    env = {
        **os.environ,
        'CUDA_VISIBLE_DEVICES': worker.gpu,
        'VLLM_NIXL_SIDE_CHANNEL_HOST': '127.0.0.1',
        'VLLM_NIXL_SIDE_CHANNEL_PORT': str(worker.side_channel_port),
        'UCX_TLS': 'sm,self,tcp' if config.buffer == 'cpu' else 'sm,self,tcp,cuda_copy,cuda_ipc',
    }
    command = [
        sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(config.model), '--served-model-name', MODEL_NAME,
        '--host', '127.0.0.1', '--port', str(worker.port), '--dtype', 'bfloat16',
        '--max-model-len', str(config.max_model_len),
        '--gpu-memory-utilization', str(config.memory),
        '--max-num-batched-tokens', str(chunk_tokens), '--max-num-seqs', str(config.max_seqs),
        '--generation-config', 'vllm', '--enable-chunked-prefill', '--no-enable-prefix-caching',
    ]
    if config.eager:
        command.append('--enforce-eager')
    if config.enable_embeds:
        command.append('--enable-prompt-embeds')
        package_root = Path(__file__).resolve().parents[2]
        import_paths = (str(package_root), env.get('PYTHONPATH'))
        env['PYTHONPATH'] = os.pathsep.join(filter(None, import_paths))
        env['VLLM_KV_CACHE_LAYOUT'] = 'HND'
    if worker.role != 'colocated':
        command.extend(['--kv-transfer-config', connector_config(worker.role, config.buffer, config.enable_embeds)])
    return command, env


def wait_until_ready(deployment: RunningDeployment):
    deadline = time.monotonic() + 360
    pending = {index: worker.url for index, worker in enumerate(deployment.placement.workers)}
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while pending:
            for index, url in list(pending.items()):
                if deployment.processes[index].poll() is not None:
                    raise RuntimeError(f'Worker {index} exited. See {deployment.logs}/worker{index}.log.')
                try:
                    response = client.get(url + '/v1/models')
                    if response.is_success and any(x['id'] == MODEL_NAME for x in response.json()['data']):
                        del pending[index]
                except httpx.RequestError:
                    pass
            if pending and time.monotonic() >= deadline:
                raise TimeoutError(f'Workers {sorted(pending)} did not start. See {deployment.logs}.')
            if pending:
                time.sleep(1)
    deployment.check_health()


def stop_workers(processes):
    for server in processes:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for server in processes:
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server.wait(timeout=5)


@contextmanager
def serving(config: VllmConfig, placement: Placement, logs: Path):
    deployment = RunningDeployment(placement, [], logs)
    logs.mkdir(parents=True, exist_ok=False)
    try:
        check_gpu_processes(placement.gpus)
        for index, worker in enumerate(placement.workers):
            command, env = worker_launch(config, worker)
            (logs / f'worker{index}_command.json').write_text(json.dumps({
                'argv': command, 'gpu': worker.gpu, 'role': worker.role,
                'side_channel_port': env['VLLM_NIXL_SIDE_CHANNEL_PORT'], 'UCX_TLS': env['UCX_TLS'],
            }, indent=2))
            with (logs / f'worker{index}.log').open('x') as log:
                deployment.processes.append(subprocess.Popen(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                ))
        wait_until_ready(deployment)
        yield deployment
    finally:
        stop_workers(deployment.processes)
