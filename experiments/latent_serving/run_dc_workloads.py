"""Run paired blocks on idle GPUs and reverse the method order or mapping."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def query(fields, kind='gpu'):
    output = subprocess.check_output(
        ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'], text=True)
    return [[part.strip() for part in row.split(',')] for row in output.splitlines() if row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--wait-seconds', type=float, default=0)
    parser.add_argument('--hf-import-shim', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('GPU indices must be distinct.')
    args.out.mkdir(exist_ok=False, parents=True)
    source = args.out / 'source'
    source.mkdir()
    for name in ('benchmark_dc_workloads.py', 'run_dc_workloads.py', 'answer_parsing.py', 'hf_semantic_pd.py'):
        shutil.copy2(Path(__file__).parent / name, source)
    (args.out / 'source_sha256.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in source.iterdir()}, indent=2))
    gpu_table = {int(row[0]): row[1:] for row in query('index,uuid,name,driver_version,memory.total')}
    candidates = {i: gpu_table[i] for i in args.gpus}
    jobs = []

    def save(stage, error=None):
        destination = args.out / 'pipeline.json'
        temp = destination.with_suffix('.tmp')
        temp.write_text(json.dumps({'stage': stage, 'controller_pid': os.getpid(), 'jobs': jobs, 'error': error}, indent=2))
        temp.replace(destination)

    active = {}
    try:
        deadline = time.monotonic() + args.wait_seconds
        save('waiting_for_idle_gpu')
        while True:
            busy = {row[0] for row in query('gpu_uuid,pid', 'compute-apps')}
            free = [i for i in args.gpus if candidates[i][0] not in busy]
            if free:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('No idle GPU became available within the configured wait.')
            time.sleep(5)
        chosen = free[:2]
        gpus = {i: candidates[i] for i in chosen}
        mode = 'GPU crossover' if len(chosen) == 2 else 'single GPU with method order reversed'
        (args.out / 'plan.json').write_text(json.dumps({
            'schema': 'official HF chain serving', 'device_comparison': mode,
            'gpus': gpus, 'model': str(args.model), 'blocks': [list(range(2, 10)), list(range(10, 18))],
            'model_loading_measured': False, 'serial1': 'closed loop, B1', 'batch4': 'closed loop, B4',
            'burst4': 'independent arrivals: 6 at 0s and 2 at 8s, FIFO max4, 25ms batch window',
            'sample_sha256': hashlib.sha256((args.root / 'samples.json').read_bytes()).hexdigest(),
        }, indent=2))

        def run_wave(block, assignments):
            busy = {row[0] for row in query('gpu_uuid,pid', 'compute-apps')}
            if any(gpus[gpu][0] in busy for _, gpu in assignments):
                raise RuntimeError('GPU became occupied before launch')
            for method, gpu in assignments:
                output = args.out / f'{method}_block{block}'
                cmd = [sys.executable, '-u', str(source / 'benchmark_dc_workloads.py'),
                       '--root', str(args.root), '--model', str(args.model),
                       '--method', method, '--block', str(block), '--output', str(output)]
                if args.hf_import_shim:
                    cmd.append('--hf-import-shim')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[gpu][0], CUDA_DEVICE_ORDER='PCI_BUS_ID')
                with (args.out / f'{method}_block{block}.log').open('w') as log:
                    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=args.root)
                job = {'method': method, 'block': block, 'gpu': gpu, 'pid': proc.pid, 'returncode': None}
                jobs.append(job)
                active[proc] = job
            save(f'block{block}')
            while active:
                expected = {gpus[job['gpu']][0]: proc.pid for proc, job in active.items()}
                collisions = [row for row in query('gpu_uuid,pid', 'compute-apps')
                              if row[0] in expected and int(row[1]) != expected[row[0]]]
                if collisions:
                    raise RuntimeError(f'Independent GPU work appeared during measurement: {collisions}')
                for proc, job in list(active.items()):
                    if (code := proc.poll()) is None:
                        continue
                    job['returncode'] = code
                    del active[proc]
                    save(f'block{block}')
                    if code:
                        raise RuntimeError(f"{job['method']} block {block} exited with {code}")
                if active:
                    time.sleep(3)

        for block in range(2):
            methods = ('latent', 'text')
            if len(chosen) == 2:
                run_wave(block, list(zip(methods, chosen if block == 0 else chosen[::-1], strict=True)))
            else:
                for method in (methods if block == 0 else methods[::-1]):
                    run_wave(block, [(method, chosen[0])])
        save('complete')
    except BaseException as exc:
        for proc, job in active.items():
            if proc.poll() is None:
                proc.terminate()
            job['returncode'] = proc.wait(timeout=15)
        save('failed', str(exc))
        raise


if __name__ == '__main__':
    main()
