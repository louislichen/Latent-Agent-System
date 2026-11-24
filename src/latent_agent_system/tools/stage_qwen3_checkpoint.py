"""Stage the exact Qwen3-4B shards, verifying the existing experiments' hashes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


SHARDS = (
    ('model-00001-of-00003.safetensors', 3957900840, '328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223'),
    ('model-00002-of-00003.safetensors', 3987450520, '6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5'),
    ('model-00003-of-00003.safetensors', 99630640, 'e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1'),
)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 2**20), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_range(dest, base, task):
    name, size, start, end = task
    part = dest / f'.{name}.{start}-{end}.range'
    expected_size = end - start + 1
    for _ in range(8):
        have = part.stat().st_size if part.exists() else 0
        if have == expected_size:
            break
        segment = part.with_name(part.name + '.segment')
        range_start = start + have
        proc = subprocess.run(['curl', '-fsSL', '--connect-timeout', '10', '--max-time', '65',
                               '-r', f'{range_start}-{end}', '-D', '-', '-o', str(segment),
                               f'{base}/{name}'], capture_output=True, text=True)
        pattern = rf'^content-range:\s*bytes {range_start}-{end}/{size}\s*$'
        if segment.exists() and re.search(pattern, proc.stdout, re.IGNORECASE | re.MULTILINE):
            n = segment.stat().st_size
            if 0 < n <= expected_size - have:
                with part.open('ab') as f:
                    f.write(segment.read_bytes())
        segment.unlink(missing_ok=True)
    if not part.exists() or part.stat().st_size != expected_size:
        raise RuntimeError(f'Model download stalled for {name}, offset {start}')
    with (dest / (name + '.partial')).open('r+b', buffering=0) as f:
        data = part.read_bytes()
        assert os.pwrite(f.fileno(), data, start) == expected_size
    return task, part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dest', type=Path)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--shards', nargs='+', type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument('--base', default='https://www.modelscope.cn/models/Qwen/Qwen3-4B/resolve/master')
    args = parser.parse_args()
    dest = args.dest.resolve()
    selected = [SHARDS[i - 1] for i in sorted(set(args.shards))]
    manifest = dest / 'completed_ranges.json'
    done = set(map(tuple, json.loads(manifest.read_text()))) if manifest.is_file() else set()
    chunk_size = 8 * 2**20
    tasks = []
    for name, size, checksum in selected:
        if (dest / name).exists():
            assert digest(dest / name) == checksum
            continue
        temporary = dest / (name + '.partial')
        if not temporary.exists():
            with temporary.open('xb') as f:
                f.truncate(size)
            done = {task for task in done if task[0] != name}
        assert temporary.stat().st_size == size
        for start in range(0, size, chunk_size):
            task = (name, size, start, min(size, start + chunk_size) - 1)
            if task not in done:
                tasks.append(task)
    start = time.monotonic()
    print(f'{len(tasks)} ranges to download', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_range, dest, args.base, task) for task in tasks]
        received = 0
        failures = []
        for i, future in enumerate(as_completed(futures), 1):
            try:
                task, part = future.result()
            except Exception as exc:
                failures.append(str(exc))
                print(exc, flush=True)
                continue
            done.add(task)
            written = dest / 'completed_ranges.tmp'
            written.write_text(json.dumps(sorted(done)))
            written.replace(manifest)
            part.unlink()
            received += task[3] - task[2] + 1
            if i % 24 == 0 or i == len(tasks):
                print(f'{i}/{len(tasks)} ranges, {received/2**30:.2f} GiB in {time.monotonic()-start:.1f}s', flush=True)
        assert not failures, failures
    for name, _, checksum in selected:
        completed = dest / name
        if not completed.exists():
            raw = dest / (name + '.partial')
            assert digest(raw) == checksum, name
            raw.rename(completed)
        print(f'{checksum}  {name}', flush=True)
    marker = 'weights_verified.json' if len(selected) == len(SHARDS) else 'shards_' + ''.join(map(str, sorted(set(args.shards)))) + '_verified.json'
    (dest / marker).write_text(json.dumps({name: checksum for name, _, checksum in selected}, indent=2))


if __name__ == '__main__':
    main()
