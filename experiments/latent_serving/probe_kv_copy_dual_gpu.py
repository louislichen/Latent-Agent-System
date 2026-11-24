"""Measure two-card KV copy component using concrete sizes from the H100 B4 run."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

from benchmark_pd_matrix import compute_pids, relocate_tensor


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('shape_source', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(set(uuids)) == 2

    def assert_exclusive():
        assert not set().union(*(compute_pids(u) for u in uuids)) - {os.getpid()}

    assert_exclusive()
    assert torch.cuda.device_count() == 2
    devices = [torch.device('cuda', i) for i in range(2)]
    peer = [torch.cuda.can_device_access_peer(a, b) for a, b in ((0, 1), (1, 0))]
    route = 'device_peer' if all(peer) else 'host_staged'
    args.out.mkdir(parents=True, exist_ok=False)
    measures = []
    per_position = 4 * 36 * 2 * 8 * 128 * 2
    with torch.inference_mode():
        for method in ('latent', 'text'):
            rows = [json.loads(l) for l in (args.shape_source / (method + '_split.jsonl')).read_text().splitlines()]
            for batch, row in enumerate(rows):
                for role in row['stages']:
                    for key, direction in (('p_to_d_kv_bytes', (0, 1)), ('d_to_p_kv_bytes', (1, 0))):
                        size = role.get(key, 0)
                        if not size:
                            continue
                        assert size % per_position == 0
                        source, target = [devices[d] for d in direction]
                        shape = (4, 8, size // per_position, 128)
                        tensors = [torch.full(shape, i, device=source, dtype=torch.bfloat16) for i in range(72)]
                        torch.cuda.synchronize(source)
                        times = []
                        for j in range(4):
                            start = time.perf_counter()
                            received = [relocate_tensor(tensor, target, route) for tensor in tensors]
                            torch.cuda.synchronize(target)
                            elapsed = time.perf_counter() - start
                            if j > 0:
                                times.append(elapsed)
                            assert all(t.device == target and t.shape == shape and t.dtype == torch.bfloat16 for t in received)
                            test = torch.stack([t[0, 0, 0, 0] for t in received]).cpu().tolist()
                            assert test == list(range(len(received)))
                            del received
                        measures.append({'method': method, 'batch': batch, 'role': role['role'], 'route': route,
                                         'direction': list(direction), 'logical_bytes': size,
                                         'times_s': times, 'median_s': statistics.median(times)})
                        del tensors
                assert_exclusive()
    totals = {m: [sum(r['median_s'] for r in measures if r['method'] == m and r['batch'] == b)
                  for b in range(2)] for m in ('latent', 'text')}
    data = {'scope': 'synthetic KV shape replicas from the H100 agent matrix; cross-device tensor copies only',
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(2)], 'gpu_uuids': uuids, 'peer': peer,
            'route': route, 'measurements': measures, 'per_batch_sum_of_medians_s': totals}
    (args.out / 'summary.json').write_text(json.dumps(data, indent=2))
    print(json.dumps({k: data[k] for k in ('gpu_names', 'peer', 'route', 'per_batch_sum_of_medians_s')}, indent=2))


if __name__ == '__main__':
    main()
