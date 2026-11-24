"""CPU-only analysis of remote, semantically validated LatentMAS profiles.

Run on the data disk; do not load weights or allocate CUDA memory.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def seconds(event):
    return (event['end_ns'] - event['start_ns']) / 1e9


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def analyze_run(path, mode, wall_key):
    rows = read_jsonl(path / (mode + '.jsonl'))
    events = []
    pipeline = json.loads((path.parent / 'pipeline.json').read_text())
    label = path.name if mode in ('reference', 'pd') else path.name.rsplit('_', 1)[0]
    parent_pid = next(j['pid'] for j in pipeline['jobs'] if j['label'] == label and j['mode'] == mode)
    worker_file = path / ('decode_worker.json' if mode == 'pd' else mode + '_worker.json')
    child_pid = json.loads(worker_file.read_text())['pid'] if worker_file.exists() else None
    for file in sorted((path / 'profile').glob('events-*.jsonl')):
        if file.name != f'events-P-{parent_pid}.jsonl' and file.name != f'events-D-{child_pid}.jsonl':
            continue
        ev = read_jsonl(file)
        child_time = defaultdict(float)
        for e in ev:
            if e['parent'] is not None:
                child_time[e['parent']] += seconds(e)
        for e in ev:
            e['cpu_exclusive_s'] = seconds(e) - child_time[e['id']]
            assert e['cpu_exclusive_s'] > -1e-6, e
        events.extend(ev)
    groups = defaultdict(list)
    for e in events:
        if e['batch'] >= 0:
            groups[(e['side'], e['role'], e['phase'], e['name'])].append(e)
    spans = []
    for (side, role, phase, name), ee in groups.items():
        spans.append({'run': str(path.name), 'mode': mode, 'side': side, 'role': role,
                      'phase': phase, 'name': name, 'count': len(ee),
                      'cpu_inclusive_s': sum(seconds(e) for e in ee),
                      'cpu_exclusive_s': sum(e['cpu_exclusive_s'] for e in ee),
                      'cuda_stream_span_s': sum(e.get('cuda_stream_span_ms', 0) / 1000 for e in ee)})
    batches = []
    for ordinal, row in enumerate(rows):
        ev = [e for e in events if e['batch'] == ordinal]
        pe = [e for e in ev if e['side'] == 'P' and e['name'] == 'prepare_chat']
        pe.sort(key=lambda e: e['start_ns'])
        assert len(pe) == 4, (path, ordinal, len(pe))
        latent_wall = (pe[3]['start_ns'] - pe[0]['start_ns']) / 1e9
        forwards = [e for e in ev if e['name'] == 'model.forward']
        br = {'run': path.name, 'mode': mode, 'batch': row['batch'], 'ordinal': ordinal,
              'wall_s': row[wall_key], 'latent_roles_wall_s': latent_wall,
              'judge_and_tail_wall_s': row[wall_key] - latent_wall,
              'handoffs': len(row.get('handoffs', [])),
              'logical_handoff_bytes': sum(h['bytes'] for h in row.get('handoffs', []))}
        for phase in ('prefill', 'latent_decode', 'text_decode'):
            ee = [e for e in forwards if e['phase'] == phase]
            br[phase + '_calls'] = len(ee)
            br[phase + '_host_s'] = sum(seconds(e) for e in ee)
            br[phase + '_cuda_stream_s'] = sum(e['cuda_stream_span_ms'] / 1000 for e in ee)
        for name in ('lm_head', 'latent.alignment', 'generation.sample', 'validation.cache_hash'):
            ee = [e for e in ev if e['name'] == name]
            br[name + '_host_s'] = sum(seconds(e) for e in ee)
        g = row.get('generation', {})
        for key in ('forward_count', 'rpc_wall_s', 'D_forward_wall_s'):
            if key in g:
                br[key] = g[key]
        batches.append(br)
    total = {k: sum(b.get(k, 0) for b in batches) for k in batches[0]
             if k not in ('run', 'mode', 'batch', 'ordinal')}
    total['n_batches'] = len(rows)
    total['n_cases'] = sum(len(row['cases']) for row in rows)
    total['latent_wall_fraction'] = total['latent_roles_wall_s'] / total['wall_s']
    total['max_process_allocated_bytes'] = max((e.get('memory_after', {}).get('peak_allocated', 0)
                                               for e in events), default=0)
    return rows, events, spans, batches, total


def kernel_windows(directory):
    result = []
    for path in sorted(directory.glob('kineto-*.json')):
        if path.name.endswith('-ops.json'):
            continue
        events = json.loads(path.read_text())['traceEvents']
        kernels = [e for e in events if e.get('cat') == 'kernel']
        heads = [e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'aten::mm'
                 and '151669' in str(e.get('args', {}).get('Input Dims'))]
        correlations = set()
        for h in heads:
            correlations.update(e.get('args', {}).get('correlation') for e in events
                                if e.get('cat') in ('cuda_runtime', 'cuda_driver')
                                and e.get('pid') == h.get('pid') and e.get('tid') == h.get('tid')
                                and h['ts'] <= e.get('ts', 0) <= h['ts'] + h['dur'])
        head_kernels = [e for e in kernels if e.get('args', {}).get('correlation') in correlations]
        total = sum(e['dur'] for e in kernels)
        head_total = sum(e['dur'] for e in head_kernels)
        top = Counter()
        for e in kernels:
            top[e['name']] += e['dur']
        result.append({'file': path.name, 'cpu_pid': heads[0]['pid'] if heads else None,
                       'kernel_count': len(kernels), 'unique_kernel_us': total,
                       'head_kernel_us': head_total,
                       'head_kernel_fraction': head_total / total if total else 0,
                       'head_matmul_shapes': [h.get('args', {}).get('Input Dims') for h in heads],
                       'top_kernels_us': top.most_common(5)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    results = root / 'results'
    out = results / 'profiling_20261116_v1/analysis'
    out.mkdir(exist_ok=True)
    provenance = {}
    for name in ('profiling_20261116_v1', 'profiling_routes_20261116_v1', 'profiling_head_20261116_v1'):
        suite = results / name
        pipeline = json.loads((suite / 'pipeline.json').read_text())
        assert pipeline['stage'] == 'complete' and all(j['returncode'] == 0 for j in pipeline['jobs']), name
        manifest = json.loads((suite / 'source_sha256.json').read_text())
        assert all(hashlib.sha256((suite / 'source' / k).read_bytes()).hexdigest() == v
                   for k, v in manifest.items()), name
        provenance[name] = {'jobs_completed': len(pipeline['jobs']), 'source_manifest_verified': True}
    runs = {}
    spans, batches, handoffs, states, all_events = [], [], [], [], []
    configs = []
    for label in ('timing_hash', 'timing_nohash'):
        for mode in ('reference', 'pd'):
            configs.append((f'existing/{label}/{mode}', results / 'profiling_20261116_v1' / label,
                            mode, 'end_to_end_s_with_observer'))
    for mode in ('affinity', 'mixed', 'split'):
        configs.append((f'routes/{mode}', results / 'profiling_routes_20261116_v1' / ('timing_' + mode),
                        mode, 'wall_s'))
    for mode in ('reference', 'pd'):
        configs.append((f'no_unused_head/{mode}', results / 'profiling_head_20261116_v1/timing',
                        mode, 'end_to_end_s_with_observer'))
    reference = None
    for label, path, mode, wall_key in configs:
        rows, ev, ss, bb, total = analyze_run(path, mode, wall_key)
        cases = [r['cases'] for r in rows]
        if reference is None:
            reference = cases
        assert cases == reference, label
        runs[label] = total
        for records in (ss, bb):
            for rec in records:
                rec['label'] = label
        spans.extend(ss)
        batches.extend(bb)
        for e in ev:
            e['run_label'] = label
        all_events.extend(ev)
        for row in rows:
            for h in row.get('handoffs', []):
                handoffs.append({'label': label, 'batch': row['batch'], **h})
            for state in row.get('latent_stage_states', []):
                states.append({'label': label, 'batch': row['batch'], 'role': state['role'], **state['state']})
    old = read_jsonl(results / 'hf_semantic100_v2/pd.jsonl')
    old_total = {'cases': sum(len(r['cases']) for r in old),
                 'judge_forward_calls_including_prefill': sum(r['generation']['forward_count'] for r in old),
                 'decode_RPC_calls': sum(r['generation']['forward_count'] - 1 for r in old),
                 'rpc_wall_s': sum(r['generation']['rpc_wall_s'] for r in old),
                 'D_forward_wall_s': sum(r['generation']['D_forward_wall_s'] for r in old)}
    old_total['rpc_residual_s'] = old_total['rpc_wall_s'] - old_total['D_forward_wall_s']
    windows = kernel_windows(results / 'profiling_20261116_v1/kernel_windows/profile')
    head_control = {}
    for mode in ('reference', 'pd'):
        before = runs[f'existing/timing_nohash/{mode}']
        after = runs[f'no_unused_head/{mode}']
        head_control[mode] = {'before_latent_wall_s': before['latent_roles_wall_s'],
                              'after_latent_wall_s': after['latent_roles_wall_s'],
                              'before_wall_s': before['wall_s'], 'after_wall_s': after['wall_s'],
                              'note': 'Single-pass instrumented timing; not a statistical speedup claim.'}
    raw_gate = read_jsonl(results / 'profiling_20261116_v1/raw_gate/reference.jsonl')[0]
    gate_checks = {}
    for mode in ('affinity', 'mixed', 'split'):
        row = read_jsonl(results / 'profiling_routes_20261116_v1' / ('gate_' + mode) / (mode + '.jsonl'))[0]
        affinity = read_jsonl(results / 'profiling_routes_20261116_v1/gate_affinity/affinity.jsonl')[0]
        assert row['cases'] == raw_gate['cases']
        assert row['latent_stage_states'] == affinity['latent_stage_states']
        gate_checks[mode] = {'case_fields_equal_raw': True, 'three_latent_KV_SHA_equal_affinity': True}
    observer_gate = {mode: read_jsonl(results / 'profiling_20261116_v1/gate' / (mode + '.jsonl'))[0]['end_to_end_s_with_observer']
                     for mode in ('reference', 'pd')}
    observer_gate['unprofiled_reference'] = raw_gate['end_to_end_s_with_observer']
    summary = {'all_timing_case_fields_equal': True, 'provenance': provenance, 'route_gate_checks': gate_checks,
               'observer_gate_wall_s_max128': observer_gate, 'runs': runs, 'head_control': head_control,
               'old_full100_audit': old_total, 'kernel_windows': windows,
               'limits': ['Single physical GPU5; CUDA IPC, no inter-GPU or network timing.',
                          'Three fixed batches / six cases, one pass per condition; no load or queue contention.',
                          'CUDA-event spans include stream idle; raw kernel activities are counted once.',
                          'Logical handoff bytes count state boundaries, not observed physical wire bytes.',
                          'Timing reset seed17 per process; skipped batches do not consume original100 RNG.']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    csv_write(out / 'per_phase.csv', spans)
    csv_write(out / 'per_batch.csv', batches)
    csv_write(out / 'logical_handoffs.csv', handoffs)
    csv_write(out / 'latent_states.csv', states)
    # CPU traces share one monotonic host clock. Keep raw GPU traces separately.
    start = min(e['start_ns'] for e in all_events)
    trace = [{'ph': 'X', 'name': e['name'], 'cat': e['role'] + '/' + e['phase'],
              'pid': e['run_label'] + '/' + e['side'], 'tid': e['pid'],
              'ts': (e['start_ns'] - start) / 1000, 'dur': seconds(e) * 1e6,
              'args': {'batch_ordinal': e['batch'], 'cuda_stream_span_ms': e.get('cuda_stream_span_ms')}}
             for e in all_events]
    (out / 'host_timeline.json').write_text(json.dumps({'traceEvents': trace}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    labels = ['Existing\ncolocated', 'Existing\nP/D RPC', 'State\naffinity', 'Agent\nhandoff', 'Phase\nsplit']
    keys = ['existing/timing_nohash/reference', 'existing/timing_nohash/pd',
            'routes/affinity', 'routes/mixed', 'routes/split']
    latent = [runs[k]['latent_roles_wall_s'] for k in keys]
    judge = [runs[k]['judge_and_tail_wall_s'] for k in keys]
    axes[0].bar(labels, latent, label='3 latent agents', color='#459c9a')
    axes[0].bar(labels, judge, bottom=latent, label='Judge + tail', color='#596bb0')
    axes[0].set_ylabel('Host wall time over 3 batches (s)')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Observed critical path (same GPU)')
    routekeys = ['routes/affinity', 'routes/mixed', 'routes/split']
    traffic = [runs[k]['logical_handoff_bytes'] / 2**20 / 3 for k in routekeys]
    axes[1].bar(['Affinity\n0 / batch', 'Agent\n3 / batch', 'Phase\n7 / batch'], traffic, color='#c58351')
    axes[1].set_ylabel('Mean logical KV handoff volume (MiB / batch)')
    axes[1].set_title('State boundaries; not network traffic')
    for i, value in enumerate(traffic):
        axes[1].text(i, value + 10, f'{value:.1f}', ha='center')
    groups = defaultdict(list)
    for h in handoffs:
        if h['label'] == 'routes/split' and h['reason'] == 'prefill_result':
            groups[h['role']].append(h['bytes'] / 2**20)
    roles = ['planner', 'critic', 'refiner', 'judger']
    means = [sum(groups[r]) / len(groups[r]) for r in roles]
    axes[2].plot(roles, means, 'o-', color='#a75d71', label='Full KV after prefill')
    axes[2].axhline(40 * 2 * 36 * 2 * 8 * 128 * 2 / 2**20,
                    color='#459c9a', linestyle='--', label='40 latent positions (11.25 MiB)')
    axes[2].set_ylabel('Logical state (MiB / batch of 2)')
    axes[2].set_title('Fixed rollout, growing continuation state')
    axes[2].legend(fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('profile_overview.' + ext), dpi=180)
    plt.close(fig)
    source = out / 'analysis_source'
    source.mkdir(exist_ok=True)
    dst = source / Path(__file__).name
    dst.write_bytes(Path(__file__).read_bytes())
    (out / 'analysis_source_sha256.json').write_text(json.dumps({dst.name: hashlib.sha256(dst.read_bytes()).hexdigest()}, indent=2))
    print(json.dumps({'runs': runs, 'head_control': head_control,
                      'kernel_windows': [{k: v for k, v in w.items() if k != 'top_kernels_us'} for w in windows]}, indent=2))


if __name__ == '__main__':
    main()
