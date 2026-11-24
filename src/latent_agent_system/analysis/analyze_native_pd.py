"""Summarize a paired native vLLM load sweep."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(run, config, reference_tokens):
    cases = run['cases']
    duration = max(c['start'] + c['wall_s'] for c in cases) - min(c['scheduled'] for c in cases)
    arrivals = [np.asarray(c['token_arrivals_s']) for c in cases]
    ttft = np.array([r[0] for r in arrivals])
    tpot = np.array([(r[-1] - r[0]) / (len(r) - 1) for r in arrivals])
    intervals = np.concatenate([np.diff(r) for r in arrivals])
    passed = int(np.count_nonzero((ttft <= config['ttft_slo_s']) & (tpot <= config['tpot_slo_s'])))
    transfer = run['transfer']
    n = transfer.get('nixl_xfer_time_seconds_count', 0)
    assert all(len(r) == len(c['token_ids']) == config['output_tokens'] and min(np.diff(r)) >= 0
               for r, c in zip(arrivals, cases, strict=True))
    return {
        'layout': run['layout'], 'offered_requests_per_s': run['rate'], 'requests': len(cases),
        'slo_attainment': passed / len(cases), 'slo_passed': passed,
        'observed_goodput_per_s': passed / duration,
        'output_tokens_per_s': sum(len(r) for r in arrivals) / duration,
        'ttft_p90_ms': float(1000 * np.percentile(ttft, 90)),
        'tpot_p90_ms': float(1000 * np.percentile(tpot, 90)),
        'itl_p99_ms': float(1000 * np.percentile(intervals, 99)),
        'arrival_jitter_p99_ms': float(1000 * np.percentile([c['start'] - c['scheduled'] for c in cases], 99)),
        'exact_match_vs_colocated': sum(c['token_ids'] == t for c, t in zip(cases, reference_tokens, strict=True)),
        'nixl_transfers': n, 'nixl_mib_per_transfer': transfer.get('nixl_bytes_transferred_sum', 0) / n / 2**20 if n else None,
        'nixl_mean_transfer_ms': transfer.get('nixl_xfer_time_seconds_sum', 0) * 1000 / n if n else None,
    }


def plot(out, rows, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.72, wspace=.35)
    fig.suptitle('Native vLLM P/D under matched text load', fontsize=17, weight='bold', y=.97)
    metrics = (('slo_attainment', 'Joint SLO attainment (%)', 100),
               ('ttft_p90_ms', 'TTFT P90 (s)', .001), ('tpot_p90_ms', 'TPOT P90 (ms)', 1))
    for layout in dict.fromkeys(r['layout'] for r in rows):
        group = sorted((r for r in rows if r['layout'] == layout), key=lambda r: r['offered_requests_per_s'])
        for ax, (metric, title, multiplier) in zip(axes, metrics, strict=True):
            ax.plot([r['offered_requests_per_s'] for r in group], [r[metric] * multiplier for r in group],
                    marker='o', label=layout)
            ax.set_title(title, weight='bold')
            ax.set_xlabel('Offered rate (requests/s)')
            ax.grid(color='#E0E5EB', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0].axhline(config['attainment_target'] * 100, color='#7E8894', linestyle='--', linewidth=1)
    axes[0].set_ylim(-3, 105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .89), frameon=False, ncol=len(labels))
    fig.text(.07, .06, f'{len(config["gpu"])} GPUs, {config["requests"]} requests per rate. '
             f'Qwen3-4B BF16, {config["prompt_tokens"]:,} input tokens, {config["output_tokens"]} output tokens. '
             f'TTFT ≤ {config["ttft_slo_s"]:g}s and TPOT ≤ {config["tpot_slo_s"] * 1000:g}ms.', color='#495766')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'goodput.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    suite = parser.parse_args().suite
    config = json.loads((suite / 'config.json').read_text())
    complete = json.loads((suite / 'complete.json').read_text())
    assert not complete['smoke']
    hashes = config['source_sha256']
    if isinstance(hashes, str):
        hashes = {'native_pd.py': hashes}
    assert all(hashlib.sha256((suite / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())
    runs = [json.loads(line) for line in (suite / 'runs.jsonl').read_text().splitlines()]
    assert len(runs) == complete['layouts'] * len(config['rates'])
    references = {r['rate']: r['cases'] for r in runs if r['layout'] == 'colocated'}
    stats = []
    for run in runs:
        baseline = references[run['rate']]
        assert len(run['cases']) == len(baseline) == config['requests']
        assert [c['index'] for c in run['cases']] == [c['index'] for c in baseline]
        gap = lambda rows: [c['scheduled'] - rows[0]['scheduled'] for c in rows]
        assert np.allclose(gap(run['cases']), gap(baseline), atol=.00001)
        stats.append(summarize(run, config, [c['token_ids'] for c in baseline]))
    capacity = {}
    for layout in dict.fromkeys(r['layout'] for r in stats):
        candidates = [r['offered_requests_per_s'] for r in stats if r['layout'] == layout
                      and r['slo_attainment'] >= config['attainment_target']]
        capacity[layout] = max(candidates, default=None)
    out = suite / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps({'config': config, 'rates': stats,
        'highest_tested_rate_meeting_target': capacity}, indent=2))
    plot(out, stats, config)
    print(json.dumps({'highest_tested_rate_meeting_target': capacity, 'rates': stats}, indent=2))


if __name__ == '__main__':
    main()
