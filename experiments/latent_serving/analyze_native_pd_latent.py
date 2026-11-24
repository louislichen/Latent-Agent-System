"""Validate hashes and summarize equal-budget native Judge replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_native_pd import summarize


KINDS = {'latent': 'LatentMAS Judge embeddings', 'matched_tokens': 'Length-matched token input'}
LAYOUTS = {'colocated': '3 colocated replicas', 'split': '1 P + 2 D'}


def plot(out, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.23, top=.72, wspace=.33)
    fig.suptitle('P/D placement for frozen LatentMAS Judge inputs', y=.96, weight='bold', fontsize=16)
    panels = [('ttft_p90_ms', 'TTFT P90 (ms)', 1), ('tpot_p90_ms', 'TPOT P90 (ms)', 1),
              ('slo_attainment', 'Joint SLO attainment (%)', 100)]
    for kind, kind_label in KINDS.items():
        for layout, layout_label in LAYOUTS.items():
            group = [r for r in stats if (r['input_kind'], r['layout']) == (kind, layout)]
            group.sort(key=lambda r: r['offered_requests_per_s'])
            for ax, (metric, title, scale) in zip(axes, panels, strict=True):
                ax.plot([r['offered_requests_per_s'] for r in group],
                        [r[metric] * scale for r in group], marker='o' if kind == 'latent' else '^',
                        linestyle='-' if kind == 'latent' else '--',
                        label=f'{kind_label}, {layout_label}',
                        color=['#316E9F', '#BD6442'][layout == 'split'])
                ax.set_title(title, weight='bold')
                ax.set_xlabel('Offered requests/s')
                ax.set_xticks(sorted({r['offered_requests_per_s'] for r in stats}))
                ax.grid(color='#E0E5EB', linewidth=.7)
                ax.spines[['top', 'right']].set_visible(False)
    axes[2].set_ylim(-3, 105)
    axes[2].axhline(90, color='#8390A2', linestyle=':', linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .885), frameon=False, ncol=2, fontsize=8)
    fig.text(.065, .066, 'Same request arrival sequence and input lengths in each pair. '
             'Joint target: TTFT ≤ 2 s and TPOT ≤ 30 ms.', color='#495766', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'latent_goodput.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    suite = parser.parse_args().suite
    cfg = json.loads((suite / 'config.json').read_text())
    done = json.loads((suite / 'complete.json').read_text())
    assert not done['smoke'] and done['count'] == cfg['count']
    assert all(hashlib.sha256((suite / name).read_bytes()).hexdigest() == digest
               for name, digest in cfg['source_sha256'].items())
    manifest = json.loads((suite / 'manifest.json').read_text())['cases']
    runs = [json.loads(line) for line in (suite / 'runs.jsonl').read_text().splitlines()]
    assert len(runs) == len(KINDS) * len(LAYOUTS) * len(cfg['rates'])
    serial = {}
    for layout in LAYOUTS:
        serial[layout] = {}
        for kind in KINDS:
            gate = json.loads((suite / layout / kind / 'serial.json').read_text())
            assert gate['exact_match'] == len(gate['cases'])
            if layout == 'split':
                assert gate['transfer']['nixl_xfer_time_seconds_count'] == len(gate['cases'])
            serial[layout][kind] = {
                'count': len(gate['cases']),
                'exact_match': gate['exact_match'],
                'ttft_p50_ms': float(np.median([c['token_arrivals_s'][0] for c in gate['cases']]) * 1000),
                'transfer': gate['transfer'],
            }
    refs = {(r['input_kind'], r['rate']): r['cases'] for r in runs if r['layout'] == 'colocated'}
    stats = []
    for run in runs:
        cases = run['cases']
        baseline = refs[run['input_kind'], run['rate']]
        assert len(cases) == len(baseline) == len(manifest)
        assert [c['input_tokens'] for c in cases] == [m['positions'] for m in manifest]
        assert np.allclose([c['scheduled'] - cases[0]['scheduled'] for c in cases],
                           [c['scheduled'] - baseline[0]['scheduled'] for c in baseline], rtol=0, atol=1e-6)
        cell = summarize(run, cfg, [c['token_ids'] for c in baseline])
        cell['input_kind'] = run['input_kind']
        cell['tpot_only_passed'] = sum(
            (c['token_arrivals_s'][-1] - c['token_arrivals_s'][0]) / (cfg['output_tokens'] - 1)
            <= cfg['tpot_slo_s'] for c in cases)
        cell['mean_base64_mib'] = float(np.mean([c['base64_bytes'] for c in cases]) / 2**20)
        if run['layout'] == 'split':
            assert cell['nixl_transfers'] == len(cases)
            cell['handoff_to_d_first_token_p90_ms'] = float(np.percentile(
                [c['token_arrivals_s'][1] - c['token_arrivals_s'][0] for c in cases], 90) * 1000)
        stats.append(cell)
    out = suite / 'analysis'
    out.mkdir(exist_ok=True)
    result = {'config': cfg, 'serial': serial, 'rates': stats,
              'prompt_lengths': [min(m['positions'] for m in manifest), max(m['positions'] for m in manifest)]}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    plot(out, stats)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
