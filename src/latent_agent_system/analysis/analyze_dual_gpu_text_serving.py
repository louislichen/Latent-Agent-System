"""Aggregate paired traces from the dual-GPU streaming comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(runs, config):
    by_run = []
    cases = [case for run in runs for case in run['cases']]
    ttft = np.array([c['stamps'][0] - c['arrival'] for c in cases])
    itls = [np.diff(c['stamps']) for c in cases]
    passed = (ttft <= config['ttft_slo_s']) & (
        np.array([x.max() for x in itls]) <= config['max_itl_slo_s'])
    for run in runs:
        rows = run['cases']
        duration = max(c['stamps'][-1] for c in rows) - min(c['arrival'] for c in rows)
        by_run.append({'replicate': run['replicate'], 'makespan_s': duration,
                       'requests_per_s': len(rows) / duration})
    copies = [c for c in cases if c['copy_started'] is not None]
    durations = [c['copy_finished'] - c['copy_started'] for c in copies]
    overlap = sum(
        any(row['stamps'][0] < c['copy_finished'] and c['copy_started'] < row['stamps'][-1]
            for row in run['cases'] if row is not c)
        for run in runs for c in run['cases'] if c['copy_started'] is not None
    )
    elapsed = sum(r['makespan_s'] for r in by_run)
    ttft_by_length = {}
    for length in sorted({c['prompt_tokens'] for c in cases}):
        rows = [c for c in cases if c['prompt_tokens'] == length]
        boundaries = [
            (c['arrival'], c['prefill_started'], c['prefill_finished'],
             c['copy_started'] if c['copy_started'] is not None else c['prefill_finished'],
             c['copy_finished'] if c['copy_finished'] is not None else c['prefill_finished'],
             c['stamps'][0])
            for c in rows
        ]
        components = np.diff(boundaries, axis=1) * 1000
        ttft_by_length[str(length)] = {
            'requests': len(rows), 'ttft_p95_ms': float(np.percentile(components.sum(axis=1), 95)),
            'mean_ms': dict(zip(('prefill_queue', 'prefill_wall', 'copy_dispatch', 'copy_wall', 'first_token_wait'),
                                components.mean(axis=0).tolist(), strict=True)),
        }
    return {
        'requests': len(cases),
        'max_arrival_jitter_ms': 1000 * max(c['arrival'] - c['scheduled'] for c in cases),
        'requests_per_s': len(cases) / elapsed,
        'output_tokens_per_s': sum(len(c['tokens']) for c in cases) / elapsed,
        'ttft_p50_ms': float(1000 * np.median(ttft)),
        'ttft_p95_ms': float(1000 * np.percentile(ttft, 95)),
        'itl_p50_ms': float(1000 * np.median(np.concatenate(itls))),
        'itl_p99_ms': float(1000 * np.percentile(np.concatenate(itls), 99)),
        'e2e_p95_ms': float(1000 * np.percentile([c['stamps'][-1] - c['arrival'] for c in cases], 95)),
        'prefill_queue_p95_ms': float(1000 * np.percentile([c['prefill_started'] - c['arrival'] for c in cases], 95)),
        'slo_pass': int(passed.sum()),
        'slo_goodput_per_s': float(passed.sum() / elapsed),
        'kv_copy_count': len(copies),
        'kv_copies_during_other_decode': overlap,
        'kv_copy_total_s': sum(durations),
        'kv_copy_mib': sum(c['kv_bytes'] for c in copies) / 1024**2,
        'long_copy_p50_ms': float(1000 * np.median(
            [d for c, d in zip(copies, durations) if c['prompt_tokens'] > 64])) if copies else None,
        'ttft_by_prompt_length': ttft_by_length,
        'replicates': by_run,
    }


def plot(out, stats, config):
    import matplotlib
    matplotlib.use('Agg')
    from latent_agent_system.analysis.plot_fonts import cjk_font_family
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': cjk_font_family(), 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.subplots_adjust(left=.058, right=.989, bottom=.22, top=.71, wspace=.34)
    fig.suptitle('两张 RTX 3090 上的并发文本服务', y=.96, fontsize=19, weight='bold')
    groups = np.arange(len(stats))
    panels = (('ttft_p95_ms', 'TTFT P95', 'ms'), ('itl_p99_ms', 'ITL P99', 'ms'),
              ('output_tokens_per_s', '输出吞吐', 'token/s'), ('slo_goodput_per_s', '达标吞吐', '请求/s'))
    for ax, (key, title, unit) in zip(axes, panels, strict=True):
        maximum = max(group[layout][key] for group in stats.values() for layout in ('unified', 'split'))
        for offset, layout, label, color in ((-.19, 'unified', '两个 P/D 合置副本', '#4F78B3'),
                                              (.19, 'split', '1 P + 1 D', '#B56846')):
            vals = [group[layout][key] for group in stats.values()]
            bars = ax.bar(groups + offset, vals, width=.33, color=color, label=label, zorder=3)
            ax.bar_label(bars, labels=[f'{v:.0f}' if unit == 'ms' else f'{v:.2f}' for v in vals], padding=3)
        ax.set_title(title, weight='bold')
        ax.set_ylabel(unit)
        ax.set_xticks(groups, ('突发到达', '间隔拉长 4 倍'))
        ax.set_ylim(0, maximum * 1.25 if maximum else 1)
        ax.grid(axis='y', color='#E6EAF0', zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .883), ncol=2, frameon=False)
    fig.text(.058, .09, f'每组 {config["replicates"]} 次，每次 12 请求。Qwen3-4B BF16，'
             '64/8,192 token 输入，固定 32 token 输出，prefill 每块 512 token。', color='#485668')
    fig.text(.058, .04, f'达标为 TTFT ≤ {config["ttft_slo_s"]:g}s 且最大 ITL ≤ '
             f'{1000 * config["max_itl_slo_s"]:g}ms，HF eager 逐请求 forward。', color='#485668')
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'dual_gpu_text_serving.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    suite = parser.parse_args().suite
    config = json.loads((suite / 'config.json').read_text())
    complete = json.loads((suite / 'complete.json').read_text())
    source = json.loads((suite / 'source_sha256.json').read_text())
    assert all(hashlib.sha256((suite / 'source' / name).read_bytes()).hexdigest() == digest
               for name, digest in source.items())
    runs = [json.loads(line) for line in (suite / 'runs.jsonl').read_text().splitlines()]
    assert len(runs) == 4 * config['replicates']
    assert sum(len(run['cases']) for run in runs) == complete['requests']
    assert complete['tokens_equal']
    paired = {(r['trace'], r['replicate'], r['layout']): r for r in runs}
    stats = {}
    for trace in ('burst', 'spread'):
        stats[trace] = {}
        for rep in range(config['replicates']):
            a, b = (paired[trace, rep, layout]['cases'] for layout in ('unified', 'split'))
            assert [(c['index'], c['tokens']) for c in a] == [(c['index'], c['tokens']) for c in b]
            for case in a + b:
                assert len(case['stamps']) == len(case['tokens']) == config['output_tokens']
                assert case['arrival'] <= case['prefill_started'] <= case['prefill_finished'] <= case['stamps'][0]
                assert min(np.diff(case['stamps'])) > 0
            assert all(c['kv_bytes'] == 0 and c['copy_started'] is None for c in a)
            assert all(c['kv_bytes'] > 0 and c['prefill_finished'] <= c['copy_started']
                       <= c['copy_finished'] <= c['stamps'][0] for c in b)
        for layout in ('unified', 'split'):
            stats[trace][layout] = summarize([paired[trace, rep, layout] for rep in range(config['replicates'])], config)
    out = suite / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps({'config': config, 'complete': complete, 'statistics': stats}, indent=2))
    plot(out, stats, config)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
