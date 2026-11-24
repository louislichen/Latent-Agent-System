"""Summarize synchronized text-token intervals from the P/D interference probe."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    from latent_agent_system.analysis.plot_fonts import cjk_font_family
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    suite = parser.parse_args().suite
    config = json.loads((suite / 'config.json').read_text())
    complete = json.loads((suite / 'complete.json').read_text())
    source = json.loads((suite / 'source_sha256.json').read_text())
    assert all(hashlib.sha256((suite / 'source' / name).read_bytes()).hexdigest() == digest
               for name, digest in source.items())
    records = [json.loads(line) for line in (suite / 'events.jsonl').read_text().splitlines()]
    assert len(records) == complete['records'] == config['replicates'] * len(config['modes'])
    assert complete['decode_tokens_equal'] and all(r['token_ids'] == records[0]['token_ids'] for r in records)
    out = suite / 'analysis'
    out.mkdir(exist_ok=True)
    stats = {}
    for mode in config['modes']:
        rows = [r for r in records if r['mode'] == mode]
        assert len(rows) == config['replicates']
        intervals = [x for r in rows for x in r['token_intervals']]
        itl_ms = np.array([x['itl_s'] * 1000 for x in intervals])
        events = [event for r in rows for event in r['prefill_events']]
        assert len(events) == (0 if mode == 'decode_only' else len(config['arrival_decode_steps']) * len(rows))
        overlap = [
            max(0, min(e['end'], r['token_intervals'][-1]['end'])
                - max(e['start'], r['token_intervals'][0]['start']))
            for r in rows for e in r['prefill_events']
        ]
        stats[mode] = {'n_token_intervals': len(intervals), 'itl_p50_ms': float(np.percentile(itl_ms, 50)),
                       'itl_p95_ms': float(np.percentile(itl_ms, 95)), 'itl_p99_ms': float(np.percentile(itl_ms, 99)),
                       'itl_max_ms': float(max(itl_ms)), 'long_prefill_completion_p50_ms': float(np.median(
                           [(e['end'] - e['start']) * 1000 for e in events])) if events else None,
                       'prefill_windows': len(events), 'prefill_windows_during_decode': sum(t > 0 for t in overlap),
                       'overlap_ms': [round(t * 1000, 3) for t in overlap]}
    (out / 'summary.json').write_text(json.dumps({'config': config, 'statistics': stats}, indent=2))
    titles = {'decode_only': '仅 decode', 'shared_full': '同卡整段 prefill',
              'shared_chunked': '同卡分块 prefill', 'isolated': '独立设备 prefill'}
    palette = {'decode_only': '#8391A2', 'shared_full': '#B56846',
               'shared_chunked': '#187D80', 'isolated': '#4F78B3'}
    plt.rcParams.update({'font.family': cjk_font_family(), 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={'width_ratios': [1.6, 1]})
    fig.subplots_adjust(left=.077, right=.98, bottom=.23, top=.74, wspace=.29)
    fig.suptitle('长 prompt 到达时，文本 decode 会停多久？', weight='bold', fontsize=17, y=.965)
    for mode in config['modes']:
        record = next(r for r in records if r['mode'] == mode and r['replicate'] == 0)
        left.plot([x['step'] for x in record['token_intervals']], [x['itl_s'] * 1000 for x in record['token_intervals']],
                  color=palette[mode], label=titles[mode], linewidth=1.35)
    for arrival in config['arrival_decode_steps']:
        left.axvline(arrival, color='#ADB6C3', linewidth=.8, linestyle='--', zorder=0)
    left.set_ylabel('相邻 token 间隔（ms）')
    left.set_xlabel('Decode 步数')
    left.set_title('一次运行的逐 token 时间线', weight='bold')
    left.grid(axis='y', color='#E6EAF0', zorder=0)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .895), frameon=False, ncol=len(labels))
    vals = [stats[mode]['itl_p99_ms'] for mode in config['modes']]
    bars = right.bar(range(len(vals)), vals, width=.63, color=[palette[m] for m in config['modes']], zorder=3)
    right.bar_label(bars, labels=[f'{x:.1f}' for x in vals], padding=4)
    right.set_ylim(0, max(vals) * 1.23)
    right.set_xticks(range(len(vals)), [titles[m].replace(' prefill', '\nprefill') for m in config['modes']])
    right.set_ylabel('相邻 token 间隔（ms）')
    right.set_title('三次运行的 ITL P99', weight='bold')
    right.grid(axis='y', color='#E6EAF0', zorder=0)
    fig.text(.077, .082, f'同一 Qwen3-4B BF16，长 prompt {config["long_tokens"]:,} token，分块 {config["chunk_tokens"]:,} token。'
                         '虚线为长请求到达。', fontsize=9, color='#485668')
    fig.text(.077, .041, '控制实验每次生成 100 个后续 token。长请求在 prefill 完成后结束。',
             fontsize=9, color='#485668')
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'text_prefill_interference.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
