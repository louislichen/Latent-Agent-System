"""Validate a two-by-two P/D comparison and draw its results."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


METHODS = ('latent', 'text')
LAYOUTS = ('colocated', 'split')
NAMES = ('LatentMAS', 'TextMAS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', type=Path)
    suite = parser.parse_args().suite
    pipeline = json.loads((suite / 'pipeline.json').read_text())
    config = json.loads((suite / 'config.json').read_text())
    assert pipeline['stage'] == 'complete' and pipeline['completed'] == 32
    checks = json.loads((suite / 'source_sha256.json').read_text())
    assert all(hashlib.sha256((suite / 'source' / f).read_bytes()).hexdigest() == h for f, h in checks.items())
    out = suite / 'analysis'
    out.mkdir(exist_ok=True)
    data = {}
    for method in METHODS:
        assert json.loads((suite / f'{method}_equivalence.json').read_text()) == {'cases': 8, 'equal': True}
        raw = {layout: [json.loads(line) for line in (suite / f'{method}_{layout}.jsonl').read_text().splitlines()]
               for layout in LAYOUTS}
        assert [b['cases'] for b in raw['colocated']] == [b['cases'] for b in raw['split']]
        data[method] = {}
        for layout, rows in raw.items():
            assert len(rows) == 2 and all(len(row['cases']) == 4 for row in rows)
            time_s = sum(row['service_s'] for row in rows)
            cases = [case for row in rows for case in row['cases']]
            data[method][layout] = {
                'total_s_for_8': time_s, 'ttft_p50_s': statistics.median(row['ttft_s'] for row in rows),
                'tasks_per_s': len(cases) / time_s,
                'judge_tokens_per_s': sum(c['final_output_tokens'] for c in cases) / time_s,
                'numeric_correct': sum(c['numeric_match'] for c in cases),
                'official_correct': sum(c['official_correct'] for c in cases),
                'mean_kv_handoffs_per_batch': statistics.mean(row['kv_handoffs'] for row in rows),
                'mean_logical_kv_mib_per_batch': statistics.mean(row['logical_kv_bytes'] for row in rows) / 2**20,
                'mean_prefill_synchronized_s_per_batch': statistics.mean(
                    sum(s.get('prefill_wall_s', 0) for s in row['stages']) for row in rows
                ) if layout == 'split' else None,
                'mean_kv_copy_s_per_batch': statistics.mean(sum(s.get('p_to_d_kv_copy_s', 0) +
                                                                 s.get('d_to_p_kv_copy_s', 0) for s in row['stages'])
                                                            for row in rows),
                'mean_p_to_d_kv_copy_s_per_batch': statistics.mean(
                    sum(s.get('p_to_d_kv_copy_s', 0) for s in row['stages']) for row in rows),
                'mean_d_to_p_kv_copy_s_per_batch': statistics.mean(
                    sum(s.get('d_to_p_kv_copy_s', 0) for s in row['stages']) for row in rows),
                'logical_kv_mib_p_to_d_by_role': [statistics.mean(row['stages'][i].get('p_to_d_kv_bytes', 0)
                                                                for row in rows) / 2**20 for i in range(4)],
                'logical_kv_mib_d_to_p_by_role': [statistics.mean(row['stages'][i].get('d_to_p_kv_bytes', 0)
                                                                for row in rows) / 2**20 for i in range(4)],
                'batches_s': [row['service_s'] for row in rows],
                'batches_ttft_s': [row['ttft_s'] for row in rows],
            }
            expected_handoffs = 0 if layout == 'colocated' else 7 if method == 'latent' else 4
            assert all(row['kv_handoffs'] == expected_handoffs for row in rows)
    (out / 'summary.json').write_text(json.dumps(data, indent=2))
    single = config['physical_gpus'] == 1
    placement = '单卡' if single else '双卡'
    transfer = 'CUDA IPC 访问同一设备上的 KV。' if single else (
        '两个进程通过 CUDA IPC 获取源卡张量引用，再通过 '
        + ('GPU P2P' if config['transport'] == 'device_peer' else 'CPU 中转') + '复制到目标卡。')
    report = [
        f'# {placement} {config["gpu"]} 上的 P/D 分工对照', '',
        '同一 Qwen3-4B BF16、同样 8 道 GSM8K，两个静态 B4。两种放置均加载两个模型副本。'
        '合置将整条 agent 链交给 P 进程，D 空闲。分离由 P 执行各 agent 的 prefill，D 执行完整 continuation，'
        f'包括 latent rollout 或文本生成与采样。{transfer}表中的逻辑 KV 体积按每次交接的完整状态求和。', '',
        '| 方法 | P/D 放置 | 两批总时间（s） | TTFT P50（s） | 任务/s | Judge TPS | KV 边界/批 | 逻辑 KV MiB/批 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for method, display in zip(METHODS, NAMES):
        for layout, label in zip(LAYOUTS, ('合置', '分离')):
            v = data[method][layout]
            report.append(f'| {display} | {label} | {v["total_s_for_8"]:.3f} | {v["ttft_p50_s"]:.3f} | '
                          f'{v["tasks_per_s"]:.4f} | {v["judge_tokens_per_s"]:.2f} | '
                          f'{v["mean_kv_handoffs_per_batch"]:.0f} | {v["mean_logical_kv_mib_per_batch"]:.1f} |')
    report.extend(['', '两种方法均通过 P/D 布局切换的输出等价检查。短 gate 还核对了前三个 latent 阶段的 KV SHA。'
                   '各方法的正式 8 题输出、评分和中间文本在两种放置下逐项相同。', ''])
    for method, display in zip(METHODS, NAMES):
        colocated, split = (data[method][layout] for layout in LAYOUTS)
        delta = split['total_s_for_8'] - colocated['total_s_for_8']
        report.append(f'- {display} 分离相对合置，两批总时间变化 {delta:+.3f} 秒'
                      f'（{100 * delta / colocated["total_s_for_8"]:+.2f}%）。'
                      f'分离的 P 端同步 prefill 区间为 {split["mean_prefill_synchronized_s_per_batch"]:.3f} 秒/批。'
                      + ('' if single else f'同步 KV 拷贝区间合计 {split["mean_kv_copy_s_per_batch"]:.3f} 秒/批，'
                         f'其中 P→D {split["mean_p_to_d_kv_copy_s_per_batch"]:.3f} 秒，'
                         f'D→P {split["mean_d_to_p_kv_copy_s_per_batch"]:.3f} 秒。'))
    report.extend(['',
                   'LatentMAS 的前 3 个 agent 将 rollout 后累计 KV 交回 P，由下一 agent 追加自己的 prompt。'
                   '因此四次 P→D 之外还发生三次 D→P。TextMAS 将文本传给下一个 agent，下一轮从该轮 prompt prefill，'
                   '每轮各有一次 P→D。TextMAS prompt 更长时，四次交接的总体积仍可能更大。', '',
                   ('这一轮是同卡串行机制对照。逻辑 KV 体积描述边界涉及的状态。'
                    '跨卡迁移和并发流水线收益需要双卡测量。' if single else
                    '这一轮比较同题串行关键路径。合置一张卡计算，另一张卡驻留空闲。双卡在线容量需要两个活跃合置副本，'
                    '并测相同请求到达率下的流水线、尾延迟和 SLO 吞吐。') + '每格两批用于机制观察。'])
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n')
    source = Path(__file__).read_bytes()
    (out / Path(__file__).name).write_bytes(source)
    (out / 'analysis_sha256.txt').write_text(hashlib.sha256(source).hexdigest() + '\n')
    draw(out, data, config)
    print(json.dumps(data, indent=2))


def draw(out, data, config):
    import matplotlib
    matplotlib.use('Agg')
    from latent_agent_system.analysis.plot_fonts import cjk_font_family
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.family': cjk_font_family(), 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    single = config['physical_gpus'] == 1
    fig, axes = plt.subplots(1, 3 if single else 4, figsize=(12.1 if single else 15.1, 4.45))
    fig.subplots_adjust(left=.075, right=.98, top=.74, bottom=.23, wspace=.39)
    fig.suptitle('LatentMAS 与 TextMAS 的 P/D 放置对照', y=.965, weight='bold', fontsize=17)
    colors = ('#408590', '#B86944')
    fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(colors, ('P/D 合置', 'P/D 分离'))],
               loc='upper center', bbox_to_anchor=(.5, .89), frameon=False, ncol=2)
    specs = [('两批总耗时', '秒 / 8 个任务', 'total_s_for_8', '.2f'),
             ('首个 Judge token（P50）', '秒 / 请求', 'ttft_p50_s', '.2f'),
             ('KV 交接的逻辑状态体积', 'MiB / 批', 'mean_logical_kv_mib_per_batch', '.0f')]
    for ax, (title, ylabel, key, fmt) in zip(axes, specs):
        biggest = max(data[m][layout][key] for m in METHODS for layout in LAYOUTS)
        for i, layout in enumerate(LAYOUTS):
            vals = [data[m][layout][key] for m in METHODS]
            bars = ax.bar([j + (i - .5) * .34 for j in range(2)], vals, width=.31, color=colors[i], zorder=3)
            labels = [format(v, fmt) for v in vals]
            if key == 'mean_logical_kv_mib_per_batch' and layout == 'split':
                labels = [f'{value}\n({data[m][layout]["mean_kv_handoffs_per_batch"]:.0f} 次)' for value, m in zip(labels, METHODS)]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
        ax.set_xticks([0, 1], NAMES)
        ax.set_ylim(0, biggest * 1.28)
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight='bold', fontsize=11)
        ax.grid(axis='y', color='#E6EAF0', zorder=0)
    if not single:
        ax = axes[-1]
        components = [('同步 P', 'mean_prefill_synchronized_s_per_batch', '#748092'),
                      ('P→D', 'mean_p_to_d_kv_copy_s_per_batch', '#D39169'),
                      ('D→P', 'mean_d_to_p_kv_copy_s_per_batch', '#964A34')]
        totals = [0.0, 0.0]
        for label, key, color in components:
            values = [data[m]['split'][key] for m in METHODS]
            bars = ax.bar(range(len(values)), values, bottom=totals, width=.47, color=color, label=label, zorder=3)
            ax.bar_label(bars, labels=[f'{v:.2f}' if v >= .10 else '' for v in values],
                         label_type='center', color='white', fontsize=9)
            totals = [total + val for total, val in zip(totals, values)]
        ax.set_xticks([0, 1], NAMES)
        ax.set_ylim(0, max(totals) * 1.32)
        ax.set_ylabel('秒 / B4')
        ax.set_title('分离路径的 prefill 与拷贝', weight='bold', fontsize=11)
        ax.legend(loc='upper left', frameon=False, fontsize=8, ncol=3, columnspacing=.7)
        ax.grid(axis='y', color='#E6EAF0', zorder=0)
    title = ('同一张 ' if config['physical_gpus'] == 1 else '两张 ') + config['gpu'].removeprefix('NVIDIA GeForce ').removeprefix('NVIDIA ')
    caption = f'{title}，两个模型副本，两个静态 B4。切换放置后两种方法的输出均逐题一致。'
    fig.text(.084, .09, caption,
             fontsize=9, color='#485668')
    fig.text(.084, .047, ('同卡 CUDA IPC 共享设备内存。' if config['physical_gpus'] == 1 else '分离条件执行真实跨卡 KV 拷贝。')
                        + 'KV 体积按各次交接的完整状态求和。',
             fontsize=9, color='#485668')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'pd_matrix.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
