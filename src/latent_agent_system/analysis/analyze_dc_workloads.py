"""Validate the paired serving runs and plot load-sensitive metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


METHODS = ('latent', 'text')
SCENARIOS = ('serial1', 'batch4', 'burst4')
LABELS = {'serial1': '串行 B1', 'batch4': '静态批 B4', 'burst4': '突发 FIFO B≤4'}
NAMES = {'latent': 'LatentMAS', 'text': 'TextMAS'}
COLORS = {'latent': '#187D80', 'text': '#C26B37'}


def read(path):
    return json.loads(path.read_text())


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, p):
    """Linear interpolation, consistent with NumPy's default percentile."""
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    k = int(index)
    return sorted_values[k] + (index - k) * (sorted_values[min(k + 1, len(sorted_values) - 1)] - sorted_values[k])


def summarize(batch_rows, cases, duration):
    correct = [c for c in cases if c['numeric_match'] and c['finish_reason'] == 'stop']
    final = sum(c['final_output_tokens'] for c in cases)
    internal = sum(c['intermediate_text_tokens'] for c in cases)
    result = {
        'n': len(cases), 'batches': len(batch_rows), 'workload_duration_s': duration,
        'requests_s': len(cases) / duration, 'judge_output_tps': final / duration,
        'all_text_output_tps': (final + internal) / duration,
        'correct_requests_s': len(correct) / duration,
        'numeric_correct': sum(c['numeric_match'] for c in cases),
        'normally_completed_numeric_correct': len(correct),
        'official_correct': sum(c['official_correct'] for c in cases),
        'judge_capped': sum(c['finish_reason'] == 'length' for c in cases),
        'judge_tokens': final, 'internal_text_tokens': internal,
        'latent_steps': sum(c['latent_rollout_steps'] for c in cases),
        'peak_allocated_gib': max(b['peak_allocated_bytes'] for b in batch_rows) / 2**30,
        'max_arrival_jitter_ms': 1000 * max(abs(c['arrival'] - c['scheduled_arrival']) for c in cases),
        'judge_post_first_batch_good_tokens_s': sum(max(c['final_output_tokens'] - 1, 0) for c in cases) /
            sum(b['finish'] - b['stages'][-1]['first_token'] for b in batch_rows),
        'mean_text_stage_padded_slots_per_task': sum(
            b['size'] * stage.get('executed_batch_decode_steps', 0) for b in batch_rows for stage in b['stages']) / len(cases),
    }
    for key in ('queue_s', 'ttft_s', 'service_ttft_s', 'e2e_s', 'judge_after_first_s', 'service_s'):
        values = [c[key] for c in cases]
        result[f'{key}_avg'] = mean(values)
        result[f'{key}_p50'] = percentile(values, .50)
        result[f'{key}_p95'] = percentile(values, .95)
    result['mean_stage_wall_s_per_request'] = [
        sum(b['size'] * b['stages'][i]['wall_s'] for b in batch_rows) / len(cases) for i in range(4)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--partial', action='store_true', help='Check only fully available pairs without exporting plots.')
    args = parser.parse_args()
    suite = args.suite
    plan = read(suite / 'plan.json')
    pipeline = read(suite / 'pipeline.json')
    if not args.partial:
        assert pipeline['stage'] == 'complete' and len(pipeline['jobs']) == 4
        assert all(j['returncode'] == 0 for j in pipeline['jobs'])
    content = read(suite / 'source_sha256.json')
    assert all(hashlib.sha256((suite / 'source' / n).read_bytes()).hexdigest() == h for n, h in content.items())
    data, batches_all, cases_all, blocks_summary, grouping = {}, [], [], [], []
    configs = {}
    for method in METHODS:
        for block in range(2):
            workdir = suite / f'{method}_block{block}'
            if not (workdir / 'config.json').is_file():
                if args.partial:
                    continue
                raise RuntimeError(f'Missing config {workdir}')
            config = read(workdir / 'config.json')
            configs[method, block] = config
            assert config['observed_vs_raw_gate_equal']
            for scenario in SCENARIOS:
                file = workdir / f'{scenario}.jsonl'
                if args.partial and not file.is_file():
                    continue
                rows = [json.loads(line) for line in file.read_text().splitlines() if line.endswith('}')]
                flat = [case for row in rows for case in row['cases']]
                if args.partial and len(flat) != len(plan['blocks'][block]):
                    continue
                assert len(flat) == len(plan['blocks'][block])
                ids = tuple(c['dataset_index'] for c in flat)
                assert len(set(ids)) == len(flat)
                if scenario == 'serial1':
                    assert all(b['size'] == 1 for b in rows)
                elif scenario == 'batch4':
                    assert all(b['size'] == 4 for b in rows)
                else:
                    assert all(1 <= b['size'] <= 4 for b in rows)
                    offsets = [round(c['scheduled_arrival'] - flat[0]['scheduled_arrival'], 2) for c in flat]
                    assert offsets == [0] * 6 + [8] * 2, offsets
                for b in rows:
                    for c in b['cases']:
                        assert -.001 <= c['queue_s'] <= c['ttft_s'] <= c['e2e_s']
                        c.update(method=method, block=block, scenario=scenario, batch=b['batch'], batch_size=b['size'],
                                 service_s=b['service_s'], judge_after_first_s=c['e2e_s'] - c['ttft_s'])
                duration = max(c['finish'] for c in flat) - min(c['arrival'] for c in flat)
                data[method, block, scenario] = (rows, flat, duration, ids)

    comparison_fields = ('config_sha256', 'gpu_name', 'gpu_total_bytes', 'torch', 'transformers', 'model_dtype',
                         'padding_side', 'vocab', 'top_k', 'temperature', 'top_p', 'seed_per_scenario', 'judge_max_tokens')
    assert all(len({cfg[field] for cfg in configs.values()}) <= 1 for field in comparison_fields)
    summary = []
    for scenario in SCENARIOS:
        valid = [b for b in range(2) if all((m, b, scenario) in data for m in METHODS)]
        if not args.partial:
            assert valid == [0, 1]
        if not valid:
            continue
        for b in valid:
            assert data['latent', b, scenario][3] == data['text', b, scenario][3]
        for method in METHODS:
            rows, cases, duration = [], [], 0
            for block in valid:
                bb, cc, d, _ = data[method, block, scenario]
                rows.extend(bb)
                cases.extend(cc)
                duration += d
                blocks_summary.append(dict(method=method, block=block, scenario=scenario, **summarize(bb, cc, d)))
                for r in bb:
                    batches_all.append(dict(method=method, block=block, scenario=scenario, batch=r['batch'],
                                            size=r['size'], service_s=r['service_s'], peak_allocated_bytes=r['peak_allocated_bytes'],
                                            **{stage['role'] + '_wall_s': stage['wall_s'] for stage in r['stages']}))
                cases_all.extend({k: v for k, v in c.items() if k not in ('token_ids', 'text')} for c in cc)
                if scenario == 'burst4' and (method, block, 'batch4') in data:
                    ref_rows, ref_cases, _, _ = data[method, block, 'batch4']
                    assert [c['dataset_index'] for c in cc] == [c['dataset_index'] for c in ref_cases]
                    grouping.append(dict(method=method, block=block, burst_batch_sizes=[r['size'] for r in bb],
                                         same_grouping=[r['size'] for r in bb] == [r['size'] for r in ref_rows],
                                         exact_answer_tokens=sum(c['token_ids'] == a['token_ids'] for c, a in zip(cc, ref_cases))))
            summary.append(dict(method=method, scenario=scenario, blocks=valid, **summarize(rows, cases, duration)))
    if args.partial:
        for r in summary:
            print(json.dumps({k: r[k] for k in ('method', 'scenario', 'n', 'ttft_s_p50', 'e2e_s_p50',
                                                'requests_s', 'judge_output_tps', 'numeric_correct')}))
        return

    out = suite / 'analysis'
    out.mkdir(exist_ok=True)
    result = {'summary': summary, 'block_summary': blocks_summary, 'batch_vs_burst': grouping,
              'configs': list(configs.values()), 'plan': plan}
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    save_csv(out / 'metrics.csv', summary)
    save_csv(out / 'per_block.csv', blocks_summary)
    save_csv(out / 'requests.csv', cases_all)
    save_csv(out / 'batches.csv', batches_all)
    source = Path(__file__).read_bytes()
    (out / Path(__file__).name).write_bytes(source)
    (out / 'analysis_sha256.txt').write_text(hashlib.sha256(source).hexdigest() + '\n')
    draw_figures(out, summary, plan)
    write_report(out, result)
    print(json.dumps(result, indent=2))


def write_report(out, result):
    rows = result['summary']
    by_key = {(r['method'], r['scenario']): r for r in rows}
    cfg = result['configs'][0]
    lines = [
        '# LatentMAS 与 TextMAS 的负载对照', '',
        f'两种方法使用同一 Qwen3-4B checkpoint、{cfg["model_dtype"]} 和官方四角色顺序实现。'
        f'每个条件使用同样的 16 道 GSM8K，运行在 {cfg["gpu_name"]}。'
        'Judge 上限均为 1024 token，TextMAS 前三阶段上限各为 1024 token，LatentMAS 前三阶段各执行 40 步。', '',
        '| 负载 | 方法 | TTFT P50 / P95（s） | 完成时延 P50 / P95（s） | 任务/s/GPU | Judge TPS/GPU | 数值正确 | 截断 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(f'| {LABELS[row["scenario"]]} | {NAMES[row["method"]]} | '
                     f'{row["ttft_s_p50"]:.2f} / {row["ttft_s_p95"]:.2f} | '
                     f'{row["e2e_s_p50"]:.2f} / {row["e2e_s_p95"]:.2f} | '
                     f'{row["requests_s"]:.3f} | {row["judge_output_tps"]:.1f} | '
                     f'{row["numeric_correct"]}/{row["n"]} | {row["judge_capped"]} |')
    lines.append('')
    for method in METHODS:
        serial, batch, burst = (by_key[method, s] for s in SCENARIOS)
        lines.append(f'- {NAMES[method]} 从 B1 到 B4 的任务吞吐比为 '
                     f'{batch["requests_s"]/serial["requests_s"]:.2f}×。B4 TTFT P50 为 '
                     f'{batch["ttft_s_p50"]:.2f} 秒，突发条件为 {burst["ttft_s_p50"]:.2f} 秒，'
                     f'突发平均排队 {burst["queue_s_avg"]:.2f} 秒。')
    latent, text = (by_key[m, 'batch4'] for m in METHODS)
    lines.extend(['',
                  f'静态 B4 下前三个 agent 合计均时分别为 '
                  f'{sum(latent["mean_stage_wall_s_per_request"][:3]):.2f} 和 '
                  f'{sum(text["mean_stage_wall_s_per_request"][:3]):.2f} 秒（LatentMAS、TextMAS）。'
                  f'Judge 阶段均时为 {latent["mean_stage_wall_s_per_request"][3]:.2f} 和 '
                  f'{text["mean_stage_wall_s_per_request"][3]:.2f} 秒。'
                  f'按正常完成且数值正确计，任务吞吐为 {latent["correct_requests_s"]:.3f} 和 '
                  f'{text["correct_requests_s"]:.3f} 任务/秒。'])
    if all(r['same_grouping'] and r['exact_answer_tokens'] == 8 for r in result['batch_vs_burst']):
        lines.append('静态 B4 和突发条件的分批及生成的 Judge token 逐题一致。')
    lines.extend(['', 'TTFT 从服务端入队到 Judge 第一个采样 token，涵盖前三个 agent。完成时间记录在整批返回时。'
                  'TPS 用实际生成的 Judge token 计数除以从首个到达到最后完成的时间，去除 EOS 及后续 padding。'
                  '吞吐的分母合计两个独立区组的持续时间。任务吞吐和正确任务吞吐可在 `metrics.csv` 查看。', '',
                  '静态 B4 仅在前一批完成时提交下一批。突发使用独立到达线程，6 个请求同时进入，8 秒后再进入 2 个。'
                  '每个副本采用整条 agent 链的 FIFO microbatch，条件间保留官方 padding。'
                  'P95 是 16 题的描述统计。图表的范围是单 GPU 推理队列，跨 GPU P/D 流水线和网络交接属于不同的测量。', '',
                  '- `dc_latency_throughput.png/pdf` 展示时延和吞吐。',
                  '- `dc_latency_token_breakdown.png/pdf` 展示队列、服务开销和生成长度。',
                  '- `dc_quality_goodput.png/pdf` 展示正常完成的正确数和正确任务吞吐。',
                  '- `metrics.csv`、`requests.csv`、`batches.csv` 和 `summary.json` 保留聚合及逐条数据。'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def draw_figures(out, summary, plan):
    import matplotlib
    matplotlib.use('Agg')
    from latent_agent_system.analysis.plot_fonts import cjk_font_family
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    plt.rcParams.update({'font.family': cjk_font_family(), 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    lookup = {(r['method'], r['scenario']): r for r in summary}
    positions = np.arange(3)
    width = .34

    def export(fig, filename):
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{filename}.{ext}', dpi=180, facecolor='white')
        plt.close(fig)

    def grouped(ax, field, ylabel, title, p95=None, number_format='.1f'):
        values_all = [r[k] for r in summary for k in (field, p95) if k]
        maximum = max(values_all)
        for i, method in enumerate(METHODS):
            records = [lookup[method, s] for s in SCENARIOS]
            x = positions + (i - .5) * width
            v = [r[field] for r in records]
            bars = ax.bar(x, v, width=width * .9, color=COLORS[method], zorder=3)
            tops = v
            if p95:
                tail = [r[p95] for r in records]
                ax.scatter(x, tail, marker='D', s=22, facecolor='white', edgecolor=COLORS[method], zorder=4)
                tops = [max(value, high) for value, high in zip(v, tail)]
            for bar, value, top in zip(bars, v, tops):
                short = value < maximum * .075
                ax.text(bar.get_x() + bar.get_width() / 2,
                        top + maximum * .026 if short else value / 2,
                        format(value, number_format), ha='center', va='bottom' if short else 'center',
                        fontsize=9, color=COLORS[method] if short else 'white', weight='bold', zorder=5)
        ax.set_xticks(positions, [LABELS[s] for s in SCENARIOS])
        ax.set_ylim(0, maximum * 1.20)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', weight='bold')
        ax.grid(axis='y', color='#E6EAF0', zorder=0)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0))
    fig.subplots_adjust(left=.094, right=.98, top=.82, bottom=.13, wspace=.31, hspace=.53)
    fig.suptitle('同一 Qwen3-4B 下的 LatentMAS 与 TextMAS', y=.975, weight='bold', fontsize=18)
    bars = [Patch(facecolor=COLORS[m], label=NAMES[m]) for m in METHODS]
    fig.legend(handles=bars + [plt.Line2D([0], [0], linestyle='none', marker='D', markerfacecolor='white',
                                        markeredgecolor='#46505C', label='时延 P95')],
               loc='upper center', bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
    grouped(axes[0, 0], 'ttft_s_p50', '秒 / 请求', '到首个 Judge token（P50）', p95='ttft_s_p95')
    grouped(axes[0, 1], 'e2e_s_p50', '秒 / 请求', '到任务完成（P50）', p95='e2e_s_p95')
    grouped(axes[1, 0], 'requests_s', '任务 / 秒 / GPU', '任务吞吐', number_format='.3f')
    grouped(axes[1, 1], 'judge_output_tps', 'token / 秒 / GPU', 'Judge 输出 TPS')
    gpu_name = next(iter(plan['gpus'].values()))[1].removeprefix('NVIDIA GeForce ').removeprefix('NVIDIA ')
    gpu_note = f'两个区组交换 {gpu_name} 卡' if len(plan['gpus']) == 2 else f'同一张 {gpu_name} 卡'
    fig.text(.094, .058, f'每个条件 16 道同题 GSM8K，{gpu_note}。突发为 6 个同时到达，8 秒后再到 2 个。',
             fontsize=9, color='#485668')
    fig.text(.094, .028, '服务器计时含排队和内部 agent 执行。任务在整批返回时记为完成。菱形为样本 P95。',
             fontsize=9, color='#485668')
    export(fig, 'dc_latency_throughput')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.22, wspace=.27)
    fig.suptitle('队列、执行和生成长度如何影响对比', y=.97, weight='bold', fontsize=17)
    qcolors = ('#B3BCCC', '#5B82AF', '#E0AC55')
    ax = axes[0]
    keys = [(scenario, method) for scenario in SCENARIOS for method in METHODS]
    group_x = [p + (i - .5) * width for p in positions for i in range(2)]
    bottom = np.zeros(len(keys))
    for name, field, color in zip(('排队', '执行至首 token', '首 token 后'),
                                  ('queue_s_avg', 'service_ttft_s_avg', 'judge_after_first_s_avg'), qcolors):
        v = [lookup[m, s][field] for s, m in keys]
        ax.bar(group_x, v, width=width * .88, bottom=bottom, color=color, label=name, zorder=3)
        bottom += np.asarray(v)
    for x, end, (_, method) in zip(group_x, bottom, keys):
        ax.text(x, end + max(bottom) * .025, 'L' if method == 'latent' else 'T', color=COLORS[method],
                fontsize=10, ha='center', weight='bold')
    ax.set_ylim(0, max(bottom) * 1.17)
    ax.set_xticks(positions, [LABELS[s] for s in SCENARIOS])
    ax.set_ylabel('平均秒 / 请求')
    ax.set_title('端到端时延分解（L=LatentMAS，T=TextMAS）', loc='left', weight='bold')
    ax.grid(axis='y', color='#E6EAF0', zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.11), ncol=3, fontsize=9, frameon=False)

    ax = axes[1]
    bottom = np.zeros(len(keys))
    for name, field, color in (('Judge 输出', 'judge_tokens', '#5B82AF'), ('中间文本', 'internal_text_tokens', '#E0AC55')):
        v = [lookup[m, s][field] / lookup[m, s]['n'] for s, m in keys]
        ax.bar(group_x, v, width=width * .88, bottom=bottom, color=color, label=name, zorder=3)
        bottom += np.asarray(v)
    for x, end, (_, method) in zip(group_x, bottom, keys):
        ax.text(x, end + max(bottom) * .025, 'L' if method == 'latent' else 'T', color=COLORS[method],
                fontsize=10, ha='center', weight='bold')
    ax.set_ylim(0, max(bottom) * 1.17)
    ax.set_xticks(positions, [LABELS[s] for s in SCENARIOS])
    ax.set_ylabel('平均生成 token / 请求')
    ax.set_title('生成的文本量', loc='left', weight='bold')
    ax.grid(axis='y', color='#E6EAF0', zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.11), ncol=2, fontsize=9, frameon=False)
    fig.text(.075, .045, 'LatentMAS 每题有 120 步中间 latent rollout。右图的单位是文本 token。', fontsize=9, color='#485668')
    export(fig, 'dc_latency_token_breakdown')

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.subplots_adjust(left=.091, right=.98, top=.72, bottom=.22, wspace=.30)
    fig.suptitle('答题质量与正确任务吞吐', y=.965, weight='bold', fontsize=17)
    fig.legend(handles=bars, loc='upper center', bbox_to_anchor=(.5, .895), ncol=2, frameon=False)
    grouped(axes[0], 'normally_completed_numeric_correct', '正确完成数 / 16 题', '正常完成且数值正确的任务',
            number_format='.0f')
    grouped(axes[1], 'correct_requests_s', '正确任务 / 秒 / GPU', '正确任务吞吐', number_format='.3f')
    fig.text(.091, .055, '以 Judge 正常结束且最终数值与答案匹配计为正确完成。每个条件使用同样的 16 道题。',
             fontsize=9, color='#485668')
    export(fig, 'dc_quality_goodput')


if __name__ == '__main__':
    main()
