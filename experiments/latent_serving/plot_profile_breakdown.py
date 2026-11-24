"""Render measured deployment cost bars on the remote data disk, without CUDA."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    out = args.root / 'results/profiling_20261116_v1/analysis'
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_fonts import cjk_font_family
    import numpy as np

    plt.rcParams.update({
        'font.family': cjk_font_family(),
        'font.size': 11, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
        'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False,
    })
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['all_timing_case_fields_equal']
    runs = summary['runs']
    keys = ['prefill_host_s', 'latent_decode_host_s', 'text_decode_host_s']
    names = ['Prefill forward', 'Latent forward', 'Judge 文本 forward', '其他']
    colors = ['#E7B75C', '#4DABA5', '#5B72AD', '#ACB3BD', '#C77963']
    deployment_colors = ['#287D9B', '#C37543']
    rows = []

    def values(label, rpc=False):
        r = runs[label]
        vals = [r[k] for k in keys]
        if rpc:
            vals.append(r.get('rpc_wall_s', 0) - r.get('D_forward_wall_s', 0))
        residual = r['wall_s'] - sum(vals)
        assert residual >= 0, (label, residual)
        vals.append(residual)
        assert abs(sum(vals) - r['wall_s']) < 1e-9
        return vals

    def draw(stem, title, labels, run_labels, rpc=False):
        categories = names[:] if not rpc else names[:3] + ['RPC 控制等开销'] + ['其他']
        stage_colors = colors[:4] if not rpc else colors[:3] + [colors[4], colors[3]]
        vv = [values(label, rpc=rpc) for label in run_labels]
        totals = [sum(v) for v in vv]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 1.65]})
        fig.subplots_adjust(left=.065, right=.985, bottom=.30, top=.80, wspace=.28)
        fig.suptitle(title, fontsize=17, y=.965, weight='bold')
        fig.text(.5, .895, '固定 3 个 batch / 6 题 · 同 GPU CUDA IPC · 关闭 KV 哈希 · 不含模型加载',
                 ha='center', fontsize=10, color='#505866')
        ax = axes[0]
        bottoms = np.zeros(2)
        for j, (category, color) in enumerate(zip(categories, stage_colors)):
            vals = [v[j] for v in vv]
            ax.bar([0, 1], vals, bottom=bottoms, width=.48, label=category, color=color,
                   edgecolor='white', linewidth=.6, zorder=3)
            for i, value in enumerate(vals):
                if value >= 5:
                    ax.text(i, bottoms[i] + value / 2, f'{value:.2f} s', ha='center', va='center',
                            color='white' if j == 2 else '#102A33', fontsize=11, weight='bold')
            bottoms += vals
        for i, total in enumerate(totals):
            ax.text(i, total + 1.1, f'{total:.2f} s', ha='center', va='bottom', fontsize=13, weight='bold')
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-.60, 1.60)
        ax.set_ylim(0, max(totals) * 1.16)
        ax.set_ylabel('三个 batch 的累计耗时（秒）')
        ax.set_title('(a) 总耗时与组成', fontsize=13, pad=13)
        ax.grid(axis='y', color='#E8EBEF', zorder=0)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False, fontsize=9,
                  ncol=2, columnspacing=1.2, handlelength=1.25)

        ax = axes[1]
        x = np.arange(len(categories))
        for i, (vals, color, label) in enumerate(zip(vv, deployment_colors, labels)):
            bars = ax.bar(x + (i - .5) * .35, vals, width=.32, color=color,
                          label=label.replace('\n', ' '), zorder=3)
            ax.bar_label(bars, labels=[f'{v:.2f}' for v in vals], padding=4, fontsize=10)
        xticklabels = ['Prefill\nforward', 'Latent\nforward', 'Judge 文本\nforward']
        if rpc:
            xticklabels.append('RPC 控制等\n（不含 D forward）')
        xticklabels.append('其他')
        ax.set_xticks(x, xticklabels)
        ax.set_ylim(0, max(max(v) for v in vv) * 1.19)
        ax.set_ylabel('三个 batch 的累计耗时（秒）')
        ax.set_title('(b) 各部分对照', fontsize=13, pad=13)
        ax.grid(axis='y', color='#E8EBEF', zorder=0)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False, fontsize=9, ncol=1)
        note = 'forward 为主机调用区间，包含发射与同步等待；“其他”为扣余，含采样、控制、IPC 和同步。'
        if rpc:
            note = 'RPC 项为 RPC wall 减 D forward wall，含控制、IPC 和等待；不是纯网络传输时间。'
        fig.text(.065, .065, note, fontsize=9, color='#505866')
        fig.text(.065, .025, '单次测量，无误差线；跨进程嵌套等待和 CUDA 父子区间未重复相加。', fontsize=9, color='#505866')
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(out / f'{stem}.{extension}', dpi=210, facecolor='white')
        plt.close(fig)
        for label, run_label, vals in zip(labels, run_labels, vv):
            for category, value in zip(categories, vals):
                rows.append({'figure': stem, 'deployment': label.replace('\n', ' '),
                             'source_run': run_label, 'component': category, 'seconds': value,
                             'total_seconds': sum(vals)})

    draw('deployment_cost_breakdown_zh', 'LatentMAS：P/D 合置与 P/D 分离的耗时分解',
         ['P/D 合置\n+ agent 间交接', 'P/D 分离\n（整阶段执行）'],
         ['routes/mixed', 'routes/split'])
    draw('existing_prototype_cost_breakdown_zh', '原 HF 部署：逐 token RPC 的开销分解',
         ['原单实例\nP/D 合置', '原 P/D 原型\n逐 token RPC'],
         ['existing/timing_nohash/reference', 'existing/timing_nohash/pd'], rpc=True)
    with (out / 'cost_breakdown_chart_data.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    source = out / 'figure_source'
    source.mkdir(exist_ok=True)
    target = source / Path(__file__).name
    target.write_bytes(Path(__file__).read_bytes())
    (out / 'figure_source_sha256.json').write_text(json.dumps({
        target.name: hashlib.sha256(target.read_bytes()).hexdigest(),
        'input_summary.json': hashlib.sha256((out / 'summary.json').read_bytes()).hexdigest(),
    }, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
