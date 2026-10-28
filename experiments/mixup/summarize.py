"""Summarize Mixup runs using real-validation selection, never test selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ('mae', 'rmse', 'mse', 'mase')
GROUP = ['dataset', 'model', 'mode', 'weight', 'checkpoint']


def read_runs(root):
    records = []
    for path in sorted(root.glob('*/*/*/lambda_*/seed_*/metrics.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        dataset, model, mode = (path.parents[i].name for i in (4, 3, 2))
        if (model, mode) != (result['model'], result['mode']):
            raise ValueError(f'Path and metrics disagree: {path}')
        if path.parents[1].name != f'lambda_{result["weight"]:.3f}':
            raise ValueError(f'Weight path and metrics disagree: {path}')
        if path.parent.name != f'seed_{result["seed"]}':
            raise ValueError(f'Seed path and metrics disagree: {path}')
        for checkpoint, parts in result['checkpoints'].items():
            row = dict(dataset=dataset, model=model, mode=mode,
                weight=float(result['weight']), seed=int(result['seed']),
                checkpoint=checkpoint, checkpoint_step=int(parts['step']),
                final_step=int(result['final_step']),
                batch_size=int(result['batch_size']),
                augmentation_probability=float(result['augmentation_probability']),
                anchor_exposures=int(result['exposures']['anchors']),
                real_partner_exposures=int(result['exposures']['real_partners']),
                synthetic_partner_exposures=int(result['exposures']['synthetic_partners']),
                unique_anchors=int(result['unique_anchors']),
                elapsed_seconds=float(result['elapsed_seconds']))
            for split in ('validation', 'test'):
                for metric in METRICS:
                    row[f'{split}_{metric}'] = parts[split][metric]
            for key, value in result['diagnostics'].items():
                row[key] = value
            records.append(row)
    if not records:
        raise ValueError(f'No completed Mixup runs in {root}')
    runs = pd.DataFrame(records)
    if runs.duplicated(GROUP + ['seed']).any():
        raise ValueError('Duplicate dataset/model/mode/weight/checkpoint/seed')
    return runs.sort_values(GROUP + ['seed']).reset_index(drop=True)


def aggregate_runs(runs):
    values = [f'{split}_{metric}' for split in ('validation', 'test') for metric in METRICS]
    values += ['checkpoint_step', 'elapsed_seconds', 'anchor_exposures',
               'real_partner_exposures', 'synthetic_partner_exposures',
               'real_history_std', 'mixed_history_std', 'real_target_std',
               'mixed_target_std']
    available = [column for column in values if column in runs]
    grouped = runs.groupby(GROUP, dropna=False)[available].agg(['mean', 'std', 'count'])
    grouped.columns = [f'{field}_{stat}' for field, stat in grouped.columns]
    return grouped.reset_index().sort_values(GROUP).reset_index(drop=True)


def paired_differences(runs):
    """Compare each run with the same model/seed/checkpoint real-only run."""
    base = runs[runs['mode'] == 'real'].copy()
    if base.duplicated(['dataset', 'model', 'seed', 'checkpoint']).any():
        raise ValueError('Real-only baseline must appear once per model/seed/checkpoint')
    baseline = base[['dataset', 'model', 'seed', 'checkpoint',
                     'test_mae', 'test_rmse', 'test_mase']].rename(columns={
        key: f'real_{key}' for key in ('test_mae', 'test_rmse', 'test_mase')})
    paired = runs.merge(baseline, on=['dataset', 'model', 'seed', 'checkpoint'],
                        how='left', validate='many_to_one')
    for metric in ('mae', 'rmse', 'mase'):
        paired[f'delta_{metric}_vs_real'] = paired[f'test_{metric}'] - paired[f'real_test_{metric}']
        paired[f'improvement_{metric}_vs_real_percent'] = np.where(
            paired[f'real_test_{metric}'] > 0,
            -100 * paired[f'delta_{metric}_vs_real'] / paired[f'real_test_{metric}'], np.nan)
    rr = runs[runs['mode'] == 'real_real'][['dataset', 'model', 'seed', 'weight',
        'checkpoint', 'test_rmse']].rename(columns={'test_rmse':'real_real_test_rmse'})
    paired = paired.merge(rr, on=['dataset', 'model', 'seed', 'weight', 'checkpoint'],
                          how='left', validate='many_to_one')
    paired['delta_rmse_vs_real_real'] = paired['test_rmse'] - paired['real_real_test_rmse']
    return paired


def select_by_validation(runs, aggregate):
    """Select only groups evaluated with the identical seed set as real-only."""
    chosen = []
    for (dataset, model, checkpoint), part in runs.groupby(
            ['dataset', 'model', 'checkpoint']):
        baseline = part[part['mode'] == 'real']
        baseline_seeds = set(baseline['seed'])
        if not baseline_seeds:
            continue
        for (mode, weight), candidate in part.groupby(['mode', 'weight']):
            if set(candidate['seed']) != baseline_seeds:
                continue
            row = aggregate[(aggregate['dataset'] == dataset)
                & (aggregate['model'] == model) & (aggregate['checkpoint'] == checkpoint)
                & (aggregate['mode'] == mode) & (aggregate['weight'] == weight)].iloc[0].to_dict()
            row['paired_seeds'] = len(baseline_seeds)
            chosen.append(row)
    if not chosen:
        return pd.DataFrame()
    eligible = pd.DataFrame(chosen)
    selected = (eligible.sort_values(['validation_rmse_mean', 'mode', 'weight'])
        .drop_duplicates(['dataset', 'model', 'checkpoint', 'mode'])
        .sort_values(['dataset', 'model', 'checkpoint', 'mode']))
    # For each model/checkpoint, compare each mode's validation-selected setting
    # with real-only using the same seeds. Do not select by test values.
    baseline = selected[selected['mode'] == 'real'][['dataset', 'model', 'checkpoint',
        'test_rmse_mean', 'test_mae_mean']].rename(columns={
            'test_rmse_mean':'real_test_rmse_mean',
            'test_mae_mean':'real_test_mae_mean'})
    selected = selected.merge(baseline, on=['dataset', 'model', 'checkpoint'], how='left')
    selected['rmse_improvement_vs_real_percent'] = np.where(
        selected['real_test_rmse_mean'] > 0,
        100 * (1 - selected['test_rmse_mean'] / selected['real_test_rmse_mean']), np.nan)
    real_real = eligible[eligible['mode'] == 'real_real'][
        ['dataset', 'model', 'checkpoint', 'weight', 'test_rmse_mean']].rename(
        columns={'test_rmse_mean': 'same_weight_real_real_test_rmse_mean'})
    selected = selected.merge(real_real,
        on=['dataset', 'model', 'checkpoint', 'weight'], how='left',
        validate='many_to_one')
    selected['rmse_improvement_vs_same_weight_real_real_percent'] = np.where(
        selected['same_weight_real_real_test_rmse_mean'] > 0,
        100 * (1 - selected['test_rmse_mean'] /
               selected['same_weight_real_real_test_rmse_mean']), np.nan)
    return selected.reset_index(drop=True)


def plot_results(root, aggregate, report_dir):
    best = aggregate[aggregate['checkpoint'] == 'best']
    for (dataset, model), part in best.groupby(['dataset', 'model']):
        fig, ax = plt.subplots(figsize=(7, 4))
        baseline = part[part['mode'] == 'real']
        if not baseline.empty:
            ax.axhline(baseline.iloc[0]['test_rmse_mean'], color='black',
                       linestyle='--', label='Real-only')
        for mode, color in [('real_real', 'tab:blue'),
                            ('real_synthetic', 'tab:orange')]:
            group = part[part['mode'] == mode].sort_values('weight')
            if not group.empty:
                ax.errorbar(group['weight'], group['test_rmse_mean'],
                    yerr=group['test_rmse_std'].fillna(0), marker='o',
                    capsize=3, color=color, label=mode)
        ax.set(xlabel='Real weight λ', ylabel='Test RMSE (original units)',
               title=f'{dataset} / {model}: validation-best checkpoint')
        ax.grid(axis='y', alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(report_dir/f'{dataset}_{model}_rmse.png', dpi=150)
        plt.close(fig)


def report_text(root, runs, aggregate, selected, paired):
    lines = ['# 真实—模拟 Mixup 实验报告', '',
        f'已完成 {len(runs) // runs.checkpoint.nunique()} 次训练；'
        f'{runs.dataset.nunique()} 个数据集、{runs.model.nunique()} 个模型。',
        '只读取已完成的 `metrics.json`。以下配置按真实验证集原尺度 RMSE 的跨 seed 均值选择；测试集不参与选择。',
        '仅当某配置与 Real-only 具有完全相同的 seed 集合时才参与选择。'
        '误差条为 seed 标准差，不是置信区间；冒烟结果只用于检查流程。', '',
        '逐运行和跨 seed 结果分别见 `runs.csv`、`aggregate.csv`；'
        '同 seed 的差值见 `paired.csv`，验证选择见 `validation_selected.csv`。', '']
    for dataset, part in selected.groupby('dataset') if not selected.empty else []:
        lines += [f'## {dataset}', '',
            '| 模型 | 权重组 | 真实权重 λ | seed 数 | 验证 RMSE | 测试 RMSE 均值 ± SD | 相对 Real-only | 相对同 λ Real–Real |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for _, row in part[part['checkpoint'] == 'best'].iterrows():
            sd = row['test_rmse_std']
            sd_text = f'{sd:.6g}' if pd.notna(sd) else '—'
            delta = row['rmse_improvement_vs_real_percent']
            delta_text = f'{delta:+.2f}%' if pd.notna(delta) else '—'
            source_delta = row['rmse_improvement_vs_same_weight_real_real_percent']
            source_text = f'{source_delta:+.2f}%' if (row['mode'] == 'real_synthetic'
                and pd.notna(source_delta)) else '—'
            lines.append(f'| {row["model"]} | {row["mode"]} | {row["weight"]:g} | '
                f'{int(row["paired_seeds"])} | {row["validation_rmse_mean"]:.6g} | '
                f'{row["test_rmse_mean"]:.6g} ± {sd_text} | {delta_text} | {source_text} |')
        baseline_path = root/dataset/'baselines.json'
        if baseline_path.exists():
            baselines = json.loads(baseline_path.read_text(encoding='utf-8')).get('test', {})
            lines += ['', '免训练基线（测试集）：']
            for name, metrics in baselines.items():
                lines.append(f'- {name}: MAE={metrics["mae"]:.6g}, RMSE={metrics["rmse"]:.6g}')
        lines.append('')
        for model in sorted(part['model'].unique()):
            image = f'{dataset}_{model}_rmse.png'
            lines += [f'![{dataset} {model} RMSE]({image})', '']
    if selected.empty:
        lines += ['没有可按相同 seed 集合比较的配置；请检查 Real-only 是否已完成。', '']
    lines += ['## 解释边界', '',
        'Real-only、Real–Real、Real–Synthetic 每步使用相同数量的真实锚点，'
        '但 Mixup 组中部分锚点已被修改，原样真实样本曝光量不同。',
        '真实—真实 Mixup 是判断线性融合本身作用的对照。'
        '仅看到真实—模拟组优于 Real-only，不能单独证明模拟机制提供了额外信息。',
        '单数据集或少量 seed 的微小测试差异不支持普遍改进结论；'
        '重叠预测窗口不能视为独立样本。', '']
    return '\n'.join(lines)


def write_compact_mase_report(report_dir, aggregate, runs):
    """Select models and mix weights on validation RMSE; display test MASE."""
    required = {'dataset', 'model', 'mode', 'weight', 'checkpoint',
                'validation_rmse_mean', 'test_mase_mean', 'test_mase_std'}
    missing = required - set(aggregate.columns)
    if missing:
        raise ValueError(f'aggregate.csv lacks columns: {sorted(missing)}')
    required_runs = {'dataset', 'model', 'mode', 'weight', 'seed', 'checkpoint'}
    missing_runs = required_runs - set(runs.columns)
    if missing_runs:
        raise ValueError(f'runs.csv lacks columns: {sorted(missing_runs)}')
    best_runs = runs[runs.checkpoint == 'best']
    best_aggregate = aggregate[(aggregate.checkpoint == 'best') &
        aggregate.test_mase_mean.notna() & aggregate.validation_rmse_mean.notna()].copy()
    if best_aggregate.empty:
        raise ValueError('No validation-best runs with finite test MASE')
    # Compare only groups with the same seed set as every real-only model of
    # that dataset. Partial shards remain in aggregate.csv, but cannot win here.
    eligible = []
    for dataset, data in best_runs.groupby('dataset'):
        base = data[data['mode'] == 'real']
        base_sets = [set(group.seed) for _, group in base.groupby('model')]
        if not base_sets or any(seeds != base_sets[0] for seeds in base_sets[1:]):
            continue
        expected = base_sets[0]
        if not expected:
            continue
        for (model, mode, weight), group in data.groupby(['model', 'mode', 'weight']):
            if model not in set(base.model) or set(group.seed) != expected:
                continue
            match = best_aggregate[(best_aggregate.dataset == dataset) &
                (best_aggregate.model == model) & (best_aggregate['mode'] == mode) &
                (best_aggregate.weight == weight)]
            if len(match) == 1:
                eligible.append(match.iloc[0].to_dict())
    if not eligible:
        raise ValueError('No complete, paired seed groups with finite test MASE')
    eligible = pd.DataFrame(eligible)
    base = eligible[eligible['mode'] == 'real'].copy()
    base = (base.sort_values(['dataset', 'validation_rmse_mean', 'model'])
        .drop_duplicates('dataset'))
    # A separate model is selected at each weight and for each Mixup source.
    per_ratio = (eligible[eligible['mode'] != 'real']
        .sort_values(['dataset', 'mode', 'weight', 'validation_rmse_mean', 'model'])
        .drop_duplicates(['dataset', 'mode', 'weight']).copy())
    per_ratio['partner_weight'] = 1 - per_ratio.weight
    baseline = base.assign(mode='real', weight=1.0)
    by_ratio = pd.concat([baseline, per_ratio], ignore_index=True)
    by_ratio['partner_weight'] = 1 - by_ratio.weight
    by_ratio = by_ratio.sort_values(['dataset', 'mode', 'partner_weight'])
    chosen = (per_ratio.sort_values(['dataset', 'mode', 'validation_rmse_mean', 'weight', 'model'])
        .drop_duplicates(['dataset', 'mode']))
    chosen = pd.concat([baseline, chosen], ignore_index=True).sort_values(['dataset', 'mode'])
    baseline_mase = base.set_index('dataset').test_mase_mean
    chosen['real_only_test_mase_mean'] = chosen.dataset.map(baseline_mase)
    chosen['mase_improvement_vs_real_percent'] = 100 * (
        1 - chosen.test_mase_mean / chosen.real_only_test_mase_mean)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    by_ratio.to_csv(report_dir/'compact_mase_by_ratio.csv', index=False)
    chosen.to_csv(report_dir/'compact_mase_best.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    modes = [('real_real', 'Real–Real'), ('real_synthetic', 'Real–Synthetic')]
    datasets = sorted(by_ratio.dataset.unique())
    colors = dict(zip(datasets, plt.get_cmap('tab10').colors[:len(datasets)]))
    models = sorted(by_ratio.model.unique())
    markers = dict(zip(models, ['o', 's', '^', 'D', 'P', 'X', 'v', '*']))
    if len(models) > len(markers):
        raise ValueError('Too many models for distinct plot markers')
    for ax, (mode, label) in zip(axes, modes):
        for dataset in datasets:
            subset = by_ratio[(by_ratio.dataset == dataset) &
                              ((by_ratio['mode'] == mode) | (by_ratio['mode'] == 'real'))]
            subset = subset.sort_values('partner_weight')
            if subset.empty:
                continue
            ax.plot(subset.partner_weight * 100, subset.test_mase_mean,
                    color=colors[dataset], alpha=.75, label=dataset)
            for _, row in subset.iterrows():
                ax.scatter(row.partner_weight * 100, row.test_mase_mean,
                    color=colors[dataset], marker=markers[row.model], s=55,
                    edgecolors='white', linewidth=.6, zorder=3)
        ax.axhline(1, color='0.5', linestyle='--', linewidth=1)
        ax.set(title=label, xlabel='Partner weight (%)',
               xticks=sorted(by_ratio.partner_weight.unique()*100))
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(axis='x', labelrotation=45)
    axes[0].set_ylabel('Test MASE')
    from matplotlib.lines import Line2D
    dataset_legend = [Line2D([0], [0], color=colors[d], marker='o', label=d)
                      for d in datasets]
    model_legend = [Line2D([0], [0], color='0.3', linestyle='None',
                     marker=markers[m], label=m) for m in models]
    axes[1].legend(handles=dataset_legend + model_legend, loc='upper left',
                   bbox_to_anchor=(1.01, 1), title='Dataset / selected model')
    fig.tight_layout()
    fig.savefig(report_dir/'compact_mase_comparison.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    def fmt(row):
        sd = row.test_mase_std
        return f'{row.test_mase_mean:.4f} ± {sd:.4f}' if pd.notna(sd) else f'{row.test_mase_mean:.4f} (n=1)'

    lines = ['# Mixup：测试 MASE 与各混合比例', '',
        '横轴为伙伴窗口权重 `1−λ`：0% 是 Real-only；另外两组分别为真实—真实、真实—模拟 Mixup。'
        '图中颜色代表数据集，点形代表该比例由**真实验证 RMSE**选出的模型。',
        '每个数据集先要求各候选组与 Real-only 具有相同 seed 集合，再按跨 seed 平均验证 RMSE 选模型和权重。'
        '测试 MASE 只用于展示，不参与选择。缺失或 MASE 分母无效的组不参与图表。',
        'MASE = 原尺度测试 MAE / 真实训练段季节性 naive MAE；低于 1 表示优于该训练段的季节性误差基准。'
        '误差中的“±”为训练 seed 标准差，不是置信区间。', '',
        '![各混合比例的测试 MASE](compact_mase_comparison.png)', '',
        '## 各数据集的验证选择结果', '',
        '| 数据集 | 方法 | 伙伴权重 | 模型 | 测试 MASE | 相对 Real-only |',
        '|---|---|---:|---|---:|---:|']
    for _, row in chosen.iterrows():
        lines.append(f'| {row.dataset} | {row["mode"]} | {1-row.weight:.3f} | '
            f'{row.model} | {fmt(row)} | {row.mase_improvement_vs_real_percent:+.2f}% |')
    for mode, label in modes:
        lines += ['', f'## {label}：各混合比例', '',
            '每个数据集与伙伴权重单独按验证 RMSE 选模型；空缺表示未完成可配对的实验。', '']
        weights = sorted(per_ratio[per_ratio['mode'] == mode].partner_weight.unique())
        lines += ['| 数据集 | Real-only | ' + ' | '.join(f'{w*100:g}%' for w in weights) + ' |',
                  '|---|---:|' + '|'.join('---:' for _ in weights) + '|']
        for dataset in datasets:
            baseline_row = base[base.dataset == dataset]
            values = [fmt(baseline_row.iloc[0]) if len(baseline_row) else '—']
            for weight in weights:
                match = by_ratio[(by_ratio.dataset == dataset) & (by_ratio['mode'] == mode) &
                                 np.isclose(by_ratio.partner_weight, weight)]
                values.append(fmt(match.iloc[0]) if len(match) else '—')
            lines.append(f'| {dataset} | ' + ' | '.join(values) + ' |')
    lines += ['', '本表不是跨数据集原始 RMSE 的平均。完整逐次结果见 `runs.csv`，'
              '各组合均值与标准差见 `aggregate.csv`；筛选后的两份 CSV 可用于后续分析。', '']
    output = report_dir/'compact_mase.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    return output


def summarize(output):
    root = Path(output)
    runs = read_runs(root)
    aggregate = aggregate_runs(runs)
    paired = paired_differences(runs)
    selected = select_by_validation(runs, aggregate)
    report_dir = root/'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(report_dir/'runs.csv', index=False)
    aggregate.to_csv(report_dir/'aggregate.csv', index=False)
    paired.to_csv(report_dir/'paired.csv', index=False)
    selected.to_csv(report_dir/'validation_selected.csv', index=False)
    plot_results(root, aggregate, report_dir)
    report = report_dir/'report.md'
    report.write_text(report_text(root, runs, aggregate, selected, paired), encoding='utf-8')
    write_compact_mase_report(report_dir, aggregate, runs)
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Mixup experiment output containing dataset/model/mode runs')
    parser.add_argument('--report-only', action='store_true',
                        help='Rebuild compact MASE report from report/runs.csv and aggregate.csv')
    args = parser.parse_args()
    if args.report_only:
        report_dir = args.output/'report'
        print(write_compact_mase_report(report_dir,
            pd.read_csv(report_dir/'aggregate.csv'), pd.read_csv(report_dir/'runs.csv')))
    else:
        summarize(args.output)
