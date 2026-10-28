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
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Mixup experiment output containing dataset/model/mode runs')
    args = parser.parse_args()
    summarize(args.output)
