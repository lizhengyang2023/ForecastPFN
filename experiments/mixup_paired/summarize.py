"""Validation-selected reports for four paired cells and optional old random controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEY = ['dataset','model','mode','strategy','weight','checkpoint']


def read_paired(root):
    rows = []
    for path in sorted(Path(root).glob('**/lambda_*/seed_*/metrics.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        dataset, model, mode, strategy = (path.parents[i].name for i in (5,4,3,2))
        if (model, mode, strategy) != (result['model'], result['mode'], result['strategy']):
            raise ValueError(f'Path and metrics disagree: {path}')
        for checkpoint, part in result['checkpoints'].items():
            row = dict(dataset=dataset, model=model, mode=mode, strategy=strategy,
                weight=float(result['weight']), seed=int(result['seed']), checkpoint=checkpoint,
                step=int(part['step']), final_step=int(result['final_step']),
                batch_size=int(result['batch_size']),
                augmentation_probability=float(result['augmentation_probability']),
                elapsed_seconds=result['elapsed_seconds'])
            for split in ('validation','test'):
                for metric in ('mae','rmse','mse','mase'):
                    row[f'{split}_{metric}'] = part[split][metric]
            row.update({f'match_{key}': value for key, value in
                result['matching']['match_rates'].items()})
            row['mean_candidate_count'] = result['matching']['mean_candidate_count']
            row['mean_log_period_distance'] = result['matching']['mean_log_period_distance']
            row['mean_phase_distance'] = result['matching']['mean_phase_distance']
            row['mean_future_phase_drift'] = result['matching']['mean_future_phase_drift']
            row['unique_partners'] = result['matching']['unique_partners']
            row.update(result['diagnostics'])
            rows.append(row)
    if not rows:
        raise ValueError(f'No paired metrics in {root}')
    runs = pd.DataFrame(rows)
    if runs.duplicated(KEY+['seed']).any():
        raise ValueError('Duplicate paired run')
    return runs


def read_random(root):
    if root is None:
        return pd.DataFrame()
    root = Path(root)
    csv = root/'report/runs.csv'
    if csv.exists():
        runs = pd.read_csv(csv)
    else:
        from experiments.mixup.summarize import read_runs
        runs = read_runs(root)
    runs = runs.copy()
    runs['strategy'] = 'random'
    return runs


def summarize(root, random_root=None):
    root = Path(root)
    paired = read_paired(root)
    random = read_random(random_root)
    if len(random):
        # Old full runs must never be used as controls for a short smoke run.
        budget = paired[['dataset','model','seed','final_step','batch_size',
            'augmentation_probability']].drop_duplicates()
        random = random.merge(budget, on=['dataset','model','seed','final_step',
            'batch_size','augmentation_probability'], how='inner', validate='many_to_one')
    runs = pd.concat([paired, random], ignore_index=True) if len(random) else paired
    grouped = runs.groupby(KEY, dropna=False)
    values = [column for column in ('validation_rmse','test_rmse','test_mae','test_mase',
        'match_period_phase','match_period_only','match_random','mean_candidate_count',
        'mean_log_period_distance','mean_phase_distance','mean_future_phase_drift',
        'unique_partners',
        'real_history_std','mixed_history_std','real_target_std','mixed_target_std')
        if column in runs]
    aggregate = grouped[values].agg(['mean','std','count'])
    aggregate.columns = [f'{key}_{stat}' for key, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    # A candidate is eligible only if all its seeds match the random real-only seeds.
    # When the old results are unavailable, use the common seed intersection of paired cells.
    eligible = []
    for (dataset, model, checkpoint), part in runs.groupby(['dataset','model','checkpoint']):
        baseline = part[(part['mode'] == 'real') & (part['strategy'] == 'random')]
        seed_set = set(baseline['seed']) if len(baseline) else None
        groups = list(part.groupby(['mode','strategy','weight']))
        if seed_set is None:
            seed_sets = [set(group['seed']) for _, group in groups if group.iloc[0]['strategy'] != 'random']
            seed_set = set.intersection(*seed_sets) if seed_sets else set()
        for (mode, strategy, weight), group in groups:
            if seed_set and set(group['seed']) == seed_set:
                eligible.append(dict(dataset=dataset, model=model, checkpoint=checkpoint,
                    mode=mode, strategy=strategy, weight=weight, paired_seeds=len(seed_set)))
    eligible = pd.DataFrame(eligible)
    if eligible.empty:
        raise ValueError('No complete same-seed comparison groups')
    comparable = aggregate.merge(eligible, on=KEY, how='inner')
    selected = (comparable.sort_values(['validation_rmse_mean','mode','strategy','weight'])
        .drop_duplicates(['dataset','model','checkpoint','mode','strategy']))
    # Compare all strategies at the same model, seed, weight, and checkpoint.
    comparisons = runs[runs['checkpoint'] == 'best'].copy()
    join = ['dataset','model','seed','checkpoint']
    baseline = comparisons[(comparisons['mode']=='real') & (comparisons['strategy']=='random')]
    if len(baseline):
        base = baseline[join+['test_rmse','test_mase']].rename(columns={
            'test_rmse':'real_test_rmse','test_mase':'real_test_mase'})
        comparisons = comparisons.merge(base, on=join, how='left', validate='many_to_one')
        comparisons['delta_rmse_vs_real'] = comparisons['test_rmse']-comparisons['real_test_rmse']
        comparisons['delta_mase_vs_real'] = comparisons['test_mase']-comparisons['real_test_mase']
    random_match = runs[(runs['strategy']=='random') & (runs['mode']!='real') &
        (runs['checkpoint']=='best')]
    if len(random_match):
        base = random_match[join+['mode','weight','test_rmse','test_mase']].rename(columns={
            'test_rmse':'random_test_rmse','test_mase':'random_test_mase'})
        comparisons = comparisons.merge(base, on=join+['mode','weight'], how='left',
            validate='many_to_one')
        comparisons['delta_rmse_vs_random'] = comparisons['test_rmse']-comparisons['random_test_rmse']
        comparisons['delta_mase_vs_random'] = comparisons['test_mase']-comparisons['random_test_mase']
    per_ratio = (comparable[comparable['checkpoint']=='best']
        .sort_values(['validation_rmse_mean','model'])
        .drop_duplicates(['dataset','mode','strategy','weight']))
    report = root/'report'
    report.mkdir(parents=True, exist_ok=True)
    for name, frame in [('runs', runs),('aggregate', aggregate),('validation_selected', selected),
                        ('paired_differences', comparisons),('compact_mase_by_ratio', per_ratio)]:
        frame.to_csv(report/f'{name}.csv', index=False)
    for dataset, part in per_ratio.groupby('dataset'):
        fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=False)
        for ax, mode in zip(axes, ('real_real','real_synthetic')):
            for strategy, sub in part[part['mode']==mode].groupby('strategy'):
                sub = sub.sort_values('weight', ascending=False)
                ax.errorbar(1-sub['weight'], sub['test_mase_mean'],
                    yerr=sub['test_mase_std'].fillna(0), marker='o', capsize=3,
                    label=strategy)
            ax.set(title=mode, xlabel='Partner weight (1−λ)', ylabel='Test MASE')
            ax.grid(axis='y', alpha=.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(report/f'{dataset}_compact_mase.png', dpi=160)
        plt.close(fig)
    diagnostic = aggregate[aggregate['checkpoint']=='best']
    for dataset, part in diagnostic.groupby('dataset'):
        part = part[part['strategy'].isin(['period_only','period_phase'])]
        part = part.groupby(['mode','strategy'])[
            ['match_period_phase_mean','match_period_only_mean','match_random_mean']].mean()
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(7,4))
        labels = [f'{mode}\n{strategy}' for mode, strategy in part.index]
        bottom = np.zeros(len(part))
        for column, color, label in [('match_period_phase_mean','tab:green','period + phase'),
                                     ('match_period_only_mean','tab:blue','period only'),
                                     ('match_random_mean','tab:gray','random fallback')]:
            values = part[column].fillna(0).to_numpy()
            ax.bar(labels, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set(ylabel='Share of augmented pairs', ylim=(0,1),
               title=f'{dataset}: actual matching and fallback')
        ax.legend()
        fig.tight_layout()
        fig.savefig(report/f'{dataset}_matching_rates.png', dpi=160)
        plt.close(fig)
    best = selected[selected['checkpoint']=='best']
    lines = ['# 周期与相位配对 Mixup 结果', '',
        f'已完成 {len(paired)//2} 个配对训练运行。按真实验证 RMSE 的跨 seed 均值选权重；测试集不参与选择。',
        '仅同 seed 集合的组可比较。误差条为训练 seed 标准差，不是置信区间。', '']
    if random_root is None or not len(random):
        lines += ['没有同模型、seed 和训练预算的随机 Mixup 对照；'
                  '与 Real-only、随机伙伴的差值暂不可报告。', '']
    else:
        lines += [f'随机对照来源：`{random_root}`。如该目录只保留报告 CSV，'
                  '请另行核对原运行的数据和模拟池签名。', '']
    lines += ['## 验证选择的组合', '',
        '差值以相同数据集、模型、权重和 checkpoint 的跨 seed 平均测试 MASE 计算；'
        '负值表示该配对组误差较低。差值仅用于解释，不参与选组。', '',
        '| 数据集 | 模型 | 来源 | 配对策略 | λ | 验证 RMSE | 测试 MASE | 相位匹配率 | ΔMASE vs 随机 | ΔMASE vs 仅周期 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for _, row in best.sort_values(['dataset','model','mode','strategy']).iterrows():
        rate = row.get('match_period_phase_mean', np.nan)
        def delta(strategy):
            reference = comparable[(comparable['dataset']==row.dataset)
                & (comparable['model']==row.model)
                & (comparable['mode']==row['mode'])
                & (comparable['strategy']==strategy)
                & (comparable['weight']==row.weight)
                & (comparable['checkpoint']=='best')]
            return f'{row.test_mase_mean-reference.iloc[0].test_mase_mean:+.4f}' if len(reference) else '—'
        lines.append(f'| {row.dataset} | {row.model} | {row["mode"]} | {row.strategy} | '
            f'{row.weight:.2f} | {row.validation_rmse_mean:.4f} | '
            f'{row.test_mase_mean:.4f} | {rate:.1%} | {delta("random")} | '
            f'{delta("period_only")} |' if pd.notna(rate) else
            f'| {row.dataset} | {row.model} | {row["mode"]} | {row.strategy} | '
            f'{row.weight:.2f} | {row.validation_rmse_mean:.4f} | '
            f'{row.test_mase_mean:.4f} | — | {delta("random")} | '
            f'{delta("period_only")} |')
    lines += ['', '## 图表与诊断', '',
        '每个数据集的 `*_compact_mase.png` 对照四种配对策略与可用的随机组；'
        '`*_matching_rates.png` 显示实际配对层级与回退占比。'
        '`compact_mase_by_ratio.csv` 为每个伙伴权重按验证 RMSE 选模型后的结果。',
        '`paired_differences.csv` 给出同模型、同 seed、同权重的测试差值；'
        '`runs.csv` 包含实际相位匹配率、回退率、候选数、周期/相位距离和振幅诊断。',
        '如果相位组主要回退到仅周期或随机组，不能把预测差异归因于相位匹配。', '']
    output = report/'report.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--random-output', type=Path)
    args = parser.parse_args()
    summarize(args.output, args.random_output)
