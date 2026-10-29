"""Plot validation-selected test MASE changes against paired Real runs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


TASKS = (
    ('real_real_phase_match', 'Real–Real\nphase match'),
    ('real_real_phase_mismatch', 'Real–Real\nphase mismatch'),
    ('real_synthetic_phase_match', 'Real–Synthetic\nphase match'),
    ('real_synthetic_phase_mismatch', 'Real–Synthetic\nphase mismatch'),
)
MODELS = ('linear', 'dlinear', 'mlp', 'gru', 'transformer')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')
LAMBDA_MARKERS = ((0.5, 'o', -0.065), (0.75, 's', 0.0), (0.9, '^', 0.065))
DISPLAY_MODELS = {'linear': 'Linear', 'dlinear': 'DLinear', 'mlp': 'MLP',
                  'gru': 'GRU', 'transformer': 'Transformer'}


def delta_table(report):
    """Mean of seed-paired percentage changes; lambda selected by validation RMSE."""
    report = Path(report)
    runs = pd.read_csv(report / 'runs.csv')
    selected = pd.read_csv(report / 'validation_selected.csv')
    runs = runs[runs.checkpoint.eq('best')]
    selected = selected[selected.checkpoint.eq('best') & selected.task.ne('real')]
    if selected.empty:
        raise ValueError('No validation-selected best-checkpoint Mixup tasks')
    keys = ['dataset', 'model', 'task', 'weight']
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate validation-selected task')
    chosen = runs.merge(selected[keys + ['test_mase_count']], on=keys,
                        how='inner', validate='many_to_one')
    if len(chosen) != int(selected.test_mase_count.sum()):
        raise ValueError('Missing runs for a validation-selected lambda')
    real = runs[runs.task.eq('real')][
        ['dataset', 'model', 'seed', 'test_mase']].rename(
            columns={'test_mase': 'real_test_mase'})
    if real.duplicated(['dataset', 'model', 'seed']).any():
        raise ValueError('Duplicate Real baseline for a model and seed')
    paired = chosen.merge(real, on=['dataset', 'model', 'seed'],
                          how='left', validate='many_to_one')
    if paired.real_test_mase.isna().any() or (paired.real_test_mase <= 0).any():
        raise ValueError('Missing or non-positive Real MASE baseline')
    paired['delta_pct'] = 100 * (paired.test_mase / paired.real_test_mase - 1)
    table = paired.groupby(keys, as_index=False).agg(
        n_seeds=('seed', 'size'),
        real_mase_mean=('real_test_mase', 'mean'),
        task_mase_mean=('test_mase', 'mean'),
        delta_pct=('delta_pct', 'mean'),
        delta_pct_sd=('delta_pct', 'std'),
    ).rename(columns={'weight': 'lambda_value'})
    return table.sort_values(['dataset', 'model', 'task']).reset_index(drop=True)


def rank_summary(report, delta):
    """Describe test winners after validation-only lambda selection."""
    report = Path(report)
    selected = pd.read_csv(report / 'validation_selected.csv')
    ranked = selected[selected.checkpoint.eq('best')][
        ['dataset', 'model', 'task', 'weight', 'test_mase_mean']].copy()
    if ranked.duplicated(['dataset', 'model', 'task']).any():
        raise ValueError('Duplicate validation-selected task for ranking')
    group = ranked.groupby(['dataset', 'model']).test_mase_mean
    ranked['test_rank'] = group.rank(method='min')
    ranked['gap_to_best_pct'] = 100 * (ranked.test_mase_mean / group.transform('min') - 1)
    ranked['first'] = ranked.test_rank.eq(1)
    ranked['within_half_percent'] = ranked.gap_to_best_pct.le(0.5)
    ranked = ranked.merge(delta[['dataset', 'model', 'task', 'delta_pct']],
                          on=['dataset', 'model', 'task'], how='left',
                          validate='one_to_one')
    ranked.loc[ranked.task.eq('real'), 'delta_pct'] = 0.0
    if ranked.delta_pct.isna().any():
        raise ValueError('Missing selected-task percentage change for ranking')
    ranked.to_csv(report / 'mase_task_rankings.csv', index=False)
    summary = ranked.groupby('task', as_index=False).agg(
        combinations=('model', 'size'),
        first_count=('first', 'sum'),
        within_half_percent_count=('within_half_percent', 'sum'),
        improved_vs_real_count=('delta_pct', lambda values: int(values.lt(0).sum())),
        median_delta_pct=('delta_pct', 'median'),
    )
    summary.to_csv(report / 'mase_task_rank_summary.csv', index=False)
    return summary


def fixed_lambda_rank_summary(report):
    """Rank Real and all four tasks separately at each fixed lambda."""
    report = Path(report)
    mixed = all_lambda_table(report)
    runs = pd.read_csv(report / 'runs.csv')
    real = runs[runs.checkpoint.eq('best') & runs.task.eq('real')]
    real = real.groupby(['dataset', 'model'], as_index=False).agg(
        task_mase_mean=('test_mase', 'mean'))
    if real.empty:
        raise ValueError('No Real baseline for fixed-lambda ranking')
    lambdas = sorted(mixed.lambda_value.unique())
    baseline = pd.concat([real.assign(lambda_value=weight, task='real',
                                      delta_pct=0.0) for weight in lambdas],
                         ignore_index=True)
    ranked = pd.concat([
        mixed[['dataset', 'model', 'lambda_value', 'task',
               'task_mase_mean', 'delta_pct']], baseline], ignore_index=True)
    keys = ['dataset', 'model', 'lambda_value']
    if ranked.duplicated(keys + ['task']).any():
        raise ValueError('Duplicate task in fixed-lambda ranking')
    group = ranked.groupby(keys).task_mase_mean
    ranked['test_rank'] = group.rank(method='min')
    ranked['gap_to_best_pct'] = 100 * (ranked.task_mase_mean / group.transform('min') - 1)
    ranked['first'] = ranked.test_rank.eq(1)
    ranked['within_half_percent'] = ranked.gap_to_best_pct.le(0.5)
    ranked = ranked.sort_values(keys + ['test_rank', 'task']).reset_index(drop=True)
    ranked.to_csv(report / 'mase_task_fixed_lambda_rankings.csv', index=False)
    summary = ranked.groupby(['lambda_value', 'task'], as_index=False).agg(
        combinations=('model', 'size'),
        first_count=('first', 'sum'),
        within_half_percent_count=('within_half_percent', 'sum'),
        improved_vs_real_count=('delta_pct', lambda values: int(values.lt(0).sum())),
        median_delta_pct=('delta_pct', 'median'),
    )
    summary.to_csv(report / 'mase_task_fixed_lambda_summary.csv', index=False)
    return summary


def _label_positions(values, gap):
    """Spread nearby point labels vertically while retaining their order."""
    ordered = sorted(values, key=lambda item: item[1])
    placed = []
    for index, value in ordered:
        placed.append((index, max(value, placed[-1][1] + gap) if placed else value))
    shift = sum(value - label for (_, value), (_, label) in zip(ordered, placed)) / len(placed)
    return {index: label + shift for index, label in placed}


def plot_report(report):
    """Write one plot per dataset and a reusable source-data CSV into report/."""
    report = Path(report)
    table = delta_table(report)
    table.to_csv(report / 'mase_delta_vs_real.csv', index=False)
    ranking = rank_summary(report, table)
    fixed_ranking = fixed_lambda_rank_summary(report)
    task_names = dict(TASKS)
    model_colors = dict(zip(MODELS, COLORS))
    images = []
    for dataset, part in table.groupby('dataset', sort=True):
        tasks = [task for task, _ in TASKS if task in set(part.task)]
        if not tasks:
            continue
        models = [model for model in MODELS if model in set(part.model)]
        models += sorted(set(part.model) - set(models))
        fig, ax = plt.subplots(figsize=(13.5, 7.2), layout='constrained')
        xloc = {task: i for i, task in enumerate(tasks)}
        for i, model in enumerate(models):
            rows = part[part.model.eq(model)].set_index('task')
            xs = [xloc[task] for task in tasks if task in rows.index]
            ys = [float(rows.loc[task, 'delta_pct']) for task in tasks if task in rows.index]
            color = model_colors.get(model, plt.cm.tab10(i % 10))
            ax.plot(xs, ys, color=color, marker='o', markersize=6,
                    linewidth=2, label=DISPLAY_MODELS.get(model, model))
        span = max(float(part.delta_pct.max() - part.delta_pct.min()), 0.75)
        label_y = {}
        for task in tasks:
            rows = part[part.task.eq(task)]
            label_y[task] = _label_positions(
                list(zip(rows.model, rows.delta_pct)), gap=0.052 * span)
        all_y = list(part.delta_pct)
        for _, row in part.iterrows():
            x = xloc[row.task]
            y = float(row.delta_pct)
            target = label_y[row.task][row.model]
            color = model_colors.get(row.model, '#333333')
            ax.plot([x + 0.018, x + 0.07], [y, target],
                    color=color, linewidth=0.7, alpha=0.7)
            ax.text(x + 0.075, target, f'λ={row.lambda_value:g}',
                    color=color, fontsize=8.5, va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.76, pad=0.5))
            all_y.append(target)
        ax.axhline(0, color='#444444', linestyle='--', linewidth=1.2)
        lower, upper = min(all_y + [0]), max(all_y + [0])
        padding = max((upper - lower) * 0.11, 0.12)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xlim(-0.12, len(tasks) - 1 + 0.43)
        ax.set_xticks(range(len(tasks)), [task_names[task] for task in tasks])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}%'))
        ax.set_ylabel('Δ test MASE vs Real (%)')
        ax.set_xlabel('Task type')
        ax.set_title(f'{dataset.upper()} — MASE change vs Real-only', loc='left',
                     fontsize=16, fontweight='bold')
        ax.grid(axis='y', color='#dddddd', linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(title='Model', ncol=min(len(models), 5), loc='upper center',
                  bbox_to_anchor=(0.5, 1.01), frameon=False)
        name = f'{dataset}_mase_delta_vs_real.png'
        fig.savefig(report / name, dpi=200, bbox_inches='tight')
        fig.savefig(report / f'{dataset}_mase_delta_vs_real.svg', bbox_inches='tight')
        plt.close(fig)
        images.append((dataset, name))
    lines = [
        '# MASE 相对 Real 的变化', '',
        '每个模型与任务的 λ 使用 `validation_selected.csv` 中按跨 seed 验证 RMSE 选出的值；'
        'checkpoint 为 `best`。对每个 seed 计算 '
        '`100 × (Mixup 测试 MASE / 同模型同 seed Real 测试 MASE − 1)`，'
        '再对 seed 求均值。负值表示优于 Real。图中每个点标注所选 λ。', '',
        '作图数据见 [`mase_delta_vs_real.csv`](mase_delta_vs_real.csv)。', '',
        '## 任务排名（描述性）', '',
        '每个数据集×模型内，先按验证 RMSE 选 λ，再用测试 MASE 给 Real 和四种任务排名。'
        '“接近第一名”暂按与第一名差距 ≤0.5% 计数，仅是敏感性描述；'
        '名次不代表统计显著性。逐组合排名见 '
        '[`mase_task_rankings.csv`](mase_task_rankings.csv)，汇总见 '
        '[`mase_task_rank_summary.csv`](mase_task_rank_summary.csv)。', '',
        '| 任务 | 第一名次数 | 距第一名 ≤0.5% | 优于 Real 次数 | MASE 变化中位数 |',
        '|---|---:|---:|---:|---:|',
    ]
    for _, row in ranking.iterrows():
        n = int(row.combinations)
        lines.append(f'| {row.task} | {int(row.first_count)}/{n} | '
                     f'{int(row.within_half_percent_count)}/{n} | '
                     f'{int(row.improved_vs_real_count)}/{n} | '
                     f'{row.median_delta_pct:+.2f}% |')
    lines.append('')
    lines += [
        '## 固定 λ 的任务排名', '',
        '在每个固定 λ 下，按同一数据集×模型的测试 MASE 均值比较 Real 与四种混合任务；'
        'Real 基线在各 λ 的比较中相同。各表重复使用相同数据集×模型，'
        '不能把不同 λ 的名次当作独立实验。'
        '“优于 Real”与上表一样使用同 seed 配对后的 MASE 百分比变化。'
        '逐组合数据见 [`mase_task_fixed_lambda_rankings.csv`]'
        '(mase_task_fixed_lambda_rankings.csv)，汇总见 '
        '[`mase_task_fixed_lambda_summary.csv`]'
        '(mase_task_fixed_lambda_summary.csv)。', '',
    ]
    for weight, part in fixed_ranking.groupby('lambda_value', sort=True):
        lines += [f'### λ={weight:g}', '',
                  '| 任务 | 第一名次数 | 距第一名 ≤0.5% | 优于 Real 次数 | MASE 变化中位数 |',
                  '|---|---:|---:|---:|---:|']
        for _, row in part.iterrows():
            n = int(row.combinations)
            lines.append(f'| {row.task} | {int(row.first_count)}/{n} | '
                         f'{int(row.within_half_percent_count)}/{n} | '
                         f'{int(row.improved_vs_real_count)}/{n} | '
                         f'{row.median_delta_pct:+.2f}% |')
        lines.append('')
    for dataset, name in images:
        lines += [f'## {dataset}', '', f'![{dataset} MASE delta]({name})', '']
    (report / 'mase_delta.md').write_text('\n'.join(lines), encoding='utf-8')
    return images


def all_lambda_table(report):
    """Every lambda's seed-paired MASE change, including validation choice flags."""
    report = Path(report)
    runs = pd.read_csv(report / 'runs.csv')
    runs = runs[runs.checkpoint.eq('best')]
    real = runs[runs.task.eq('real')][
        ['dataset', 'model', 'seed', 'test_mase']].rename(
            columns={'test_mase': 'real_test_mase'})
    if real.duplicated(['dataset', 'model', 'seed']).any():
        raise ValueError('Duplicate Real baseline for a model and seed')
    mixed = runs[runs.task.ne('real')]
    keys = ['dataset', 'model', 'task', 'weight']
    if mixed.duplicated(keys + ['seed']).any():
        raise ValueError('Duplicate Mixup run for a lambda and seed')
    paired = mixed.merge(real, on=['dataset', 'model', 'seed'],
                         how='left', validate='many_to_one')
    if paired.real_test_mase.isna().any() or (paired.real_test_mase <= 0).any():
        raise ValueError('Missing or non-positive Real MASE baseline')
    paired['delta_pct'] = 100 * (paired.test_mase / paired.real_test_mase - 1)
    table = paired.groupby(keys, as_index=False).agg(
        n_seeds=('seed', 'size'),
        real_mase_mean=('real_test_mase', 'mean'),
        task_mase_mean=('test_mase', 'mean'),
        delta_pct=('delta_pct', 'mean'),
        delta_pct_sd=('delta_pct', 'std'),
    )
    selected = pd.read_csv(report / 'validation_selected.csv')
    selected = selected[selected.checkpoint.eq('best') & selected.task.ne('real')]
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate validation-selected task')
    selected = selected[keys].assign(validation_selected=True)
    table = table.merge(selected, on=keys, how='left', validate='one_to_one')
    table['validation_selected'] = table.validation_selected.fillna(False).astype(bool)
    table = table.rename(columns={'weight': 'lambda_value'})
    if not table.groupby(['dataset', 'model', 'task']).validation_selected.sum().eq(1).all():
        raise ValueError('Expected one validation-selected lambda per task')
    return table.sort_values(['dataset', 'model', 'task', 'lambda_value']).reset_index(drop=True)


def same_lambda_phase_summary(report, table):
    """Compare match and mismatch with dataset, model and lambda held fixed."""
    report = Path(report)
    wide = table.pivot(index=['dataset', 'model', 'lambda_value'],
                       columns='task', values='delta_pct')
    contrasts = []
    for source in ('real_real', 'real_synthetic'):
        match = f'{source}_phase_match'
        mismatch = f'{source}_phase_mismatch'
        if match not in wide or mismatch not in wide:
            continue
        pair = wide[[match, mismatch]].dropna().reset_index()
        pair['source'] = source
        pair['match_advantage_pct'] = pair[mismatch] - pair[match]
        contrasts.append(pair[['dataset', 'model', 'lambda_value', 'source',
                               'match_advantage_pct']])
    if not contrasts:
        raise ValueError('No same-lambda phase contrasts')
    detail = pd.concat(contrasts, ignore_index=True)
    detail.to_csv(report / 'mase_phase_same_lambda.csv', index=False)
    summary = detail.groupby(['source', 'lambda_value'], as_index=False).agg(
        combinations=('model', 'size'),
        match_better_count=('match_advantage_pct',
                            lambda values: int(values.gt(0).sum())),
        median_match_advantage_pct=('match_advantage_pct', 'median'),
    )
    summary.to_csv(report / 'mase_phase_same_lambda_summary.csv', index=False)
    return summary


def plot_all_lambdas(report):
    """Draw all lambda points and each model's min–max envelope per dataset."""
    report = Path(report)
    table = all_lambda_table(report)
    table.to_csv(report / 'mase_delta_all_lambdas.csv', index=False)
    phase_summary = same_lambda_phase_summary(report, table)
    sensitivity = table.groupby(['dataset', 'model', 'task'], as_index=False).agg(
        min_delta_pct=('delta_pct', 'min'), max_delta_pct=('delta_pct', 'max'))
    sensitivity['lambda_range_pct'] = sensitivity.max_delta_pct - sensitivity.min_delta_pct
    sensitivity.to_csv(report / 'mase_lambda_sensitivity.csv', index=False)
    task_names = dict(TASKS)
    model_colors = dict(zip(MODELS, COLORS))
    images = []
    for dataset, part in table.groupby('dataset', sort=True):
        tasks = [task for task, _ in TASKS if task in set(part.task)]
        if not tasks:
            continue
        models = [model for model in MODELS if model in set(part.model)]
        models += sorted(set(part.model) - set(models))
        fig, ax = plt.subplots(figsize=(13.5, 7.2), layout='constrained')
        positions = {task: i for i, task in enumerate(tasks)}
        for i, model in enumerate(models):
            rows = part[part.model.eq(model)]
            color = model_colors.get(model, plt.cm.tab10(i % 10))
            envelope = rows.groupby('task').delta_pct.agg(['min', 'max'])
            available = [task for task in tasks if task in envelope.index]
            xs = [positions[task] for task in available]
            ax.fill_between(xs, envelope.loc[available, 'min'].to_numpy(),
                            envelope.loc[available, 'max'].to_numpy(),
                            color=color, alpha=0.10, linewidth=0)
            chosen = rows[rows.validation_selected].set_index('task')
            chosen_tasks = [task for task in tasks if task in chosen.index]
            ax.plot([positions[task] for task in chosen_tasks],
                    [chosen.loc[task, 'delta_pct'] for task in chosen_tasks],
                    color=color, linewidth=1.9, alpha=0.90,
                    marker='o', markersize=4,
                    label=DISPLAY_MODELS.get(model, model))
            for weight, marker, offset in LAMBDA_MARKERS:
                at_weight = rows[rows.lambda_value.eq(weight)]
                ax.scatter([positions[task] + offset for task in at_weight.task],
                           at_weight.delta_pct, color=color, marker=marker,
                           s=40, edgecolors='white', linewidths=0.6, zorder=4)
        ax.axhline(0, color='#444444', linestyle='--', linewidth=1.2)
        lower, upper = min(float(part.delta_pct.min()), 0), max(float(part.delta_pct.max()), 0)
        padding = max((upper - lower) * 0.12, 0.15)
        ax.set_ylim(lower - padding, upper + padding)
        ax.set_xlim(-0.2, len(tasks) - 1 + 0.22)
        ax.set_xticks(range(len(tasks)), [task_names[task] for task in tasks])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:+g}%'))
        ax.set_ylabel('Δ test MASE vs Real (%)')
        ax.set_xlabel('Task type')
        ax.set_title(f'{dataset.upper()} — all λ values (line = selected; band = range)', loc='left',
                     fontsize=16, fontweight='bold')
        ax.grid(axis='y', color='#dddddd', linewidth=0.7)
        ax.set_axisbelow(True)
        model_handles, _ = ax.get_legend_handles_labels()
        markers = [Line2D([], [], linestyle='None', marker=marker, markersize=7,
                          markerfacecolor='#555555', markeredgecolor='white',
                          label=f'λ={weight:g}')
                   for weight, marker, _ in LAMBDA_MARKERS]
        ax.legend(handles=model_handles + markers, ncol=len(models) + len(markers),
                  loc='lower center', bbox_to_anchor=(0.5, 1.035), frameon=False)
        name = f'{dataset}_mase_delta_all_lambdas.png'
        fig.savefig(report / name, dpi=200, bbox_inches='tight')
        fig.savefig(report / f'{dataset}_mase_delta_all_lambdas.svg', bbox_inches='tight')
        plt.close(fig)
        images.append((dataset, name))
    lines = [
        '# 全部 λ 的 MASE 相对 Real 变化', '',
        '每个点为同模型、同 seed 的测试 MASE 相对 Real 百分比变化，先按 seed 配对计算，'
        '再取平均；负值表示改善。圆点、方点、三角形分别为 λ=0.5、0.75、0.9，'
        '为便于识别在横轴上略微错开。实线连接按验证 RMSE 选择的 λ；'
        '半透明色带连接每个任务三个 λ 的最小值和最大值，只表示观测范围，'
        '不代表置信区间或任务之间的连续变化。', '',
        '作图数据见 [`mase_delta_all_lambdas.csv`](mase_delta_all_lambdas.csv)。', '',
        '三个 λ 的最大与最小增益之差（百分点）见 '
        '[`mase_lambda_sensitivity.csv`](mase_lambda_sensitivity.csv)；'
        '它描述参数敏感性，不是 seed 不确定性。', '',
        '## 固定 λ 的相位对照', '',
        '在同一数据集、模型和 λ 下计算“不匹配增益 − 匹配增益”；正值表示相位匹配更好。'
        '这是描述性统计，详细差值见 '
        '[`mase_phase_same_lambda.csv`](mase_phase_same_lambda.csv)。', '',
        '| 伙伴来源 | λ | 匹配更好次数 | 匹配优势中位数 |',
        '|---|---:|---:|---:|',
    ]
    for _, row in phase_summary.iterrows():
        lines.append(f'| {row.source} | {row.lambda_value:g} | '
                     f'{int(row.match_better_count)}/{int(row.combinations)} | '
                     f'{row.median_match_advantage_pct:+.2f}% |')
    lines.append('')
    for dataset, name in images:
        lines += [f'## {dataset}', '', f'![{dataset} all-lambda MASE delta]({name})', '']
    (report / 'mase_delta_all_lambdas.md').write_text('\n'.join(lines), encoding='utf-8')
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Experiment output directory containing report/')
    parser.add_argument('--all-lambdas', action='store_true',
                        help='Plot all lambda values with a translucent min–max envelope')
    args = parser.parse_args()
    images = (plot_all_lambdas if args.all_lambdas else plot_report)(args.output / 'report')
    print(f'Wrote {len(images)} dataset plots to {args.output / "report"}')


if __name__ == '__main__':
    main()
