"""Summarize strategy effects and pool statistics without test-set selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(root):
    root = Path(root)
    report = root/'report'
    report.mkdir(exist_ok=True)
    rows = []
    for path in sorted(root.glob('*/*/*/p_*/seed_*/metrics.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        d,s = path.parents[4].name, path.parents[3].name
        for checkpoint, values in r['checkpoints'].items():
            rows.append(dict(dataset=d, strategy=s, model=r['model'], seed=r['seed'],
                             fraction=r['fraction'], checkpoint=checkpoint,
                             validation_rmse=values['validation']['rmse'],
                             test_rmse=values['test']['rmse'], test_mae=values['test']['mae'],
                             test_mase=values['test']['mase'], best_step=r['best_step'],
                             real_exposures=r['exposures']['real'], synthetic_exposures=r['exposures']['synthetic'],
                             train_seconds=r['train_seconds']))
    if not rows:
        raise ValueError('No completed corpus runs')
    runs = pd.DataFrame(rows)
    runs.to_csv(report/'runs.csv', index=False)
    keys = ['dataset','strategy','model','fraction','checkpoint']
    aggregate = runs.groupby(keys, as_index=False).agg(
        validation_rmse_mean=('validation_rmse','mean'), test_rmse_mean=('test_rmse','mean'),
        test_mase_mean=('test_mase','mean'), test_mase_std=('test_mase','std'),
        seeds=('seed','nunique'), train_seconds_mean=('train_seconds','mean'))
    aggregate.to_csv(report/'aggregate.csv', index=False)
    best = aggregate[aggregate.checkpoint == 'best']
    selected = best.sort_values(['validation_rmse_mean','strategy','model','fraction']).drop_duplicates('dataset')
    selected.to_csv(report/'validation_selected.csv', index=False)
    per_strategy = best.sort_values(['validation_rmse_mean','model','fraction']).drop_duplicates(['dataset','strategy'])
    per_strategy.to_csv(report/'validation_selected_by_strategy.csv', index=False)
    paired = runs[runs.checkpoint == 'best'].pivot_table(index=['dataset','model','seed','fraction'],
        columns='strategy', values='test_mase').reset_index()
    if 'original' in paired:
        for strategy in ('filtered','parameter','parameter_filtered'):
            if strategy in paired:
                paired[f'{strategy}_minus_original'] = paired[strategy] - paired['original']
    paired.to_csv(report/'paired_mase.csv', index=False)
    pools = []
    for path in sorted(root.glob('*/*_corpus_manifest.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        pools.append(dict(dataset=path.parent.name, strategy=meta.get('strategy','original'),
                          candidates=meta.get('candidate_windows',meta['pool_windows']),
                          pool_windows=meta['pool_windows'], independent_series=meta['independent_series'],
                          distance_before=meta.get('distance_before'), distance_after=meta.get('distance_after'),
                          generation_seconds=meta['generation_seconds'],
                          multipliers=json.dumps(meta.get('multipliers',{}))))
    pd.DataFrame(pools).to_csv(report/'corpus_stats.csv', index=False)
    for dataset, part in best.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(9,5))
        for (strategy, model), group in part.groupby(['strategy','model']):
            group = group.sort_values('fraction')
            ax.plot(group.fraction, group.test_mase_mean, marker='o', label=f'{strategy}/{model}')
        ax.set(xlabel='Synthetic exposure fraction', ylabel='Test MASE', title=f'{dataset}: independent synthetic corpora')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, bbox_to_anchor=(1.02,1), loc='upper left')
        fig.tight_layout()
        fig.savefig(report/f'{dataset}_mase.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    stats = pd.DataFrame(pools)
    lines = ['# 统计匹配合成池实验', '',
             f'完成 {len(runs)//2} 次训练。所有方案与比例只按真实验证 RMSE 选择；测试集仅用于报告。',
             '逐 run、聚合、配对 MASE 和池统计分别见 `runs.csv`、`aggregate.csv`、`paired_mase.csv`、`corpus_stats.csv`。', '']
    for _, row in per_strategy.iterrows():
        lines.append(f'- {row.dataset}/{row.strategy}: 验证选择 {row.model}, p={row.fraction:g}; '
                     f'测试 MASE={row.test_mase_mean:.4g}。')
    for dataset in sorted(runs.dataset.unique()):
        lines += ['', f'![{dataset} MASE]({dataset}_mase.png)']
    (report/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


