"""One report for real-only and the four controlled Mixup tasks."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .plot_mase_delta import plot_report


def summarize(root):
    root=Path(root)
    rows=[]
    for path in sorted(root.glob('*/*/*/lambda_*/seed_*/metrics.json')):
        result=json.loads(path.read_text(encoding='utf-8'))
        dataset,model,task=(path.parents[i].name for i in (4,3,2))
        for checkpoint,values in result['checkpoints'].items():
            row=dict(dataset=dataset,model=model,task=task,weight=result['weight'],
                     seed=result['seed'],checkpoint=checkpoint,
                     checkpoint_step=values['step'],eligible_anchors=result['eligible_anchors'],
                     elapsed_seconds=result['elapsed_seconds'])
            for split in ('validation','test'):
                for metric in ('mae','rmse','mse','mase'):
                    row[f'{split}_{metric}']=values[split][metric]
            row.update(result['diagnostics'])
            rows.append(row)
    if not rows:
        raise ValueError('No completed controlled Mixup runs')
    report=root/'report'
    report.mkdir(exist_ok=True)
    runs=pd.DataFrame(rows)
    key=['dataset','model','task','weight','checkpoint']
    if runs.duplicated(key+['seed']).any():
        raise ValueError('Duplicate task/seed/checkpoint')
    runs.to_csv(report/'runs.csv',index=False)
    fields=['validation_rmse','test_mae','test_rmse','test_mase']
    grouped=runs.groupby(key)[fields].agg(['mean','std','count'])
    grouped.columns=['_'.join(c) for c in grouped.columns]
    aggregate=grouped.reset_index()
    aggregate.to_csv(report/'aggregate.csv',index=False)
    selected=[]
    for (dataset,model,task,checkpoint),part in aggregate.groupby(
            ['dataset','model','task','checkpoint']):
        baseline=runs[(runs.dataset==dataset)&(runs.model==model)&
                      (runs.task=='real')&(runs.checkpoint==checkpoint)]
        required=set(baseline.seed)
        eligible=[]
        for _,candidate in part.iterrows():
            seeds=set(runs[(runs.dataset==dataset)&(runs.model==model)&
                (runs.task==task)&(runs.weight==candidate.weight)&
                (runs.checkpoint==checkpoint)].seed)
            if required and seeds==required:
                eligible.append(candidate)
        if eligible:
            best=min(eligible,key=lambda r:(r.validation_rmse_mean,r.weight))
            selected.append(best.to_dict())
    selected=pd.DataFrame(selected)
    selected.to_csv(report/'validation_selected.csv',index=False)
    base=runs[runs.task=='real'][['dataset','model','seed','checkpoint','test_mase']].rename(
        columns={'test_mase':'real_test_mase'})
    paired=runs.merge(base,on=['dataset','model','seed','checkpoint'],how='left',
                      validate='many_to_one')
    paired['delta_mase_vs_real']=paired.test_mase-paired.real_test_mase
    for source in ('real_real','real_synthetic'):
        matched=runs[runs.task==f'{source}_phase_match'][
            ['dataset','model','seed','checkpoint','weight','test_mase']].rename(
            columns={'test_mase':f'{source}_matched_mase'})
        paired=paired.merge(matched,on=['dataset','model','seed','checkpoint','weight'],
                            how='left',validate='many_to_one')
        paired[f'delta_mase_vs_{source}_match']=paired.test_mase-paired[f'{source}_matched_mase']
    paired.to_csv(report/'paired_differences.csv',index=False)
    delta_images=plot_report(report)
    lines=['# 受控相位 Mixup 汇总','',
           f'已完成 {len(runs)//2} 次运行；{runs.dataset.nunique()} 个数据集。',
           '所有任务使用同一历史可靠、四类伙伴均可用的真实锚点池；无周期不匹配的回退。',
           '相位匹配与不匹配均先通过相同周期和未来漂移门槛；Real-only 使用同一锚点池。',
           'best checkpoint 和每个任务的 λ 均按真实验证 RMSE 选择；测试集只用于报告。',
           'MASE 越低越好；± 为训练 seed 标准差。详细结果见 runs.csv、aggregate.csv、'
           'validation_selected.csv、paired_differences.csv。','',
           f'各数据集相对 Real 的 MASE 百分比折线图（{len(delta_images)} 张）见 '
           '[mase_delta.md](mase_delta.md)。','']
    if (report/'mase_delta_all_lambdas.md').exists():
        lines += ['全部 λ 的三点及范围带图见 '
                  '[mase_delta_all_lambdas.md](mase_delta_all_lambdas.md)。','']
    if not selected.empty:
        for dataset,part in selected[selected.checkpoint=='best'].groupby('dataset'):
            support=json.loads((root/dataset/'support.json').read_text(encoding='utf-8'))
            lines += [f'## {dataset}','',
                f'共同可用锚点：{support["eligible_anchors"]}/{support["total_real_windows"]} '
                f'({100*support["eligible_fraction"]:.1f}%)。','',
                '| 模型 | 任务 | λ | 验证 RMSE | 测试 MASE 均值 ± SD |',
                '|---|---|---:|---:|---:|']
            for _,row in part.sort_values(['model','task']).iterrows():
                sd=row.test_mase_std
                spread=f'{sd:.4f}' if pd.notna(sd) else 'n/a'
                lines.append(f'| {row.model} | {row.task} | {row.weight:g} | '
                    f'{row.validation_rmse_mean:.4f} | {row.test_mase_mean:.4f} ± {spread} |')
            lines.append('')
    (report/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(report/'report.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.output)


if __name__=='__main__':
    main()
