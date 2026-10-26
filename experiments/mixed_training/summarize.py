"""CSV, Markdown and PNG reports; choose proportions by validation, never test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(root):
    root=Path(root)
    report_dir=root/'report'
    report_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    locations={}
    for path in sorted(root.glob('*/*/p_*/seed_*/metrics.json')):
        r=json.loads(path.read_text())
        dataset=path.parents[3].name
        locations[(dataset,r['model'],r['fraction'],r['seed'])]=path.parent
        for checkpoint,parts in r['checkpoints'].items():
            record=dict(dataset=dataset,model=r['model'],fraction=r['fraction'],seed=r['seed'],
                checkpoint=checkpoint,best_step=r['best_step'],parameters=r['parameters'],
                train_seconds=r['train_seconds'],validation_seconds=r['validation_seconds'],
                total_seconds=r['total_seconds'],real_exposures=r['exposures']['real'],
                synthetic_exposures=r['exposures']['synthetic'])
            for split in ['validation','test']:
                for metric in ['mae','rmse','mse','mase']:
                    record[f'{split}_{metric}']=parts[split][metric]
            records.append(record)
    if not records:
        raise ValueError('No completed runs')
    runs=pd.DataFrame(records)
    runs.to_csv(report_dir/'runs.csv',index=False)
    keys=['dataset','model','fraction','checkpoint']
    metric_names=['validation_mae','validation_rmse','test_mae','test_rmse','test_mase','total_seconds']
    grouped=runs.groupby(keys)[metric_names].agg(['mean','std','count'])
    grouped.columns=['_'.join(c) for c in grouped.columns]
    aggregate=grouped.reset_index()
    aggregate.to_csv(report_dir/'aggregate.csv',index=False)
    selections=[]
    for (dataset,model,checkpoint),part in aggregate.groupby(['dataset','model','checkpoint']):
        # Mean validation RMSE across seeds determines one proportion per model.
        selected=part.sort_values(['validation_rmse_mean','fraction']).iloc[0].to_dict()
        real_only=part[part.fraction==0]
        selected['relative_rmse_improvement_percent']=None
        if len(real_only) and real_only.iloc[0].test_rmse_mean>0:
            selected['relative_rmse_improvement_percent']=100*(1-selected['test_rmse_mean']/real_only.iloc[0].test_rmse_mean)
        selections.append(selected)
    selected=pd.DataFrame(selections)
    selected.to_csv(report_dir/'validation_selected.csv',index=False)
    lines=['# Mixed real/synthetic training report','',
        f'Completed runs: {len(runs)//2}. Both final and validation-best checkpoints are reported.',
        'Proportions are selected by mean REAL validation RMSE across seeds, not by test ranking.',
        'Test metrics are in original target units. Error bars are sample standard deviations across training seeds, not confidence intervals.',
        'The 100% synthetic group still uses target-domain validation for checkpoint selection; it is not a strict untouched-domain zero-shot benchmark.','']
    for dataset,part in aggregate.groupby('dataset'):
        baseline_path=root/dataset/'baselines.json'
        baselines=json.loads(baseline_path.read_text())['test'] if baseline_path.exists() else {}
        models=sorted(part.model.unique())
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for col,checkpoint in enumerate(['final','best']):
            for row,metric in enumerate(['mae','rmse']):
                ax=axes[row,col]
                for model in models:
                    d=part[(part.model==model)&(part.checkpoint==checkpoint)].sort_values('fraction')
                    ax.errorbar(d.fraction*100,d[f'test_{metric}_mean'],
                        yerr=d[f'test_{metric}_std'].fillna(0),marker='o',capsize=2,label=model)
                for baseline in ['last','seasonal_naive']:
                    if baseline in baselines:
                        ax.axhline(baselines[baseline][metric],linestyle='--' if baseline=='last' else ':',
                                   color='0.4',label=baseline)
                ax.set(title=f'{dataset}: {checkpoint} checkpoint',xlabel='Synthetic exposure (%)',
                       ylabel=f'Test {metric.upper()} (original units)',xticks=np.arange(0,101,12.5))
                ax.tick_params(axis='x',labelrotation=45)
        axes[0,0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(report_dir/f'{dataset}_ratios.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        d=selected[selected.dataset==dataset]
        for ax,metric in zip(axes,['mae','rmse']):
            for j,checkpoint in enumerate(['final','best']):
                values=d[d.checkpoint==checkpoint].set_index('model').loc[models]
                ax.bar(np.arange(len(models))+(j-.5)*.35,values[f'test_{metric}_mean'],width=.35,
                       yerr=values[f'test_{metric}_std'].fillna(0),capsize=3,label=checkpoint)
            ax.set(xticks=np.arange(len(models)),xticklabels=models,ylabel=f'Test {metric.upper()} (original units)',
                   title=f'{dataset}: validation-selected proportion per model/checkpoint')
            ax.legend()
        fig.tight_layout();fig.savefig(report_dir/f'{dataset}_models.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(len(models),2,figsize=(12,2.7*len(models)),squeeze=False)
        for i,model in enumerate(models):
            p=float(d[(d.model==model)&(d.checkpoint=='best')].iloc[0].fraction)
            seed=int(runs[(runs.dataset==dataset)&(runs.model==model)].seed.min())
            location=locations[(dataset,model,p,seed)]
            log=pd.read_csv(location/'train_log.csv')
            for field,label in [('train_loss','mixed'),('real_loss','real'),('synthetic_loss','synthetic')]:
                if log[field].notna().any():
                    axes[i,0].plot(log.step,log[field],label=label)
            axes[i,0].set(title=f'{model} p={p:g}, seed={seed}',xlabel='Update step',ylabel='Normalized training MSE')
            axes[i,0].legend(fontsize=7)
            axes[i,1].plot(log.step,log.validation_rmse)
            axes[i,1].set(title='Fixed real validation set',xlabel='Update step',ylabel='RMSE (original units)')
        fig.tight_layout();fig.savefig(report_dir/f'{dataset}_learning.png',dpi=140);plt.close(fig)
        fig,axes=plt.subplots(len(models),1,figsize=(11,2.3*len(models)),squeeze=False)
        for i,model in enumerate(models):
            p=float(d[(d.model==model)&(d.checkpoint=='best')].iloc[0].fraction)
            seed=int(runs[(runs.dataset==dataset)&(runs.model==model)].seed.min())
            with np.load(locations[(dataset,model,p,seed)]/'predictions.npz') as saved:
                h=np.arange(1,saved['truth'].shape[1]+1)
                axes[i,0].plot(h,saved['truth'][0],color='black',label='Observed future')
                for label in ['final','best']:
                    axes[i,0].plot(h,saved[label][0],label=label)
                axes[i,0].set(title=f'{model}, p={p:g}; first test origin (not chosen by quality)',
                              xlabel='Forecast horizon (steps)',ylabel='Target (original units)')
                axes[i,0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(report_dir/f'{dataset}_predictions.png',dpi=140);plt.close(fig)
        lines += [f'## {dataset}','', 'Validation-selected settings (mean ± seed SD):','']
        for _,r in d.iterrows():
            std=r.test_rmse_std
            std_text=f'{std:.6g}' if pd.notna(std) else 'n/a (one seed)'
            lines.append(f'- {r.model}, {r.checkpoint}: p={r.fraction:.3f}, test MAE={r.test_mae_mean:.6g}, '
                         f'RMSE={r.test_rmse_mean:.6g} ± {std_text}, seeds={int(r.test_rmse_count)}.')
        lines += ['', 'Training-free baselines:','']
        for name,m in baselines.items():
            lines.append(f'- {name}: MAE={m["mae"]:.6g}, RMSE={m["rmse"]:.6g}.')
        lines += ['',f'![Ratios]({dataset}_ratios.png)','',f'![Model comparison]({dataset}_models.png)','',
                  f'![Learning curves]({dataset}_learning.png)','',f'![Forecast examples]({dataset}_predictions.png)','']
    for cache in sorted((root/'assets').glob('*.npz')):
        with np.load(cache) as samples:
            fig,axes=plt.subplots(3,2,figsize=(11,7))
            for i,ax in enumerate(axes.flat):
                ax.plot(samples['clean_examples'][i],label='Clean')
                ax.plot(samples['examples'][i],alpha=.7,label='Observed')
                ax.set(title=f'Original ForecastPFN generator: series {i}',xlabel='Time index (configured frequency)',ylabel='Synthetic value')
            axes[0,0].legend();fig.tight_layout()
            file=f'synthetic_{cache.stem}.png'
            fig.savefig(report_dir/file,dpi=140);plt.close(fig)
            lines += [f'![Synthetic examples]({file})','']
    lines += ['## Interpretation limits','',
        'Every dataset trains its own models. All proportions share the full real training pool and a fixed synthetic pool; only exposure shares vary. ',
        'Fixed batch size and update count do not equalize compute across architectures. High synthetic shares reduce real exposure; attribution requires separate exposure-controlled experiments.',
        'Seeds quantify optimization/sampling variation; overlapping test windows are not independent observations. No significance claim or independent-window confidence interval is made.',
        'These are designated single-target tasks, not the original paper aggregate or multivariate benchmark. Weather uses complete hourly means.',
        'See runs.csv for every seed, aggregate.csv for all ratios, and validation_selected.csv for validation-selected settings.','']
    (report_dir/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(report_dir/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.output)
