"""Unified controlled Mixup runner: prepare once, then run disjoint task shards."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import torch

from experiments.utils.constants import ALL_DATASETS
from experiments.utils.datasets import load_real, sha256, write_json
from experiments.utils.evaluate import evaluate_baselines
from experiments.utils.synthetic import load_synthetic
from .controlled import TASKS, build_pools, load_pools, save_pools
from .engine import run_one

PACKAGE = Path(__file__).resolve().parent
BASE_CONFIG = PACKAGE.parent/'sample_mixed/configs/default.json'
TRAINING_FILES = ('run.py', 'engine.py', 'sampling.py', 'controlled.py', 'period_features.py')
DEFAULT_CONTROLLED_DATASETS = ('etth1','etth2','electricity','traffic','weather')


def configuration(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config['datasets'] = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))['datasets']
    names = args.datasets or (list(DEFAULT_CONTROLLED_DATASETS) if args.stage == 'full' else ['etth1'])
    if args.stage == 'smoke' or getattr(args,'smoke_config',False):
        config.update(models=['linear','gru'], seeds=[42], weights=[.75],
                      steps=50, eval_every=25, synthetic_series=32)
    for key in ('models','seeds','weights','steps','synthetic_series','windows_per_series'):
        value = getattr(args,key)
        if value is not None:
            config[key] = value
    if config['steps'] < 1 or config['eval_every'] < 1:
        raise ValueError('steps and eval_every must be positive')
    if not 0 < config['augmentation_probability'] <= 1:
        raise ValueError('augmentation_probability must be in (0,1]')
    if abs(config['batch_size']*config['augmentation_probability']-
           round(config['batch_size']*config['augmentation_probability'])) > 1e-8:
        raise ValueError('Noninteger augmentation quota')
    if any(not 0 <= w <= 1 for w in config['weights']):
        raise ValueError('Mixup weights must be in [0,1]')
    tasks = args.tasks or TASKS
    if any(task not in TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError(f'Unknown or duplicate task: {tasks}')
    if not tasks or not config['models'] or not config['seeds'] or not config['weights']:
        raise ValueError('Empty task grid')
    for label, values in (('datasets',names),('models',config['models']),
                          ('seeds',config['seeds']),('weights',config['weights'])):
        if len(set(values)) != len(values):
            raise ValueError(f'Duplicate {label} in task grid')
    return config, names, tasks


def jobs(config, names, tasks):
    return [(dataset, model, seed, weight, task)
        for dataset, model, seed, task in itertools.product(
            names, config['models'], config['seeds'], tasks)
        for weight in ([1.] if task == 'real' else config['weights'])]


def identity():
    paths = [PACKAGE/name for name in TRAINING_FILES]
    paths += sorted((PACKAGE.parent/'utils').glob('*.py'))
    return {str(path.relative_to(PACKAGE.parent)).replace('\\','/'):sha256(path)
            for path in paths}


def prepare(root, data_root, config, names, tasks, code_hashes):
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'experiment.json'
    metadata = dict(config=config, datasets=names, tasks=list(tasks), code_hashes=code_hashes,
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    if marker.exists():
        previous=json.loads(marker.read_text(encoding='utf-8'))
        if (previous['code_hashes'] != code_hashes or previous['config'] != config
                or previous['datasets'] != names or previous['tasks'] != list(tasks)):
            raise ValueError('Prepared code or config changed; use a new output directory')
        for dataset in names:
            spec=config['datasets'][dataset]
            cache=root/'assets'/f'{spec["frequency"]}_{spec["context"]}_{spec["horizon"]}.npz'
            if not cache.exists() or not (root/'assets'/f'{dataset}_controlled.npz').exists():
                raise FileNotFoundError(f'Incomplete prepared assets for {dataset}')
        print('PREPARED assets already match this run',flush=True)
        return
    for dataset in names:
        spec = config['datasets'][dataset]
        splits, manifest = load_real(data_root,spec,config['normalization_epsilon'])
        cache_name = f'{spec["frequency"]}_{spec["context"]}_{spec["horizon"]}.npz'
        synthetic, syn_meta = load_synthetic(root/'assets'/cache_name,spec,config)
        eligible, pools, summary = build_pools(splits['train']['x'], synthetic['x'],
            splits['train']['origins'], spec, config['matching'])
        write_json(root/dataset/'support.json',summary)
        if not len(eligible):
            raise ValueError(f'{dataset}: no common-support anchors under the frozen matching rules; '
                f'see {root/dataset/"support.json"}')
        if summary['low_support']:
            print(f'WARNING {dataset}: only {summary["eligible_fraction"]:.1%} common support',flush=True)
        save_pools(root/'assets'/f'{dataset}_controlled.npz',eligible,pools)
        write_json(root/dataset/'manifest.json',manifest)
        write_json(root/dataset/'synthetic_manifest.json',syn_meta)
        write_json(root/dataset/'baselines.json',
            evaluate_baselines(splits,spec,manifest['mase_denominator']))
        print(f'PREPARED {dataset}: eligible={len(eligible)}/{len(splits["train"]["x"])}',flush=True)
    write_json(marker,metadata)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','smoke','screen','full'],required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PACKAGE/'configs/default.json')
    parser.add_argument('--datasets',nargs='+',choices=ALL_DATASETS)
    parser.add_argument('--models',nargs='+',choices=['linear','dlinear','mlp','gru','transformer'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--weights',nargs='+',type=float)
    parser.add_argument('--tasks',nargs='+',choices=TASKS)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--synthetic-series',type=int)
    parser.add_argument('--windows-per-series',type=int)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--shard-id',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    parser.add_argument('--smoke-config',action='store_true',help='Apply smoke defaults during prepare')
    args=parser.parse_args(argv)
    config,names,tasks=configuration(args)
    if not 0 <= args.shard_id < args.num_shards:
        parser.error('Expected 0 <= shard-id < num-shards')
    torch.set_num_threads(config['threads'])
    code_hashes=identity()
    if args.stage=='prepare':
        prepare(args.output,args.data_root,config,names,tasks,code_hashes)
        return
    marker=args.output/'experiment.json'
    if not marker.exists():
        raise FileNotFoundError(f'Run --stage prepare first: {marker}')
    previous=json.loads(marker.read_text(encoding='utf-8'))
    if (previous['config'] != config or previous['datasets'] != names or previous['tasks'] != list(tasks)
            or previous['code_hashes'] != code_hashes):
        raise ValueError('Prepared config or code changed; use a new output directory')
    device=(torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if args.device=='auto' else torch.device(args.device))
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error(f'CUDA unavailable: {device}')
    grid=jobs(config,names,tasks)
    assigned=[job for i,job in enumerate(grid) if i % args.num_shards==args.shard_id]
    print(f'SHARD {args.shard_id}/{args.num_shards}: {len(assigned)}/{len(grid)} jobs on {device}',flush=True)
    run_config={k:v for k,v in config.items() if k not in ('models','seeds','weights','datasets')}
    by_dataset={name:[] for name in names}
    for job in assigned:
        by_dataset[job[0]].append(job)
    for dataset, subset in by_dataset.items():
        if not subset:
            continue
        spec=config['datasets'][dataset]
        splits,manifest=load_real(args.data_root,spec,config['normalization_epsilon'])
        cache=args.output/'assets'/f'{spec["frequency"]}_{spec["context"]}_{spec["horizon"]}.npz'
        synthetic,syn_meta=load_synthetic(cache,spec,config)
        eligible,pools=load_pools(args.output/'assets'/f'{dataset}_controlled.npz')
        key=dict(training=code_hashes,synthetic=syn_meta['signature'],
                 support_sha256=sha256(args.output/'assets'/f'{dataset}_controlled.npz'))
        for number,(_,model,seed,weight,task) in enumerate(subset,1):
            path=args.output/dataset/model/task/f'lambda_{weight:.3f}'/f'seed_{seed}'
            result,skipped=run_one(path,model,task,weight,seed,spec,run_config,
                splits,synthetic,manifest,device,key,eligible,pools)
            print(f'[{number}/{len(subset)}] {dataset}/{model}/{task}/seed={seed}/lambda={weight:g} '
                  f'{"SKIP" if skipped else "DONE"} val={result["checkpoints"]["best"]["validation"]["rmse"]:.6g}',flush=True)


if __name__=='__main__':
    main()
