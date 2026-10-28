"""Run real-only, real-real and real-synthetic Mixup experiments."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.utils.datasets import load_real, sha256, write_json
from experiments.utils.constants import ALL_DATASETS
from experiments.utils.evaluate import evaluate_baselines
from experiments.utils.synthetic import load_synthetic
from .engine import run_one

PACKAGE = Path(__file__).resolve().parent
BASE_CONFIG = PACKAGE.parent/'sample_mixed/configs/default.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','smoke','screen','full'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PACKAGE/'configs/default.json')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS)
    parser.add_argument('--models', nargs='+', choices=['linear','dlinear','mlp','gru','transformer'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--modes', nargs='+', choices=['real','real_real','real_synthetic'])
    parser.add_argument('--steps', type=int)
    parser.add_argument('--synthetic-series', type=int)
    parser.add_argument('--windows-per-series', type=int)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config['datasets'] = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))['datasets']
    names = args.datasets or (ALL_DATASETS if args.stage == 'full' else ['exchange'])
    modes = args.modes or ['real','real_real','real_synthetic']
    if args.stage == 'smoke':
        config.update(models=['linear','gru'], seeds=[42], weights=[0.75],
                      steps=50, eval_every=25)
    for key in ['models','seeds','weights','steps','synthetic_series','windows_per_series']:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if config['steps'] < 1 or config['eval_every'] < 1:
        raise ValueError('steps and eval_every must be positive')
    if any(not 0 <= w <= 1 for w in config['weights']):
        raise ValueError('Mixup weights must be in [0, 1]')
    if not 0 <= config['augmentation_probability'] <= 1:
        raise ValueError('Invalid augmentation probability')
    if abs(config['batch_size']*config['augmentation_probability']
           -round(config['batch_size']*config['augmentation_probability'])) > 1e-8:
        raise ValueError('Batch size times augmentation probability must be integer')
    torch.set_num_threads(config['threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    code_hashes = {str(path.relative_to(PACKAGE.parent)):sha256(path)
        for path in [*PACKAGE.glob('*.py'),
            *(PACKAGE.parent/'utils').glob('*.py')]
        if path.name not in ['tests.py','verify.py']}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(config=config, datasets=names, modes=modes, python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, device=str(device),
        code_hashes=code_hashes)
    experiment_path = args.output/'experiment.json'
    if experiment_path.exists():
        old = json.loads(experiment_path.read_text(encoding='utf-8'))
        if old['code_hashes'] != code_hashes:
            raise ValueError('Training code changed; use a new output directory')
    write_json(experiment_path, metadata)
    jobs = [(model,seed,mode, 1.0 if mode == 'real' else weight)
        for model,seed,mode in itertools.product(config['models'],config['seeds'],modes)
        for weight in ([1.0] if mode == 'real' else config['weights'])]
    print(f'Stage={args.stage}, datasets={names}, jobs={len(jobs)*len(names)}, device={device}', flush=True)
    started = time.perf_counter()
    for dataset in names:
        spec = config['datasets'][dataset]
        splits, manifest = load_real(args.data_root,spec,config['normalization_epsilon'])
        write_json(args.output/dataset/'manifest.json', manifest)
        cache_name = f'{spec["frequency"]}_{spec["context"]}_{spec["horizon"]}.npz'
        synthetic, syn_meta = load_synthetic(args.output/'assets'/cache_name,spec,config)
        write_json(args.output/dataset/'synthetic_manifest.json', syn_meta)
        write_json(args.output/dataset/'baselines.json',
            evaluate_baselines(splits,spec,manifest['mase_denominator']))
        print(f'{dataset}: {manifest["window_counts"]}, synthetic={len(synthetic["x"])}',flush=True)
        if args.stage == 'prepare':
            continue
        run_config = {k:v for k,v in config.items()
            if k not in ['models','seeds','weights','datasets']}
        for index,(model,seed,mode,weight) in enumerate(jobs,1):
            output = args.output/dataset/model/mode/f'lambda_{weight:.3f}'/f'seed_{seed}'
            result, skipped = run_one(output,model,mode,weight,seed,spec,run_config,splits,
                synthetic,manifest,device,dict(training=code_hashes,synthetic=syn_meta['signature']))
            print(f'[{index}/{len(jobs)}] {dataset}/{model}/{mode}/λ={weight:g}/seed={seed} '
                f'{"SKIP" if skipped else "DONE"} val={result["checkpoints"]["best"]["validation"]["rmse"]:.6g}',flush=True)
    print(f'Elapsed {(time.perf_counter()-started)/60:.2f} min',flush=True)


if __name__ == '__main__':
    main()
