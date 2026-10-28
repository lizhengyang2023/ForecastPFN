"""Run one paired Mixup cell per process; four cells can occupy four GPUs."""
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
from .engine import run_one
from .partner_index import PartnerIndex
from .period_features import extract
from .diagnostics import feature_summary

PACKAGE = Path(__file__).resolve().parent
BASE_CONFIG = PACKAGE.parent/'sample_mixed/configs/default.json'


def feature_cache(path, real_x, synthetic_x, spec, matching, source_signature, code_hashes):
    signature = dict(real_sha256=source_signature['real'],
        synthetic=source_signature['synthetic'], spec=spec, matching=matching,
        feature_code=code_hashes['mixup_paired/period_features.py'])
    marker = path.with_suffix('.json')
    if path.exists() and marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != signature:
            raise ValueError(f'Feature cache changed: {path}')
        with np.load(path) as saved:
            return ({key[5:]: saved[key] for key in saved.files if key.startswith('real_')},
                    {key[4:]: saved[key] for key in saved.files if key.startswith('syn_')})
    arguments = {key: matching[key] for key in
        ('min_cycles', 'min_period', 'power_fraction', 'min_autocorrelation')}
    real = extract(real_x, spec['season'], **arguments)
    syn = extract(synthetic_x, spec['season'], **arguments)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f'real_{key}': value for key, value in real.items()},
        **{f'syn_{key}': value for key, value in syn.items()})
    write_json(marker, signature)
    return real, syn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'screen', 'full'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, help='Shared read-only synthetic assets prepared first')
    parser.add_argument('--config', type=Path, default=PACKAGE/'configs/default.json')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS)
    parser.add_argument('--models', nargs='+', choices=['linear','dlinear','mlp','gru','transformer'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--synthetic-series', type=int)
    parser.add_argument('--mode', choices=['real_real','real_synthetic'])
    parser.add_argument('--strategy', choices=['period_only','period_phase'])
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    if args.stage != 'prepare' and (args.mode is None or args.strategy is None):
        parser.error('--mode and --strategy are required for training')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config['datasets'] = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))['datasets']
    names = args.datasets or (ALL_DATASETS if args.stage == 'full' else ['etth1'])
    if args.stage == 'smoke':
        config.update(models=['linear','gru'], seeds=[42], weights=[.75],
                      steps=50, eval_every=25)
    for key in ('models','seeds','weights','steps','synthetic_series'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if not config['models'] or not config['seeds'] or not config['weights']:
        parser.error('models, seeds, weights must be nonempty')
    if config['steps'] < 1 or config['eval_every'] < 1:
        parser.error('steps and eval_every must be positive')
    if any(not 0 <= w <= 1 for w in config['weights']):
        parser.error('weights must be in [0,1]')
    if not 0 < config['augmentation_probability'] <= 1:
        parser.error('paired augmentation probability must be in (0,1]')
    if abs(config['batch_size'] * config['augmentation_probability'] -
           round(config['batch_size'] * config['augmentation_probability'])) > 1e-8:
        parser.error('batch_size * augmentation_probability must be integer')
    torch.set_num_threads(config['threads'])
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'auto' else torch.device(args.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error(f'CUDA unavailable: {device}')
    code_hashes = {str(path.relative_to(PACKAGE.parent)).replace('\\','/'): sha256(path)
        for path in [*PACKAGE.glob('*.py'), *(PACKAGE.parent/'utils').glob('*.py'),
                     PACKAGE.parent/'mixup/sampling.py']
        if path.name not in ('tests.py','verify.py','summarize.py')}
    args.output.mkdir(parents=True, exist_ok=True)
    meta_path = args.output/'experiment.json'
    metadata = dict(config=config, datasets=names, mode=args.mode, strategy=args.strategy,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        device=str(device), code_hashes=code_hashes)
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding='utf-8'))
        if (previous['code_hashes'] != code_hashes or previous['config'] != config
                or previous['mode'] != args.mode or previous['strategy'] != args.strategy):
            raise ValueError('Code or config changed; use another output directory')
    write_json(meta_path, metadata)
    jobs = list(itertools.product(config['models'], config['seeds'], config['weights']))
    for dataset in names:
        spec = config['datasets'][dataset]
        splits, manifest = load_real(args.data_root, spec, config['normalization_epsilon'])
        cache = (args.assets or args.output/'assets') / f'{spec["frequency"]}_{spec["context"]}_{spec["horizon"]}.npz'
        if args.assets is not None and (not cache.exists() or not cache.with_suffix('.json').exists()):
            raise FileNotFoundError(f'Run --stage prepare before parallel jobs: {cache}')
        synthetic, syn_meta = load_synthetic(cache, spec, config)
        write_json(args.output/dataset/'manifest.json', manifest)
        write_json(args.output/dataset/'synthetic_manifest.json', syn_meta)
        write_json(args.output/dataset/'baselines.json',
            evaluate_baselines(splits, spec, manifest['mase_denominator']))
        if args.stage == 'prepare':
            print(f'PREPARED {dataset}: {len(synthetic["x"])} synthetic windows', flush=True)
            continue
        matching = config['matching']
        real_features, syn_features = feature_cache(
            args.output/'features'/f'{dataset}.npz', splits['train']['x'], synthetic['x'],
            spec, matching, dict(real=manifest['sha256'], synthetic=syn_meta['signature']),
            code_hashes)
        write_json(args.output/dataset/'feature_diagnostics.json',
            feature_summary(real_features, syn_features))
        features = real_features if args.mode == 'real_real' else syn_features
        index = PartnerIndex(real_features, features,
            origins=splits['train']['origins'] if args.mode == 'real_real' else None,
            same_source=args.mode == 'real_real', horizon=spec['horizon'],
            overlap_gap=spec['context'] + spec['horizon'],
            **{key: matching[key] for key in ('period_tolerance','phase_tolerance',
                                                'drift_tolerance','top_k')})
        run_config = {key: value for key, value in config.items()
            if key not in ('models','seeds','weights','datasets')}
        identity = dict(training=code_hashes, synthetic=syn_meta['signature'],
                        matching=matching)
        for number, (model, seed, weight) in enumerate(jobs, 1):
            path = args.output/dataset/model/args.mode/args.strategy/f'lambda_{weight:.3f}'/f'seed_{seed}'
            result, skipped = run_one(path, model, args.mode, args.strategy, weight,
                seed, spec, run_config, splits, synthetic, manifest, device, identity, index)
            print(f'[{number}/{len(jobs)}] {dataset} {args.mode}/{args.strategy} '
                f'{model} λ={weight:g} seed={seed} {"SKIP" if skipped else "DONE"} '
                f'val={result["checkpoints"]["best"]["validation"]["rmse"]:.6g}', flush=True)


if __name__ == '__main__':
    main()
