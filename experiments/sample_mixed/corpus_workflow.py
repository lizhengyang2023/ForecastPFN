"""Independent-window corpus experiment, with deterministic job shards."""
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
from .corpus import STRATEGIES, build
from .engine import run_one

PACKAGE = Path(__file__).parent


def configuration(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config.update(json.loads((PACKAGE/'configs/corpus.json').read_text(encoding='utf-8')))
    for key in ('models', 'seeds', 'fractions', 'steps', 'synthetic_series',
                'windows_per_series', 'candidate_multiplier'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.stage == 'smoke' or args.smoke_config:
        config.update(steps=8, eval_every=4, seeds=[42], fractions=[0, .5],
                      synthetic_series=8, candidate_multiplier=3, calibration_series=4)
    names = args.datasets or (['etth1'] if args.stage == 'smoke' or args.smoke_config else
                              ['etth1', 'weather', 'exchange'] if args.stage == 'screen' else ALL_DATASETS)
    strategies = args.strategies or list(STRATEGIES)
    if config['synthetic_series'] < 1 or config['candidate_multiplier'] < 1:
        raise ValueError('Pool sizes must be positive')
    return config, names, strategies


def jobs(config, names, strategies):
    return list(itertools.product(names, strategies, config['models'], config['seeds'], config['fractions']))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['prepare', 'smoke', 'screen', 'full'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=PACKAGE/'configs/default.json')
    p.add_argument('--datasets', nargs='+', choices=ALL_DATASETS)
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES)
    p.add_argument('--models', nargs='+', choices=['linear','dlinear','mlp','gru','transformer'])
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--fractions', nargs='+', type=float)
    p.add_argument('--steps', type=int)
    p.add_argument('--synthetic-series', type=int)
    p.add_argument('--windows-per-series', type=int)
    p.add_argument('--candidate-multiplier', type=int)
    p.add_argument('--device', default='auto')
    p.add_argument('--shard-id', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--smoke-config', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config, names, strategies = configuration(args)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Invalid shard')
    from experiments.utils.sampler import quota
    for fraction in config['fractions']:
        quota(config['batch_size'], fraction)
    torch.set_num_threads(config['threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    identity = {str(p.relative_to(PACKAGE.parent)): sha256(p) for p in sorted([
        *PACKAGE.glob('*.py'), *(PACKAGE.parent/'utils').glob('*.py')])
        if p.name not in ('summarize.py', 'verify.py', 'tests.py')}
    identity['sample_mixed/configs/corpus.json'] = sha256(PACKAGE/'configs/corpus.json')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output/'corpus_experiment.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding='utf-8'))
        if old['code_hashes'] != identity or old['config'] != config or old['datasets'] != names or old['strategies'] != strategies:
            raise ValueError('Experiment changed; use a new output directory')
    elif args.stage != 'prepare' and args.num_shards > 1:
        raise ValueError('Run prepare before sharded training')
    else:
        write_json(metadata_path, dict(config=config, datasets=names, strategies=strategies,
                                       code_hashes=identity, python=platform.python_version(), torch=torch.__version__))
    pools = {}
    real_data = {}
    for dataset in names:
        spec = config['datasets'][dataset]
        splits, manifest = load_real(args.data_root, spec, config['normalization_epsilon'])
        real_data[dataset] = (splits, manifest)
        dataset_dir = args.output/dataset
        if args.stage == 'prepare':
            write_json(dataset_dir/'manifest.json', manifest)
            write_json(dataset_dir/'baselines.json', evaluate_baselines(splits, spec, manifest['mase_denominator']))
        elif not (dataset_dir/'manifest.json').exists():
            raise ValueError('Run prepare first')
        if json.loads((dataset_dir/'manifest.json').read_text(encoding='utf-8'))['sha256'] != manifest['sha256']:
            raise ValueError('Real dataset changed')
        for strategy in strategies:
            cache = args.output/'assets'/f'{dataset}_{strategy}.npz'
            if args.stage != 'prepare' and not cache.exists():
                raise ValueError(f'Missing prepared pool: {cache}')
            pool, meta = build(cache, spec, config, splits['train']['x'], strategy)
            pools[(dataset, strategy)] = (pool, meta)
            if args.stage == 'prepare':
                write_json(dataset_dir/f'{strategy}_corpus_manifest.json', meta)
                print(f'{dataset}/{strategy}: {meta["pool_windows"]} windows', flush=True)
    if args.stage == 'prepare':
        return
    all_jobs = jobs(config, names, strategies)
    for index, (dataset, strategy, model, seed, fraction) in enumerate(all_jobs):
        if index % args.num_shards != args.shard_id:
            continue
        spec = config['datasets'][dataset]
        splits, manifest = real_data[dataset]
        pool, pool_meta = pools[(dataset, strategy)]
        output = args.output/dataset/strategy/model/f'p_{fraction:.3f}'/f'seed_{seed}'
        run_config = {k:v for k,v in config.items() if k not in ('models','seeds','fractions','datasets')}
        result, skipped = run_one(output, model, fraction, seed, spec, run_config,
                                  splits, pool, manifest, device,
                                  dict(training=identity, synthetic=pool_meta['signature']))
        print(f'[{index+1}/{len(all_jobs)}] {dataset}/{strategy}/{model}/{seed}/{fraction:g} '
              f'{"SKIP" if skipped else "DONE"} val={result["checkpoints"]["best"]["validation"]["rmse"]:.6g}', flush=True)


