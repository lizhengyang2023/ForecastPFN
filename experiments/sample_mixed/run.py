"""CLI for smoke, screening, full-grid preparation/execution and cost estimates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.utils.datasets import load_real, sha256, write_json
from experiments.utils.constants import ALL_DATASETS
from .engine import run_one
from experiments.utils.evaluate import evaluate_baselines
from experiments.utils.synthetic import load_synthetic

PACKAGE = Path(__file__).resolve().parent
def estimate(config, names, root, profile, output):
    """Extrapolate measured smoke training and evaluation separately; not a promise."""
    measured = []
    for path in Path(profile).glob('*/*/p_*/seed_*/metrics.json'):
        result = json.loads(path.read_text())
        manifest = json.loads((path.parent/'split_manifest.json').read_text())
        result['spec'] = manifest['specification']
        result['counts'] = manifest['window_counts']
        measured.append(result)
    if not measured:
        raise ValueError('No completed profile runs found')
    rows = []
    for dataset in names:
        spec = config['datasets'][dataset]
        _, manifest = load_real(root, spec, config['normalization_epsilon'])
        for model in config['models']:
            samples = [r for r in measured if r['model']==model]
            if not samples:
                raise ValueError(f'No timing profile for {model}')
            rates = []
            for r in samples:
                length_factor = spec['context']/r['spec']['context']
                # Attention's sequence cost grows faster; conservative shape heuristic.
                factor = length_factor**2 if model=='transformer' else length_factor
                training = r['train_seconds']/r['final_step']*config['steps']*factor
                evals = (config['steps']+config['eval_every']-1)//config['eval_every']
                old_config = json.loads(next(Path(profile).glob(
                    f'*/{model}/p_{r["fraction"]:.3f}/seed_{r["seed"]}/config.json')).read_text())['config']
                old_evals = (r['final_step']+old_config['eval_every']-1)//old_config['eval_every']
                validation = r['validation_seconds']/old_evals*evals*manifest['window_counts']['validation']/r['counts']['validation']*factor
                final_eval = r['test_and_save_seconds']*manifest['window_counts']['test']/r['counts']['test']*factor
                rates.append(training+validation+final_eval)
            runs = len(config['seeds'])*len(config['fractions'])
            median = float(np.median(rates))
            rows.append(dict(dataset=dataset, model=model, runs=runs, seconds_per_run=median,
                             estimated_seconds=median*runs))
    total = sum(r['estimated_seconds'] for r in rows)
    value = dict(rows=rows, runs=sum(r['runs'] for r in rows), estimated_hours=total/3600,
        planning_range_hours=[total*.7/3600,total*1.8/3600],
        caveats='Same hardware only; excludes first-time synthetic generation, plotting and queueing. Shape scaling is heuristic; smoke warm-up can inflate small-model timing. Range is not a confidence interval.')
    write_json(Path(output)/'cost_estimate.json', value)
    print(json.dumps(value, indent=2), flush=True)


def main():
    if '--corpus' in sys.argv[1:]:
        from .corpus_workflow import main as corpus_main
        corpus_main([arg for arg in sys.argv[1:] if arg != '--corpus'])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke','screen','full','prepare','estimate'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PACKAGE/'configs/default.json')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS)
    parser.add_argument('--models', nargs='+', choices=['linear','dlinear','mlp','gru','transformer'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--fractions', nargs='+', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--profile-output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    names = args.datasets or (ALL_DATASETS if args.stage in ['full','prepare','estimate'] else ['exchange'])
    if args.stage == 'smoke':
        config.update(steps=50, eval_every=25, seeds=[42], fractions=[0,.5,1])
    for key in ['models','seeds','fractions','steps']:
        if getattr(args,key) is not None:
            config[key] = getattr(args,key)
    if config['steps'] < 1 or config['eval_every'] < 1:
        raise ValueError('steps and eval_every must be positive')
    from experiments.utils.sampler import quota
    for fraction in config['fractions']:
        quota(config['batch_size'],fraction)
    if args.stage == 'estimate':
        if not args.profile_output:
            parser.error('--profile-output is required for estimate')
        estimate(config,names,args.data_root,args.profile_output,args.output)
        return
    torch.set_num_threads(config['threads'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else torch.device(args.device)
    identity = {str(p.relative_to(PACKAGE.parent)):sha256(p)
                for p in sorted([*PACKAGE.glob('*.py'), *(PACKAGE.parent/'utils').glob('*.py')])
                if p.name not in ['summarize.py','verify.py','tests.py','corpus.py','corpus_workflow.py',
                                  'corpus_report.py','corpus_audit.py','launch.py']}
    metadata = dict(config=config, datasets=names, python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, device=str(device), device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
        code_hashes=identity, description='Independent model per real dataset; source fractions are per-batch exposures; no test tuning')
    args.output.mkdir(parents=True,exist_ok=True)
    metadata_path = args.output/'experiment.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        # Allow disjoint dataset/model/seed shards into one output, but run-level signatures still protect results.
        if previous['code_hashes'] != identity:
            raise ValueError('Training code changed since this output was created; choose a new output directory')
    write_json(metadata_path,metadata)
    total = len(names)*len(config['models'])*len(config['seeds'])*len(config['fractions'])
    done, started = 0, time.perf_counter()
    print(f'Stage={args.stage}, planned_runs={total}, device={metadata["device_name"]}',flush=True)
    for dataset in names:
        spec = config['datasets'][dataset]
        splits, manifest = load_real(args.data_root,spec,config['normalization_epsilon'])
        write_json(args.output/dataset/'manifest.json',manifest)
        cache_name = f'{spec["frequency"]}_{spec["context"]}_{spec["horizon"]}.npz'
        synthetic,syn_metadata = load_synthetic(args.output/'assets'/cache_name,spec,config)
        write_json(args.output/dataset/'synthetic_manifest.json',syn_metadata)
        print(f'{dataset}: windows={manifest["window_counts"]}, synthetic={len(synthetic["x"])}, rejected={syn_metadata["rejected_nonfinite_series"]}',flush=True)
        if args.stage=='prepare':
            continue
        for model,seed,fraction in itertools.product(config['models'],config['seeds'],config['fractions']):
            destination = args.output/dataset/model/f'p_{fraction:.3f}'/f'seed_{seed}'
            # Ignore grid membership in per-run config identity so CLI sharding remains resumable.
            run_config = {k:v for k,v in config.items() if k not in ['models','seeds','fractions','datasets']}
            result,skipped = run_one(destination,model,fraction,seed,spec,run_config,splits,
                synthetic,manifest,device,dict(training=identity,synthetic=syn_metadata['signature']))
            done += 1
            elapsed = time.perf_counter()-started
            print(f'[{done}/{total}] {dataset}/{model}/seed={seed}/p={fraction:.3f} '
                  f'{"SKIP" if skipped else "DONE"} train={result["train_wall_seconds"]:.2f}s '
                  f'best_val_rmse={result["checkpoints"]["best"]["validation"]["rmse"]:.6g} '
                  f'elapsed={elapsed/60:.1f}min',flush=True)
        write_json(args.output/dataset/'baselines.json',evaluate_baselines(splits,spec,manifest['mase_denominator']))
    print(f'Finished in {(time.perf_counter()-started)/60:.2f} minutes',flush=True)


if __name__=='__main__':
    main()
