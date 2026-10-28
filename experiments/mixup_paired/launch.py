"""Prepare shared assets, run four independent GPU jobs, verify, and summarize."""
import argparse
import subprocess
import sys
from pathlib import Path

from experiments.utils.constants import ALL_DATASETS


CELLS = [(mode, strategy) for mode in ('real_real','real_synthetic')
         for strategy in ('period_only','period_phase')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke','screen','full'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--weights', nargs='+', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--synthetic-series', type=int)
    parser.add_argument('--devices', nargs=4, default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    parser.add_argument('--random-output', type=Path,
                        help='Completed old random Mixup output for matched comparisons')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    common = ['--data-root', str(args.data_root)]
    for name in ('config','datasets','models','seeds','weights','steps','synthetic_series'):
        value = getattr(args, name)
        if value is not None:
            flag = '--' + name.replace('_','-')
            common += [flag] + ([str(v) for v in value] if isinstance(value,list) else [str(value)])
    prepared = args.output/'prepared'
    prepare = [sys.executable,'-m','experiments.mixup_paired.run','--stage','prepare',
        '--output',str(prepared),*common]
    print('Preparing shared synthetic assets', flush=True)
    subprocess.run(prepare, check=True)
    logs = args.output/'logs'
    logs.mkdir(exist_ok=True)
    running = []
    for (mode, strategy), device in zip(CELLS, args.devices):
        name = f'{mode}_{strategy}'
        command = [sys.executable,'-m','experiments.mixup_paired.run',
            '--stage',args.stage,'--output',str(args.output/name),
            '--assets',str(prepared/'assets'),
            '--mode',mode,'--strategy',strategy,'--device',device,*common]
        log = (logs/f'{name}.log').open('w', encoding='utf-8')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        running.append((name, process, log))
        print(f'{name} -> {device}; log={logs/name}.log', flush=True)
    failed = []
    for name, process, log in running:
        code = process.wait()
        log.close()
        print(f'{name}: exit={code}', flush=True)
        if code:
            failed.append(name)
    if failed:
        raise RuntimeError(f'Failed jobs: {failed}; inspect {logs}')
    from .verify import verify
    from .summarize import summarize
    dataset_count = len(args.datasets) if args.datasets else (len(ALL_DATASETS)
        if args.stage == 'full' else 1)
    models = len(args.models) if args.models else (2 if args.stage == 'smoke' else 5)
    seeds = len(args.seeds) if args.seeds else (1 if args.stage == 'smoke' else 3)
    weights = len(args.weights) if args.weights else (1 if args.stage == 'smoke' else 3)
    verify(args.output, expected=4 * dataset_count * models * seeds * weights)
    summarize(args.output, args.random_output)


if __name__ == '__main__':
    main()
