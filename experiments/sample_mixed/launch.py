"""Prepare corpus pools once, then shard complete training runs across devices."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .corpus_workflow import configuration, jobs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['smoke','screen','full'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/default.json')
    for name in ('datasets','strategies','models'):
        p.add_argument('--'+name, nargs='+')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--fractions', nargs='+', type=float)
    for name in ('steps','synthetic-series','windows-per-series','candidate-multiplier'):
        p.add_argument('--'+name, type=int)
    p.add_argument('--devices', nargs='+', default=['cuda:0'])
    args = p.parse_args(argv)
    if len(set(args.devices)) != len(args.devices):
        p.error('Each worker needs a distinct device')
    args.smoke_config = False
    config, names, strategies = configuration(args)
    common = ['--data-root', str(args.data_root), '--output', str(args.output), '--config', str(args.config)]
    for name in ('datasets','strategies','models','seeds','fractions','steps','synthetic_series',
                 'windows_per_series','candidate_multiplier'):
        value = getattr(args, name)
        if value is not None:
            common += ['--'+name.replace('_','-')]
            common += [str(item) for item in value] if isinstance(value, list) else [str(value)]
    smoke = ['--smoke-config'] if args.stage == 'smoke' else []
    subprocess.run([sys.executable, '-m', 'experiments.sample_mixed.run', '--corpus', '--stage', 'prepare',
                    *smoke, *common], check=True)
    logs = args.output/'logs'
    logs.mkdir(exist_ok=True)
    running = []
    for shard, device in enumerate(args.devices):
        path = logs/f'worker_{shard}.log'
        stream = path.open('a', encoding='utf-8')
        stream.write(f'\n=== {datetime.now(timezone.utc).isoformat()} shard={shard} device={device} ===\n')
        stream.flush()
        command = [sys.executable, '-m', 'experiments.sample_mixed.run', '--corpus', '--stage', args.stage,
                   '--shard-id', str(shard), '--num-shards', str(len(args.devices)), '--device', device,
                   *common]
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        running.append((process, stream, path))
        print(f'shard {shard} -> {device}; log={path}', flush=True)
    failed = []
    for process, stream, path in running:
        status = process.wait()
        stream.close()
        if status:
            failed.append(str(path))
    if failed:
        raise RuntimeError(f'Failed shards: {failed}')
    from .verify import verify
    from .summarize import summarize
    verify(args.output, expected=len(jobs(config, names, strategies)))
    summarize(args.output)


if __name__ == '__main__':
    main()
