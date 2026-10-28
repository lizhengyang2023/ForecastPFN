"""Prepare once and run deterministic Mixup job shards on separate devices."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .run import configuration, jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['smoke','screen','full'],required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/default.json')
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--models',nargs='+')
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--weights',nargs='+',type=float)
    parser.add_argument('--tasks',nargs='+')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--synthetic-series',type=int)
    parser.add_argument('--windows-per-series',type=int)
    parser.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1','cuda:2','cuda:3'])
    args=parser.parse_args(argv)
    if not args.devices or len({d for d in args.devices if d!='cpu'}) != sum(d!='cpu' for d in args.devices):
        parser.error('Provide distinct GPU devices for each worker')
    config,names,tasks=configuration(args)
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    common=['--data-root',str(args.data_root),'--output',str(root),'--config',str(args.config)]
    for name in ('datasets','models','seeds','weights','tasks','steps','synthetic_series','windows_per_series'):
        value=getattr(args,name)
        if value is not None:
            common += ['--'+name.replace('_','-')]
            common += [str(v) for v in value] if isinstance(value,list) else [str(value)]
    prepare=[sys.executable,'-m','experiments.mixup.run','--stage','prepare',*common]
    if args.stage=='smoke':
        prepare.append('--smoke-config')
    subprocess.run(prepare,check=True)
    logs=root/'logs'
    logs.mkdir(exist_ok=True)
    running=[]
    for shard,device in enumerate(args.devices):
        command=[sys.executable,'-m','experiments.mixup.run','--stage',args.stage,
                 '--shard-id',str(shard),'--num-shards',str(len(args.devices)),
                 '--device',device,*common]
        path=logs/f'worker_{shard}.log'
        stream=path.open('a',encoding='utf-8')
        stream.write(f'\n=== {datetime.now(timezone.utc).isoformat()} shard={shard} device={device} ===\n')
        stream.flush()
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        running.append((shard,process,stream,path))
        print(f'shard {shard} -> {device}; log={path}',flush=True)
    failed=[]
    for shard,process,stream,path in running:
        status=process.wait()
        stream.close()
        print(f'shard {shard}: exit={status}',flush=True)
        if status:
            failed.append((shard,str(path)))
    if failed:
        raise RuntimeError(f'Failed shards: {failed}')
    from .verify import verify
    from .summarize import summarize
    verify(root,expected=len(jobs(config,names,tasks)))
    summarize(root)


if __name__=='__main__':
    main()
