"""Verify the unified controlled Mixup grid and saved predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utils.evaluate import metrics
from .controlled import load_pools
from .run import jobs


def verify(root, expected=None):
    root=Path(root)
    metadata=json.loads((root/'experiment.json').read_text(encoding='utf-8'))
    config=metadata['config']
    planned=jobs(config,metadata['datasets'],metadata['tasks'])
    if expected is not None and len(planned)!=expected:
        raise AssertionError(f'Expected {expected} planned jobs, found {len(planned)}')
    found=sorted(root.glob('*/*/*/lambda_*/seed_*/metrics.json'))
    if len(found)!=len(planned):
        raise AssertionError(f'Found {len(found)} completed runs; planned {len(planned)}')
    expected_keys={(d,m,s,w,t) for d,m,s,w,t in planned}
    seen=set()
    initial={}
    for path in found:
        result=json.loads(path.read_text(encoding='utf-8'))
        dataset,model,task=(path.parents[i].name for i in (4,3,2))
        key=(dataset,model,result['seed'],result['weight'],task)
        if key not in expected_keys or key in seen:
            raise AssertionError(f'Unexpected or duplicate run: {path}')
        seen.add(key)
        if result['task']!=task or path.parents[1].name!=f'lambda_{result["weight"]:.3f}':
            raise AssertionError(f'Path and metrics disagree: {path}')
        manifest=json.loads((path.parent/'split_manifest.json').read_text(encoding='utf-8'))
        with np.load(path.parent/'predictions.npz') as saved:
            if len(saved['origins'])!=manifest['window_counts']['test']:
                raise AssertionError(f'Test window count changed: {path}')
            for name in ('best','final'):
                actual=metrics(saved[f'{name}_test'],saved['truth'],manifest['mase_denominator'])
                for metric in ('mae','rmse','mse','mase'):
                    recorded=result['checkpoints'][name]['test'][metric]
                    if recorded is not None and not np.isclose(actual[metric],recorded,rtol=1e-8,atol=1e-10):
                        raise AssertionError(f'{path}: {name} {metric}')
        eligible,pools=load_pools(root/'assets'/f'{dataset}_controlled.npz')
        with np.load(path.parent/'exposures.npz') as exposure:
            counts={field:int(exposure[field].sum()) for field in exposure.files}
            if counts!=result['exposures'] or np.any(exposure['anchors'][np.setdiff1d(
                    np.arange(len(exposure['anchors'])),eligible)]):
                raise AssertionError(f'Exposure mismatch: {path}')
        total=result['batch_size']*result['final_step']
        quota=0 if task=='real' else int(total*result['augmentation_probability'])
        if counts['anchors']!=total or counts['real_partners']+counts['synthetic_partners']!=quota:
            raise AssertionError(f'Quota mismatch: {path}')
        if task.startswith('real_real') and counts['real_partners']!=quota:
            raise AssertionError(f'Real partner source mismatch: {path}')
        if task.startswith('real_synthetic') and counts['synthetic_partners']!=quota:
            raise AssertionError(f'Synthetic partner source mismatch: {path}')
        pair_key=(dataset,model,result['seed'])
        if pair_key in initial and initial[pair_key]!=result['initial_digest']:
            raise AssertionError(f'Initial weights differ: {path}')
        initial[pair_key]=result['initial_digest']
        if task!='real' and np.any(pools[task][eligible,0]<0):
            raise AssertionError(f'Ineligible matched pool: {dataset}/{task}')
    if seen!=expected_keys:
        raise AssertionError('Missing task combinations')
    print(f'PASS: {len(found)} complete controlled Mixup runs',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-runs',type=int)
    args=parser.parse_args()
    verify(args.output,args.expected_runs)


if __name__=='__main__':
    main()
