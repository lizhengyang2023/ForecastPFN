"""Check saved Mixup predictions, exposures, and paired initialization."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utils.evaluate import metrics


def verify(root, expected=None):
    root = Path(root)
    paths = sorted(root.glob('*/*/*/lambda_*/seed_*/metrics.json'))
    if expected is not None and len(paths) != expected:
        raise AssertionError(f'Found {len(paths)} runs, expected {expected}')
    if not paths:
        raise AssertionError('No completed runs')
    initials = {}
    for path in paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        manifest = json.loads((path.parent/'split_manifest.json').read_text(encoding='utf-8'))
        config = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        with np.load(path.parent/'predictions.npz') as saved:
            assert len(saved['origins']) == manifest['window_counts']['test']
            assert np.all(saved['origins'] >= manifest['validation_end_exclusive'])
            assert np.all(saved['origins']+config['spec']['horizon'] <= manifest['processed_rows'])
            for name in ['best','final']:
                actual = metrics(saved[f'{name}_test'],saved['truth'],manifest['mase_denominator'])
                reported = result['checkpoints'][name]['test']
                for key in ['mae','rmse','mse','mase']:
                    if reported[key] is not None:
                        assert np.isclose(actual[key],reported[key],rtol=1e-8,atol=1e-10)
        with np.load(path.parent/'exposures.npz') as exp:
            counts = {k:int(exp[k].sum()) for k in exp.files}
            assert counts == result['exposures']
            total = result['batch_size']*result['final_step']
            assert counts['anchors'] == total
            partners = counts['real_partners']+counts['synthetic_partners']
            expected_partners = 0 if result['mode']=='real' else int(total*result['augmentation_probability'])
            assert partners == expected_partners
            assert (counts['real_partners']>0) == (result['mode']=='real_real')
            assert (counts['synthetic_partners']>0) == (result['mode']=='real_synthetic')
        key = (path.parents[4].name,result['model'],result['seed'])
        if key in initials:
            assert initials[key] == result['initial_digest']
        else:
            initials[key] = result['initial_digest']
    print(f'PASS: {len(paths)} runs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-runs', type=int)
    args = parser.parse_args()
    verify(args.output,args.expected_runs)
