"""Audit saved predictions, exposure counts, and paired initialization."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utils.evaluate import metrics


def verify(root, expected=None):
    paths = sorted(Path(root).glob('**/lambda_*/seed_*/metrics.json'))
    if expected is not None and len(paths) != expected:
        raise AssertionError(f'Found {len(paths)} runs; expected {expected}')
    if not paths:
        raise AssertionError('No completed paired runs')
    initials = {}
    for path in paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        request = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        manifest = json.loads((path.parent/'split_manifest.json').read_text(encoding='utf-8'))
        dataset, model, mode, strategy = (path.parents[i].name for i in (5,4,3,2))
        assert (model, mode, strategy) == (result['model'], result['mode'], result['strategy'])
        assert path.parents[1].name == f'lambda_{result["weight"]:.3f}'
        assert path.parent.name == f'seed_{result["seed"]}'
        assert request['mode'] == mode and request['strategy'] == strategy
        with np.load(path.parent/'predictions.npz') as saved:
            assert len(saved['origins']) == manifest['window_counts']['test']
            assert np.all(saved['origins'] >= manifest['validation_end_exclusive'])
            assert np.all(saved['origins'] + request['spec']['horizon'] <= manifest['processed_rows'])
            for name in ('best','final'):
                actual = metrics(saved[f'{name}_test'], saved['truth'],
                    manifest['mase_denominator'])
                recorded = result['checkpoints'][name]['test']
                for key in ('mae','rmse','mse','mase'):
                    if recorded[key] is not None:
                        assert np.isclose(actual[key], recorded[key], rtol=1e-8, atol=1e-10)
        with np.load(path.parent/'exposures.npz') as saved:
            exposure = {key: int(saved[key].sum()) for key in saved.files}
            assert exposure == result['exposures']
            total = result['batch_size'] * result['final_step']
            assert exposure['anchors'] == total
            assert exposure['real_partners'] + exposure['synthetic_partners'] == int(
                total * result['augmentation_probability'])
            assert (exposure['real_partners'] > 0) == (mode == 'real_real')
        assert sum(result['matching']['match_counts'].values()) == int(
            total * result['augmentation_probability'])
        if strategy == 'period_only':
            assert result['matching']['match_counts']['period_phase'] == 0
        key = (dataset, model, result['seed'])
        if key in initials:
            assert initials[key] == result['initial_digest']
        else:
            initials[key] = result['initial_digest']
    print(f'PASS: {len(paths)} paired runs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-runs', type=int)
    args = parser.parse_args()
    verify(args.output, args.expected_runs)
