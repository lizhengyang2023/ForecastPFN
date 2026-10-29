"""Verify corpus runs, quotas, pool size and saved test predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.utils.evaluate import metrics
from .corpus_workflow import jobs


def verify(root, expected=None):
    root = Path(root)
    experiment = json.loads((root/'corpus_experiment.json').read_text(encoding='utf-8'))
    planned = {(d,s,m,f'p_{p:.3f}',f'seed_{seed}') for d,s,m,seed,p in
               jobs(experiment['config'], experiment['datasets'], experiment['strategies'])}
    paths = sorted(root.glob('*/*/*/p_*/seed_*/metrics.json'))
    found = {tuple(part.name for part in path.parents[4::-1]) for path in paths}
    if found != planned:
        raise ValueError(f'Incomplete grid: missing={len(planned-found)}, extra={len(found-planned)}')
    if expected is not None and len(paths) != expected:
        raise ValueError(f'Expected {expected}, found {len(paths)}')
    initial = {}
    for path in paths:
        d,s,m,p,seed = tuple(part.name for part in path.parents[4::-1])
        result = json.loads(path.read_text(encoding='utf-8'))
        request = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        pool = json.loads((root/d/f'{s}_corpus_manifest.json').read_text(encoding='utf-8'))
        assert pool['pool_windows'] == experiment['config']['synthetic_series'] * experiment['config']['windows_per_series']
        assert request['identity']['synthetic'] == pool['signature']
        assert result['final_step'] == experiment['config']['steps']
        key = d,m,int(seed.split('_')[1])
        if key in initial:
            assert initial[key] == result['initial_weights_sha256']
        initial[key] = result['initial_weights_sha256']
        with np.load(path.parent/'exposures.npz') as exposure:
            assert exposure['real'].sum() == result['exposures']['real']
            assert exposure['synthetic'].sum() == result['exposures']['synthetic']
            assert len(exposure['synthetic']) == pool['pool_windows']
        manifest = json.loads((path.parent/'split_manifest.json').read_text(encoding='utf-8'))
        with np.load(path.parent/'predictions.npz') as predictions:
            for checkpoint in ('best','final'):
                recalculated = metrics(predictions[checkpoint], predictions['truth'], manifest['mase_denominator'])
                for field in ('mae','rmse','mse'):
                    np.testing.assert_allclose(recalculated[field], result['checkpoints'][checkpoint]['test'][field])
    print(f'PASS: {len(paths)} corpus runs; complete grid, pool identities, exposures, paired initialization, predictions.')


