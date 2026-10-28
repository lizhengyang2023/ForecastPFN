"""Audit saved runs independently of the training loop."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.utils.evaluate import metrics


def verify(root, expected=None, require_grid=False, grid_datasets=None):
    paths = sorted(Path(root).glob('*/*/p_*/seed_*/metrics.json'))
    if require_grid:
        plan=json.loads((Path(__file__).resolve().parent/'configs/default.json').read_text())
        names=grid_datasets or list(plan['datasets'])
        planned={
            (dataset,model,f'p_{fraction:.3f}',f'seed_{seed}')
            for dataset in names
            for model in plan['models']
            for seed in plan['seeds']
            for fraction in plan['fractions']
        }
        found={tuple(part.name for part in (path.parents[3],path.parents[2],path.parents[1],path.parent)) for path in paths}
        missing=sorted(planned-found)
        extra=sorted(found-planned)
        if missing or extra:
            raise ValueError(f'Incomplete grid: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}')
    if expected is not None:
        assert len(paths)==expected,(len(paths),expected)
    assert paths,'No completed runs'
    initial,origins={},{}
    for path in paths:
        r=json.loads(path.read_text())
        config=json.loads((path.parent/'config.json').read_text())
        assert config['model']==path.parents[2].name
        assert f'p_{config["fraction"]:.3f}'==path.parents[1].name
        assert f'seed_{config["seed"]}'==path.parent.name
        assert r['final_step']==config['config']['steps']
        manifest=json.loads((path.parent/'split_manifest.json').read_text())
        log=pd.read_csv(path.parent/'train_log.csv')
        dataset=path.parents[3].name
        key=(dataset,r['model'],r['seed'])
        if key in initial:
            assert initial[key]==r['initial_weights_sha256'],'Initialization differs across p'
        initial[key]=r['initial_weights_sha256']
        with np.load(path.parent/'exposures.npz') as exposure:
            for source in ['real','synthetic']:
                count=exposure[source]
                assert count.sum()==r['exposures'][source]
                assert count.max()-count.min()<=1
            assert sum(r['exposures'].values())==config['config']['batch_size']*r['final_step']
            assert r['exposures']['synthetic']==r['fraction']*sum(r['exposures'].values())
        assert r['best_step']==int(log.loc[log.validation_rmse.idxmin(),'step'])
        np.testing.assert_allclose(r['checkpoints']['best']['validation']['rmse'],log.validation_rmse.min())
        np.testing.assert_allclose(r['checkpoints']['final']['validation']['rmse'],log.validation_rmse.iloc[-1])
        with np.load(path.parent/'predictions.npz') as saved:
            assert len(saved['test_origins'])==manifest['window_counts']['test']
            assert saved['test_origins'][0]>=manifest['validation_end_exclusive']
            assert saved['test_origins'][-1]+config['spec']['horizon']==manifest['processed_rows']
            if dataset in origins:
                np.testing.assert_array_equal(origins[dataset],saved['test_origins'])
            origins[dataset]=saved['test_origins']
            for checkpoint in ['best','final']:
                independent=metrics(saved[checkpoint],saved['truth'],manifest['mase_denominator'])
                for metric in ['mae','rmse','mse']:
                    np.testing.assert_allclose(independent[metric],r['checkpoints'][checkpoint]['test'][metric],rtol=1e-10)
    print(f'PASS: {len(paths)} runs; paired initialization, fixed splits/origins, exact quotas, balanced exposures, validation-only checkpoint selection, independently recomputed saved prediction metrics.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-runs',type=int)
    parser.add_argument('--require-grid',action='store_true',help='Check every combination in configs/default.json (all seven datasets by default).')
    parser.add_argument('--datasets',nargs='+',help='Dataset subset for --require-grid, e.g. exchange for screening.')
    args=parser.parse_args()
    verify(args.output,args.expected_runs,args.require_grid,args.datasets)
