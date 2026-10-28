"""Fast invariants for split leakage, exact mixing and source-model wiring."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.utils.datasets import load_real, normalize
from experiments.utils.evaluate import metrics, baseline_predictions
from experiments.utils.models import build_model
from experiments.utils.sampler import CyclingSampler, quota
from experiments.utils.synthetic import load_synthetic


class Invariants(unittest.TestCase):
    def test_split_targets_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            values=np.arange(1000,dtype='float32')
            pd.DataFrame({'date':pd.date_range('2000-01-01',periods=1000),'y':values}).to_csv(Path(directory)/'data.csv',index=False)
            spec=dict(file='data.csv',target='y',frequency='D',context=36,horizon=10,season=7)
            splits,manifest=load_real(directory,spec,1e-4)
            for name,lo,hi in [('train',36,700),('validation',700,800),('test',800,1000)]:
                d=splits[name]
                self.assertEqual(d['origins'][0],lo)
                self.assertEqual(d['origins'][-1]+10,hi)
                np.testing.assert_allclose(d['y']*d['scale']+d['mean'],d['raw_y'],atol=1e-4)
            changed=splits['train']['raw_y']+10000
            x,_,mean,scale=normalize(splits['train']['raw_x'],changed,1e-4)
            np.testing.assert_array_equal(x,splits['train']['x'])
            self.assertEqual(manifest['mase_denominator'],7)

    def test_sampler_prefix_and_counts(self):
        a,b=CyclingSampler(11,42),CyclingSampler(11,42)
        np.testing.assert_array_equal(a.take(50),np.concatenate([b.take(7),b.take(43)]))
        self.assertEqual(a.exposures.sum(),50)
        self.assertLessEqual(a.exposures.max()-a.exposures.min(),1)
        for p in np.arange(9)/8:
            nr,ns=quota(64,float(p))
            self.assertEqual(nr+ns,64)
            # Source-weighted loss is exactly the concatenated-window mean.
            losses=np.arange(64,dtype='float64')
            weighted=(1-p)*losses[:nr].mean() if nr else 0
            weighted+=p*losses[nr:].mean() if ns else 0
            self.assertAlmostEqual(weighted,losses.mean())

    def test_models_have_finite_gradients(self):
        torch.set_num_threads(1)
        for name in ['linear','dlinear','mlp','gru','transformer']:
            model=build_model(name,dict(context=36,horizon=10,kernel=7))
            out=model(torch.randn(8,36))
            self.assertEqual(tuple(out.shape),(8,10))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_metrics_and_seasonal_baseline(self):
        split=dict(raw_x=np.arange(14)[None,:],y=np.zeros((1,10)))
        np.testing.assert_array_equal(baseline_predictions(split,7)['seasonal_naive'],[[7,8,9,10,11,12,13,7,8,9]])
        m=metrics(np.array([[1.,3.]]),np.array([[0.,1.]]),2.)
        self.assertEqual(m['mae'],1.5)
        self.assertAlmostEqual(m['rmse'],np.sqrt(2.5))
        self.assertEqual(m['mase'],.75)

    def test_causal_weather_fill_excludes_imputed_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            dates=pd.date_range('2000-01-01',periods=6000,freq='10min')
            frame=pd.DataFrame({'date':dates,'y':np.arange(6000,dtype=float)})
            # One missing interval and one identical duplicate, both in train.
            frame=frame.drop(index=np.arange(1000,1009))
            frame=pd.concat([frame,frame.iloc[[20]]]).sort_values('date')
            frame.to_csv(Path(directory)/'data.csv',index=False)
            spec=dict(file='data.csv',target='y',frequency='H',context=36,horizon=10,
                      season=24,duplicate_policy='drop_identical',regularize='10min',
                      resample='1h',samples_per_bin=6)
            splits,manifest=load_real(directory,spec,1e-4)
            self.assertEqual(manifest['dropped_identical_duplicates'],1)
            self.assertEqual(manifest['causally_filled_raw_points'],9)
            for split in splits.values():
                targets=split['origins'][:,None]+np.arange(10)
                self.assertFalse(np.isin(targets,[166,167,168]).any())

    def test_hourly_synthetic_generation_with_compatible_alias(self):
        temporary_root = Path(__file__).resolve().parents[1]/'results'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            spec=dict(frequency='H',context=4,horizon=2)
            config=dict(synthetic_series=1,windows_per_series=1,data_seed=42,
                        normalization_epsilon=1e-4)
            synthetic,_=load_synthetic(Path(directory)/'hourly.npz',spec,config)
            self.assertEqual(synthetic['x'].shape,(1,4))
            self.assertEqual(synthetic['y'].shape,(1,2))


if __name__=='__main__':
    unittest.main()
