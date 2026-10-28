"""Check validation-only selection and paired baselines in a report fixture."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from .summarize import summarize


class SummaryTests(unittest.TestCase):
    def test_selection_uses_validation_not_test(self):
        scratch = Path(__file__).resolve().parents[1]/'results'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            for seed in (42, 43):
                for mode, weight, val, test in (
                    ('real', 1.0, 2.0, 2.0),
                    ('real_real', .75, 1.8, 1.9),
                    ('real_synthetic', .75, 1.5, 1.7),
                    ('real_synthetic', .5, 1.7, 1.1)):
                    path = root/'exchange'/'linear'/mode/f'lambda_{weight:.3f}'/f'seed_{seed}'
                    path.mkdir(parents=True)
                    score = lambda x: dict(mae=x,rmse=x,mse=x*x,mase=x,
                        horizon_mae=[x],horizon_rmse=[x],windows=1)
                    result = dict(model='linear',mode=mode,weight=weight,seed=seed,
                        final_step=50,batch_size=64,augmentation_probability=.5,
                        exposures=dict(anchors=3200,
                            real_partners=1600 if mode=='real_real' else 0,
                            synthetic_partners=1600 if mode=='real_synthetic' else 0),
                        unique_anchors=100,elapsed_seconds=1.0,diagnostics={},
                        checkpoints={name:dict(step=50,validation=score(val),test=score(test))
                                     for name in ('best','final')})
                    (path/'metrics.json').write_text(json.dumps(result),encoding='utf-8')
            report = summarize(root)
            selected = pd.read_csv(root/'report'/'validation_selected.csv')
            row = selected[(selected['mode']=='real_synthetic') &
                           (selected['checkpoint']=='best')].iloc[0]
            self.assertEqual(row['weight'],.75)
            self.assertAlmostEqual(row['test_rmse_mean'],1.7)
            self.assertAlmostEqual(
                row['rmse_improvement_vs_same_weight_real_real_percent'],
                100*(1-1.7/1.9))
            paired = pd.read_csv(root/'report'/'paired.csv')
            chosen = paired[(paired['mode']=='real_synthetic') &
                            (paired['weight']==.75) & (paired['checkpoint']=='best')]
            self.assertTrue((chosen['delta_rmse_vs_real'] == -.3).all() or
                            all(abs(chosen['delta_rmse_vs_real']+.3)<1e-12))
            self.assertTrue(report.exists())
            self.assertTrue((root/'report'/'exchange_linear_rmse.png').exists())


if __name__ == '__main__':
    unittest.main()
