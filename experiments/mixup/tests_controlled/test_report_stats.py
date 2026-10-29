"""Check paired percentage changes and validation-selected task ranking."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.mixup.plot_mase_delta import (
    all_lambda_table, delta_table, fixed_lambda_rank_summary, rank_summary,
    same_lambda_phase_summary,
)


class ReportStatisticsTest(unittest.TestCase):
    def test_seed_pairing_selection_and_same_lambda_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            rows = []
            def add(task, weight, values):
                for seed, mase in zip((42, 43), values):
                    rows.append(dict(dataset='etth1', model='linear', task=task,
                                     weight=weight, seed=seed, checkpoint='best',
                                     test_mase=mase))
            add('real', 1.0, (1.0, 2.0))
            add('real_synthetic_phase_match', 0.5, (0.9, 1.8))
            add('real_synthetic_phase_match', 0.9, (1.2, 1.6))
            add('real_synthetic_phase_mismatch', 0.5, (1.1, 2.2))
            add('real_synthetic_phase_mismatch', 0.9, (1.1, 2.2))
            pd.DataFrame(rows).to_csv(report / 'runs.csv', index=False)
            pd.DataFrame([
                dict(dataset='etth1', model='linear', task='real', weight=1.0,
                     checkpoint='best', test_mase_mean=1.5, test_mase_count=2),
                dict(dataset='etth1', model='linear', task='real_synthetic_phase_match',
                     weight=0.5, checkpoint='best', test_mase_mean=1.35,
                     test_mase_count=2),
                dict(dataset='etth1', model='linear', task='real_synthetic_phase_mismatch',
                     weight=0.5, checkpoint='best', test_mase_mean=1.65,
                     test_mase_count=2),
            ]).to_csv(report / 'validation_selected.csv', index=False)
            chosen = delta_table(report)
            match = chosen[chosen.task.eq('real_synthetic_phase_match')].iloc[0]
            self.assertAlmostEqual(match.delta_pct, -10.0)
            self.assertEqual(match.lambda_value, 0.5)
            ranking = rank_summary(report, chosen)
            winner = ranking[ranking.first_count.eq(1)].iloc[0]
            self.assertEqual(winner.task, 'real_synthetic_phase_match')
            fixed = fixed_lambda_rank_summary(report)
            self.assertEqual(set(fixed.lambda_value), {0.5, 0.9})
            half_win = fixed[fixed.lambda_value.eq(0.5) & fixed.first_count.eq(1)]
            self.assertEqual(half_win.iloc[0].task, 'real_synthetic_phase_match')
            self.assertEqual(fixed[fixed.task.eq('real')].combinations.tolist(), [1, 1])
            all_weights = all_lambda_table(report)
            self.assertEqual(len(all_weights), 4)
            phase = same_lambda_phase_summary(report, all_weights)
            half = phase[phase.lambda_value.eq(0.5)].iloc[0]
            self.assertEqual(half.match_better_count, 1)
            self.assertAlmostEqual(half.median_match_advantage_pct, 20.0)


if __name__ == '__main__':
    unittest.main()
