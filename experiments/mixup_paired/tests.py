"""Focused matching, leakage, stream, and training replay checks."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .engine import run_one
from .partner_index import PartnerIndex
from .period_features import extract, circular_distance
from .sampling import BatchSampler


class PairedTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(96)
        self.histories = np.stack([np.cos(2*np.pi*t/24 + phase)
            for phase in (0, .15, np.pi, .2, .3, .4)]).astype('float32')
        self.features = extract(self.histories, 24)

    def test_period_phase_and_unreliable(self):
        np.testing.assert_allclose(self.features['period'], 24)
        self.assertTrue(self.features['reliable'].all())
        self.assertGreater(circular_distance(self.features['phase'][0],
            self.features['phase'][2]), 3.0)
        flat = extract(np.zeros((2,96), dtype='float32'), 24)
        self.assertFalse(flat['reliable'].any())
        long_period = extract(self.histories[:1], 52)
        self.assertNotEqual(long_period['period'][0], 52)

    def test_matching_and_fallback(self):
        index = PartnerIndex(self.features, self.features,
            same_source=True, origins=np.arange(6)*200,
            overlap_gap=96, horizon=24)
        rng = np.random.default_rng(2)
        selected, level, _ = index.choose(0, 'period_phase', rng)
        self.assertEqual(level, 'period_phase')
        self.assertNotEqual(selected, 0)
        self.assertLess(circular_distance(self.features['phase'][0],
            self.features['phase'][selected]), np.pi/4)
        self.assertEqual(index.choose(0,'period_only',rng)[1], 'period_only')
        weak = {key: value.copy() for key, value in self.features.items()}
        weak['reliable'][0] = False
        fallback = PartnerIndex(weak, self.features)
        self.assertEqual(fallback.choose(0,'period_phase',rng)[1], 'random')

    def test_shared_anchor_stream_and_replay(self):
        index = PartnerIndex(self.features, self.features)
        streams = [BatchSampler(6,6,4,.5,42,index,'real_synthetic',strategy)
            for strategy in ('period_only','period_phase')]
        for _ in range(4):
            a, b = (stream.take() for stream in streams)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            self.assertEqual(a[1].sum(), 2)
        replay = BatchSampler(6,6,4,.5,42,index,'real_synthetic','period_phase')
        original = BatchSampler(6,6,4,.5,42,index,'real_synthetic','period_phase')
        for _ in range(4):
            for left, right in zip(replay.take(), original.take()):
                np.testing.assert_array_equal(left, right)

    def test_complete_training_replay(self):
        rng = np.random.default_rng(7)
        def split(count):
            x = rng.normal(size=(count, 96)).astype('float32')
            y = rng.normal(size=(count, 4)).astype('float32')
            return dict(x=x, y=y, raw_y=y.copy(), mean=np.zeros((count,1)),
                        scale=np.ones((count,1)), origins=np.arange(count)*200)
        splits = dict(train=split(12), validation=split(5), test=split(6))
        synthetic = split(8)
        spec = dict(context=96, horizon=4, kernel=3)
        manifest = dict(sha256='fixture', mase_denominator=1.,
            window_counts=dict(train=12,validation=5,test=6),
            validation_end_exclusive=0, processed_rows=2000)
        config = dict(steps=3,batch_size=4,augmentation_probability=.5,
            learning_rate=.001,clip_grad=1.,eval_every=1,eval_batch_size=10)
        index = PartnerIndex(extract(splits['train']['x'],24),
            extract(synthetic['x'],24), horizon=4)
        root = Path(__file__).resolve().parents[1]/'results'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            results = []
            for suffix in ('a','b'):
                result, _ = run_one(Path(directory)/suffix, 'linear','real_synthetic',
                    'period_phase',.75,42,spec,config,splits,synthetic,manifest,
                    torch.device('cpu'),{'fixture':True},index)
                results.append(result)
            self.assertEqual(results[0]['initial_digest'], results[1]['initial_digest'])
            self.assertEqual(results[0]['checkpoints'], results[1]['checkpoints'])
            self.assertEqual(results[0]['matching'], results[1]['matching'])


if __name__ == '__main__':
    unittest.main()
