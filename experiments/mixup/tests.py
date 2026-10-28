"""Determinism and target-consistency checks for the Mixup sampler."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from .sampling import BatchSampler, blend
from .engine import run_one


class MixupTests(unittest.TestCase):
    def test_shared_anchors_and_determinism(self):
        streams = [BatchSampler(17, 11, 8, .5, 42) for _ in range(3)]
        for _ in range(12):
            real = streams[0].take('real')
            rr = streams[1].take('real_real')
            rs = streams[2].take('real_synthetic')
            np.testing.assert_array_equal(real[0], rr[0])
            np.testing.assert_array_equal(real[0], rs[0])
            np.testing.assert_array_equal(rr[1], rs[1])
            self.assertEqual(rr[1].sum(), 4)
            self.assertFalse(np.any(rr[2] == rr[0][rr[1]]))
        replay = BatchSampler(17, 11, 8, .5, 42)
        first = BatchSampler(17, 11, 8, .5, 42)
        for _ in range(12):
            a = first.take('real_synthetic')
            b = replay.take('real_synthetic')
            for left,right in zip(a,b):
                np.testing.assert_array_equal(left,right)
        self.assertEqual(int(streams[2].anchor.exposures.sum()), 96)
        self.assertEqual(int(streams[2].synthetic_partner.exposures.sum()), 48)

    def test_same_weight_in_history_and_target(self):
        x = torch.tensor([[2.,4.],[6.,8.]])
        y = torch.tensor([[10.,12.],[14.,16.]])
        px = torch.tensor([[10.,20.]])
        py = torch.tensor([[30.,40.]])
        mx,my = blend(x,y,px,py,torch.tensor([True,False]),.75)
        torch.testing.assert_close(mx[0], torch.tensor([4.,8.]))
        torch.testing.assert_close(my[0], torch.tensor([15.,19.]))
        torch.testing.assert_close(mx[1],x[1])
        torch.testing.assert_close(my[1],y[1])

    def test_invalid_batch_fraction(self):
        with self.assertRaises(ValueError):
            BatchSampler(10,10,7,.5,42)

    def test_complete_training_replay(self):
        rng = np.random.default_rng(7)
        def split(count):
            x = rng.normal(size=(count,4)).astype('float32')
            y = rng.normal(size=(count,2)).astype('float32')
            return dict(x=x,y=y,raw_y=y.copy(),mean=np.zeros((count,1)),
                        scale=np.ones((count,1)),origins=np.arange(count))
        splits = dict(train=split(12), validation=split(5), test=split(6))
        synthetic = split(8)
        spec = dict(context=4,horizon=2,kernel=3)
        manifest = dict(sha256='fixture',mase_denominator=1.0,
                        window_counts=dict(train=12,validation=5,test=6),
                        validation_end_exclusive=0,processed_rows=10)
        config = dict(steps=3,batch_size=4,augmentation_probability=.5,
            learning_rate=.001,clip_grad=1.,eval_every=1,eval_batch_size=10)
        temporary_root = Path(__file__).resolve().parents[1]/'results'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            outputs = []
            for suffix in ['a','b']:
                result,_ = run_one(Path(directory)/suffix,'linear','real_synthetic',.75,42,
                    spec,config,splits,synthetic,manifest,torch.device('cpu'),{'fixture':True})
                outputs.append(result)
            self.assertEqual(outputs[0]['initial_digest'],outputs[1]['initial_digest'])
            self.assertEqual(outputs[0]['checkpoints'],outputs[1]['checkpoints'])
            for suffix in ['a','b']:
                with np.load(Path(directory)/suffix/'predictions.npz') as saved:
                    if suffix == 'a':
                        first = saved['final_test'].copy()
                    else:
                        np.testing.assert_array_equal(first,saved['final_test'])


if __name__ == '__main__':
    unittest.main()
