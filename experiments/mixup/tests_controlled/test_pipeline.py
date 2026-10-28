"""Focused tests for common support, task sharding, and saved results."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.utils.datasets import write_json
from experiments.mixup.controlled import TASKS, build_pools, save_pools
from experiments.mixup.engine import run_one
from experiments.mixup.period_features import extract
from experiments.mixup.run import jobs
from experiments.mixup.run import DEFAULT_CONTROLLED_DATASETS
from experiments.mixup.sampling import BatchSampler
from experiments.mixup.summarize import summarize
from experiments.mixup.verify import verify


class ControlledPipeline(unittest.TestCase):
    def setUp(self):
        time=np.arange(96)
        phase=np.arange(12)*2*np.pi/12
        self.real=np.stack([np.cos(2*np.pi*time/24+p) for p in phase]).astype('float32')
        self.synthetic=np.stack([np.cos(2*np.pi*time/24+p+.02) for p in phase]).astype('float32')
        self.origins=np.arange(12)*200
        self.spec=dict(context=96,horizon=4,kernel=3,season=24)
        self.matching=dict(min_cycles=3,min_period=4,power_fraction=.15,
            min_autocorrelation=.15,period_tolerance=.1,
            phase_tolerance=np.pi/4,drift_tolerance=np.pi/4,top_k=8)
        self.eligible,self.pools,self.summary=build_pools(
            self.real,self.synthetic,self.origins,self.spec,self.matching)

    def test_common_support_and_phase_separation(self):
        self.assertEqual(len(self.eligible),12)
        self.assertEqual(set(self.pools),set(TASKS)-{'real'})
        for task,matrix in self.pools.items():
            self.assertTrue((matrix[self.eligible,0]>=0).all())
            for anchor in self.eligible:
                candidates=matrix[anchor][matrix[anchor]>=0]
                self.assertTrue(len(candidates))
                if task.startswith('real_real'):
                    self.assertTrue((np.abs(self.origins[candidates]-self.origins[anchor])>=100).all())
                real_features=extract(self.real,24)
                partner_features=(real_features if task.startswith('real_real')
                                  else extract(self.synthetic,24))
                anchor_period=real_features['period'][anchor]
                partner_period=partner_features['period'][candidates]
                self.assertTrue((np.abs(partner_period/anchor_period-1)
                                 <=self.matching['period_tolerance']).all())
                drift=2*np.pi*self.spec['horizon']*np.abs(1/anchor_period-1/partner_period)
                self.assertTrue((drift<=self.matching['drift_tolerance']).all())
                phase=np.arange(12)*2*np.pi/12
                distance=np.abs(np.angle(np.exp(1j*(phase[anchor]-phase[candidates]))))
                if task.endswith('_phase_match'):
                    self.assertTrue((distance<=np.pi/4+.03).all())
                else:
                    self.assertTrue((distance>np.pi/4-.03).all())

    def test_sampler_streams_and_quota(self):
        samplers=[BatchSampler(self.eligible,self.pools,12,12,8,.5,42,task)
                  for task in TASKS]
        for _ in range(3):
            draws=[sampler.take() for sampler in samplers]
            for anchors,_,_ in draws[1:]:
                np.testing.assert_array_equal(anchors,draws[0][0])
            for _,mask,_ in draws[2:]:
                np.testing.assert_array_equal(mask,draws[1][1])
            for sampler,(anchors,mask,partners) in zip(samplers[1:],draws[1:]):
                self.assertEqual(mask.sum(),4)
                self.assertEqual(len(partners),4)
                self.assertTrue((self.pools[sampler.task][anchors[mask],0]>=0).all())

    def test_shards_partition_full_grid(self):
        config=dict(models=['linear','gru'],seeds=[42,43],weights=[.5,.75])
        grid=jobs(config,['etth1','exchange'],TASKS)
        shards=[[job for i,job in enumerate(grid) if i%4==shard] for shard in range(4)]
        self.assertEqual(sum(map(len,shards)),len(grid))
        self.assertEqual(set().union(*(set(part) for part in shards)),set(grid))
        self.assertEqual(len(grid),2*2*2*(1+4*2))
        self.assertEqual(len(DEFAULT_CONTROLLED_DATASETS)*5*3*(1+4*3),975)

    def test_no_fallback_when_common_support_is_empty(self):
        flat=np.zeros_like(self.real)
        eligible,_,summary=build_pools(flat,self.synthetic,self.origins,self.spec,self.matching)
        self.assertEqual(len(eligible),0)
        self.assertEqual(summary['eligible_fraction'],0)
        with self.assertRaisesRegex(ValueError,'Empty common-support'):
            BatchSampler(eligible,self.pools,12,12,8,.5,42,'real_real_phase_match')

    def test_training_verification_and_report(self):
        rng=np.random.default_rng(7)
        def split(x):
            y=rng.normal(size=(len(x),4)).astype('float32')
            return dict(x=x,y=y,raw_y=y,mean=np.zeros((len(x),1)),
                        scale=np.ones((len(x),1)),origins=self.origins[:len(x)])
        splits=dict(train=split(self.real),validation=split(self.real[:4]),test=split(self.real[:5]))
        synthetic=split(self.synthetic)
        manifest=dict(sha256='fixture',mase_denominator=1.,
            window_counts=dict(test=5),validation_end_exclusive=0,processed_rows=2000)
        config=dict(models=['linear'],seeds=[42],weights=[.75],steps=2,batch_size=8,
            augmentation_probability=.5,learning_rate=.001,clip_grad=1.,
            eval_every=1,eval_batch_size=16)
        root=Path(__file__).resolve().parents[2]/'results'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            output=Path(directory)
            write_json(output/'experiment.json',dict(config=config,datasets=['etth1'],tasks=TASKS))
            write_json(output/'etth1/support.json',self.summary)
            (output/'assets').mkdir()
            save_pools(output/'assets/etth1_controlled.npz',self.eligible,self.pools)
            for dataset,model,seed,weight,task in jobs(config,['etth1'],TASKS):
                path=output/dataset/model/task/f'lambda_{weight:.3f}'/f'seed_{seed}'
                result,_=run_one(path,model,task,weight,seed,self.spec,config,splits,
                    synthetic,manifest,torch.device('cpu'),{'test':True},self.eligible,self.pools)
                self.assertEqual(result['exposures']['anchors'],16)
            verify(output,expected=5)
            summarize(output)
            self.assertTrue((output/'report/report.md').exists())


if __name__=='__main__':
    unittest.main()
