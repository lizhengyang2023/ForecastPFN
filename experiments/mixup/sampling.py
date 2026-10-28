"""Shared anchor and augmentation streams for all controlled Mixup tasks."""
import numpy as np

from experiments.utils.sampler import CyclingSampler
from .controlled import TASKS


class BatchSampler:
    def __init__(self, eligible, pools, real_count, synthetic_count,
                 batch_size, probability, seed, task):
        count = batch_size*probability
        if task not in TASKS or not 0 <= probability <= 1 or abs(count-round(count)) > 1e-8:
            raise ValueError('Invalid task or augmentation quota')
        if not len(eligible):
            raise ValueError('Empty common-support anchor pool')
        self.eligible, self.pools, self.task = eligible, pools, task
        self.batch_size, self.augmented = batch_size, int(round(count))
        self.anchor = CyclingSampler(len(eligible), seed+1000)
        self.position_rng = np.random.default_rng(seed+4000)
        self.partner_rng = np.random.default_rng(seed+5000)
        self.real_partner_exposures = np.zeros(real_count, dtype=np.int64)
        self.synthetic_partner_exposures = np.zeros(synthetic_count, dtype=np.int64)

    def take(self):
        anchors = self.eligible[self.anchor.take(self.batch_size)]
        mask = np.zeros(self.batch_size, dtype=bool)
        if self.task != 'real':
            mask[self.position_rng.permutation(self.batch_size)[:self.augmented]] = True
        if self.task == 'real':
            return anchors, mask, np.empty(0, dtype=np.int64)
        candidates = self.pools[self.task][anchors[mask]]
        counts = (candidates >= 0).sum(axis=1)
        if np.any(counts == 0):
            raise AssertionError('Common-support pool contains an ineligible anchor')
        choices = np.array([self.partner_rng.integers(int(n)) for n in counts])
        partners = candidates[np.arange(len(choices)), choices].astype(np.int64)
        exposure = (self.real_partner_exposures if self.task.startswith('real_real')
                    else self.synthetic_partner_exposures)
        np.add.at(exposure, partners, 1)
        return anchors, mask, partners


def blend(real_x, real_y, partner_x, partner_y, mask, weight):
    """Mix history and future targets with the same scalar weight."""
    if not 0 <= weight <= 1:
        raise ValueError('Mixup weight must be in [0,1]')
    x, y = real_x.clone(), real_y.clone()
    x[mask] = weight*x[mask] + (1-weight)*partner_x
    y[mask] = weight*y[mask] + (1-weight)*partner_y
    return x, y
