"""Deterministic anchor, partner and augmentation-position streams."""
import numpy as np

from experiments.utils.sampler import CyclingSampler


class BatchSampler:
    def __init__(self, real_count, synthetic_count, batch_size, probability, seed):
        count = batch_size * probability
        if not 0 <= probability <= 1 or abs(count-round(count)) > 1e-8:
            raise ValueError('batch_size * augmentation probability must be an integer')
        self.real_count = real_count
        self.batch_size = batch_size
        self.augmented = int(round(count))
        self.anchor = CyclingSampler(real_count, seed+1000)
        self.real_partner = CyclingSampler(real_count, seed+2000)
        self.synthetic_partner = CyclingSampler(synthetic_count, seed+3000)
        self.position_rng = np.random.default_rng(seed+4000)

    def take(self, mode):
        anchors = self.anchor.take(self.batch_size)
        mask = np.zeros(self.batch_size, dtype=bool)
        if mode != 'real':
            mask[self.position_rng.permutation(self.batch_size)[:self.augmented]] = True
        n = int(mask.sum())
        partners = np.empty(0, dtype='int64')
        if mode == 'real_real':
            if self.real_count < 2:
                raise ValueError('Real-real Mixup needs two distinct training windows')
            partners = self.real_partner.take(n)
            # Resolve collisions without altering the shared anchor stream.
            collisions = partners == anchors[mask]
            partners[collisions] = (partners[collisions]+1) % self.real_count
        elif mode == 'real_synthetic':
            partners = self.synthetic_partner.take(n)
        elif mode != 'real':
            raise ValueError(f'Unknown mode {mode}')
        return anchors, mask, partners


def blend(real_x, real_y, partner_x, partner_y, mask, weight):
    """Use the same weight for every point in both history and target."""
    if not 0 <= weight <= 1:
        raise ValueError('Mixup weight must be in [0, 1]')
    x, y = real_x.clone(), real_y.clone()
    x[mask] = weight*x[mask] + (1-weight)*partner_x
    y[mask] = weight*y[mask] + (1-weight)*partner_y
    return x, y
