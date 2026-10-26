"""Exact batch quotas and independent, reproducible source streams."""
import numpy as np


class CyclingSampler:
    def __init__(self, size, seed):
        if size <= 0:
            raise ValueError('Empty source pool')
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(size)
        self.position = 0
        self.exposures = np.zeros(size, dtype='int64')

    def take(self, count):
        parts = []
        while count:
            remaining = min(count, len(self.order)-self.position)
            part = self.order[self.position:self.position+remaining]
            parts.append(part)
            self.exposures[part] += 1
            self.position += remaining
            count -= remaining
            if self.position == len(self.order):
                self.order = self.rng.permutation(len(self.order))
                self.position = 0
        return np.concatenate(parts) if parts else np.empty(0, dtype='int64')


def quota(batch_size, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    synthetic = batch_size*fraction
    if abs(synthetic-round(synthetic)) > 1e-8:
        raise ValueError('batch_size * fraction must be an integer for exact quotas')
    return batch_size-int(round(synthetic)), int(round(synthetic))
