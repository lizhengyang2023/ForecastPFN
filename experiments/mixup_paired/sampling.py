"""Preserve the old anchor/mask streams while replacing partner selection."""
import numpy as np

from experiments.utils.sampler import CyclingSampler
from experiments.mixup.sampling import blend
from .diagnostics import add_histogram, empty_histograms


class BatchSampler:
    def __init__(self, real_count, synthetic_count, batch_size, probability, seed,
                 index, mode, strategy):
        count = batch_size * probability
        if not 0 <= probability <= 1 or abs(count - round(count)) > 1e-8:
            raise ValueError('batch_size * probability must be an integer')
        if mode not in ('real_real', 'real_synthetic'):
            raise ValueError(mode)
        self.mode, self.strategy, self.index = mode, strategy, index
        self.batch_size, self.augmented = batch_size, int(round(count))
        self.anchor = CyclingSampler(real_count, seed + 1000)
        self.position_rng = np.random.default_rng(seed + 4000)
        self.partner_rng = np.random.default_rng(seed + 5000)
        self.real_partner_exposures = np.zeros(real_count, dtype=np.int64)
        self.synthetic_partner_exposures = np.zeros(synthetic_count, dtype=np.int64)
        self.level_counts = {key: 0 for key in ('period_phase', 'period_only', 'random')}
        self.candidate_total = 0
        self.period_distance_total = 0.
        self.phase_distance_total = 0.
        self.future_drift_total = 0.
        self.matched_distance_count = 0
        self.histograms = empty_histograms()

    def take(self):
        anchors = self.anchor.take(self.batch_size)
        mask = np.zeros(self.batch_size, dtype=bool)
        mask[self.position_rng.permutation(self.batch_size)[:self.augmented]] = True
        partners = np.empty(self.augmented, dtype=np.int64)
        for position, anchor_id in enumerate(anchors[mask]):
            partner, level, count = self.index.choose(int(anchor_id), self.strategy,
                self.partner_rng)
            partners[position] = partner
            self.level_counts[level] += 1
            self.candidate_total += count
            if level != 'random':
                a, p = self.index.anchor, self.index.partner
                period_distance = float(abs(np.log(
                    a['period'][anchor_id] / p['period'][partner])))
                phase_distance = float(abs(np.angle(np.exp(1j *
                    (a['phase'][anchor_id] - p['phase'][partner])))))
                future_drift = float(2 * np.pi * self.index.horizon * abs(
                    1 / a['period'][anchor_id] - 1 / p['period'][partner]))
                self.period_distance_total += period_distance
                self.phase_distance_total += phase_distance
                self.future_drift_total += future_drift
                for key, value in [('log_period_distance', period_distance),
                    ('phase_distance', phase_distance), ('future_phase_drift', future_drift)]:
                    add_histogram(self.histograms, key, value)
                self.matched_distance_count += 1
        exposures = (self.real_partner_exposures if self.mode == 'real_real'
                     else self.synthetic_partner_exposures)
        np.add.at(exposures, partners, 1)
        return anchors, mask, partners

    def diagnostics(self):
        count = sum(self.level_counts.values())
        return dict(match_counts=self.level_counts,
            match_rates={key: value / count if count else 0 for key, value in self.level_counts.items()},
            mean_candidate_count=self.candidate_total / count if count else 0,
            mean_log_period_distance=self.period_distance_total / self.matched_distance_count
                if self.matched_distance_count else None,
            mean_phase_distance=self.phase_distance_total / self.matched_distance_count
                if self.matched_distance_count else None,
            mean_future_phase_drift=self.future_drift_total / self.matched_distance_count
                if self.matched_distance_count else None,
            histograms=self.histograms,
            unique_partners=int(np.count_nonzero(self.real_partner_exposures
                if self.mode == 'real_real' else self.synthetic_partner_exposures)))
