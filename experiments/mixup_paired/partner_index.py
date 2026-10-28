"""Candidate lookup using history-only features and explicit fallbacks."""
import numpy as np
from scipy.spatial import cKDTree

from .period_features import circular_distance


class PartnerIndex:
    def __init__(self, anchor_features, partner_features, origins=None,
                 same_source=False, period_tolerance=.1, phase_tolerance=np.pi/4,
                 drift_tolerance=np.pi/4, top_k=8, horizon=24, overlap_gap=0):
        if not 0 < period_tolerance < 1 or not 0 < phase_tolerance <= np.pi:
            raise ValueError('Invalid period or phase tolerance')
        if drift_tolerance <= 0 or top_k < 1 or horizon < 1:
            raise ValueError('Invalid matching parameters')
        self.anchor = anchor_features
        self.partner = partner_features
        self.origins = origins
        self.same_source = same_source
        self.period_tolerance = period_tolerance
        self.phase_tolerance = phase_tolerance
        self.drift_tolerance = drift_tolerance
        self.top_k = top_k
        self.horizon = horizon
        self.overlap_gap = overlap_gap
        self.valid = np.flatnonzero(partner_features['reliable'])
        self.sorted_ids = self.valid[np.argsort(partner_features['period'][self.valid])]
        self.sorted_periods = partner_features['period'][self.sorted_ids]
        if len(self.valid):
            p = partner_features['period'][self.valid]
            phase = partner_features['phase'][self.valid]
            coords = np.column_stack((np.log(p) / period_tolerance,
                np.cos(phase) / phase_tolerance,
                np.sin(phase) / phase_tolerance))
            self.tree = cKDTree(coords)
        else:
            self.tree = None
        self.pool_size = len(partner_features['period'])
        if same_source and (origins is None or len(origins) != self.pool_size):
            raise ValueError('Real-real matching needs training origins')

    def _allowed(self, anchor_id, ids):
        if not self.same_source:
            return ids
        return ids[np.abs(self.origins[ids] - self.origins[anchor_id]) >= self.overlap_gap]

    def _period_candidates(self, anchor_id):
        period = float(self.anchor['period'][anchor_id])
        lower = np.searchsorted(self.sorted_periods, period * (1 - self.period_tolerance))
        upper = np.searchsorted(self.sorted_periods, period * (1 + self.period_tolerance), side='right')
        ids = self._allowed(anchor_id, self.sorted_ids[lower:upper])
        if not len(ids):
            return ids
        drift = 2 * np.pi * self.horizon * np.abs(
            1 / period - 1 / self.partner['period'][ids])
        return ids[drift <= self.drift_tolerance]

    def choose(self, anchor_id, strategy, rng):
        if strategy not in ('period_only', 'period_phase'):
            raise ValueError(strategy)
        period_ids = np.empty(0, dtype=np.int64)
        if self.anchor['reliable'][anchor_id]:
            period_ids = self._period_candidates(anchor_id)
        if strategy == 'period_phase' and len(period_ids) and self.tree is not None:
            p = self.anchor['period'][anchor_id]
            phase = self.anchor['phase'][anchor_id]
            point = [np.log(p) / self.period_tolerance,
                np.cos(phase) / self.phase_tolerance,
                np.sin(phase) / self.phase_tolerance]
            # Radius encloses all points passing both exact period and phase gates.
            period_radius = -np.log(1 - self.period_tolerance) / self.period_tolerance
            phase_radius = 2 * np.sin(self.phase_tolerance / 2) / self.phase_tolerance
            positions = self.tree.query_ball_point(point,
                r=np.hypot(period_radius, phase_radius) + 1e-8)
            nearest = self._allowed(anchor_id, self.valid[np.asarray(positions, dtype=int)])
            allowed = np.intersect1d(nearest, period_ids, assume_unique=False)
            distances = circular_distance(phase, self.partner['phase'][allowed])
            allowed = allowed[distances <= self.phase_tolerance]
            distances = distances[distances <= self.phase_tolerance]
            if len(allowed):
                order = np.lexsort((allowed, distances))[:self.top_k]
                selected = int(rng.choice(allowed[order]))
                return selected, 'period_phase', len(allowed)
        if len(period_ids):
            return int(rng.choice(period_ids)), 'period_only', len(period_ids)
        if not self.same_source:
            return int(rng.integers(self.pool_size)), 'random', 0
        for _ in range(1000):
            candidate = int(rng.integers(self.pool_size))
            if abs(self.origins[candidate] - self.origins[anchor_id]) >= self.overlap_gap:
                return candidate, 'random', 0
        raise ValueError('No eligible real partner after overlap exclusion')
