"""Compact, bounded matching diagnostics for long training runs."""
import numpy as np


HISTOGRAM_EDGES = {
    'phase_distance': [0, np.pi/8, np.pi/4, np.pi/2, np.pi],
    'log_period_distance': [0, .02, .05, .1, np.inf],
    'future_phase_drift': [0, np.pi/8, np.pi/4, np.pi/2, np.inf],
}


def empty_histograms():
    return {key: [0] * (len(edges) - 1) for key, edges in HISTOGRAM_EDGES.items()}


def add_histogram(histograms, key, value):
    edges = HISTOGRAM_EDGES[key]
    bin_id = min(int(np.searchsorted(edges, value, side='right')) - 1,
                 len(edges) - 2)
    histograms[key][max(0, bin_id)] += 1


def feature_summary(real, synthetic):
    def one(features):
        reliable = features['reliable']
        return dict(total=len(reliable), reliable=int(reliable.sum()),
            reliable_fraction=float(reliable.mean()) if len(reliable) else 0,
            median_period=float(np.median(features['period'][reliable]))
                if reliable.any() else None)
    return dict(real=one(real), synthetic=one(synthetic))
