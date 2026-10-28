"""History-only, common-support partner pools for a controlled phase contrast."""
import numpy as np

from .period_features import circular_distance, extract


TASKS = ('real', 'real_real_phase_match', 'real_real_phase_mismatch',
         'real_synthetic_phase_match', 'real_synthetic_phase_mismatch')


def build_pools(real_x, synthetic_x, origins, spec, matching):
    keys = ('min_cycles', 'min_period', 'power_fraction', 'min_autocorrelation')
    settings = {key: matching[key] for key in keys}
    real = extract(real_x, spec['season'], **settings)
    synthetic = extract(synthetic_x, spec['season'], **settings)
    tolerance = matching['period_tolerance']
    phase_limit = matching['phase_tolerance']
    drift_limit = matching['drift_tolerance']
    top_k = matching['top_k']
    pools = {}
    for source, partner in (('real_real', real), ('real_synthetic', synthetic)):
        valid = np.flatnonzero(partner['reliable'])
        order = np.argsort(partner['period'][valid], kind='stable')
        sorted_ids = valid[order]
        sorted_periods = partner['period'][sorted_ids]
        matched = np.full((len(real_x), top_k), -1, dtype=np.int32)
        mismatched = np.full_like(matched, -1)
        for anchor in np.flatnonzero(real['reliable']):
            period = float(real['period'][anchor])
            lo = np.searchsorted(sorted_periods, period * (1-tolerance))
            hi = np.searchsorted(sorted_periods, period * (1+tolerance), side='right')
            ids = sorted_ids[lo:hi]
            if source == 'real_real':
                ids = ids[np.abs(origins[ids]-origins[anchor]) >= spec['context']+spec['horizon']]
            if not len(ids):
                continue
            drift = 2*np.pi*spec['horizon']*np.abs(1/period-1/partner['period'][ids])
            ids = ids[drift <= drift_limit]
            if not len(ids):
                continue
            distance = circular_distance(real['phase'][anchor], partner['phase'][ids])
            for matrix, keep, rank in (
                (matched, distance <= phase_limit, distance),
                (mismatched, distance > phase_limit, np.pi-distance)):
                candidates, scores = ids[keep], rank[keep]
                if len(candidates):
                    selected = np.lexsort((candidates, scores))[:top_k]
                    matrix[anchor, :len(selected)] = candidates[selected]
        pools[f'{source}_phase_match'] = matched
        pools[f'{source}_phase_mismatch'] = mismatched
    eligible = np.flatnonzero(np.logical_and.reduce([
        (pool[:, 0] >= 0) for pool in pools.values()])).astype(np.int64)
    summary = dict(total_real_windows=len(real_x), eligible_anchors=len(eligible),
        eligible_fraction=float(len(eligible)/len(real_x)),
        reliable_real_fraction=float(real['reliable'].mean()),
        reliable_synthetic_fraction=float(synthetic['reliable'].mean()),
        candidate_anchors={key:int((value[:,0]>=0).sum()) for key,value in pools.items()},
        phase_rule=f'match <= {phase_limit:g} rad; mismatch > {phase_limit:g} rad',
        no_fallback=True, low_support=bool(len(eligible)/len(real_x)<.1))
    return eligible, pools, summary


def save_pools(path, eligible, pools):
    np.savez_compressed(path, eligible=eligible, **pools)


def load_pools(path):
    with np.load(path) as saved:
        return saved['eligible'], {key:saved[key] for key in saved.files if key!='eligible'}
