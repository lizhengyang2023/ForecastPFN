"""Build independent synthetic pools matched to training-history statistics."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from experiments.utils.datasets import normalize, sha256, write_json
from experiments.utils.synthetic import GENERATOR, load_synthetic

STRATEGIES = ('original', 'filtered', 'parameter', 'parameter_filtered')
FEATURES = ('low_power', 'mid_power', 'high_power', 'season_power',
            'slope', 'difference', 'lag_correlation', 'tail')


def features(x, season):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite history windows')
    length = x.shape[1]
    t = np.arange(length) - (length - 1) / 2
    slope = (x @ t) / np.dot(t, t)
    residual = x - x.mean(1, keepdims=True) - slope[:, None] * t
    power = np.abs(np.fft.rfft(residual, axis=1)) ** 2
    power[:, 0] = 0
    freq = np.fft.rfftfreq(length)
    total = np.maximum(power.sum(1), 1e-10)
    bands = [power[:, (freq > a) & (freq <= b)].sum(1) / total
             for a, b in ((0, 1/16), (1/16, 1/4), (1/4, .5))]
    index = int(round(length / season)) if season else 0
    seasonal = power[:, max(1, index-1):min(power.shape[1], index+2)].sum(1) / total
    lag = min(int(season), length // 2) if season else 1
    left, right = residual[:, :-lag], residual[:, lag:]
    corr = (left * right).sum(1) / np.maximum(np.sqrt((left**2).sum(1) * (right**2).sum(1)), 1e-10)
    difference = np.sqrt(np.mean(np.diff(x, axis=1)**2, axis=1))
    tail = np.mean(np.abs(x) > 2, axis=1)
    return np.column_stack([*bands, seasonal, slope * length, difference, corr, tail])


def reference(real_x, season):
    value = features(real_x, season)
    median = np.median(value, axis=0)
    q25, q75 = np.quantile(value, [.25, .75], axis=0)
    scale = np.maximum(q75 - q25, .05)
    return value, median, scale


def distribution_distance(real, candidate, median, scale):
    return float(np.mean(np.abs((np.quantile(real, [.1,.25,.5,.75,.9], axis=0)
                                - np.quantile(candidate, [.1,.25,.5,.75,.9], axis=0)) / scale)))


def select(real, candidate, series_ids, size, season):
    if len(candidate) < size:
        raise ValueError('Candidate pool is smaller than final pool')
    real_f, median, scale = reference(real, season)
    cand_f = features(candidate, season)
    # Quantile bins preserve both weak- and strong-seasonality histories.
    cuts = np.quantile(real_f[:, 3], [1/3, 2/3])
    real_bins = np.searchsorted(cuts, real_f[:, 3], side='right')
    cand_bins = np.searchsorted(cuts, cand_f[:, 3], side='right')
    allocation = np.bincount(real_bins, minlength=3) / len(real_bins) * size
    quotas = np.floor(allocation).astype(int)
    for b in np.argsort(-(allocation - quotas))[:size - quotas.sum()]:
        quotas[b] += 1
    selected = []
    per_series = {}
    selected_set = set()
    for b in range(3):
        taken = 0
        indices = np.flatnonzero(cand_bins == b)
        targets = real_f[real_bins == b]
        if not len(targets):
            continue
        # Nearest real shape, with robustly scaled feature coordinates.
        distances = cdist(np.clip((cand_f[indices]-median)/scale, -8, 8),
                          np.clip((targets-median)/scale, -8, 8)).min(axis=1)
        for i in indices[np.argsort(distances, kind='stable')]:
            sid = int(series_ids[i])
            if per_series.get(sid, 0) < 4 and taken < quotas[b]:
                selected.append(int(i))
                selected_set.add(int(i))
                per_series[sid] = per_series.get(sid, 0) + 1
                taken += 1
    if len(selected) < size:
        distances = cdist(np.clip((cand_f-median)/scale, -8, 8),
                          np.clip((real_f-median)/scale, -8, 8)).min(axis=1)
        for i in np.argsort(distances, kind='stable'):
            sid = int(series_ids[i])
            if int(i) not in selected_set and per_series.get(sid, 0) < 4:
                selected.append(int(i))
                per_series[sid] = per_series.get(sid, 0) + 1
                if len(selected) == size:
                    break
    if len(selected) != size:
        raise ValueError(f'Insufficient stratum coverage: selected {len(selected)}/{size}; increase candidates')
    return np.array(selected), real_f, cand_f


def generate_pool(spec, config, series_count, multipliers, seed):
    sys.path.insert(0, str(GENERATOR))
    try:
        from config_variables import Config
        from generate_series import generate
        from generate_series_components import make_series
        from pandas.tseries.frequencies import to_offset
        Config.set_freq_variables(True)
        Config.set_transition(False)
        frequency_index = {'H':1, 'D':2, 'W':3}[spec['frequency']]
        old_state = np.random.get_state()
        np.random.seed(seed)
        l, h, count = spec['context'], spec['horizon'], config['windows_per_series']
        length = l + count*h
        raw_x, raw_y, ids = [], [], []
        rejected = 0
        try:
            for sid in range(series_count):
                for _ in range(100):
                    cfg, frame = generate(n=length, freq_index=frequency_index)
                    cfg.scale.linear *= multipliers['trend']
                    cfg.scale.exp = 1 + (cfg.scale.exp - 1) * multipliers['trend']
                    for field in ('a', 'm', 'w', 'h', 'minute'):
                        setattr(cfg.scale, field, getattr(cfg.scale, field) * multipliers['seasonal'])
                    cfg.noise_config.scale *= multipliers['noise']
                    generated = make_series(cfg, to_offset(Config.frequencies[frequency_index][0]), length, frame.index[0], {}, False)
                    observed = np.asarray(generated['values'] * generated['noise'], dtype='float32')
                    if np.isfinite(observed).all():
                        break
                    rejected += 1
                else:
                    raise ValueError('Generator repeatedly returned nonfinite series')
                for offset in range(count):
                    origin = l + offset*h
                    raw_x.append(observed[origin-l:origin])
                    raw_y.append(observed[origin:origin+h])
                    ids.append(sid)
        finally:
            np.random.set_state(old_state)
    finally:
        sys.path.remove(str(GENERATOR))
    x, y, _, _ = normalize(np.array(raw_x), np.array(raw_y), config['normalization_epsilon'])
    return dict(x=x, y=y, series_ids=np.array(ids)), rejected


def build(cache, spec, config, real_x, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    if strategy == 'original':
        pool, meta = load_synthetic(cache, spec, config)
        real_f, median, scale = reference(real_x, spec['season'])
        distance = distribution_distance(real_f, features(pool['x'], spec['season']), median, scale)
        meta = dict(meta, strategy='original', candidate_windows=len(pool['x']),
                    distance_before=distance, distance_after=distance,
                    multipliers=dict(trend=1., seasonal=1., noise=1.), feature_names=FEATURES)
        return pool, meta
    sources = {p.name:sha256(p) for p in GENERATOR.glob('*.py')}
    signature = dict(strategy=strategy, spec=spec, series=config['synthetic_series'],
                     windows=config['windows_per_series'], candidates=config['candidate_multiplier'],
                     seed=config['data_seed'], epsilon=config['normalization_epsilon'],
                     calibration_grid=config['calibration_grid'], sources=sources,
                     real_history_sha256=sha256_array(real_x))
    cache = Path(cache)
    meta_path = cache.with_suffix('.json')
    if cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['signature'] != signature:
            raise ValueError(f'Incompatible corpus cache: {cache}')
        with np.load(cache) as saved:
            return {key:saved[key] for key in saved.files}, meta
    started = time.perf_counter()
    size = config['synthetic_series'] * config['windows_per_series']
    candidate_series = config['synthetic_series'] * (config['candidate_multiplier'] if 'filtered' in strategy else 1)
    real_f, median, scale = reference(real_x, spec['season'])
    unit = dict(trend=1., seasonal=1., noise=1.)
    chosen = unit.copy()
    if strategy.startswith('parameter'):
        # One-factor-at-a-time coordinate search; no validation or test input.
        for field in ('trend', 'seasonal', 'noise'):
            trials = []
            for value in config['calibration_grid']:
                setting = {**chosen, field:float(value)}
                trial, _ = generate_pool(spec, config, min(config['calibration_series'], config['synthetic_series']), setting,
                                         config['data_seed'] + 100)
                score = distribution_distance(real_f, features(trial['x'], spec['season']), median, scale)
                trials.append((score, abs(value-1), value))
            chosen[field] = float(min(trials)[2])
    candidate, rejected = generate_pool(spec, config, candidate_series, chosen, config['data_seed'])
    candidate_f = features(candidate['x'], spec['season'])
    before = distribution_distance(real_f, candidate_f, median, scale)
    if 'filtered' in strategy:
        indices, _, _ = select(real_x, candidate['x'], candidate['series_ids'], size, spec['season'])
    else:
        indices = np.arange(size)
    result = {key:value[indices] for key,value in candidate.items()}
    after = distribution_distance(real_f, candidate_f[indices], median, scale)
    meta = dict(signature=signature, strategy=strategy, multipliers=chosen,
                candidate_windows=len(candidate['x']), pool_windows=len(indices),
                independent_series=int(len(np.unique(result['series_ids']))),
                rejected_nonfinite_series=rejected, distance_before=before, distance_after=after,
                feature_names=FEATURES, generation_seconds=time.perf_counter()-started)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    write_json(meta_path, meta)
    return result, meta


def sha256_array(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
