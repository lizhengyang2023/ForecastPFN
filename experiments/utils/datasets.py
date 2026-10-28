"""Chronological splits, context-only normalization and explicit window origins."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic replacement: a half-written JSON must not look like a completed run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def normalize(x, y, epsilon):
    mean = x.mean(axis=1, keepdims=True)
    scale = np.maximum(x.std(axis=1, keepdims=True), epsilon)
    return ((x-mean)/scale).astype('float32'), ((y-mean)/scale).astype('float32'), mean, scale


def windows(values, origins, context, horizon, epsilon):
    x = values[origins[:, None] + np.arange(-context, 0)]
    y = values[origins[:, None] + np.arange(horizon)]
    xn, yn, mean, scale = normalize(x, y, epsilon)
    return dict(x=xn, y=yn, mean=mean, scale=scale, raw_x=x, raw_y=y, origins=origins)


def load_real(root, spec, epsilon):
    path = Path(root) / spec['file']
    frame = pd.read_csv(path, usecols=['date', spec['target']])
    source_rows = len(frame)
    dropped_duplicates = 0
    if spec.get('duplicate_policy') == 'drop_identical':
        frame = frame.drop_duplicates(subset=['date',spec['target']]).reset_index(drop=True)
        dropped_duplicates = source_rows-len(frame)
    dates = pd.to_datetime(frame['date'], errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f'{path}: timestamps must be unique and increasing')
    values = pd.to_numeric(frame[spec['target']], errors='raise').to_numpy(dtype='float32')
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: nonfinite observations; specify a causal cleaning policy first')
    valid_target = np.ones(len(values), dtype=bool)
    filled_points = 0
    if spec.get('regularize'):
        series = pd.Series(values, index=pd.DatetimeIndex(dates))
        grid = pd.date_range(series.index[0],series.index[-1],freq=spec['regularize'])
        if not series.index.isin(grid).all():
            raise ValueError('Off-grid observations cannot be silently dropped')
        regular = series.reindex(grid)
        valid_target = regular.notna().to_numpy()
        filled_points = int((~valid_target).sum())
        # Past-only fill is allowed in histories; targets touching it are excluded.
        values = regular.ffill().to_numpy(dtype='float32')
        dates = pd.Series(grid)
    dropped_bins = 0
    if spec.get('resample'):
        series = pd.Series(values, index=pd.DatetimeIndex(dates))
        observed = pd.Series(valid_target.astype(int), index=pd.DatetimeIndex(dates))
        grouped = series.resample(spec['resample'], label='right', closed='left')
        means, counts = grouped.mean(), grouped.count()
        valid = counts == spec['samples_per_bin']
        # Only edge bins can be incomplete; silently dropping internal gaps is unsafe.
        good = np.flatnonzero(valid.to_numpy())
        if not len(good) or not valid.iloc[good[0]:good[-1]+1].all():
            raise ValueError('Missing/incomplete internal aggregation bins')
        dropped_bins = int((~valid).sum())
        valid_target = observed.resample(spec['resample'], label='right', closed='left').min()[valid].to_numpy().astype(bool)
        values = means[valid].to_numpy(dtype='float32')
        dates = pd.Series(means[valid].index)
    differences = pd.DatetimeIndex(dates).to_series().diff().dropna()
    expected = pd.Timedelta({'D':'1d', 'H':'1h', 'W':'7d'}[spec['frequency']])
    if not (differences == expected).all():
        raise ValueError(f'{path}: irregular frequency; do not silently relabel timestamps')
    n = len(values)
    train_end, val_end = int(n*.7), int(n*.8)
    l, h = spec['context'], spec['horizon']
    intervals = {'train':(l, train_end), 'validation':(train_end, val_end), 'test':(val_end, n)}
    splits = {}
    for name, (start, end) in intervals.items():
        origins = np.arange(max(l, start), end-h+1, dtype='int64')
        origins = origins[valid_target[origins[:,None]+np.arange(h)].all(axis=1)]
        if not len(origins):
            raise ValueError(f'{path}: empty {name} split for L={l}, H={h}')
        splits[name] = windows(values, origins, l, h, epsilon)
        assert np.all(origins+h <= end)
    season = spec['season']
    if season > l or season >= train_end:
        raise ValueError('Seasonal baseline requires sufficient context/training history')
    valid_pairs = valid_target[season:train_end] & valid_target[:train_end-season]
    denominator = float(np.abs(values[season:train_end]-values[:train_end-season])[valid_pairs].mean())
    manifest = dict(file=str(path.resolve()), sha256=sha256(path), target=spec['target'],
        source_rows=source_rows, processed_rows=n, specification=spec, dropped_edge_bins=dropped_bins,
        dropped_identical_duplicates=dropped_duplicates,
        causally_filled_raw_points=filled_points, excluded_target_timepoints=int((~valid_target).sum()),
        train_end_exclusive=train_end, validation_end_exclusive=val_end,
        train_last_date=str(dates.iloc[train_end-1]), validation_first_date=str(dates.iloc[train_end]),
        validation_last_date=str(dates.iloc[val_end-1]), test_first_date=str(dates.iloc[val_end]),
        last_date=str(dates.iloc[-1]), window_counts={k:len(v['x']) for k,v in splits.items()},
        mase_denominator=denominator if denominator > 1e-12 else None,
        evaluation='rolling origin, frozen weights, all horizons stay in their split')
    return splits, manifest
