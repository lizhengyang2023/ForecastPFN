"""Estimate periodic features from normalized histories, never targets."""
import numpy as np


def extract(x, season, min_cycles=3, min_period=4, power_fraction=.15,
            min_autocorrelation=.15, batch_size=4096):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite [windows, context] histories')
    n, length = x.shape
    frequencies = np.fft.rfftfreq(length)
    candidates = np.flatnonzero((frequencies > 0)
        & (1 / np.maximum(frequencies, 1e-12) >= min_period)
        & (1 / np.maximum(frequencies, 1e-12) <= length / min_cycles))
    result = {key: np.zeros(n, dtype=dtype) for key, dtype in
        [('period', 'float32'), ('phase', 'float32'), ('amplitude', 'float32'),
         ('confidence', 'float32'), ('autocorrelation', 'float32'),
         ('slope', 'float32'), ('reliable', 'bool')]}
    if not len(candidates):
        return result
    t = np.arange(length, dtype=np.float64) - (length - 1)
    t_centered = t - t.mean()
    norm_t = np.dot(t_centered, t_centered)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        values = x[start:stop]
        slope = ((values - values.mean(axis=1, keepdims=True)) @ t_centered) / norm_t
        detrended = values - values.mean(axis=1, keepdims=True) - slope[:, None] * t_centered
        spectrum = np.fft.rfft(detrended, axis=1)
        power = np.abs(spectrum[:, candidates]) ** 2
        chosen = candidates[np.argmax(power, axis=1)]
        # Prefer a task's seasonal fundamental when its spectral power is competitive.
        seasonal = int(round(length / season)) if season else 0
        if seasonal in candidates:
            seasonal_power = np.abs(spectrum[:, seasonal]) ** 2
            use_season = seasonal_power >= .5 * power.max(axis=1)
            chosen[use_season] = seasonal
        rows = np.arange(stop - start)
        selected = spectrum[rows, chosen]
        period = length / chosen
        lag = np.rint(period).astype(int)
        corr = np.array([np.dot(v[:-p], v[p:]) /
            np.sqrt(np.dot(v[:-p], v[:-p]) * np.dot(v[p:], v[p:]) + 1e-12)
            for v, p in zip(detrended, lag)])
        confidence = np.abs(selected) ** 2 / (power.sum(axis=1) + 1e-12)
        # FFT phase is referred to the first sample; move it to the last history point.
        phase = np.angle(selected) + 2 * np.pi * chosen * (length - 1) / length
        phase = np.angle(np.exp(1j * phase))
        reliable = (confidence >= power_fraction) & (corr >= min_autocorrelation)
        reliable &= detrended.var(axis=1) > 1e-8
        result['period'][start:stop] = period
        result['phase'][start:stop] = phase
        result['amplitude'][start:stop] = 2 * np.abs(selected) / length
        result['confidence'][start:stop] = confidence
        result['autocorrelation'][start:stop] = corr
        result['slope'][start:stop] = slope
        result['reliable'][start:stop] = reliable
    return result


def circular_distance(left, right):
    return np.abs(np.angle(np.exp(1j * (left - right))))
