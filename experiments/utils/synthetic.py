"""Call the actual ForecastPFN NumPy/SciPy generator, without a TF runtime.

All fractions use the same finite pool. Fractions refer to exposures, not pool
cardinality; this also defines the p=1 endpoint without an infinite pool.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from .datasets import normalize, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / 'src/synthetic_generation'


def load_synthetic(cache, spec, config):
    sources = {name:sha256(GENERATOR/name) for name in [
        'generate_series.py', 'generate_series_components.py', 'utils.py',
        'series_config.py', 'config_variables.py', 'prior_constants.py']}
    signature = dict(context=spec['context'], horizon=spec['horizon'], frequency=spec['frequency'],
        series=config['synthetic_series'], windows=config['windows_per_series'],
        seed=config['data_seed'], epsilon=config['normalization_epsilon'], sources=sources,
        target='observed', transition=False, version=1)
    cache = Path(cache)
    meta_path = cache.with_suffix('.json')
    if cache.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata['signature'] == signature:
            with np.load(cache) as saved:
                return {key:saved[key] for key in saved.files}, metadata
        raise ValueError(f'Incompatible synthetic cache: {cache}; choose a new output directory')
    # This package uses fully qualified imports elsewhere; legacy bare module
    # imports are confined to the original generator directory.
    sys.path.insert(0, str(GENERATOR))
    try:
        from generate_series import generate
        from config_variables import Config
        Config.set_freq_variables(True)
        Config.set_transition(False)
        frequency_index = {'H':1, 'D':2, 'W':3}[spec['frequency']]
        rng_state = np.random.get_state()
        np.random.seed(config['data_seed'])
        started = time.perf_counter()
        l, h, count = spec['context'], spec['horizon'], config['windows_per_series']
        length = l + count*h
        raw_x, raw_y, series_ids = [], [], []
        examples, clean_examples = [], []
        rejected = 0
        try:
            for sid in range(config['synthetic_series']):
                for attempt in range(100):
                    _, frame = generate(n=length, freq_index=frequency_index)
                    clean = frame.series_values.to_numpy(dtype='float32')
                    observed = clean*frame.noise.to_numpy(dtype='float32')
                    if np.isfinite(observed).all():
                        break
                    rejected += 1
                else:
                    raise ValueError('Generator repeatedly returned nonfinite series')
                if sid < 6:
                    examples.append(observed)
                    clean_examples.append(clean)
                for offset in range(count):
                    origin = l+offset*h
                    raw_x.append(observed[origin-l:origin])
                    raw_y.append(observed[origin:origin+h])
                    series_ids.append(sid)
        finally:
            np.random.set_state(rng_state)
    finally:
        sys.path.remove(str(GENERATOR))
    x, y, _, _ = normalize(np.array(raw_x), np.array(raw_y), config['normalization_epsilon'])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite normalized synthetic windows')
    data = dict(x=x, y=y, series_ids=np.array(series_ids), examples=np.array(examples),
                clean_examples=np.array(clean_examples))
    metadata = dict(signature=signature, generation_seconds=time.perf_counter()-started,
        rejected_nonfinite_series=rejected, pool_windows=len(x), independent_series=config['synthetic_series'],
        note='Original generator, observed history/target; four nonoverlapping target windows per series by default')
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    write_json(meta_path, metadata)
    return data, metadata
