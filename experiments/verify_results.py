"""Independently recompute saved metrics and baseline forecasts with NumPy."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent / 'results'
for directory in sorted(root.glob('quick_validation*')):
    if not (directory/'summary.json').exists():
        continue
    s = json.loads((directory/'summary.json').read_text())
    data = np.load(directory/'samples.npz')
    saved = np.load(directory/'predictions.npz')
    ids = saved['series_ids']
    assert np.all(data['split'][ids]=='test')
    assert len(np.unique(ids))==s['ntest']
    h = s['horizon']
    hist = data['observed'][ids,-h-s['context']:-h]
    expected = hist[:,-7:][:,np.arange(h)%7]
    np.testing.assert_array_equal(expected,saved['SeasonalNaive-7'])
    scales=[]
    for row in hist:
        nonzero = row[row!=0]
        clipped = np.clip(row,0,nonzero.mean()+2*nonzero.std())
        clipped = clipped[clipped!=0]
        scales.append(clipped.mean()+clipped.std()+1e-4)
    np.testing.assert_allclose(saved['scale'],np.repeat(np.array(scales)[:,None],h,axis=1),rtol=2e-6)
    for metric in s['metrics']:
        truth = data[metric['target']][ids,-h:]
        error = saved[metric['model']]-truth
        np.testing.assert_allclose(metric['mse'],np.mean(error**2),rtol=1e-6)
        np.testing.assert_allclose(metric['mae'],np.mean(abs(error)),rtol=1e-6)
        np.testing.assert_allclose(metric['nmse'],np.mean((error/saved['scale'])**2),rtol=1e-6)
        assert np.isfinite(error).all()
    print(directory.name+': PASS split IDs, seasonal baseline, independent scale, all MSE/MAE/NMSE, finite predictions')
