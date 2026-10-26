"""Small, series-disjoint validation using the repository generator/model.

Run with conda base Python; optional workspace-local .experiment_deps is preferred.
No downloaded weights or real-world data are used.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.experiment_deps').exists():
    sys.path.insert(0, str(ROOT / '.experiment_deps'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import argparse
import json
import time
import platform
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='experiments/results/quick_validation')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT / 'src/synthetic_generation'))
    from config_variables import Config
    from generate_series import generate
    Config.set_freq_variables(False)
    ntrain, nval, ntest, length, context, horizon = 768, 128, 256, 64, 36, 10
    clean, observed, dates, configs = [], [], [], []
    for i in range(ntrain + nval + ntest):
        cfg, frame = generate(n=length, freq_index=0)
        y = frame.series_values.to_numpy(dtype='float32')
        obs = y * frame.noise.to_numpy(dtype='float32')
        clean.append(y)
        observed.append(obs)
        d = frame.index
        dates.append(np.stack([d.year, d.month, d.day, d.dayofweek + 1, d.dayofyear], -1))
        configs.append(str(cfg))
    clean, observed, dates = np.array(clean), np.array(observed), np.array(dates, dtype='int32')
    assert np.isfinite(clean).all() and np.isfinite(observed).all()
    fingerprints = [hashlib.sha256(x.tobytes()).hexdigest() for x in clean]
    assert len(set(fingerprints)) == len(fingerprints), 'Duplicate series'
    np.savez_compressed(out / 'samples.npz', clean=clean, observed=observed, dates=dates,
                        split=np.repeat(['train', 'validation', 'test'], [ntrain, nval, ntest]))
    # Both source directories use bare module names; clear only their shared imports.
    sys.path.remove(str(ROOT / 'src/synthetic_generation'))
    for module in ['constants', 'config_variables', 'utils']:
        sys.modules.pop(module, None)
    sys.path.insert(0, str(ROOT / 'src/training'))
    from models import TransformerModel, TransformerBlock
    model = TransformerModel(scaler='robust')
    # Original block implementations, reduced width/head count for a CPU smoke run.
    model.encoder1 = TransformerBlock(key_dim=18, heads=2)
    model.encoder2 = TransformerBlock(key_dim=18, heads=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def inputs(ids, origins, offsets):
        history_indices = origins[:, None] - context + np.arange(context)[None, :]
        target_indices = origins + offsets
        x = dict(history=observed[ids[:, None], history_indices],
                 ts=dates[ids[:, None], history_indices],
                 target_ts=dates[ids, target_indices][:, None, :],
                 task=np.ones(len(ids), dtype='int32'))
        return x, clean[ids, target_indices][:, None]

    def full_inputs(ids):
        return inputs(np.repeat(ids, horizon), np.full(len(ids)*horizon, length-horizon),
                      np.tile(np.arange(horizon), len(ids)))

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square((y-pred['result'])/pred['scale']))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def infer(x):
        return model(x, training=False)

    def predict(x):
        p, s = [], []
        for b in range(0, len(x['task']), 64):
            result = infer({k:v[b:b+64] for k,v in x.items()})
            p.append(result['result'].numpy())
            s.append(result['scale'].numpy())
        return np.concatenate(p), np.concatenate(s)

    vx, vy = full_inputs(np.arange(ntrain, ntrain+nval))
    # As in train.py: build eagerly before tracing ragged robust scaling.
    model({k:v[:2] for k,v in vx.items()})
    vp, vs = predict(vx)
    initial = float(np.mean(((vp-vy)/vs)**2))
    print(f'TensorFlow {tf.__version__}, parameters={model.count_params()}, initial val={initial:.6f}', flush=True)
    records = [dict(epoch=0, train_loss=None, val_loss=initial, elapsed_s=time.perf_counter()-started)]
    best, best_epoch = initial, 0
    model.save_weights(str(out / 'best.weights.h5'))
    rng = np.random.default_rng(args.seed+1)
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs+1):
        losses = []
        for _ in range(args.steps):
            ids = rng.integers(0, ntrain, 32)
            origins = rng.integers(context, length-horizon+1, 32)
            x, y = inputs(ids, origins, rng.integers(0, horizon, 32))
            losses.append(float(train_step(x,y)))
        vp, vs = predict(vx)
        val = float(np.mean(((vp-vy)/vs)**2))
        assert np.isfinite(val) and np.isfinite(losses).all()
        records.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val,
                            elapsed_s=time.perf_counter()-started))
        if val < best:
            best, best_epoch = val, epoch
            model.save_weights(str(out / 'best.weights.h5'))
        pd.DataFrame(records).to_csv(out/'loss.csv', index=False)
        print(f'epoch {epoch:02}: train={np.mean(losses):.6f} val={val:.6f}', flush=True)
    training_seconds = time.perf_counter()-training_start
    model.load_weights(str(out / 'best.weights.h5'))
    ids = np.arange(ntrain+nval, ntrain+nval+ntest)
    tx, ty = full_inputs(ids)
    t0 = time.perf_counter()
    pp, ss = predict(tx)
    inference_seconds = time.perf_counter()-t0
    pred, scale = pp.reshape(ntest,horizon), ss.reshape(ntest,horizon)
    hist = observed[ids, length-horizon-context:length-horizon]
    predictions = {'ForecastPFN-small': pred,
        'SeasonalNaive-7': hist[:, -7:][:, np.arange(horizon)%7],
        'LastValue': np.repeat(hist[:,-1:],horizon,axis=1),
        'ContextMean': np.repeat(hist.mean(1,keepdims=True),horizon,axis=1),
        'Drift': hist[:,-1:] + (hist[:,-1:]-hist[:,:1])/(context-1)*np.arange(1,horizon+1)}
    metrics, per_series = [], {}
    for target_name, truth in [('clean',clean[ids,-horizon:]),('observed',observed[ids,-horizon:])]:
        for name, prediction in predictions.items():
            error = prediction-truth
            normalized = np.mean((error/scale)**2,axis=1)
            per_series[target_name+'/'+name] = normalized
            metrics.append(dict(target=target_name,model=name,mse=float(np.mean(error**2)),
                mae=float(np.mean(abs(error))),nmse=float(normalized.mean()),
                smape=float(200*np.mean(abs(error)/(abs(prediction)+abs(truth)+1e-8)))))
    boot = np.random.default_rng(123)
    comparison = {}
    for target in ['clean','observed']:
        diff = per_series[target+'/ForecastPFN-small']-per_series[target+'/SeasonalNaive-7']
        draws = diff[boot.integers(0,ntest,(2000,ntest))].mean(1)
        comparison[target] = dict(nmse_difference=float(diff.mean()),
             paired_bootstrap_95ci=np.quantile(draws,[.025,.975]).tolist(),
             series_win_rate=float(np.mean(diff<0)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    horizon_rows = []
    for name, prediction in predictions.items():
        for h in range(horizon):
            horizon_rows.append(dict(model=name,horizon=h+1,
                nmse=float(np.mean(((prediction[:,h]-clean[ids,-horizon+h])/scale[:,h])**2))))
    pd.DataFrame(horizon_rows).to_csv(out/'horizon_metrics.csv',index=False)
    np.savez_compressed(out/'predictions.npz', series_ids=ids, scale=scale, **predictions)
    sample_rows=[]
    for j in range(6):
        for t in range(length):
            sample_rows.append(dict(series_id=int(ids[j]),t=t,clean=float(clean[ids[j],t]),
                observed=float(observed[ids[j],t]),
                **{name:float(p[j,t-(length-horizon)]) if t>=length-horizon else None for name,p in predictions.items()}))
    pd.DataFrame(sample_rows).to_csv(out/'sample_examples.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot([r['epoch'] for r in records[1:]],[r['train_loss'] for r in records[1:]],label='Train (sampled batches)')
    ax.plot([r['epoch'] for r in records],[r['val_loss'] for r in records],label='Validation (fixed series/horizons)')
    ax.set(xlabel='Epoch',ylabel='Scale-normalized MSE',title='ForecastPFN-small learning curve')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'loss.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3,2,figsize=(12,9))
    for j, ax in enumerate(axes.flat):
        ax.plot(np.arange(length),observed[ids[j]],color='0.65',label='Observed')
        ax.plot(np.arange(length),clean[ids[j]],color='black',label='Clean signal')
        for name in ['ForecastPFN-small','SeasonalNaive-7']:
            ax.plot(np.arange(length-horizon,length),predictions[name][j],label=name)
        ax.axvline(length-horizon-.5,color='0.4',linestyle=':')
        ax.set(title=f'Test series {j} | {configs[ids[j]]}',xlabel='Day index',ylabel='Synthetic value')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'samples.png',dpi=150);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for ax,target in zip(axes,['clean','observed']):
        rows=[r for r in metrics if r['target']==target]
        ax.barh([r['model'] for r in rows],[r['nmse'] for r in rows],color='steelblue')
        ax.set(title=f'Test target: {target}',xlabel='Scale-normalized MSE (lower is better)')
    fig.tight_layout();fig.savefig(out/'baselines.png',dpi=150);plt.close(fig)
    summary = dict(seed=args.seed,python=platform.python_version(),tensorflow=tf.__version__,numpy=np.__version__,
        pandas=pd.__version__,device='CPU',parameters=model.count_params(),ntrain=ntrain,nval=nval,ntest=ntest,
        context=context,horizon=horizon,length=length,frequency='daily',epochs=args.epochs,steps=args.steps,batch=32,
        learning_rate=.001,best_epoch=best_epoch,initial_val_loss=initial,best_val_loss=best,
        training_seconds=training_seconds,inference_seconds=inference_seconds,total_seconds=time.perf_counter()-started,
        negative_clean_fraction=float(np.mean(clean<0)),metrics=metrics,comparison=comparison,
        source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['src/training/models.py','src/training/scalers.py','src/synthetic_generation/generate_series.py']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
