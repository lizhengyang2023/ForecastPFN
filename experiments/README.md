# Quick validation

For the revised real/synthetic mixing experiment (eight evenly spaced synthetic
shares and five small models), see [the Markdown validation plan](real_synthetic_validation_plan.md).
It supersedes the earlier canvas design; the mixing experiment has not yet run.

This experiment imports the repository's `generate` function, `TransformerModel`,
`TransformerBlock`, and robust scaler. It bypasses the cloud-specific TFRecord
writer and the hard-coded 700-epoch training entry point. No pretrained weights
are used. Three unused `tensorflow_io` imports and a per-forward debug print were
removed from the imported source files; numerical model logic is unchanged.

## Run

On this machine, conda base is `D:\software\Anacoda\python.exe` (Python 3.11.7).
The `conda run` launcher was broken, so the base interpreter is called directly.
TensorFlow 2.15.1 is installed into workspace-local `.experiment_deps`, which the
script prepends to `sys.path`; base packages are not changed.

```powershell
cd D:\Code\TimeSeries\ForecastPFN
& D:\software\Anacoda\python.exe -m pip install --target .experiment_deps tensorflow-cpu==2.15.1
& D:\software\Anacoda\python.exe experiments/quick_validation.py
& D:\software\Anacoda\python.exe experiments/quick_validation.py --epochs 100 --output experiments/results/quick_validation_100
& D:\software\Anacoda\python.exe experiments/verify_results.py
& D:\software\Anacoda\python.exe experiments/render_report.py
```

Defaults: seed 42; 768 training / 128 validation / 256 test independent daily
series, each 64 points; 36-point context, 10-step forecast; 25 epochs x 24 steps
x batch 32 = 19,200 randomly sampled training tasks. Each task selects one
series, one origin between 36 and 54, and one offset 0–9. Validation/test use the
final origin (54) and all ten horizons. Splits occur at series level before
sampling tasks. The first six test series are plotted without selecting examples
based on prediction quality.

The two original Transformer blocks are instantiated with two heads and key
dimension 18 (original: four heads, key dimension 72). Calendar encoding, task
embedding, robust scaling, GELU feed-forward layers, and ReLU output remain.
Only point prediction is trained, not mean/std auxiliary tasks. Adam uses 0.001
instead of the original 0.0001. The best checkpoint is selected exclusively by
validation scale-normalized MSE, including epoch 0 as a candidate.

## Metrics and interpretation

Training predicts the clean signal from noisy context, matching the original
point task. Evaluation reports both clean and noisy observation targets. All
methods see the same noisy context. Future observations are never used as inputs.
SeasonalNaive-7 repeats the last seven context points, LastValue repeats the last
point, ContextMean repeats the context mean, and Drift extrapolates the
first-to-last context slope. Predictions are not clipped for baselines.

MSE/MAE are in original synthetic units. NMSE means `mean((prediction-truth)^2 /
scale^2)`, where scale is the original robust scaler computed from context only;
the identical scale is used for every method. This is not variance-normalized
NMSE. sMAPE is `200*mean(abs(error)/(abs(prediction)+abs(truth)+1e-8))`.
The paired 95% bootstrap interval resamples entire test series (2,000 draws),
not correlated horizon points. It measures test-sample uncertainty for this one
trained model, not training-seed variability.

Training loss averages sampled batches while validation loss covers fixed
unseen series and all ten horizons, so they are not directly identical samples.
The generator can produce negative values, while the original scaler clips its
inputs and the output is ReLU; this original limitation is retained and the
negative fraction is recorded. The original noise implementation subtracts its
median, not its exact expectation; we retain that behavior too.

This is a small in-distribution synthetic validation, not a reproduction of the
paper's real-world zero-shot benchmark. Training time excludes dependency
installation and includes epoch validation/checkpointing. Inference time is
batched CPU evaluation for 256 series x 10 independently queried horizons.

## Outputs

`results/quick_validation/` contains `summary.json`, `loss.csv`, `metrics.csv`,
six-series `sample_examples.csv`, `loss.png`, `samples.png`, `baselines.png`,
the best weights, full sampled data, and full predictions. Binary weights/data
and downloaded dependencies are ignored by Git but remain available locally.

The 100-epoch exploratory follow-up is saved separately in
`results/quick_validation_100/`; its report includes the initial 25-epoch result.
This follow-up was chosen after observing underfitting in the first run, so it is
not a pre-registered independent test. Both runs use identical generated data and
initialization. Local dependency versions are recorded in `requirements-local.txt`.
