"""Original-unit metrics, complete evaluation batches and training-free baselines."""
import numpy as np
import torch


def metrics(prediction, truth, mase_denominator):
    error = prediction.astype('float64')-truth.astype('float64')
    if not np.isfinite(error).all():
        raise FloatingPointError('Nonfinite evaluation predictions')
    mae = float(np.abs(error).mean())
    return dict(mae=mae, rmse=float(np.sqrt(np.mean(error**2))), mse=float(np.mean(error**2)),
        mase=mae/mase_denominator if mase_denominator else None,
        horizon_mae=np.abs(error).mean(0).tolist(),
        horizon_rmse=np.sqrt((error**2).mean(0)).tolist(), windows=len(truth))


@torch.inference_mode()
def predict(model, split, device, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(split['x']), batch_size):
        x = torch.as_tensor(split['x'][start:start+batch_size], device=device)
        normalized = model(x).cpu().numpy()
        predictions.append(normalized*split['scale'][start:start+batch_size]+split['mean'][start:start+batch_size])
    return np.concatenate(predictions)


def baseline_predictions(split, season):
    x, h = split['raw_x'], split['y'].shape[1]
    return {'last':np.repeat(x[:, -1:], h, axis=1),
            'mean':np.repeat(x.mean(1, keepdims=True), h, axis=1),
            'seasonal_naive':x[:, -season:][:, np.arange(h)%season]}


def evaluate_baselines(splits, spec, denominator):
    return {name:{model:metrics(pred, split['raw_y'], denominator)
                  for model, pred in baseline_predictions(split, spec['season']).items()}
            for name, split in splits.items() if name != 'train'}
