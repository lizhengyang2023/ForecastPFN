"""Train one deterministic Mixup run, select on real validation, then evaluate test."""
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.utils.datasets import write_json
from experiments.utils.evaluate import metrics, predict
from experiments.utils.models import build_model
from experiments.utils.randomness import seed_everything
from .sampling import BatchSampler, blend


def run_one(output, model_name, mode, strategy, weight, seed, spec, config, splits,
            synthetic, manifest, device, identity, index):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = dict(model=model_name, mode=mode, strategy=strategy, weight=weight, seed=seed, spec=spec,
                   config=config, real_sha256=manifest['sha256'], identity=identity,
                   device=str(device))
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    result_path = output/'metrics.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['signature'] != signature:
            raise ValueError(f'Configuration changed in {output}; use a new output directory')
        return result, True
    write_json(output/'config.json', request)
    write_json(output/'split_manifest.json', manifest)
    seed_everything(seed)
    model = build_model(model_name, spec).to(device)
    initial_digest = hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes()
        for t in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    sampler = BatchSampler(len(splits['train']['x']), len(synthetic['x']),
        config['batch_size'], config['augmentation_probability'], seed, index, mode, strategy)
    real_x = torch.as_tensor(splits['train']['x'], device=device)
    real_y = torch.as_tensor(splits['train']['y'], device=device)
    syn_x = torch.as_tensor(synthetic['x'], device=device)
    syn_y = torch.as_tensor(synthetic['y'], device=device)
    best_rmse, best_step, best_state = float('inf'), 0, None
    best_validation = None
    started = time.perf_counter()
    rows = []
    diagnostic_sum = np.zeros(4, dtype='float64')
    diagnostic_count = 0
    for step in range(1, config['steps']+1):
        anchors, mask_np, partners = sampler.take()
        mask = torch.as_tensor(mask_np, device=device)
        x, y = real_x[anchors], real_y[anchors]
        if mode == 'real_real':
            px, py = real_x[partners], real_y[partners]
        elif mode == 'real_synthetic':
            px, py = syn_x[partners], syn_y[partners]
        x, y = blend(x, y, px, py, mask, weight)
        # Mean history/target standard deviations before and after blending.
        diagnostic_sum += np.array([real_x[anchors][mask].std(dim=1).mean().item(),
            x[mask].std(dim=1).mean().item(), real_y[anchors][mask].std(dim=1).mean().item(),
            y[mask].std(dim=1).mean().item()]) * len(partners)
        diagnostic_count += len(partners)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x)-y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}: {output}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
        optimizer.step()
        if step % config['eval_every'] == 0 or step == config['steps']:
            validation_prediction = predict(model, splits['validation'], device,
                config['eval_batch_size'])
            validation = metrics(validation_prediction, splits['validation']['raw_y'],
                manifest['mase_denominator'])
            rows.append(dict(step=step, train_loss=float(loss.detach().cpu()),
                validation_mae=validation['mae'], validation_rmse=validation['rmse'],
                anchor_exposures=int(sampler.anchor.exposures.sum()),
                partner_exposures=int(sampler.real_partner_exposures.sum()
                    + sampler.synthetic_partner_exposures.sum())))
            if validation['rmse'] < best_rmse:
                best_rmse, best_step, best_validation = validation['rmse'], step, validation
                best_state = copy.deepcopy(model.state_dict())
    with (output/'train_log.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(best_state, output/'best.pt')
    final_state = copy.deepcopy(model.state_dict())
    torch.save(final_state, output/'final.pt')
    checkpoints = {}
    predictions = {}
    for name, state, step in [('best', best_state, best_step),
                              ('final', final_state, config['steps'])]:
        model.load_state_dict(state)
        val_pred = predict(model, splits['validation'], device, config['eval_batch_size'])
        test_pred = predict(model, splits['test'], device, config['eval_batch_size'])
        checkpoints[name] = dict(step=step,
            validation=metrics(val_pred, splits['validation']['raw_y'], manifest['mase_denominator']),
            test=metrics(test_pred, splits['test']['raw_y'], manifest['mase_denominator']))
        predictions[f'{name}_test'] = test_pred
    assert abs(checkpoints['best']['validation']['rmse']-best_validation['rmse']) < 1e-9
    np.savez_compressed(output/'predictions.npz', truth=splits['test']['raw_y'],
        origins=splits['test']['origins'], **predictions)
    np.savez_compressed(output/'exposures.npz', anchors=sampler.anchor.exposures,
        real_partners=sampler.real_partner_exposures,
        synthetic_partners=sampler.synthetic_partner_exposures)
    result = dict(signature=signature, model=model_name, mode=mode, strategy=strategy,
        weight=weight,
        seed=seed, initial_digest=initial_digest, final_step=config['steps'],
        batch_size=config['batch_size'], augmentation_probability=config['augmentation_probability'],
        exposures=dict(anchors=int(sampler.anchor.exposures.sum()),
            real_partners=int(sampler.real_partner_exposures.sum()),
            synthetic_partners=int(sampler.synthetic_partner_exposures.sum())),
        unique_anchors=int((sampler.anchor.exposures>0).sum()),
        diagnostics=dict(zip(['real_history_std','mixed_history_std','real_target_std',
                              'mixed_target_std'], (diagnostic_sum/diagnostic_count).tolist()))
            if diagnostic_count else {},
        matching=sampler.diagnostics(),
        checkpoints=checkpoints, elapsed_seconds=time.perf_counter()-started)
    write_json(result_path, result)  # Completion marker is written last.
    return result, False
