"""One independently initialized run: fixed updates, validation selection, then test."""
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
from experiments.utils.sampler import CyclingSampler, quota


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_one(output, name, fraction, seed, spec, config, splits, synthetic, manifest, device, identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    request = dict(model=name, fraction=fraction, seed=seed, spec=spec, config=config,
                   real_sha256=manifest['sha256'], identity=identity, device=str(device))
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    result_path = output/'metrics.json'
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing['signature'] != signature:
            raise ValueError(f'Completed run configuration differs: {output}; use another output directory')
        return existing, True
    # An incomplete run is restarted deterministically, not resumed midway.
    write_json(output/'config.json', request)
    write_json(output/'split_manifest.json', manifest)
    seed_everything(seed)
    model = build_model(name, spec).to(device)
    initial_digest = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()
                                             for v in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    nr, ns = quota(config['batch_size'], fraction)
    real_sampler = CyclingSampler(len(splits['train']['x']), seed+1000)
    syn_sampler = CyclingSampler(len(synthetic['x']), seed+2000)
    shuffle = np.random.default_rng(seed+3000)
    real_x = torch.as_tensor(splits['train']['x'], device=device)
    real_y = torch.as_tensor(splits['train']['y'], device=device)
    syn_x = torch.as_tensor(synthetic['x'], device=device)
    syn_y = torch.as_tensor(synthetic['y'], device=device)
    best_score, best_step, best_state = float('inf'), 0, None
    best_val = None
    train_seconds = validation_seconds = checkpoint_seconds = 0.
    losses = []
    rows = []
    started = time.perf_counter()
    synchronize(device)
    interval_started = time.perf_counter()
    fields = ['step','train_loss','real_loss','synthetic_loss','validation_mae','validation_rmse',
              'real_exposures','synthetic_exposures','elapsed_seconds','train_seconds']
    with (output/'train_log.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(1, config['steps']+1):
            model.train()
            ri, si = real_sampler.take(nr), syn_sampler.take(ns)
            # Whole windows are concatenated along batch, never along time.
            x = torch.cat([real_x[ri], syn_x[si]], dim=0)
            y = torch.cat([real_y[ri], syn_y[si]], dim=0)
            source = torch.cat([torch.zeros(nr, device=device), torch.ones(ns, device=device)])
            order = torch.as_tensor(shuffle.permutation(config['batch_size']), device=device)
            x, y, source = x[order], y[order], source[order]
            optimizer.zero_grad(set_to_none=True)
            window_loss = (model(x)-y).square().mean(dim=1)
            loss = window_loss.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {output}, step={step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(torch.stack([loss.detach(),
                window_loss[source==0].detach().mean() if nr else loss.detach()*0,
                window_loss[source==1].detach().mean() if ns else loss.detach()*0]))
            if step % config['eval_every'] == 0 or step == config['steps']:
                synchronize(device)
                train_seconds += time.perf_counter()-interval_started
                t0 = time.perf_counter()
                prediction = predict(model, splits['validation'], device, config['eval_batch_size'])
                val = metrics(prediction, splits['validation']['raw_y'], manifest['mase_denominator'])
                validation_seconds += time.perf_counter()-t0
                means = torch.stack(losses).mean(0).cpu().tolist()
                losses = []
                row = dict(step=step, train_loss=means[0], real_loss=means[1] if nr else None,
                    synthetic_loss=means[2] if ns else None, validation_mae=val['mae'], validation_rmse=val['rmse'],
                    real_exposures=nr*step, synthetic_exposures=ns*step,
                    elapsed_seconds=time.perf_counter()-started, train_seconds=train_seconds)
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                if val['rmse'] < best_score:
                    best_score, best_step, best_val = val['rmse'], step, val
                    t0 = time.perf_counter()
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    checkpoint_seconds += time.perf_counter()-t0
                synchronize(device)
                interval_started = time.perf_counter()
    train_wall_seconds = time.perf_counter()-started
    assert best_state is not None
    assert int(real_sampler.exposures.sum()) == nr*config['steps']
    assert int(syn_sampler.exposures.sum()) == ns*config['steps']
    np.savez_compressed(output/'exposures.npz', real=real_sampler.exposures, synthetic=syn_sampler.exposures)
    checkpoints = {'final':copy.deepcopy(model.state_dict()), 'best':best_state}
    report = dict(signature=signature, model=name, fraction=fraction, seed=seed,
        parameters=sum(p.numel() for p in model.parameters()), initial_weights_sha256=initial_digest,
        best_step=best_step, final_step=config['steps'], selection_metric='validation original-unit RMSE',
        train_seconds=train_seconds, validation_seconds=validation_seconds,
        train_wall_seconds=train_wall_seconds, checkpoint_copy_seconds=checkpoint_seconds,
        source_batch_counts={'real':nr,'synthetic':ns},
        exposures={'real':int(real_sampler.exposures.sum()), 'synthetic':int(syn_sampler.exposures.sum())},
        unique_exposed={'real':int((real_sampler.exposures>0).sum()),'synthetic':int((syn_sampler.exposures>0).sum())},
        best_exposures={'real':nr*best_step,'synthetic':ns*best_step}, checkpoints={})
    saved = dict(test_origins=splits['test']['origins'], truth=splits['test']['raw_y'])
    # The test set is accessed only after all optimization/selection is finished.
    test_started = time.perf_counter()
    for label, state in checkpoints.items():
        model.load_state_dict(state)
        torch.save({k:v.detach().cpu() for k,v in state.items()}, output/f'{label}.pt')
        prediction = predict(model, splits['test'], device, config['eval_batch_size'])
        test = metrics(prediction, splits['test']['raw_y'], manifest['mase_denominator'])
        report['checkpoints'][label] = dict(validation=best_val if label=='best' else val, test=test)
        saved[label] = prediction
    report['test_and_save_seconds'] = time.perf_counter()-test_started
    np.savez_compressed(output/'predictions.npz', **saved)
    report['total_seconds'] = time.perf_counter()-started
    write_json(result_path, report)  # Completion marker, written last.
    return report, False
