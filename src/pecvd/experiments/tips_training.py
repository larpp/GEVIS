"""Train/evaluate an explicit TIPS-native GEVIS experiment, no PECVD assets needed."""
from pathlib import Path
import copy
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .tips_dataset import TipsData, sha
from .tips_model import TipsModel, SyntheticCases, classification_metrics
from .fusion_answers import KINDS, QUESTIONS, ANSWER_CRITERIA
from .peald_readout import auxiliary_losses
from .tips_optimization import PairedBatchSampler, BatchProgress

SCHEMA = 'gevis.tips_native_scratch.v1'


def configure(config):
    if config['mode'] != 'tips_native_scratch':
        raise ValueError('This entry explicitly trains on TIPS; it is not PECVD frozen transfer')
    if config['window_length'] % config['patch_length'] or not 2 <= config['window_length'] // config['patch_length'] <= 64:
        raise ValueError('Window must contain 2..64 complete native patches')
    if config['fusion_width'] % 4:
        raise ValueError('Fusion width must be divisible by four attention heads')
    for name in ['encoder_epochs', 'answer_epochs', 'batch_size', 'train_windows_per_run',
                 'evaluation_windows_per_run', 'encoder_layers', 'patience']:
        if config[name] < 1:
            raise ValueError('Positive training budgets required: ' + name)
    if config['cases_per_window'] < 24 or config['learning_rate'] <= 0:
        raise ValueError('At least 24 cases/window and a positive learning rate required')
    if config.get('trace_cache_mb', 256) < 0 or config.get('log_every_batches', 50) < 1 or config.get('log_every_seconds', 10) <= 0:
        raise ValueError('Nonnegative trace cache and positive progress intervals required')
    if not config['target_sensors'] or any(n.startswith('ValveAct_')
                                          for n in config['target_sensors']):
        raise ValueError('Choose numeric sensors as perturbation targets; states remain context only')


def json_write(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert), encoding='utf-8')


def data_for(config, gevis_root):
    root = Path(gevis_root)
    resolve = lambda s: (root / s).resolve()
    return TipsData(resolve(config['tips_root']), resolve(config['split_dir']),
        resolve(config['columns_file']) if config.get('columns_file') else None,
        main_steps=config['main_steps'], selected_sensors=config.get('selected_sensors'),
        selection_identity=config.get('selection'),
        window_length=config['window_length'], stride=config['stride'],
        window_scope=config['window_scope'], seed=config['seed'])


def experiment_identity(data, config):
    files = ['tips_dataset.py', 'tips_model.py', 'tips_training.py', 'tips_profiles.py', 'tips_optimization.py', 'fusion_answers.py',
             'injection.py', 'peald_readout.py', '../multiscale_patchtst.py',
             '../temporal_patchtst.py', '../temporal_bridge.py', '../gevis_backbone.py',
             '../answer_display.py', '../relational_features.py']
    folder = Path(__file__).parent
    return dict(schema=SCHEMA, data=data.identity, config=config,
        source_sha256={name: sha(folder / name) for name in files},
        questions={k: QUESTIONS[k] for k in KINDS},
        options={k: list(ANSWER_CRITERIA[k]) for k in KINDS})


def choose_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly choose --device cpu if intended')
    return device


def batch_to(batch, device):
    values = [batch[k].to(device) for k in ['current', 'reference', 'center', 'step', 'position']]
    if 'current_key' in batch:
        values.append(dict(current=batch['current_key'], reference=batch['reference_key']))
    return values


def progress_for(label, loader, config, output):
    def save(payload):
        temporary = output / 'progress.json.tmp'
        json_write(temporary, payload)
        temporary.replace(output / 'progress.json')
    return BatchProgress(label, len(loader), every=config.get('log_every_batches', 50),
        seconds=config.get('log_every_seconds', 10), callback=save)


def loss_for(logits, labels):
    ce = torch.stack([F.cross_entropy(z, labels[:, i]) for i, z in enumerate(logits)]).mean()
    consistency, ordinal = auxiliary_losses(logits, labels[:, 2])
    return ce + .2 * consistency + .2 * ordinal


@torch.no_grad()
def score(model, loader, device, *, timing=False, collect_rows=True, progress=None):
    model.eval(); labels = []; predictions = []; rows = []; latency = []
    for batch_index, batch in enumerate(loader, 1):
        args = batch_to(batch, device)
        if timing and device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model(*args)
        if timing and device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000
        if timing:
            latency.append(dict(batch_size=len(batch['center']), batch_ms=elapsed,
                                amortized_ms=elapsed / len(batch['center'])))
        y = batch['labels'].numpy(); pred = torch.stack([z.argmax(-1) for z in logits], -1).cpu().numpy()
        labels.append(y); predictions.append(pred)
        for j in range(len(y) if collect_rows else 0):
            for i, kind in enumerate(KINDS):
                rows.append(dict(run_id=int(batch['run_id'][j]), step=int(batch['step'][j]),
                    window_index=int(batch['window_index'][j]), center_sensor=model.sensors[int(batch['center'][j])],
                    answer_type=kind, label=int(y[j, i]), prediction=int(pred[j, i])))
        if progress:
            progress.update(batch_index)
    if not labels:
        raise ValueError('No evaluation examples')
    return classification_metrics(np.concatenate(labels), np.concatenate(predictions)), rows, latency


def case_loader(windows, data, config, partition, *, shuffle=False):
    seed = config['seed'] + {'train': 0, 'validation': 100003, 'test': 200003}[partition]
    cases = SyntheticCases(windows, data.requested, config['target_sensors'],
        cases_per_window=config['cases_per_window'], seed=seed,
        varying_by_step={step: st['varying'] for step, st in data.stats.items()})
    sampler = PairedBatchSampler(cases, config['batch_size'], shuffle=shuffle, seed=seed)
    return torch.utils.data.DataLoader(cases, batch_sampler=sampler, num_workers=0)


def train(data, config, output, identity, device):
    """TRAIN optimization / Validation model selection. Never reads Test sensor rows."""
    if (output / 'model.pt').exists():
        raise FileExistsError('Checkpoint exists. Use evaluate or choose a new --output directory.')
    stats = data.fit()
    missing = set(config['target_sensors']) - set(data.requested)
    if missing:
        raise ValueError('Target sensors absent from input columns: ' + str(sorted(missing)))
    unusable = [s for s in config['target_sensors'] if not any(
        st['varying'][data.requested.index(s)] for st in stats.values())]
    if len(unusable) == len(config['target_sensors']):
        raise ValueError('All selected targets have no TRAIN variation: ' + str(unusable))
    if unusable:
        print('TRAIN-constant selected sensors retained without synthetic intervention:', unusable, flush=True)
    print('Preparing Train/Validation windows; reserved and Test measurements are not read.', flush=True)
    windows = data.windows('train', config['train_windows_per_run'])
    validation = data.windows('validation', config['evaluation_windows_per_run'])
    if {w['run_id'] for w in windows} & {w['run_id'] for w in validation}:
        raise ValueError('Train/Validation overlap')
    model = TipsModel(data.requested, config).to(device)
    encoder_loader = torch.utils.data.DataLoader(np.stack([w['values'] for w in windows]),
        batch_size=config['batch_size'], shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(config['seed']))
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=config['learning_rate'])
    history = []
    for epoch in range(config['encoder_epochs']):
        model.encoder.train(); losses = []; loss_total = 0.
        progress = progress_for(f'Encoder {epoch + 1}/{config["encoder_epochs"]}', encoder_loader, config, output)
        for batch_index, x in enumerate(encoder_loader, 1):
            x = x.to(device); target, _ = model.encoder.patchify(x)
            loss = F.mse_loss(model.encoder.blind_prediction(x), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite encoder loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
            loss_total += losses[-1]
            progress.update(batch_index, loss_total / batch_index)
        history.append(dict(stage='encoder', epoch=epoch + 1, train_loss=float(np.mean(losses))))
        print(f'Encoder {epoch + 1}/{config["encoder_epochs"]}: loss={np.mean(losses):.6f}', flush=True)
        json_write(output / 'history.json', history)
    model.freeze_encoder()
    train_loader = case_loader(windows, data, config, 'train', shuffle=True)
    val_loader = case_loader(validation, data, config, 'validation')
    print(f'Answer examples: Train={len(train_loader.dataset):,}, Validation={len(val_loader.dataset):,}; '
          f'batches/epoch={len(train_loader):,}+{len(val_loader):,}; all original cases retained. '
          f'Frozen encoder host cache limit={config.get("trace_cache_mb", 256)} MiB.', flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=1e-4)
    best = None; stale = 0; best_state = None; best_scores = None
    for epoch in range(config['answer_epochs']):
        model.train(); losses = []; loss_total = 0.
        progress = progress_for(f'Answers {epoch + 1}/{config["answer_epochs"]}', train_loader, config, output)
        for batch_index, batch in enumerate(train_loader, 1):
            logits = model(*batch_to(batch, device))
            loss = loss_for(logits, batch['labels'].to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite answer loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step(); losses.append(float(loss.detach()))
            loss_total += losses[-1]
            progress.update(batch_index, loss_total / batch_index)
        scores, _, _ = score(model, val_loader, device, collect_rows=False,
            progress=progress_for(f'Validation {epoch + 1}/{config["answer_epochs"]}', val_loader, config, output))
        key = float(np.mean([scores[k]['macro_f1'] for k in KINDS]))
        history.append(dict(stage='answers', epoch=epoch + 1, train_loss=float(np.mean(losses)),
                            validation=scores, frozen_trace_cache=model.trace_cache.stats()))
        print(f'Answers {epoch + 1}/{config["answer_epochs"]}: validation mean macro-F1={key:.4f}', flush=True)
        if best is None or key > best:
            best = key; stale = 0; best_scores = copy.deepcopy(scores)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        json_write(output / 'history.json', history)
        if stale >= config['patience']:
            break
    if best_state is None:
        raise RuntimeError('No optimized model; refusing to save an untrained checkpoint')
    # Recheck the immutable inputs and loaded-code identity across the full fit span.
    verified = TipsData(data.root, data.split, data.columns_file,
        selected_sensors=data.requested if data.columns_file is None else None,
        selection_identity=data.selection_identity,
        main_steps=config['main_steps'], window_length=config['window_length'],
        stride=config['stride'], window_scope=config['window_scope'], seed=config['seed'])
    if experiment_identity(verified, config) != identity:
        raise ValueError('Source data/config/code changed during training; no model saved')
    payload = dict(schema=SCHEMA, identity=identity, config=config, sensors=data.requested,
        train_stats=stats, state=best_state, validation=best_scores,
        trained_steps=sorted({w['step'] for w in windows}),
        train_runs=sorted({w['run_id'] for w in windows}),
        validation_runs=sorted({w['run_id'] for w in validation}),
        mode='TIPS-native encoder pretraining then Fusion+head training; not frozen PECVD transfer')
    for name, path in config.get('selection', {}).get('files', {}).items():
        if sha(path) != config['selection']['sha256'][name]:
            raise ValueError('Selection profile changed during training; no model saved: ' + name)
    with (output / 'model.pt').open('xb') as f:
        torch.save(payload, f)
    pd.DataFrame(data.audit).to_csv(output / 'window_audit.csv', index=False)
    json_write(output / 'training_summary.json', {k: v for k, v in payload.items() if k not in ('state', 'train_stats')})
    print('Saved Validation-selected model:', output / 'model.pt', flush=True)


def evaluate(data, config, output, identity, device):
    checkpoint = output / 'model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError('Train the TIPS-native model first: ' + str(checkpoint))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload['identity'] != identity:
        raise ValueError('Checkpoint data/config/code identity mismatch; no silent reuse')
    data.stats = {int(k): v for k, v in payload['train_stats'].items()}
    windows = data.windows('test', config['evaluation_windows_per_run'])
    if set(payload['train_runs'] + payload['validation_runs']) & {w['run_id'] for w in windows}:
        raise ValueError('Test overlaps fitted or selected Runs')
    supported = set(payload['trained_steps'])
    if any(w['step'] not in supported for w in windows):
        raise ValueError('Test contains an untrained Main Step')
    model = TipsModel(data.requested, config).to(device)
    model.load_state_dict(payload['state'], strict=True); model.freeze_encoder()
    loader = case_loader(windows, data, config, 'test')
    # Warm-up is excluded. Batch latency is not a single-window/Edge KPI.
    with torch.no_grad():
        model.eval(); model(*batch_to(next(iter(loader)), device))
    metrics, rows, latency = score(model, loader, device, timing=True,
        progress=progress_for('Test', loader, config, output))
    report = dict(task='held-out paired synthetic changes; not real fault accuracy',
        reference='exact pre-injection Test window; not cross-run deployment reference',
        split_policy='original TIPS whole-Run chronological partitions; reserved_transfer excluded',
        metrics=metrics, checkpoint_sha256=sha(checkpoint),
        latency_scope='batched model forward with bounded frozen trace cache including text/context; excludes I/O, injection, queues; not cold unseen-window latency',
        frozen_trace_cache=model.trace_cache.stats(),
        batch_ms_mean=float(np.mean([r['batch_ms'] for r in latency])),
        batch_ms_p95=float(np.quantile([r['batch_ms'] for r in latency], .95)),
        amortized_ms_mean=float(np.mean([r['amortized_ms'] for r in latency])))
    details = pd.DataFrame(rows)
    by_step = []; by_sensor = []
    for column, target in [('step', by_step), ('center_sensor', by_sensor)]:
        for (name, kind), g in details.groupby([column, 'answer_type']):
            y = g.label.to_numpy(); p = g.prediction.to_numpy(); count = len(ANSWER_CRITERIA[kind])
            matrix = np.zeros((count, count), dtype=int); np.add.at(matrix, (y, p), 1)
            tp = matrix.diagonal(); support = matrix.sum(0) + matrix.sum(1)
            f1 = np.divide(2 * tp, support, out=np.zeros(count), where=support > 0)
            target.append({column: name, 'answer_type': kind, 'questions': len(g),
                           'accuracy': float((y == p).mean()), 'macro_f1': float(f1[support > 0].mean())})
    details.to_csv(output / 'test_predictions.csv', index=False)
    pd.DataFrame(by_step).to_csv(output / 'test_by_step.csv', index=False)
    pd.DataFrame(by_sensor).to_csv(output / 'test_by_sensor.csv', index=False)
    pd.DataFrame(latency).to_csv(output / 'test_batch_latency.csv', index=False)
    pd.DataFrame(data.audit).to_csv(output / 'test_window_audit.csv', index=False)
    json_write(output / 'test_metrics.json', report)
    print(json.dumps(metrics, indent=2), flush=True)
    return report


def run(config, gevis_root, *, action='train-evaluate', device='auto', output=None):
    from .tips_profiles import resolve_config
    config = resolve_config(config, gevis_root)
    configure(config)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    data = data_for(config, gevis_root)
    identity = experiment_identity(data, config)
    stamp = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    output = Path(output) if output else Path(gevis_root) / 'artifacts/tips_peald_scratch' / stamp
    if action == 'prepare':
        data.fit(); windows = data.windows('train', config['train_windows_per_run'])
        print(dict(train_runs=len(data.rows['train']), validation_runs=len(data.rows['valid']),
            test_runs=len(data.rows['test']), reserved_runs=len(data.rows['reserved_transfer']),
            input_columns=len(data.requested), train_windows=len(windows),
            selected_main_steps=config['main_steps'], selected_sensors=data.requested,
            eligible_main_steps=sorted({w['step'] for w in windows}), output=str(output)))
        return
    selected = choose_device(device)
    output.mkdir(parents=True, exist_ok=True)
    print('Experiment:', SCHEMA, 'device:', selected, 'output:', output, flush=True)
    if action in ('train', 'train-evaluate'):
        train(data, config, output, identity, selected)
    if action in ('evaluate', 'train-evaluate'):
        return evaluate(data, config, output, identity, selected)
