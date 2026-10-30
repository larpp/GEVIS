"""Fixed six STEP 1 encoders + complete STEP 2; cache features, train readout only."""
from pathlib import Path
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from .tips_dataset import sha
from .tips_profiles import resolve_config
from .tips_training import (data_for, choose_device, batch_to, json_write, loss_for, progress_for)
from .tips_model import SyntheticCases, classification_metrics
from .tips_optimization import PairedBatchSampler
from .tips_frozen_source import FrozenSource, inspect_bundle
from .fusion_answers import ContextReadout, KINDS

SCHEMA = 'gevis.tips_frozen_step12_readout.v1'


def identity_for(data, config, source):
    folder = Path(__file__).parent
    files = ['tips_frozen_training.py', 'tips_frozen_source.py', 'tips_dataset.py',
        'tips_profiles.py', 'tips_model.py', 'tips_training.py', 'tips_optimization.py',
        'fusion_answers.py', 'questions.py', 'frozen_transfer.py', 'injection.py', 'peald_readout.py',
        '../encoder_loading.py', '../pretrained_patchtst.py', '../legacy_patchtst.py',
        '../temporal_bridge.py', '../gevis_backbone.py', '../answer_display.py']
    return dict(schema=SCHEMA, data=data.identity, config=config, frozen_source=source,
        sources={f: sha(folder / f) for f in files})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cases_for(windows, data, config, partition):
    return SyntheticCases(windows, data.requested, config['target_sensors'],
        cases_per_window=config['cases_per_window'],
        seed=config['seed'] + dict(train=0, validation=100003, test=200003)[partition],
        varying_by_step={s: v['varying'] for s, v in data.stats.items()})


def cached_features(source, windows, data, config, partition, output, identity, device):
    """Final post-Cross-attention features are computed ONCE, never per epoch."""
    cases = cases_for(windows, data, config, partition)
    folder = output / 'features' / partition; folder.mkdir(parents=True, exist_ok=True)
    paths = {n: folder / (n + '.npy') for n in ('x', 'y', 'meta')}
    marker = folder / 'manifest.json'; cache_key = digest(dict(identity=identity, partition=partition))
    if marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['identity'] != cache_key or saved['rows'] != len(cases):
            raise ValueError('Frozen feature cache identity mismatch')
        for name, path in paths.items():
            if sha(path) != saved['sha256'][name]:
                raise ValueError('Frozen feature cache hash mismatch: ' + name)
        print(f'Reusing completed frozen {partition} features: {len(cases):,} examples', flush=True)
    else:
        required = len(cases) * (3 * source.feature_dim * 4 + 3 * 8 + 4 * 8)
        if required > config.get('feature_cache_limit_mb', 4096) * 1024**2:
            raise ValueError('Feature cache partition exceeds feature_cache_limit_mb')
        arrays = dict(x=np.lib.format.open_memmap(str(paths['x']) + '.tmp', mode='w+', dtype='float32',
                          shape=(len(cases), 3, source.feature_dim)),
                      y=np.lib.format.open_memmap(str(paths['y']) + '.tmp', mode='w+', dtype='int64', shape=(len(cases), 3)),
                      meta=np.lib.format.open_memmap(str(paths['meta']) + '.tmp', mode='w+', dtype='int64', shape=(len(cases), 4)))
        # Sequential center pairs maximize reference reuse. No training shuffle
        # is needed until the final frozen representations have been prepared.
        loader = torch.utils.data.DataLoader(cases,
            batch_sampler=PairedBatchSampler(cases, config['feature_batch_size']), num_workers=0)
        progress = progress_for(f'Frozen features {partition}', loader, config, output)
        offset = 0
        source.to_device(device)
        with torch.no_grad():
            for bi, batch in enumerate(loader, 1):
                features = source.features(*batch_to(batch, device)).cpu().numpy()
                if not np.isfinite(features).all():
                    raise ValueError('Nonfinite frozen source features')
                end = offset + len(features)
                arrays['x'][offset:end] = features
                arrays['y'][offset:end] = batch['labels'].numpy()
                arrays['meta'][offset:end] = torch.stack([batch[k] for k in
                    ('run_id', 'step', 'center', 'window_index')], -1).numpy()
                offset = end; progress.update(bi)
        for name in list(arrays):
            arrays[name].flush(); del arrays[name]
            Path(str(paths[name]) + '.tmp').replace(paths[name])
        source.verify_frozen()
        json_write(marker, dict(identity=cache_key, rows=len(cases), feature_dim=source.feature_dim,
            sha256={name: sha(path) for name, path in paths.items()},
            frozen_source_unchanged=True, columns=['run_id', 'step', 'center', 'window_index']))
    result = {n: np.load(path, mmap_mode='r') for n, path in paths.items()}
    if result['x'].shape != (len(cases), 3, source.feature_dim):
        raise ValueError('Feature cache shape mismatch')
    return result


def feature_batches(cache, batch_size, *, shuffle=False, seed=42):
    order = np.arange(len(cache['y']))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        ix = order[start:start + batch_size]
        yield torch.from_numpy(np.array(cache['x'][ix], copy=True)), torch.from_numpy(np.array(cache['y'][ix], copy=True))


def fit_normalization(head, cache):
    count = 0; total = np.zeros(cache['x'].shape[-1]); squares = total.copy()
    for start in range(0, len(cache['x']), 1024):
        values = np.asarray(cache['x'][start:start + 1024], dtype=np.float64).reshape(-1, len(total))
        count += len(values); total += values.sum(0); squares += (values * values).sum(0)
    mean = total / count; scale = np.sqrt(np.maximum(0, squares / count - mean * mean))
    with torch.no_grad():
        head.feature_mean.copy_(torch.from_numpy(mean))
        head.feature_scale.copy_(torch.from_numpy(np.where(scale > 1e-6, scale, 1.)))


def loader_size(cache, batch_size):
    return range((len(cache['y']) + batch_size - 1) // batch_size)


@torch.no_grad()
def score(head, cache, config, output, device, label):
    head.eval(); predictions = []; timings = []
    progress = progress_for(label, loader_size(cache, config['batch_size']), config, output)
    for bi, (x, _) in enumerate(feature_batches(cache, config['batch_size']), 1):
        x = x.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        pred = torch.stack([head(x[:, i], k).argmax(-1) for i, k in enumerate(KINDS)], -1)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - started) * 1000)
        predictions.append(pred.cpu().numpy()); progress.update(bi)
    predictions = np.concatenate(predictions)
    return classification_metrics(np.asarray(cache['y']), predictions), predictions, timings


def train(data, source, config, output, identity, device):
    if (output / 'model.pt').exists():
        raise FileExistsError('Readout checkpoint exists; use evaluate or a new output directory')
    stats = data.fit()
    if not any(any(v['varying'][data.requested.index(s)] for s in config['target_sensors']) for v in stats.values()):
        raise ValueError('All selected target sensors are TRAIN-constant')
    windows = data.windows('train', config['train_windows_per_run'])
    validation = data.windows('validation', config['evaluation_windows_per_run'])
    if {w['run_id'] for w in windows} & {w['run_id'] for w in validation}:
        raise ValueError('Train/Validation run overlap')
    train_cache = cached_features(source, windows, data, config, 'train', output, identity, device)
    val_cache = cached_features(source, validation, data, config, 'validation', output, identity, device)
    source.verify_frozen()
    # Source modules are never supplied to an optimizer. Release GPU resources
    # during head fitting; persistent disk features are all the head needs.
    source.to_device(torch.device('cpu'))
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    head = ContextReadout(source.feature_dim, config['head_width'])
    fit_normalization(head, train_cache); head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    history = []; best = None; best_state = None; stale = 0; started = time.perf_counter()
    print(f'Frozen STEP 1/2: 0 trainable parameters; readout: {sum(p.numel() for p in head.parameters()):,}. '
          f'Cached Train={len(train_cache["y"]):,}, Validation={len(val_cache["y"]):,}.', flush=True)
    for epoch in range(config['answer_epochs']):
        head.train(); loss_sum = 0.; batches = 0
        progress = progress_for(f'Readout {epoch + 1}/{config["answer_epochs"]}',
            loader_size(train_cache, config['batch_size']), config, output)
        for bi, (x, y) in enumerate(feature_batches(train_cache, config['batch_size'], shuffle=True,
                                                  seed=config['seed'] + epoch), 1):
            x, y = x.to(device), y.to(device)
            logits = [head(x[:, i], k) for i, k in enumerate(KINDS)]
            loss = loss_for(logits, y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite readout loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step(); loss_sum += float(loss.detach()); batches += 1
            progress.update(bi, loss_sum / batches)
        metrics, _, _ = score(head, val_cache, config, output, device, f'Validation {epoch + 1}')
        key = float(np.mean([metrics[k]['macro_f1'] for k in KINDS]))
        history.append(dict(stage='readout_only', epoch=epoch + 1, train_loss=loss_sum / batches,
                            validation=metrics, frozen_source_updated=False))
        if best is None or key > best:
            best = key; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        json_write(output / 'history.json', history)
        print(f'Readout {epoch + 1}: Validation mean macro-F1={key:.4f}; STEP 1/2 remain frozen', flush=True)
        if stale >= config['patience'] or time.perf_counter() - started >= config.get('head_max_seconds', float('inf')):
            break
    source.verify_frozen()
    # Recheck source files, code, profiles, immutable split and source Parquets.
    verified = data_for(config, Path(config['_gevis_root']))
    if identity_for(verified, config, inspect_bundle(config, config['_gevis_root'])) != identity:
        raise ValueError('Data/config/code/frozen source changed during training; checkpoint refused')
    for name, path in config.get('selection', {}).get('files', {}).items():
        if sha(path) != config['selection']['sha256'][name]:
            raise ValueError('Selection profile changed during training')
    payload = dict(schema=SCHEMA, identity=identity, state=best_state, sensors=data.requested,
        train_stats=stats, trained_steps=sorted({w['step'] for w in windows}),
        train_runs=sorted({w['run_id'] for w in windows}), validation_runs=sorted({w['run_id'] for w in validation}),
        feature_dim=source.feature_dim, frozen_source_hash=source.initial_hash,
        source_weights_updated=False, head_fit_seconds=time.perf_counter() - started,
        input_adaptation=source.adaptation)
    with (output / 'model.pt').open('xb') as stream:
        torch.save(payload, stream)
    json_write(output / 'training_summary.json', {k: v for k, v in payload.items() if k not in ('state', 'train_stats')})
    pd.DataFrame(data.audit).to_csv(output / 'window_audit.csv', index=False)
    print('Saved readout-only checkpoint:', output / 'model.pt', flush=True)


def evaluate(data, source, config, output, identity, device):
    payload = torch.load(output / 'model.pt', map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload['identity'] != identity:
        raise ValueError('Frozen readout checkpoint identity mismatch')
    if payload['frozen_source_hash'] != source.initial_hash:
        raise ValueError('Source tensor identity differs from the frozen training bundle')
    data.stats = {int(s): v for s, v in payload['train_stats'].items()}
    windows = data.windows('test', config['evaluation_windows_per_run'])
    if {w['run_id'] for w in windows} & set(payload['train_runs'] + payload['validation_runs']):
        raise ValueError('Test run overlaps training or validation')
    if not {w['step'] for w in windows} <= set(payload['trained_steps']):
        raise ValueError('Test contains untrained target Main Steps')
    cache = cached_features(source, windows, data, config, 'test', output, identity, device)
    source.to_device(torch.device('cpu'))
    head = ContextReadout(payload['feature_dim'], config['head_width']).to(device)
    head.load_state_dict(payload['state'], strict=True)
    metrics, pred, timings = score(head, cache, config, output, device, 'Test readout')
    meta = np.asarray(cache['meta']); rows = []
    for i, kind in enumerate(KINDS):
        rows.append(pd.DataFrame(dict(run_id=meta[:, 0], step=meta[:, 1],
            center_sensor=[data.requested[c] for c in meta[:, 2]], window_index=meta[:, 3],
            answer_type=kind, label=cache['y'][:, i], prediction=pred[:, i])))
    details = pd.concat(rows, ignore_index=True)
    details.to_csv(output / 'test_predictions.csv', index=False)
    for col, name in [('step', 'test_by_step.csv'), ('center_sensor', 'test_by_sensor.csv')]:
        groups = []
        for (value, kind), g in details.groupby([col, 'answer_type']):
            i = KINDS.index(kind)
            # Use the same pooled confusion/active-class F1 definition.
            truth = np.zeros((len(g), 3), dtype=int); predictions = truth.copy()
            truth[:, i] = g.label; predictions[:, i] = g.prediction
            result = classification_metrics(truth, predictions)[kind]
            groups.append({col: value, 'answer_type': kind, 'questions': len(g),
                           'accuracy': result['accuracy'], 'macro_f1': result['macro_f1']})
        pd.DataFrame(groups).to_csv(output / name, index=False)
    report = dict(task='held-out paired synthetic changes; not verified equipment-fault accuracy',
        metrics=metrics, source_weights_updated=False, frozen_source=source.identity,
        input_adaptation=source.adaptation, checkpoint_sha256=sha(output / 'model.pt'),
        latency_scope='cached readout head ONLY; excludes STEP 1/2, source feature extraction and I/O; not full inference latency',
        cached_head_batch_ms_mean=float(np.mean(timings)))
    source.verify_frozen()
    json_write(output / 'test_metrics.json', report)
    pd.DataFrame(data.audit).to_csv(output / 'test_window_audit.csv', index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    return report


def run(config, gevis_root, *, action='train-evaluate', device='auto', output=None):
    config = resolve_config(config, gevis_root)
    if config['mode'] != 'tips_frozen_step12':
        raise ValueError('This entry always freezes the six STEP 1 encoders and complete STEP 2')
    if action not in ('prepare', 'train', 'evaluate', 'train-evaluate'):
        raise ValueError('Unknown action')
    for key in ('window_length', 'stride', 'feature_batch_size', 'batch_size', 'head_width',
                'answer_epochs', 'patience', 'train_windows_per_run', 'evaluation_windows_per_run'):
        if config[key] < 1:
            raise ValueError('Positive budget required: ' + key)
    if config['learning_rate'] <= 0 or config['cases_per_window'] < 24:
        raise ValueError('Positive learning rate and at least 24 cases required')
    if (config.get('trace_cache_mb', 256) < 0 or config.get('feature_cache_limit_mb', 4096) <= 0
        or config.get('head_max_seconds', 300) <= 0 or config.get('log_every_batches', 50) < 1
        or config.get('log_every_seconds', 10) <= 0):
        raise ValueError('Nonnegative trace cache and positive feature/time/log budgets required')
    forbidden = {'encoder_epochs', 'encoder_layers', 'fusion_width', 'text_encoder', 'patch_length'} & config.keys()
    if forbidden:
        raise ValueError('Source architecture/text/patches come only from frozen checkpoints: ' + str(sorted(forbidden)))
    config['_gevis_root'] = str(Path(gevis_root).resolve())
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    print('Loading and verifying ALL SIX frozen STEP 1 encoders + complete STEP 2; no source training.', flush=True)
    source = FrozenSource(config, gevis_root, config.get('selected_sensors') or
        (Path(gevis_root) / config['columns_file']).read_text(encoding='utf-8-sig').splitlines())
    data = data_for(config, gevis_root)
    identity = identity_for(data, config, source.identity)
    output = Path(output) if output else Path(gevis_root) / 'artifacts/tips_peald_frozen' / digest(identity)[:20]
    print(dict(mode=config['mode'], source_steps=source.encoder_names,
        active_source_encoder=source.source_step, source_patch_length=source.patch_length,
        selected_target_steps=config['main_steps'], sensors=len(data.requested),
        source_trainable_parameters=sum(p.numel() for p in source.parameters() if p.requires_grad),
        adaptation=source.adaptation, output=str(output)), flush=True)
    if action == 'prepare':
        data.fit(); windows = data.windows('train', config['train_windows_per_run'])
        print(dict(train_windows=len(windows), eligible_main_steps=sorted({w['step'] for w in windows}),
            train_examples=len(cases_for(windows, data, config, 'train'))), flush=True)
        source.verify_frozen()
        return dict(frozen_source=source.identity, source_trainable_parameters=0, output=str(output))
    selected = choose_device(device); output.mkdir(parents=True, exist_ok=True)
    json_write(output / 'frozen_source_summary.json', dict(identity=source.identity,
        tensor_sha256=source.initial_hash, adaptation=source.adaptation, source_trainable_parameters=0))
    if action in ('train', 'train-evaluate'):
        train(data, source, config, output, identity, selected)
    if action in ('evaluate', 'train-evaluate'):
        return evaluate(data, source, config, output, identity, selected)
