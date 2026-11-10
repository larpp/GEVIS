"""User-run Teacher loading and source-bound data. No data/model I/O on import."""
from copy import copy
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from pecvd import gevis_training
from pecvd.experiments import ald_head_experiments as heads, ald_preprocessing as preprocessing
from pecvd.experiments import decision_training as d
from .runtime import validate_original_head


def load_teacher(root, *, domain='PEALD', checkpoint=None, format='domain',
                 allow_legacy_cvd=False, allow_legacy_ald=True, choose=input):
    root = Path(root).resolve()
    if domain not in ('PECVD', 'PEALD') or format not in ('domain', 'detailed'):
        raise ValueError('Select saved PECVD/PEALD domain and domain/detailed package format')
    if format == 'detailed':
        runtime = d.load(root, domain=domain, checkpoint=checkpoint).runtime
        runtime.target_sensors = {s: list(m['sensor_columns']) for s, (_, m) in runtime.encoders.items()}
    else:
        if checkpoint is None:
            candidates = sorted((root / 'artifacts/gevis_heads' / domain.lower()).glob('*/answer_head.pt'))
            matches = []
            for path in candidates:
                payload = torch.load(path, map_location='cpu', weights_only=False)
                identity = payload.get('identity', {})
                if (payload.get('schema') == heads.SCHEMA and identity.get('domain') == domain
                        and identity.get('choice', {}).get('mode') == 'softmax'):
                    matches.append(path)
            if not matches:
                raise FileNotFoundError('No original-five Softmax Teacher. Check WORKSPACE_PATH/DOMAIN/CHECKPOINT; no fitting fallback.')
            if len(matches) == 1:
                checkpoint = matches[0]
            else:
                print(pd.DataFrame({'checkpoint': [str(p.relative_to(root)) for p in matches]}).to_string())
                if choose is None:
                    raise ValueError('Multiple original-five Teachers; set CHECKPOINT explicitly')
                try:
                    index = int(choose('사용할 원본 Head 번호: '))
                except (ValueError, TypeError):
                    raise ValueError('Integer Teacher index required') from None
                if not 0 <= index < len(matches):
                    raise ValueError('Teacher index outside table')
                checkpoint = matches[index]
        runtime = heads.load_head(root, checkpoint, allow_legacy_cvd=allow_legacy_cvd,
                                  allow_legacy_ald=allow_legacy_ald)
    if runtime.domain != domain:
        raise ValueError('Teacher domain differs')
    validate_original_head(runtime.module.FUSION_ANSWER_HEAD)
    runtime.distilation_input_policy = format
    return runtime


def attach_prepared(teacher, prepared):
    """Refuse changed normalization, sensor order, native cadence or Run split.

    Keep the packaged TRAIN normalizer/references. A current preparation with
    incompatible scalars is not silently adopted or used to refit the Teacher.
    """
    if teacher.domain == 'PEALD':
        prepared = preprocessing.canonicalize_run_lineages(prepared)
    saved = teacher.prepared
    for split in ('train', 'validation', 'test'):
        if set(map(str, prepared.split_runs[split])) != set(map(str, saved.split_runs[split])):
            raise ValueError('Prepared Run partitions differ from Teacher: ' + split)
    parts = prepared.split_runs
    if any(set(parts[a]) & set(parts[b]) for a, b in
           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Run partitions overlap')
    for step, (_, meta) in teacher.encoders.items():
        if step not in prepared.windows or step not in prepared.metadata:
            raise ValueError('Prepared native step missing from Teacher support')
        other = prepared.metadata[step]
        if list(meta['sensor_columns']) != list(other['sensor_columns']):
            raise ValueError('Teacher sensor order differs')
        for name in ('mean', 'std'):
            if not np.array_equal(np.asarray(meta[name]).reshape(-1), np.asarray(other[name]).reshape(-1)):
                raise ValueError('Teacher TRAIN normalization differs: ' + name)
        for name in ('window_length', 'cadence_seconds'):
            if meta[name] != other[name]:
                raise ValueError('Teacher native input contract differs: ' + name)
    for field in ('source_files', 'split_runs'):
        if field in saved.identity and d._hash(saved.identity[field]) != d._hash(prepared.identity.get(field)):
            raise ValueError('Prepared source identity differs: ' + field)
    scoped = copy(prepared)
    scoped.windows = {s: prepared.windows[s] for s in teacher.encoders}
    scoped.metadata = {s: meta for s, (_, meta) in teacher.encoders.items()}
    scoped.identity = saved.identity
    teacher.prepared = scoped
    return scoped


def prepare_for_teacher(root, teacher, *, max_memory_mb=4096, progress=print):
    """Explicit user action for KD data; not needed for merged model inference."""
    if teacher.domain == 'PEALD':
        identity = teacher.prepared.identity
        if 'preprocessing' not in identity or 'native_config' not in identity:
            raise ValueError('Teacher preparation policy unavailable; pass its original prepared object to attach_prepared')
        source_steps = set(teacher.source_steps.values())
        if len(source_steps) != 1:
            raise ValueError('A single source patch contract is required for ALD control48')
        prepared = gevis_training.prepare_ald(root, teacher.source_runtime, encoder=next(iter(source_steps)),
            preprocessing=preprocessing.PreprocessingConfig(**identity['preprocessing']),
            native=preprocessing.NativeConfig(**identity['native_config']),
            max_memory_mb=max_memory_mb, progress=progress)
    else:
        prepared = gevis_training.prepare_cvd(root, teacher.source_runtime)
    return attach_prepared(teacher, prepared)


def sample_cases(cases, limit, seed):
    """Bound each step, reserving an available normal/family/stage combination."""
    rng = np.random.default_rng(seed)
    selected = []
    for step in sorted({c['step'] for c in cases}):
        pool = [c for c in cases if c['step'] == step]
        order = list(map(int, rng.permutation(len(pool))))
        groups = {}
        for index in order:
            case = pool[index]
            levels = tuple(sorted(set(map(int, np.asarray(case['targets'])[:, 2]))))
            groups.setdefault((case['kind'], levels), index)
        reserve = list(groups.values())
        if len(reserve) > limit:
            raise ValueError('Increase cases_per_step to retain available family/stage coverage')
        chosen = reserve + [i for i in order if i not in set(reserve)][:max(0, limit - len(reserve))]
        selected.extend(pool[i] for i in chosen)
    return selected


def cases_for(teacher, split, config):
    if not hasattr(teacher.prepared, 'windows'):
        raise ValueError('Prepare/attach source-bound KD windows first; original Head is not retrained')
    if split == 'train':
        cases, coverage, exclusions = d.training_cases(teacher, teacher.references, teacher.config)
    elif split in ('validation', 'test'):
        cases, coverage = d.controlled_cases(teacher, split, teacher.config, teacher.references)
        exclusions = []
    else:
        raise ValueError('Explicit train/validation/test split required')
    if teacher.distilation_input_policy == 'domain':
        cases, coverage = heads._continuous_interventions(cases, coverage, teacher.target_sensors)
    cases, audit = d.supported_cases(teacher, cases, teacher.references, training=split == 'train')
    if not cases:
        raise ValueError('No supported ' + split + ' KD cases; inspect native reference coverage')
    cases = sample_cases(cases, getattr(config, split + '_cases_per_step'), config.seed + {'train': 0, 'validation': 101, 'test': 202}[split])
    return cases, dict(generation=coverage, support=audit, exclusions=exclusions,
        selected_cases=len(cases), lineages=sorted({c['lineage_id'] for c in cases}))
