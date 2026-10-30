"""Explicit frozen source Fast transfer; no target fitting or rule fallback.

This experiment intentionally evaluates unseen target phases. It does not mark
them as trained source Steps or relax the production predict_arrays guard.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd

from . import fast_experiments as fast, fusion_answers as fa
from .frozen_transfer import external_pairs, NativeTransferUnsupported
from .process_inputs import step_description
from .reporting import ExperimentReport


def bind(source_runtime, config, frame, *, source_step, pairs_per_step=2, seed=42):
    fast.require_runtime(source_runtime)
    if not 1 <= pairs_per_step <= 8:
        raise ValueError('PEALD pair limit must be 1..8 per Step')
    # The source encoder is chosen before inspecting target results.
    fast._steps(source_runtime, [source_step])
    source = source_runtime.module
    encoder, meta = source_runtime.encoders[source_step]
    if not getattr(encoder, 'supports_native_sensor_transfer', False):
        raise NativeTransferUnsupported('This source encoder cannot natively represent unseen PEALD sensors; no ID substitution or fallback')
    names = list(config.sensor_columns)
    if not names or len(names) != len(set(names)):
        raise ValueError('Explicit unique target measurement channels required')
    paired, support = external_pairs(frame, config, int(meta['window_length']),
                                     limit=pairs_per_step, seed=seed)
    module = SimpleNamespace(**source.__dict__)
    module.STEP_ENCODERS = {}
    for step, (_, mean, scale) in paired.items():
        module.STEP_ENCODERS[step] = (encoder, dict(meta, sensor_columns=names,
            mean=mean, std=scale, step_descriptor=step_description(config, step),
            sensor_descriptors=dict(getattr(config, 'sensor_descriptors', {}) or {})))
    runtime = SimpleNamespace(module=module, encoders=module.STEP_ENCODERS,
        external_pairs=paired, transfer_metadata=dict(source_encoder_step=source_step,
            source_trained_steps=list(getattr(source, 'FUSION_ANSWER_TRAINING', {}).get('trained_steps', [])),
            target_steps_are_source_trained=False, weights_updated=False,
            native_sensor_transfer=True, input_calibration='PEALD reference runs only',
            window_alignment='same occurrence/cohort/native row offset; no resampling',
            target_truth='known injected changes only; original faults unlabelled'))
    return runtime, support


def _windows(runtime, steps):
    selected = list(runtime.encoders) if steps is None else list(steps)
    if not selected or len(selected) != len(set(selected)) or any(s not in runtime.encoders for s in selected):
        raise ValueError('Choose supported PEALD Steps from SUPPORT; none are silently skipped')
    for step in selected:
        paired, mean, scale = runtime.external_pairs[step]
        names = list(runtime.encoders[step][1]['sensor_columns'])
        for i, (current, reference) in enumerate(paired['test']):
            if current.attrs.get('reference_split') != 'train':
                raise ValueError('PEALD comparison requires reference-run input calibration')
            if current.attrs['evaluation_run_id'] in current.attrs['reference_run_ids']:
                raise ValueError('Test run overlaps reference runs')
            yield step, i, names, (current[names].to_numpy(float)-mean)/scale, (
                reference[names].to_numpy(float)-mean)/scale, current.attrs


def controlled(runtime, *, steps=None, seed=42):
    """Use target Test traces as bases; operator truth never enters forward."""
    cases = []
    for step, i, names, base, _, attrs in _windows(runtime, steps):
        for variant in fa.paired_variants(base, i % len(names), i, seed+600003):
            sid = variant.get('sensor', i % len(names))
            centers = [names[sid]] + ([names[(sid+1) % len(names)]] if len(names)>1 else [])
            targets = {}
            for center in centers:
                changed = bool(variant['mask'][:, names.index(center)].any())
                targets[center] = dict(BOOLEAN=int(changed),
                    CHOICE=fa.TYPE_TARGET[variant['kind']] if changed else 0,
                    ORDINAL=variant['level'] if changed else 0)
            cases.append(dict(step=step, current=variant['observed'], reference=base.copy(),
                centers=centers, targets=targets, changed_mask=variant['mask'], kind=variant['kind'],
                strength=variant['strength'], lag_samples=variant.get('lag_samples', 0),
                strength_unit='native_samples' if variant['kind']=='delay' else 'target_reference_std',
                challenge='peald_frozen_transfer', position=attrs['position_bin']/9,
                run_id=attrs['evaluation_run_id'], pair_id=attrs['evaluation_pair_id']+'/'+variant['tag']))
    report = fast.score_cases(runtime, cases)
    report.metadata.update(runtime.transfer_metadata)
    report.metadata['experiment'] = 'PEALD frozen transfer synthetic comparison'
    return report


def original(runtime, *, steps=None, centers=None, center_batch_size=8):
    if not 1 <= center_batch_size <= 32:
        raise ValueError('Center batch size must be 1..32')
    fast.require_runtime(runtime)
    rows = []
    for step, i, names, current, reference, attrs in _windows(runtime, steps):
        selected = names if centers is None else list(centers)
        if not selected or len(selected)!=len(set(selected)) or any(c not in names for c in selected):
            raise ValueError('Select unique configured PEALD center sensors')
        for start in range(0, len(selected), center_batch_size):
            features, tasks = fa.fusion_features(runtime.module, current, reference, step,
                selected[start:start+center_batch_size], attrs['position_bin']/9)
            for answer in fast._decode(fa.require_head(runtime.module), features, tasks, runtime.module.DEVICE):
                rows.append(dict(answer, step=step, pair_index=i, pair_id=attrs['evaluation_pair_id'],
                                 run_id=attrs['evaluation_run_id']))
    details = pd.DataFrame(rows)
    return ExperimentReport('PEALD original predictions without independent fault truth',
        pd.DataFrame([dict(questions=len(details), accuracy_available=False,
            metric_target='unlabelled PEALD original cross-run answers')]), details=details,
        diagnostics={'answer_consistency': fast._consistency(rows)},
        metadata=dict(runtime.transfer_metadata, decision_system=fa.SYSTEM, model_evaluated=True, rule_fallback=False,
                      original_cross_run_fault_accuracy_available=False))
