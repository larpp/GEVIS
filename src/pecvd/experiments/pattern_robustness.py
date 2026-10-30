"""External fixed-pattern test generator. Never fits, selects or saves models."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from . import fast_experiments as fast, fusion_answers as fa
from .injection import Injection, inject


def protocol_digest(plan):
    validate(plan)
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate(plan):
    if plan.get('schema') != 'gevis.fixed_pattern_test.v1' or not plan.get('patterns'):
        raise ValueError('Nonempty fixed-pattern protocol required')
    if plan.get('origin') not in ('illustrative', 'train_review', 'validation_review', 'inspected_test'):
        raise ValueError('Explicit pattern provenance required')
    if plan.get('training_exposure') != 'not_audited':
        raise ValueError('This runner cannot certify unseen-training conditions')
    if not isinstance(plan.get('seed'), int) or len(plan['patterns']) > 32:
        raise ValueError('Integer seed and bounded pattern count required')
    ids = []
    for rule in plan['patterns']:
        ids.append(rule['id'])
        if not isinstance(rule['id'], str) or not rule['id'] or not 1 <= len(rule['components']) <= 4:
            raise ValueError('Named bounded patterns required')
        for op in rule['components']:
            if op['kind'] not in (*fa.AMPLITUDE_TARGET, 'delay'):
                raise ValueError('Unsupported fixed-pattern operation')
            if not isinstance(op['sensor'], int) or isinstance(op['sensor'], bool) or op['sensor'] < 0:
                raise ValueError('Explicit nonnegative sensor index required')
            if not np.isfinite(op['strength']) or op['strength'] <= 0:
                raise ValueError('Finite positive pattern strength required')
            if not 0 <= op['start'] < op['end'] <= 1 or op.get('direction', 1) not in (-1, 1):
                raise ValueError('Valid fractional span and direction required')
            if op['kind'] in ('delay', 'variance') and op.get('direction', 1) != 1:
                raise ValueError('Delay/noise reversal is not a supported negative operation')
            if op['kind'] == 'delay' and (int(op['strength']) != op['strength'] or op['start'] != 0 or op['end'] != 1):
                raise ValueError('Delay uses an integer whole-window native-sample lag')
    if len(set(ids)) != len(ids) or 'control' in ids:
        raise ValueError('Unique non-control pattern IDs required')


def _level(op):
    values = (1., 2., 3., 4.) if op['kind'] == 'delay' else (.5, 1., 2., 4.)
    return next((i + 1 for i, value in enumerate(values) if np.isclose(op['strength'], value, rtol=0, atol=1e-9)), None)


def cases(runtime, windows, plan):
    stamp = protocol_digest(plan)
    result = []
    for window in windows:
        if window.get('split') != 'test':
            raise ValueError('Fixed-pattern evaluation requires explicit Test windows')
        step = window['step']; names = runtime.encoders[step][1]['sensor_columns']
        base = np.asarray(window['current'], dtype=float)
        if base.ndim != 2 or base.shape != (runtime.encoders[step][1]['window_length'], len(names)) or len(base) < 4 or not np.isfinite(base).all():
            raise ValueError('Finite native standardized Test window required')
        used = sorted({op['sensor'] for rule in plan['patterns'] for op in rule['components']})
        if max(used) >= len(names):
            raise ValueError('Pattern sensor index outside this Step; define a compatible protocol')
        extra = next((i for i in range(len(names)) if i not in used), None)
        centers = used + ([extra] if extra is not None else [])
        variants = [('control', base.copy(), [])]
        for rule in plan['patterns']:
            current = base.copy(); applied = []
            for index, op in enumerate(rule['components']):
                before = current.copy()
                if op['kind'] == 'delay':
                    current, _ = fa.delay_trace(current, op['sensor'], int(op['strength']))
                else:
                    start = int(len(base) * op['start'])
                    end = min(len(base), max(start + 1, int(np.ceil(len(base) * op['end']))))
                    current, _ = inject(current, Injection(op['kind'], op['sensor'], start, end,
                        op['strength'], plan['seed'] + index), np.ones(len(names)))
                    if op.get('direction', 1) < 0:
                        current = 2 * before - current
                if (current[:, op['sensor']] != before[:, op['sensor']]).any():
                    applied.append(op)
            variants.append((rule['id'], current, applied))
        for name, current, applied in variants:
            mask = current != base; targets = {}
            for sid in centers:
                changed = bool(mask[:, sid].any())
                label = dict(BOOLEAN=int(changed))
                relevant = [op for op in applied if op['sensor'] == sid]
                if not changed:
                    label.update(CHOICE=0, ORDINAL=0)
                elif len(relevant) == 1:
                    label['CHOICE'] = fa.TYPE_TARGET[relevant[0]['kind']]
                    level = _level(relevant[0])
                    if level is not None:
                        label['ORDINAL'] = level
                targets[names[sid]] = label
            result.append(dict(step=step, current=current, reference=base.copy(), centers=[names[i] for i in centers],
                targets=targets, changed_mask=mask, kind='equal' if name == 'control' else 'fixed_pattern',
                challenge=name, strength=0., strength_unit='see_locked_protocol', position=window.get('position', 0.),
                run_id=str(window['run_id']), pair_id=f'{step}/{window["pair_id"]}/{stamp}/{name}', split='test'))
    if not result:
        raise ValueError('No fixed-pattern evaluation inputs')
    return result


def _fingerprint(runtime):
    h = hashlib.sha256()
    models = [runtime.module.GEVIS_MODEL, fa.require_head(runtime.module)]
    models += [encoder for encoder, meta in runtime.encoders.values()]
    models += list(getattr(runtime, 'step_heads', {}).values())
    seen = set()
    for model in models:
        if id(model) in seen:
            continue
        seen.add(id(model))
        for name, value in model.state_dict().items():
            h.update(name.encode()); h.update(str((value.shape, value.dtype)).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def source_windows(runtime, *, steps=None, limit=2, seed=42):
    if not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ValueError('Bounded evaluation window limit 1..8 required')
    if hasattr(runtime, 'prepared'):
        from .peald_training import _select
        return [dict(step=step, current=w['current'], run_id=w['run_id'], pair_id=w['pair_id'],
                     position=w['position'], split='test')
                for step in fast._steps(runtime, steps)
                for w in _select(runtime.prepared.windows[step]['test'], limit, seed)]
    config = fa.FusionConfig(max_windows_per_step=limit, seed=seed)
    return [dict(step=step, current=w['current'], run_id=str(w['run_id']), pair_id=str(w.get('cache_index', i)),
                 position=float(w['position']) / 9, split='test')
            for step, i, w in fast._source_rows(runtime, 'test', steps, config)]


def compare(models, windows, plan, *, locked_digest):
    """All supplied candidates are reported; no best-model selection or export."""
    if not models or protocol_digest(plan) != locked_digest:
        raise ValueError('Freeze the exact protocol before evaluating any model')
    plan = copy.deepcopy(plan); windows = list(windows)
    baseline = next(iter(models.values()))
    inputs = cases(baseline, windows, plan)
    before = {name: _fingerprint(runtime) for name, runtime in models.items()}
    reports = {}; summaries = []
    for name, runtime in models.items():
        fast.require_runtime(runtime)
        forbidden = (getattr(runtime.module, 'FUSION_ANSWER_TRAINING', None) or {})
        seen = {str(r) for key in ('train_runs', 'validation_runs') for r in forbidden.get(key, [])}
        if seen & {str(w['run_id']) for w in windows}:
            raise ValueError('Evaluation run overlaps candidate TRAIN/Validation')
        rows = []
        for step in dict.fromkeys(c['step'] for c in inputs):
            fast._steps(runtime, [step])
            a = baseline.encoders[step][1]; b = runtime.encoders[step][1]
            if a['sensor_columns'] != b['sensor_columns'] or a['window_length'] != b['window_length'] or any(
                    not np.array_equal(np.asarray(a.get(k)), np.asarray(b.get(k))) for k in ('mean', 'std')):
                raise ValueError('Candidates must share the exact prepared input contract')
            scoped = runtime
            if hasattr(runtime, 'step_heads'):
                from .peald_step_heads import for_step
                scoped = for_step(runtime, step)
            report = fast.score_cases(scoped, [c for c in inputs if c['step'] == step])
            rows.extend(report.details.to_dict('records'))
            rows.extend(report.diagnostics['unscored_answers'].to_dict('records'))
        scored = [row for row in rows if row['scored']]
        report = fast._report('Locked fixed-pattern model answers', scored, all_answers=rows,
            metadata=dict(protocol_sha256=locked_digest, pattern_origin=plan['origin'],
                training_exposure='not_audited', weights_sha256=before[name], model_selection=False,
                run_overlap_check='checked' if 'train_runs' in forbidden and 'validation_runs' in forbidden else 'unavailable'))
        reports[name] = report
        for pattern, group in report.details.groupby('challenge', sort=False):
            sub = fast._report(pattern, group.to_dict('records'))
            table = sub.summary.assign(model=name, pattern=pattern)
            table['changed_questions'] = table.answer_type.map(group.loc[group.label.ne(0)].groupby('answer_type').size()).fillna(0).astype(int)
            table['unscored_questions'] = [sum(row['challenge'] == pattern and row['answer_type'] == kind and not row['scored']
                                               for row in rows) for kind in table.answer_type]
            summaries.append(table)
    if any(_fingerprint(runtime) != before[name] for name, runtime in models.items()):
        raise RuntimeError('Model weights changed during frozen evaluation; results rejected')
    return fast.FastSuite(pd.concat(summaries, ignore_index=True), reports,
        pd.concat([r.details.assign(model=name) for name, r in reports.items()], ignore_index=True),
        metadata=dict(protocol_sha256=locked_digest, origin=plan['origin'], training_exposure='not_audited',
            weights_preserved=True, test_used_for_selection=False, actual_fault_truth=False, automatic_export=False))
