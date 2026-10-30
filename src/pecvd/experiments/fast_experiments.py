"""Offline experiments for the deployed fixed-question Fusion answer model.

No numerical detector, automatic export or model fallback. Only the explicitly
requested feature-control experiment fits temporary TRAIN/Validation readouts.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import time

import numpy as np
import pandas as pd
import torch

from . import fusion_answers as fa
from .reporting import ExperimentReport


CHALLENGES = ('standard', 'amplitude', 'location_duration', 'negative', 'compound', 'delay')
FEATURE_VIEWS = ('full_retrained', 'fusion_query_only', 'selected_numeric_only',
                 'attended_selected_only', 'position_only')


@dataclass
class FastSuite:
    summary: pd.DataFrame
    reports: dict = field(default_factory=dict)
    details: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def require_runtime(runtime):
    if getattr(runtime.module, 'DECISION_SYSTEM', None) != fa.SYSTEM:
        raise RuntimeError('Fast experiments require fusion_answers, not a comparator runtime')
    fa.require_head(runtime.module)
    return runtime


def load_runtime(root):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from .evaluation import load_experiment_runtime
    return require_runtime(load_experiment_runtime(root))


def _steps(runtime, steps):
    selected = list(runtime.encoders) if steps is None else list(steps)
    if not selected or any(step not in runtime.encoders for step in selected):
        raise ValueError('Select at least one trained Step')
    training = getattr(runtime.module, 'FUSION_ANSWER_TRAINING', None)
    if training and any(step not in training.get('trained_steps', []) for step in selected):
        raise ValueError('Selected Step has no trained answer-head support')
    return selected


def _source_rows(runtime, split, steps, config):
    from .discovery_calibration import cached_comparisons
    if split not in ('train', 'validation', 'test') or config.max_windows_per_step < 1:
        raise ValueError('Known split and positive window limit required')
    for step in _steps(runtime, steps):
        for i, row in enumerate(cached_comparisons(runtime.module.ENCODER_ROOT, step,
                split=split, config=fa.sampling_config(config))):
            if row.get('split', split) != split:
                raise ValueError('Cache row belongs to another split')
            yield step, i, row


def controlled_cases(runtime, *, challenge='standard', steps=None, config=fa.FusionConfig()):
    """TEST self-reference interventions; labels are used only after model forward."""
    require_runtime(runtime)
    if challenge not in CHALLENGES:
        raise ValueError('Unknown controlled challenge')
    cases = []
    for step, i, row in _source_rows(runtime, 'test', steps, config):
        from .injection import Injection, inject
        base = np.asarray(row['current'], dtype=float)
        names = list(runtime.encoders[step][1]['sensor_columns'])
        if len(base) < 4:
            raise ValueError('Controlled challenges need at least four samples')
        sid = i % len(names); level = 1 + i % 4
        if challenge in ('delay','standard'):
            for v in fa.paired_variants(base,sid,i,config.seed+200003):
                if v['kind'] not in ('equal','delay'):continue
                if challenge=='standard' and v['kind']=='equal':continue
                target=v.get('sensor',sid)
                centers=[names[target]]+([names[(target+1)%len(names)]] if len(names)>1 else [])
                targets={}
                for center in centers:
                    changed=bool(v['mask'][:,names.index(center)].any())
                    targets[center]=dict(BOOLEAN=int(changed),CHOICE=fa.TYPE_TARGET['delay'] if changed else 0,
                        ORDINAL=v['level'] if changed else 0)
                cases.append(dict(step=step,current=v['observed'],reference=base.copy(),centers=centers,
                    targets=targets,changed_mask=v['mask'],kind=v['kind'],strength=v['strength'],
                    lag_samples=v.get('lag_samples',0),strength_unit='native_samples',challenge=challenge,
                    position=float(row['position'])/9,run_id=str(row['run_id']),
                    pair_id=f'{step}/{i}/{v["tag"]}',split='test'))
            if challenge=='delay':continue
        centers = [names[sid]] + ([names[(sid + 1) % len(names)]] if len(names) > 1 else [])
        families = ('BOOLEAN',) if challenge == 'compound' else (
            ('BOOLEAN', 'CHOICE') if challenge == 'amplitude' else fa.KINDS)
        variants = [('equal', base.copy(), np.zeros_like(base, dtype=bool), 0.)]
        for k, kind in enumerate(fa.AMPLITUDE_TARGET):
            if challenge == 'negative' and kind == 'variance':
                # Noise sign reversal is not a decrease in variance.
                continue
            strength = ((.25, .75, 1.5, 3., 6.)[(i + k) % 5] if challenge == 'amplitude'
                        else (.5, 1., 2., 4.)[level - 1])
            start, end = len(base) // 4, max(len(base) // 4 + 2, 3 * len(base) // 4)
            if challenge == 'location_duration':
                width = max(2, len(base) // (2, 4, 8)[i % 3])
                rng = np.random.default_rng(config.seed + i * 23 + k)
                start = int(rng.integers(0, len(base) - width + 1)); end = start + width
            observed, truth = inject(base, Injection(kind, sid, start, end, strength,
                config.seed + i * 11 + k), np.ones(len(names)))
            if challenge == 'negative':
                observed = 2 * base - observed
            if challenge == 'compound':
                partner = 'level_shift' if kind == 'drift' else 'drift'
                observed, _ = inject(observed, Injection(partner, sid, start, end,
                    strength, config.seed + i * 11 + 100), np.ones(len(names)))
            mask = observed != base
            if mask.any():
                variants.append((f'compound_{kind}' if challenge == 'compound' else kind,
                                 observed, mask, strength))
        for kind, observed, mask, strength in variants:
            targets = {}
            for center in centers:
                changed = bool(mask[:, names.index(center)].any())
                targets[center] = dict(BOOLEAN=int(changed))
                if 'CHOICE' in families:
                    targets[center]['CHOICE'] = fa.TYPE_TARGET[kind] if changed else 0
                if 'ORDINAL' in families:
                    targets[center]['ORDINAL'] = level if changed else 0
            cases.append(dict(step=step, current=observed, reference=base.copy(), centers=centers,
                targets=targets, changed_mask=mask, kind=kind, strength=strength,
                challenge=challenge, position=float(row['position']) / 9,
                run_id=str(row['run_id']), pair_id=f'{step}/{i}/{kind}', split='test'))
    if not cases:
        raise ValueError('No controlled Test cases')
    return cases


@contextmanager
def _question_probe(model, mode):
    if mode not in ('original', 'question_withheld', 'mismatched_question'):
        raise ValueError('Unknown question probe')
    handle = None
    if mode != 'original':
        def change_text(module, args):
            # Only question text changes. Answer types/options remain fixed.
            changed = list(args)
            changed[1] = (['Question withheld.'] * len(args[1]) if mode == 'question_withheld' else
                [fa.QUESTIONS[fa.KINDS[(fa.KINDS.index(kind) + 1) % len(fa.KINDS)]] for kind in args[2]])
            return tuple(changed)
        handle = model.register_forward_pre_hook(change_text)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def _decode(head, features, tasks, device):
    result = []
    for feature, (center, kind) in zip(features, tasks):
        p = head(feature[None].to(device), kind).softmax(-1)[0].cpu()
        index = int(p.argmax())
        result.append(dict(center_sensor=center, answer_type=kind, prediction=index,
            answer=fa.ANSWER_CRITERIA[kind][index],
            probabilities=dict(zip(fa.ANSWER_CRITERIA[kind], map(float, p)))))
    return result


def _consistency(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()
    wide = frame.pivot(index=['pair_id', 'center_sensor'], columns='answer_type', values='prediction')
    mismatch = (wide['BOOLEAN'].ne(0) != wide['CHOICE'].ne(0)) | (
        wide['BOOLEAN'].ne(0) != wide['ORDINAL'].ne(0))
    return pd.DataFrame([dict(answer_groups=len(wide), inconsistent_groups=int(mismatch.sum()),
                              inconsistency_rate=float(mismatch.mean()))])


def _report(name, scored, *, all_answers=None, metadata=None):
    details = pd.DataFrame(scored)
    if details.empty:
        raise ValueError('No answers with defined independent synthetic targets')
    details['correct'] = details.prediction.eq(details.label)
    diagnostics = {}
    if all_answers is not None:
        diagnostics['answer_consistency'] = _consistency(all_answers)
        diagnostics['unscored_answers'] = pd.DataFrame([r for r in all_answers if not r['scored']])
    return ExperimentReport(name, pd.DataFrame(), details=details,
        errors=details.loc[~details.correct].copy(), diagnostics=diagnostics,
        metadata=dict(decision_system=fa.SYSTEM, model_evaluated=True, split='test',
            rule_fallback=False, deployed_weights_updated=False, test_used_for_fitting=False,
            original_cross_run_fault_accuracy_available=False, **(metadata or {})))


def score_cases(runtime, cases, *, probe='original'):
    require_runtime(runtime); m = runtime.module
    scored, all_answers = [], []
    with _question_probe(m.GEVIS_MODEL, probe):
        for case in cases:
            features, tasks = fa.fusion_features(m, case['current'], case['reference'],
                case['step'], case['centers'], case['position'])
            for answer in _decode(fa.require_head(m), features, tasks, m.DEVICE):
                center, kind = answer['center_sensor'], answer['answer_type']
                row = dict(answer, step=case['step'], pair_id=case['pair_id'], run_id=case['run_id'],
                    kind=case['kind'], challenge=case['challenge'], strength=case['strength'],
                    lag_samples=case.get('lag_samples',0),strength_unit=case.get('strength_unit','train_std'),
                    injected=case['kind'] != 'equal', split='test',
                    center_is_perturbed=bool(case['changed_mask'][:,
                        list(runtime.encoders[case['step']][1]['sensor_columns']).index(center)].any()),
                    scored=kind in case['targets'][center])
                all_answers.append(row)
                if row['scored']:
                    scored.append(dict(row, label=case['targets'][center][kind]))
    return _report('Fast Fusion controlled answers', scored, all_answers=all_answers,
        metadata=dict(question_probe=probe, target_scope='original fixed task; changed text is an input stress probe'))


def _suite(reports, **metadata):
    summary = pd.concat([r.summary.assign(variant=name) for name, r in reports.items()], ignore_index=True)
    details = pd.concat([r.details.assign(variant=name) for name, r in reports.items()], ignore_index=True)
    return FastSuite(summary, reports, details, metadata=dict(model=fa.SYSTEM,
        deployed_weights_updated=False, test_used_for_selection=False, **metadata))


def run_question_probes(runtime, *, steps=None, config=fa.FusionConfig()):
    cases = controlled_cases(runtime, steps=steps, config=config)
    reports = {probe: score_cases(runtime, cases, probe=probe)
               for probe in ('original', 'question_withheld', 'mismatched_question')}
    suite = _suite(reports, experiment='frozen_question_input_sensitivity')
    baseline = reports['original'].details.set_index(['pair_id', 'center_sensor', 'answer_type'])
    rows = []
    for name, report in reports.items():
        current = report.details.set_index(['pair_id', 'center_sensor', 'answer_type'])
        for kind in fa.KINDS:
            changed = current.xs(kind, level='answer_type').prediction.ne(
                baseline.xs(kind, level='answer_type').prediction)
            rows.append(dict(probe=name, answer_type=kind, cases=len(changed),
                             prediction_change_rate=float(changed.mean())))
    suite.diagnostics['prediction_changes'] = pd.DataFrame(rows)
    suite.metadata['limitation'] = ('Frozen out-of-training input probes, not fair retrained ablations. '
        'Answer-family metadata is held fixed. Fixed task text does not establish free-form language understanding.')
    return suite


def run_generalization(runtime, *, steps=None, config=fa.FusionConfig(), challenges=CHALLENGES):
    if not challenges or len(set(challenges)) != len(challenges) or any(c not in CHALLENGES for c in challenges):
        raise ValueError('Select unique known controlled challenges')
    reports = {name: score_cases(runtime, controlled_cases(runtime, challenge=name, steps=steps, config=config))
               for name in challenges}
    return _suite(reports, experiment='held_out_perturbation_generalization',
        limitation='Amplitude: no ordinal targets; compound: Boolean only. All controls have exact self-reference truth.')


def feature_mask(dimension, view):
    if (dimension - 2) % 4 or dimension <= 2 or view not in FEATURE_VIEWS:
        raise ValueError('Known view and four equal Fusion feature blocks plus position required')
    d = (dimension - 2) // 4
    mask = torch.zeros(dimension)
    mask[-2:] = 1
    blocks = {'full_retrained': (0, 1, 2, 3), 'fusion_query_only': (0, 1),
              'selected_numeric_only': (2,), 'attended_selected_only': (3,), 'position_only': ()}
    for block in blocks[view]:
        mask[block*d:(block+1)*d] = 1
    return mask


def _masked(rows, view):
    return [dict(row, features=row['features'] * feature_mask(row['features'].numel(), view)) for row in rows]


def _score_features(head, examples, device):
    rows = []
    for e in examples:
        answer = _decode(head, e['features'][None], [(e['center_sensor'], e['answer_type'])], device)[0]
        rows.append({k: v for k, v in e.items() if k != 'features'} | answer)
    return _report('Matched feature-control Test answers', rows)


def run_feature_controls(runtime, *, steps=None, config=fa.FusionConfig(), views=FEATURE_VIEWS):
    """Fit equally sized masked heads on identical TRAIN/Validation data only."""
    require_runtime(runtime); m = runtime.module
    if 'full_retrained' not in views:
        raise ValueError('A matched full_retrained control is required')
    if len(set(views)) != len(views) or any(view not in FEATURE_VIEWS for view in views):
        raise ValueError('Unique known feature views required')
    selected = _steps(runtime, steps)
    data = {s: fa.controlled_examples(m, s, config, selected) for s in ('train', 'validation', 'test')}
    ids = {s: {e['run_id'] for e in rows} for s, rows in data.items()}
    if any(ids[a] & ids[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('TRAIN/Validation/Test run overlap; no feature controls fitted')
    reports = {'deployed_reference': _score_features(fa.require_head(m), data['test'], m.DEVICE)}
    fits = []; history = []
    for view in views:
        print(f'Feature control {view}: TRAIN fit / Validation epoch selection / Test scoring', flush=True)
        head, training = fa.fit_readout(_masked(data['train'], view), _masked(data['validation'], view), config, m.DEVICE)
        reports[view] = _score_features(head, _masked(data['test'], view), m.DEVICE)
        fits.append(dict(view=view, parameters=training['trainable_parameters'],
            optimizer_updates=training['optimizer_updates'], validation_loss=training['validation_loss'],
            train_runs=len(ids['train']), validation_runs=len(ids['validation']), test_runs=len(ids['test'])))
        history.extend(dict(view=view, **row) for row in training['history'])
        del head
    suite = _suite(reports, experiment='matched_retrained_feature_controls', temporary_control_heads_trained=True,
        limitation='Deployed scores are reference only. Compare retrained full vs matched masks; position remains in every view.')
    suite.summary = suite.summary.loc[suite.summary.variant.ne('deployed_reference')].copy()
    suite.diagnostics.update(training=pd.DataFrame(fits), history=pd.DataFrame(history),
                             deployed_reference=reports['deployed_reference'].summary)
    baseline = reports['full_retrained'].summary.set_index('answer_type')
    suite.summary['accuracy_delta_vs_retrained_full'] = [
        row.answer_accuracy - baseline.loc[row.answer_type, 'answer_accuracy']
        for row in suite.summary.itertuples()]
    return suite


def original_cases(runtime, *, steps=None, centers=None, config=fa.FusionConfig()):
    require_runtime(runtime); cases = []
    for step, i, row in _source_rows(runtime, 'test', steps, config):
        names = list(runtime.encoders[step][1]['sensor_columns'])
        selected = names if centers is None else [name for name in names if name in centers]
        if not selected:
            raise ValueError('No selected sensors in a requested Step')
        cases.append(dict(step=step, current=np.asarray(row['current']).copy(),
            reference=np.asarray(row['reference']).copy(), centers=selected,
            position=float(row['position']) / 9, run_id=str(row['run_id']),
            cohort=row['cohort'],
            reference_run_ids=list(row.get('reference_run_ids', [])), pair_id=f'{step}/{i}', split='test'))
    if not cases:
        raise ValueError('No original Test comparisons')
    return cases


def run_original_review(runtime, *, steps=None, centers=None, config=fa.FusionConfig(), center_batch_size=8):
    if center_batch_size < 1:
        raise ValueError('Positive center batch size required')
    cases = original_cases(runtime, steps=steps, centers=centers, config=config)
    m = runtime.module; rows = []
    for i, case in enumerate(cases):
        for start in range(0, len(case['centers']), center_batch_size):
            features, tasks = fa.fusion_features(m, case['current'], case['reference'], case['step'],
                case['centers'][start:start + center_batch_size], case['position'])
            rows.extend(dict(answer, step=case['step'], pair_id=case['pair_id'], case_index=i,
                run_id=case['run_id'], reference_run_ids=case['reference_run_ids'])
                for answer in _decode(fa.require_head(m), features, tasks, m.DEVICE))
    report = ExperimentReport('Original fast Fusion answers: independent truth unavailable',
        pd.DataFrame([dict(cases=len(cases), answers=len(rows), accuracy_available=False)]),
        details=pd.DataFrame(rows), diagnostics={'answer_consistency': _consistency(rows)},
        metadata=dict(decision_system=fa.SYSTEM, model_evaluated=True, deployed_weights_updated=False,
                      original_cross_run_fault_accuracy_available=False))
    from .evaluation_audit import original_alarm_rates
    report.diagnostics['original_alarm_rates'] = original_alarm_rates(report.details)
    return report, cases


def inspect_original(runtime, case, center):
    """Return actual answer attention; no claim that attention explains correctness."""
    require_runtime(runtime); m = runtime.module
    if center not in case['centers']:
        raise ValueError('Select a center included in this review case')
    captured = []
    def capture(model, args, output):
        captured.append(output[2].detach().cpu())
    handle = m.GEVIS_MODEL.register_forward_hook(capture)
    try:
        features, tasks = fa.fusion_features(m, case['current'], case['reference'],
            case['step'], [center], case['position'])
    finally:
        handle.remove()
    names = list(runtime.encoders[case['step']][1]['sensor_columns'])
    attention = captured[0].float().reshape(3, len(names), -1).numpy()
    return dict(answers=_decode(fa.require_head(m), features, tasks, m.DEVICE), attention=attention,
                sensors=names, center=center, step=case['step'], truth_available=False)


def plot_review(runtime, case, center):
    import matplotlib.pyplot as plt
    result = inspect_original(runtime, case, center)
    sid = result['sensors'].index(center)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes[0, 0].plot(case['current'][:, sid], label='current')
    axes[0, 0].plot(case['reference'][:, sid], label='matched TRAIN-run reference median')
    axes[0, 0].set(title=f'{case["step"]} / {center}', xlabel='sample within model window',
                   ylabel='source-TRAIN standardized value'); axes[0, 0].legend()
    for axis, answer in zip((axes[0, 1], axes[1, 0], axes[1, 1]), result['answers']):
        axis.bar(list(answer['probabilities']), list(answer['probabilities'].values()))
        axis.set(title=answer['answer_type'] + ': ' + answer['answer'], ylim=(0, 1))
        axis.tick_params(axis='x', labelrotation=25)
    fig.suptitle('Original comparison; independent truth unavailable; softmax is not fault probability')
    fig.tight_layout()
    return fig, result


def benchmark(runtime, *, steps=None, config=fa.FusionConfig(), repeats=20, warmup=3):
    if repeats < 1 or warmup < 0:
        raise ValueError('Positive repeats and nonnegative warmup required')
    cases = original_cases(runtime, steps=steps, config=config)
    m = runtime.module; device = torch.device(m.DEVICE)
    groups = {step: [case for case in cases if case['step'] == step] for step in dict.fromkeys(c['step'] for c in cases)}
    cases = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]
    on_cuda = device.type == 'cuda'
    def sync():
        if on_cuda:
            torch.cuda.synchronize(device)
    def forward(case):
        return fa.predict_arrays(m, case['current'], case['reference'], case['step'],
                                 case['centers'][0], case['position'])
    for group in groups.values():
        for i in range(warmup):
            forward(group[i % len(group)])
    sync()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for i in range(repeats):
        case = cases[i % len(cases)]
        sync(); started = time.perf_counter(); answers = forward(case); sync()
        rows.append(dict(repetition=i, step=case['step'], pair_id=case['pair_id'],
                         latency_ms=1000 * (time.perf_counter() - started), answers=len(answers)))
    frame = pd.DataFrame(rows); times = frame.latency_ms.to_numpy()
    summary = pd.DataFrame([dict(device=str(device), measured_windows=len(frame), questions_per_window=3,
        mean_ms_per_window=float(times.mean()), median_ms_per_window=float(np.median(times)),
        p95_ms_per_window=float(np.quantile(times, .95)), windows_per_second=float(1000 / times.mean()))])
    memory = pd.DataFrame([dict(device=str(device), metric='peak allocated CUDA MiB',
        value=torch.cuda.max_memory_allocated(device) / 2**20)]) if on_cuda else pd.DataFrame()
    return FastSuite(summary, details=frame, diagnostics={'device_memory': memory}, metadata=dict(
        experiment='actual_frozen_inference_cost', deployed_weights_updated=False, warmup_windows=warmup * len(groups),
        timing_scope='standardized window -> STEP1 numeric state -> question Fusion -> three head answers; excludes disk/preprocessing',
        limitation='CPU RAM not measured. Repeated-window timing is hardware-dependent; no quality claim.'))


def show_suite(suite, display):
    display(suite.summary)
    for key, table in suite.diagnostics.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            print(key); display(table)
    if suite.metadata.get('limitation'):
        print(suite.metadata['limitation'])
