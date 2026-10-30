"""One frozen request path: native window -> TRAIN reference -> GEVIS answers.

No CSV lookup, labels, injection, training or cross-request numeric-state cache.
The request unit is a model window, not a certified physical equipment cycle.
"""
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import fusion_answers as fa, fast_experiments as fast, peald_step_heads

RECIPE = 'gevis.frozen_train_reference_request.v1'


def sync(module):
    device = torch.device(module.DEVICE)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict_pair(runtime, case, centers, *, center_batch_size=8):
    """Equivalent learned features; one numeric encoding for all center batches."""
    if not 1 <= center_batch_size <= 32:
        raise ValueError('Center batch size must be in 1..32')
    step = case['step']
    scoped = runtime.step_runtimes[step] if hasattr(runtime, 'step_runtimes') else runtime
    fast._steps(scoped, [step])
    if hasattr(scoped, 'prepared'):
        scoped = peald_step_heads.for_step(scoped, step)
    original = scoped.module
    module = SimpleNamespace(**original.__dict__)
    head = fa.require_head(module)
    names = list(module.STEP_ENCODERS[step][1]['sensor_columns'])
    if not centers or len(set(centers)) != len(centers) or any(c not in names for c in centers):
        raise ValueError('Unique model-supported selected sensors required')
    state = None; encoder_ms = 0.; fusion_ms = 0.; head_ms = 0.

    def numeric_once(encoder, current, reference):
        nonlocal state, encoder_ms
        if state is None:
            sync(module); start = time.perf_counter()
            state = original.numeric_state(encoder, current, reference)
            sync(module); encoder_ms = 1000*(time.perf_counter()-start)
        return state

    # The module namespace is local to this call. Shared runtime methods/weights
    # are never replaced; a later window must execute its own numeric encoder.
    module.numeric_state = numeric_once
    answers = []
    for offset in range(0, len(centers), center_batch_size):
        batch = centers[offset:offset+center_batch_size]
        previous_encoder = encoder_ms
        sync(module); start = time.perf_counter()
        features, tasks = fa.fusion_features(module, case['current'], case['reference'],
            step, batch, case.get('position', 0.))
        sync(module)
        fusion_ms += 1000*(time.perf_counter()-start)-(encoder_ms-previous_encoder)
        start = time.perf_counter()
        features = features.to(module.DEVICE)
        decoded = {}
        for kind in fa.KINDS:
            indices = [i for i, (_, k) in enumerate(tasks) if k == kind]
            probabilities = head(features[indices], kind).softmax(-1).cpu()
            for index, values in zip(indices, probabilities):
                choice = int(values.argmax())
                decoded[index] = dict(center_sensor=tasks[index][0], answer_type=kind,
                    prediction=choice, answer=fa.ANSWER_CRITERIA[kind][choice],
                    probabilities=dict(zip(fa.ANSWER_CRITERIA[kind], map(float, values))))
        answers.extend(decoded[i] for i in range(len(tasks)))
        sync(module); head_ms += 1000*(time.perf_counter()-start)
    return answers, dict(encoder_ms=encoder_ms, fusion_ms=fusion_ms, head_ms=head_ms)


class InferenceSession:
    """Startup-only frozen reference bank; one process_window call per input."""
    def __init__(self, runtime, *, center_batch_size=8):
        if not 1 <= center_batch_size <= 32:
            raise ValueError('Center batch size must be in 1..32')
        self.runtime = runtime
        self.center_batch_size = center_batch_size
        self.modules = {}; self.metadata = {}; self.bank = {}; self.references = {}
        if hasattr(runtime, 'step_runtimes'):
            steps = list(runtime.step_runtimes)
        else:
            trained = (getattr(runtime.module, 'FUSION_ANSWER_TRAINING', None) or {}).get('trained_steps')
            steps = fast._steps(runtime, [s for s in runtime.encoders if trained is None or s in trained])
        for step in steps:
            scoped = runtime.step_runtimes[step] if hasattr(runtime, 'step_runtimes') else runtime
            fast._steps(scoped, [step])
            self.modules[step] = scoped.module
            meta = dict(scoped.encoders[step][1])
            for key in ('mean', 'std'):
                meta[key] = np.asarray(meta[key]).copy()
                meta[key].setflags(write=False)
            self.metadata[step] = meta
            if not hasattr(scoped, 'prepared'):
                # Source PECVD has already-standardized TRAIN caches. Read only
                # TRAIN values; never use the Test current or paired reference.
                from pathlib import Path
                with np.load(Path(scoped.module.ENCODER_ROOT)/'windows'/f'{step}.npz', allow_pickle=False) as cache:
                    runs = {s: cache[f'{s}_process_id'].astype(str) for s in ('train', 'validation', 'test')}
                    if any(set(runs[a]) & set(runs[b]) for a, b in
                           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
                        raise ValueError('Run overlap in cached partitions')
                    for run, cohort, position, values in zip(runs['train'], cache['train_cohort_key'].astype(str),
                            cache['train_position_bin'], cache['train']):
                        key = (step, str(cohort), int(position))
                        self.bank.setdefault(key, {}).setdefault(str(run), np.array(values, copy=True))
                continue
            for row in scoped.prepared.windows[step]['train']:
                if row.get('split', 'train') != 'train':
                    raise ValueError('Only TRAIN reference windows allowed at startup')
                recipe, _, offset, tick = row['cohort']
                key = (step, recipe, offset, tick)
                self.bank.setdefault(key, {}).setdefault(row['run_id'], row['current'])
        # Common independent-input references are prepared before timed requests.
        for key, by_run in self.bank.items():
            self.references[(key, None)] = self._median(by_run, None)

    @staticmethod
    def _median(by_run, excluded_run):
        values = [v for r, v in by_run.items() if r != excluded_run][:16]
        if not values:
            return None
        value = np.median(np.stack(values), axis=0)
        value.setflags(write=False)
        return value

    def _reference(self, step, cohort, run_id):
        if not isinstance(cohort, (tuple, list)):
            raise ValueError('Explicit structured reference cohort required')
        if isinstance(cohort, (tuple, list)) and len(cohort) == 2:
            # PECVD source cache: recipe/context cohort and relative position bin.
            key = (step, str(cohort[0]), int(cohort[1]))
        elif len(cohort) == 4:
            recipe, _, offset, tick = cohort
            key = (step, recipe, offset, tick)
        elif len(cohort) == 3:
            recipe, offset, tick = cohort
            key = (step, recipe, offset, tick)
        else:
            raise ValueError('Explicit recipe/native offset/time cohort required')
        bank = self.bank.get(key, {})
        excluded = run_id if run_id in bank else None
        cache_key = (key, excluded)
        if cache_key not in self.references:
            self.references[cache_key] = self._median(bank, excluded)
        return self.references[cache_key], min(16, len(bank)-int(excluded is not None))

    def process_window(self, *, current, step, cohort, run_id, centers=None,
                       position=0., standardized=False):
        """No reference/truth argument: Test originals cannot enter inference."""
        module = self.modules.get(step)
        if module is not None:
            sync(module)
        started = time.perf_counter()
        result = dict(status='NOT_EVALUATED', reason=None, step=step, answers=[],
            reference_count=0, reference_policy='frozen_TRAIN_matching_context_excluding_same_run',
            processing_unit='one complete native Step model window; all selected sensors; three answers per sensor')
        times = dict(reference_ms=0., normalization_ms=0., encoder_ms=0., fusion_ms=0., head_ms=0.)
        if module is None:
            result['reason'] = 'unsupported_step'
        else:
            meta = self.metadata[step]
            names = list(meta['sensor_columns'])
            selected = names if centers is None else list(centers)
            start = time.perf_counter()
            values = np.asarray(current, dtype=np.float32)
            if values.shape != (meta['window_length'], len(names)) or not np.isfinite(values).all():
                raise ValueError('One complete finite native model window required')
            if not standardized:
                values = ((values-meta['mean'])/meta['std']).astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite normalized input')
            times['normalization_ms'] = 1000*(time.perf_counter()-start)
            start = time.perf_counter()
            reference, count = self._reference(step, cohort, run_id)
            times['reference_ms'] = 1000*(time.perf_counter()-start)
            if reference is None:
                result['reason'] = 'no_matching_independent_TRAIN_reference'
            else:
                answers, stages = predict_pair(self.runtime, dict(step=step, current=values,
                    reference=reference, position=position), selected, center_batch_size=self.center_batch_size)
                times.update(stages)
                result.update(status='OK', answers=answers, reference_count=count,
                              selected_sensors=len(selected))
            sync(module)
        times['request_ms'] = 1000*(time.perf_counter()-started)
        result['timings'] = times
        return result


def latency_tables(requests, *, domain='PEALD', goal_ms=50.):
    frame = pd.DataFrame(requests)
    ok = frame.loc[frame.status.eq('OK')] if not frame.empty else frame
    def summary(group, **identity):
        values = group.request_ms.to_numpy()
        return dict(identity, measured_windows=len(group),
            mean_ms_per_window=float(values.mean()), median_ms_per_window=float(np.median(values)),
            p95_ms_per_window=float(np.quantile(values, .95)), p99_ms_per_window=float(np.quantile(values, .99)),
            windows_per_second=float(1000/values.mean()),
            mean_request_within_50ms=bool(values.mean() <= goal_ms),
            fraction_requests_over_50ms=float((values > goal_ms).mean()))
    return (pd.DataFrame([summary(ok, domain=domain)]) if len(ok) else pd.DataFrame(),
            pd.DataFrame([summary(g, domain=domain, step=s) for s, g in ok.groupby('step', sort=False)]) if len(ok) else pd.DataFrame())


def run_cases(session, cases, *, domain='PEALD', progress=None):
    rows = []; requests = []
    for i, case in enumerate(cases):
        result = session.process_window(current=case.get('raw_current', case['current']),
            step=case['step'], cohort=case['cohort'], run_id=case['run_id'],
            centers=case.get('centers'), position=case.get('position', 0.),
            standardized='raw_current' not in case)
        pair_id = case.get('pair_id', str(i))
        requests.append(dict(request_id=pair_id, step=case['step'], status=result['status'],
            reason=result['reason'], selected_sensors=result.get('selected_sensors', 0),
            answers=len(result['answers']), **result['timings']))
        rows.extend(dict(a, pair_id=pair_id, step=case['step']) for a in result['answers'])
        if progress is not None and (i == 0 or (i+1) % 25 == 0):
            progress(f'Completed input windows: {i+1}', flush=True)
    summary, by_step = latency_tables(requests, domain=domain)
    return fast.FastSuite(summary, details=pd.DataFrame(rows), diagnostics=dict(
        requests=pd.DataFrame(requests), by_step=by_step, answer_consistency=fast._consistency(rows)),
        metadata=dict(domain=domain, inference_recipe=RECIPE, truth_available=False,
            cycle_KPI_verified=False, timing_scope='complete model-window request -> normalization/reference selection -> encoder/Fusion/head -> answer dictionaries; excludes disk, startup, segmentation and transport',
            processing_unit='all selected sensors and three answers per complete Step model window',
            limitation='No independent accuracy without truth. Window request timing is not a certified physical-cycle/Edge KPI.'))
