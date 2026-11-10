"""Batched, request-local control48 inference using unchanged saved weights.

This sibling module deliberately leaves heads/distilation/*.py unchanged:
existing merged package code/content checks remain strict and loadable.
No cross-request numeric/feature cache, dtype switch, TF32 switch, or fitting.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from heads.distilation.runtime import Session as ReferenceSession, sync
from heads.distilation.student import GevisStudent, inputs, boundary_tail
from heads.distilation.mapping import remap_reply
from heads.distilation.mapping import metrics
from pecvd.experiments import ald_preprocessing as preprocessing, decision_training as d, fusion_answers as fa

EXECUTION_POLICY = 'control48_batched_head_request_local_bound_grids_v1'


def normalize_numeric_array(current, meta):
    """Equivalent numeric ndarray shortcut; preserve original object/valve rules.

    Check raw valve values before float32 conversion: rounding a nonbinary
    float64 value to 0/1 must not turn an invalid valve into a valid one.
    Negative zero and bool/string cases use the original string encoder.
    """
    names = meta['sensor_columns']
    if (isinstance(current, np.ndarray) and current.dtype.kind in 'fiu'
            and current.shape == (meta['window_length'], len(names))):
        valves = [i for i, name in enumerate(names) if name.startswith('ValveAct_')]
        valid = bool(np.isfinite(current).all())
        if valves:
            values = current[:, valves]
            valid = valid and bool(np.isin(values, [0, 1]).all())
            if current.dtype.kind == 'f':
                valid = valid and not bool(((values == 0) & np.signbit(values)).any())
        if valid:
            with np.errstate(over='ignore', invalid='ignore'):
                raw = np.asarray(current, np.float32)
            if np.isfinite(raw).all():
                return ((raw - np.asarray(meta['mean'])) / np.asarray(meta['std'])).astype(np.float32)
    frame = current if isinstance(current, pd.DataFrame) else pd.DataFrame(current, columns=names)
    return preprocessing.transform_live(frame, meta)


class _WindowFeatures:
    """Same Student operators; reuse bound grids/means only within one request."""
    def __init__(self, student, state):
        self.student = student
        self.state = state
        self.grids = [student.key_norm(state + student.boundary.weight[bound]) for bound in range(2)]
        self.means = [grid.mean(1) for grid in self.grids]

    def features(self, centers, tail, kinds):
        s = self.student
        channels, patches, _ = self.state.shape
        n, k = len(centers), len(kinds)
        ids = centers.repeat_interleave(k)
        kind_ids = torch.tensor([fa.KINDS.index(kind) for kind in kinds], device=self.state.device).repeat(n)
        query = s.question_vectors[kind_ids]
        parts = []
        for grid, mean in zip(self.grids, self.means):
            local = mean[ids]
            q = s.query_norm(s.text(query) + local)
            answer, attention = s.cross(q[None], grid.reshape(1, -1, s.width),
                grid.reshape(1, -1, s.width), need_weights=True)
            weights = attention[0].reshape(n * k, channels, patches)[torch.arange(n * k, device=self.state.device), ids]
            attended = (grid[ids] * weights[..., None]).sum(1) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            fused = s.fused_out(s.answer_norm(answer[0] + q))
            parts.append(torch.cat([fused, query, s.numeric_out(local), s.numeric_out(attended)], -1))
        exact_tail = tail.repeat_interleave(k, dim=0)
        return torch.cat(parts + [exact_tail], -1).reshape(n, k, 8 * s.dimension + 10)


class FastSession(ReferenceSession):
    def __init__(self, runtime, student=None, *, center_batch_size=None):
        super().__init__(runtime, student, center_batch_size=center_batch_size)
        if self.student is not None and not isinstance(self.student, GevisStudent):
            raise ValueError('Original control48 GevisStudent required')
        if self.student is not None and (self.student.training or runtime.module.FUSION_ANSWER_HEAD.training):
            raise ValueError('Frozen eval-mode Student and original Head required')

    @torch.no_grad()
    def process_window(self, *, current, step, recipe_key, offset_samples, lineage_id,
                       loop_key='', centers=None, sensor_columns=None, standardized=False, run_id=None):
        request = dict(current=current, step=step, recipe_key=recipe_key, offset_samples=offset_samples,
            lineage_id=lineage_id, loop_key=loop_key, centers=centers, sensor_columns=sensor_columns,
            standardized=standardized, run_id=run_id)
        if self.student is None:
            return super().process_window(**request)
        r = self.runtime
        def withheld(reason):
            return dict(status='NOT_EVALUATED', reason=reason, answers=[])
        if step not in r.encoders:
            return withheld('unsupported_native_context')
        meta = r.encoders[step][1]
        names = meta['sensor_columns']
        if sensor_columns is not None and list(sensor_columns) != names:
            return withheld('incomplete_observed_sensor_grid')
        if not isinstance(lineage_id, str) or not lineage_id:
            return withheld('source_lineage_required')
        if lineage_id in set(r.references['basis_lineages']):
            return withheld('reference_basis_lineage_not_independent')
        record = r.reference_bank.get(d._key(step, recipe_key, loop_key, offset_samples))
        if record is None:
            return withheld('native_normal_reference_unobserved')
        targets = r.target_sensors[step]
        chosen = list(targets) if centers is None else list(centers)
        if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(names):
            raise ValueError('Unique known centers required')
        if set(chosen) - set(targets):
            raise ValueError('Context-only sensor cannot be selected as a prediction target')
        try:
            if standardized:
                values = np.asarray(current, np.float32)
            elif self.input_policy == 'detailed':
                values = np.asarray(current, np.float32)
                if values.shape != (meta['window_length'], len(names)) or not np.isfinite(values).all():
                    raise ValueError('Finite complete native window required')
                values = ((values - np.asarray(meta['mean'])) / np.asarray(meta['std'])).astype(np.float32)
            else:
                values = normalize_numeric_array(current, meta)
            if values.shape != (meta['window_length'], len(names)) or not np.isfinite(values).all():
                raise ValueError('Finite complete native window required')
        except (ValueError, TypeError) as error:
            return withheld(str(error))
        device = r.module.DEVICE
        head = r.module.FUSION_ANSWER_HEAD
        name_ids = {name: i for i, name in enumerate(names)}
        center_ids = torch.tensor([name_ids[c] for c in chosen], device=device)
        # Exact reference tail once for all centers (rather than recomputing
        # complete below/above arrays for every center batch and answer stage).
        tails = boundary_tail(values, record, chosen, names, offset_samples, float(meta['cadence_seconds'])).to(device)
        window = _WindowFeatures(self.student, self.student.encode(inputs(values, record).to(device)))
        decoded = {}
        def infer(selected, ids, tail, kinds):
            chunks = [window.features(ids[start:start + self.center_batch_size],
                tail[start:start + self.center_batch_size], kinds)
                for start in range(0, len(selected), self.center_batch_size)]
            features = torch.cat(chunks, 0)
            probabilities = {}
            labels_by_kind = {}
            offsets = {}
            # At most one original-Head call per answer type, for the whole
            # request. Queue typed GPU work before transferring its results.
            for j, kind in enumerate(kinds):
                logits = head(features[:, j], kind).clone()
                if kind == 'CHOICE' and hasattr(head, 'choice_probabilities'):
                    probabilities[kind] = head.choice_probabilities(logits)
                    labels_by_kind[kind] = head.choice_labels
                    offsets[kind] = 1
                else:
                    if kind != 'BOOLEAN':
                        logits[:, 0] = -torch.inf
                    probabilities[kind] = logits.softmax(-1)
                    labels_by_kind[kind] = fa.ANSWER_CRITERIA[kind]
                    offsets[kind] = 0
            for kind in kinds:
                probs = probabilities[kind].cpu()
                labels = labels_by_kind[kind]
                offset = offsets[kind]
                predictions = (probs.argmax(-1) + offset).tolist()
                vectors = probs.tolist()
                for center, prediction, vector in zip(selected, predictions, vectors):
                    decoded[center, kind] = dict(center_sensor=center, answer_type=kind,
                        prediction=prediction, answer=labels[prediction - offset],
                        probabilities=dict(zip(labels, vector)), evaluated=True, status='EVALUATED')
        infer(chosen, center_ids, tails, ('BOOLEAN',))
        positions = [i for i, c in enumerate(chosen) if decoded[c, 'BOOLEAN']['prediction'] == 1]
        if positions:
            positive = [chosen[i] for i in positions]
            indices = torch.tensor(positions, device=device)
            infer(positive, center_ids[indices], tails[indices], ('CHOICE', 'ORDINAL'))
        answers = [decoded.get((c, k), dict(center_sensor=c, answer_type=k, prediction=None,
            answer=None, probabilities=None, evaluated=False, status='NOT_APPLICABLE', reason='BOOLEAN_No'))
            for c in chosen for k in fa.KINDS]
        return remap_reply(dict(status='OK', reason=None, answers=answers, run_id=run_id,
            reference_count=record['reference_runs'], selected_sensors=len(chosen),
            evaluated_answers=sum(a['evaluated'] for a in answers), conditional_answering=True))


def reply_parity(reference, candidate, *, atol=1e-5, rtol=1e-4):
    """Compare original-five decisions too; mapping must not conceal a change."""
    if reference['status'] != candidate['status'] or reference.get('reason') != candidate.get('reason'):
        return False, np.inf
    left, right = reference['answers'], candidate['answers']
    if len(left) != len(right):
        return False, np.inf
    maximum = 0.
    for a, b in zip(left, right):
        for key in ('center_sensor', 'answer_type', 'evaluated', 'prediction', 'original_five_prediction', 'answer'):
            if a.get(key) != b.get(key):
                return False, np.inf
        for key in ('probabilities', 'original_five_probabilities'):
            p, q = a.get(key), b.get(key)
            if p is None or q is None:
                if p != q:
                    return False, np.inf
                continue
            if list(p) != list(q):
                return False, np.inf
            x, y = np.array(list(p.values())), np.array(list(q.values()))
            maximum = max(maximum, float(np.max(np.abs(x - y))))
            if not np.allclose(x, y, atol=atol, rtol=rtol):
                return False, maximum
    return True, maximum


def benchmark_requests(runtime, requests, *, batches=(8, 16, 32), repeats=5, warmup=2, split='validation'):
    """Parity first, then identical whole-window wall timing, without labels/KD.

    No inference state survives a request. Timed requests contain raw input so
    normalization, reference, all targets, decoding and final mapping are timed.
    """
    if (split not in ('validation', 'test') or not requests
            or any(type(n) is not int or n < 1 for n in (repeats, warmup))):
        raise ValueError('Heldout requests and positive timing budgets required')
    if getattr(runtime, 'student', None) is None:
        raise ValueError('Merged Student runtime required; do not substitute Teacher timings')
    if any(request.get('standardized', False) for request in requests):
        raise ValueError('Raw requests required for whole-window latency comparison')
    if not batches or len(set(batches)) != len(batches) or any(type(n) is not int or not 1 <= n <= 32 for n in batches):
        raise ValueError('Unique center batches in 1..32 required')
    sessions = {'reference': ReferenceSession(runtime)}
    sessions.update({f'batch{size}': FastSession(runtime, center_batch_size=size) for size in batches})
    parity = []
    for window, request in enumerate(requests):
        reference = sessions['reference'].process_window(**request)
        if reference['status'] != 'OK':
            raise ValueError('Reference request withheld; do not report an answered latency')
        for name, session in sessions.items():
            if name == 'reference':
                continue
            equal, difference = reply_parity(reference, session.process_window(**request))
            parity.append(dict(runtime=name, window=window, passed=equal, max_probability_difference=difference))
            if not equal:
                raise ValueError(f'Original-five/gate/ordinal/probability parity failed for {name}; timing refused')
    records = []
    device = runtime.module.DEVICE
    names = tuple(sessions)
    for window, request in enumerate(requests):
        for repeat in range(warmup + repeats):
            offset = (window + repeat) % len(names)
            for name in names[offset:] + names[:offset]:
                sync(device); start = time.perf_counter()
                reply = sessions[name].process_window(**request)
                sync(device); elapsed = (time.perf_counter() - start) * 1000
                if reply['status'] != 'OK':
                    raise ValueError('Timing request withheld')
                if repeat >= warmup:
                    records.append(dict(runtime=name, window=window, repeat=repeat, split=split, wall_ms=elapsed,
                        sensors=len(request['sensor_columns']), selected_sensors=reply['selected_sensors'],
                        evaluated_answers=reply['evaluated_answers']))
    frame = pd.DataFrame(records)
    summary = pd.DataFrame([dict(runtime=name, split=split, mean_ms=g.wall_ms.mean(),
        p95_ms=np.percentile(g.wall_ms, 95), requests=len(g), windows=g.window.nunique(),
        min_sensors=g.sensors.min(), max_sensors=g.sensors.max(),
        mean_evaluated_answers=g.evaluated_answers.mean(), device=str(device),
        scope='same raw Window: normalization/reference/Student/original Head/final mapping')
        for name, g in frame.groupby('runtime', sort=False)])
    return dict(summary=summary, requests=frame, parity=pd.DataFrame(parity), execution=EXECUTION_POLICY,
        runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def heldout_requests(experiment, split='validation', windows=16):
    """Reuse prepared, bound cases. Does not refit or reopen Test for selection."""
    if split not in ('validation', 'test') or type(windows) is not int or windows < 1:
        raise ValueError('Heldout split and positive window count required')
    if split == 'test' and not experiment.test_done:
        raise ValueError('Finish original sealed Test before its timing-only comparison')
    if split not in experiment.datasets:
        raise ValueError('Keep the existing encoded heldout study; this timing helper will not prepare/encode new data')
    cases = experiment.datasets[split]['cases']
    ids = np.linspace(0, len(cases) - 1, min(windows, len(cases)), dtype=int)
    return [experiment.request(cases[int(i)], raw=True) for i in ids]


def choose_batch(validation_report, target_ms=50.):
    table = validation_report['summary']
    if (not table.split.eq('validation').all() or validation_report['parity'].empty
            or not validation_report['parity'].passed.all()):
        raise ValueError('Choose execution batch on parity-checked Validation only')
    if not np.isfinite(target_ms) or target_ms <= 0:
        raise ValueError('Positive finite mean-latency target required')
    candidates = table.loc[table.runtime.str.startswith('batch')]
    if candidates.empty:
        raise ValueError('No accelerated Validation candidate')
    qualified = candidates.loc[candidates.mean_ms.le(target_ms)]
    ranked = qualified if not qualified.empty else candidates
    row = ranked.sort_values(['p95_ms', 'mean_ms', 'runtime']).iloc[0]
    return dict(center_batch_size=int(row.runtime[5:]), selected_on='validation',
        mean_ms=float(row.mean_ms), p95_ms=float(row.p95_ms), mean_target_ms=target_ms,
        mean_target_met=bool(row.mean_ms <= target_ms), weights_changed=False, seed_search=False,
        runtime_source_sha256=validation_report['runtime_source_sha256'])


def verify_scored_cases(experiment, runtime, *, split='validation', center_batch_size=32, progress=print):
    """No refitting: compare the reference and batched runtimes on scored centers.

    Produces same-protocol F1 tables for both runtimes. Uses already prepared
    cases and their original targets; never changes the original sealed study.
    """
    if split not in ('validation', 'test') or split not in experiment.datasets:
        raise ValueError('Existing heldout cases required')
    if getattr(runtime, 'student', None) is None:
        raise ValueError('Merged Student runtime required')
    if split == 'test' and not experiment.test_done:
        raise ValueError('Original Test must already be evaluated; do not use it to choose a runtime')
    reference = ReferenceSession(runtime)
    candidate = FastSession(runtime, center_batch_size=center_batch_size)
    truth, old, new, rows = [], [], [], []
    def decisions(reply, centers):
        answers = {(a['center_sensor'], a['answer_type']): a for a in reply['answers']}
        return np.array([[0 if not answers[c, k]['evaluated'] else
            answers[c, k].get('original_five_prediction', answers[c, k]['prediction'])
            for k in fa.KINDS] for c in centers])
    cases = experiment.datasets[split]['cases']
    for index, case in enumerate(cases):
        request = experiment.request(case, raw=False, selected_centers=True)
        expected, found = reference.process_window(**request), candidate.process_window(**request)
        passed, difference = reply_parity(expected, found)
        if expected['status'] != 'OK' or not passed:
            raise ValueError(f'Scored-center parity failed at case {index}; stop before claiming unchanged F1')
        truth.append(np.asarray(case['targets']))
        old.append(decisions(expected, case['centers']))
        new.append(decisions(found, case['centers']))
        rows.append(dict(case=index, questions=len(case['centers']), passed=passed, max_probability_difference=difference))
        if progress and (index == 0 or (index + 1) % 100 == 0 or index + 1 == len(cases)):
            progress(f'{split} scored-case parity {index + 1}/{len(cases)}', flush=True)
    tables = [metrics(np.concatenate(truth), np.concatenate(predicted), model=name, split=split)[0]
              for name, predicted in (('reference_runtime', old), ('batched_runtime', new))]
    return dict(metrics=pd.concat(tables, ignore_index=True), parity=pd.DataFrame(rows), split=split,
        scope='same existing normalized scored inputs; same original-five decisions, gate and ordinal; no training')
