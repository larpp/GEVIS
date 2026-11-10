"""Source-bound control48 KD, Validation selection, sealed Test and wall timing."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import time
import uuid

import numpy as np
import pandas as pd
import torch

from pecvd.gevis_backbone import ANSWER_TYPES
from pecvd.experiments import decision_training as d, fusion_answers as fa
from . import data
from .student import GevisStudent, inputs, boundary_tail
from .objective import Config, legacy_loss, head_logits, original_predictions, legacy_validation_selection
from .mapping import metrics, remap
from .runtime import Session, sync, teacher_binding, save_merged, validate_original_head


class Study:
    def __init__(self, root, teacher, config=Config(), *, progress=print, question_vectors=None):
        config.validate()
        self.root = Path(root).resolve()
        self.teacher = teacher
        self.config = config
        self.progress = progress
        device = torch.device(teacher.module.DEVICE)
        self.head = teacher.module.FUSION_ANSWER_HEAD.to(device)
        validate_original_head(self.head)
        self.head.requires_grad_(False).eval()
        # The official loader restores weights on CPU even when DEVICE is CUDA.
        # fusion_features normally aligns the whole model before _text, but KD
        # queries are extracted before that first forward. Align all registered
        # children (including E5 embeddings and projection) before making IDs.
        teacher.module.GEVIS_MODEL.to(device).requires_grad_(False).eval()
        for encoder, _ in teacher.encoders.values():
            encoder.to(device).requires_grad_(False).eval()
        if not hasattr(teacher, 'distilation_input_policy'):
            from pecvd.experiments.ald_head_experiments import ExperimentHead
            teacher.distilation_input_policy = 'domain' if isinstance(self.head, ExperimentHead) else 'detailed'
        if not hasattr(teacher, 'target_sensors'):
            teacher.target_sensors = {s: list(m['sensor_columns']) for s, (_, m) in teacher.encoders.items()}
        lengths = {int(e.patch_len) for e, _ in teacher.encoders.values()}
        if len(lengths) != 1:
            raise ValueError('control48 requires one source patch length across selected steps')
        self.patch_length = lengths.pop()
        if any(int(m['window_length']) % self.patch_length for _, m in teacher.encoders.values()):
            raise ValueError('Teacher windows must contain complete source patches')
        if question_vectors is None:
            model = teacher.module.GEVIS_MODEL
            with torch.no_grad():
                ids = torch.tensor([ANSWER_TYPES[k] for k in fa.KINDS], device=device)
                question_vectors = model._text([fa.QUESTIONS[k] for k in fa.KINDS], device) + model.type_emb(ids)
        self.queries = question_vectors.detach().float().cpu().clone()
        if self.queries.shape[0] != 3 or self.head.feature_mean.numel() != 8 * self.queries.shape[1] + 10:
            raise ValueError('Teacher query and original-head 8D+10 contract differs')
        self.binding = teacher_binding(teacher)
        self.folder = self.root / '.cache/distilation' / uuid.uuid4().hex
        self.datasets = {}
        self.student = None
        self.training = None
        self.validation = None
        self.validation_timing = None
        self.selection = None
        self.test_opened = False
        self.test_done = False
        self._parity_passed = False

    def assert_preserved(self):
        if teacher_binding(self.teacher) != self.binding:
            raise ValueError('Teacher/head/reference/input binding changed during distillation')
        if any(p.requires_grad or p.grad is not None for p in self.head.parameters()):
            raise ValueError('Original Head must stay frozen without gradients')
        if self.student is not None and not torch.equal(self.student.question_vectors.detach().cpu(), self.queries):
            raise ValueError('Fixed Teacher question vectors changed')

    def _open_training(self):
        if self.test_opened or self.selection is not None:
            raise ValueError('Selection is frozen/Test opened; further fitting or optimization is closed')

    def dataset(self, split):
        if split == 'test' and self.selection is None:
            raise ValueError('Freeze Validation selection before opening Test')
        if split in self.datasets:
            return self.datasets[split]
        cases, coverage = data.cases_for(self.teacher, split, self.config)
        lineages = {c['lineage_id'] for c in cases}
        for other in self.datasets.values():
            if lineages & set(other['coverage']['lineages']):
                raise ValueError('KD split source lineages overlap')
        paths = []
        device = self.teacher.module.DEVICE
        folder = self.folder / split
        folder.mkdir(parents=True, exist_ok=False)
        with torch.no_grad():
            for case_index, case in enumerate(cases):
                values = d.case_current(case)
                record = self.teacher.reference_bank[tuple(case['reference_key'])]
                names = self.teacher.encoders[case['step']][1]['sensor_columns']
                x = inputs(values, record)
                input_file = f'inputs-{case_index:07d}.pt'
                torch.save(x, folder / input_file)
                for start in range(0, len(case['centers']), self.config.center_batch_size):
                    centers = case['centers'][start:start + self.config.center_batch_size]
                    scoped = dict(case, centers=centers, targets=case['targets'][start:start + len(centers)])
                    features = d._features(self.teacher, scoped, self.teacher.reference_bank).float().cpu()
                    logits = {k: v.cpu() for k, v in head_logits(self.head, features.to(device)).items()}
                    row = dict(inputs_file=input_file, centers=torch.tensor([names.index(c) for c in centers]),
                        tail=boundary_tail(values, record, centers, names, case['offset_samples'],
                            float(self.teacher.encoders[case['step']][1]['cadence_seconds'])),
                        y=torch.as_tensor(scoped['targets'], dtype=torch.long), features=features,
                        teacher=logits, case_index=case_index, centers_names=centers)
                    path = folder / f'{len(paths):07d}.pt'
                    torch.save(row, path)
                    paths.append(path)
                if self.progress and (case_index == 0 or (case_index + 1) % 20 == 0 or case_index + 1 == len(cases)):
                    self.progress(f'Frozen Teacher {split} cases {case_index + 1}/{len(cases)}; disk rows={len(paths)}', flush=True)
        if not paths:
            raise ValueError('No encoded KD centers')
        result = dict(paths=paths, cases=cases, coverage=coverage)
        self.datasets[split] = result
        return result

    @staticmethod
    def row(path):
        row = torch.load(path, map_location='cpu', weights_only=False)
        row['x'] = torch.load(Path(path).parent / row['inputs_file'], map_location='cpu', weights_only=False)
        return row

    def _weights(self, dataset):
        labels = np.concatenate([self.row(path)['y'].numpy() for path in dataset['paths']])
        weights = {}
        for j, kind in enumerate(fa.KINDS):
            selected = labels[:, j] if j == 0 else labels[(labels[:, 0] == 1) & (labels[:, j] > 0), j]
            counts = np.bincount(selected, minlength=len(fa.ANSWER_CRITERIA[kind]))
            expected = getattr(self.head, 'allowed_' + kind).detach().cpu().numpy().copy()
            if j:
                expected[0] = False
            if np.any(expected & (counts == 0)):
                raise ValueError('Increase TRAIN sample: missing original-head ' + kind + ' class support')
            active = counts > 0
            values = np.zeros(len(counts), np.float32)
            values[active] = counts.sum() / (active.sum() * counts[active])
            weights[kind] = torch.tensor(values)
        return weights

    @torch.no_grad()
    def _score(self, model, dataset, config=None):
        device = self.teacher.module.DEVICE
        truth, prediction = [], []
        loss = 0.
        cfg = config or self.config
        if model is not None:
            model.eval()
        for path in dataset['paths']:
            row = self.row(path)
            features = row['features'].to(device) if model is None else model(
                row['x'].to(device), row['centers'].to(device), row['tail'].to(device))
            logits = head_logits(self.head, features)
            truth.append(row['y'].numpy())
            prediction.append(original_predictions(logits).cpu().numpy())
            if model is not None:
                value, _ = legacy_loss(logits, {k: v.to(device) for k, v in row['teacher'].items()},
                    row['y'].to(device), features, row['features'].to(device), self.head.feature_scale, cfg)
                loss += float(value)
        return np.concatenate(truth), np.concatenate(prediction), loss / len(dataset['paths'])

    def fit(self, config=None):
        self._open_training()
        cfg = config or self.config
        cfg.validate()
        # Sampling/normalization do not change across optional search trials.
        locked = ('seed', 'center_batch_size', 'train_cases_per_step', 'validation_cases_per_step', 'test_cases_per_step')
        if any(getattr(cfg, k) != getattr(self.config, k) for k in locked):
            raise ValueError('Keep data/sampling settings fixed across KD trials')
        train, validation = self.dataset('train'), self.dataset('validation')
        weights = self._weights(train)
        held = np.concatenate([self.row(p)['y'].numpy() for p in validation['paths']])
        if set(held[:, 0]) != {0, 1} or any(not (held[:, j] > 0).any() for j in (1, 2)):
            raise ValueError('Validation requires both Boolean classes and known typed positives')
        device = self.teacher.module.DEVICE
        torch.manual_seed(cfg.seed)
        model = GevisStudent(self.queries, self.patch_length, 48, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        history, best, best_key = [], None, None
        stale = updates = 0
        started = time.perf_counter()
        for epoch in range(cfg.epochs):
            if history and time.perf_counter() - started >= cfg.max_seconds:
                break
            model.train()
            order = np.random.default_rng(cfg.seed + epoch).permutation(len(train['paths']))
            total = 0.
            for start in range(0, len(order), cfg.batch_cases):
                batch = order[start:start + cfg.batch_cases]
                optimizer.zero_grad(set_to_none=True)
                for index in batch:
                    row = self.row(train['paths'][int(index)])
                    features = model(row['x'].to(device), row['centers'].to(device), row['tail'].to(device))
                    loss, _ = legacy_loss(head_logits(self.head, features),
                        {k: v.to(device) for k, v in row['teacher'].items()}, row['y'].to(device),
                        features, row['features'].to(device), self.head.feature_scale, cfg, weights)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite Student loss')
                    (loss / len(batch)).backward()
                    total += float(loss.detach()) / len(batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                updates += 1
            truth, predicted, val_loss = self._score(model, validation, cfg)
            score, key = legacy_validation_selection(truth, predicted, val_loss)
            history.append(dict(epoch=epoch + 1, train_loss=total / max(1, math.ceil(len(order) / cfg.batch_cases)),
                validation_loss=val_loss, **{k + '_f1': v for k, v in d.typed_f1(score).items()}))
            if best_key is None or key > best_key:
                best_key = key
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                selected_epoch, selected_updates = epoch + 1, updates
                stale = 0
            else:
                stale += 1
            if self.progress:
                self.progress(f'Student control48 epoch {epoch + 1}: original-five Validation {d.typed_f1(score)}', flush=True)
            if stale >= cfg.patience:
                break
        model.load_state_dict(best, strict=True)
        self.student = model.requires_grad_(False).eval()
        self.training = dict(objective='legacy', selected_epoch=selected_epoch, optimizer_updates=selected_updates,
            attempted_updates=updates, history=history, selection_split='validation', original_head_frozen=True,
            teacher_queries_frozen=True, epoch_selection='original-five legacy Validation before final remapping',
            config=cfg.__dict__.copy())
        self.validation = self.validation_timing = None
        self._parity_passed = False
        self.assert_preserved()
        return pd.DataFrame(history)

    def adopt_student(self, student, training_record, original_head):
        """Explicitly merge a verified live control48 without retraining its Head.

        The caller supplies the original live fit record and original Head;
        shape alone cannot establish that a Student used the intended Teacher.
        """
        from .runtime import head_digest
        self._open_training()
        if (getattr(student, 'width', None) != 48 or len(student.temporal.layers) != 2
                or student.patch_length != self.patch_length or training_record.get('objective') != 'legacy'):
            raise ValueError('Verified live control48 legacy record required')
        if head_digest(original_head) != head_digest(self.head):
            raise ValueError('Live Student original Head differs from selected Teacher')
        if not torch.equal(student.question_vectors.detach().cpu(), self.queries):
            raise ValueError('Live Student Teacher question vectors differ')
        model = GevisStudent(self.queries, self.patch_length, 48, 2).to(self.teacher.module.DEVICE)
        model.load_state_dict(student.state_dict(), strict=True)
        self.student = model.requires_grad_(False).eval()
        self.training = deepcopy(training_record)
        self.training.update(objective='legacy', config=self.config.__dict__.copy(),
            adopted_live_student=True, original_head_frozen=True, teacher_queries_frozen=True)
        self.validation = self.validation_timing = None
        self._parity_passed = False
        self.assert_preserved()
        return self.student

    def report(self, split='validation'):
        if split not in ('validation', 'test') or self.student is None:
            raise ValueError('Fitted Student and Validation/Test split required')
        if split == 'test' and self.selection is None:
            raise ValueError('Freeze Validation selection before Test')
        dataset = self.dataset(split)
        tables, details, support = [], {}, {}
        for name, model in (('teacher', None), ('control48', self.student)):
            truth, prediction, _ = self._score(model, dataset)
            table, support[name] = metrics(truth, prediction, model=name, split=split)
            tables.append(table)
            details[name] = dict(truth=truth, original_five_prediction=prediction, final_prediction=remap(prediction))
        result = dict(metrics=pd.concat(tables, ignore_index=True), support=support,
            coverage=dataset['coverage'], split=split, details=details)
        if split == 'validation':
            if self.test_opened:
                raise ValueError('Validation selection closed after Test opens')
            self.validation = result
        return result

    def parity_report(self, windows=2):
        self._open_training()
        if self.student is None:
            raise ValueError('Fit the Student first')
        dataset = self.dataset('validation')
        rows = []
        for case_index, case in enumerate(dataset['cases'][:windows]):
            expected = {name: [] for name in ('teacher', 'control48')}
            for path in dataset['paths']:
                row = self.row(path)
                if row['case_index'] != case_index:
                    continue
                for name, model in (('teacher', None), ('control48', self.student)):
                    with torch.no_grad():
                        features = row['features'].to(self.teacher.module.DEVICE) if model is None else model(
                            row['x'].to(self.teacher.module.DEVICE), row['centers'].to(self.teacher.module.DEVICE),
                            row['tail'].to(self.teacher.module.DEVICE))
                        expected[name].append(original_predictions(head_logits(self.head, features)).cpu().numpy())
            for name, model in (('teacher', None), ('control48', self.student)):
                reply = Session(self.teacher, model, center_batch_size=self.config.center_batch_size).process_window(
                    **self.request(case, raw=False, selected_centers=True))
                if reply['status'] != 'OK':
                    raise ValueError('Supported parity request withheld')
                by = {(a['center_sensor'], a['answer_type']): a for a in reply['answers']}
                observed = np.array([[by[c, k]['prediction'] if by[c, k]['evaluated'] else 0
                    for k in fa.KINDS] for c in case['centers']])
                passed = np.array_equal(remap(np.concatenate(expected[name])), observed)
                rows.append(dict(model=name, window=case_index, prediction_parity=passed))
        frame = pd.DataFrame(rows)
        self._parity_passed = not frame.empty and bool(frame.prediction_parity.all())
        if not self._parity_passed:
            raise ValueError('Cached/single-request prediction parity failed; do not time this path')
        return frame

    def request(self, case, *, raw=True, selected_centers=False):
        values = d.case_current(case)
        meta = self.teacher.encoders[case['step']][1]
        if raw:
            values = (values * np.asarray(meta['std']) + np.asarray(meta['mean'])).astype(np.float32)
        return dict(current=values, step=case['step'], recipe_key=case['recipe_key'], loop_key=case['loop_key'],
            offset_samples=int(case['offset_samples']), lineage_id=case['lineage_id'], run_id=case['run_id'],
            sensor_columns=case['sensor_columns'], standardized=not raw,
            centers=case['centers'] if selected_centers else None)

    def benchmark(self, split='validation', *, merged=None):
        if split not in ('validation', 'test') or not self._parity_passed:
            raise ValueError('Validation runtime parity must pass before timing')
        if split == 'test' and not self.test_done:
            raise ValueError('Evaluate the sealed Test before its timing')
        if split == 'validation' and self.test_opened:
            raise ValueError('Validation timing closed after Test opens')
        cases = self.dataset(split)['cases']
        # Cases are fixed before timing; no support scan/encoding cache in the timer.
        indices = np.linspace(0, len(cases) - 1, min(self.config.benchmark_windows, len(cases)), dtype=int)
        if merged is not None and getattr(merged, 'teacher_binding', None) != self.binding:
            raise ValueError('Merged timing package has a different Teacher/input binding')
        sessions = {'teacher': Session(self.teacher, center_batch_size=self.config.center_batch_size),
                    'control48': (Session(merged, center_batch_size=self.config.center_batch_size) if merged is not None
                        else Session(self.teacher, self.student, center_batch_size=self.config.center_batch_size))}
        rows = []
        device = self.teacher.module.DEVICE
        for window, index in enumerate(indices):
            request = self.request(cases[int(index)], raw=True)
            for repeat in range(self.config.warmup + self.config.repeats):
                names = ('teacher', 'control48') if (window + repeat) % 2 == 0 else ('control48', 'teacher')
                for name in names:
                    sync(device); start = time.perf_counter()
                    reply = sessions[name].process_window(**request)
                    sync(device); elapsed = (time.perf_counter() - start) * 1000
                    if reply['status'] != 'OK':
                        raise ValueError('Timing request withheld; do not count as an answered latency')
                    if repeat >= self.config.warmup:
                        rows.append(dict(model=name, split=split, window=window, repeat=repeat,
                            wall_ms=elapsed, sensors=len(request['sensor_columns']),
                            evaluated_answers=sum(a['evaluated'] for a in reply['answers'])))
        frame = pd.DataFrame(rows)
        summary = pd.DataFrame([dict(model=name, split=split, mean_ms=g.wall_ms.mean(),
            p95_ms=np.percentile(g.wall_ms, 95), requests=len(g), windows=g.window.nunique(),
            device=str(device), merged_package_loaded=bool(merged is not None and name == 'control48'),
            scope='whole Window: normalization/reference/GEVIS/original Head/final argmax mapping')
            for name, g in frame.groupby('model', sort=False)])
        result = dict(summary=summary, requests=frame)
        if split == 'validation':
            self.validation_timing = result
        self.assert_preserved()
        return result

    def freeze_selection(self):
        self._open_training()
        if self.validation is None or self.validation_timing is None or not self._parity_passed:
            raise ValueError('Validation F1, runtime parity and actual timing are required')
        table = self.validation['metrics']
        f1 = table.loc[table.model.eq('control48'), 'f1']
        timing = self.validation_timing['summary'].loc[lambda t: t.model.eq('control48')].iloc[0]
        support = self.validation['support']['control48']
        complete = (set(support['boolean_classes']) == {0, 1}
            and all(support['source_choice_support'].values()) and all(support['ordinal_support'].values()))
        self.selection = dict(model='control48', selected_on='validation',
            quality_target=self.config.quality_target, coverage_complete=bool(complete),
            quality_pass=bool(complete and np.isfinite(f1).all() and f1.ge(self.config.quality_target).all()),
            mean_ms=float(timing.mean_ms), p95_ms=float(timing.p95_ms),
            config=deepcopy(self.training['config']), test_used=False)
        return deepcopy(self.selection)

    def evaluate_test(self):
        if self.selection is None or self.test_done:
            raise ValueError('Freeze Validation selection; evaluate Test once without reselection')
        self.test_opened = True
        result = self.report('test')
        self.test_done = True
        self.assert_preserved()
        return result

    def save(self):
        if self.selection is None:
            raise ValueError('Freeze Validation selection before saving merged deployment')
        self.assert_preserved()
        effective = Config(**self.training['config'])
        return save_merged(self.root, self.teacher, self.student, config=effective, training=self.training,
            reports=dict(selection=self.selection, validation=self.validation['metrics'].to_dict('records'),
                         validation_timing=self.validation_timing['summary'].to_dict('records')))

    def merged_parity_report(self, merged, windows=2):
        """Check the saved/loadable package against the selected in-memory model."""
        if self.student is None:
            raise ValueError('Selected Student required')
        if getattr(merged, 'teacher_binding', None) != self.binding:
            raise ValueError('Merged package differs from selected Teacher/input binding')
        original = Session(self.teacher, self.student, center_batch_size=self.config.center_batch_size)
        restored = Session(merged, center_batch_size=self.config.center_batch_size)
        rows = []
        for index, case in enumerate(self.dataset('validation')['cases'][:windows]):
            request = self.request(case, raw=True, selected_centers=True)
            expected = original.process_window(**request)
            found = restored.process_window(**request)
            decisions = lambda reply: [(a['center_sensor'], a['answer_type'], a['prediction'],
                a.get('original_five_prediction'), a['evaluated']) for a in reply['answers']]
            passed = expected['status'] == found['status'] == 'OK' and decisions(expected) == decisions(found)
            rows.append(dict(window=index, merged_prediction_parity=passed))
        result = pd.DataFrame(rows)
        if result.empty or not result.merged_prediction_parity.all():
            raise ValueError('Merged package prediction parity failed')
        return result
