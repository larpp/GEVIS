"""Frozen GEVIS + independently fitted ALD softmax/grouped/sigmoid heads.

Features and Run partitions are shared across variants. Validation alone selects
epochs. Packages live outside the existing CVD/ALD completion pointers.
"""
from dataclasses import asdict, dataclass
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import current_input as current, decision_training as d, fusion_answers as fa
from . import ald_preprocessing as preprocessing
from .feature_store import FeatureRows

SCHEMA = 'gevis.domain_head.v1'
DEFAULT_GROUPS = ((1, 3), (2,), (4,), (5,))


class PreparationSupportError(ValueError):
    def __init__(self, message, summary, audit):
        super().__init__(message)
        self.summary = summary
        self.audit = audit


@dataclass(frozen=True)
class ChoiceConfig:
    mode: str = 'softmax'
    groups: tuple = DEFAULT_GROUPS
    sigmoid_threshold: float = .5

    def validate(self):
        if self.mode not in ('softmax', 'grouped', 'sigmoid'):
            raise ValueError('Choose softmax, grouped or sigmoid')
        flat = [v for group in self.groups for v in group]
        if sorted(flat) != list(range(1, 6)) or any(not group for group in self.groups):
            raise ValueError('CHOICE groups must partition positive classes 1..5 exactly once')
        if not math.isfinite(self.sigmoid_threshold) or not 0 < self.sigmoid_threshold < 1:
            raise ValueError('Fixed sigmoid threshold must lie in (0,1)')


def group_targets(values, config):
    result = values.clone() if isinstance(values, torch.Tensor) else np.array(values, copy=True)
    for index, group in enumerate(config.groups, 1):
        for value in group:
            result[values == value] = index
    return result


class ExperimentHead(fa.ContextReadout):
    def __init__(self, dimension, width, choice=ChoiceConfig()):
        choice.validate()
        super().__init__(dimension, width)
        self.choice = choice

    def choice_logits(self, logits):
        if self.choice.mode == 'grouped':
            return torch.stack([torch.logsumexp(logits[:, list(g)], -1) for g in self.choice.groups], -1)
        return logits[:, 1:]

    def choice_probabilities(self, logits):
        scores = self.choice_logits(logits)
        return scores.sigmoid() if self.choice.mode == 'sigmoid' else scores.softmax(-1)

    @property
    def choice_labels(self):
        labels = fa.ANSWER_CRITERIA['CHOICE']
        return (tuple(' / '.join(labels[i] for i in g) for g in self.choice.groups)
                if self.choice.mode == 'grouped' else labels[1:])


def _loss(logits, target, kind, head, weights):
    known = target >= (0 if kind == 'BOOLEAN' else 1)
    if not known.any():
        return logits.sum() * 0
    logits, target = logits[known], target[known]
    if kind != 'CHOICE':
        return F.cross_entropy(logits, target, weight=weights[kind], reduction='sum')
    scores = head.choice_logits(logits)
    if head.choice.mode == 'grouped':
        target = group_targets(target, head.choice)
    if head.choice.mode == 'sigmoid':
        truth = F.one_hot(target - 1, 5).to(scores.dtype)
        active = head.allowed_CHOICE[1:]
        # Single-label synthetic truth trains independent one-vs-rest detectors.
        # No synthetic overlap label is invented for similar-looking families.
        return F.binary_cross_entropy_with_logits(scores[:, active], truth[:, active],
            pos_weight=weights['sigmoid'][active], reduction='sum') / active.sum()
    return F.cross_entropy(scores, target - 1, weight=weights[kind], reduction='sum')


def _predictions(head, logits):
    gate = logits['BOOLEAN'].argmax(-1)
    choice = head.choice_probabilities(logits['CHOICE']).argmax(-1) + 1
    ordinal = logits['ORDINAL'].clone()
    ordinal[:, 0] = -torch.inf
    return torch.stack([gate, torch.where(gate == 1, choice, 0),
                        torch.where(gate == 1, ordinal.argmax(-1), 0)], -1)


def fit_head(train, validation, config, choice=ChoiceConfig(), *, device='cpu', domain='PEALD', progress=print):
    config.validate(); choice.validate(); torch.manual_seed(config.seed)
    if {c['lineage_id'] for c in train} & {c['lineage_id'] for c in validation}:
        raise ValueError('ALD TRAIN/Validation lineages overlap')
    x = FeatureRows(train, config.max_feature_mb * 1024**2)
    vx = FeatureRows(validation, config.max_feature_mb * 1024**2)
    y = torch.as_tensor(np.concatenate([c['targets'] for c in train]), dtype=torch.long)
    vy = torch.as_tensor(np.concatenate([c['targets'] for c in validation]), dtype=torch.long)
    if x.shape[1:] != vx.shape[1:] or len(y) != x.shape[0] or len(vy) != vx.shape[0]:
        raise ValueError('ALD features/targets do not match')
    head = ExperimentHead(x.shape[-1], config.width, choice).to(device)
    mean, scale = x.moments(config.batch_size)
    head.feature_mean.copy_(mean.to(device)); head.feature_scale.copy_(scale.clamp_min(1e-5).to(device))
    positive = np.flatnonzero(y[:, 0].numpy() == 1)
    negative = np.flatnonzero(y[:, 0].numpy() == 0)
    if not len(positive) or not len(negative):
        raise ValueError('ALD head needs both normal and positive TRAIN examples')
    weights = {'BOOLEAN': None}
    for i, kind in enumerate(fa.KINDS):
        labels = y[:, i] if kind == 'BOOLEAN' else y[y[:, 0] == 1, i]
        labels = labels[labels >= 0]
        counts = torch.bincount(labels, minlength=len(fa.ANSWER_CRITERIA[kind])).float()
        head.get_buffer('allowed_' + kind).copy_((counts > 0).to(device))
        held = vy[:, i] if kind == 'BOOLEAN' else vy[vy[:, 0] == 1, i]
        held = held[held > ( -1 if kind == 'BOOLEAN' else 0)]
        if not len(held) or not (counts[held] > 0).all():
            raise ValueError(f'Validation {kind} has missing TRAIN class support')
        if kind == 'BOOLEAN':
            continue
        if kind == 'CHOICE':
            active = counts[1:] > 0
            weights['sigmoid'] = ((counts[1:].sum() - counts[1:]) / counts[1:].clamp_min(1)).to(device)
            if choice.mode == 'grouped':
                counts = torch.stack([counts[list(g)].sum() for g in choice.groups])
            else:
                counts = counts[1:]
        active = counts > 0
        w = torch.zeros_like(counts)
        w[active] = counts.sum() / (active.sum() * counts[active])
        weights[kind] = w.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history = []; best = None; best_key = None; stale = 0; updates = 0
    started = time.perf_counter()

    def score(features, targets):
        predictions = []; total_loss = 0.
        with torch.no_grad():
            for start in range(0, len(targets), config.batch_size):
                ids = np.arange(start, min(len(targets), start + config.batch_size))
                batch = features.take(ids).to(device)
                target = targets[ids].to(device).clone()
                target[target[:, 0] == 0, 1:] = -1
                logits = {k: head(batch[:, i], k) for i, k in enumerate(fa.KINDS)}
                total_loss += sum(float(_loss(logits[k], target[:, i], k, head, weights)) for i, k in enumerate(fa.KINDS))
                predictions.append(_predictions(head, logits).cpu().numpy())
        truth = targets.numpy().copy()
        if choice.mode == 'grouped':
            truth[:, 1] = group_targets(truth[:, 1], choice)
        return d._scores(truth, np.concatenate(predictions)), total_loss / max(1, len(targets))

    for epoch in range(config.epochs):
        if history and time.perf_counter() - started >= config.max_seconds:
            break
        rng = np.random.default_rng(config.seed + epoch)
        draws = max(len(positive), len(negative))
        pos = np.resize(rng.permutation(positive), draws); neg = np.resize(rng.permutation(negative), draws)
        half = max(1, config.batch_size // 2)
        head.train(); loss_sum = 0.; rows = 0
        for start in range(0, draws, half):
            ids = np.r_[pos[start:start + half], neg[start:start + half]]; rng.shuffle(ids)
            batch = x.take(ids).to(device); target = y[ids].to(device).clone()
            target[target[:, 0] == 0, 1:] = -1
            optimizer.zero_grad(set_to_none=True)
            loss = sum(_loss(head(batch[:, i], k), target[:, i], k, head, weights)
                       for i, k in enumerate(fa.KINDS)) / (3 * len(ids))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite ALD head loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); optimizer.step()
            updates += 1; loss_sum += float(loss.detach()) * len(ids); rows += len(ids)
        head.eval(); metrics, val_loss = score(vx, vy)
        key = d.selection_key(metrics, val_loss)
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / rows, validation_loss=val_loss, **metrics))
        if best_key is None or key > best_key:
            best_key = key; stale = 0
            best = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            selected_epoch = epoch + 1; selected_updates = updates
        else:
            stale += 1
        if progress:
            progress(f'{domain} {choice.mode} epoch {epoch + 1}: Validation F1={d.typed_f1(metrics)}', flush=True)
        if stale >= config.patience:
            break
    head.load_state_dict(best, strict=True); head.requires_grad_(False).eval()
    return head, dict(history=history, selected_epoch=selected_epoch, optimizer_updates=selected_updates,
        attempted_optimizer_updates=updates, fit_seconds=time.perf_counter() - started,
        warm_started=False, feature_normalization='ALD_TRAIN', choice=asdict(choice),
        selection_policy=d.SELECTION_POLICY, target_domain=domain)


class Session:
    """Native inference with the packaged ALD preprocessing and CHOICE decoder."""
    def __init__(self, runtime):
        self.runtime = runtime

    @torch.no_grad()
    def process_window(self, *, current, step, recipe_key, offset_samples, lineage_id,
                       loop_key='', centers=None, sensor_columns=None, standardized=False, run_id=None):
        r = self.runtime
        def withheld(reason):
            return dict(status='NOT_EVALUATED', reason=reason, answers=[])
        if step not in r.encoders:
            return withheld('unsupported_native_context')
        meta = r.encoders[step][1]; names = meta['sensor_columns']
        if sensor_columns is not None and list(sensor_columns) != names:
            return withheld('incomplete_observed_sensor_grid')
        if not isinstance(lineage_id, str) or not lineage_id:
            return withheld('source_lineage_required')
        if lineage_id in set(r.references['basis_lineages']):
            return withheld('reference_basis_lineage_not_independent')
        record = r.reference_bank.get(d._key(step, recipe_key, loop_key, offset_samples))
        if record is None:
            return withheld('native_normal_reference_unobserved')
        targets = getattr(r, 'target_sensors', {}).get(step, [s for s in names if not s.startswith('ValveAct_')])
        chosen = list(targets) if centers is None else list(centers)
        if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(names):
            raise ValueError('Unique known ALD centers required')
        if set(chosen) - set(targets):
            raise ValueError('Context-only sensor cannot be selected as a prediction target')
        try:
            if not standardized:
                frame = current if isinstance(current, pd.DataFrame) else pd.DataFrame(current, columns=names)
                values = preprocessing.transform_live(frame, meta)
            else:
                values = np.asarray(current, np.float32)
            if values.shape != (meta['window_length'], len(names)) or not np.isfinite(values).all():
                raise ValueError('Finite complete native ALD window required')
        except (ValueError, TypeError) as error:
            return withheld(str(error))
        head = r.module.FUSION_ANSWER_HEAD; decoded = {}
        def infer(batch, kinds):
            features, tasks = d._window_features(r, values, record, step, batch, offset_samples,
                r.config.center_batch_size, answer_types=kinds)
            features = features.reshape(-1, features.shape[-1]).to(r.module.DEVICE)
            for i, (center, kind) in enumerate(tasks):
                logits = head(features[i:i + 1], kind)
                if kind == 'CHOICE':
                    probs = head.choice_probabilities(logits)[0].cpu()
                    labels = head.choice_labels
                    prediction = int(probs.argmax()) + 1
                else:
                    if kind == 'ORDINAL':
                        logits[:, 0] = -torch.inf
                    probs = logits.softmax(-1)[0].cpu(); labels = fa.ANSWER_CRITERIA[kind]
                    prediction = int(probs.argmax())
                answer = dict(center_sensor=center, answer_type=kind, prediction=prediction,
                    answer=labels[prediction - 1] if kind == 'CHOICE' else labels[prediction],
                    probabilities=dict(zip(labels, map(float, probs))), evaluated=True, status='EVALUATED')
                if kind == 'CHOICE':
                    answer['choice_mode'] = head.choice.mode
                    answer['probability_semantics'] = ('independent_sigmoid_scores' if head.choice.mode == 'sigmoid'
                                                        else 'conditional_class_softmax')
                    answer['detected_types'] = ([label for label, prob in zip(labels, probs)
                        if prob >= head.choice.sigmoid_threshold] if head.choice.mode == 'sigmoid'
                        else [answer['answer']])
                    if head.choice.mode == 'sigmoid':
                        answer['top1_type'] = answer['answer']
                        answer['prediction_semantics'] = 'top1_for_comparison'
                        answer['answer'] = ' / '.join(answer['detected_types']) or None
                decoded[center, kind] = answer
        infer(chosen, ('BOOLEAN',))
        positive = [c for c in chosen if decoded[c, 'BOOLEAN']['prediction'] == 1]
        if positive:
            infer(positive, ('CHOICE', 'ORDINAL'))
        answers = [decoded.get((c, k), dict(center_sensor=c, answer_type=k, prediction=None,
            answer=None, probabilities=None, evaluated=False, status='NOT_APPLICABLE', reason='BOOLEAN_No'))
            for c in chosen for k in fa.KINDS]
        return dict(status='OK', reason=None, answers=answers, run_id=run_id, conditional_answering=True)


def _codes():
    return dict(d._codes(), **{Path(p).name: d.p.digest(p) for p in
        (__file__, preprocessing.__file__, Path(__file__).parents[1] / 'gevis.py',
         Path(__file__).parents[1] / 'gevis_runtime.py', Path(__file__).with_name('ald_input_store.py'))})


def save_head(root, runtime, head, training, choice, preparation_policy):
    identity = dict(schema=SCHEMA, domain=runtime.domain, config=asdict(runtime.config), choice=asdict(choice),
        preprocessing=asdict(preparation_policy) if runtime.domain == 'PEALD' else None,
        preparation=runtime.prepared.identity,
        metadata=runtime.prepared.metadata, source_steps=runtime.source_steps,
        target_sensors=runtime.target_sensors, context_sensors=runtime.context_sensors,
        source=current.source_identity(runtime), codes=_codes(), references=d._hash(runtime.references))
    path = Path(root) / 'artifacts/gevis_heads' / runtime.domain.lower() / d._hash(identity)[:24] / 'answer_head.pt'
    payload = dict(schema=SCHEMA, identity=identity, references=runtime.references,
        dimension=head.feature_mean.numel(), training=training,
        state={k: v.detach().cpu() for k, v in head.state_dict().items()})
    payload['content_sha256'] = current.payload_digest(payload)
    current._atomic_torch(payload, path)
    runtime.module.FUSION_ANSWER_PATH = path
    return path


def load_head(root, checkpoint, *, gevis=None, allow_legacy_cvd=False, allow_legacy_ald=False):
    """Restore an explicit experiment; no automatic fitting or model selection."""
    from pecvd.gevis import load_gevis
    root = Path(root).resolve(); path = Path(checkpoint)
    path = (path if path.is_absolute() else root / path).resolve()
    path.relative_to(root / 'artifacts/gevis_heads')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('content_sha256') != current.payload_digest(payload):
        raise ValueError('ALD experiment package schema/content differs')
    identity = payload['identity']; domain = d._domain(identity['domain'])
    path.relative_to(root / 'artifacts/gevis_heads' / domain.lower())
    if path.parent.name != d._hash(identity)[:24]:
        raise ValueError('Experiment checkpoint namespace differs')
    actual_codes = _codes()
    changed = {name for name in set(identity['codes']) | set(actual_codes)
               if identity['codes'].get(name) != actual_codes.get(name)}
    if changed:
        # Explicit inference-only compatibility: never ignore backbone, feature,
        # reference, or CVD preparation changes. Payload and source-weight checks
        # below remain mandatory, as does strict state_dict loading.
        permitted = {'ald_preprocessing.py', 'ald_input_store.py', 'ald_head_experiments.py'}
        cvd_compatible = allow_legacy_cvd and domain == 'PECVD' and changed <= permitted
        # ALD preprocessing/input-store differences are never accepted here.
        ald_compatible = (allow_legacy_ald and domain == 'PEALD'
                          and changed <= {'ald_head_experiments.py'})
        if not (cvd_compatible or ald_compatible):
            raise ValueError(f'Experiment source differs: {sorted(changed)}')
        import warnings
        warnings.warn(f'Loading legacy {domain} head with current inference code; historical '
                      'prediction parity is not verified. Changed files: '
                      + ', '.join(sorted(changed)), RuntimeWarning, stacklevel=2)
    if identity['references'] != d._hash(payload['references']):
        raise ValueError('ALD experiment reference identity differs')
    config = d.Config(**identity['config']); config.validate()
    choice = ChoiceConfig(**identity['choice']); choice.validate()
    if domain == 'PEALD':
        preprocessing.PreprocessingConfig(**identity['preprocessing']).validate()
    gevis = gevis or load_gevis(root, encoder_steps=sorted(set(identity['source_steps'].values())))
    if set(identity['source_steps'].values()) - set(gevis.encoders):
        raise ValueError('Experiment requires an encoder missing from GEVIS')
    prepared = SimpleNamespace(metadata=identity['metadata'], identity=identity['preparation'],
        split_runs=identity['preparation']['split_runs'])
    source = SimpleNamespace(module=gevis.module, encoders=gevis.encoders)
    encoders = {step: (gevis.encoders[identity['source_steps'][step]][0], meta)
                for step, meta in identity['metadata'].items()}
    target_sensors = identity.get('target_sensors')
    context_sensors = identity.get('context_sensors')
    if target_sensors is None or context_sensors is None:
        if not (allow_legacy_cvd and domain == 'PECVD'):
            raise ValueError('Checkpoint lacks explicit sensor roles')
        # Match Session's original default for checkpoints predating role metadata.
        target_sensors = {s: [n for n in m['sensor_columns'] if not n.startswith('ValveAct_')]
                          for s, m in identity['metadata'].items()}
        context_sensors = {s: [n for n in m['sensor_columns'] if n.startswith('ValveAct_')]
                           for s, m in identity['metadata'].items()}
    module = SimpleNamespace(**gevis.module.__dict__); module.STEP_ENCODERS = encoders
    runtime = SimpleNamespace(module=module, encoders=encoders, prepared=prepared, source_runtime=source,
        source_steps=identity['source_steps'], domain=domain, config=config,
        target_sensors=target_sensors, context_sensors=context_sensors,
        references=payload['references'], reference_bank=d._bank(payload['references']))
    if current.source_identity(runtime) != identity['source']:
        raise ValueError('Frozen GEVIS differs from ALD experiment training')
    head = ExperimentHead(payload['dimension'], config.width, choice)
    head.load_state_dict(payload['state'], strict=True)
    runtime.module.FUSION_ANSWER_HEAD = head.requires_grad_(False).eval().to(runtime.module.DEVICE)
    runtime.module.FUSION_ANSWER_TRAINING = payload['training']; runtime.module.FUSION_ANSWER_PATH = path
    return runtime


def evaluate_encoded(head, cases, config, device):
    """Same heldout examples for every variant; grouped metrics share one taxonomy."""
    features = FeatureRows(cases, config.max_feature_mb * 1024**2)
    truth = np.concatenate([c['targets'] for c in cases]); predictions = []; probabilities = []; boolean_scores = []
    with torch.no_grad():
        for start in range(0, len(truth), config.batch_size):
            batch = features.take(np.arange(start, min(len(truth), start + config.batch_size))).to(device)
            logits = {k: head(batch[:, i], k) for i, k in enumerate(fa.KINDS)}
            predictions.append(_predictions(head, logits).cpu().numpy())
            probabilities.append(head.choice_probabilities(logits['CHOICE']).cpu().numpy())
            boolean_scores.append(logits['BOOLEAN'].softmax(-1)[:, 1].cpu().numpy())
    predicted = np.concatenate(predictions); probs = np.concatenate(probabilities)
    native_truth = truth.copy()
    if head.choice.mode == 'grouped':
        native_truth[:, 1] = group_targets(native_truth[:, 1], head.choice)
    metrics = d._scores(native_truth, predicted)
    common_truth = truth.copy(); common_truth[:, 1] = group_targets(truth[:, 1], head.choice)
    common_prediction = predicted.copy()
    if head.choice.mode != 'grouped':
        common_prediction[:, 1] = group_targets(predicted[:, 1], head.choice)
    grouped = d._scores(common_truth, common_prediction)
    summary = dict(mode=head.choice.mode, questions=len(truth),
        boolean_f1=metrics['boolean_positive_f1'], normal_fpr=metrics['normal_false_positive_rate'],
        choice_exact_macro_f1=metrics['macro_f1']['CHOICE'] if head.choice.mode != 'grouped' else np.nan,
        choice_common_group_macro_f1=grouped['macro_f1']['CHOICE'],
        ordinal_macro_f1=metrics['macro_f1']['ORDINAL'])
    details = pd.DataFrame([dict(step=c['step'], run_id=c['run_id'], case_id=c['pair_id'],
        center_sensor=name, offset_samples=c['offset_samples'], injected=c['injected'],
        intervention_family=c['kind']) for c in cases for name in c['centers']])
    for name, values in dict(boolean_truth=truth[:, 0], boolean_prediction=predicted[:, 0],
        choice_truth=truth[:, 1], choice_prediction=predicted[:, 1],
        choice_group_truth=common_truth[:, 1], choice_group_prediction=common_prediction[:, 1],
        ordinal_truth=truth[:, 2], ordinal_prediction=predicted[:, 2]).items():
        details[name] = values
    details['boolean_answer'] = [fa.ANSWER_CRITERIA['BOOLEAN'][i] for i in predicted[:, 0]]
    details['boolean_yes_score'] = np.concatenate(boolean_scores)
    details['choice_top1'] = [head.choice_labels[i - 1] if gate else None for gate, i in predicted[:, :2]]
    details['detected_types'] = [([label for label, value in zip(head.choice_labels, score)
        if value >= head.choice.sigmoid_threshold] if head.choice.mode == 'sigmoid'
        else [head.choice_labels[index - 1]]) if gate else []
        for gate, index, score in zip(predicted[:, 0], predicted[:, 1], probs)]
    details['choice_answer'] = [' / '.join(items) or None for items in details['detected_types']]
    details['ordinal_answer'] = [fa.ANSWER_CRITERIA['ORDINAL'][i] if gate else None
                                 for gate, i in predicted[:, [0, 2]]]
    known = truth[:, 1] > 0
    if head.choice.mode == 'sigmoid' and known.any():
        found = probs[known] >= head.choice.sigmoid_threshold
        found[predicted[known, 0] != 1] = False
        target = np.eye(5, dtype=bool)[truth[known, 1] - 1]
        tp = int((found & target).sum()); fp = int((found & ~target).sum()); fn = int((~found & target).sum())
        summary.update(sigmoid_set_precision=tp / max(1, tp + fp), sigmoid_set_recall=tp / max(1, tp + fn),
            sigmoid_set_f1=2 * tp / max(1, 2 * tp + fp + fn),
            sigmoid_no_type_rate=float((~found.any(1)).mean()),
            sigmoid_multiple_type_rate=float((found.sum(1) > 1).mean()))
    confusion = pd.crosstab(details.loc[known, 'choice_group_truth'],
                            details.loc[known, 'choice_group_prediction'], dropna=False)
    sensor_rows = []
    for (step, name), group in details.groupby(['step', 'center_sensor'], sort=False):
        true = group.boolean_truth.eq(1); found = group.boolean_prediction.eq(1)
        tp = int((true & found).sum()); fp = int((~true & found).sum()); fn = int((true & ~found).sum())
        sensor_rows.append(dict(step=step, sensor=name, windows=len(group), true_positive=tp,
            false_positive=fp, missed_change=fn, boolean_f1=2 * tp / max(1, 2 * tp + fp + fn)))
    return dict(summary=summary, details=details, common_group_confusion=confusion,
                sensor_summary=pd.DataFrame(sensor_rows))


def run_comparison(root, gevis, *, encoder_step='DEP.', prepared=None,
                   preparation_policy=preprocessing.PreprocessingConfig(),
                   native_config=preprocessing.NativeConfig(), config=d.Config(),
                   choices=None, steps=None, domain='PEALD', progress=print):
    """Prepare once, cache frozen features once, fit three ALD-only heads, Test once.

    This call explicitly trains. Model files use a separate identity directory;
    existing STEP1/2/3 weights and pointers are not changed.
    """
    root = Path(root).resolve(); domain = d._domain(domain)
    config.validate(); preparation_policy.validate(); native_config.validate()
    if domain == 'PEALD' and encoder_step not in gevis.encoders:
        raise ValueError('Load the selected GEVIS encoder before ALD experiments')
    choices = tuple(ChoiceConfig(mode=m) for m in ('softmax', 'grouped', 'sigmoid')) if choices is None else tuple(choices)
    if not choices or len({c.mode for c in choices}) != len(choices):
        raise ValueError('Choose unique experiment modes')
    for choice in choices:
        choice.validate()
    if len({d._hash(c.groups) for c in choices}) != 1:
        raise ValueError('Comparisons require the same common CHOICE grouping')
    source = SimpleNamespace(module=gevis.module, encoders=gevis.encoders)
    if prepared is None:
        if domain != 'PEALD':
            raise ValueError('Prepare the CVD input before head training')
        prepared = preprocessing.prepare(root, config=native_config,
            patch_length=gevis.encoders[encoder_step][0].patch_len,
            policy=preparation_policy, progress=progress)
    elif domain == 'PEALD' and d._hash(prepared.identity.get('preprocessing')) != d._hash(asdict(preparation_policy)):
        raise ValueError('Prepared ALD data uses a different preprocessing policy')
    if domain == 'PEALD' and ('delivery_metadata_identity' in prepared.identity or
            prepared.identity.get('reserved_delivery_groups')):
        raise ValueError('This prepared data used external Test membership; rerun prepare_ald from data_complete only')
    if (domain == 'PEALD' and 'native_config' in prepared.identity and
            d._hash(prepared.identity['native_config']) != d._hash(asdict(native_config))):
        raise ValueError('Prepared ALD Run/recipe policy differs; rerun preprocessing')
    if (domain == 'PEALD' and 'patch_length' in prepared.identity and
            prepared.identity['patch_length'] != gevis.encoders[encoder_step][0].patch_len):
        raise ValueError('Prepared ALD patch length differs from selected GEVIS encoder')
    if domain == 'PEALD':
        prepared = preprocessing.canonicalize_run_lineages(prepared, progress=progress)
    for name, checksum in prepared.identity.get('source_files', {}).items():
        if d.p.digest(root / name) != checksum:
            raise ValueError('ALD source CSV changed since preprocessing; prepare it again')
    selected = list(prepared.windows) if steps is None else list(steps)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(prepared.windows):
        raise ValueError('Select unique supported ALD native Steps')
    original_prepared = prepared; prepared = copy(original_prepared)
    prepared.windows = {s: original_prepared.windows[s] for s in selected}
    prepared.metadata = {s: original_prepared.metadata[s] for s in selected}
    prepared.identity = dict(original_prepared.identity, selected_steps=selected)
    partitions = prepared.split_runs
    if any(set(partitions[a]) & set(partitions[b]) for a, b in
           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('ALD Run splits overlap')
    roles = preprocessing.sensor_roles(prepared)
    target_sensors = {s: roles.loc[roles.step.eq(s) & roles.role.eq('prediction_target'), 'sensor'].tolist() for s in selected}
    context_sensors = {s: roles.loc[roles.step.eq(s) & roles.role.eq('context_only'), 'sensor'].tolist() for s in selected}
    if not any(target_sensors.values()):
        raise ValueError('No continuous prediction targets; remaining sensors are context-only')
    runtime = d.bind(source, prepared, domain, source_step=encoder_step); runtime.config = config
    runtime.target_sensors = target_sensors; runtime.context_sensors = context_sensors
    frozen = current.source_identity(runtime)
    runtime.references = d.fit_references(runtime, config); runtime.reference_bank = d._bank(runtime.references)
    train, coverage, exclusions = d.training_cases(runtime, runtime.references, config)
    train, coverage = _continuous_interventions(train, coverage, target_sensors)
    train, train_audit = d.supported_cases(runtime, train, runtime.references, training=True)
    validation, validation_coverage = d.controlled_cases(runtime, 'validation', config, runtime.references)
    validation, validation_coverage = _continuous_interventions(validation, validation_coverage, target_sensors)
    validation, validation_audit = d.supported_cases(runtime, validation, runtime.references)
    support_audit = pd.DataFrame(train_audit + validation_audit)
    support_summary = (support_audit.groupby(['split', 'supported', 'reason'], dropna=False)
        .agg(cases=('pair_id', 'size'), sensor_questions=('requested_targets', 'sum')).reset_index()
        if not support_audit.empty else pd.DataFrame())
    if progress:
        progress(f'Supported cases before encoding: TRAIN={len(train)}, Validation={len(validation)}', flush=True)
    if not train or not validation:
        reasons = support_summary.to_string(index=False) if not support_summary.empty else 'No candidate cases generated'
        raise PreparationSupportError('No supported TRAIN/Validation cases. Exclusion counts:\n' + reasons,
                                      support_summary, support_audit)
    cache = root / '.cache/ald-experiment-features'
    train = d.encoded_cases(runtime, train, runtime.references, cache_root=cache, source=frozen, progress=progress)
    validation = d.encoded_cases(runtime, validation, runtime.references, cache_root=cache, source=frozen, progress=progress)
    models = {}; summaries = []; reports = {}; paths = {}; histories = {}
    for choice in choices:
        head, training = fit_head(train, validation, config, choice, device=runtime.module.DEVICE, domain=domain, progress=progress)
        runtime.module.FUSION_ANSWER_HEAD = head; runtime.module.FUSION_ANSWER_TRAINING = training
        paths[choice.mode] = save_head(root, runtime, head, training, choice, preparation_policy)
        models[choice.mode] = head; histories[choice.mode] = training
    # No Test cases/features are inspected until all fitting and epoch selection end.
    test, test_coverage = d.controlled_cases(runtime, 'test', config, runtime.references)
    test, test_coverage = _continuous_interventions(test, test_coverage, target_sensors)
    test, test_audit = d.supported_cases(runtime, test, runtime.references)
    if not test:
        raise ValueError('No supported heldout ALD Test cases')
    test = d.encoded_cases(runtime, test, runtime.references, cache_root=cache, source=frozen, progress=progress)
    for mode, head in models.items():
        report = evaluate_encoded(head, test, config, runtime.module.DEVICE)
        report['summary'].update(checkpoint=str(paths[mode]), selected_epoch=histories[mode]['selected_epoch'])
        reports[mode] = report; summaries.append(report['summary'])
    if current.source_identity(runtime) != frozen:
        raise RuntimeError('Frozen GEVIS changed during ALD experiments')
    return SimpleNamespace(summary=pd.DataFrame(summaries), reports=reports, checkpoints=paths,
        prepared=prepared, coverage=pd.DataFrame(coverage), validation_coverage=pd.DataFrame(validation_coverage),
        test_coverage=pd.DataFrame(test_coverage), support_audit=pd.DataFrame(train_audit + validation_audit + test_audit),
        exclusions=exclusions, training=histories, support_summary=support_summary, runtime=runtime)


def _continuous_interventions(cases, coverage, target_sensors=None):
    # Binary valves remain context/gate observations. Continuous drift/variance
    # injections must not manufacture impossible valve values such as 1.7.
    def eligible(step, name):
        return name in target_sensors.get(step, ()) if target_sensors is not None else not name.startswith('ValveAct_')
    retained = []
    for case in cases:
        step = case.get('step')
        indices = [i for i, name in enumerate(case['centers']) if eligible(step, name)]
        if not indices or (case['injected'] and any(case['targets'][i, 0] > 0
                for i, name in enumerate(case['centers']) if not eligible(step, name))):
            continue
        retained.append(dict(case, centers=[case['centers'][i] for i in indices],
                             targets=np.asarray(case['targets'])[indices]))
    coverage = [dict(c, status='excluded_binary_valve_intervention', generated=0)
        if not eligible(c.get('step'), c.get('center_sensor', '')) else c for c in coverage]
    return retained, coverage
