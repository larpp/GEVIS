"""Joint target-head objectives; inference remains the original learned argmax.

All losses use TRAIN. Epoch selection uses Validation only. No model scores or
thresholds are fitted to Test; synthetic-change F1 is not equipment-fault F1.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import fusion_answers as fa

RECIPE = 'peald_grouped_consistency_ordinal_v1'


def grouped(examples, split, device):
    groups = {}
    for row in examples:
        if row['split'] != split:
            raise ValueError('Only TRAIN optimization and Validation selection are allowed')
        key = tuple(row[k] for k in ('run_id', 'step', 'pair_id', 'center_sensor'))
        family = groups.setdefault(key, {})
        kind = row['answer_type']
        if kind not in fa.KINDS or kind in family:
            raise ValueError('Unique three-answer groups required')
        label = row['label']
        if not isinstance(label, (int, np.integer)) or not 0 <= label < len(fa.ANSWER_CRITERIA[kind]):
            raise ValueError('Supported integer answer label required')
        family[kind] = row
    if not groups or any(set(g) != set(fa.KINDS) for g in groups.values()):
        raise ValueError('Complete three-answer groups required')
    x = torch.stack([torch.stack([g[k]['features'] for k in fa.KINDS]) for g in groups.values()]).to(device)
    y = torch.tensor([[g[k]['label'] for k in fa.KINDS] for g in groups.values()], device=device)
    if not torch.isfinite(x).all():
        raise ValueError('Finite Fusion features required')
    if not torch.equal(y[:, 0].bool(), y[:, 1].bool()) or not torch.equal(y[:, 0].bool(), y[:, 2].bool()):
        raise ValueError('Contradictory synthetic group labels')
    return x, y


def auxiliary_losses(logits, ordinal_labels):
    probabilities = [value.softmax(-1) for value in logits]
    changed = torch.stack([1 - p[:, 0] for p in probabilities], dim=-1)
    # Penalize disagreement of the three learned change probabilities, not labels.
    consistency = ((changed - changed.mean(-1, keepdim=True)) ** 2).mean()
    ordinal = probabilities[2]
    target = F.one_hot(ordinal_labels, ordinal.shape[-1]).to(ordinal.dtype)
    # Ranked probability score: crossing more ordinal boundaries costs more.
    distance = ((ordinal.cumsum(-1)[:, :-1] - target.cumsum(-1)[:, :-1]) ** 2).mean()
    return consistency, distance


def metrics(logits, labels):
    macro = []
    binary_f1 = None
    predictions = []
    for i, (kind, output) in enumerate(zip(fa.KINDS, logits)):
        pred = output.argmax(-1)
        truth = labels[:, i]
        predictions.append(pred)
        scores = []
        for c in range(len(fa.ANSWER_CRITERIA[kind])):
            tp = int(((pred == c) & (truth == c)).sum())
            fp = int(((pred == c) & (truth != c)).sum())
            fn = int(((pred != c) & (truth == c)).sum())
            if tp + fp + fn:
                scores.append(2 * tp / (2 * tp + fp + fn))
            if kind == 'BOOLEAN' and c == 1:
                binary_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        macro.append(float(np.mean(scores)))
    mismatch = (predictions[0].bool() != predictions[1].bool()) | (predictions[0].bool() != predictions[2].bool())
    changed = labels[:, 2] > 0
    return dict(boolean_f1=binary_f1, macro_f1=float(np.mean(macro)),
        choice_macro_f1=macro[1], ordinal_macro_f1=macro[2],
        ordinal_changed_accuracy=float((predictions[2][changed] == labels[changed, 2]).float().mean()) if changed.any() else None,
        inconsistency_rate=float(mismatch.float().mean()))


def selection_key(scores, loss, goal):
    # The goal is a Validation selection criterion, never an inference threshold.
    passed = scores['boolean_f1'] >= goal
    return (int(passed), scores['macro_f1'] if passed else scores['boolean_f1'],
            scores['boolean_f1'] if passed else scores['macro_f1'], -loss)


def fit_readout(train, validation, config, device='cpu', *, initial_head=None):
    config.validate()
    if {e['run_id'] for e in train} & {e['run_id'] for e in validation}:
        raise ValueError('TRAIN/Validation run overlap')
    x, y = grouped(train, 'train', device)
    vx, vy = grouped(validation, 'validation', device)
    torch.manual_seed(config.seed)
    head = fa.ContextReadout(x.shape[-1], config.width).to(device)
    if initial_head is not None:
        # Preserve the common feature coordinates when adapting its weights.
        head.load_state_dict(initial_head.state_dict(), strict=True)
    with torch.no_grad():
        flattened = x.flatten(0, 1)
        if initial_head is None:
            head.feature_mean.copy_(flattened.mean(0))
            scale = flattened.std(0, unbiased=False)
            head.feature_scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)))
    weights = []
    for i, kind in enumerate(fa.KINDS):
        counts = torch.bincount(y[:, i], minlength=len(fa.ANSWER_CRITERIA[kind]))
        allowed = counts > 0
        if allowed.sum() < 2 or not allowed[vy[:, i]].all():
            raise ValueError('TRAIN needs coverage of every Validation class and at least two answers')
        getattr(head, 'allowed_' + kind).copy_(allowed)
        weights.append(torch.where(allowed, counts.sum() / counts.clamp_min(1) / allowed.sum(), 0).float())

    def objective(inputs, labels):
        logits = [head(inputs[:, i], kind) for i, kind in enumerate(fa.KINDS)]
        ce = torch.stack([F.cross_entropy(z, labels[:, i], weight=weights[i]) for i, z in enumerate(logits)]).mean()
        consistency, distance = auxiliary_losses(logits, labels[:, 2])
        return ce + config.consistency_weight * consistency + config.ordinal_weight * distance, logits

    optimizer = torch.optim.AdamW(head.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(config.seed)
    history = []; updates = 0; stale = 0; best = None; best_state = None; best_scores = None
    started = time.perf_counter()
    # batch_size remains a question-row budget; each group contains three rows.
    batch_groups = max(1, config.batch_size // len(fa.KINDS))
    for epoch in range(config.epochs):
        head.train(); losses = []
        order = torch.randperm(len(y), generator=generator)
        for offset in range(0, len(order), batch_groups):
            if time.perf_counter() - started >= config.max_seconds:
                break
            ix = order[offset:offset + batch_groups].to(device)
            loss, _ = objective(x[ix], y[ix])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite answer loss; no checkpoint')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step(); updates += 1; losses.append(float(loss.detach()))
        if not losses:
            break
        head.eval()
        with torch.no_grad():
            value, logits = objective(vx, vy)
            if not torch.isfinite(value):
                raise ValueError('Nonfinite Validation loss; no checkpoint')
            value = float(value)
            scores = metrics(logits, vy)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=value,
                            **{'validation_' + k: v for k, v in scores.items()}))
        print(f'PEALD readout epoch {epoch+1}/{config.epochs}: validation loss {value:.4f}, '
              f'BOOLEAN F1 {scores["boolean_f1"]:.4f}', flush=True)
        key = selection_key(scores, value, config.boolean_f1_goal)
        if best is None or key > best:
            best = key; best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            best_scores = dict(scores, validation_loss=value, epoch=epoch + 1); stale = 0
        else:
            stale += 1
        if stale >= config.patience or time.perf_counter() - started >= config.max_seconds:
            break
    if not updates or best_state is None:
        raise RuntimeError('No trained answer weights; refusing random-head export')
    head.load_state_dict(best_state); head.requires_grad_(False).eval()
    return head, dict(optimizer_updates=updates, history=history,
        validation_loss=best_scores['validation_loss'], selected_validation=best_scores,
        validation_boolean_goal_met=best_scores['boolean_f1'] >= config.boolean_f1_goal,
        selection_policy='Validation BOOLEAN F1 goal, then mean task macro F1; no Test selection',
        trainable_parameters=sum(p.numel() for p in head.parameters()), fit_seconds=time.perf_counter() - started,
        train_runs=sorted({e['run_id'] for e in train}), validation_runs=sorted({e['run_id'] for e in validation}),
        trained_steps=sorted({e['step'] for e in train}), training_recipe=RECIPE, ordinal_target=fa.ORDINAL_TARGET)
