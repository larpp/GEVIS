"""User-confirmed legacy control48 objective; only Student receives gradients."""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from pecvd.experiments import fusion_answers as fa, decision_training as d


@dataclass(frozen=True)
class Config:
    seed: int = 42
    encoder_width: int = 48
    encoder_layers: int = 2
    learning_rate: float = .001
    temperature: float = 2.
    alpha: float = .5
    feature_weight: float = 1.
    epochs: int = 40
    patience: int = 8
    batch_cases: int = 8
    max_seconds: float = 1200.
    center_batch_size: int = 8
    train_cases_per_step: int = 64
    validation_cases_per_step: int = 64
    test_cases_per_step: int = 64
    benchmark_windows: int = 16
    repeats: int = 5
    warmup: int = 2
    quality_target: float = .96

    def validate(self):
        if self.encoder_width != 48 or self.encoder_layers != 2:
            raise ValueError('control48 fixes Student width=48 and Transformer layers=2')
        for name in ('epochs', 'patience', 'batch_cases', 'center_batch_size', 'train_cases_per_step',
                     'validation_cases_per_step', 'test_cases_per_step', 'benchmark_windows', 'repeats', 'warmup'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer budgets required: ' + name)
        if self.center_batch_size > 32 or type(self.seed) is not int:
            raise ValueError('Center batch <=32 and integer seed required')
        for name in ('learning_rate', 'temperature', 'feature_weight', 'max_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('Positive finite setting required: ' + name)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1 or not 0 < self.quality_target <= 1:
            raise ValueError('alpha in [0,1] and quality target in (0,1] required')


def head_logits(head, features):
    return {kind: head(features[:, j], kind) for j, kind in enumerate(fa.KINDS)}


def original_predictions(logits):
    gate = logits['BOOLEAN'].argmax(-1)
    choice = logits['CHOICE'][:, 1:].argmax(-1) + 1
    ordinal = logits['ORDINAL'][:, 1:].argmax(-1) + 1
    return torch.stack([gate, torch.where(gate == 1, choice, 0), torch.where(gate == 1, ordinal, 0)], -1)


def legacy_validation_selection(truth, prediction, loss):
    """V1/control48 epoch selection precedes V7's final answer remapping."""
    score = d._scores(truth, prediction)
    return score, d.selection_key(score, loss)


def legacy_loss(student_logits, teacher_logits, targets, student_features, teacher_features,
                feature_scale, config, weights=None):
    ce = student_features.sum() * 0
    kd = ce
    for j, kind in enumerate(fa.KINDS):
        selected = torch.ones(len(targets), dtype=torch.bool, device=targets.device) if j == 0 else targets[:, 0].eq(1)
        if not selected.any():
            continue
        offset = 0 if j == 0 else 1
        scores = student_logits[kind][selected, offset:]
        teacher = teacher_logits[kind].detach()[selected, offset:]
        temperature = config.temperature
        kd = kd + F.kl_div(F.log_softmax(scores / temperature, -1),
            F.softmax(teacher / temperature, -1), reduction='batchmean') * temperature**2
        labels = targets[selected, j] - offset
        known = labels >= 0
        if known.any():
            weight = None if weights is None else weights[kind][offset:].to(scores.device)
            ce = ce + F.cross_entropy(scores[known], labels[known], weight=weight)
    scale = feature_scale[:-10].detach().clamp_min(1e-5).to(student_features.device)
    feature = F.smooth_l1_loss(student_features[..., :-10] / scale, teacher_features.detach()[..., :-10] / scale)
    total = (1 - config.alpha) * ce + config.alpha * kd + config.feature_weight * feature
    return total, dict(ce=float(ce.detach()), kl=float(kd.detach()), feature=float(feature.detach()))
