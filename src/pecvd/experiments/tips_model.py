"""Explicit TIPS-from-scratch GEVIS experiment, not a PECVD transfer checkpoint."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pecvd.gevis_backbone import GEVISBackbone, E5TextEncoder, ANSWER_TYPES
from pecvd.multiscale_patchtst import MultiscalePatchEncoder
from pecvd.temporal_bridge import MULTISCALE_DIM
from .fusion_answers import ContextReadout, KINDS, QUESTIONS, ANSWER_CRITERIA
from .tips_optimization import FrozenTraceCache, numeric_state_from_encoded, trace_key


class CachedE5(E5TextEncoder):
    """Cache frozen E5 pooling only; projection and norm still receive gradients."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}

    def forward(self, texts, device):
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if any(any('\uac00' <= c <= '\ud7a3' or c == '\ufffd' for c in t) for t in missing):
            raise ValueError('English model text is required')
        for offset in range(0, len(missing), 64):
            selected = missing[offset:offset + 64]
            encoded = self.tokenizer(['query: ' + t for t in selected],
                max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                h = self.model(**encoded).last_hidden_state
                mask = encoded['attention_mask'][..., None].to(h.dtype)
                pooled = F.normalize((h * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)
            self._cache.update({t: p.detach().cpu() for t, p in zip(selected, pooled)})
        pooled = torch.stack([self._cache[t] for t in texts]).to(device)
        return self.norm(self.projection(pooled))


class TipsModel(nn.Module):
    def __init__(self, sensors, config):
        super().__init__()
        self.sensors = list(sensors)
        self.config = dict(config)
        self.encoder = MultiscalePatchEncoder(len(sensors), config['patch_length'],
            d_model=96, nhead=8, layers=config['encoder_layers'], dropout=0.)
        # Initialize the same public GEVIS Fusion components, with an explicit text backend.
        # Construct the byte branch first to avoid loading E5 twice.
        self.fusion = GEVISBackbone(MULTISCALE_DIM, config['fusion_width'], 0., 'byte')
        if config['text_encoder'] != 'byte':
            self.fusion.text = CachedE5(config['fusion_width'],
                model_name=config['text_encoder'])
        self.readout = ContextReadout(4 * config['fusion_width'] + 2, config['head_width'])
        self.trace_cache = FrozenTraceCache(config.get('trace_cache_mb', 256))

    def freeze_encoder(self):
        self.encoder.requires_grad_(False).eval()
        self.trace_cache = FrozenTraceCache(self.config.get('trace_cache_mb', 256))

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def answer_features(self, current, reference, centers, steps, positions, trace_keys=None):
        with torch.no_grad():
            encoded = self.trace_cache.encode_pair(self.encoder, current, reference, trace_keys)
            state = numeric_state_from_encoded(self.encoder, current, reference, encoded)
        batch, count, _ = state.shape
        patches = count // len(self.sensors)
        keys = self.fusion.numeric_key_norm(self.fusion.numeric(self.fusion.numeric_input_norm(state)))
        # One context per token. The selected-center role is numeric context, never question wording.
        descriptors = []
        for center, step in zip(centers.tolist(), steps.tolist()):
            descriptors.append([f'Main step {step}; sensor {sensor}; patch {p}; '
                + ('analysis center' if j == center else 'context sensor')
                for j, sensor in enumerate(self.sensors) for p in range(patches)])
        unique = list(dict.fromkeys(t for row in descriptors for t in row))
        lookup = {t: i for i, t in enumerate(unique)}
        text = self.fusion.text(unique, current.device)
        ids = torch.tensor([[lookup[t] for t in row] for row in descriptors], device=current.device)
        described = self.fusion.descriptor_norm(text[ids])
        grid = keys.reshape(batch, len(self.sensors), patches, -1)
        selected = grid[torch.arange(batch, device=current.device), centers]
        question = self.fusion.text([QUESTIONS[k] for k in KINDS], current.device)
        types = torch.tensor([ANSWER_TYPES[k] for k in KINDS], device=current.device)
        query = (question + self.fusion.type_emb(types))[None].expand(batch, -1, -1)
        gate = self.fusion.fusion_gate(torch.cat([keys, described], -1))
        fused_keys = keys + gate * described
        fused, attn = self.fusion.cross(query, fused_keys, keys, need_weights=True)
        fused = self.fusion.norm(fused + query)
        weights = attn.reshape(batch, len(KINDS), len(self.sensors), patches)
        local_weights = weights[torch.arange(batch, device=current.device), :, centers]
        local = (selected[:, None] * local_weights[..., None]).sum(2) / local_weights.sum(2, keepdim=True).clamp_min(1e-8)
        features = []
        for i, kind in enumerate(KINDS):
            features.append(torch.cat([fused[:, i], query[:, i], selected.mean(1), local[:, i],
                torch.stack([positions, 1 - positions], -1)], -1))
        return features

    def forward(self, current, reference, centers, steps, positions, trace_keys=None):
        if current.shape[-1] != len(self.sensors) or reference.shape != current.shape:
            raise ValueError('Input must match the selected profile sensor grid')
        allowed = self.config['main_steps']
        if allowed != 'all' and not set(steps.tolist()) <= set(allowed):
            raise ValueError('Prediction requested a Main Step outside the profile')
        features = self.answer_features(current, reference, centers, steps, positions, trace_keys)
        return [self.readout(x, kind) for x, kind in zip(features, KINDS)]


class SyntheticCases(torch.utils.data.Dataset):
    """Deterministic continuous-sensor interventions; no state/setpoint injection."""
    TYPES = ('equal', 'level_shift', 'variance', 'drift', 'spike', 'delay')

    def __init__(self, windows, sensors, targets, *, cases_per_window=24, seed=42,
                 varying_by_step=None):
        self.windows, self.sensors = windows, list(sensors)
        self.targets = [self.sensors.index(s) for s in targets]
        if not self.targets or cases_per_window < 24:
            raise ValueError('Targets and at least 24 cases per window required for six types/four strengths')
        self.count, self.seed = cases_per_window, seed
        self.varying = varying_by_step
        # Cover every selected numeric center in every supported Step, even when
        # a small window budget would otherwise leave most sensors unassessed.
        by_step = {}
        for i, window in enumerate(windows):
            by_step.setdefault(window['step'], []).append(i)
        self.plan = [(indices[j % len(indices)], self.targets[j % len(self.targets)])
            for indices in by_step.values() for j in range(max(len(indices), len(self.targets)))]
        self.reference_keys = [trace_key(np.asarray(w['values'], dtype=np.float32)) for w in windows]
        self._last_example = None
        self._last_observed = None

    def __len__(self):
        return len(self.plan) * self.count * 2

    def __getitem__(self, index):
        example, center_variant = divmod(int(index), 2)
        plan_index, variant = divmod(example, self.count)
        wi, target = self.plan[plan_index]
        window = self.windows[wi]; base = window['values']
        kind = variant % 6; level = (variant // 6) % 4 + 1
        if self._last_example == example:
            observed, current_key = self._last_observed
        else:
            rng = np.random.default_rng(self.seed + example * 7919)
            width = max(2, int(round(len(base) * rng.uniform(.25, .9))))
            start = int(rng.integers(len(base) - width + 1))
            observed = self._intervene(base, window, target, kind, level, start, width, wi, variant, example)
            # Preserve the original label rule before float32 conversion.
            current_key = trace_key(np.asarray(observed, dtype=np.float32))
            self._last_example, self._last_observed = example, (observed, current_key)
        # Counterfactual analysis center: the same waveform, a different center, independent labels.
        center = target if center_variant == 0 or len(self.sensors) == 1 else (target + 1) % len(self.sensors)
        changed = bool(np.any(observed[:, center] != base[:, center]))
        label = [int(changed), kind if changed else 0, level if changed else 0]
        return dict(current=torch.tensor(observed, dtype=torch.float32),
            reference=torch.tensor(base, dtype=torch.float32), center=center,
            step=window['step'], position=np.float32(window['position']), labels=torch.tensor(label),
            run_id=window['run_id'], window_index=wi, kind=kind, target=target,
            current_key=current_key, reference_key=self.reference_keys[wi])

    def _intervene(self, base, window, target, kind, level, start, width, wi, variant, example):
        from .injection import Injection, inject
        observed = base.copy()
        usable = self.varying is None or self.varying[window['step']][target]
        if kind == 5 and usable:
            observed[:, target] = base[np.maximum(0, np.arange(len(base)) - level), target]
        elif kind and usable:
            observed, _ = inject(base, Injection(self.TYPES[kind], target, start,
                start + width, (.5, 1., 2., 4.)[level - 1], self.seed + example), np.ones(base.shape[1]))
            if kind != 2 and (wi + variant // 6) % 2:
                observed = 2 * base - observed
        return observed


def classification_metrics(labels, predictions):
    result = {}
    for i, kind in enumerate(KINDS):
        n = len(ANSWER_CRITERIA[kind]); matrix = np.zeros((n, n), dtype=int)
        np.add.at(matrix, (labels[:, i], predictions[:, i]), 1)
        support = matrix.sum(1); predicted = matrix.sum(0); tp = matrix.diagonal()
        precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(n), where=precision + recall > 0)
        active = support + predicted > 0
        result[kind] = dict(accuracy=float(tp.sum() / matrix.sum()),
            macro_f1=float(f1[active].mean()), confusion=matrix.tolist(),
            class_support=support.tolist(), class_f1=f1.tolist(),
            majority_baseline=float(support.max() / support.sum()))
        if kind == 'BOOLEAN':
            result[kind].update(precision=float(precision[1]), recall=float(recall[1]), f1=float(f1[1]))
    return result
