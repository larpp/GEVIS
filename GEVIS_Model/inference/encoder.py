"""Head-independent, explicit current/reference feature extraction."""
from dataclasses import dataclass

import torch
from torch import nn

from ..encoder.multiscale_patchtst import MultiscalePatchEncoder
from ..fusion.model import GEVIS, numeric_state
from ..fusion.text import token_descriptions


@dataclass
class FusionOutput:
    fused: torch.Tensor
    query: torch.Tensor
    attention: torch.Tensor
    selected: torch.Tensor
    local: torch.Tensor

    @property
    def features(self):
        """Original 4*d_model feature blocks, without task-specific coordinates."""
        return torch.cat([self.fused, self.query, self.selected, self.local], -1)


class FusionEncoder(nn.Module):
    def __init__(self, encoder, fusion):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion

    def forward(self, current, reference, *, sensor_names, questions, answer_types,
                centers, phase='process', sensor_descriptors=None):
        """One window, multiple explicit (question, center, type) tasks.

        current/reference are TRAIN-standardized tensors [1,time,sensors].
        No head, anomaly injection, reference fitting, or implicit normalization.
        """
        names = list(sensor_names)
        if (current.ndim != 3 or current.shape[0] != 1 or current.shape != reference.shape
                or current.shape[-1] != len(names) or len(set(names)) != len(names)):
            raise ValueError('Matching [1,time,sensors] inputs and unique sensor_names required')
        if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
            raise ValueError('Finite complete windows required; missing values are not imputed')
        if not questions or len(questions) != len(centers) or len(centers) != len(answer_types):
            raise ValueError('Aligned nonempty questions, centers and answer_types required')
        if any(center not in names for center in centers):
            raise ValueError('Unknown center sensor')
        if any(kind not in ('BOOLEAN', 'CHOICE', 'ORDINAL') for kind in answer_types):
            raise ValueError('Supported answer type tags: BOOLEAN, CHOICE, ORDINAL')
        state = numeric_state(self.encoder, current, reference)
        patches = state.shape[1] // len(names)
        descriptions = token_descriptions(phase,
            [(sensor_descriptors or {}).get(name, name) for name in names], patches)
        contexts = []
        for center in centers:
            # Exact center binding used by the historical Fusion extractor.
            sid = names.index(center)
            contexts.append([d + ('; UI analysis center.' if i // patches == sid else '; context sensor.')
                             for i, d in enumerate(descriptions)])
        fused, query, attention = self.fusion(
            state.expand(len(questions), -1, -1), list(questions), list(answer_types),
            [[] for _ in questions], contexts, return_fusion=True)
        keys = self.fusion.numeric_key_norm(self.fusion.numeric(self.fusion.numeric_input_norm(state)))
        grid = keys.reshape(1, len(names), patches, -1)[0]
        selected = torch.stack([grid[names.index(c)].mean(0) for c in centers])
        weights = attention.reshape(len(questions), len(names), patches)
        local = torch.stack([
            (grid[names.index(c)] * weights[i, names.index(c), :, None]).sum(0)
            / weights[i, names.index(c)].sum().clamp_min(1e-8)
            for i, c in enumerate(centers)])
        return FusionOutput(fused, query, attention, selected, local)


def create_encoder(*, n_sensors, patch_length=4, latent_dim=96, fusion_dim=192,
                   heads=8, layers=3, dropout=0., text_encoder='byte', text_revision=None):
    """Create untrained modules. Creation never implies pretrained performance."""
    encoder = MultiscalePatchEncoder(n_sensors, patch_length, latent_dim, heads, layers, dropout)
    if latent_dim != 96:
        raise ValueError('The preserved 440-D comparison bridge requires latent_dim=96')
    fusion = GEVIS(440, fusion_dim, dropout, text_encoder, text_revision)
    return FusionEncoder(encoder, fusion)
