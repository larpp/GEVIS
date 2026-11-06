"""Frozen STEP1/2 features, independent of domain decision heads."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from .experiments import fusion_answers as fusion


class GEVIS:
    def __init__(self, root, source, encoder_steps=None):
        self.root = Path(root).resolve()
        selected = list(source.encoders) if encoder_steps is None else list(encoder_steps)
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(source.encoders):
            raise ValueError('Select unique saved GEVIS encoder Steps')
        self.encoder_steps = tuple(selected)
        self.encoders = {s: source.encoders[s] for s in selected}
        self.module = SimpleNamespace(**source.module.__dict__)
        self.module.STEP_ENCODERS = self.encoders
        self.module.FUSION_ANSWER_HEAD = None
        self.module.FUSION_ANSWER_TRAINING = None
        self.module.FUSION_ANSWER_PATH = None
        self.module.GEVIS_MODEL.requires_grad_(False).eval()
        for encoder, _ in self.encoders.values():
            encoder.requires_grad_(False).eval()

    def describe(self):
        return pd.DataFrame([dict(encoder_step=s, patch_length=e.patch_len,
            window_length=m['window_length'], frozen=True, decision_head=False)
            for s, (e, m) in self.encoders.items()])

    @torch.no_grad()
    def encode(self, current, reference, *, step, centers, metadata=None, answer_types=None):
        """Return question-conditioned Fusion features and (sensor, kind) tasks.

        Inputs are finite, already standardized native arrays. ALD callers supply
        their fitted metadata; this object never fits a normalizer or a head.
        """
        if step not in self.encoders:
            raise ValueError('Encoder Step was not loaded')
        module = SimpleNamespace(**self.module.__dict__)
        encoder, original = self.encoders[step]
        module.STEP_ENCODERS = {step: (encoder, original if metadata is None else metadata)}
        return fusion.fusion_features(module, current, reference, step, centers,
                                      answer_types=answer_types)


def load_gevis(root, *, encoder_steps=None):
    """Load library model factories and weights, without executing notebooks."""
    from .gevis_runtime import load_source
    root = Path(root).resolve()
    return GEVIS(root, load_source(root, encoder_steps=encoder_steps), encoder_steps)
