"""Checkpoint-compatible text encoder and attention inputs."""
from __future__ import annotations
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
TARGET_FUSION_MODE="descriptor_anchor_v4"
REFERENCE_FUSION_MODE="numeric_first_generic_v11"
DEFAULT_TEXT_ENCODER="intfloat/e5-small-v2"
DEFAULT_TEXT_ENCODER_REVISION="ffb93f3bd4047442299a41ebb6fa998a38507c52"

def token_descriptions(phase,sensor_names,patch_count):
    if patch_count<1 or not sensor_names: raise ValueError('Nonempty sensor/patch grid required')
    scopes=['early interval' if p<patch_count/3 else 'late interval' if p>=2*patch_count/3 else 'middle interval' for p in range(patch_count)]
    return [f'Process phase {phase}; sensor {sensor}; relative time {scope}.' for sensor in sensor_names for scope in scopes]

def attention_inputs(numeric_keys, descriptors, text_query, typed_query, gate_layer, mode='legacy'):
    """One train/runtime fusion contract; no target IDs or oracle metadata.

    Named-target candidate anchors keys in sensor/time descriptions. Numeric
    features still contribute through a learned bounded gate; answer type enters
    the fused residual but cannot itself redirect sensor attention.
    """
    if mode not in ('legacy',TARGET_FUSION_MODE,REFERENCE_FUSION_MODE): raise ValueError('Unknown GEVIS fusion mode')
    if descriptors is None:
        if mode!= 'legacy': raise ValueError('Target binding requires token descriptions')
        return typed_query,numeric_keys
    gate=gate_layer(torch.cat([numeric_keys,descriptors],-1))
    if mode==REFERENCE_FUSION_MODE:
        return text_query,numeric_keys+gate*descriptors
    if mode==TARGET_FUSION_MODE:
        return text_query,descriptors+gate*numeric_keys
    return typed_query,numeric_keys+gate*descriptors

class E5TextEncoder(nn.Module):
    """Frozen English E5 encoder with a small trainable projection into GEVIS space."""

    def __init__(
        self,
        d_model: int,
        model_name: str = DEFAULT_TEXT_ENCODER,
        *,
        revision: str = DEFAULT_TEXT_ENCODER_REVISION,
        max_length: int = 128,
        tokenizer=None,
        model=None,
    ):
        super().__init__()
        if tokenizer is None or model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "E5 text encoding requires transformers; install config/requirements-a.txt"
                ) from error
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModel.from_pretrained(model_name, revision=revision)
        self.model_name = model_name
        self.revision = revision
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        hidden_size = int(self.model.config.hidden_size)
        self.projection = nn.Linear(hidden_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        # The pretrained encoder stays deterministic; only projection/norm train.
        self.model.eval()
        return self

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if any(any('\uac00' <= char <= '\ud7a3' or char == '\ufffd' for char in text) for text in texts):
            raise ValueError('English E5 input required: use English questions, options and metadata descriptors; damaged text is not accepted')
        prefixed = [text if text.startswith("query: ") else f"query: {text}" for text in texts]
        encoded = self.tokenizer(
            prefixed,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = F.normalize(pooled, p=2, dim=-1)
        return self.norm(self.projection(pooled))

def frozen_text(model, texts, device, *, capacity=4096, chunk_size=32):
    """Reuse text only while the encoder is frozen and in evaluation mode."""
    encoder = model.text
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        model._text_cache = OrderedDict()
        return encoder(texts, device)
    cache = getattr(model, '_text_cache', None)
    if cache is None:
        cache = model._text_cache = OrderedDict()
    unique = list(dict.fromkeys(texts))
    missing = [text for text in unique if text not in cache]
    embeddings = {text: cache[text] for text in unique if text in cache}
    with torch.no_grad():
        for start in range(0, len(missing), chunk_size):
            items = missing[start:start + chunk_size]
            values = encoder(items, device).detach().cpu()
            for text, value in zip(items, values):
                embeddings[text] = value
                cache[text] = value
                while len(cache) > capacity:
                    cache.popitem(last=False)
    return torch.stack([embeddings[text] for text in texts]).to(device)
