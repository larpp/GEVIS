"""Verified STEP 1/2 reuse. Every source tensor is frozen, including E5/Fusion.

Legacy unseen-channel adaptation is explicit; it is not a native portable model.
No learned input projection, source reinitialization, or target-Step search.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from pecvd.encoder_loading import load_encoder
from pecvd.gevis_backbone import (GEVISBackbone, E5TextEncoder, BackbonePretrainHeads,
    ANSWER_TYPES, attention_inputs, token_descriptions)
from pecvd.temporal_bridge import supports, numeric_state
from .frozen_transfer import FrozenChannelAdapter
from .fusion_answers import KINDS, QUESTIONS
from .questions import center_descriptions
from .tips_dataset import sha, inside
from .tips_optimization import trace_key

SOURCE_STEPS = ('Gap-up', 'PRE-HEAT', 'STAB1', 'DEP.', 'PURGE', 'PUMPING')
CONTRACT = 'tips.frozen_step12.explicit_channel_adapter.v1'


def tensor_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def inspect_bundle(config, root):
    root = Path(root).resolve()
    paths = {k: (root / config[k]).resolve() for k in ('source_manifest', 'source_backbone')}
    for key, path in paths.items():
        expected = config.get(key + '_sha256')
        if not expected or sha(path) != expected:
            raise ValueError('Pinned frozen source hash mismatch: ' + key)
    manifest = json.loads(paths['source_manifest'].read_text(encoding='utf-8'))
    if set(manifest['step_checkpoints']) != set(SOURCE_STEPS):
        raise ValueError('The complete six-encoder STEP 1 bundle is required')
    encoders = {}
    for step in SOURCE_STEPS:
        row = manifest['step_checkpoints'][step]
        path = inside(paths['source_manifest'].parent, row['path'])
        if sha(path) != row['sha256']:
            raise ValueError('STEP 1 checkpoint hash mismatch: ' + step)
        encoders[step] = dict(path=str(path), sha256=row['sha256'])
    return dict(contract=CONTRACT, manifest_sha256=sha(paths['source_manifest']),
        backbone_sha256=sha(paths['source_backbone']), encoders=encoders,
        manifest_path=str(paths['source_manifest']), backbone_path=str(paths['source_backbone']),
        source_encoder_step=config['source_encoder_step'], input_adapter=config['input_adapter'])


class FrozenSource(nn.Module):
    def __init__(self, config, root, sensors):
        super().__init__()
        self.identity = inspect_bundle(config, root)
        self.sensors = list(sensors)
        self.allowed_steps = config['main_steps']
        payload = torch.load(self.identity['backbone_path'], map_location='cpu', weights_only=False)
        if payload.get('schema') not in ('gevis.backbone.v9', 'gevis.backbone.generic.v10'):
            raise ValueError('Unsupported explicit STEP 2 checkpoint schema')
        if payload.get('question_language') != 'English':
            raise ValueError('English STEP 2 checkpoint required')
        bridge = payload['step1_bridge']
        if (bridge['manifest_sha256'] != self.identity['manifest_sha256'] or
            bridge['encoder_hashes'] != {s: e['sha256'] for s, e in self.identity['encoders'].items()}):
            raise ValueError('STEP 2 was not trained with this exact six-encoder STEP 1 bundle')
        self.encoder_names = list(SOURCE_STEPS)
        self.encoders = nn.ModuleList()
        self.metadata = {}
        for step in self.encoder_names:
            checkpoint = torch.load(self.identity['encoders'][step]['path'], map_location='cpu', weights_only=False)
            if checkpoint.get('step') != step:
                raise ValueError('STEP 1 step identity mismatch')
            self.encoders.append(load_encoder(checkpoint).requires_grad_(False).eval())
            self.metadata[step] = checkpoint
        self.source_step = config['source_encoder_step']
        if self.source_step not in self.encoder_names:
            raise ValueError('Unknown source_encoder_step; choose one of the six frozen encoders')
        source = self.encoders[self.encoder_names.index(self.source_step)]
        meta = self.metadata[self.source_step]
        adapter = config['input_adapter']
        if getattr(source, 'supports_native_sensor_transfer', False):
            self.adaptation = 'native transferable source forward'
        elif self.sensors == list(meta['sensor_columns']):
            self.adaptation = 'exact source sensor order; original joint forward'
        elif adapter == 'legacy_channel_independent':
            source = FrozenChannelAdapter(source, len(sensors))
            self.adaptation = ('explicit legacy adapter: mean frozen source sensor embedding; '
                               'independent temporal attention per target channel; not native joint transfer')
        else:
            raise ValueError('Unseen sensors require the explicit legacy_channel_independent adapter')
        # Alias of an already registered frozen module; no new trainable tensors.
        self.active_encoder = source
        self.patch_length = source.patch_len
        if config['window_length'] % self.patch_length or not 2 <= config['window_length'] // self.patch_length <= 64:
            raise ValueError('window_length must contain 2..64 complete source patches; no interpolation')
        width = int(payload['d_model']); numeric_dim = int(payload['numeric_dim'])
        self.backbone = GEVISBackbone(numeric_dim, width, 0., 'byte')
        if payload['text_encoder_name'] != 'byte':
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            name, revision = payload['text_encoder_name'], payload['text_encoder_revision']
            tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=True)
            text_model = AutoModel.from_config(AutoConfig.from_pretrained(name, revision=revision, local_files_only=True))
            self.backbone.text = E5TextEncoder(width, model_name=name, revision=revision,
                                              tokenizer=tokenizer, model=text_model)
        self.backbone.load_state_dict(payload['backbone_state'], strict=True)
        self.backbone.fusion_mode = payload.get('fusion_mode', 'legacy')
        self.pretrain_heads = BackbonePretrainHeads(width)
        self.pretrain_heads.load_state_dict(payload['pretrain_heads_state'], strict=True)
        self.numeric_dim, self.feature_dim = numeric_dim, 4 * width + 2
        self.requires_grad_(False).eval()
        self._text_cache = {}
        self._trace_cache = OrderedDict()
        self._trace_bytes = 0
        self._trace_limit = int(config.get('trace_cache_mb', 256) * 1024**2)
        self.initial_hash = tensor_hash(self)

    def train(self, mode=True):
        super().train(False)
        return self

    def to_device(self, device):
        # Unselected source encoders stay on CPU, fully loaded and frozen.
        self.active_encoder.to(device)
        self.backbone.to(device)
        return self

    def verify_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise RuntimeError('A STEP 1/2 source parameter was unfrozen or received a gradient')
        if tensor_hash(self) != self.initial_hash:
            raise RuntimeError('Frozen STEP 1/2 tensor values changed')

    @torch.no_grad()
    def text(self, texts, device):
        missing = list(dict.fromkeys(t for t in texts if t not in self._text_cache))
        for start in range(0, len(missing), 64):
            subset = missing[start:start + 64]
            values = self.backbone.text(subset, device)
            self._text_cache.update({t: v.detach().cpu().clone() for t, v in zip(subset, values)})
        return torch.stack([self._text_cache[t] for t in texts]).to(device)

    @torch.no_grad()
    def legacy_state(self, current, reference, trace_keys):
        """Original 392-D bridge, reusing deterministic tokens for reconstruction."""
        joined = torch.cat([current, reference], 0)
        keys = list(trace_keys['current']) + list(trace_keys['reference']) if trace_keys else [
            trace_key(x.cpu().numpy()) for x in joined]
        unique = list(dict.fromkeys(keys)); values = {}; missing = []
        for key in unique:
            if key in self._trace_cache:
                values[key] = self._trace_cache[key].to(current.device)
                self._trace_cache.move_to_end(key)
            else:
                missing.append(key)
        if missing:
            indices = torch.tensor([keys.index(k) for k in missing], device=current.device)
            encoded = self.active_encoder.tokens(joined.index_select(0, indices))
            for key, token in zip(missing, encoded):
                values[key] = token
                size = token.numel() * token.element_size()
                if size <= self._trace_limit:
                    while self._trace_bytes + size > self._trace_limit:
                        _, old = self._trace_cache.popitem(last=False)
                        self._trace_bytes -= old.numel() * old.element_size()
                    self._trace_cache[key] = token.detach().to('cpu', copy=True)
                    self._trace_bytes += size
        a, b = torch.stack([values[k] for k in keys]).chunk(2)
        cp, _ = self.active_encoder.patchify(current); rp, _ = self.active_encoder.patchify(reference)
        cr, rr = self.active_encoder.decoder(a), self.active_encoder.decoder(b)
        cm, rm = cp.mean(-1), rp.mean(-1)
        evidence = torch.stack([cm, rm, (cm-rm).abs(), cp.std(-1, unbiased=False),
            rp.std(-1, unbiased=False), (cp-rp).abs().mean(-1),
            (cr-cp).square().mean(-1), (rr-rp).square().mean(-1)], -1)
        return torch.cat([a, b, a-b, (a-b).abs(), evidence], -1)

    @torch.no_grad()
    def features(self, current, reference, centers, steps, positions, trace_keys=None):
        if current.shape != reference.shape or current.shape[-1] != len(self.sensors):
            raise ValueError('Source input must match the selected target sensor grid')
        if self.allowed_steps != 'all' and not set(steps.tolist()) <= set(self.allowed_steps):
            raise ValueError('Feature request contains a Main Step outside the selection profile')
        if supports(self.active_encoder):
            state = numeric_state(self.active_encoder, current, reference)
        else:
            state = self.legacy_state(current, reference, trace_keys)
        if state.shape[-1] != self.numeric_dim:
            raise ValueError('Frozen source encoder/Fusion feature contract mismatch; no padding/truncation')
        model = self.backbone; batch, count, _ = state.shape
        patches = count // len(self.sensors)
        keys = model.numeric_key_norm(model.numeric(model.numeric_input_norm(state)))
        descriptions = [center_descriptions(token_descriptions(f'Main step {step}', self.sensors, patches),
            self.sensors, patches, self.sensors[center]) for step, center in zip(steps.tolist(), centers.tolist())]
        unique = list(dict.fromkeys(t for row in descriptions for t in row))
        lookup = {t: i for i, t in enumerate(unique)}
        ids = torch.tensor([[lookup[t] for t in row] for row in descriptions], device=current.device)
        described = model.descriptor_norm(self.text(unique, current.device)[ids])
        question = self.text([QUESTIONS[k] for k in KINDS], current.device)[None].expand(batch, -1, -1)
        types = torch.tensor([ANSWER_TYPES[k] for k in KINDS], device=current.device)
        query = question + model.type_emb(types)[None]
        attention_query, fused_keys = attention_inputs(keys, described, question, query,
            model.fusion_gate, model.fusion_mode)
        # Original STEP 2 uses the same fused keys for both K and V.
        fused, attention = model.cross(attention_query, fused_keys, fused_keys, need_weights=True)
        fused = model.norm(fused + query)
        selected = keys.reshape(batch, len(self.sensors), patches, -1)[torch.arange(batch, device=current.device), centers]
        weights = attention.reshape(batch, len(KINDS), len(self.sensors), patches)[torch.arange(batch, device=current.device), :, centers]
        local = (selected[:, None] * weights[..., None]).sum(2) / weights.sum(2, keepdim=True).clamp_min(1e-8)
        return torch.cat([fused, query, selected.mean(1)[:, None].expand_as(query), local,
            torch.stack([positions, 1-positions], -1)[:, None].expand(-1, len(KINDS), -1)], -1)
