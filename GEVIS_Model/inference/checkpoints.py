"""Load preserved native Encoder/Fusion checkpoints without training or notebooks."""
import json
from pathlib import Path
import torch

from .encoder import FusionEncoder
from ..encoder.multiscale_patchtst import MultiscalePatchEncoder, BACKEND, CONTRACT
from ..fusion.model import GEVIS
from ..fusion.temporal_bridge import install_checkpoint_adapter
from ..utils.files import digest


def load_encoder_bundle(manifest_path, backbone_path, *, step='DEP.', device='cpu',
                        text_model_path=None):
    """Paths may be anywhere; manifest-relative encoder files retain hash checks.

    Only trusted torch checkpoints should be loaded (historical files contain
    numpy metadata and require weights_only=False). E5 weights are a separate
    dependency; supply text_model_path for a local Hugging Face snapshot.
    """
    manifest_path = Path(manifest_path).resolve()
    backbone_path = Path(backbone_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'gevis.encoder_manifest.rich_ssl.v1':
        raise ValueError('Unsupported encoder manifest schema')
    if step not in manifest['step_checkpoints']:
        raise ValueError(f'Encoder not available: {step}; choose {list(manifest["step_checkpoints"])}')
    item = manifest['step_checkpoints'][step]
    encoder_path = (manifest_path.parent / item['path']).resolve()
    encoder_path.relative_to(manifest_path.parent)
    if digest(encoder_path) != item['sha256']:
        raise ValueError(f'Encoder checksum differs: {step}')
    checkpoint = torch.load(backbone_path, map_location='cpu', weights_only=False)
    if checkpoint.get('schema') != 'gevis.backbone.generic.v10':
        raise ValueError('Unsupported Fusion checkpoint schema')
    if checkpoint.get('step1_bridge', {}).get('manifest_sha256') != digest(manifest_path):
        raise ValueError('Fusion/encoder manifest identity mismatch')
    record = torch.load(encoder_path, map_location='cpu', weights_only=False)
    if record.get('encoder_backend') != BACKEND:
        raise ValueError('Portable loader supports the preserved native multiscale backend only')
    if record.get('reconstruction_contract') != CONTRACT:
        raise ValueError('Encoder reconstruction contract differs')
    cfg = record['model_config']
    encoder = MultiscalePatchEncoder(len(record['sensor_columns']), record['patch_length'], **cfg)
    encoder.load_state_dict(record['model_state'], strict=True)
    fusion = GEVIS(int(checkpoint['numeric_dim']), int(checkpoint['d_model']), 0.,
                   text_model_path or checkpoint['text_encoder_name'], checkpoint['text_encoder_revision'])
    state = checkpoint['backbone_state']
    install_checkpoint_adapter(fusion, state, checkpoint['numeric_dim'], checkpoint['d_model'])
    components = ((fusion.text, 'text'), (fusion.numeric_input_norm, 'numeric_input_norm'),
        (fusion.numeric, 'numeric'), (fusion.numeric_key_norm, 'numeric_key_norm'),
        (fusion.descriptor_norm, 'descriptor_norm'), (fusion.fusion_gate, 'fusion_gate'),
        (fusion.type_emb, 'type_emb'), (fusion.cross, 'cross'), (fusion.answer_norm, 'norm'))
    for target, name in components:
        target.load_state_dict({k.removeprefix(name + '.'): v for k, v in state.items()
                                if k.startswith(name + '.')}, strict=True)
    for name in ('support_head', 'semantic_heads'):
        getattr(fusion, name).load_state_dict({k.removeprefix(name + '.'): v
            for k, v in checkpoint['pretrain_heads_state'].items() if k.startswith(name + '.')}, strict=True)
    fusion.fusion_mode = checkpoint.get('fusion_mode', 'legacy')
    model = FusionEncoder(encoder, fusion).to(device).requires_grad_(False).eval()
    return model, record


def load_numeric_encoder(path, *, device='cpu'):
    """Restore the explicit new-data SSL format; no Fusion or normalizer is implied."""
    record = torch.load(path, map_location='cpu', weights_only=False)
    if record.get('schema') != 'gevis.numeric_ssl.portable.v1':
        raise ValueError('Expected a portable numeric SSL checkpoint')
    model = MultiscalePatchEncoder(record['n_sensors'], record['patch_length'], **record['model_config'])
    model.load_state_dict(record['model_state'], strict=True)
    return model.to(device).eval(), record
