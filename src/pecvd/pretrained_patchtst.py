"""Pinned public PatchTST encoder, adapted to native GEVIS patch grids.

Reuses the original HF/IBM patch embedding, positional encoding, CLS and every
Transformer layer. Native patches are interpolated to the public patch width;
positions are interpolated to the native patch count, not to physical seconds.
TRAIN-standardized amplitudes are NOT re-normalized within each input window.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID='ibm-granite/granite-timeseries-patchtst'
REVISION='7fe295d8bc8fbac8041b60ab351882634165517f'
BACKEND='hf_patchtst_rich_ssl_v1'
PUBLIC_FILES_SHA256='4ed9daefc3f6193295a3bf03f35ae45e2876c66ecaf3081cbcf96e2721c283e5'


def checkpoint_hash(path):
    h=hashlib.sha256()
    for name in ('config.json','model.safetensors'):
        file=Path(path)/name
        if not file.is_file(): raise FileNotFoundError(f'Missing public checkpoint file {name}; run scripts/setup_patchtst.py')
        h.update(name.encode())
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


class PretrainedPatchEncoder(nn.Module):
    def __init__(self,n_sensors,patch_len,d_model=96,*,hf_config):
        super().__init__()
        from transformers import PatchTSTConfig,PatchTSTModel
        cfg=PatchTSTConfig(**hf_config)
        if not cfg.share_embedding or cfg.channel_attention:
            raise ValueError('Channel-independent shared public PatchTST required')
        self.n_sensors=n_sensors; self.patch_len=patch_len; self.d_model=d_model
        self.hf_config=cfg.to_dict()
        self.pretrained=PatchTSTModel(cfg).encoder
        width=cfg.d_model
        self.mask_token=nn.Parameter(torch.zeros(width))
        self.projection=nn.Linear(width,d_model)
        self.norm=nn.LayerNorm(d_model)
        self.decoder=nn.Linear(d_model,patch_len)
        self.local_statistics=nn.Sequential(nn.Linear(5,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        # The public encoder starts frozen. Fine-tuning the final layers is an
        # explicit, small-LR STEP 1 option, not accidental whole-model training.
        self.pretrained.requires_grad_(False)

    @classmethod
    def from_public(cls,path,n_sensors,patch_len,d_model=96):
        from transformers import PatchTSTForPrediction
        identity=checkpoint_hash(path)
        if identity!=PUBLIC_FILES_SHA256: raise ValueError('Public PatchTST files do not match pinned weights/config')
        public,info=PatchTSTForPrediction.from_pretrained(str(path),local_files_only=True,use_safetensors=True,output_loading_info=True)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Incomplete public PatchTST weight load; random fallback forbidden')
        model=cls(n_sensors,patch_len,d_model,hf_config=public.config.to_dict())
        model.pretrained.load_state_dict(public.model.encoder.state_dict(),strict=True)
        model.pretrained_identity={'model_id':MODEL_ID,'revision':REVISION,'files_sha256':identity,
            'pretraining':'ETTh1 forecasting; not manufacturing or FM-r2',
            'adaptation':'native-patch interpolation; interpolated public positions; projected latent width'}
        return model

    def configure_finetuning(self,last_layers=1):
        if not 0<=last_layers<=len(self.pretrained.layers): raise ValueError('Invalid pretrained unfreeze count')
        self.pretrained.requires_grad_(False)
        for layer in list(self.pretrained.layers)[len(self.pretrained.layers)-last_layers:] if last_layers else []:
            layer.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        # Do not drift frozen BatchNorm running statistics during adaptation.
        for module in self.pretrained.modules():
            if not any(p.requires_grad for p in module.parameters()): module.eval()
        return self

    def patchify(self,x):
        if x.ndim!=3 or x.shape[1]%self.patch_len or x.shape[1]<self.patch_len:
            raise ValueError('Finite complete native time/sensor patches required')
        b,t,s=x.shape; p=t//self.patch_len
        return x.reshape(b,p,self.patch_len,s).permute(0,3,1,2).reshape(b,s*p,self.patch_len),p

    def channel_independent_tokens(self,x,mask=None):
        z,p=self.patchify(x); b=len(x); sensors=x.shape[2]
        if mask is not None:
            if mask.shape!=z.shape[:2]: raise ValueError('Mask/native token grid mismatch')
            # Hide BEFORE patch interpolation AND local-statistic extraction.
            z=torch.where(mask[...,None],torch.zeros_like(z),z)
        sampled=F.interpolate(z.reshape(-1,1,self.patch_len),size=self.hf_config['patch_length'],mode='linear',align_corners=False)
        # Use the verified shared projection directly; HF's wrapper otherwise
        # rejects a changed channel COUNT despite shared channel-independent weights.
        h=self.pretrained.embedder.input_embedding(sampled.reshape(b,sensors,p,-1))
        if mask is not None: h=torch.where(mask.reshape(b,sensors,p,1),self.mask_token,h)
        pe=self.pretrained.positional_encoder
        positions=pe.position_enc[1:] if pe.use_cls_token else pe.position_enc
        position=F.interpolate(positions.T[None],size=p,mode='linear',align_corners=False)[0].T
        h=pe.positional_dropout(h+position)
        if pe.use_cls_token: h=torch.cat([(pe.cls_token+pe.position_enc[:1]).expand(b,sensors,1,-1),h],dim=2)
        for layer in self.pretrained.layers: h=layer(hidden_state=h,output_attentions=False)[0]
        if pe.use_cls_token: h=h[:,:,1:]
        statistics=torch.stack([z.mean(-1),z.std(-1,unbiased=False),z[...,0],z[...,-1],z.abs().amax(-1)],-1)
        local=self.local_statistics(statistics)
        if mask is not None: local=local.masked_fill(mask[...,None],0)
        return self.norm(self.projection(h.reshape(b,sensors*p,-1))+local)

    def tokens(self,x,mask=None):
        if x.shape[2]!=self.n_sensors: raise ValueError('Use explicit channel-independent adapter for external sensors')
        return self.channel_independent_tokens(x,mask)

    def forward(self,x,mask=None): return self.decoder(self.tokens(x,mask))


SUPPORTED_ENCODER_BACKENDS=frozenset(('legacy',BACKEND,'hf_patchtst_official_ssl_v1','hf_patchtst_legacy_joint_v1','hf_patchtst_portable_joint_v1','gevis_native_temporal_blind_v2','gevis_native_multiscale_blind_v3'))


def load_encoder(checkpoint,legacy_class):
    """Local reconstruction only; loading a trained GEVIS bundle never downloads."""
    cfg=checkpoint['model_config']
    if checkpoint.get('encoder_backend')=='gevis_native_multiscale_blind_v3':
        from .multiscale_patchtst import MultiscalePatchEncoder,CONTRACT
        if checkpoint.get('reconstruction_contract')!=CONTRACT:raise ValueError('Multiscale blind reconstruction contract mismatch')
        model=MultiscalePatchEncoder(len(checkpoint['sensor_columns']),checkpoint['patch_length'],**cfg)
    elif checkpoint.get('encoder_backend')=='gevis_native_temporal_blind_v2':
        from .temporal_patchtst import TemporalPatchEncoder, CONTRACT
        if checkpoint.get('reconstruction_contract')!=CONTRACT:raise ValueError('Temporal blind reconstruction contract mismatch')
        model=TemporalPatchEncoder(len(checkpoint['sensor_columns']),checkpoint['patch_length'],**cfg)
    elif checkpoint.get('encoder_backend') in ('hf_patchtst_legacy_joint_v1','hf_patchtst_portable_joint_v1'):
        from .hybrid_patchtst import HybridPatchEncoder,PortableJointPatchEncoder
        cls=PortableJointPatchEncoder if checkpoint['encoder_backend']=='hf_patchtst_portable_joint_v1' else HybridPatchEncoder
        model=cls(len(checkpoint['sensor_columns']),checkpoint['patch_length'],checkpoint['window_length'],
            public_config=checkpoint['pretrained_config'],**cfg)
        model.pretrained_identity=deepcopy(checkpoint['pretrained_identity'])
    elif checkpoint.get('encoder_backend')=='hf_patchtst_official_ssl_v1':
        from .official_patchtst import OfficialPatchEncoder
        model=OfficialPatchEncoder(len(checkpoint['sensor_columns']),hf_config=checkpoint['pretrained_config'])
        model.pretrained_identity=deepcopy(checkpoint['pretrained_identity'])
    elif checkpoint.get('encoder_backend')==BACKEND:
        model=PretrainedPatchEncoder(len(checkpoint['sensor_columns']),checkpoint['patch_length'],
            cfg['d_model'],hf_config=checkpoint['pretrained_config'])
        model.pretrained_identity=deepcopy(checkpoint['pretrained_identity'])
    elif checkpoint.get('encoder_backend','legacy')=='legacy':
        model=legacy_class(len(checkpoint['sensor_columns']),checkpoint['patch_length'],**cfg)
    else: raise ValueError(f"Unknown STEP 1 encoder backend {checkpoint.get('encoder_backend')!r}; supported: {sorted(SUPPORTED_ENCODER_BACKENDS)}")
    model.load_state_dict(checkpoint['model_state'],strict=True)
    return model


def detail_loss(prediction,target):
    """Explicit fluctuation and peak reconstruction; no synthetic fault labels."""
    variability=(prediction.std(-1,unbiased=False)-target.std(-1,unbiased=False)).square().mean()
    peaks=(prediction-target).square().amax(-1).mean()
    return variability,peaks
