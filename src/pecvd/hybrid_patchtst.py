"""Original joint-sensor encoder plus a pinned public PatchTST feature residual.

Public weights are reused in the PUBLIC branch, not transplanted into an
incompatible custom Transformer. Native sensor/time tokens and raw-scale path
remain intact. The public branch alone resamples each native patch to width 12.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .legacy_patchtst import PatchEncoder
from .official_patchtst import OfficialPatchEncoder

BACKEND='hf_patchtst_legacy_joint_v1'
PORTABLE_BACKEND='hf_patchtst_portable_joint_v1'


class HybridPatchEncoder(nn.Module):
    def __init__(self,n_sensors,patch_len,window_length,*,public_config,
                 d_model=96,nhead=8,layers=3,dropout=.05):
        super().__init__()
        if window_length % patch_len or not 2 <= window_length//patch_len <= 64:
            raise ValueError('Original native grid must contain 2..64 complete patches')
        self.n_sensors=n_sensors;self.patch_len=patch_len;self.d_model=d_model
        self.window_length=window_length
        self.legacy=PatchEncoder(n_sensors,patch_len,d_model,nhead,layers,dropout)
        self.public=OfficialPatchEncoder(n_sensors,hf_config=public_config)
        if public_config['context_length']!=(window_length//patch_len)*public_config['patch_length']:
            raise ValueError('Public context must map one-to-one onto native patches')
        self.public_projection=nn.Linear(self.public.d_model,d_model)
        self.public_norm=nn.LayerNorm(d_model)
        self.public_gain=nn.Parameter(torch.zeros(()))
        self.hf_config=deepcopy(public_config);self.pretrained_identity=None
        self.configure_finetuning(None)

    @classmethod
    def from_public(cls,path,n_sensors,patch_len,window_length,**legacy_config):
        public=OfficialPatchEncoder.from_public(path,n_sensors,(window_length//patch_len)*12)
        model=cls(n_sensors,patch_len,window_length,public_config=public.hf_config,**legacy_config)
        model.public=public;model.configure_finetuning(None)
        model.pretrained_identity={**deepcopy(public.pretrained_identity),
            'architecture':'original joint-sensor PatchEncoder + gated public feature residual',
            'public_usage':'official temporal features; official SSL head unused and frozen',
            'input_adaptation':'public branch only: each native patch linearly resampled to public width; original target/grid/raw path unchanged',
            'new_parameters':'128-to-native projection, layer norm and zero-initialized residual gain',
            'legacy_initialization':'not yet bound'}
        return model

    @property
    def pretrained(self):return self.public.pretrained

    @property
    def decoder(self):return self.legacy.decoder

    def configure_finetuning(self,last_layers=None):
        self.legacy.requires_grad_(True)
        self.public.configure_finetuning(last_layers)
        self.public.hf.head.requires_grad_(False)  # NOT used for reconstruction.

    def train(self,mode=True):
        super().train(mode)
        # Preserve the public branch's frozen-state policy after recursive train.
        self.public.train(mode)
        return self

    def patchify(self,x):
        if x.ndim!=3 or x.shape[1]!=self.window_length or not torch.isfinite(x).all():
            raise ValueError('Finite input on the original native context is required')
        return self.legacy.patchify(x)

    def _public_tokens(self,x,mask):
        z,patches=self.patchify(x);batch,time,sensors=x.shape
        if mask is not None:
            if mask.shape!=z.shape[:2]:raise ValueError('Mask/native grid mismatch')
            # Remove hidden values BEFORE any interpolation or window scaling.
            z=z.masked_fill(mask[...,None],0.)
        width=self.hf_config['patch_length']
        resized=F.interpolate(z.reshape(-1,1,self.patch_len),size=width,mode='linear',align_corners=False)
        series=resized.reshape(batch,sensors,patches*width).transpose(1,2)
        return self.public.channel_independent_tokens(series,mask)

    def _tokens(self,x,mask,*,joint):
        z,patches=self.patchify(x);batch,time,sensors=x.shape
        if joint and sensors!=self.n_sensors:raise ValueError('Native sensor order/count required')
        content=self.legacy.input(z)
        if mask is not None:
            if mask.shape!=z.shape[:2]:raise ValueError('Mask/native grid mismatch')
            content=torch.where(mask[...,None],self.legacy.mask_token,content)
        public=self.public_norm(self.public_projection(self._public_tokens(x,mask)))
        content=content+self.public_gain.tanh()*public
        position=self.legacy.patch_emb(torch.arange(patches,device=x.device))
        if joint:
            sid=torch.arange(sensors,device=x.device).repeat_interleave(patches)
            h=content+self.legacy.sensor_emb(sid)[None]+position.repeat(sensors,1)[None]
            return self.legacy.norm(self.legacy.encoder(h))
        # Explicit frozen external adapter: neutral source ID, independent channels.
        h=content.reshape(batch*sensors,patches,self.d_model)
        h=h+self.legacy.sensor_emb.weight.mean(0)+position[None]
        return self.legacy.norm(self.legacy.encoder(h)).reshape(batch,sensors*patches,self.d_model)

    def tokens(self,x,mask=None):return self._tokens(x,mask,joint=True)

    def forward(self,x,mask=None):return self.decoder(self.tokens(x,mask))

    def channel_independent_tokens(self,x,mask=None):return self._tokens(x,mask,joint=False)

    def channel_independent_forward(self,x,mask=None):return self.decoder(self.channel_independent_tokens(x,mask))

    def load_legacy(self,checkpoint,*,sensor_columns,mean,std,run_split,source_sha256):
        """Exact local reuse only; incompatible data/model identity must fail closed."""
        import numpy as np
        expected=self.legacy.state_dict()
        if checkpoint.get('encoder_backend','legacy')!='legacy':raise ValueError('Warm start requires the original legacy encoder')
        if list(checkpoint['sensor_columns'])!=list(sensor_columns):raise ValueError('Legacy sensor order mismatch')
        if checkpoint['window_length']!=self.window_length or checkpoint['patch_length']!=self.patch_len:
            raise ValueError('Legacy native grid mismatch')
        if checkpoint.get('source_sha256')!=source_sha256 or checkpoint.get('run_split')!=run_split:
            raise ValueError('Legacy source/split mismatch; no cross-split warm start')
        for key,value in [('mean',mean),('std',std)]:
            if not np.allclose(np.asarray(checkpoint[key]).reshape(-1),np.asarray(value).reshape(-1),rtol=1e-5,atol=1e-6):
                raise ValueError('Legacy TRAIN scaler mismatch')
        state=checkpoint['model_state']
        if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape for k in expected):
            raise ValueError('Legacy architecture mismatch')
        # Shapes do not encode head count/dropout; verify those independently.
        cfg=checkpoint['model_config']
        first=self.legacy.encoder.layers[0]
        if (cfg['nhead']!=first.self_attn.num_heads or cfg['layers']!=len(self.legacy.encoder.layers)
                or cfg['dropout']!=first.dropout.p):raise ValueError('Legacy attention configuration mismatch')
        self.legacy.load_state_dict(state,strict=True)
        if self.pretrained_identity is not None:self.pretrained_identity['legacy_initialization']='exact compatible local legacy checkpoint'


class PortableJointPatchEncoder(HybridPatchEncoder):
    """Same permutation-equivariant joint path for source and unseen sensor sets.

    Channel identity is a shared trainable token, not an equipment-specific index.
    Time positions, raw patches, public temporal features and joint sensor/time
    attention remain. Sensor meanings are supplied to STEP 2 as descriptors.
    This changes the source training architecture: old hybrid weights alone are
    NOT a trained portable model. Legacy initialization is only a warm start.
    """
    supports_native_sensor_transfer=True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.shared_sensor_token=nn.Parameter(self.legacy.sensor_emb.weight.detach().mean(0).clone())
        self.legacy.sensor_emb.requires_grad_(False)  # Retained solely for exact legacy state provenance.

    def configure_finetuning(self,last_layers=None):
        super().configure_finetuning(last_layers)
        self.legacy.sensor_emb.requires_grad_(False)

    @classmethod
    def from_public(cls,*args,**kwargs):
        model=super().from_public(*args,**kwargs)
        model.pretrained_identity.update(
            architecture='portable shared-sensor token + native joint attention + public residual',
            sensor_transfer='same learned shared token and joint forward for every sensor count; no external averaging')
        return model

    def load_legacy(self,*args,**kwargs):
        super().load_legacy(*args,**kwargs)
        with torch.no_grad():self.shared_sensor_token.copy_(self.legacy.sensor_emb.weight.mean(0))
        self.legacy.sensor_emb.requires_grad_(False)
        self.pretrained_identity['legacy_initialization']='legacy temporal/joint warm start; sensor IDs replaced BEFORE source training'

    def tokens(self,x,mask=None):
        z,patches=self.patchify(x);batch,_,sensors=x.shape
        if sensors<1:raise ValueError('Nonempty sensor set required')
        content=self.legacy.input(z)
        if mask is not None:
            if mask.shape!=z.shape[:2]:raise ValueError('Mask/native grid mismatch')
            content=torch.where(mask[...,None],self.legacy.mask_token,content)
        public=self.public_norm(self.public_projection(self._public_tokens(x,mask)))
        positions=self.legacy.patch_emb(torch.arange(patches,device=x.device)).repeat(sensors,1)
        h=content+self.public_gain.tanh()*public+self.shared_sensor_token+positions[None]
        return self.legacy.norm(self.legacy.encoder(h))
