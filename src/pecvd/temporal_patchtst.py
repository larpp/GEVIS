"""Native temporal/cross-sensor SSL candidate. No public-pretraining claim."""
import math
import torch
from torch import nn

BACKEND = 'gevis_native_temporal_blind_v2'
CONTRACT = 'complementary_hidden_reconstruction_temporal_v2'


class AxialBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(width, heads, 4*width, dropout,
                                                   batch_first=True, norm_first=True)
        self.sensor = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(2*width, width)
        self.local = nn.Conv1d(width, width, 3, padding=1, groups=width)

    def forward(self, x):
        b,s,p,d=x.shape
        h=x.reshape(b*s,p,d)
        h=self.temporal(h+self.local(h.transpose(1,2)).transpose(1,2))
        h=h.reshape(b,s,p,d)
        q=self.norm(h).transpose(1,2).reshape(b*p,s,d)
        v=self.sensor(q,q,q,need_weights=False)[0].reshape(b,p,s,d).transpose(1,2)
        return h+torch.sigmoid(self.gate(torch.cat([h,v],-1)))*v


class TemporalPatchEncoder(nn.Module):
    supports_native_sensor_transfer=True
    def __init__(self,n_sensors,patch_len,d_model=96,nhead=8,layers=3,dropout=.05):
        super().__init__()
        if patch_len<2 or d_model%2: raise ValueError('Patch width >=2 and even latent width required')
        self.n_sensors=n_sensors;self.patch_len=patch_len;self.d_model=d_model
        self.input=nn.Linear(patch_len,d_model)
        self.detail=nn.Linear(patch_len-1+3,d_model)
        self.mask_token=nn.Parameter(torch.zeros(d_model))
        self.blocks=nn.ModuleList([AxialBlock(d_model,nhead,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(d_model)
        self.decoder=nn.Linear(d_model,patch_len)
        self.reconstruction_contract=CONTRACT

    def patchify(self,x):
        if x.ndim!=3 or x.shape[2]<1 or x.shape[1]%self.patch_len or not torch.isfinite(x).all():
            raise ValueError('Complete native time/sensor patches required')
        b,t,s=x.shape;p=t//self.patch_len
        if not 2<=p<=64:raise ValueError('Temporal position capacity is 2..64 complete patches')
        return x.reshape(b,p,self.patch_len,s).permute(0,3,1,2).reshape(b,s*p,self.patch_len),p

    def tokens(self,x,mask=None):
        z,p=self.patchify(x);b=len(x)
        if mask is not None:
            if mask.shape!=z.shape[:2] or mask.dtype!=torch.bool:raise ValueError('Boolean native-grid mask required')
            z=z.masked_fill(mask[...,None],0)
        stats=torch.stack([z.mean(-1),z.std(-1,unbiased=False),z[...,-1]-z[...,0]],-1)
        h=self.input(z)+self.detail(torch.cat([z.diff(dim=-1),stats],-1))
        if mask is not None:h=torch.where(mask[...,None],self.mask_token,h)
        pos=torch.arange(p,device=x.device,dtype=h.dtype)[:,None]
        freq=torch.exp(torch.arange(0,self.d_model,2,device=x.device,dtype=h.dtype)*(-math.log(10000.)/self.d_model))
        pe=torch.stack([(pos*freq).sin(),(pos*freq).cos()],-1).flatten(-2)
        h=h.reshape(b,x.shape[-1],p,self.d_model)+pe
        h=self.encode_grid(h)
        return self.norm(h).reshape(b,x.shape[-1]*p,self.d_model)

    def encode_grid(self,h):
        for block in self.blocks:h=block(h)
        return h

    def forward(self,x,mask=None):return self.decoder(self.tokens(x,mask))

    def blind_prediction(self,x,mask=None):
        z,p=self.patchify(x)
        if mask is None:
            mask=(torch.arange(x.shape[-1],device=x.device)[:,None]+torch.arange(p,device=x.device)[None,:])%2==0
            mask=mask.reshape(1,-1).expand(len(x),-1)
        first=self(x,mask);second=self(x,~mask)
        return torch.where(mask[...,None],first,second)


def blind_objective(model,x,mask,config,*,return_prediction=False):
    """All detail losses see held-out predictions, not identity reconstructions."""
    target,p=model.patchify(x);pred=model.blind_prediction(x,mask).float();target=target.float()
    mse=(pred-target).square().mean()
    # Temporal differences include patch boundaries, not just within-patch slopes.
    a=pred.reshape(len(x),model.n_sensors,-1);b=target.reshape_as(a)
    slope=(a.diff(dim=-1)-b.diff(dim=-1)).square().mean()
    def relation(v):
        v=v-v.mean(-1,keepdim=True)
        v=v/v.square().sum(-1,keepdim=True).clamp_min(1e-4).sqrt()
        return v@v.transpose(-1,-2)
    relationship=(relation(a)-relation(b)).square().mean()
    variability=(pred.std(-1,unbiased=False)-target.std(-1,unbiased=False)).square().mean()
    peak=(pred-target).square().amax(-1).mean()
    parts=dict(masked_mse=mse,visible_mse=mse*0,slope=slope,relationship=relationship,variability=variability,peak=peak)
    total=mse+config.slope_weight*slope+config.relationship_weight*relationship+config.variability_weight*variability+config.peak_weight*peak
    return (total,parts,pred,target) if return_prediction else (total,parts)
