"""Long-window native encoder: short-context learning plus coarse global memory.

Identical rules for every Step; no manufacturing names or labels are embedded.
All pooling sees masked content only. Full native rows/token order are retained.
"""
from collections import Counter,defaultdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .temporal_patchtst import TemporalPatchEncoder,AxialBlock,blind_objective

BACKEND='gevis_native_multiscale_blind_v3'
CONTRACT='complementary_hidden_multiscale_v3'


class MultiscalePatchEncoder(TemporalPatchEncoder):
    def __init__(self,n_sensors,patch_len,d_model=96,nhead=8,layers=3,dropout=.05,
                 local_patches=8,coarse_factor=4):
        super().__init__(n_sensors,patch_len,d_model,nhead,layers,dropout)
        if local_patches<2 or coarse_factor<2:raise ValueError('Local/global context sizes must be >=2')
        self.local_patches=local_patches;self.coarse_factor=coarse_factor
        self.global_block=AxialBlock(d_model,nhead,dropout)
        self.scale_gate=nn.Linear(2*d_model,d_model)
        self.reconstruction_contract=CONTRACT

    def encode_grid(self,h):
        b,s,p,d=h.shape
        # One batched local call when divisible; no cross-window mixing.
        if p%self.local_patches==0:
            chunks=p//self.local_patches
            local=h.reshape(b,s,chunks,self.local_patches,d).permute(0,2,1,3,4).reshape(b*chunks,s,self.local_patches,d)
            local=super().encode_grid(local)
            local=local.reshape(b,chunks,s,self.local_patches,d).permute(0,2,1,3,4).reshape_as(h)
        else:
            encode_local=super().encode_grid
            local=torch.cat([encode_local(h[:,:,start:start+self.local_patches])
                             for start in range(0,p,self.local_patches)],2)
        coarse_count=max(2,math.ceil(p/self.coarse_factor))
        coarse=F.adaptive_avg_pool1d(h.reshape(b*s,p,d).transpose(1,2),coarse_count)
        coarse=self.global_block(coarse.transpose(1,2).reshape(b,s,coarse_count,d))
        global_context=F.interpolate(coarse.reshape(b*s,coarse_count,d).transpose(1,2),size=p,
                                     mode='linear',align_corners=False).transpose(1,2).reshape_as(h)
        return local+torch.sigmoid(self.scale_gate(torch.cat([local,global_context],-1)))*global_context


def crop_window(x,patch_len,divisor,*,random):
    """Crop each example independently; never concatenate runs or resample time."""
    patches=x.shape[1]//patch_len
    size=max(2,patches//divisor)
    if size>=patches:return x
    starts=(torch.randint(patches-size+1,(len(x),),device=x.device) if random else
            torch.full((len(x),),(patches-size)//2,device=x.device,dtype=torch.long))
    indices=starts[:,None]*patch_len+torch.arange(size*patch_len,device=x.device)[None,:]
    return x.gather(1,indices[...,None].expand(-1,-1,x.shape[-1]))


def lag_relationship_loss(pred,target,max_lag=4):
    """Reconstruct cross-sensor signed lag correlations, not causal labels."""
    def corr(a,b):
        a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
        a=a/a.square().sum(-1,keepdim=True).clamp_min(1e-4).sqrt()
        b=b/b.square().sum(-1,keepdim=True).clamp_min(1e-4).sqrt()
        return a@b.transpose(-1,-2)
    losses=[]
    for lag in range(1,min(max_lag,pred.shape[-1]-2)+1):
        losses.append((corr(pred[...,:-lag],pred[...,lag:])-corr(target[...,:-lag],target[...,lag:])).square().mean())
    return torch.stack(losses).mean() if losses else pred.sum()*0


def multiscale_objective(model,x,mask,config):
    # Fixed full-context blind MSE remains checkpoint selection, never Test.
    full,parts,pred,target=blind_objective(model,x,mask,config,return_prediction=True)
    crops=[];crop_mses=[]
    divisors=(int(torch.randint(0,2,()).item())*2+2,) if model.training else (2,4)
    for divisor in divisors:
        short=crop_window(x,model.patch_len,divisor,random=model.training)
        _,p=model.patchify(short)
        crop_mask=(torch.arange(short.shape[-1],device=x.device)[:,None]+torch.arange(p,device=x.device)[None,:])%2==0
        crop_mask=crop_mask.reshape(1,-1).expand(len(x),-1)
        loss,short_parts=blind_objective(model,short,crop_mask,config)
        crops.append(loss);crop_mses.append(short_parts['masked_mse'])
    # Reuse hidden reconstruction semantics; only hidden predictions feed lags.
    pred=pred.reshape(len(x),x.shape[-1],-1);target=target.reshape_as(pred)
    lag=lag_relationship_loss(pred,target)
    parts.update(crop_mse=torch.stack(crop_mses).mean(),lag_relationship=lag)
    return full+.5*torch.stack(crops).mean()+.25*config.relationship_weight*lag,parts


def balanced_window_weights(metadata):
    """Equal total mass per run, then per occupied within-run position bin."""
    runs=list(map(str,metadata['process_id']));positions=list(metadata['position_bin'])
    if not runs or len(runs)!=len(positions):raise ValueError('Aligned nonempty TRAIN window metadata required')
    counts=Counter(zip(runs,positions));bins=defaultdict(set)
    for run,position in counts:bins[run].add(position)
    return [1./(len(bins[run])*counts[(run,position)]) for run,position in zip(runs,positions)]


def training_sampler(model,size,seed,training):
    weights=getattr(model,'window_sample_weights',None)
    if not training or weights is None:return None
    if len(weights)!=size:raise ValueError('TRAIN sampler/window identity mismatch')
    return torch.utils.data.WeightedRandomSampler(weights,size,replacement=True,
                    generator=torch.Generator().manual_seed(seed))
