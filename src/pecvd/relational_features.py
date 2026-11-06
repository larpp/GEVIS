"""Observed paired sensor/time features. No intervention IDs or labels accepted."""
import torch
from torch.nn import functional as F

CONTRACT='paired_local_relationship_lag_stuck_32_v1'


def relational_features(current,reference,patch_len):
    """Return [batch, sensor*patch, 32], preserving the encoder's native order."""
    if current.shape!=reference.shape or current.ndim!=3 or current.shape[1]%patch_len:
        raise ValueError('Aligned complete paired native patches required')
    b,t,s=current.shape;p=t//patch_len
    width=3*patch_len
    def windows(x):
        x=x.float().transpose(1,2).reshape(b*s,1,t)
        return F.pad(x,(patch_len,patch_len),mode='replicate').unfold(-1,width,patch_len).reshape(b,s,p,width)
    a=windows(current);r=windows(reference)
    def centered(x):return x-x.mean(-1,keepdim=True)
    def unit(x):
        x=centered(x)
        return x/x.square().sum(-1,keepdim=True).clamp_min(1e-8).sqrt()
    def pairs(x,y):return torch.einsum('bspw,bkpw->bpsk',unit(x),unit(y))
    ac=pairs(a,a);rc=pairs(r,r)
    diagonal=torch.eye(s,dtype=torch.bool,device=a.device)[None,None]
    ac=ac.masked_fill(diagonal,0);rc=rc.masked_fill(diagonal,0);delta=ac-rc
    count=max(1,s-1)
    # Partner is selected from observed reference only, never injected truth.
    partner=rc.abs().argmax(-1,keepdim=True)
    ref_partner=rc.gather(-1,partner).squeeze(-1)
    cur_partner=ac.gather(-1,partner).squeeze(-1)
    def pair_axis(v):return v.transpose(1,2)
    features=[pair_axis(delta.abs().sum(-1)/count),pair_axis(delta.abs().amax(-1)),
        pair_axis(delta.square().sum(-1).div(count).sqrt()),pair_axis(cur_partner-ref_partner),
        pair_axis(ref_partner),pair_axis(cur_partner),
        pair_axis(rc.clamp_min(0).sum(-1)/count),pair_axis(ac.clamp_min(0).sum(-1)/count),
        pair_axis(rc.clamp_max(0).sum(-1)/count),pair_axis(ac.clamp_max(0).sum(-1)/count)]
    for lag in (1,2):
        difference=(pairs(a[...,:-lag],a[...,lag:])-pairs(r[...,:-lag],r[...,lag:])).masked_fill(diagonal,0)
        features.extend([pair_axis(difference.abs().amax(-1)),pair_axis(difference.abs().sum(-1)/count)])
    for v in (r,a):
        covariance=torch.einsum('bspw,bkpw->bpsk',centered(v),centered(v))/width
        features.append(pair_axis(covariance.masked_fill(diagonal,0).abs().sum(-1)/count))
    # Native sample shifts, not arbitrary sensor IDs or physical seconds.
    lags=(-4,-2,-1,0,1,2,4)
    correlations=[];errors=[]
    for lag in lags:
        aa,rr=(a[...,lag:],r[...,:-lag]) if lag>0 else ((a[...,:lag],r[...,-lag:]) if lag<0 else (a,r))
        correlations.append((unit(aa)*unit(rr)).sum(-1))
        errors.append((aa-rr).abs().mean(-1))
    correlations=torch.stack(correlations,-1);errors=torch.stack(errors,-1)
    best=correlations.argmax(-1,keepdim=True);zero=correlations[...,3]
    best=torch.where(correlations.gather(-1,best)>zero[...,None]+1e-6,best,torch.full_like(best,3))
    best_corr=correlations.gather(-1,best).squeeze(-1)
    lag_values=torch.tensor(lags,device=a.device,dtype=a.dtype)
    features.extend([zero,best_corr,lag_values[best.squeeze(-1)]/4,best_corr-zero,
        correlations[...,4]-correlations[...,2],correlations[...,5]-correlations[...,1],
        correlations[...,6]-correlations[...,0],errors.gather(-1,best).squeeze(-1)-errors[...,3]])
    da=a.diff(dim=-1);dr=r.diff(dim=-1)
    def longest_equal(d):
        run=torch.zeros_like(d[...,0]);longest=run.clone()
        for i in range(d.shape[-1]):
            run=torch.where(d[...,i].eq(0),run+1,torch.zeros_like(run));longest=torch.maximum(longest,run)
        return longest/d.shape[-1]
    features.extend([da.abs().mean(-1),dr.abs().mean(-1),(da-dr).abs().mean(-1),
        da.eq(0).float().mean(-1),dr.eq(0).float().mean(-1),longest_equal(da),longest_equal(dr),
        torch.log((a.std(-1,unbiased=False)+1e-4)/(r.std(-1,unbiased=False)+1e-4))])
    return torch.stack(features,-1).reshape(b,s*p,32)
