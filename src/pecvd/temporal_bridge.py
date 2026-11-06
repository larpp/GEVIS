"""Shared STEP2/runtime bridge: observed latent + paired temporal evidence."""
import torch
from torch import nn

CONTRACT='paired_temporal_blind_408_v2'
NUMERIC_DIM=408  # four 96-D latent blocks + 16 temporal + historical trailing 8
MULTISCALE_DIM=440
MULTISCALE_CONTRACT='paired_relational_multiscale_440_v3'


def supports(encoder):
    from .temporal_patchtst import CONTRACT as reconstruction
    from .multiscale_patchtst import CONTRACT as multiscale
    return getattr(encoder,'reconstruction_contract',None) in (reconstruction,multiscale)


@torch.no_grad()
def numeric_state(encoder,current,reference):
    if not supports(encoder):raise ValueError('Explicit temporal reconstruction contract required')
    if current.shape!=reference.shape:raise ValueError('Aligned current/reference windows required')
    paired=torch.cat([current,reference],0)
    a,b=encoder.tokens(paired).chunk(2,0)
    cp,p=encoder.patchify(current);rp,_=encoder.patchify(reference)
    cr,rr=encoder.blind_prediction(paired).chunk(2,0)
    cm=cp.mean(-1);rm=rp.mean(-1);cs=cp.std(-1,unbiased=False);rs=rp.std(-1,unbiased=False)
    cd=cp.diff(dim=-1);rd=rp.diff(dim=-1)
    def progression(v):
        v=v.reshape(len(current),current.shape[-1],p)
        return torch.cat([torch.zeros_like(v[...,:1]),v.diff(dim=-1)],-1).reshape_as(cm)
    cprog=progression(cm);rprog=progression(rm)
    delta=cp-rp
    centered_c=cp-cm[...,None];centered_r=rp-rm[...,None]
    corr=(centered_c*centered_r).sum(-1)/(centered_c.square().sum(-1)*centered_r.square().sum(-1)).clamp_min(1e-8).sqrt()
    extra=torch.stack([cm-rm,cs-rs,cd.mean(-1),rd.mean(-1),(cd-rd).mean(-1),
        delta.abs().amax(-1),delta.square().mean(-1).sqrt(),(cd-rd).std(-1,unbiased=False),
        cd.std(-1,unbiased=False),rd.std(-1,unbiased=False),cprog,rprog,cprog-rprog,corr,
        (cr-rr).mean(-1),(cr-rr).abs().mean(-1)],-1)
    old=torch.stack([cm,rm,(cm-rm).abs(),cs,rs,delta.abs().mean(-1),
                     (cr-cp).square().mean(-1),(rr-rp).square().mean(-1)],-1)
    from .multiscale_patchtst import CONTRACT as multiscale
    multi=getattr(encoder,'reconstruction_contract',None)==multiscale
    if multi:
        from .relational_features import relational_features
        extra=torch.cat([extra,relational_features(current,reference,encoder.patch_len)],-1)
    state=torch.cat([a,b,a-b,(a-b).abs(),extra,old],-1)
    if state.shape[-1]!=(MULTISCALE_DIM if multi else NUMERIC_DIM):raise ValueError('Temporal candidate requires 96-D encoder latents')
    return state


class GatedTemporalAdapter(nn.Module):
    """Keep numeric amplitudes separate from latent LayerNorm before fusion."""
    def __init__(self,in_features,out_features):
        super().__init__()
        if in_features not in (NUMERIC_DIM,MULTISCALE_DIM):raise ValueError('Unknown temporal bridge width')
        self.in_features=in_features;self.out_features=out_features
        self.latent=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,out_features))
        self.temporal=nn.Sequential(nn.Linear(in_features-384,out_features),nn.GELU(),nn.Linear(out_features,out_features))
        self.gate=nn.Linear(2*out_features,out_features)

    def forward(self,x):
        h=self.latent(x[...,:384]);v=x[...,384:]
        v=self.temporal(v.sign()*torch.log1p(v.abs()))
        return h+torch.sigmoid(self.gate(torch.cat([h,v],-1)))*v


def install(model,numeric_dim,d_model):
    if numeric_dim in (NUMERIC_DIM,MULTISCALE_DIM):
        model.numeric_input_norm=nn.Identity()
        model.numeric=GatedTemporalAdapter(numeric_dim,d_model)


def install_checkpoint_adapter(model,state,numeric_dim,d_model):
    """Reconstruct the saved adapter, including in kernels with stale GEVIS classes.

    Do not infer a legacy adapter from width alone or relax strict loading.
    Component load_state_dict remains responsible for complete tensor validation.
    """
    if 'numeric.latent.0.weight' in state:
        model.numeric_input_norm=nn.Identity()
        model.numeric=GatedTemporalAdapter(numeric_dim,d_model)
    elif 'numeric.weight' in state and 'numeric_input_norm.weight' in state:
        model.numeric_input_norm=nn.LayerNorm(numeric_dim)
        model.numeric=nn.Linear(numeric_dim,d_model)
    else:
        raise ValueError('Unrecognized checkpoint numeric adapter; refusing partial weight loading')
