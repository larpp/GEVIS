"""Control48 Student copied from the user-provided reference; no import I/O."""
import math
import numpy as np
import torch
from torch import nn
from pecvd.experiments import fusion_answers as fa


INPUT_CONTRACT = 'asinh_current_lower_upper_signed_scaled_residuals_C5T_v1'


def inputs(values, record):
    """Exact five-channel order confirmed by the user; no second normalization."""
    values = np.asarray(values, np.float32)
    lo = np.asarray(record['lower'], np.float32)
    hi = np.asarray(record['upper'], np.float32)
    if (values.ndim != 2 or values.shape != lo.shape or hi.shape != lo.shape
            or not all(np.isfinite(v).all() for v in (values, lo, hi)) or (lo > hi).any()):
        raise ValueError('Finite complete current/bounds grid required')
    scale = np.maximum(hi - lo, 1.)
    channels = np.stack([values, lo, hi, (values - lo) / scale, (values - hi) / scale], axis=1)
    return torch.as_tensor(np.asinh(channels).transpose(2, 1, 0).copy())

def boundary_tail(values,record,centers,names,offset,cadence):
    lower=np.asarray(record['lower']);upper=np.asarray(record['upper']);span=upper-lower
    scale=np.maximum(span,1.);below=np.maximum(lower-values,0.)/scale;above=np.maximum(values-upper,0.)/scale
    rows=[]
    for name in centers:
        sid=names.index(name);lo=below[:,sid];hi=above[:,sid]
        rows.append([float(np.mean(lo>0)),float(np.mean(hi>0)),float(lo.mean()),float(hi.mean()),
            float(lo.max()),float(hi.max()),float(span[:,sid].mean()),float(span[:,sid].max()),
            math.log1p(offset*cadence),math.log1p(len(values)*cadence)])
    return torch.tensor(rows,dtype=torch.float32)

class GevisStudent(nn.Module):
    """Channel-independent patch Transformer and small text/numeric Fusion."""
    def __init__(self,question_vectors,patch_length,width=48,layers=2):
        super().__init__()
        if question_vectors.ndim!=2 or question_vectors.shape[0]!=3 or question_vectors.shape[1]<1 or not torch.isfinite(question_vectors).all():
            raise ValueError('Three finite Teacher text queries required')
        if patch_length<1 or width<8 or width%4:raise ValueError('Invalid patch/attention shape')
        self.width=width;self.patch_length=int(patch_length);self.dimension=int(question_vectors.shape[1])
        self.register_buffer('question_vectors',question_vectors.detach().float().cpu().clone())
        self.patch=nn.Linear(5*self.patch_length,width)
        block=nn.TransformerEncoderLayer(width,4,dim_feedforward=2*width,dropout=0.,
            activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.sensor_position=nn.Linear(1,width)
        self.boundary=nn.Embedding(2,width)
        self.text=nn.Linear(self.dimension,width)
        self.query_norm=nn.LayerNorm(width);self.key_norm=nn.LayerNorm(width)
        self.cross=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.answer_norm=nn.LayerNorm(width)
        self.fused_out=nn.Linear(width,self.dimension)
        self.numeric_out=nn.Linear(width,self.dimension)

    def encode(self,x):
        channels,_,length=x.shape
        if length%self.patch_length:raise ValueError('Native window must contain complete source-sized patches')
        patches=x.unfold(-1,self.patch_length,self.patch_length).permute(0,2,1,3)
        z=self.patch(patches.reshape(channels,-1,5*self.patch_length))
        positions=torch.arange(z.shape[1],device=z.device,dtype=z.dtype)[:,None]
        frequencies=torch.exp(torch.arange(0,self.width,2,device=z.device,dtype=z.dtype)*(-math.log(10000.)/self.width))
        pe=torch.zeros(z.shape[1],self.width,device=z.device,dtype=z.dtype)
        pe[:,0::2]=torch.sin(positions*frequencies);pe[:,1::2]=torch.cos(positions*frequencies)
        z=self.temporal(z+pe[None])
        sensor=torch.linspace(-1,1,channels,device=z.device,dtype=z.dtype)[:,None]
        return z+self.sensor_position(sensor)[:,None]

    def features(self,state,centers,tail,kinds=fa.KINDS):
        channels,patches,_=state.shape;n=len(centers);k=len(kinds)
        ids=centers.repeat_interleave(k)
        kind_ids=torch.tensor([fa.KINDS.index(kind) for kind in kinds],device=state.device).repeat(n)
        query=self.question_vectors[kind_ids];parts=[]
        for bound in range(2):
            grid=self.key_norm(state+self.boundary.weight[bound]);local=grid.mean(1)[ids]
            q=self.query_norm(self.text(query)+local)
            answer,attention=self.cross(q[None],grid.reshape(1,-1,self.width),grid.reshape(1,-1,self.width),need_weights=True)
            weights=attention[0].reshape(n*k,channels,patches)[torch.arange(n*k,device=state.device),ids]
            attended=(grid[ids]*weights[...,None]).sum(1)/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            fused=self.fused_out(self.answer_norm(answer[0]+q))
            parts.append(torch.cat([fused,query,self.numeric_out(local),self.numeric_out(attended)],-1))
        exact_tail=tail.to(state.device).repeat_interleave(k,dim=0)
        return torch.cat(parts+[exact_tail],-1).reshape(n,k,8*self.dimension+10)

    def forward(self,x,centers,tail,kinds=fa.KINDS):
        return self.features(self.encode(x),centers,tail,kinds)
