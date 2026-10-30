"""Original custom encoder kept only for explicit legacy comparisons."""
import torch
from torch import nn

class PatchEncoder(nn.Module):
    def __init__(self,n_sensors,patch_len,d_model=64,nhead=4,layers=2,dropout=.1):
        super().__init__(); self.n_sensors=n_sensors; self.patch_len=patch_len; self.d_model=d_model
        self.input=nn.Linear(patch_len,d_model); self.sensor_emb=nn.Embedding(n_sensors,d_model); self.patch_emb=nn.Embedding(64,d_model); self.mask_token=nn.Parameter(torch.zeros(d_model))
        block=nn.TransformerEncoderLayer(d_model,nhead,4*d_model,dropout,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d_model); self.decoder=nn.Linear(d_model,patch_len)
    def patchify(self,x):
        b,t,s=x.shape; p=t//self.patch_len
        return x[:,:p*self.patch_len].reshape(b,p,self.patch_len,s).permute(0,3,1,2).reshape(b,s*p,self.patch_len),p
    def tokens(self,x,mask=None):
        z,p=self.patchify(x); b=z.shape[0]; sid=torch.arange(self.n_sensors,device=x.device).repeat_interleave(p); pid=torch.arange(p,device=x.device).repeat(self.n_sensors)
        content=self.input(z)
        if mask is not None: content=torch.where(mask[...,None],self.mask_token[None,None],content)
        h=content+self.sensor_emb(sid)[None]+self.patch_emb(pid)[None]
        return self.norm(self.encoder(h))
    def forward(self,x,mask=None): return self.decoder(self.tokens(x,mask))
