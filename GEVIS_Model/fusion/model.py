"""Original GEVIS Fusion tensor structure, independent of domain heads."""
import math
import torch
from torch import nn
from . import text as gevis_backbone
from .text import frozen_text
ANSWER_TYPES={'BOOLEAN':0,'CHOICE':1,'ORDINAL':2}

def numeric_state(encoder,current,reference):
    from .temporal_bridge import supports, numeric_state as temporal_state
    if supports(encoder): return temporal_state(encoder,current,reference)
    a=encoder.tokens(current); b=encoder.tokens(reference)
    current_patch,_=encoder.patchify(current); reference_patch,_=encoder.patchify(reference)
    current_reconstruction=encoder(current); reference_reconstruction=encoder(reference)
    fdc_features=torch.stack([current_patch.mean(-1),reference_patch.mean(-1),(current_patch.mean(-1)-reference_patch.mean(-1)).abs(),current_patch.std(-1,unbiased=False),reference_patch.std(-1,unbiased=False),(current_patch-reference_patch).abs().mean(-1),(current_reconstruction-current_patch).pow(2).mean(-1),(reference_reconstruction-reference_patch).pow(2).mean(-1)],-1)
    return torch.cat([a,b,a-b,(a-b).abs(),fdc_features],-1)

class ByteTextEncoder(nn.Module):
    def __init__(self,d_model=128,max_bytes=320,layers=2,dropout=.1):
        super().__init__(); self.max_bytes=max_bytes; self.token=nn.Embedding(257,d_model,padding_idx=0); self.position=nn.Embedding(max_bytes,d_model)
        block=nn.TransformerEncoderLayer(d_model,4,4*d_model,dropout,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d_model)
    def forward(self,texts,device):
        encoded=[list(text.encode('utf-8')[:self.max_bytes]) for text in texts]; width=max(1,max(map(len,encoded)))
        ids=torch.zeros((len(texts),width),dtype=torch.long,device=device)
        for i,raw in enumerate(encoded):
            if raw: ids[i,:len(raw)]=torch.tensor([x+1 for x in raw],device=device)
        mask=ids.eq(0); pos=torch.arange(width,device=device)[None]
        z=self.encoder(self.token(ids)+self.position(pos),src_key_padding_mask=mask)
        keep=(~mask).unsqueeze(-1); return self.norm((z*keep).sum(1)/keep.sum(1).clamp_min(1))

class GEVIS(nn.Module):
    def __init__(self,numeric_dim=256,d_model=128,dropout=.1,text_encoder_name='byte',text_encoder_revision=None):
        super().__init__(); self.text_encoder_name=text_encoder_name; self.text_encoder_revision=text_encoder_revision; self.text=ByteTextEncoder(d_model,dropout=dropout) if text_encoder_name=='byte' else gevis_backbone.E5TextEncoder(d_model,model_name=text_encoder_name,revision=text_encoder_revision or gevis_backbone.DEFAULT_TEXT_ENCODER_REVISION); self.numeric_input_norm=nn.LayerNorm(numeric_dim); self.numeric=nn.Linear(numeric_dim,d_model); self.numeric_key_norm=nn.LayerNorm(d_model); self.descriptor_norm=nn.LayerNorm(d_model); self.fusion_gate=nn.Sequential(nn.Linear(2*d_model,d_model),nn.Sigmoid()); self.type_emb=nn.Embedding(3,d_model)
        from .temporal_bridge import install
        install(self,numeric_dim,d_model)
        self.cross=nn.MultiheadAttention(d_model,4,dropout,batch_first=True); self.answer_norm=nn.LayerNorm(d_model); self.support_head=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,64),nn.GELU(),nn.Linear(64,1)); self.change_head=nn.Sequential(nn.LayerNorm(2*d_model),nn.Linear(2*d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,1))
        self.semantic_heads=nn.ModuleDict({name:nn.Linear(d_model,size) for name,size in {'statistic':5,'direction':3,'time_scope':3}.items()})
        self.criterion_proj=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.LayerNorm(d_model)); self.temperature=nn.Parameter(torch.tensor(0.0))
    def load_state_dict(self,*args,**kwargs):
        self._text_cache=None
        return super().load_state_dict(*args,**kwargs)
    def _text(self,texts,device): return frozen_text(self,texts,device)
    def support_probability(self,questions): return self.support_head(self._text(questions,next(self.parameters()).device)).squeeze(-1).sigmoid()
    def forward(self,numeric_tokens,questions,answer_types,criteria,token_descriptions=None,return_fusion=False):
        device=numeric_tokens.device; type_ids=torch.tensor([ANSWER_TYPES[x] for x in answer_types],device=device)
        if numeric_tokens.shape[-1]!=self.numeric.in_features: raise ValueError(f'numeric state width {numeric_tokens.shape[-1]} does not match GEVIS input width {self.numeric.in_features}; load a GEVIS backbone matching the numeric encoder')
        text_query=self._text(questions,device); q=text_query+self.type_emb(type_ids); numeric_keys=self.numeric_key_norm(self.numeric(self.numeric_input_norm(numeric_tokens))); descriptors=None
        if token_descriptions is not None:
            if len(token_descriptions)!=len(numeric_tokens) or any(len(items)!=numeric_tokens.shape[1] for items in token_descriptions): raise ValueError('token_descriptions must match [batch, numeric_tokens]')
            flat_descriptions=[item for items in token_descriptions for item in items]
            unique=list(dict.fromkeys(flat_descriptions)); lookup={text:index for index,text in enumerate(unique)}
            indices=torch.tensor([lookup[text] for text in flat_descriptions],device=device,dtype=torch.long)
            descriptors=self.descriptor_norm(self._text(unique,device).index_select(0,indices).reshape_as(numeric_keys))
        from .text import attention_inputs
        attention_query,keys=attention_inputs(numeric_keys,descriptors,text_query,q,self.fusion_gate,getattr(self,'fusion_mode','legacy'))
        answer,attention=self.cross(attention_query[:,None],keys,keys,need_weights=True); answer=self.answer_norm(answer[:,0]+q)
        fusion=(answer,q,attention[:,0])
        if return_fusion: return fusion
        return self.decode(fusion,answer_types,criteria),attention[:,0]
    def decode(self,fusion,answer_types,criteria):
        answer,q,_=fusion; device=answer.device
        change_logit=self.change_head(torch.cat([answer,q],-1)).squeeze(-1)
        scale=self.temperature.exp().clamp(1e-2,100)
        flat=[item for items in criteria for item in items]; all_emb=self.criterion_proj(self._text(flat,device)); logits=[]; offset=0
        for i,items in enumerate(criteria):
            emb=all_emb[offset:offset+len(items)]; offset+=len(items); score=scale*(emb@answer[i])/math.sqrt(answer.shape[-1])
            if answer_types[i]=='BOOLEAN' and len(items)==2: score=score+torch.stack([-change_logit[i],change_logit[i]])
            logits.append(score)
        return logits

def numeric_token_descriptions(step,meta,patch_count,batch_size):
    names=meta.get('sensor_descriptors',{name:name for name in meta['sensor_columns']}); phase=meta.get('step_descriptor',step)
    one=gevis_backbone.token_descriptions(phase,[names.get(sensor,sensor) for sensor in meta['sensor_columns']],patch_count)
    return [one for _ in range(batch_size)]
