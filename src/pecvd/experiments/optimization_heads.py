"""Shared temporal/type outputs. Synthetic mechanism truth is not physical cause.

Target construction is training/scoring only. Model forward and interval decoding
accept no target sensor, injected interval or mechanism metadata.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .injection import KINDS, Injection, inject

TYPE_LABELS=('no_injected_change',)+KINDS


def patch_rows(length,patches):
    if length<1 or patches<1:raise ValueError('Positive native grid required')
    bins=np.array_split(np.arange(length),min(length,patches))
    rows=np.array([(int(ids[0]),int(ids[-1])+1) for ids in bins])
    return rows[np.floor(np.linspace(0,len(rows)-1,patches)).astype(int)] if len(rows)!=patches else rows


def temporal_targets(changed_mask,patches):
    """Any observed changed native row marks its patch; never an input feature."""
    mask=np.asarray(changed_mask)
    if mask.ndim!=2 or mask.dtype!=np.bool_:raise ValueError('Boolean time/sensor truth mask required')
    return np.stack([mask[a:b].any(0) for a,b in patch_rows(len(mask),patches)],1).astype(np.float32)


def compatible_type_targets(before,after,spec,scale):
    """Admit exact observational aliases at identical injection parameters.

    This does not discover every physically possible mechanism or create real fault
    labels. The full original benchmark remains; ambiguous cases are not dropped.
    """
    result=np.zeros(len(TYPE_LABELS),dtype=bool)
    if spec is None:
        result[0]=True;return result
    for i,kind in enumerate(KINDS,1):
        proposal,_=inject(before,Injection(kind,spec.sensor,spec.start,spec.end,spec.strength,spec.seed),scale)
        result[i]=np.array_equal(proposal,np.asarray(after))
    if not result.any() or not result[TYPE_LABELS.index(spec.kind)]:
        raise ValueError('Injected observation and truth contract disagree')
    return result


def decode_interval(probabilities,length,*,threshold=.5):
    """One displayed contiguous component containing the predicted peak.

    No truth-dependent best-overlap selection. Boundaries are patch-resolution
    native rows, not physical seconds; absent support returns None.
    """
    values=np.asarray(probabilities,float)
    if values.ndim!=1 or not values.size or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError('Finite patch probabilities required')
    if not 0<threshold<1:raise ValueError('Interior decoding threshold required')
    peak=int(values.argmax());active=values>=threshold
    if not active[peak]:return None
    left=right=peak
    while left and active[left-1]:left-=1
    while right+1<len(values) and active[right+1]:right+=1
    rows=patch_rows(length,len(values))
    return [int(rows[left,0]),int(rows[right,1])]


class GroundedOutputHeads(nn.Module):
    def __init__(self,token_width,context_width):
        super().__init__()
        self.condition=nn.Linear(context_width,2*token_width)
        self.temporal=nn.Linear(token_width,1)
        self.kind=nn.Sequential(nn.Linear(context_width,token_width),nn.GELU(),nn.Linear(token_width,len(TYPE_LABELS)))

    def forward(self,tokens,context):
        # tokens B,S,P,D; context B,S,C, already question/numeric-conditioned.
        gamma,beta=self.condition(context).chunk(2,-1)
        conditioned=tokens*(1+gamma.tanh().unsqueeze(-2))+beta.unsqueeze(-2)
        return dict(patch_logits=self.temporal(conditioned).squeeze(-1),type_logits=self.kind(context))


def output_loss(output,patch_targets,type_targets,*,positive_weight=20.):
    """Partial-label type likelihood: aliases need not receive arbitrary unique IDs."""
    patch_logits=output['patch_logits'];type_logits=output['type_logits']
    allowed=type_targets.bool()
    if allowed.shape!=type_logits.shape or not allowed.any(-1).all():raise ValueError('Nonempty admissible type labels required')
    if patch_targets.shape!=patch_logits.shape:raise ValueError('Matching patch labels required')
    temporal=F.binary_cross_entropy_with_logits(patch_logits,patch_targets,
        pos_weight=patch_logits.new_tensor(positive_weight))
    log_prob=F.log_softmax(type_logits,-1)
    per_sensor=-torch.logsumexp(log_prob.masked_fill(~allowed,float('-inf')),-1)
    changed=~allowed[...,0]
    # A single changed channel must not be drowned by the many untouched channels.
    groups=[per_sensor[mask].mean() for mask in (changed,~changed) if mask.any()]
    kind=torch.stack(groups).mean()
    return temporal+kind


class GroundedOutputTrial:
    """Fit output heads on a frozen detector, with independent TRAIN selection.

    Detector scores/thresholds are unchanged. This isolates decoding capacity from
    detection changes before considering alternative representation learning.
    """
    def __init__(self,detector,*,seed=81001):
        self.detector,self.seed=detector,seed

    def fit(self,train,selection,*,epochs=20,batch_size=32,learning_rate=.001):
        if not train or not selection or any(e['split']!='train' for e in train+selection):
            raise ValueError('TRAIN-only head fitting/selection required')
        if {e['run_id'] for e in train}&{e['run_id'] for e in selection}:
            raise ValueError('Head fitting/selection run leakage')
        if epochs<1 or batch_size<1 or learning_rate<=0:raise ValueError('Positive training budget required')
        detector=self.detector;device=detector.device
        def tensors(rows):
            features=np.stack([detector.transform(e['features'],e.get('step')) for e in rows])
            return (torch.tensor((features-detector.mean)/detector.scale,device=device),
                torch.tensor(np.stack([e['patch_targets'] for e in rows]),device=device),
                torch.tensor(np.stack([e['type_targets'] for e in rows]),device=device))
        tx,tp,tt=tensors(train);vx,vp,vt=tensors(selection)
        changed=tp.any(-1).any(-1)
        if not changed.any() or not (~changed).any():raise ValueError('Controls and changes required')
        devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            self.heads=GroundedOutputHeads(detector.width,detector.width*3).to(device)
            optimizer=torch.optim.AdamW(self.heads.parameters(),lr=learning_rate,weight_decay=.01)
            best=float('inf');best_state=None
            detector.model.eval()
            for epoch in range(epochs):
                self.heads.train()
                weights=torch.where(changed,1/changed.sum(),1/(~changed).sum()).float()
                indices=torch.multinomial(weights,len(tx),replacement=True)
                for ids in indices.split(batch_size):
                    with torch.no_grad():tokens,context=detector.model.encode(tx[ids])
                    loss=output_loss(self.heads(tokens,context),tp[ids],tt[ids])
                    optimizer.zero_grad();loss.backward()
                    nn.utils.clip_grad_norm_(self.heads.parameters(),5.);optimizer.step()
                self.heads.eval();losses=[]
                with torch.no_grad():
                    # Equal control/change selection weight, independent of generator prevalence.
                    changed_validation=vp.any(-1).any(-1)
                    for mask in (changed_validation,~changed_validation):
                        ids=torch.where(mask)[0];total=0.
                        if not len(ids):raise ValueError('Selection controls and changes required')
                        for chunk in ids.split(batch_size):
                            tokens,context=detector.model.encode(vx[chunk])
                            total+=float(output_loss(self.heads(tokens,context),vp[chunk],vt[chunk]))*len(chunk)
                        losses.append(total/len(ids))
                value=sum(losses)/len(losses)
                if value<best:
                    best=value;self.best_epoch=epoch+1
                    best_state={k:v.detach().cpu().clone() for k,v in self.heads.state_dict().items()}
            self.heads.load_state_dict(best_state);self.heads.eval()
        return self

    @torch.no_grad()
    def predict(self,features,step=None,query=None):
        detector=self.detector;x=detector.transform(features,step)
        if x.ndim!=3:raise ValueError('One sensor/patch/feature window required')
        tokens,context=detector.model.encode(torch.tensor((x[None]-detector.mean)/detector.scale,device=detector.device),query=query)
        result=self.heads(tokens,context)
        return dict(patch_probabilities=result['patch_logits'][0].sigmoid().cpu().numpy(),
                    type_probabilities=result['type_logits'][0].softmax(-1).cpu().numpy(),
                    sensor_scores=detector.model.head(context).squeeze(-1)[0].cpu().numpy())
