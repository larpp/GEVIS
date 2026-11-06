"""Optional generic localization curriculum; not change/no-change or cause labels."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

VERSION='gevis.sensor_discrimination.v12'

def center_relation_pairs(current, reference, patch_length, *, seed):
    """Synthetic paired relationship disruption, NOT real causal labels.

    A shared anchor and two candidate partners preserve identical reference
    context across each pair; only the perturbed partner moves. Full-window
    reversal preserves each partner's marginal value distribution while changing
    its temporal relationship. Flat anchors use a declared synthetic waveform.
    """
    from pecvd.gevis_backbone import reference_evidence_pairs
    if current.ndim!=3 or current.shape!=reference.shape:
        raise ValueError('Aligned current/reference arrays required')
    batch,length,sensors=current.shape
    if (patch_length<1 or length<2*patch_length or length%patch_length
            or not torch.isfinite(current).all() or not torch.isfinite(reference).all()):
        raise ValueError('Finite arrays with at least two complete patches required')
    if sensors<3:
        return reference_evidence_pairs(current,reference,patch_length,seed=seed,temporal=True)
    if length%patch_length:raise ValueError('Complete patch grid required')
    generator=torch.Generator(device=current.device).manual_seed(seed)
    centers=torch.randint(sensors,(batch,),generator=generator,device=current.device)
    ids=torch.stack(((centers+1)%sensors,(centers+2)%sensors),1).flatten()
    source=current.detach().repeat_interleave(2,0).clone()
    changed=source.clone();mask=torch.zeros(2*batch,length//patch_length,dtype=torch.bool,device=current.device)
    for pair in range(batch):
        center=int(centers[pair]);wave=current[pair,:,center].detach().clone()
        if not bool((wave-wave.flip(0)).abs().max()>1e-5):
            wave=torch.linspace(-1.,1.,length,device=current.device,dtype=current.dtype)
        for row in (2*pair,2*pair+1):
            for sensor in (center,int(ids[2*pair]),int(ids[2*pair+1])):
                source[row,:,sensor]=wave
            changed[row]=source[row]
            changed[row,:,ids[row]]=wave.flip(0)
            mask[row]=(wave.flip(0)-wave).reshape(-1,patch_length).ne(0).any(-1)
    return changed,source,{'sensor':ids,'patch':mask.long().argmax(-1),
        'patch_count':length//patch_length,'patch_mask':mask,
        'cross_run':torch.zeros(2*batch,dtype=torch.bool,device=current.device),
        'center_sensor':centers.repeat_interleave(2)}

def source_identity():
    from . import questions
    return hashlib.sha256(Path(__file__).read_bytes()+b'\nQUESTION_CONTRACT\n'+
                          Path(questions.__file__).read_bytes()).hexdigest()

def discrimination_loss(attention,targets,*,margin=.5,pair_weight=.25,hard_weight=.25):
    from pecvd.gevis_backbone import numeric_evidence_loss
    if min(margin,pair_weight,hard_weight)<0:raise ValueError('Nonnegative loss parameters required')
    loss,metrics=numeric_evidence_loss(attention,targets)
    grid=attention.reshape(len(attention),-1,int(targets['patch_count'])).clamp_min(1e-8)
    mass=grid.sum(-1);mass=mass/mass.sum(-1,keepdim=True);logp=mass.log()
    ids=targets['sensor'];rows=torch.arange(len(ids),device=ids.device)
    if mass.shape[1]<2 or len(ids)%2:raise ValueError('Paired examples with two or more sensors required')
    positive=logp[rows,ids]
    negatives=logp.scatter(1,ids[:,None],-torch.inf)
    hard=F.relu(margin-positive+negatives.max(-1).values).mean()
    first,second=ids[::2],ids[1::2]
    if torch.any(first==second):raise ValueError('Counterfactual pair must move to another sensor')
    # Same source, interval and perturbation; attention must follow the moved signal.
    pair_rows=torch.arange(len(first),device=ids.device)
    a,b=logp[::2],logp[1::2]
    movement=(a[pair_rows,first]-a[pair_rows,second])+(b[pair_rows,second]-b[pair_rows,first])
    paired=F.relu(2*margin-movement).mean()
    correct=mass.argmax(-1).eq(ids)
    metrics.update(hard_sensor_margin_loss=float(hard.detach()),moved_sensor_loss=float(paired.detach()),
        moved_sensor_pair_accuracy=float((correct[::2]&correct[1::2]).float().mean()),
        evidence_sensor_top3=float(mass.topk(min(3,mass.shape[1]),dim=-1).indices.eq(ids[:,None]).any(-1).float().mean()))
    return loss+hard_weight*hard+pair_weight*paired,metrics

class TrainReferencePairs(torch.utils.data.Dataset):
    """Validation current windows use TRAIN references, like deployed discovery.

    One reference per other TRAIN process; same cohort and position bucket.
    No validation/Test waveform is admitted as a reference.
    """
    def __init__(self,current,meta,train,train_meta,seed,limit,*,split):
        if split not in ('train','validation'):raise ValueError('Training or selection partition only')
        self.current=current;self.train=train;self.seed=seed;self.members={}
        ids=np.asarray(meta['process_id']).astype(str);train_ids=np.asarray(train_meta['process_id']).astype(str)
        if split=='validation' and set(ids)&set(train_ids):raise ValueError('TRAIN/Validation process overlap')
        for i,(run,cohort,position) in enumerate(zip(ids,meta['cohort_key'],meta['position_bin'])):
            found={}
            for j in np.flatnonzero((np.asarray(train_meta['cohort_key'])==cohort)&(np.asarray(train_meta['position_bin'])==position)&(train_ids!=run)):
                found.setdefault(train_ids[j],int(j))
            if found:self.members[i]=list(found.values())
        indices=list(self.members)
        if limit and len(indices)>limit:indices=np.random.default_rng(seed).choice(indices,limit,replace=False).tolist()
        if not indices:raise ValueError('No matching TRAIN references')
        self.indices=indices

    def __len__(self):return len(self.indices)

    def __getitem__(self,item):
        index=self.indices[item];pool=self.members[index]
        chosen=np.random.default_rng(self.seed+item).choice(pool,min(16,len(pool)),replace=False)
        return torch.from_numpy(self.current[index]),torch.from_numpy(np.median(self.train[chosen],axis=0).astype('float32'))
