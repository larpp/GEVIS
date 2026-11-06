"""Bounded in-memory preparation and small-head adaptation for GEVIS.

No CSV reads or implicit fitting occur on import. Explicit answer fitting can
save one full source-bound checkpoint. Text and sensor encoders stay frozen;
configured experiments adapt small fusion and answer components. Quality
remains subject to held-out evaluation.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import time
import math

import torch
from torch import nn

from .answer_display import ANSWER_CRITERIA as DEFAULT_ANSWER_CRITERIA


def frozen_text(model, texts, device, *, capacity=4096, chunk_size=32):
    """Reuse text only while the encoder is frozen and in evaluation mode."""
    encoder = model.text
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        model._text_cache = OrderedDict()
        return encoder(texts, device)
    cache = getattr(model, '_text_cache', None)
    if cache is None:
        cache = model._text_cache = OrderedDict()
    unique = list(dict.fromkeys(texts))
    missing = [text for text in unique if text not in cache]
    embeddings = {text: cache[text] for text in unique if text in cache}
    with torch.no_grad():
        for start in range(0, len(missing), chunk_size):
            items = missing[start:start + chunk_size]
            values = encoder(items, device).detach().cpu()
            for text, value in zip(items, values):
                embeddings[text] = value
                cache[text] = value
                while len(cache) > capacity:
                    cache.popitem(last=False)
    return torch.stack([embeddings[text] for text in texts]).to(device)


def configure_fast(model):
    """Freeze pretrained computation; train only criterion/change/temperature."""
    model.requires_grad_(False)
    model.criterion_proj.requires_grad_(True)
    model.change_head.requires_grad_(True)
    model.temperature.requires_grad_(True)
    model.eval()
    model._text_cache = OrderedDict()
    return [p for p in model.parameters() if p.requires_grad]


def head_loss(logits, rewards, *, reward_weight=.1, tie_tolerance=1e-5):
    """Tie-aware supervision supplies gradients even for collapsed answers."""
    rewards = rewards.detach()
    targets = (rewards >= rewards.amax(-1, keepdim=True) - tie_tolerance).float()
    targets = targets / targets.sum(-1, keepdim=True)
    supervision = -(targets * logits.log_softmax(-1)).sum(-1).mean()
    return supervision - reward_weight * (logits.softmax(-1) * rewards).sum(-1).mean()


def cache_size_bytes(cache):
    return sum(t.numel() * t.element_size() for item in cache for t in item['fusion']) + sum(
        item['rewards'].numel() * item['rewards'].element_size() for item in cache
    )


def fit_heads(model, cache, criteria, *, epochs=8, batch_size=64,
              learning_rate=1e-3, patience=2, max_seconds=120., seed=42,
              validation=None, progress=print, require_trained_selection=False):
    """Cooperative time limit; restore the best validation state, including baseline.

    The caller times preparation separately. A running batch/validation cannot
    be preempted, so max_seconds is a budget checked between operations.
    """
    if not cache or epochs < 1 or max_seconds <= 0 or batch_size < 1:
        raise ValueError('Nonempty cache and positive epochs/budget/batch size required')
    params = configure_fast(model)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    baseline = validation(model) if validation else 0.
    def snapshot():
        if require_trained_selection:
            return {k:p.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
        return copy.deepcopy(model.state_dict())
    best_score, best_state = baseline, snapshot()
    if require_trained_selection:
        best_score = -float('inf')  # Never export the random pre-fit baseline.
    updates = 0
    stale, history = 0, []
    for epoch in range(1, epochs + 1):
        if time.perf_counter() - started >= max_seconds:
            break
        model.eval()  # frozen fusion and text stay deterministic
        model.criterion_proj.train(); model.change_head.train()
        losses = []
        for item_index in torch.randperm(len(cache), generator=generator).tolist():
            item = cache[item_index]
            order = torch.randperm(len(item['rewards']), generator=generator)
            for start in range(0, len(order), batch_size):
                if time.perf_counter() - started >= max_seconds:
                    break
                indices = order[start:start + batch_size]
                fusion = tuple(t[indices].to(device) for t in item['fusion'])
                answer_type = item['answer_type']
                logits = torch.stack(model.decode(fusion, [answer_type] * len(indices), [criteria[answer_type]] * len(indices)))
                loss = head_loss(logits, item['rewards'][indices].to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite answer-training loss; no checkpoint exported')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step(); updates += 1; losses.append(float(loss.detach()))
            if time.perf_counter() - started >= max_seconds:
                break
        if not losses:
            break
        model.eval()
        score = validation(model) if validation else -sum(losses) / len(losses)
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation score; no checkpoint exported')
        row = {'epoch': epoch, 'train_loss': sum(losses) / len(losses),
               'validation_score': score, 'elapsed_seconds': time.perf_counter() - started}
        history.append(row); progress(row)
        if score > best_score + 1e-5:
            best_score, best_state, stale = score, snapshot(), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if require_trained_selection and (not history or not updates):
        raise RuntimeError('No completed answer-head updates; no checkpoint exported')
    if require_trained_selection:
        with torch.no_grad():
            for name,parameter in model.named_parameters():
                if name in best_state:parameter.copy_(best_state[name].to(parameter.device))
    else:
        model.load_state_dict(best_state)
    model.eval()
    return history, {'baseline_score': baseline, 'best_score': best_score,
                     'fit_seconds': time.perf_counter() - started,
                     'trainable_parameters': sum(p.numel() for p in params),
                     'cache_bytes': cache_size_bytes(cache), 'optimizer_updates': updates}


@torch.no_grad()
def answer_probabilities(model, numeric_tokens, questions, answer_types, criteria,
                         token_descriptions=None):
    """Process-independent inference on an explicitly adapted token batch.

    An encoder adapter supplies [batch,tokens,features] and optional descriptions.
    Equal feature width alone is NOT a claim of compatible token semantics.
    Returned softmax values are uncalibrated class scores, not fault probabilities.
    """
    if numeric_tokens.ndim != 3 or min(numeric_tokens.shape) < 1:
        raise ValueError('Encoder adapter must supply nonempty [batch,tokens,features]')
    if numeric_tokens.shape[-1] != model.numeric.in_features:
        raise ValueError('Encoder adapter feature width does not match trained GEVIS; no padding/truncation')
    batch=len(numeric_tokens)
    if not (len(questions)==len(answer_types)==len(criteria)==batch):
        raise ValueError('One question, answer type and ordered option list per batch item required')
    if any(not isinstance(q,str) or not q.strip() for q in questions):
        raise ValueError('Nonempty questions required')
    if any(k not in ('BOOLEAN','CHOICE','ORDINAL') for k in answer_types):
        raise ValueError('Unsupported typed answer contract')
    if any(not opts or any(not isinstance(v,str) for v in opts) for opts in criteria):
        raise ValueError('Ordered text answer options required')
    if not torch.isfinite(numeric_tokens).all():
        raise ValueError('Nonfinite adapted input')
    trained_options=getattr(model,'answer_criteria',None)
    if trained_options is not None and any(list(opts)!=list(trained_options[k]) for k,opts in zip(answer_types,criteria)):
        raise ValueError('Inference option text/order must match the trained decoder contract')
    model.eval()
    fusion=model(numeric_tokens,questions,answer_types,criteria,token_descriptions,return_fusion=True)
    logits=model.decode(fusion,answer_types,criteria)
    if len(logits)!=batch:
        raise ValueError('Incomplete answer logits')
    result=[]
    for values,opts in zip(logits,criteria):
        if values.ndim!=1 or len(values)!=len(opts) or not torch.isfinite(values).all():
            raise ValueError('Invalid answer logits')
        result.append(values.softmax(-1))
    return result


def load_answer_checkpoint(model, checkpoint_path, *, backbone_sha256, criteria=None):
    """Strictly restore the SAME GEVIS, with no process-specific fallback."""
    from pathlib import Path
    checkpoint=torch.load(Path(checkpoint_path),map_location='cpu',weights_only=False)
    if checkpoint.get('schema') not in ('gevis.question_model.v15','gevis.typed_answer.v1'):
        raise ValueError('Unsupported trained answer checkpoint schema')
    if not backbone_sha256 or checkpoint.get('backbone_sha256')!=backbone_sha256:
        raise ValueError('Answer checkpoint belongs to a different backbone')
    state=checkpoint.get('model_state')
    if not isinstance(state,dict) or not all(any(k.startswith(prefix) for k in state)
            for prefix in ('change_head.','criterion_proj.')) or 'temperature' not in state:
        raise ValueError('Trained answer decoder tensors are missing; refusing random-head softmax')
    if criteria is not None and checkpoint.get('criteria')!={k:list(v) for k,v in criteria.items()}:
        raise ValueError('Trained answer option ordering does not match the current contract')
    if checkpoint.get('answer_contract') in ('fixed_english_center_whole_proxy_v1','fixed_english_center_whole_proxy_v2'):
        training=checkpoint.get('training',{})
        if training.get('optimizer_updates',0)<1 or not training.get('frozen_backbone_unchanged'):
            raise ValueError('Missing actual answer-training evidence')
    model.load_state_dict(state,strict=True)
    model.requires_grad_(False).eval();model._text_cache=None
    model.answer_criteria=checkpoint.get('criteria')
    return checkpoint


def fit_answer_checkpoint(model, train, validation, criteria, backbone_path, *,
                          train_run_ids, validation_run_ids, epochs=8,
                          batch_size=64, learning_rate=1e-3, max_seconds=120.,
                          patience=2, seed=42, progress=print):
    """Fit existing GEVIS answer heads; save full state after held-out selection.

    Encoder/fusion computation is cached and frozen. Targets are explicit numeric
    comparison proxies, never verified equipment faults. Only TRAIN is optimized;
    disjoint Validation selects a trained snapshot; Test is not accepted here.
    """
    from pathlib import Path
    import hashlib
    path=Path(backbone_path)
    destination=path.parent/'gevis_answer_model.pt'
    if destination.exists():
        raise FileExistsError('Answer checkpoint already exists; load it or choose an explicit new run')
    fitting=set(map(str,train_run_ids));heldout=set(map(str,validation_run_ids))
    if not fitting or not heldout or fitting & heldout:
        raise ValueError('Nonempty disjoint TRAIN/Validation run identities required')
    expected={'BOOLEAN':2,'CHOICE':6,'ORDINAL':5}
    if {k:len(v) for k,v in criteria.items()}!=expected:
        raise ValueError('Expected ordered BOOLEAN/CHOICE/ORDINAL options (2/6/5)')
    for cache in (train,validation):
        if {item['answer_type'] for item in cache}!=set(expected):
            raise ValueError('Both splits require all three answer types')
        for item in cache:
            rewards=item['rewards']
            if rewards.ndim!=2 or not len(rewards) or rewards.shape[1]!=expected[item['answer_type']]:
                raise ValueError('Invalid proxy reward dimensions')
            if not torch.isfinite(rewards).all() or len(item['fusion'])!=3:
                raise ValueError('Invalid cached fusion/rewards')
            if any(len(t)!=len(rewards) or not torch.isfinite(t).all() for t in item['fusion']):
                raise ValueError('Invalid cached fusion batch')
    def validation_score(candidate):
        device=next(candidate.parameters()).device
        losses=[]
        with torch.no_grad():
            candidate.eval()
            for item in validation:
                fusion=tuple(t.to(device) for t in item['fusion']);kind=item['answer_type']
                logits=torch.stack(candidate.decode(fusion,[kind]*len(item['rewards']),
                                                     [criteria[kind]]*len(item['rewards'])))
                losses.append(float(head_loss(logits,item['rewards'].to(device),reward_weight=0.)))
        return -sum(losses)/len(losses)
    def tensor_digest(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
    frozen={k:tensor_digest(v) for k,v in model.state_dict().items()
            if not k.startswith(('change_head.','criterion_proj.')) and k!='temperature'}
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()
             if k.startswith(('change_head.','criterion_proj.')) or k=='temperature'}
    history,report=fit_heads(model,train,criteria,epochs=epochs,batch_size=batch_size,
        learning_rate=learning_rate,max_seconds=max_seconds,patience=patience,
        seed=seed,progress=progress,validation=validation_score,require_trained_selection=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if any(tensor_digest(state[k])!=v for k,v in frozen.items()):
        raise RuntimeError('Frozen GEVIS backbone changed; refusing export')
    if not any(not torch.equal(state[k],v) for k,v in initial.items()):
        raise RuntimeError('Answer parameters did not change; refusing random-head export')
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite trained state; refusing export')
    report.update(selection='disjoint Validation loss among trained snapshots',
                  target='numeric comparison proxy; not verified fault labels',
                  train_runs=len(fitting),validation_runs=len(heldout),
                  frozen_backbone_unchanged=True)
    report['class_support']={}
    for split,cache in (('train',train),('validation',validation)):
        report['class_support'][split]={kind:torch.bincount(torch.cat([
            item['rewards'].argmax(-1) for item in cache if item['answer_type']==kind]),
            minlength=size).tolist() for kind,size in expected.items()}
    source_hash=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):source_hash.update(block)
    checkpoint=dict(schema='gevis.typed_answer.v1',model_state=state,
        backbone_sha256=source_hash.hexdigest(),
        numeric_dim=int(model.numeric.in_features),criteria={k:list(v) for k,v in criteria.items()},
        answer_contract='fixed_english_center_whole_proxy_v2',training=report,
        training_config=dict(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,
                             max_seconds=max_seconds,patience=patience,seed=seed))
    with destination.open('xb') as output:
        torch.save(checkpoint,output)
    model.requires_grad_(False).eval()
    model.answer_criteria=checkpoint['criteria']
    return destination,history,report


def pack_boolean_records(records, *, max_bytes=256*1024**2):
    """Pack sensor-selected patches, masking unequal phase lengths explicitly."""
    if not records: raise ValueError('No diagnostic records')
    width = max(item['states'].shape[1] for item in records)
    features = records[0]['states'].shape[2]
    rows = sum(len(item['labels']) for item in records)
    if rows*width*features*4 > max_bytes: raise MemoryError('Diagnostic cache exceeds memory budget; reduce pair limit')
    states, labels, lengths = [], [], []
    for item in records:
        x, y = item['states'].detach().cpu().float(), item['labels'].detach().cpu().long()
        if x.ndim!=3 or x.shape[2]!=features or len(x)!=len(y): raise ValueError('Diagnostic state/label shape mismatch')
        states.append(nn.functional.pad(x,(0,0,0,width-x.shape[1])))
        labels.append(y); lengths.append(torch.full((len(y),),x.shape[1],dtype=torch.long))
    return {'states':torch.cat(states),'labels':torch.cat(labels),'lengths':torch.cat(lengths)}


def configure_boolean_fusion(model):
    """Unfreeze the small numeric fusion; keep language and semantic classifiers fixed."""
    model.requires_grad_(False)
    for name in ('numeric','cross','answer_norm','change_head','criterion_proj'):
        getattr(model,name).requires_grad_(True)
    model.temperature.requires_grad_(True)
    model.eval(); model._text_cache=OrderedDict()
    return [p for p in model.parameters() if p.requires_grad]


def boolean_fusion_logits(model, states, lengths):
    """Fixed operation: selected sensor, whole window, absolute mean difference >1.

    Sensor selection happens before fusion. No free-language parser is involved.
    The threshold defines proxy labels, not equipment faults.
    """
    device=states.device
    query=model._text(['Has the absolute mean level difference exceeded one normalized unit over the full interval?'],device).expand(len(states),-1)
    query=query+model.type_emb(torch.zeros(len(states),device=device,dtype=torch.long))
    keys=model.numeric(states)
    mask=torch.arange(states.shape[1],device=device)[None]>=lengths[:,None]
    answer,attention=model.cross(query[:,None],keys,keys,key_padding_mask=mask,need_weights=True)
    answer=model.answer_norm(answer[:,0]+query)
    return torch.stack(model.decode((answer,query,attention[:,0]),['BOOLEAN']*len(states),[['No','Yes']]*len(states)))


class BooleanNumericProbe(nn.Module):
    """Small learned numeric baseline, not a label-computing oracle."""
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(10,16),nn.GELU(),nn.Linear(16,2))
    def forward(self,states,lengths):
        keep=(torch.arange(states.shape[1],device=states.device)[None]<lengths[:,None])[...,None]
        values=(states[...,-8:]*keep).sum(1)/lengths[:,None]
        delta=values[:,0]-values[:,1]
        features=torch.cat([values,delta[:,None],delta.abs()[:,None]],-1)
        return self.net(features)


@torch.no_grad()
def evaluate_boolean(model, cache, logits_fn, *, batch_size=128):
    model.eval(); device=next(model.parameters()).device; predictions=[]; loss_sum=0.
    for start in range(0,len(cache['labels']),batch_size):
        x=cache['states'][start:start+batch_size].to(device); lengths=cache['lengths'][start:start+batch_size].to(device)
        y=cache['labels'][start:start+batch_size].to(device); logits=logits_fn(model,x,lengths)
        loss_sum+=float(nn.functional.cross_entropy(logits,y,reduction='sum'))
        predictions.append(logits.argmax(-1).cpu())
    pred=torch.cat(predictions); y=cache['labels']; n=len(y)
    tp=int(((y==1)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    fp=int(((y==0)&(pred==1)).sum()); tn=int(((y==0)&(pred==0)).sum())
    rate=lambda a,b: a/b if b else float('nan')
    recall,specificity=rate(tp,tp+fn),rate(tn,tn+fp)
    return {'pairs':n,'accuracy':float((pred==y).float().mean()),
            'balanced_accuracy':(recall+specificity)/2,
            'change_cases':tp+fn,'no_change_cases':tn+fp,'true_positive':tp,'false_negative':fn,
            'false_positive':fp,'true_negative':tn,'change_recall':recall,
            'false_alarm_rate':rate(fp,fp+tn),'precision':rate(tp,tp+fp),
            'validation_loss':loss_sum/n},pred


def fit_boolean(model, train, validation, logits_fn, *, epochs=12, batch_size=128,
                learning_rate=1e-3, max_seconds=120., patience=3, seed=42, progress=print):
    """Balanced TRAIN sampling only; natural validation selects the snapshot."""
    y=train['labels']; positive=(y==1).nonzero().flatten(); negative=(y==0).nonzero().flatten()
    if not len(positive) or not len(negative):
        raise ValueError('Diagnostic TRAIN needs both classes; no synthetic positives will be invented')
    if batch_size<2 or epochs<1 or max_seconds<=0: raise ValueError('Invalid diagnostic fitting budget')
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=learning_rate,weight_decay=1e-4)
    device=next(model.parameters()).device; rng=torch.Generator().manual_seed(seed); started=time.perf_counter()
    baseline,_=evaluate_boolean(model,validation,logits_fn)
    has_both=baseline['change_cases']>0 and baseline['no_change_cases']>0
    score=lambda metrics: metrics['balanced_accuracy'] if has_both else -metrics['validation_loss']
    best_score=score(baseline); best_loss=baseline['validation_loss']; best_state=copy.deepcopy(model.state_dict()); stale=0; history=[]
    for epoch in range(1,epochs+1):
        if time.perf_counter()-started>=max_seconds: break
        model.eval()  # deterministic frozen text; fusion dropout need not add noise here
        losses=[]
        for _ in range(max(1,(len(y)+batch_size-1)//batch_size)):
            if time.perf_counter()-started>=max_seconds: break
            pos=positive[torch.randint(len(positive),(batch_size//2,),generator=rng)]
            neg=negative[torch.randint(len(negative),(batch_size-batch_size//2,),generator=rng)]
            indices=torch.cat([pos,neg]); indices=indices[torch.randperm(len(indices),generator=rng)]
            logits=logits_fn(model,train['states'][indices].to(device),train['lengths'][indices].to(device))
            loss=nn.functional.cross_entropy(logits,y[indices].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params,1.); optimizer.step()
            losses.append(float(loss.detach()))
        if not losses: break
        metrics,_=evaluate_boolean(model,validation,logits_fn); current=score(metrics)
        row={'epoch':epoch,'train_loss':sum(losses)/len(losses),**metrics,'elapsed_seconds':time.perf_counter()-started}
        history.append(row); progress(row)
        improved=current>best_score+1e-5 or (abs(current-best_score)<=1e-5 and metrics['validation_loss']<best_loss-1e-5)
        if improved:
            best_score=current; best_loss=metrics['validation_loss']; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if stale>=patience: break
    model.load_state_dict(best_state); model.eval()
    return history,{'train_positive':len(positive),'train_negative':len(negative),
                    'validation_has_both_classes':has_both,'selection_metric':'balanced_accuracy' if has_both else 'validation_loss',
                    'trainable_parameters':sum(p.numel() for p in params),'fit_seconds':time.perf_counter()-started,
                    'LJR_weight':0.,'baseline_validation':baseline,'best_score':best_score}


TYPED_SIZES = {'BOOLEAN':2,'CHOICE':6,'ORDINAL':5}
TYPED_CRITERIA = {'BOOLEAN':['No','Yes'],
    'CHOICE':['No clear change','Mean level','Variability','Trend slope','Local change','Reconstruction error'],
    'ORDINAL':['No change','Small','Moderate','Large','Very large']}


def typed_targets(features, statistic, direction, *, thresholds=(1.,1.,1.,1.,1.), ordinal_edges=(1.,2.,3.,4.), false_alarm_cost=1., miss_cost=1.):
    """Explicit numeric proxy contract; no verified-fault claim or Gaussian assumption."""
    threshold=torch.as_tensor(thresholds,device=features.device,dtype=features.dtype)
    if features.shape[-1]!=5 or len(threshold)!=5 or (threshold<=0).any(): raise ValueError('Five positive operation thresholds required')
    edges=torch.as_tensor(ordinal_edges,device=features.device,dtype=features.dtype)
    if len(edges)!=4 or (edges<=0).any() or not (edges[1:]>edges[:-1]).all(): raise ValueError('Four increasing ordinal boundaries required')
    if false_alarm_cost<0 or miss_cost<0: raise ValueError('Reward costs must be nonnegative')
    scores=features/threshold
    values=scores.gather(1,statistic[:,None]).squeeze(1)
    directed=torch.where(direction==0,values,torch.where(direction==1,-values,values.abs()))
    strength=scores.abs(); largest,feature=strength.max(-1)
    labels={'BOOLEAN':(directed>1.).long(),'CHOICE':torch.where(largest>1.,feature+1,torch.zeros_like(feature)),
            'ORDINAL':torch.bucketize(largest,edges,right=False)}
    rewards={}
    for kind,label in labels.items():
        choices=torch.arange(TYPED_SIZES[kind],device=features.device)[None]
        if kind=='ORDINAL': reward=1.-2*(choices-label[:,None]).abs()/4.
        else: reward=torch.where(choices==label[:,None],1.,-1.)
        if kind=='BOOLEAN':
            reward=torch.where((label[:,None]==0)&(choices==1),-false_alarm_cost,reward)
            reward=torch.where((label[:,None]==1)&(choices==0),-miss_cost,reward)
        rewards[kind]=reward.float()
    return labels,rewards,scores


def structured_answer_probabilities(features, statistic, direction, *, thresholds=(1.,1.,1.,1.,1.), temperature=.75):
    """Execute declared numeric operations and return soft answer scores.

    These are deterministic comparison scores, not learned fault probabilities.
    ``features`` contains level, variability, slope, local difference and
    reconstruction-residual operations for the sensor/scope named by a question.
    """
    if temperature <= 0:
        raise ValueError('Positive answer temperature required')
    threshold=torch.as_tensor(thresholds,device=features.device,dtype=features.dtype)
    if features.ndim!=2 or features.shape[1]!=5 or threshold.shape!=(5,) or (threshold<=0).any():
        raise ValueError('Expected rows of five operations and five positive thresholds')
    if statistic.shape!=direction.shape or len(statistic)!=len(features):
        raise ValueError('One statistic and direction are required per feature row')
    scores=features/threshold
    requested=scores.gather(1,statistic[:,None]).squeeze(1)
    directed=torch.where(direction==0,requested,torch.where(direction==1,-requested,requested.abs()))
    boolean=torch.stack([1.-directed,directed-1.],-1).div(temperature).softmax(-1)
    strength=scores.abs(); largest=strength.max(-1).values
    # Keep soft-answer argmax boundaries identical to ``typed_targets``:
    # no-change versus a named feature crosses at 1.0, not 0.5.
    choice=torch.cat([(1.-largest)[:,None],strength-1.],-1).div(temperature).softmax(-1)
    # Midpoints between these centers are exactly the declared ordinal edges
    # [1, 2, 3, 4]. At an exact edge argmax selects the lower class, matching
    # torch.bucketize(..., right=False) in ``typed_targets``.
    centers=torch.arange(5,device=features.device,dtype=features.dtype)+.5
    ordinal=(-(largest[:,None]-centers[None]).abs()/temperature).softmax(-1)
    return {'BOOLEAN':boolean,'CHOICE':choice,'ORDINAL':ordinal},scores


def pack_typed_records(records, *, max_bytes=256*1024**2):
    """One shared token bank and row indices: never duplicate it for question types."""
    width=max(record['states'].shape[1] for record in records)
    total=sum(len(record['states']) for record in records)
    dim=records[0]['states'].shape[-1]
    if total*width*dim*4>max_bytes: raise MemoryError('Integrated cache budget exceeded; reduce pair_limit')
    states=[]; lengths=[]; parts={name:[] for name in ('bank_index','features','statistic','direction','scope','step_id')}
    labels={kind:[] for kind in TYPED_SIZES}; rewards={kind:[] for kind in TYPED_SIZES}; offset=0
    for record in records:
        x=record['states'].detach().cpu().float()
        states.append(nn.functional.pad(x,(0,0,0,width-x.shape[1])))
        lengths.append(torch.full((len(x),),x.shape[1],dtype=torch.long))
        for name in parts:
            value=record[name].detach().cpu()
            parts[name].append(value+offset if name=='bank_index' else value)
        for kind in labels:
            labels[kind].append(record['labels'][kind].cpu()); rewards[kind].append(record['rewards'][kind].cpu())
        offset+=len(x)
    result={name:torch.cat(values) for name,values in parts.items()}
    result.update(states=torch.cat(states),lengths=torch.cat(lengths),labels={k:torch.cat(v) for k,v in labels.items()},rewards={k:torch.cat(v) for k,v in rewards.items()})
    return result


def typed_batch(cache, indices, device):
    bank=cache['bank_index'][indices]
    result={name:cache[name][indices].to(device) for name in ('features','statistic','direction','scope')}
    result.update(states=cache['states'][bank].to(device),lengths=cache['lengths'][bank].to(device))
    return result


class TypedFusionScorer(nn.Module):
    """Trainable small numeric fusion; declared slots separate target binding from NLP."""
    def __init__(self, base):
        super().__init__(); self.base=base
        configure_boolean_fusion(base)
        dim=base.numeric.out_features
        self.operation=nn.Embedding(45,dim)
        self.summary=nn.Sequential(nn.Linear(5,dim),nn.GELU(),nn.LayerNorm(dim))
        # Newly attached layers default to CPU even when the pretrained base is
        # already on CUDA. Place the complete wrapper before fitting/inference.
        self.to(next(base.parameters()).device)
    def forward(self,batch,kind):
        base=self.base; x=batch['states']; device=x.device
        operation=batch['statistic']*9+batch['direction']*3+batch['scope']
        scope_names=('full','early','late'); stat_names=('mean level','variability','trend slope','local change','reconstruction error'); directions=('increase','decrease','change')
        questions=[]
        for st,dr,sc in zip(batch['statistic'].tolist(),batch['direction'].tolist(),batch['scope'].tolist()):
            questions.append(f'Over the {scope_names[sc]} interval, did {stat_names[st]} {directions[dr]} relative to the reference?' if kind=='BOOLEAN' else f'Over the {scope_names[sc]} interval, '+('what kind of change is present?' if kind=='CHOICE' else 'what is the degree of change?'))
        query=base._text(questions,device)+base.type_emb(torch.full((len(x),),list(TYPED_SIZES).index(kind),device=device))+self.operation(operation)
        keys=base.numeric(x); positions=torch.arange(x.shape[1],device=device)[None]
        ratio=(positions+.5)/batch['lengths'][:,None]
        mask=positions>=batch['lengths'][:,None]
        scope_mask=torch.where(batch['scope'][:,None]==1,ratio>=1/3,torch.where(batch['scope'][:,None]==2,ratio<2/3,torch.zeros_like(mask)))
        # Very short windows may have no token in a third; retain one valid token.
        combined=mask|scope_mask; empty=combined.all(1)
        if empty.any(): combined[empty,0]=False
        answer,attention=base.cross(query[:,None],keys,keys,key_padding_mask=combined,need_weights=True)
        answer=base.answer_norm(answer[:,0]+query+self.summary(batch['features']))
        return torch.stack(base.decode((answer,query,attention[:,0]),[kind]*len(x),[TYPED_CRITERIA[kind]]*len(x)))


class TypedNumericProbe(nn.Module):
    def __init__(self):
        super().__init__(); self.shared=nn.Sequential(nn.Linear(16,32),nn.GELU())
        self.heads=nn.ModuleDict({kind:nn.Linear(32,size) for kind,size in TYPED_SIZES.items()})
    def forward(self,batch,kind):
        context=torch.cat([batch['features'],nn.functional.one_hot(batch['statistic'],5),nn.functional.one_hot(batch['direction'],3),nn.functional.one_hot(batch['scope'],3)],-1).float()
        return self.heads[kind](self.shared(context))


def expected_calibration_error(probabilities, labels, *, bins=10):
    """Top-label ECE for held-out answer probabilities."""
    if probabilities.ndim != 2 or len(probabilities) != len(labels) or bins < 1:
        raise ValueError('Probability matrix, matching labels, and positive bins required')
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0., 1., bins + 1, device=probabilities.device)
    error = probabilities.new_tensor(0.)
    for index in range(bins):
        selected = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1])
        if selected.any():
            error += selected.float().mean() * (confidence[selected].mean() - correct[selected].mean()).abs()
    return float(error)


def typed_metrics(probabilities, labels, rewards, kind, *, threshold=.5):
    pred=(probabilities[:,1]>=threshold).long() if kind=='BOOLEAN' else probabilities.argmax(-1)
    n=len(labels); classes=TYPED_SIZES[kind]; present=labels.unique()
    recalls=[float((pred[labels==cls]==cls).float().mean()) for cls in present]
    f1=[]
    for cls in range(classes):
        tp=int(((labels==cls)&(pred==cls)).sum()); fp=int(((labels!=cls)&(pred==cls)).sum()); fn=int(((labels==cls)&(pred!=cls)).sum())
        f1.append(2*tp/max(1,2*tp+fp+fn))
    chosen=rewards.gather(1,pred[:,None]).squeeze(1)
    row={'pairs':n,'accuracy':float((pred==labels).float().mean()),'balanced_accuracy':sum(recalls)/len(recalls),
         'macro_f1':sum(f1)/classes,'classes_present':len(present),'classes_expected':classes,
         'prediction_largest_share':float(torch.bincount(pred,minlength=classes).max()/n),
         'target_largest_share':float(torch.bincount(labels,minlength=classes).max()/n),
         'mean_selected_reward':float(chosen.mean()),'regret':float((rewards.max(1).values-chosen).mean())}
    true_probability=probabilities.gather(1,labels[:,None]).squeeze(1)
    confidence=probabilities.max(-1).values
    row.update(nll=float(-true_probability.clamp_min(1e-8).log().mean()),
               brier=float((probabilities-nn.functional.one_hot(labels,classes)).square().sum(-1).mean()),
               expected_calibration_error=expected_calibration_error(probabilities,labels),
               mean_confidence=float(confidence.mean()),
               mean_true_answer_probability=float(true_probability.mean()),
               confidence_accuracy_gap=float(confidence.mean()-(pred==labels).float().mean()))
    if kind=='ORDINAL': row.update(ordinal_mae=float((pred-labels).abs().float().mean()))
    if kind=='BOOLEAN':
        tp=int(((labels==1)&(pred==1)).sum()); fn=int(((labels==1)&(pred==0)).sum()); fp=int(((labels==0)&(pred==1)).sum()); tn=int(((labels==0)&(pred==0)).sum())
        rate=lambda a,b:a/b if b else float('nan')
        row.update(change_cases=tp+fn,no_change_cases=fp+tn,true_positive=tp,false_negative=fn,false_positive=fp,true_negative=tn,
                   change_recall=rate(tp,tp+fn),false_alarm_rate=rate(fp,fp+tn),precision=rate(tp,tp+fp))
        if not tp+fn or not tn+fp: row['balanced_accuracy']=float('nan')
    return row


@torch.no_grad()
def predict_typed(model,cache,*,batch_size=128):
    model.eval(); device=next(model.parameters()).device; output={kind:[] for kind in TYPED_SIZES}
    started=time.perf_counter()
    for start in range(0,len(cache['bank_index']),batch_size):
        indices=torch.arange(start,min(start+batch_size,len(cache['bank_index'])))
        batch=typed_batch(cache,indices,device)
        for kind in output: output[kind].append(model(batch,kind).softmax(-1).cpu())
    if device.type=='cuda': torch.cuda.synchronize(device)
    return {kind:torch.cat(parts) for kind,parts in output.items()},(time.perf_counter()-started)*1000/len(cache['bank_index'])


def typed_validation_score(predictions,cache):
    values=[]
    for kind,prob in predictions.items():
        metrics=typed_metrics(prob,cache['labels'][kind],cache['rewards'][kind],kind)
        if math.isfinite(metrics['balanced_accuracy']): values.append(metrics['balanced_accuracy'])
    return sum(values)/len(values) if values else 0.


def typed_loss(logits, labels, rewards, *, kind, class_priors, reward_weight=0., ordinal_weight=.2):
    loss=nn.functional.cross_entropy(logits,labels)
    prob=logits.softmax(-1)
    if kind=='ORDINAL':
        target=nn.functional.one_hot(labels,5).float()
        loss=loss+ordinal_weight*(prob.cumsum(-1)-target.cumsum(-1)).square().mean()
    if reward_weight:
        # Undo balanced-sampler priors for LJR's natural TRAIN expected utility.
        importance=class_priors[labels]*int((class_priors>0).sum())
        loss=loss-reward_weight*(importance*(prob*rewards).sum(-1)).mean()
    return loss


def fit_typed_comparison(base,train,validation,*,epochs=12,batch_size=64,steps_per_epoch=20,
                         learning_rate=1e-3,reward_weight=.2,max_seconds=240.,seed=42,progress=print):
    """Paired identical batches/updates for CE vs CE+LJR; TEST never enters this call."""
    if epochs<1 or batch_size<1 or steps_per_epoch<1 or max_seconds<=0: raise ValueError('Positive integrated training budgets required')
    torch.manual_seed(seed)
    initial=TypedFusionScorer(copy.deepcopy(base))
    models={'GEVIS_CE':copy.deepcopy(initial),'GEVIS_CE_LJR':copy.deepcopy(initial),'numeric_probe':TypedNumericProbe().to(next(base.parameters()).device)}
    device=next(base.parameters()).device
    optimizers={name:torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=1e-4) for name,model in models.items()}
    groups={kind:[(train['labels'][kind]==cls).nonzero().flatten() for cls in range(size)] for kind,size in TYPED_SIZES.items()}
    groups={kind:[g for g in values if len(g)] for kind,values in groups.items()}
    if any(not groups[kind] for kind in groups): raise ValueError('Empty typed TRAIN labels')
    priors={kind:torch.bincount(train['labels'][kind],minlength=size).float().to(device)/len(train['bank_index']) for kind,size in TYPED_SIZES.items()}
    initial_predictions={name:predict_typed(model,validation)[0] for name,model in models.items()}
    best={name:typed_validation_score(pred,validation) for name,pred in initial_predictions.items()}
    best_loss={name:sum(float(-pred[kind].gather(1,validation['labels'][kind][:,None]).clamp_min(1e-8).log().mean()) for kind in TYPED_SIZES)/3 for name,pred in initial_predictions.items()}
    snapshots={name:copy.deepcopy(model.state_dict()) for name,model in models.items()}
    history=[]; updates=0; rng=torch.Generator().manual_seed(seed); started=time.perf_counter()
    for epoch in range(1,epochs+1):
        if time.perf_counter()-started>=max_seconds: break
        losses={name:[] for name in models}
        for _ in range(steps_per_epoch):
            if time.perf_counter()-started>=max_seconds: break
            selections={}
            for kind,available in groups.items():
                selections[kind]=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(available)),),generator=rng)] for g in available])
            # A deadline is checked only outside this pair: compared branches get equal updates.
            for name,model in models.items():
                model.eval(); parts=[]
                for kind,indices in selections.items():
                    batch=typed_batch(train,indices,device)
                    logits=model(batch,kind)
                    parts.append(typed_loss(logits,train['labels'][kind][indices].to(device),train['rewards'][kind][indices].to(device),kind=kind,class_priors=priors[kind],reward_weight=reward_weight if name=='GEVIS_CE_LJR' else 0.))
                loss=torch.stack(parts).mean(); optimizers[name].zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.); optimizers[name].step(); losses[name].append(float(loss.detach()))
            updates+=1
        if not losses['GEVIS_CE']: break
        for name,model in models.items():
            pred,latency=predict_typed(model,validation); score=typed_validation_score(pred,validation)
            nll=sum(float(-pred[kind].gather(1,validation['labels'][kind][:,None]).clamp_min(1e-8).log().mean()) for kind in TYPED_SIZES)/3
            if score>best[name]+1e-5 or (abs(score-best[name])<=1e-5 and nll<best_loss[name]):
                best[name]=score; best_loss[name]=nll; snapshots[name]=copy.deepcopy(model.state_dict())
            row={'epoch':epoch,'method':name,'updates':updates,'train_loss':sum(losses[name])/len(losses[name]),'validation_score':score,'validation_nll':nll,'ms_per_question_group':latency,'elapsed_seconds':time.perf_counter()-started}
            history.append(row); progress(row)
    for name,model in models.items(): model.load_state_dict(snapshots[name]); model.eval()
    report={'paired_updates':updates,'fit_seconds':time.perf_counter()-started,'reward_weight':reward_weight,
            'trainable_parameters':{name:sum(p.numel() for p in model.parameters() if p.requires_grad) for name,model in models.items()},
            'train_class_counts':{kind:torch.bincount(train['labels'][kind],minlength=size).tolist() for kind,size in TYPED_SIZES.items()},
            'selection':'validation score/NLL only','initial_weights_equal':True,'sampler':'TRAIN present-class balanced; identical batches across methods'}
    return models,history,report


def choose_boolean_threshold(probabilities,labels,*,minimum_recall=.8):
    """Validation-only lower-FPR operating point with an explicit recall constraint."""
    if not (labels==1).any() or not (labels==0).any(): return .5,{'status':'INSUFFICIENT_VALIDATION_CLASSES'}
    dummy=torch.zeros(len(labels),2); choices=[]
    for threshold in torch.linspace(.05,.95,91).tolist():
        metrics=typed_metrics(probabilities,labels,dummy,'BOOLEAN',threshold=threshold)
        if metrics['change_recall']>=minimum_recall: choices.append((metrics['false_alarm_rate'],-metrics['change_recall'],abs(threshold-.5),threshold))
    if not choices: return .5,{'status':'NO_FEASIBLE_VALIDATION_THRESHOLD'}
    threshold=min(choices)[-1]
    return threshold,{'status':'VALIDATION_ONLY','minimum_recall':minimum_recall}


def choose_temperature(probabilities,labels):
    """Validation-only probability calibration for any type; class argmax stays fixed."""
    logs=probabilities.clamp_min(1e-8).log(); values=[]
    for temperature in (.5,.75,1.,1.5,2.,3.):
        adjusted=(logs/temperature).softmax(-1)
        nll=float(-adjusted.gather(1,labels[:,None]).clamp_min(1e-8).log().mean())
        values.append((nll,abs(temperature-1),temperature))
    return min(values)[-1]
