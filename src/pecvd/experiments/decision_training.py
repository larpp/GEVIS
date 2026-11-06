"""Detailed STEP3: frozen Fusion and an MLP over independent TRAIN MIN/MAX.

Only current measurements and known native context enter prediction. The normal
reference is fitted on independent TRAIN source lineages, never the exact clean
pre-intervention waveform. No Test fitting, threshold tuning or numeric answers.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import ast, json, math, time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import current_input as current, fusion_answers as fa, peald_training as p
from .reporting import ExperimentReport
from .feature_store import FeatureCache, FeatureRows

SCHEMA = 'gevis.detailed_normal_envelope_mlp.v3'
RECIPE = 'known_boundary_excess_all_splits_v6'
ORDINAL_POLICY = 'measured_boundary_excess_half_one_two_four_train_normalized_width'
SELECTION_POLICY = 'full_population_end_to_end_v5'
INPUT_MODE = 'current_with_packaged_TRAIN_minmax_envelope'
TARGET_POLICY = 'inclusive_native_TRAIN_minmax; outside_any_sample; not equipment fault truth'
_REVIEWED_V6_CODE='157dbf91b82f696adcb060a72658d95908301352780e928d31ce4de8f6f5cb56'
_REVIEWED_V6_AST=('8ddf600f9120b35092a67e00d22822289fe47a172ffdc8411b2e80146625222c',
    'a0550c352f9ff7568b5093cdf40bb9a5c6634cf9203c08c3e492bf2d135c283b')


def _reviewed_v6_inference_matches():
    names={'Config','_hash','_domain','pointer','_key','ReferenceBank','_bank','case_current','_features',
        '_window_features','RowReceiver','Session','load'}
    constants={'SCHEMA','RECIPE','ORDINAL_POLICY','SELECTION_POLICY','INPUT_MODE','TARGET_POLICY'}
    tree=ast.parse(Path(__file__).read_text(encoding='utf8'))
    nodes=[n for n in tree.body if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
        or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets))]
    signature=current.hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    return signature in _REVIEWED_V6_AST




@dataclass(frozen=True)
class Config:
    seed: int = 42
    train_windows_per_step: int = 64
    validation_windows_per_step: int = 4
    test_windows_per_step: int = 4
    examples_per_sensor_family_level: int = 2
    reference_fraction: float = .35
    epochs: int = 80
    patience: int = 12
    batch_size: int = 256
    width: int = 96
    learning_rate: float = .001
    max_seconds: float = 600.
    center_batch_size: int = 8
    max_feature_mb: int = 1024
    target_f1: float = .96

    def validate(self):
        for name in ('train_windows_per_step','validation_windows_per_step','test_windows_per_step',
                     'examples_per_sensor_family_level','epochs','patience',
                     'batch_size','width','center_batch_size','max_feature_mb'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError('Positive detailed learning budgets required')
        if not 0<self.reference_fraction<1 or self.batch_size<2:
            raise ValueError('Independent reference and fitting populations required')
        if not (math.isfinite(self.max_seconds) and self.max_seconds>0 and
                math.isfinite(self.learning_rate) and self.learning_rate>0):
            raise ValueError('Finite positive MLP learning rate/time required')
        if self.center_batch_size>32:
            raise ValueError('Center batch must be 1..32')
        if not 0<self.target_f1<=1:raise ValueError('F1 goal must be in (0,1]')


def _hash(value):
    return current._sha(value)


def _domain(domain):
    if domain not in ('PECVD','PEALD'):
        raise ValueError('Explicit PECVD or PEALD learning domain required')
    return domain


def pointer(domain):
    return '.cache/external/'+_domain(domain).lower()+'/detailed_decision_head.json'


def bind(source, prepared, domain, *, source_step='DEP.'):
    from .pecvd_current_data import input_metadata
    _domain(domain)
    prepared.metadata={s:input_metadata(m) for s,m in prepared.metadata.items()}
    if domain=='PECVD':
        encoders={s:(source.encoders[s][0],dict(meta)) for s,meta in prepared.metadata.items()}
        source_steps={s:s for s in encoders}
    else:
        transferred=current.runtime_for(source,prepared,source_step=source_step)
        encoders=transferred.encoders;source_steps={s:source_step for s in encoders}
    module=SimpleNamespace(**source.module.__dict__)
    module.STEP_ENCODERS=encoders;module.FUSION_ANSWER_HEAD=None;module.FUSION_ANSWER_TRAINING=None
    module.GEVIS_MODEL.requires_grad_(False).eval()
    for encoder,meta in encoders.values():encoder.requires_grad_(False).eval()
    return SimpleNamespace(module=module,encoders=encoders,source_runtime=source,
        source_steps=source_steps,prepared=prepared,domain=domain)


def _key(step, recipe, loop, offset):
    if not isinstance(offset,(int,np.integer)) or offset<0:
        raise ValueError('Known nonnegative native elapsed offset required')
    return (str(step),str(recipe),str(loop),int(offset))


def window_key(window):
    cohort=window['cohort']
    if len(cohort)!=4:
        raise ValueError('Complete recipe/loop/native-start/elapsed context required')
    return _key(window['step'],cohort[0],cohort[1],int(cohort[-1]))


def _lineage(window):
    value=window.get('lineage_id')
    if not value:raise ValueError('Canonical source-Run lineage missing')
    return str(value)


def _complete(runtime, window):
    meta=runtime.encoders[window['step']][1]
    return (list(window.get('sensor_columns',meta['sensor_columns']))==list(meta['sensor_columns'])
        and np.asarray(window['current']).shape==(meta['window_length'],len(meta['sensor_columns']))
        and np.isfinite(window['current']).all())


def fit_references(runtime, config):
    """Disjoint TRAIN lineages fit references; remaining TRAIN lineages fit MLP."""
    config.validate();partitions=runtime.prepared.split_runs
    if any(set(partitions[a])&set(partitions[b]) for a,b in
           [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Original Run partitions overlap')
    windows=[];by_recipe=defaultdict(set)
    for step,pools in runtime.prepared.windows.items():
        for w in pools['train']:
            if w['run_id'] not in partitions['train'] or w.get('split')!='train':
                raise ValueError('References may use original TRAIN only')
            if w.get('injected',False):raise ValueError('Synthetic data cannot fit normal references')
            lineage=_lineage(w)
            if _complete(runtime,w):
                windows.append(w);by_recipe[str(w['cohort'][0])].add(lineage)
    basis=set()
    for recipe,lineages in sorted(by_recipe.items()):
        ordered=sorted(lineages,key=lambda x:_hash((config.seed,recipe,x)))
        if len(ordered)<4:continue
        count=min(len(ordered)-2,max(2,round(len(ordered)*config.reference_fraction)))
        for lineage in ordered:
            if len(lineages&basis)>=count:break
            if lineage not in basis and all(len(runs-basis)>2 for runs in by_recipe.values() if lineage in runs):
                basis.add(lineage)
    # A fixed fraction can miss normal positions that TRAIN can support. Extend
    # the independent basis using TRAIN coverage only, retaining two fitting
    # lineages per recipe. Never inspect external/Test values to fill a hole.
    positions=defaultdict(set);by_lineage=defaultdict(set)
    for w in windows:
        lineage=_lineage(w);step,recipe,loop,offset=window_key(w)
        for tick in range(offset,offset+len(w['current'])):
            key=(step,recipe,loop,tick);positions[key].add(lineage);by_lineage[lineage].add(key)
    deficits={key:2-len(runs&basis) for key,runs in positions.items() if len(runs)>=2 and len(runs&basis)<2}
    while deficits:
        eligible=[lineage for lineage in by_lineage if lineage not in basis and all(
            len(runs-basis)>2 for runs in by_recipe.values() if lineage in runs)]
        gains={lineage:sum(key in deficits for key in by_lineage[lineage]) for lineage in eligible}
        if not gains or max(gains.values())==0:break
        chosen=min(gains,key=lambda lineage:(-gains[lineage],_hash((config.seed,lineage))))
        basis.add(chosen)
        for key in by_lineage[chosen]:
            if key in deficits:
                deficits[key]-=1
                if deficits[key]==0:del deficits[key]
    all_lineages={_lineage(w) for w in windows};fitting=all_lineages-basis
    if len(basis)<2 or len(fitting)<2:
        raise ValueError('Need independent normal-reference and MLP fitting TRAIN lineages')
    grouped=defaultdict(dict)
    for w in windows:
        lineage=_lineage(w)
        if lineage in basis:
            values=np.asarray(w['current'],np.float32)
            previous=grouped[window_key(w)].get(lineage)
            grouped[window_key(w)][lineage]=(values.copy(),values.copy()) if previous is None else (
                np.minimum(previous[0],values),np.maximum(previous[1],values))
    records=[]
    for key,by_run in sorted(grouped.items()):
        # Every eligible basis Run contributes its extrema, including duplicates
        # at the same native coordinate. Do not trim tails or average normals.
        chosen=sorted(by_run)
        lower=np.minimum.reduce([by_run[r][0] for r in chosen])
        upper=np.maximum.reduce([by_run[r][1] for r in chosen])
        records.append(dict(key=list(key),lower=lower,upper=upper,reference_runs=len(chosen),lineages=chosen))
    if not records:raise ValueError('No native normal reference windows')
    support=dict(native_positions=len(positions),two_run_positions=sum(len(runs)>=2 for runs in positions.values()),
        covered_positions=sum(len(runs&basis)>=2 for runs in positions.values()),
        one_run_only_positions=sum(len(runs)<2 for runs in positions.values()),
        fitting_reserve_limited_positions=len(deficits))
    return dict(records=records,basis_lineages=sorted(basis),fitting_lineages=sorted(fitting),support_selection=support,
        target_policy=TARGET_POLICY,fit_policy='all_eligible_original_TRAIN_basis_extrema_disjoint_MLP_lineages_fixed_at_deployment')


class ReferenceBank(dict):
    """Align overlapping bounds by native tick, without bridging support holes."""
    def __init__(self,reference):
        super().__init__((tuple(r['key']),r) for r in reference['records'])
        self.points={};self.widths={}
        for r in reference['records']:
            context=tuple(r['key'][:3]);offset=int(r['key'][3])
            self.widths[context]=len(r['lower'])
            for i,(lower,upper) in enumerate(zip(r['lower'],r['upper'])):
                key=(*context,offset+i);old=self.points.get(key)
                self.points[key]=(lower.copy(),upper.copy(),set(r['lineages'])) if old is None else (
                    np.minimum(old[0],lower),np.maximum(old[1],upper),old[2]|set(r['lineages']))

    def get(self,key,default=None):
        key=tuple(key);context=key[:3];width=self.widths.get(context)
        if width is None:return default
        rows=[self.points.get((*context,key[3]+i)) for i in range(width)]
        if any(r is None or len(r[2])<2 for r in rows):return default
        return dict(key=list(key),lower=np.stack([r[0] for r in rows]),upper=np.stack([r[1] for r in rows]),
            reference_runs=min(len(r[2]) for r in rows))

    def __contains__(self,key):return self.get(key) is not None

    def __getitem__(self,key):
        value=self.get(key)
        if value is None:raise KeyError(key)
        return value


def _bank(reference):return ReferenceBank(reference)


def _common(window):
    return dict(step=window['step'],split=window['split'],run_id=window['run_id'],
        lineage_id=_lineage(window),original_id=window['pair_id'],reference_key=window_key(window),
        recipe_key=window['cohort'][0],loop_key=str(window['cohort'][1]),offset_samples=int(window['cohort'][-1]),
        sensor_columns=list(window['sensor_columns']))


def case_current(case):
    if 'sensor_update' not in case:return case['current']
    values=np.array(case['current'],copy=True)
    sensor,changed=case['sensor_update'];values[:,sensor]=changed
    return values


def envelope_targets(case,record):
    """Inclusive interval truth; operator/strength truth needs a clean inside base."""
    observed=case_current(case);lower=record['lower'];upper=record['upper']
    outside=np.any((observed<lower)|(observed>upper),axis=0)
    base=case.get('original_current',case['current'])  # label provenance only
    base_outside=np.any((base<lower)|(base>upper),axis=0)
    original=np.asarray(case['targets']);targets=np.zeros_like(original)
    for i,name in enumerate(case['centers']):
        sid=case['sensor_columns'].index(name)
        if outside[sid]:
            targets[i]=(1,-1,-1)
            if case['injected'] and original[i,0] and not base_outside[sid]:
                targets[i]=original[i]
    return targets


def training_cases(runtime, reference, config):
    """Independent fitting originals plus known boundary-relative interventions."""
    from . import boundary_injection as boundary
    bank=_bank(reference);fitting=set(reference['fitting_lineages']);originals=[];excluded=[]
    for step,pools in runtime.prepared.windows.items():
        names=list(runtime.encoders[step][1]['sensor_columns'])
        pool=[w for w in pools['train'] if _lineage(w) in fitting and _complete(runtime,w) and window_key(w) in bank]
        for w in p._select(pool,config.train_windows_per_step,config.seed):
            base=np.array(w['current'],np.float32,copy=True)
            originals.append(dict(_common(w),current=base,original_current=base,centers=names,
                targets=np.zeros((len(names),3),np.int64),kind='equal',injected=False,pair_id=w['pair_id']+'/normal'))
        excluded.append(dict(step=step,reason='incomplete_grid_or_missing_reference',windows=sum(
            _lineage(w) in fitting and (not _complete(runtime,w) or window_key(w) not in bank) for w in pools['train'])))
    extra,coverage=boundary.generate(originals,bank,seed=config.seed,
        examples_per_combination=config.examples_per_sensor_family_level,
        sensor_inventory={s:m['sensor_columns'] for s,(e,m) in runtime.encoders.items()},split='train')
    if not extra:raise ValueError('No supported known boundary interventions; inspect sensor/family/stage coverage')
    return originals+extra,coverage,excluded


def evaluation_cases(runtime, split, config):
    """Retain the case schedule; interval labels are assigned after support checks."""
    if split not in ('validation','test'):raise ValueError('Held-out split required')
    cfg=current.Config(seed=config.seed,validation_windows_per_step=config.validation_windows_per_step,
        test_windows_per_step=config.test_windows_per_step)
    cases=current.cases(runtime,split,config=cfg)
    lookup={(s,w['pair_id']):w for s,pools in runtime.prepared.windows.items() for w in pools[split]}
    for case in cases:
        w=lookup[(case['step'],case['original_id'])]
        case.update(_common(w))
        case['original_current']=w['current']  # never passed to encoder/Fusion
    return cases


def controlled_cases(runtime,split,config,reference):
    """Declared new heldout protocol; historical case construction stays separate."""
    from . import boundary_injection as boundary
    originals=[c for c in evaluation_cases(runtime,split,config) if not c['injected']]
    bank=_bank(reference)
    extra,coverage=boundary.generate(originals,bank,seed=config.seed+{'validation':100003,'test':200003}[split],
        examples_per_combination=1,sensor_inventory={s:m['sensor_columns'] for s,(e,m) in runtime.encoders.items()},split=split)
    return originals+extra,coverage


def supported_cases(runtime, cases, reference, *, training=False):
    bank=_bank(reference);basis=set(reference['basis_lineages']);fitting=set(reference['fitting_lineages']);rows=[];audit=[]
    for case in cases:
        meta=runtime.encoders[case['step']][1];reason=None
        if case['lineage_id'] in basis:reason='reference_basis_lineage_not_independent'
        elif not training and case['lineage_id'] in fitting:reason='heldout_lineage_overlaps_TRAIN'
        elif list(case['sensor_columns'])!=list(meta['sensor_columns']):reason='incomplete_observed_sensor_grid'
        elif tuple(case['reference_key']) not in bank:reason='native_normal_reference_unobserved'
        audit.append(dict(step=case['step'],split=case['split'],pair_id=case['pair_id'],
            requested_targets=len(case['centers']),changed_targets=int(case['targets'][:,0].sum()),
            supported=reason is None,reason=reason))
        if reason is None:
            targets=envelope_targets(case,bank[tuple(case['reference_key'])])
            audit[-1].update(injected_targets=int(case['targets'][:,0].sum()),
                changed_targets=int(targets[:,0].sum()),unknown_typed_targets=int((targets[:,1:]<0).sum()))
            rows.append(dict(case,targets=targets))
    return rows,audit


def _features(runtime, case, bank):
    return _window_features(runtime,case_current(case),bank[tuple(case['reference_key'])],
        case['step'],case['centers'],case['offset_samples'],runtime.config.center_batch_size)[0]


def _window_features(runtime,observed,record,step,centers,offset,center_batch_size,*,answer_types=None):
    """Shared training/inference path: two boundaries, never a central waveform."""
    values=[];tasks_all=[];module=SimpleNamespace(**runtime.module.__dict__);original=module.numeric_state;states={}
    def numeric_once(encoder,a,b):
        # Distinct boundary states must survive center chunk reuse.
        key=b.detach().cpu().numpy().tobytes()
        if key not in states:states[key]=original(encoder,a,b)
        return states[key]
    module.numeric_state=numeric_once
    names=list(runtime.encoders[step][1]['sensor_columns']);cadence=float(runtime.encoders[step][1]['cadence_seconds'])
    lower=record['lower'];upper=record['upper'];span=upper-lower
    kinds=fa.KINDS if answer_types is None else tuple(answer_types)
    query={} if answer_types is None else dict(answer_types=kinds)
    scale=np.maximum(span,1.)  # TRAIN-standardized units; finite for constant bounds.
    below=np.maximum(lower-observed,0.)/scale;above=np.maximum(observed-upper,0.)/scale
    for start in range(0,len(centers),center_batch_size):
        batch=centers[start:start+center_batch_size]
        low,tasks=fa.fusion_features(module,observed,lower,step,batch,position=0.,**query)
        high,_=fa.fusion_features(module,observed,upper,step,batch,position=0.,**query)
        evidence=[]
        for name,kind in tasks:
            sid=names.index(name);lo=below[:,sid];hi=above[:,sid]
            evidence.append([float(np.mean(lo>0)),float(np.mean(hi>0)),float(lo.mean()),float(hi.mean()),
                float(lo.max()),float(hi.max()),float(span[:,sid].mean()),float(span[:,sid].max())])
        coordinates=torch.tensor([[math.log1p(offset*cadence),math.log1p(len(observed)*cadence)]]*len(tasks))
        x=torch.cat([low[:,:-2],high[:,:-2],torch.tensor(evidence,dtype=low.dtype),coordinates],-1)
        values.append(x.reshape(len(batch),len(kinds),-1));tasks_all.extend(tasks)
    return torch.cat(values),tasks_all


def encoded_cases(runtime,cases,reference,*,cache_root=None,source=None,progress=None):
    bank=_bank(reference);checksum=[]
    for c in cases:
        checksum.append(dict(key=c['pair_id'],step=c['step'],reference_key=c['reference_key'],
            current_sha256=current.hashlib.sha256(np.ascontiguousarray(case_current(c)).tobytes()).hexdigest(),
            centers=c['centers'],targets=c['targets'].tolist()))
    identity=dict(recipe=RECIPE,code=p.digest(__file__),source=source,references=_hash(reference),cases=checksum)
    path=Path(cache_root)/(_hash(identity)[:24]+'.pt') if cache_root is not None else None
    if path is not None and path.is_file():
        # Existing exact caches stay immutable and memory mapped on CPU.
        record=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if record.get('identity')!=identity or record.get('content_sha256')!=current._feature_content(identity,record['features']):
            raise ValueError('Detailed Fusion feature cache identity/content differs')
        features=record['features']
        if len(features)!=len(cases):raise ValueError('Incomplete detailed Fusion cache')
        if progress:progress('Reused memory-mapped frozen Fusion features',flush=True)
        for c,f in zip(cases,features):
            if not isinstance(f,torch.Tensor) or f.requires_grad or f.ndim!=3 or f.shape[:2]!=(len(c['centers']),3) or not torch.isfinite(f).all():
                raise ValueError('Invalid detailed Fusion feature grid')
    elif path is not None:
        cache=FeatureCache(path.with_suffix(''),identity,runtime.config.max_feature_mb*1024**2)
        features=[];started=time.perf_counter();reused=0
        for i,case in enumerate(cases):
            feature=cache.get(i,len(case['centers']))
            if feature is None:
                value=_features(runtime,case,bank).float().cpu()
                if value.shape[:2]!=(len(case['centers']),3):raise ValueError('Invalid detailed Fusion feature grid')
                feature=cache.put(i,value)
                del value
            else:reused+=1
            features.append(feature)
            if progress and (i==0 or (i+1)%100==0 or i+1==len(cases)):
                progress(f'Disk Fusion features {i+1}/{len(cases)}; reused={reused}; {time.perf_counter()-started:.1f}s',flush=True)
        cache.finish(features)
    else:
        features=[];used=0
        for case in cases:
            value=_features(runtime,case,bank).float().cpu();used+=value.numel()*value.element_size()
            if used>runtime.config.max_feature_mb*1024**2:raise MemoryError('In-memory features exceed budget; provide cache_root for disk streaming')
            features.append(value)
    return [dict(c,features=f) for c,f in zip(cases,features)]


def fit_mlp(train,validation,config,device='cpu',progress=None,*,initial_state=None):
    """Mixed-label MLP batches; complete epochs; natural-population Validation."""
    config.validate();torch.manual_seed(config.seed)
    if {c['lineage_id'] for c in train}&{c['lineage_id'] for c in validation}:
        raise ValueError('MLP TRAIN/Validation source lineages overlap')
    working_bytes=config.max_feature_mb*1024**2
    x=FeatureRows(train,working_bytes);vx=FeatureRows(validation,working_bytes)
    y=torch.as_tensor(np.concatenate([c['targets'] for c in train]),device='cpu')
    vy=torch.as_tensor(np.concatenate([c['targets'] for c in validation]),device='cpu')
    if len(y)!=x.shape[0] or len(vy)!=vx.shape[0]:raise ValueError('Feature/target row counts differ')
    metric_y=y.clone();metric_vy=vy.clone()
    # Type/strength are supervised only when true BOOLEAN is Yes. Normal rows
    # teach the gate, without asking positive-family heads to learn class zero.
    y[y[:,0].eq(0),1:]=-1;vy[vy[:,0].eq(0),1:]=-1
    if x.shape[1:]!=vx.shape[1:]:raise ValueError('Fusion feature dimensions differ')
    head=fa.ContextReadout(x.shape[-1],config.width).to(device)
    if initial_state is not None:head.load_state_dict(initial_state,strict=True)
    if initial_state is None:
        average,scale=x.moments(config.batch_size);head.feature_mean.copy_(average)
        head.feature_scale.copy_(torch.where(scale>1e-5,scale,torch.ones_like(scale)))
    weights={};positive=np.flatnonzero(y[:,0].cpu().numpy()==1);negative=np.flatnonzero(y[:,0].cpu().numpy()==0)
    if not len(positive) or not len(negative):raise ValueError('MLP needs normal and changed TRAIN targets')
    for i,k in enumerate(fa.KINDS):
        known=y[:,i]>=0;held=vy[:,i]>=0
        counts=torch.bincount(y[known,i],minlength=len(fa.ANSWER_CRITERIA[k]));allowed=counts>0
        if initial_state is not None:allowed=allowed|getattr(head,'allowed_'+k).detach().cpu()
        if not known.any() or not held.any():raise ValueError('Changed typed TRAIN/Validation support required for each answer family')
        if (k=='BOOLEAN' and allowed.sum()<2) or not (y[known,i]>0).any() or not (vy[held,i]>0).any():
            raise ValueError('Changed typed TRAIN/Validation support required; all-No family F1 cannot meet the goal')
        getattr(head,'allowed_'+k).copy_(allowed)
        if not allowed[vy[held,i]].all():raise ValueError('Validation answer absent in detailed TRAIN support')
        if k=='BOOLEAN':weights[k]=None;continue
        changed=torch.bincount(y[y[:,0].eq(1)&known,i],minlength=len(counts)).float();active=changed>0
        values=torch.zeros_like(changed);values[0]=1.
        if active.any():values[active]=changed.sum()/(active.sum()*changed[active])
        weights[k]=values.to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=config.learning_rate,weight_decay=1e-4)
    history=[];best=None;best_key=None;best_score=None;stale=0;updates=0;best_updates=0;started=time.perf_counter();stop='epochs_completed';best_epoch=None
    def score(features,truth):
        predictions=[];loss=0.;n=len(truth)
        loss_targets=truth.clone();loss_targets[loss_targets[:,0].eq(0),1:]=-1
        with torch.no_grad():
            for start in range(0,n,config.batch_size):
                batch=features.take(np.arange(start,min(start+config.batch_size,n))).to(device)
                logits={k:head(batch[:,i],k) for i,k in enumerate(fa.KINDS)}
                target=loss_targets[start:start+config.batch_size].to(device)
                gate=logits['BOOLEAN'].argmax(-1);pred=[gate]
                for kind in ('CHOICE','ORDINAL'):
                    conditional=logits[kind].clone();conditional[:,0]=-torch.inf
                    pred.append(torch.where(gate.eq(1),conditional.argmax(-1),torch.zeros_like(gate)))
                predictions.append(torch.stack(pred,-1).cpu().numpy())
                loss+=sum(float(F.cross_entropy(logits[k],target[:,i],weight=weights[k],ignore_index=-1,reduction='sum')) for i,k in enumerate(fa.KINDS))
        return _scores(truth.cpu().numpy(),np.concatenate(predictions)),loss/max(1,int((loss_targets>=0).sum()))
    for epoch in range(config.epochs):
        if history and time.perf_counter()-started>=config.max_seconds:stop='wall_time_budget_after_complete_epoch';break
        rng=np.random.default_rng(config.seed+epoch);draws=max(len(positive),len(negative))
        pos=np.resize(rng.permutation(positive),draws);neg=np.resize(rng.permutation(negative),draws)
        half=max(1,config.batch_size//2);loss_sum=0.;target_count=0;head.train()
        for start in range(0,draws,half):
            chosen=np.r_[pos[start:start+half],neg[start:start+half]];rng.shuffle(chosen)
            ids=torch.as_tensor(chosen,device='cpu');optimizer.zero_grad(set_to_none=True)
            batch=x.take(chosen).to(device);target=y[ids].to(device)
            loss=sum(F.cross_entropy(head(batch[:,i],k),target[:,i],weight=weights[k],ignore_index=-1,reduction='sum') for i,k in enumerate(fa.KINDS))/(3*len(ids))
            if not torch.isfinite(loss):raise ValueError('Nonfinite detailed MLP loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();updates+=1
            loss_sum+=float(loss.detach())*len(ids);target_count+=len(ids)
        head.eval();train_score,_=score(x,metric_y);validation_score,val_loss=score(vx,metric_vy)
        history.append(dict(epoch=epoch+1,epoch_completed=True,train_loss=loss_sum/target_count,
            validation_loss=val_loss,train_positive_f1=train_score['boolean_positive_f1'],
            train_normal_fpr=train_score['normal_false_positive_rate'],**validation_score,
            validation_f1=typed_f1(validation_score),
            elapsed_seconds=time.perf_counter()-started))
        key=selection_key(validation_score,val_loss)
        if best_key is None or key>best_key:
            best_key=key;best_epoch=epoch+1;best_updates=updates;best_score=validation_score;best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};stale=0
        else:stale+=1
        if progress:progress(f'MLP epoch {epoch+1}: TRAIN F1={train_score["boolean_positive_f1"]:.4f}; '
            f'Validation F1={typed_f1(validation_score)}, FPR={validation_score["normal_false_positive_rate"]:.4f}',flush=True)
        if stale>=config.patience:stop='validation_patience';break
    if best is None:raise ValueError('No trained detailed MLP state')
    head.load_state_dict(best,strict=True);head.requires_grad_(False).eval()
    return head,dict(history=history,selected_epoch=best_epoch,optimizer_updates=best_updates,attempted_optimizer_updates=updates,stop_reason=stop,
        fit_seconds=time.perf_counter()-started,training_recipe=RECIPE,
        selection_policy=SELECTION_POLICY,
        warm_started=initial_state is not None,
        feature_storage='CPU/disk rows; device minibatches only',
        feature_normalization='source_frozen' if initial_state is not None else 'target_TRAIN',
        answer_class_support='source_union_target_TRAIN' if initial_state is not None else 'target_TRAIN',
        selected_validation_f1=typed_f1(best_score),target_f1=config.target_f1,
        validation_goal_met=min(typed_f1(best_score).values())>=config.target_f1,
        selection='full-population conditional weakest typed Validation F1; mean typed F1; lower normal FPR; lower loss')


def _scores(truth,predictions):
    # Boolean interval truth is complete. Unknown family/strength is never No.
    truth=np.asarray(truth);predictions=np.asarray(predictions)
    score=current._scores(np.maximum(truth,0),predictions)
    counts={}
    for i,kind in enumerate(fa.KINDS):
        known=truth[:,i]>=0;f1=[];counts[kind]=int(known.sum())
        for label in range(0 if kind=='BOOLEAN' else 1,len(fa.ANSWER_CRITERIA[kind])):
            t=truth[known,i]==label;p=predictions[known,i]==label
            if t.any() or p.any():f1.append(2*int((t&p).sum())/max(1,int(t.sum()+p.sum())))
        score['macro_f1'][kind]=float(np.mean(f1)) if f1 else float('nan')
    complete=np.all(truth>=0,axis=1)
    score['joint_accuracy']=float(np.all(truth[complete]==predictions[complete],axis=1).mean()) if complete.any() else float('nan')
    score['always_no_joint_accuracy']=float(np.all(truth[complete]==0,axis=1).mean()) if complete.any() else float('nan')
    score['scored_targets']=counts;score['joint_scored_targets']=int(complete.sum())
    score['routed_normal_typed_calls']={kind:int(((truth[:,i]==0)&(predictions[:,0]==1)).sum())
        for i,kind in enumerate(fa.KINDS) if kind!='BOOLEAN'}
    return score


def typed_f1(score):
    return {k:float(score['boolean_positive_f1'] if k=='BOOLEAN' else score['macro_f1'][k]) for k in fa.KINDS}


def selection_key(score,loss):
    values=list(typed_f1(score).values())
    return (min(values),float(np.mean(values)),-score['normal_false_positive_rate'],-loss)


def _codes():
    here=Path(__file__)
    paths=[here.with_name(name) for name in ('decision_training.py','decision_replay.py','pecvd_current_data.py',
        'peald_current_data.py','peald_training.py','current_input.py','fusion_answers.py','injection.py','boundary_injection.py','feature_store.py')]
    paths += [here.parent.parent/name for name in ('temporal_bridge.py','relational_features.py')]
    return {path.name:p.digest(path) for path in paths}


def train_or_load(root,runtime,*,config=Config(),allow_training=False,progress=print):
    from .pecvd_current_data import input_metadata
    root=Path(root).resolve();config.validate();runtime.config=config
    if runtime.domain=='PEALD' and (not getattr(runtime,'initial_head_identity',None) or not getattr(runtime,'initial_head_state',None)):
        raise ValueError('Prepare the selected pretrained CVD encoder/head before ALD training')
    initial=getattr(runtime,'initial_head_identity',None)
    if initial is not None:
        path=Path(initial['checkpoint']).expanduser();path=(path if path.is_absolute() else root/path).resolve()
        path.relative_to(root/'artifacts/gevis_decision_training/pecvd')
        initial=dict(initial,checkpoint=path.relative_to(root).as_posix());runtime.initial_head_identity=initial
        if (p.digest(path)!=initial['checkpoint_sha256']
                or current.payload_digest(dict(state=runtime.initial_head_state))!=initial['state_sha256']):
            raise ValueError('Pretrained CVD head state/checkpoint changed before target fitting')
    runtime.prepared.metadata={s:input_metadata(m) for s,m in runtime.prepared.metadata.items()}
    runtime.encoders={s:(encoder,runtime.prepared.metadata[s]) for s,(encoder,_) in runtime.encoders.items()}
    runtime.module.STEP_ENCODERS=runtime.encoders
    source=current.source_identity(runtime)
    identity=dict(schema=SCHEMA,recipe=RECIPE,domain=runtime.domain,config=asdict(config),
        selection_policy=SELECTION_POLICY,ordinal_policy=ORDINAL_POLICY,
        initial_head=getattr(runtime,'initial_head_identity',None),
        source=source,source_steps=runtime.source_steps,metadata=runtime.prepared.metadata,
        preparation=runtime.prepared.identity,codes=_codes(),fusion_helper_sha256=p.digest(fa.__file__))
    path=root/'artifacts/gevis_decision_training'/runtime.domain.lower()/_hash(identity)[:24]/'answer_head.pt'
    if path.is_file():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        validate_payload(payload,path,runtime.domain)
        if _hash(payload['identity'])!=_hash(identity):raise ValueError('Detailed head identity changed')
        head=fa.ContextReadout(payload['dimension'],config.width)
        head.load_state_dict(payload['state'],strict=True);training=payload['training'];reference=payload['references']
    else:
        if not allow_training:
            raise FileNotFoundError('Run STEP3 detailed training for the selected domain; legacy weights cannot substitute')
        started=time.perf_counter();reference=fit_references(runtime,config)
        train,coverage,excluded=training_cases(runtime,reference,config)
        train,train_audit=supported_cases(runtime,train,reference,training=True)
        validation,validation_coverage=controlled_cases(runtime,'validation',config,reference)
        validation,validation_audit=supported_cases(runtime,validation,reference)
        if not train or not validation:raise ValueError('No independent supported TRAIN/Validation cases; inspect native reference support')
        # A type/level remaining inside the normal envelope is not forced positive
        # to satisfy a coverage quota. Actual supported classes are checked below.
        if progress:
            progress(f'Detailed {runtime.domain}: {len(train)} TRAIN cases, {len(validation)} supported Validation cases. '
                'Fusion preparation is separate from MLP fitting.',flush=True)
        cache=root/'.cache/detailed-fusion-features'
        train=encoded_cases(runtime,train,reference,cache_root=cache,source=source,progress=progress)
        validation=encoded_cases(runtime,validation,reference,cache_root=cache,source=source,progress=progress)
        preparation_seconds=time.perf_counter()-started
        head,training=fit_mlp(train,validation,config,runtime.module.DEVICE,progress,
            initial_state=getattr(runtime,'initial_head_state',None))
        training.update(input_mode=INPUT_MODE,reference_mode='packaged_independent_TRAIN',
            checkpoint_binding=SCHEMA,target_domain=runtime.domain,trained_steps=list(runtime.encoders),ordinal_policy=ORDINAL_POLICY,
            original_runs_normal=False,target_policy=TARGET_POLICY,coverage=coverage,validation_coverage=validation_coverage,
            boundary_augmentation_audit=[row for row in excluded if row.get('boundary_augmentation')],
            preparation_exclusions=[row for row in excluded if not row.get('boundary_augmentation')],
            conditional_answering=True,typed_training='known_BOOLEAN_positive_only',
            initial_head=getattr(runtime,'initial_head_identity',None),
            confidence_abstention=False,reference_support_selection=reference['support_selection'],
            support_audit=train_audit+validation_audit,preparation_seconds=preparation_seconds,
            heldout_normal_envelope=[dict(split='validation',step=case['step'],
                questions=len(case['centers']),outside=int(case['targets'][:,0].sum()))
                for case in validation if not case['injected']],
            reference_basis_lineages=len(reference['basis_lineages']),fitting_lineages=len(reference['fitting_lineages']))
        if current.source_identity(runtime)!=source or _codes()!=identity['codes'] or p.digest(fa.__file__)!=identity['fusion_helper_sha256']:
            raise ValueError('Frozen source/code changed during detailed fitting')
        if initial is not None and p.digest(root/initial['checkpoint'])!=initial['checkpoint_sha256']:
            raise ValueError('Pretrained CVD checkpoint changed during target fitting')
        inputs=dict(runtime.prepared.identity.get('source_files',{}))
        inputs.update(runtime.prepared.identity.get('cache_files',{}))
        for rel,checksum in inputs.items():
            if p.digest(root/rel)!=checksum:raise ValueError('Original training input changed')
        payload=dict(schema=SCHEMA,identity=identity,references=reference,dimension=head.feature_mean.numel(),
            state={k:v.detach().cpu() for k,v in head.state_dict().items()},training=training,
            process_columns=dict(runtime.prepared.process.source_columns))
        payload['content_sha256']=current.payload_digest(payload)
        current._atomic_torch(payload,path)
    runtime.references=reference;runtime.reference_bank=_bank(reference)
    runtime.module.FUSION_ANSWER_HEAD=head.requires_grad_(False).eval().to(runtime.module.DEVICE)
    runtime.module.FUSION_ANSWER_TRAINING=training;runtime.module.FUSION_ANSWER_PATH=path
    record=dict(schema=SCHEMA,domain=runtime.domain,checkpoint=path.relative_to(root).as_posix(),sha256=p.digest(path))
    handoff=root/pointer(runtime.domain);handoff.parent.mkdir(parents=True,exist_ok=True)
    temporary=handoff.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(handoff)
    return training


def validate_payload(payload,path,domain):
    if payload.get('schema')!=SCHEMA or payload.get('content_sha256')!=current.payload_digest(payload):
        raise ValueError('Detailed decision package schema/content differs')
    identity=payload['identity'];training=payload['training'];reference=payload['references']
    if identity.get('domain')!=_domain(domain) or training.get('target_domain')!=domain:
        raise ValueError('Another domain head cannot substitute')
    if _hash(training.get('initial_head'))!=_hash(identity.get('initial_head')):
        raise ValueError('Pretrained source head provenance differs')
    if domain=='PEALD' and not identity.get('initial_head'):raise ValueError('Missing pretrained CVD warm-start binding')
    codes=_codes();previous={k:v for k,v in codes.items() if k!='feature_store.py'}
    previous['decision_training.py']=_REVIEWED_V6_CODE
    reviewed_previous=(identity.get('codes')==previous and _reviewed_v6_inference_matches())
    if (identity.get('recipe')!=RECIPE or not (identity.get('codes')==codes or reviewed_previous)
            or identity.get('fusion_helper_sha256')!=p.digest(fa.__file__)):
        raise ValueError('Detailed decision source/recipe changed; preserve old weights and train a new package')
    if identity.get('selection_policy')!=SELECTION_POLICY or training.get('selection_policy')!=SELECTION_POLICY:
        raise ValueError('Validation selection policy differs from model identity')
    if identity.get('ordinal_policy')!=ORDINAL_POLICY or training.get('ordinal_policy')!=ORDINAL_POLICY:
        raise ValueError('Boundary-excess ordinal contract differs; run STEP3 for the new recipe')
    if Path(path).parent.name!=_hash(identity)[:24]:raise ValueError('Detailed package namespace differs')
    if training.get('input_mode')!=INPUT_MODE or training.get('checkpoint_binding')!=SCHEMA or training.get('optimizer_updates',0)<1:
        raise ValueError('Missing actual detailed MLP fitting evidence')
    if not training.get('conditional_answering') or training.get('typed_training')!='known_BOOLEAN_positive_only':
        raise ValueError('Conditional source head recipe is required')
    if set(training.get('trained_steps',[]))!=set(identity['metadata']):raise ValueError('Trained Step inventory differs')
    Config(**identity['config']).validate()
    for meta in identity['metadata'].values():
        names=meta['sensor_columns'];mean=np.asarray(meta['mean']).reshape(-1);scale=np.asarray(meta['std']).reshape(-1)
        if (not names or len(names)!=len(set(names)) or mean.shape!=(len(names),) or scale.shape!=mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any()
                or not math.isfinite(meta['cadence_seconds']) or meta['cadence_seconds']<=0):
            raise ValueError('Invalid detailed normalizer/native timing metadata')
    basis=set(reference['basis_lineages']);fitting=set(reference['fitting_lineages'])
    if reference.get('target_policy')!=TARGET_POLICY or training.get('target_policy')!=TARGET_POLICY:
        raise ValueError('Normal envelope target policy differs')
    if len(basis)<2 or len(fitting)<2 or basis&fitting:raise ValueError('Reference/MLP source lineages overlap')
    if len(_bank(reference))!=len(reference['records']):raise ValueError('Duplicate normal reference context')
    for record in reference['records']:
        step,recipe,loop,offset=record['key'];key=_key(step,recipe,loop,offset)
        if step not in identity['metadata']:raise ValueError('Unknown reference Step')
        meta=identity['metadata'][step];lower=np.asarray(record['lower']);upper=np.asarray(record['upper'])
        if (lower.shape!=(meta['window_length'],len(meta['sensor_columns'])) or upper.shape!=lower.shape
                or not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower>upper).any()):
            raise ValueError('Invalid packaged normal reference grid')
        if len(set(record['lineages']))!=record['reference_runs'] or record['reference_runs']<1 or not set(record['lineages'])<=basis:
            raise ValueError('Reference lineage provenance differs')
    if reviewed_previous:training=dict(training,runtime_compatibility='reviewed_v6_inference_only')
    return identity,training


def load(root,*,domain='PECVD',checkpoint=None):
    """Models, scalar metadata and packaged references only; no process CSV/cache."""
    from .external_test import InputContract
    root=Path(root).resolve();_domain(domain)
    if checkpoint is None:
        handoff=root/pointer(domain)
        if not handoff.is_file():raise FileNotFoundError('No detailed '+domain+' model. Run STEP3 for that domain first.')
        record=json.loads(handoff.read_text(encoding='utf-8'))
        if record.get('schema')!=SCHEMA or record.get('domain')!=domain:raise ValueError('Detailed completion pointer domain/schema differs')
        checkpoint=root/record['checkpoint']
        if p.digest(checkpoint)!=record['sha256']:raise ValueError('Detailed checkpoint changed')
    path=Path(checkpoint).expanduser();path=(path if path.is_absolute() else root/path).resolve()
    path.relative_to(root/'artifacts/gevis_decision_training'/domain.lower())
    before=p.digest(path);payload=torch.load(path,map_location='cpu',weights_only=False)
    identity,training=validate_payload(payload,path,domain);source=current.load_frozen_source(root)
    encoders={s:(source.encoders[identity['source_steps'][s]][0],dict(meta)) for s,meta in identity['metadata'].items()}
    module=SimpleNamespace(**source.module.__dict__);module.STEP_ENCODERS=encoders
    runtime=SimpleNamespace(module=module,encoders=encoders,source_runtime=source,source_steps=identity['source_steps'],domain=domain,
        references=payload['references'],reference_bank=_bank(payload['references']),config=Config(**identity['config']),
        prepared=SimpleNamespace(identity=identity['preparation'],metadata=identity['metadata'],
            split_runs=identity['preparation']['split_runs']))
    if current.source_identity(runtime)!=identity['source']:raise ValueError('Frozen GEVIS source weights/recipe differ')
    head=fa.ContextReadout(payload['dimension'],runtime.config.width);head.load_state_dict(payload['state'],strict=True)
    module.FUSION_ANSWER_HEAD=head.requires_grad_(False).eval().to(module.DEVICE)
    module.FUSION_ANSWER_TRAINING=training;module.FUSION_ANSWER_PATH=path
    if p.digest(path)!=before:raise ValueError('Detailed checkpoint changed while loading')
    return InputContract(runtime,SimpleNamespace(source_columns=payload['process_columns']),domain=domain)


class RowReceiver:
    """Validated native sensor rows -> one causal rolling request per new row.

    Callers own raw timestamp/unit validation. For raw ALD records use the native
    Receiver, retaining SubCycle completeness and provisional Main semantics.
    """
    def __init__(self,session):
        from collections import deque
        self.session=session;self._deque=deque;self.reset()

    def reset(self):self._rows=self._deque();self._key=None;self._last_offset=None

    def push(self,values,*,step,recipe_key,offset_samples,lineage_id,quality_validated,
             run_id=None,occurrence_id=None,loop_key='',sensor_columns=None,centers=None,standardized=False):
        meta=self.session.metadata.get(step)
        if meta is None or quality_validated is not True:
            self.reset();return dict(status='NOT_EVALUATED',reason='native_quality_not_validated' if meta else 'unsupported_step',answers=[])
        if not lineage_id or occurrence_id is None:
            self.reset();return dict(status='NOT_EVALUATED',reason='source_lineage_and_occurrence_required',answers=[])
        if not isinstance(offset_samples,(int,np.integer)) or isinstance(offset_samples,(bool,np.bool_)) or offset_samples<0:
            self.reset();raise ValueError('Known nonnegative native row offset required')
        names=list(meta['sensor_columns']) if sensor_columns is None else list(sensor_columns)
        row=np.asarray(values,np.float32)
        if names!=list(meta['sensor_columns']) or row.shape!=(len(names),) or not np.isfinite(row).all():
            self.reset();return dict(status='NOT_EVALUATED',reason='incomplete_or_invalid_sensor_row',answers=[])
        key=(lineage_id,run_id,occurrence_id,step,recipe_key,str(loop_key),tuple(names),
            None if centers is None else tuple(centers),bool(standardized))
        gap=self._key==key and self._last_offset is not None and offset_samples!=self._last_offset+1
        if key!=self._key or gap:
            self.reset();self._key=key;self._rows=self._deque(maxlen=meta['window_length'])
        self._rows.append(row.copy());self._last_offset=int(offset_samples)
        if len(self._rows)<meta['window_length']:
            return dict(status='NOT_READY',reason='native_gap_reset' if gap else 'initial_window_warmup',
                observed_rows=len(self._rows),required_rows=meta['window_length'],answers=[])
        start=int(offset_samples)-meta['window_length']+1
        reply=self.session.process_window(current=np.stack(self._rows),step=step,recipe_key=recipe_key,
            loop_key=loop_key,offset_samples=start,run_id=run_id,lineage_id=lineage_id,
            centers=centers,sensor_columns=names,standardized=standardized)
        reply.update(window_start_offset=start,window_end_offset=int(offset_samples)+1,
            prediction_scope='rolling_window_ending_at_new_row')
        return reply


class Session:
    supports_reference_context=True

    def __init__(self,runtime,*,center_batch_size=8):
        training=runtime.module.FUSION_ANSWER_TRAINING or {}
        if training.get('checkpoint_binding')!=SCHEMA or training.get('optimizer_updates',0)<1:
            raise ValueError('A trained detailed domain head is required')
        if type(center_batch_size) is not int or not 1<=center_batch_size<=32:raise ValueError('Center batch must be 1..32')
        self.runtime=runtime;self.metadata={s:dict(m) for s,(e,m) in runtime.encoders.items()};self.center_batch_size=center_batch_size

    @torch.no_grad()
    def process_window(self,*,current,step,recipe_key=None,loop_key='',offset_samples=0,
                       run_id=None,lineage_id=None,centers=None,sensor_columns=None,standardized=False):
        from .online_inference import sync
        runtime=self.runtime;module=runtime.module;sync(module);started=time.perf_counter()
        timings=dict(normalization_ms=0.,reference_ms=0.,encoder_ms=0.,fusion_ms=0.,head_ms=0.)
        reply=dict(status='NOT_EVALUATED',reason='unsupported_step',step=step,answers=[],input_mode=INPUT_MODE,reference_count=0)
        def finish():
            sync(module);timings['request_ms']=1000*(time.perf_counter()-started);reply['timings']=timings;return reply
        if step not in self.metadata:return finish()
        meta=self.metadata[step];names=list(meta['sensor_columns'])
        observed=names if sensor_columns is None else list(sensor_columns)
        if observed!=names:reply['reason']='incomplete_observed_sensor_grid';return finish()
        chosen=names if centers is None else list(centers)
        if not chosen or len(chosen)!=len(set(chosen)) or any(c not in names for c in chosen):raise ValueError('Unique known centers required')
        if not isinstance(lineage_id,str) or not lineage_id:
            reply['reason']='source_lineage_required';return finish()
        if lineage_id in set(runtime.references['basis_lineages']):
            reply['reason']='reference_basis_lineage_not_independent';return finish()
        key=_key(step,recipe_key,loop_key,offset_samples)
        record=runtime.reference_bank.get(key)
        if record is None:reply['reason']='native_normal_reference_unobserved';return finish()
        now=time.perf_counter();values=np.asarray(current,np.float32)
        if values.shape!=(meta['window_length'],len(names)) or not np.isfinite(values).all():raise ValueError('Finite complete native sensor window required')
        if not standardized:values=((values-np.asarray(meta['mean']))/np.asarray(meta['std'])).astype(np.float32)
        timings['normalization_ms']=1000*(time.perf_counter()-now)
        local=SimpleNamespace(**module.__dict__);original=module.numeric_state;states={}
        def timed_numeric(encoder,a,b):
            key=b.detach().cpu().numpy().tobytes()
            if key in states:return states[key]
            sync(module);t=time.perf_counter();state=original(encoder,a,b);sync(module)
            timings['encoder_ms']+=1000*(time.perf_counter()-t);states[key]=state;return state
        local.numeric_state=timed_numeric
        local_runtime=SimpleNamespace(module=local,encoders=runtime.encoders)
        head=module.FUSION_ANSWER_HEAD;decoded={}
        def infer(batch,kinds):
            previous=timings['encoder_ms'];now=time.perf_counter()
            features,tasks=_window_features(local_runtime,values,record,step,batch,offset_samples,self.center_batch_size,answer_types=kinds)
            features=features.reshape(-1,features.shape[-1])
            sync(module);timings['fusion_ms']+=1000*(time.perf_counter()-now)-(timings['encoder_ms']-previous)
            now=time.perf_counter();features=features.to(module.DEVICE)
            for i,(center,kind) in enumerate(tasks):
                logits=head(features[i:i+1],kind).clone()
                if kind!='BOOLEAN':
                    if not getattr(head,'allowed_'+kind)[1:].any():raise ValueError('Source head has no trained positive answer class')
                    logits[:,0]=-torch.inf
                probabilities=logits.softmax(-1)[0].cpu();prediction=int(probabilities.argmax())
                decoded[(center,kind)]=dict(center_sensor=center,answer_type=kind,prediction=prediction,
                    answer=fa.ANSWER_CRITERIA[kind][prediction],evaluated=True,status='EVALUATED',
                    probability_scope='unconditional' if kind=='BOOLEAN' else 'given_BOOLEAN_Yes',
                    probabilities=dict(zip(fa.ANSWER_CRITERIA[kind],map(float,probabilities))))
            sync(module);timings['head_ms']+=1000*(time.perf_counter()-now)
        infer(chosen,('BOOLEAN',))
        positive=[center for center in chosen if decoded[(center,'BOOLEAN')]['prediction']==1]
        if positive:infer(positive,('CHOICE','ORDINAL'))
        answers=[]
        for center in chosen:
            for kind in fa.KINDS:
                answers.append(decoded.get((center,kind),dict(center_sensor=center,answer_type=kind,
                    prediction=None,answer=None,probabilities=None,evaluated=False,status='NOT_APPLICABLE',
                    reason='BOOLEAN_No')))
        reply.update(status='OK',reason=None,answers=answers,reference_count=record['reference_runs'],selected_sensors=len(chosen),run_id=run_id,
            evaluated_answers=sum(a['evaluated'] for a in answers),conditional_answering=True)
        return finish()


def evaluate(runtime,*,steps=None,boundary_challenge=False,protocol='controlled'):
    from . import fast_experiments as fast
    selected=set(runtime.encoders if steps is None else steps)
    if not selected<=set(runtime.encoders):raise ValueError('Requested Step was not trained')
    if protocol not in ('controlled','historical'):raise ValueError('Choose controlled or historical evaluation explicitly')
    if boundary_challenge and protocol!='controlled':raise ValueError('Boundary protocol and historical comparison cannot be mixed')
    if protocol=='controlled':requested,intervention_coverage=controlled_cases(runtime,'test',runtime.config,runtime.references)
    else:
        requested=evaluation_cases(runtime,'test',runtime.config);intervention_coverage=[]
        # New ordinal outputs express measured excess, not historical amplitude
        # or lag stages. Retain the old waveforms/family labels, never mix units.
        requested=[dict(c,targets=np.array(c['targets'],copy=True)) for c in requested]
        for c in requested:c['targets'][c['targets'][:,0]>0,2]=-1
    requested=[c for c in requested if c['step'] in selected]
    intervention_coverage=[c for c in intervention_coverage if c['step'] in selected]
    cases,audit=supported_cases(runtime,requested,runtime.references);rows=[];all_answers=[];session=Session(runtime,center_batch_size=runtime.config.center_batch_size)
    typed=[]
    for case in cases:
        reply=session.process_window(current=case_current(case),step=case['step'],centers=case['centers'],
            sensor_columns=case['sensor_columns'],recipe_key=case['recipe_key'],loop_key=case['loop_key'],
            offset_samples=case['offset_samples'],lineage_id=case['lineage_id'],run_id=case['run_id'],standardized=True)
        if reply['status']!='OK':raise RuntimeError('Supported Test case unexpectedly withheld')
        for answer in reply['answers']:
            i=case['centers'].index(answer['center_sensor']);j=fa.KINDS.index(answer['answer_type'])
            label=int(case['targets'][i,j]);row=dict(answer,scored=label>=0 and (answer['evaluated'] or label>0),step=case['step'],split='test',
                pair_id=case['pair_id'],run_id=case['run_id'],injected=case['injected'],
                center_is_perturbed=bool(case['injected'] and np.any(case_current(case)[:,case['sensor_columns'].index(answer['center_sensor'])]!=case['original_current'][:,case['sensor_columns'].index(answer['center_sensor'])])),
                target_known=label>=0,target_label=label,intervention_family=case['kind'],
                requested_stage=case.get('injection',{}).get('realized_stage'),
                strength_unit=case.get('injection',{}).get('strength_unit'),realized_excess=case.get('injection',{}).get('maximum_normalized_excess'))
            if protocol=='historical' and answer['answer_type']=='ORDINAL':
                label=-1;row.update(scored=False,target_known=False,target_label=-1)
            all_answers.append(row)
            if answer['answer_type']!='BOOLEAN' and label>0:
                typed.append(dict(answer_type=answer['answer_type'],detected=answer['evaluated'],
                    correct=answer['prediction']==label))
            if label>=0 and (answer['evaluated'] or label>0):
                # A missed BOOLEAN gate is an end-to-end typed miss. The 0 here
                # is a scoring sentinel, never a displayed head probability.
                rows.append(dict(row,label=label,prediction=0 if answer['prediction'] is None else answer['prediction'],
                    gate_miss=not answer['evaluated']))
    coverage_rows=[]
    for item in intervention_coverage:
        matching=[a for a in all_answers if a['step']==item['step'] and a['center_sensor']==item['center_sensor']
            and a['intervention_family']==item['family'] and a['requested_stage']==item['level'] and a['center_is_perturbed']]
        coverage_rows.append(dict(item,supported=sum(a['answer_type']=='BOOLEAN' for a in matching),
            **{'evaluated_'+kind:sum(a['answer_type']==kind and a['evaluated'] for a in matching) for kind in fa.KINDS},
            **{'scored_'+kind:sum(a['answer_type']==kind and a['scored'] for a in matching) for kind in fa.KINDS}))
    intervention_table=pd.DataFrame(coverage_rows)
    answer_table=pd.DataFrame([dict(answer_type=kind,requested_questions=sum(len(c['centers']) for c in requested),
        supported_questions=sum(a['answer_type']==kind for a in all_answers),
        unsupported_questions=sum(len(c['centers']) for c in requested)-sum(a['answer_type']==kind for a in all_answers),
        evaluated_questions=sum(a['answer_type']==kind and a['evaluated'] for a in all_answers),
        scored_questions=sum(r['answer_type']==kind for r in rows),known_positive_questions=sum(a['answer_type']==kind and a['target_label']>0 for a in all_answers),
        unknown_truth_questions=sum(a['answer_type']==kind and not a['target_known'] for a in all_answers)) for kind in fa.KINDS])
    if not rows:return ExperimentReport('Detailed Test NOT_EVALUATED',pd.DataFrame(),diagnostics=dict(
        support_audit=pd.DataFrame(audit),boolean_populations=pd.DataFrame(),
        answer_coverage=answer_table,intervention_coverage=intervention_table))
    report=fast._report('Detailed MIN/MAX envelope held-out Test',rows,all_answers=all_answers,
        metadata=dict(domain=runtime.domain,model_schema=SCHEMA,training_recipe=RECIPE,input_mode=INPUT_MODE,
            checkpoint=str(runtime.module.FUSION_ANSWER_PATH),target_policy=TARGET_POLICY,metric_target=TARGET_POLICY,
            selection_policy=(runtime.module.FUSION_ANSWER_TRAINING or {}).get('selection_policy',SELECTION_POLICY),
            real_fault_truth=False,historical_paired_scores_comparable=False,conditional_answering=True,
            ordinal_policy=ORDINAL_POLICY if protocol=='controlled' else 'historical_ordinal_unscored_incompatible_units',
            evaluation_protocol='known_boundary_excess_all_splits_v6' if protocol=='controlled' else 'preserved_original_relative_interventions'))
    b=report.details.loc[report.details.answer_type.eq('BOOLEAN')].copy()
    b['population']=np.where(~b.injected,'nominal_original',np.where(b.center_is_perturbed,'injected_sensor','unchanged_sensor'))
    report.diagnostics['boolean_populations']=pd.DataFrame([dict(population=name,questions=len(g),
        true_negatives=int((g.label.eq(0)&g.prediction.eq(0)).sum()),false_positives=int((g.label.eq(0)&g.prediction.eq(1)).sum()),
        false_negatives=int((g.label.eq(1)&g.prediction.eq(0)).sum()),true_positives=int((g.label.eq(1)&g.prediction.eq(1)).sum())) for name,g in b.groupby('population',sort=False)])
    report.diagnostics['support_audit']=pd.DataFrame(audit)
    report.diagnostics['intervention_coverage']=intervention_table
    report.diagnostics['answer_coverage']=answer_table
    report.diagnostics['conditional_typed_metrics']=pd.DataFrame([dict(answer_type=kind,
        known_changed_questions=len(group),gate_detected=sum(v['detected'] for v in group),
        correct=sum(v['correct'] for v in group),
        accuracy_given_detected=(sum(v['correct'] for v in group)/sum(v['detected'] for v in group)) if any(v['detected'] for v in group) else np.nan,
        end_to_end_accuracy=sum(v['correct'] for v in group)/len(group))
        for kind in ('CHOICE','ORDINAL') if (group:=[v for v in typed if v['answer_type']==kind])])
    nominal=b.loc[~b.injected]
    report.diagnostics['heldout_normal_envelope']=pd.DataFrame([dict(nominal_original_questions=len(nominal),
        outside_TRAIN_envelope=int(nominal.label.sum()),predicted_yes=int(nominal.prediction.sum()),
        equipment_fault_truth_available=False,target_policy=TARGET_POLICY)])
    return report


def benchmark(runtime,*,steps=None,repeats=3,warmup=1,progress=print):
    from .online_inference import latency_tables
    if repeats<1 or warmup<0:raise ValueError('Positive repeats and nonnegative warmup required')
    selected=set(runtime.encoders if steps is None else steps)
    examples=[c for c in evaluation_cases(runtime,'test',runtime.config) if not c['injected'] and c['step'] in selected]
    examples,_=supported_cases(runtime,examples,runtime.references);per_step={c['step']:c for c in examples};requests=[]
    session=Session(runtime,center_batch_size=runtime.config.center_batch_size)
    for step,c in per_step.items():
        for i in range(warmup+repeats):
            reply=session.process_window(current=c['current'],step=step,recipe_key=c['recipe_key'],loop_key=c['loop_key'],
                offset_samples=c['offset_samples'],lineage_id=c['lineage_id'],standardized=True)
            if i>=warmup:requests.append(dict(request_id=f'{step}/{i}',step=step,status=reply['status'],reason=reply['reason'],
                answers=len(reply['answers']),evaluated_answers=reply.get('evaluated_answers',0),
                conditional_answering=True,selected_sensors=len(c['centers']),**reply['timings']))
        if progress:progress('Detailed complete-window timing: '+step,flush=True)
    latency,by_step=latency_tables(requests)
    return ExperimentReport('Detailed domain complete-window timing',pd.DataFrame(),diagnostics=dict(
        requests=pd.DataFrame(requests),request_latency=latency,request_latency_by_step=by_step))
