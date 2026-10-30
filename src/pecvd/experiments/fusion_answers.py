"""Fixed questions + selected numeric center + real Fusion -> learned answers.

Controlled paired training truth is not equipment fault truth. No numerical
detector, component assignment or post-hoc alert threshold decides an answer.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from pecvd.answer_display import QUESTIONS, FUSION_ANSWER_CRITERIA as ANSWER_CRITERIA

SYSTEM = 'fusion_answers'
SCHEMA = 'gevis.fixed_question_contextual_fusion.v3'
TRAINING_RECIPE = 'bidirectional_variable_span_delay_v3'
ORDINAL_TARGET = 'local amplitude stage (.5,1,2,4); Delay stage is native lag samples (1,2,3,4); not physical severity'
KINDS = ('BOOLEAN','CHOICE','ORDINAL')
AMPLITUDE_TARGET = {'level_shift':1,'variance':2,'drift':3,'spike':4}
TYPE_TARGET = dict(AMPLITUDE_TARGET, delay=5)


@dataclass(frozen=True)
class FusionConfig:
    seed: int = 42
    max_windows_per_step: int = 32
    epochs: int = 40
    batch_size: int = 64
    width: int = 96
    learning_rate: float = .001
    max_seconds: float = 180.
    patience: int = 8


class ContextReadout(nn.Module):
    def __init__(self, dimension, width=96):
        super().__init__()
        self.register_buffer('feature_mean',torch.zeros(dimension))
        self.register_buffer('feature_scale',torch.ones(dimension))
        self.shared = nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,width),nn.GELU())
        self.answers = nn.ModuleDict({kind:nn.Linear(width,len(ANSWER_CRITERIA[kind])) for kind in KINDS})
        for kind in KINDS:self.register_buffer('allowed_'+kind,torch.ones(len(ANSWER_CRITERIA[kind]),dtype=torch.bool))

    def forward(self, features, kind):
        features=(features-self.feature_mean)/self.feature_scale
        logits=self.answers[kind](self.shared(features))
        return logits.masked_fill(~getattr(self,'allowed_'+kind),-1e9)


@torch.no_grad()
def fusion_features(module, current, reference, step, centers, position=0., *, with_evidence=False):
    """One frozen numeric encoding, actual question Fusion for each UI center.

    Sensor names occur only in numeric descriptions; the question strings are
    the unchanged public English questions. No labels enter this function.
    """
    model=module.GEVIS_MODEL.to(module.DEVICE).eval()
    encoder,meta=module.STEP_ENCODERS[step];names=list(meta['sensor_columns'])
    if not centers or any(c not in names for c in centers):raise ValueError('Explicit known analysis center required')
    current=np.asarray(current,dtype=np.float32);reference=np.asarray(reference,dtype=np.float32)
    if current.ndim!=2 or current.shape!=reference.shape or current.shape[1]!=len(names):
        raise ValueError('Matching standardized current/reference sensor arrays required')
    if not np.isfinite(current).all() or not np.isfinite(reference).all():raise ValueError('Finite traces required')
    encoder.to(module.DEVICE).eval()
    a=torch.as_tensor(current[None],device=module.DEVICE)
    b=torch.as_tensor(reference[None],device=module.DEVICE)
    state=module.numeric_state(encoder,a,b)
    if state.shape[1]%len(names):raise ValueError('Complete sensor/patch grid required')
    patches=state.shape[1]//len(names)
    descriptions=module.numeric_token_descriptions(step,meta,patches,1)[0]
    from .questions import center_descriptions
    tasks=[(center,kind) for center in centers for kind in KINDS]
    contexts=[center_descriptions(descriptions,names,patches,center) for center,kind in tasks]
    texts=[QUESTIONS[kind] for center,kind in tasks]
    output=model(state.expand(len(tasks),-1,-1),texts,[k for c,k in tasks],
        [list(ANSWER_CRITERIA[k]) for c,k in tasks],contexts,return_fusion=True)
    fused,query,attention=output
    keys=model.numeric_key_norm(model.numeric(model.numeric_input_norm(state)))
    grid=keys.reshape(1,len(names),patches,-1)[0]
    selected=torch.stack([grid[names.index(center)].mean(0) for center,kind in tasks])
    weights=attention.reshape(len(tasks),len(names),patches)
    local=torch.stack([(grid[names.index(center)]*weights[i,names.index(center),:,None]).sum(0)
        /weights[i,names.index(center)].sum().clamp_min(1e-8) for i,(center,kind) in enumerate(tasks)])
    coordinate=torch.tensor([[float(position),1-float(position)]]*len(tasks),device=module.DEVICE)
    features=torch.cat([fused,query,selected,local,coordinate],-1).detach().cpu()
    if not torch.isfinite(features).all():raise ValueError('Nonfinite Fusion features')
    if with_evidence:
        from .reconstruction_evidence import evidence
        return features,tasks,evidence(state,names,tasks)
    return features,tasks


def sampling_config(config):
    # Cache selection only: no numerical policy is fitted or queried here.
    from .window_policy import WindowConfig
    return WindowConfig(max_windows_per_step=config.max_windows_per_step,seed=config.seed)


def delay_trace(base, sensor, lag):
    """Causal shift in native samples; hold first sample, never wrap or add noise."""
    base=np.asarray(base,dtype=float)
    if base.ndim!=2 or not np.isfinite(base).all() or not 0<=sensor<base.shape[1] or not 1<=lag<len(base) or int(lag)!=lag:
        raise ValueError('Finite array, known sensor and integer native lag required')
    observed=base.copy()
    observed[:,sensor]=base[np.maximum(0,np.arange(len(base))-int(lag)),sensor]
    return observed,observed!=base


def paired_variants(base, sensor, index, seed):
    """Original controls plus signed, variable-span interventions; copy-only.

    Stages (.5, 1, 2, 4) describe the requested local perturbation amplitude,
    including variance noise scale. They do not measure window-integrated impact.
    A sign reversal of added noise is NOT a decrease in variance.
    """
    from .injection import Injection, inject
    base=np.asarray(base,dtype=float)
    if base.ndim!=2 or len(base)<4 or not np.isfinite(base).all():
        raise ValueError('Finite time/sensor window with at least four samples required')
    if not 0<=sensor<base.shape[1]:raise ValueError('Known intervention sensor required')
    yield dict(tag='equal',kind='equal',observed=base.copy(),mask=np.zeros_like(base,dtype=bool),
        level=0,strength=0.,direction=0,start=0,end=len(base),augmentation='equal')
    rng=np.random.default_rng(seed+index*7919)
    for k,kind in enumerate(AMPLITUDE_TARGET):
        specs=[(1,index%4+1,len(base)//4,max(len(base)//4+2,3*len(base)//4),'standard')]
        # Every directional family sees both signs. Noise gets two independent
        # spans, never an incorrectly named negative-variance target.
        for j in range(2):
            width=max(2,min(len(base),round(len(base)*rng.uniform(.125,.875))))
            start=int(rng.integers(0,len(base)-width+1))
            specs.append((1 if kind=='variance' or j==0 else -1,
                int(rng.integers(1,5)),start,start+width,'variable_span'))
        for j,(direction,level,start,end,augmentation) in enumerate(specs):
            strength=(.5,1.,2.,4.)[level-1]
            observed,_=inject(base,Injection(kind,sensor,start,end,strength,
                seed+index*31+k*7+j),np.ones(base.shape[1]))
            if direction<0:observed=2*base-observed
            mask=observed!=base
            if mask.any():
                yield dict(tag=f'{kind}/{j}',kind=kind,observed=observed,mask=mask,
                    level=level,strength=strength,direction=direction,start=start,end=end,
                    augmentation=augmentation)
    eligible=np.flatnonzero(np.ptp(base,axis=0)>0)
    target=sensor if sensor in eligible or not len(eligible) else int(eligible[index%len(eligible)])
    maximum=min(4,len(base)-1)
    for j in range(2):
        lag=1+(index+j*max(1,maximum//2))%maximum
        observed,mask=delay_trace(base,target,lag)
        yield dict(tag=f'delay/{j}',kind='delay',observed=observed,mask=mask,sensor=target,
            level=lag,strength=float(lag),lag_samples=lag,strength_unit='native_samples',
            direction=0,start=0,end=len(base),augmentation='sample_delay')


def controlled_examples(module, split, config=FusionConfig(), steps=None):
    """Known equality controls and actual perturbations of held-out source waves.

    The reference is the SAME source window, not an unlabelled other-run window.
    A different selected center on the identical perturbed input has a different
    target. Real cross-run normality is never inferred from 'not injected'.
    """
    from .discovery_calibration import cached_comparisons
    if split not in ('train','validation','test'):raise ValueError('Known source split required')
    split_seed=config.seed+{'train':0,'validation':100003,'test':200003}[split]
    examples=[]
    for step in (list(module.STEP_ENCODERS) if steps is None else steps):
        names=list(module.STEP_ENCODERS[step][1]['sensor_columns'])
        print(f'Fusion answer features: {split} / {step}',flush=True)
        for i,row in enumerate(cached_comparisons(module.ENCODER_ROOT,step,split=split,config=sampling_config(config))):
            if row.get('split',split)!=split:raise ValueError('Cache row belongs to another split')
            base=row['current'];sid=i%len(names)
            for variant in paired_variants(base,sid,i,split_seed):
                kind=variant['kind'];observed=variant['observed'];mask=variant['mask']
                target=variant.get('sensor',sid)
                centers=[names[target]]+([names[(target+1)%len(names)]] if len(names)>1 else [])
                features,tasks,reconstruction=fusion_features(module,observed,base,step,centers,
                    float(row['position'])/9,with_evidence=True)
                pair_id=f'{step}/{row["run_id"]}/{row.get("cache_index",i)}/{variant["tag"]}'
                for feature,(center,answer_type),error in zip(features,tasks,reconstruction):
                    changed=bool(mask[:,names.index(center)].any())
                    label=(int(changed) if answer_type=='BOOLEAN' else
                        TYPE_TARGET[kind] if changed and answer_type=='CHOICE' else
                        variant['level'] if changed and answer_type=='ORDINAL' else 0)
                    examples.append(dict(features=feature,answer_type=answer_type,label=label,
                        run_id=str(row['run_id']),step=step,split=split,center_sensor=center,
                        pair_id=pair_id,kind=kind,injected=bool(mask.any()),
                        direction=variant['direction'],strength=variant['strength'],
                        strength_unit=variant.get('strength_unit','train_std'),
                        lag_samples=variant.get('lag_samples',0),
                        intervention_start=variant['start'],intervention_end=variant['end'],
                        augmentation=variant['augmentation'],
                        center_is_perturbed=changed,question=QUESTIONS[answer_type],**error))
        module.STEP_ENCODERS[step][0].cpu()
    if not examples:raise ValueError('No source windows available for Fusion answer task')
    return examples


def fit_readout(train, validation, config=FusionConfig(), device='cpu'):
    if min(config.epochs,config.width,config.batch_size,config.patience)<1 or config.learning_rate<=0 or config.max_seconds<=0:
        raise ValueError('Positive bounded answer training parameters required')
    if not train or not validation:raise ValueError('Nonempty TRAIN/Validation examples required')
    if any(e['split']!='train' for e in train) or any(e['split']!='validation' for e in validation):
        raise ValueError('Only TRAIN optimization and Validation selection are allowed')
    if {e['run_id'] for e in train}&{e['run_id'] for e in validation}:raise ValueError('TRAIN/Validation run overlap')
    torch.manual_seed(config.seed)
    head=ContextReadout(len(train[0]['features']),config.width).to(device)
    features=torch.stack([e['features'] for e in train]).to(device)
    with torch.no_grad():
        head.feature_mean.copy_(features.mean(0));scale=features.std(0,unbiased=False)
        head.feature_scale.copy_(torch.where(scale>0,scale,torch.ones_like(scale)))
    groups={};weights={}
    for kind in KINDS:
        group=[e for e in train if e['answer_type']==kind]
        held=[e for e in validation if e['answer_type']==kind]
        if not group or not held:raise ValueError('Each split needs all three fixed questions')
        y=torch.tensor([e['label'] for e in group],device=device)
        counts=torch.bincount(y,minlength=len(ANSWER_CRITERIA[kind]));allowed=counts>0
        if allowed.sum()<2:raise ValueError('At least two distinct conditional answers per question required')
        getattr(head,'allowed_'+kind).copy_(allowed)
        if any(not allowed[e['label']] for e in held):raise ValueError('Validation has an untrained answer class; increase TRAIN sample coverage')
        weights[kind]=torch.where(allowed,counts.sum()/counts.clamp_min(1)/allowed.sum(),torch.zeros_like(counts)).float()
        groups[kind]=(torch.stack([e['features'] for e in group]).to(device),y,
            torch.stack([e['features'] for e in held]).to(device),torch.tensor([e['label'] for e in held],device=device))
    optimizer=torch.optim.AdamW(head.parameters(),lr=config.learning_rate,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(config.seed);best=float('inf');best_state=None
    stale=0;updates=0;history=[];started=time.perf_counter()
    for epoch in range(config.epochs):
        head.train();losses=[]
        for kind,(x,y,vx,vy) in groups.items():
            order=torch.randperm(len(y),generator=generator)
            for offset in range(0,len(order),config.batch_size):
                if time.perf_counter()-started>=config.max_seconds:break
                ix=order[offset:offset+config.batch_size].to(device)
                loss=F.cross_entropy(head(x[ix],kind),y[ix],weight=weights[kind])
                if not torch.isfinite(loss):raise ValueError('Nonfinite answer loss; no checkpoint')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.)
                optimizer.step();updates+=1;losses.append(float(loss.detach()))
        if not losses:break
        head.eval()
        with torch.no_grad():
            value=float(torch.stack([F.cross_entropy(head(vx,kind),vy,weight=weights[kind])
                for kind,(x,y,vx,vy) in groups.items()]).mean())
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=value))
        print(f'Fusion readout epoch {epoch+1}/{config.epochs}: validation loss {value:.4f}',flush=True)
        if value<best:
            best=value;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};stale=0
        else:stale+=1
        if stale>=config.patience or time.perf_counter()-started>=config.max_seconds:break
    if updates<1 or best_state is None:raise RuntimeError('No trained answer weights; refusing random-head export')
    head.load_state_dict(best_state);head.requires_grad_(False).eval()
    return head,dict(optimizer_updates=updates,history=history,validation_loss=best,
        trainable_parameters=sum(p.numel() for p in head.parameters()),fit_seconds=time.perf_counter()-started,
        train_runs=sorted({e['run_id'] for e in train}),validation_runs=sorted({e['run_id'] for e in validation}),
        trained_steps=sorted({e['step'] for e in train}),
        training_recipe=TRAINING_RECIPE,ordinal_target=ORDINAL_TARGET,
        target_contract='selected-center known paired interventions; not equipment fault truth')


def identity(module, config=FusionConfig()):
    from .discovery_calibration import digest,numeric_runtime_source_digest
    from pecvd import gevis_backbone,gevis_fast,temporal_bridge,relational_features
    from . import reconstruction_evidence
    manifest=Path(module.ENCODER_ROOT)/'encoder_manifest.json'
    doc=json.loads(manifest.read_text(encoding='utf-8'))
    notebook=Path(module.ROOT)/'notebooks/STEP_3.1_run_gevis_inference.ipynb'
    executable=json.loads(notebook.read_text(encoding='utf-8'))
    input_source=''.join(''.join(c['source']) for c in executable['cells'] if c.get('id') in ('03-types','03-function'))
    return dict(schema=SCHEMA,config=asdict(config),questions={k:QUESTIONS[k] for k in KINDS},
        criteria={k:list(ANSWER_CRITERIA[k]) for k in KINDS},backbone_sha256=digest(module.BACKBONE_PATH),
        manifest_sha256=digest(manifest),encoder_sha256={s:digest(Path(module.ENCODER_ROOT)/v['path']) for s,v in doc['step_checkpoints'].items()},
        cache_sha256={s:digest(Path(module.ENCODER_ROOT)/'windows'/f'{s}.npz') for s in doc['step_checkpoints']},
        numeric_runtime_sha256=numeric_runtime_source_digest(module.ROOT),
        input_source_sha256=hashlib.sha256(input_source.encode('utf-8')).hexdigest(),
        dependency_sha256={Path(m.__file__).name:digest(m.__file__)
            for m in (gevis_backbone,gevis_fast,temporal_bridge,relational_features,reconstruction_evidence)},
        source_sha256=digest(__file__))


def checkpoint_path(module, config=FusionConfig()):
    stamp=hashlib.sha256(json.dumps(identity(module,config),sort_keys=True).encode()).hexdigest()[:20]
    return Path(module.BACKBONE_PATH).parent/f'fusion_answers_{stamp}.pt'


def attach(module, config=FusionConfig()):
    module.FUSION_ANSWER_HEAD=None
    path=checkpoint_path(module,config);module.FUSION_ANSWER_PATH=path
    if not path.is_file():return
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!=SCHEMA or payload.get('identity')!=identity(module,config):
        raise ValueError('Fusion answer checkpoint identity mismatch; no rule fallback')
    if payload.get('training',{}).get('optimizer_updates',0)<1:raise ValueError('Missing actual answer-head training evidence')
    head=ContextReadout(payload['dimension'],config.width)
    head.load_state_dict(payload['state'],strict=True);head.requires_grad_(False).eval()
    module.FUSION_ANSWER_HEAD=head.to(module.DEVICE)
    module.FUSION_ANSWER_TRAINING=payload['training']


def require_head(module):
    head=getattr(module,'FUSION_ANSWER_HEAD',None)
    if head is None:raise RuntimeError('Fusion answer head missing. Run STEP 3 fusion-answer-train; no q99/GMM fallback.')
    return head.to(module.DEVICE).eval()


def prepare(runtime, config=FusionConfig()):
    module=runtime.module;attach(module,config)
    if module.FUSION_ANSWER_HEAD is not None:return module.FUSION_ANSWER_TRAINING
    before=identity(module,config);started=time.perf_counter()
    train=controlled_examples(module,'train',config);validation=controlled_examples(module,'validation',config)
    head,training=fit_readout(train,validation,config,module.DEVICE)
    training['augmentation_coverage']=[dict(split=split,kind=kind,direction=direction,
        augmentation=augmentation,sensor_comparisons=len(group))
        for split,examples in (('train',train),('validation',validation))
        for (kind,direction,augmentation),group in pd.DataFrame([
            dict(kind=e.get('kind','synthetic'),direction=e.get('direction',0),
                 augmentation=e.get('augmentation','fixture'))
            for e in examples if e['answer_type']=='BOOLEAN']).groupby(
                ['kind','direction','augmentation'],sort=False)]
    training['preparation_seconds']=time.perf_counter()-started-training['fit_seconds']
    training['fit_packages']={'torch':torch.__version__,'numpy':np.__version__}
    if identity(module,config)!=before:raise ValueError('Sources or frozen artifacts changed during answer training')
    path=checkpoint_path(module,config);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema=SCHEMA,identity=before,dimension=head.feature_mean.numel(),
        state={k:v.detach().cpu() for k,v in head.state_dict().items()},training=training)
    with path.open('xb') as stream:torch.save(payload,stream)
    module.FUSION_ANSWER_HEAD=head;module.FUSION_ANSWER_PATH=path;module.FUSION_ANSWER_TRAINING=training
    return training


@torch.no_grad()
def predict_arrays(module,current,reference,step,center_sensor,position=0.):
    head=require_head(module)
    training=getattr(module,'FUSION_ANSWER_TRAINING',None)
    if training is not None and step not in training.get('trained_steps',[]):
        raise RuntimeError('This Step has no trained Fusion answer examples; no rule fallback')
    features,tasks=fusion_features(module,current,reference,step,[center_sensor],position)
    answers=[]
    for feature,(center,kind) in zip(features,tasks):
        probabilities=head(feature[None].to(module.DEVICE),kind).softmax(-1)[0].cpu()
        index=int(probabilities.argmax());options=ANSWER_CRITERIA[kind]
        answers.append(dict(answer_type=kind,question=QUESTIONS[kind],answer=options[index],
            candidate_answer=options[index],answer_index=index,accepted=True,center_sensor=center,
            probabilities=dict(zip(options,map(float,probabilities))),
            untrained_options=[option for i,option in enumerate(options) if not bool(getattr(head,'allowed_'+kind)[i])],
            decision_source=SCHEMA,score_meaning='learned class softmax, not calibrated equipment fault probability',
            warnings=['synthetic comparison training; cross-run fault quality unverified']))
    return answers


def review(module,current_rows,reference_rows,step,reference_info,*,center_sensor):
    if center_sensor is None:raise ValueError('UI analysis center required; never infer it from the question')
    require_head(module)
    meta=module.STEP_ENCODERS[step][1]
    for rows in (current_rows,reference_rows):
        if len(rows)!=meta['window_length']:
            raise ValueError('Provide one exact model window or use review_selected_step for the complete timeline')
        if module.RUN_COLUMN in rows and rows[module.RUN_COLUMN].nunique()!=1:raise ValueError('Do not mix process runs')
        if module.STEP_COLUMN in rows and set(rows[module.STEP_COLUMN])!={step}:raise ValueError('Do not mix Steps')
        if 'step_number' in rows and rows.step_number.nunique()!=1:raise ValueError('Do not mix Step occurrences')
    a,ac=module.prepare_window(current_rows,meta);b,bc=module.prepare_window(reference_rows,meta)
    answers=predict_arrays(module,a.cpu().numpy()[0],b.cpu().numpy()[0],step,center_sensor,ac.get('position_ratio',0.))
    return dict(contract=SCHEMA,answers=answers,accepted=True,changed=answers[0]['answer_index']==1,
        center_sensor=center_sensor,step=step,events=[],winner=None,ranked_evidence=[],
        change_decision_source='question-conditioned Fusion + trained contextual readout argmax',
        warnings=['no trained time-boundary head in this answer experiment'],
        reference=reference_info,criterion_coverage=1.)


def answer_questions(module,current,reference,step,questions,reference_info,*,center_sensor):
    for question in questions:
        if question.answer_type not in KINDS or question.text!=QUESTIONS[question.answer_type]:
            raise ValueError('This experiment supports the unchanged fixed questions only')
        if list(question.criteria)!=list(ANSWER_CRITERIA[question.answer_type]):raise ValueError('Trained option order required')
    result=review(module,current,reference,step,reference_info,center_sensor=center_sensor)
    lookup={a['answer_type']:a for a in result['answers']}
    return [lookup[q.answer_type] for q in questions]


def predict_original(runtime,*,steps=None,pairs_per_step=2,seed=42,**unused):
    from .discovery_calibration import load_discovery_pairs
    from .reporting import ExperimentReport
    module=runtime.module;require_head(module);rows=[]
    for step in (list(runtime.encoders) if steps is None else steps):
        names=list(runtime.encoders[step][1]['sensor_columns'])
        for i,(current,reference) in enumerate(load_discovery_pairs(runtime,step,split='test',limit=pairs_per_step,seed=seed)):
            center=names[i%len(names)]
            result=review(module,current,reference,step,None,center_sensor=center)
            rows.extend(dict(step=step,pair_index=i,center_sensor=center,**a) for a in result['answers'])
    details=pd.DataFrame(rows)
    return ExperimentReport('Original cross-run Fusion answers without independent truth',
        pd.DataFrame([dict(questions=len(details),accuracy_available=False,metric_target='unlabelled original cross-run predictions')]),
        details=details,metadata=dict(model_evaluated=True,decision_system=SYSTEM,rule_fallback=False,
            original_cross_run_fault_accuracy_available=False))


def evaluate(runtime, *, split='test', steps=None, config=FusionConfig(), **unused):
    if split not in ('validation','test'):raise ValueError('Held-out answer evaluation only')
    from .evaluation import binary_metrics
    from .reporting import ExperimentReport
    module=runtime.module;head=require_head(module)
    examples=controlled_examples(module,split,config,steps);rows=[]
    with torch.no_grad():
        for kind in KINDS:
            group=[e for e in examples if e['answer_type']==kind]
            probabilities=head(torch.stack([e['features'] for e in group]).to(module.DEVICE),kind).softmax(-1).cpu()
            for e,p in zip(group,probabilities):
                rows.append({k:v for k,v in e.items() if k!='features'}|dict(prediction=int(p.argmax()),correct=int(p.argmax())==e['label']))
    details=pd.DataFrame(rows);summary=[]
    for kind,group in details.groupby('answer_type',sort=False):
        row=dict(answer_type=kind,questions=len(group),answer_accuracy=float(group.correct.mean()),
            majority_baseline_accuracy=float(group.label.value_counts().max()/len(group)),
            metric_target='known paired perturbation answer; not actual equipment fault')
        if kind=='BOOLEAN':row.update(binary_metrics(group.label,group.prediction))
        summary.append(row)
    boolean=details[details.answer_type.eq('BOOLEAN')]
    swapped=[g for _,g in boolean.groupby('pair_id',sort=False) if g.label.nunique()>1]
    diagnostics={'center_swap':pd.DataFrame([dict(pairs=len(swapped),
        joint_answer_accuracy=float(np.mean([g.correct.all() for g in swapped])) if swapped else np.nan)]),
        'off_center':pd.DataFrame([dict(cases=len(g),false_change_rate=float(g.prediction.eq(1).mean()))
            for _,g in boolean[boolean.injected & ~boolean.center_is_perturbed].groupby('answer_type')]),
        'confusion':pd.crosstab([details.answer_type,details.label],details.prediction).reset_index()}
    diagnostics['intervention_metrics']=details[details.center_is_perturbed].groupby(
        ['augmentation','direction','kind','answer_type'],sort=False).agg(
            questions=('correct','size'),answer_accuracy=('correct','mean')).reset_index()
    from .reconstruction_evidence import summary as reconstruction_summary
    diagnostics['reconstruction_metrics']=reconstruction_summary(details)
    delay=details[details.kind.eq('delay') & details.center_is_perturbed]
    diagnostics['delay_metrics']=delay.groupby(['lag_samples','answer_type'],sort=False).agg(
        questions=('correct','size'),answer_accuracy=('correct','mean')).reset_index()
    by_step=details.groupby(['step','answer_type'],sort=False).agg(questions=('correct','size'),answer_accuracy=('correct','mean')).reset_index()
    return ExperimentReport('Fixed-question Fusion answers',pd.DataFrame(summary),by_step,details,
        details[~details.correct].copy(),diagnostics=diagnostics,
        metadata=dict(status='COMPLETED',model_evaluated=True,split=split,decision_system=SYSTEM,
            decision_source=SCHEMA,rule_fallback=False,test_used_for_fitting=False,
            training_recipe=TRAINING_RECIPE,ordinal_target=ORDINAL_TARGET,
            evaluation_role='development benchmark; previously inspected Test is not confirmatory',
            evaluation_domain='held-out source windows, exact equality controls and paired perturbations',
            original_cross_run_fault_accuracy_available=False))


def review_selected_step(module, frame, reference_frame, process_key, step, center_sensor, *, step_number=None):
    """Select one run and Step occurrence; cover its timeline, never last-window only.

    Return learned answers per window. No hand-written whole-Step alert rule.
    The reference frame must already contain one compatible reference Step.
    """
    from pecvd.gevis_ui import derive_step_label
    frame=derive_step_label(frame)
    rows=frame.loc[frame[module.RUN_COLUMN].eq(process_key)&frame[module.STEP_COLUMN].eq(step)].copy()
    if step_number is not None and 'step_number' in rows:rows=rows[rows.step_number.eq(step_number)]
    if rows.empty:raise ValueError('Selected process/Step has no samples')
    if 'step_number' in rows and rows.step_number.nunique()!=1:
        raise ValueError('Select one Step occurrence explicitly; repeated segments cannot be concatenated')
    reference_frame=derive_step_label(reference_frame)
    if reference_frame[module.RUN_COLUMN].nunique()!=1 or set(reference_frame[module.STEP_COLUMN])!={step}:
        raise ValueError('Provide exactly one compatible reference run/Step')
    if 'step_number' in reference_frame and reference_frame.step_number.nunique()!=1:raise ValueError('One reference Step occurrence required')
    rows=rows.sort_values(module.ORDER_COLUMN);ref=reference_frame.sort_values(module.ORDER_COLUMN)
    width=module.STEP_ENCODERS[step][1]['window_length']
    if min(len(rows),len(ref))<width:raise ValueError('Selected Step shorter than trained model window')
    starts=list(range(0,len(rows)-width+1,width));last=len(rows)-width
    if starts[-1]!=last:starts.append(last)
    results=[]
    for start in starts:
        ratio=start/max(1,len(rows)-width);rstart=round(ratio*(len(ref)-width))
        result=review(module,rows.iloc[start:start+width],ref.iloc[rstart:rstart+width],step,None,center_sensor=center_sensor)
        results.append(dict(start_row=start,end_row=start+width,position_ratio=ratio,answers=result['answers']))
    return dict(process_key=process_key,step=step,center_sensor=center_sensor,windows=results,
        whole_step_decision=None,meaning='learned window answers across selected Step; no rule aggregation')
