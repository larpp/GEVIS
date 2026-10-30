"""Question/center-conditioned learned decision candidate, independent of old gates.

TRAIN interventions supervise a review task, not verified equipment faults.
No input function accepts intervention truth. Adoption requires a separate
Validation comparison; existing deployment remains the comparator.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, math, os, time
import numpy as np
import torch
from .optimization_models import (paired_patch_features, continuous_difference_features,
                                  PairedEncoderTrial, PairedPatchDetector)
from .optimization_heads import (GroundedOutputTrial, GroundedOutputHeads, TYPE_LABELS,
                                 temporal_targets, compatible_type_targets, decode_interval, patch_rows)

SCHEMA='gevis.center_decision.v1'


def _no_auxiliary_features(*args):
    """The learned route does not need the old numerical decision operations."""
    return None


@dataclass(frozen=True)
class CenterConfig:
    seed:int=91001
    train_runs_per_step:int=32
    validation_runs_per_step:int=16
    patches:int=32
    width:int=48
    epochs:int=30
    output_epochs:int=15
    batch_size:int=32
    max_cases:int=7
    learning_rate:float=.001

    def validate(self):
        if min(self.train_runs_per_step,self.validation_runs_per_step)<8:
            raise ValueError('Independent fit/selection/calibration/evaluation runs required')
        if min(self.patches,self.width,self.epochs,self.output_epochs,self.batch_size,self.max_cases)<1 or self.width%4:
            raise ValueError('Positive budgets and four-head compatible width required')
        if self.max_cases>14 or self.learning_rate<=0:raise ValueError('Bounded positive training budget required')


def _pool(values,patches):
    bins=patch_rows(len(values),patches)
    return np.stack([values[a:b].mean(0) for a,b in bins],axis=1)


def pair_context(current,reference,center):
    """Native local center/partner lag correlations; never realign the observations."""
    from .window_policy import _windows
    a,b=np.asarray(current,float),np.asarray(reference,float)
    if a.ndim!=2 or a.shape!=b.shape or not 0<=center<a.shape[1] or not np.isfinite([a,b]).all():
        raise ValueError('Finite aligned arrays and known UI center required')
    out=[]
    for width in (9,33):
        def correlations(x,offset):
            anchor=_windows(x[:,center:center+1],width)
            partner=_windows(x[np.clip(np.arange(len(x))+offset,0,len(x)-1)],width)
            anchor=anchor-anchor.mean(-1,keepdims=True);partner=partner-partner.mean(-1,keepdims=True)
            norm=np.linalg.norm(anchor,axis=-1)*np.linalg.norm(partner,axis=-1)
            return np.divide((anchor*partner).sum(-1),norm,out=np.zeros_like(norm),where=norm>1e-8)
        for lag in (-4,-2,0,2,4):
            ac,bc=correlations(a,lag),correlations(b,lag)
            out.extend([ac,bc,ac-bc])
    return np.stack(out,-1).astype('float32')


def center_features(current,reference,state,sensor_probability,patch_probability,center,patches):
    """Use frozen encoder states AND question attention; no detector threshold here."""
    a,b=np.asarray(current,float),np.asarray(reference,float)
    if isinstance(state,torch.Tensor):state=state.detach().cpu().numpy()
    state=np.asarray(state);sensors=a.shape[1]
    if state.ndim!=3 or len(state)!=1 or state.shape[1]%sensors or state.shape[-1]<8:
        raise ValueError('Aligned frozen numeric bridge required')
    grid=state[0].reshape(sensors,-1,state.shape[-1])
    # Preserve eight explicit numeric channels and sixteen summaries of latents.
    # Original STEP1/STEP2 weights and their full forwards remain unchanged.
    latent=np.stack([v.mean(-1) if v.shape[-1] else np.zeros(grid.shape[:2])
                     for v in np.array_split(grid[...,:-8],16,axis=-1)],-1)
    latent=np.concatenate([latent,grid[...,-8:]],-1)
    native=np.floor(np.linspace(0,grid.shape[1]-1,patches)).astype(int)
    latent=latent[:,native]
    base=continuous_difference_features(a,b,paired_patch_features(a,b,patches))
    anchor=np.broadcast_to(base[center:center+1],base.shape)
    relation=_pool(pair_context(a,b,center),patches)
    prior=np.stack([np.broadcast_to(np.asarray(sensor_probability)[:,None],(sensors,patches)),
                    np.asarray(patch_probability)[:,native]],-1)
    role=np.zeros((sensors,patches,1),dtype='float32');role[center]=1.
    result=np.concatenate([base,anchor,relation,np.sign(latent)*np.log1p(np.abs(latent)),prior,role],-1)
    if not np.isfinite(result).all():raise ValueError('Finite learned decision features required')
    return result.astype('float32')


@torch.no_grad()
def encode_features(module,current,reference,step,center,config):
    from .discovery import _learned_generic_attention
    encoder,meta=module.STEP_ENCODERS[step];encoder.to(module.DEVICE).eval()
    a=torch.as_tensor(current[None],dtype=torch.float32,device=module.DEVICE)
    b=torch.as_tensor(reference[None],dtype=torch.float32,device=module.DEVICE)
    state=module.numeric_state(encoder,a,b)
    sensor,patch=_learned_generic_attention(module,state,step,meta,state.shape[1]//len(meta['sensor_columns']),
                                          center_sensor=meta['sensor_columns'][center])
    return center_features(current,reference,state,sensor,patch,center,config.patches)


def targets_for_center(before,after,truth,spec,center,patches):
    """Supervision only: actual changed center or observed changed center pair.

    A context intervention on an unrelated flat center has no pair label. Pair
    labels mean changed local correlation, not a physical root cause.
    """
    sensors=before.shape[1];mask=np.zeros_like(before,dtype=bool)
    target=None
    if spec is not None and truth['effective']:
        if spec.sensor==center:
            mask=truth['changed_mask'].copy();target=center
        else:
            pair=pair_context(after,before,center)
            difference=np.abs(pair[:,spec.sensor,2::3]).max(-1)>1e-6
            if difference.any():mask[:,spec.sensor]=difference;target=spec.sensor
    return target,temporal_targets(mask,patches)


class CenterDecision:
    def __init__(self,detector,outputs,thresholds,config):
        self.detector,self.outputs,self.thresholds,self.config=detector,outputs,thresholds,config

    def predict_features(self,features,step,center,quantile=.99):
        output=self.outputs.predict(features,step)
        scores=output['sensor_scores'];group=self.thresholds[str(quantile)][step];threshold=group['threshold']
        thresholds=np.full_like(scores,group.get('pair_threshold',threshold))
        thresholds[center]=group.get('self_threshold',threshold)
        return {**output,'threshold':threshold,'sensor_thresholds':thresholds,'changed':scores>thresholds,'center':center,
                'ordinal_edges':group['ordinal_edges'],
                'self_ordinal_edges':group.get('self_ordinal_edges',group['ordinal_edges']),
                'pair_ordinal_edges':group.get('pair_ordinal_edges',group['ordinal_edges'])}

    @torch.no_grad()
    def review(self,module,current_rows,reference_rows,step,reference_info,*,center_sensor,scopes=None):
        from .questions import CRITERIA,DISCOVERY_QUESTIONS,SENSOR_QUESTION
        from .window_policy import CHOICES,SCHEMA as BASE_SCHEMA,OPERATIONS
        from .evidence import prepare_inputs
        from .discovery import _learned_generic_attention
        started=time.perf_counter()
        if scopes not in (None,['whole'],('whole',)):
            raise ValueError('Learned candidate supports the full UI-selected window only')
        if reference_info.process_count<1:raise ValueError('Reference processes required')
        encoder,meta=module.STEP_ENCODERS[step];names=list(meta['sensor_columns']);center=names.index(center_sensor)
        encoder.to(module.DEVICE).eval()
        a,b,ac,bc,state,patches,_=prepare_inputs(current_rows,reference_rows,step,encoder,meta,module.DEVICE,
            module.prepare_window,module.numeric_state,_no_auxiliary_features)
        if abs(ac.get('position_bin',0)-bc.get('position_bin',0))>1:raise ValueError('Incomparable reference position')
        attention,patch=_learned_generic_attention(module,state,step,meta,patches,center_sensor=center_sensor)
        length=int(a.shape[1])
        features=center_features(a[0].cpu().numpy(),b[0].cpu().numpy(),state,attention,patch,center,self.config.patches)
        q=round(1-module.DISCOVERY_POLICY.config.nominal_window_alert_rate,2)
        prediction=self.predict_features(features,step,center,q);scores=prediction['sensor_scores']
        thresholds=prediction['sensor_thresholds'];events=[];candidates=[]
        mapping={'level_shift':'level','drift':'slope','spike':'local_change','variance':'variability',
                 'stuck':'stuck','delay':'delay','relationship':'relationship'}
        def row(sid):
            own=sid==center;kind=int(prediction['type_probabilities'][sid,1:].argmax())+1
            mechanism=TYPE_LABELS[kind];statistic=mapping[mechanism]
            score=float(scores[sid]);threshold=float(thresholds[sid]);changed=bool(score>threshold)
            interval=decode_interval(prediction['patch_probabilities'][sid],length) if changed else None
            # A conditional decoder can have no supported patch. Never invent a
            # full-window interval to improve IoU.
            ratios={names[s]:float(np.exp(np.clip(scores[s]-thresholds[s],-40,40))) for s in range(len(names)) if s!=center}
            return {'sensor':center_sensor,'partner_sensor':None if own else names[sid],
                'time_scope':'whole','scope_rows':[0,length],'statistic':statistic,
                'statistic_index':OPERATIONS.index(statistic),'predicted_mechanism':mechanism,
                'normalized_score':float(np.exp(np.clip(score-threshold,-40,40))),
                'window_threshold':1.,'learned_logit':score,'learned_logit_threshold':threshold,
                'changed':changed,'detected_rows':interval,
                'related_sensors':[] if own else [names[sid]],'related_sensor_scores':{} if own else ratios,
                'ordinal_index':min(4,int(np.sum(score>np.asarray(prediction['self_ordinal_edges'] if own else prediction['pair_ordinal_edges'])))) if changed else 0,
                'pointwise':{'exceedance_intervals':[interval] if interval else [],'scope_rows':[0,length]},
                'localization_basis':'learned temporal head; conditioned on learned calibrated alert'}
        for sid in range(len(names)):
            r=row(sid);candidates.append(r)
            if r['changed']:events.append(r)
        events.sort(key=lambda r:-r['normalized_score']);winner=events[0] if events else None
        def view(relation):
            selected=[r for r in events if (r['partner_sensor'] is not None)==relation]
            supported=not relation or len(names)>1
            return {'accepted':supported,'changed':bool(selected) if supported else None,'events':selected,
                    'winner':selected[0] if selected else None,
                    'related_sensors':[r['partner_sensor'] for r in selected if r['partner_sensor']],
                    'meaning':'learned center-pair review signal, not cause' if relation else 'learned selected-center review signal'}
        own,relation=view(False),view(True);changed=bool(events)
        choice=winner['statistic_index']+1 if winner else 0;grade=winner['ordinal_index'] if winner else 0
        labels={'BOOLEAN':int(changed),'CHOICE':choice,'ORDINAL':grade};answers=[]
        probability=float(1/(1+np.exp(-np.clip(scores.max(),-40,40))))
        for kind,options in {**CRITERIA,'CHOICE':CHOICES}.items():
            index=labels[kind]
            if kind=='BOOLEAN':probabilities=dict(zip(options,[1-probability,probability]))
            elif kind=='CHOICE':
                sid=center if not winner or winner['partner_sensor'] is None else names.index(winner['partner_sensor'])
                values=prediction['type_probabilities'][sid].copy();probabilities={v:0. for v in options}
                if changed:
                    values[0]=0.;values/=max(float(values.sum()),1e-12)
                for i,label in enumerate(TYPE_LABELS):
                    mapped=0 if i==0 else OPERATIONS.index(mapping[label])+1
                    probabilities[options[mapped]]+=float(values[i])
            else:probabilities={v:float(i==index) for i,v in enumerate(options)}
            answers.append({'answer_type':kind,'question':DISCOVERY_QUESTIONS[kind],'answer':options[index],
                'candidate_answer':options[index],'accepted':True,'probabilities':probabilities,
                'confidence':{'answer':probabilities[options[index]]},'evidence':winner,
                'coverage_warnings':['synthetic_training_not_verified_fault_probability'],
                'decision_source':SCHEMA,'score_meaning':'synthetic review-task score; ORDINAL is empirical rarity'})
        partner_ids=[sid for sid in range(len(names)) if sid!=center]
        partner_scores=scores[partner_ids];mass=np.exp(partner_scores-partner_scores.max()) if len(partner_ids) else np.array([])
        mass=mass/mass.sum() if len(mass) else mass
        distribution=dict(zip([names[i] for i in partner_ids],map(float,mass)))
        partner=max(distribution,key=distribution.get) if distribution else None
        answers.append({'answer_type':'CHOICE_SENSOR','typed_family':'CHOICE','question':SENSOR_QUESTION,
            'answer':partner,'candidate_answer':partner,'accepted':bool(distribution),'probabilities':distribution,
            'confidence':{'answer':distribution.get(partner)},'reference_only':not bool(relation['changed']),
            'evidence':winner,'decision_source':SCHEMA,'score_meaning':'relative learned partner scores, not fault probability'})
        summary={'center_sensor':center_sensor,'self_changed':own['changed'],'relationship_changed':relation['changed'],
            'related_sensors':relation['related_sensors'],'self_type':own['winner']['statistic'] if own['winner'] else None,
            'self_interval':own['winner']['detected_rows'] if own['winner'] else None,
            'relationship_interval':relation['winner']['detected_rows'] if relation['winner'] else None}
        for answer in answers:answer['center_analysis']=summary
        return {'contract':SCHEMA,'base_contract':BASE_SCHEMA,'step':step,'center_sensor':center_sensor,
            'analysis_mode':'center_and_relationship','accepted':True,'changed':changed,'winner':winner,
            'events':events,'ranked_evidence':candidates,'answers':answers,'self_analysis':own,
            'relationship_analysis':relation,'context_sensors':names,'selected_sensors':[center_sensor],
            'selected_scopes':['whole'],'attention_sensor_scores':dict(zip(names,map(float,attention))),
            'attention_top_sensor':names[int(attention.argmax())],'complete_coverage':True,'criterion_coverage':1.,
            'missing_criteria':[],'warnings':['learned_intervention_task_not_verified_fault_probability'],
            'latency_ms':(time.perf_counter()-started)*1000,'threshold_source':'Validation run-max learned scores',
            'change_decision_source':'learned head; old numerical candidate gate is not applied'}


def source_identity():
    from . import optimization_models,optimization_heads,questions,injection,evaluation,discovery
    from pecvd import temporal_bridge,temporal_patchtst,multiscale_patchtst,relational_features
    modules=(optimization_models,optimization_heads,questions,injection,evaluation,discovery,temporal_bridge,temporal_patchtst,multiscale_patchtst,relational_features)
    return {Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}|{
        'center_decision.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'center_training.py':hashlib.sha256(Path(__file__).with_name('center_training.py').read_bytes()).hexdigest()}


def candidate_path(backbone_path,config=CenterConfig()):
    import json
    fingerprint=hashlib.sha256(json.dumps({'sources':source_identity(),'config':asdict(config)},sort_keys=True).encode()).hexdigest()[:12]
    return Path(backbone_path).parent/f'center_decision_{fingerprint}.pt'


def artifact_identity(module,config):
    from .discovery_calibration import identity_for
    from .window_policy import WindowConfig
    return {'sources':source_identity(),'config':asdict(config),
            'bridge':identity_for(module.ENCODER_ROOT,module.BACKBONE_PATH,WindowConfig())}


def save_candidate(head,path,identity,adoption):
    if not adoption.get('accepted'):raise ValueError('Rejected candidate must not become a runtime artifact')
    detector=head.detector
    payload={'schema':SCHEMA,'identity':identity,'config':asdict(head.config),'adoption':adoption,
        'thresholds':head.thresholds,'mean':torch.as_tensor(detector.mean),'scale':torch.as_tensor(detector.scale),
        'query':detector.model.query.detach().cpu(),'features':len(detector.mean),
        'detector':{k:v.detach().cpu() for k,v in detector.model.state_dict().items()},
        'outputs':{k:v.detach().cpu() for k,v in head.outputs.heads.state_dict().items()}}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(payload,stream)


def load_candidate(module,path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    config=CenterConfig(**payload['config']);config.validate()
    if payload.get('schema')!=SCHEMA or payload['identity']!=artifact_identity(module,config):
        raise ValueError('Learned center artifact identity mismatch; no fallback or old-weight overwrite')
    if not payload['adoption'].get('accepted'):raise ValueError('Unaccepted candidate artifact')
    detector=PairedEncoderTrial('paired_cross_attention',payload['query'],width=config.width,device=module.DEVICE)
    detector.mean=payload['mean'].numpy();detector.scale=payload['scale'].numpy();detector.control_scales={}
    detector.model=PairedPatchDetector(payload['features'],payload['query'],'paired_cross_attention',config.width).to(module.DEVICE)
    detector.model.load_state_dict(payload['detector']);detector.model.eval()
    outputs=GroundedOutputTrial(detector);outputs.heads=GroundedOutputHeads(config.width,config.width*3).to(module.DEVICE)
    outputs.heads.load_state_dict(payload['outputs']);outputs.heads.eval()
    return CenterDecision(detector,outputs,payload['thresholds'],config)


def selected_candidate(module):
    path=os.environ.get('GEVIS_CENTER_DECISION_PATH')
    if not path:return None
    path=str(Path(path).resolve())
    cached=getattr(module,'_CENTER_DECISION_CACHE',None)
    if cached is None or cached[0]!=path:
        cached=(path,load_candidate(module,path));module._CENTER_DECISION_CACHE=cached
    return cached[1]
