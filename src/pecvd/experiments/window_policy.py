"""Frozen-model, operation-aware discovery. No labels or Test fitting.

Train estimates sensor/operation scales. Untouched Validation run maxima set
the complete-window threshold. These are empirical controls, NOT verified
normal labels or a certified false-positive probability.
"""
from dataclasses import asdict,dataclass
import math
import numpy as np
from .evidence import segments
from .questions import CRITERIA

SCHEMA='gevis.window_evidence.v2'
OPERATIONS=('level','variability','slope','local_change','residual','stuck','delay','relationship')
CORE_KINDS=('level_shift','drift','spike','variance')


def active_operations(config):
    return OPERATIONS[:5] if config.operation_profile=='core' else OPERATIONS
CHOICES=(*CRITERIA['CHOICE'],'Stuck','Delay','Sensor relationship')
MODE='window_calibrated_v2'


def contract():
    return {'schema':SCHEMA,'decision':'Validation run-max complete-window threshold',
        'normalization':'Train sensor/operation window-max empirical quantiles',
        'localization':'same winning operation trace; half-open row intervals; centered offline rolling features',
        'operations':list(OPERATIONS),'typed_outputs':'one winning event for BOOLEAN/CHOICE/ORDINAL and displayed evidence',
        'step2_cross_attention_executed':True,'test_used_for_fitting':False,
        'population_false_alarm_guarantee':False,'meaning':'reference-relative review signal, not confirmed fault'}


@dataclass(frozen=True)
class WindowConfig:
    gate_scope:str='full_window'
    operation_profile:str='all'
    noise_scale_floor:bool=False
    scale_quantile:float=.95
    nominal_window_alert_rate:float=.05
    threshold_multiplier:float=1.
    local_width:int=9
    max_lag:int=4
    delay_min_improvement:float=.25
    merge_gap:int=1
    min_duration:int=2
    min_train_runs:int=8
    min_validation_runs:int=4
    max_windows_per_step:int=256
    max_calibration_windows_per_step:int=2048
    max_references:int=16
    seed:int=42

    def validate(self):
        if self.operation_profile not in ('all','core'):raise ValueError('Unknown operation profile')
        if self.operation_profile=='core' and self.gate_scope!='position_rank_v4':
            raise ValueError('Core operation profile requires position_rank_v4 criteria')
        if not np.isfinite(self.threshold_multiplier) or not 0<self.threshold_multiplier<=1:
            raise ValueError('Threshold multiplier must lie inside (0,1]')
        if self.gate_scope not in ('full_window','center_views_v3','position_rank_v4'):
            raise ValueError(f'Unknown gate scope: {self.gate_scope!r}; expected full_window, center_views_v3 or position_rank_v4')
        if not 0<self.scale_quantile<1 or not 0<self.nominal_window_alert_rate<1:
            raise ValueError('Quantiles/rates must lie inside (0,1)')
        if self.local_width<3 or self.local_width%2!=1 or self.max_lag<1:
            raise ValueError('Odd local width >=3 and positive lag required')
        if not 0<self.delay_min_improvement<1:raise ValueError('Delay improvement fraction must lie inside (0,1)')
        if min(self.min_train_runs,self.min_validation_runs,self.min_duration)<1 or self.merge_gap<0:
            raise ValueError('Positive sample/duration limits required')
        if self.max_windows_per_step<self.min_train_runs or self.max_calibration_windows_per_step<self.min_train_runs or self.max_references<1:
            raise ValueError('Invalid bounded preparation budget')


def _windows(x,width):
    half=width//2
    return np.lib.stride_tricks.sliding_window_view(np.pad(x,((half,half),(0,0)),mode='edge'),width,axis=0)


def operation_traces(current,reference,residual=None,config=WindowConfig()):
    """Return T x sensors x operations and pairwise relation traces.

    Inputs are scaled with source TRAIN constants, never window-fitted scaling.
    Rolling evidence is centered/offline and has width-dependent boundary blur.
    Delay is local MSE improvement under a signed reference shift, not causality.
    Relationship is symmetric: report sensor pairs, never claim a unique cause.
    """
    config.validate();a=np.asarray(current,float);b=np.asarray(reference,float)
    if a.ndim!=2 or a.shape!=b.shape or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matching finite time/sensor arrays required')
    residual=np.asarray(residual,float)
    if residual.shape!=a.shape or not np.isfinite(residual).all() or (residual<0).any():
        raise ValueError('Finite nonnegative STEP 1 residual-difference trace required')
    width=min(config.local_width,len(a) if len(a)%2 else len(a)-1)
    aw,bw=_windows(a,width),_windows(b,width)
    am,bm=aw.mean(-1),bw.mean(-1)
    ast,bst=aw.std(-1),bw.std(-1)
    t=np.arange(width)-(width-1)/2
    slope=np.abs(((aw-bw)*t).sum(-1)/np.square(t).sum())
    # Reduction in variation is scored separately from variance increase.
    variance=np.maximum(ast-bst,0.)
    if config.operation_profile=='core':
        zero=np.zeros_like(a)
        return np.stack([np.abs(am-bm),variance,slope,np.abs(a-b),residual,zero,zero,zero],-1),{
            'relationship':np.zeros((len(a),a.shape[1],a.shape[1])),
            'lag_rows':np.zeros_like(a,dtype=int),'local_width':width}
    stuck=np.maximum(bst-ast,0.)*np.clip(1-ast/np.maximum(bst,1e-6),0,1)
    base=np.square(aw-bw).mean(-1);best=base.copy();lag=np.zeros_like(base,dtype=int)
    maxlag=min(config.max_lag,(width-1)//2)
    for offset in range(-maxlag,maxlag+1):
        if not offset:continue
        shifted=b[np.clip(np.arange(len(b))+offset,0,len(b)-1)]
        error=np.square(aw-_windows(shifted,width)).mean(-1)
        improved=error<best
        lag=np.where(improved,offset,lag);best=np.minimum(best,error)
    delay=np.sqrt(np.maximum(base-best,0.))
    # Flat/near-flat segments contain no identifiable lag information.
    identifiable=(ast>1e-6)&(bst>1e-6)&((base-best)>config.delay_min_improvement*np.maximum(base,1e-8))
    delay=np.where(identifiable,delay,0.);lag=np.where(identifiable,lag,0)
    ac=aw-am[...,None];bc=bw-bm[...,None]
    an=ac/np.maximum(np.linalg.norm(ac,axis=-1,keepdims=True),1e-8)
    bn=bc/np.maximum(np.linalg.norm(bc,axis=-1,keepdims=True),1e-8)
    pair=np.abs(np.einsum('tsw,tqw->tsq',an,an)-np.einsum('tsw,tqw->tsq',bn,bn))
    for sid in range(a.shape[1]):pair[:,sid,sid]=0.
    relation=pair.max(-1)
    if residual is None:raise ValueError('Frozen STEP 1 residual trace is required')
    residual=np.asarray(residual,float)
    if residual.shape!=a.shape or not np.isfinite(residual).all() or (residual<0).any():
        raise ValueError('Finite nonnegative STEP 1 residual-difference trace required')
    traces=np.stack([np.abs(am-bm),variance,slope,np.abs(a-b),residual,stuck,delay,relation],-1)
    return traces,{'relationship':pair,'lag_rows':lag,'local_width':width}


def _record(record,split,config):
    if record.get('split')!=split:raise ValueError('Calibration partition violation; Test never fits policy')
    if not record.get('run_id') or not record.get('reference_run_ids'):
        raise ValueError('Explicit independent current/reference run identities required')
    if str(record['run_id']) in set(map(str,record['reference_run_ids'])):
        raise ValueError('Self-run reference is forbidden')
    trace,_=operation_traces(record['current'],record['reference'],record['residual'],config)
    if trace.shape[1]!=len(record['sensors']):raise ValueError('Sensor grid mismatch')
    return trace


def _fit_window_policy_core(train_records,validation_records,identity,config=WindowConfig()):
    config.validate();train=list(train_records);validation=list(validation_records)
    train_ids={str(r['run_id']) for r in train}|{str(v) for r in train for v in r.get('reference_run_ids',[])}
    val_ids={str(r['run_id']) for r in validation}
    if train_ids&val_ids:raise ValueError('Train/Validation run overlap')
    groups={};normalized={};run_features={}
    for r in train:
        trace=_record(r,'train',config)
        if not set(map(str,r['reference_run_ids']))<=train_ids:raise ValueError('Train references must belong to Train')
        key=r['step'];sensors=list(r['sensors'])
        if key in groups and groups[key]['sensors']!=sensors:raise ValueError('Sensor order changed')
        groups.setdefault(key,{'sensors':sensors})
        run=(key,str(r['run_id']));feature=trace.max(axis=0)
        run_features[run]=np.maximum(run_features.get(run,feature),feature)
    for step,group in groups.items():
        features=[v for (s,_),v in run_features.items() if s==step]
        raw=np.quantile(features,config.scale_quantile,axis=0,method='higher')
        # Lag improvement is often exactly zero on matched Train traces. Its
        # resolution is bounded by the observed Train point-difference noise
        # (same amplitude units), not division by floating-point epsilon.
        raw[:,OPERATIONS.index('delay')]=np.maximum(raw[:,OPERATIONS.index('delay')],raw[:,OPERATIONS.index('local_change')])
        if config.noise_scale_floor:
            # TRAIN-derived amplitude floor; never divide a constant-channel
            # operation by epsilon. Inputs already use TRAIN standard units.
            point=raw[:,OPERATIONS.index('local_change')]
            positive=point[point>0]
            pooled=float(np.median(positive)) if len(positive) else 1.
            noise=np.maximum(point,pooled)
            floors=np.repeat(noise[:,None],len(OPERATIONS),axis=1)
            floors[:,OPERATIONS.index('slope')]/=config.local_width
            floors[:,OPERATIONS.index('residual')]=noise**2
            relation=raw[:,OPERATIONS.index('relationship')]
            positive=relation[relation>0]
            floors[:,OPERATIONS.index('relationship')]=float(np.median(positive)) if len(positive) else 1.
            raw=np.maximum(raw,floors)
        scales=np.maximum(raw,32*np.finfo(np.float32).eps*np.maximum(raw,1.))
        normalized[step]=scales
        group.update(scales=scales.tolist(),train_runs=len(features))
    maxima={};window_scores={};center_maxima={};drivers={}
    for r in validation:
        trace=_record(r,'validation',config)
        if not set(map(str,r['reference_run_ids']))<=train_ids:raise ValueError('Validation references must come from Train only')
        step=r['step']
        if step not in groups or list(r['sensors'])!=groups[step]['sensors']:raise ValueError('Validation sensor/Step grid mismatch')
        normalized_trace=trace/normalized[step][None]
        score=float(normalized_trace.max());run=(step,str(r['run_id']))
        if run not in maxima or score>maxima[run]:
            tid,sid,op=np.unravel_index(normalized_trace.argmax(),normalized_trace.shape)
            drivers[run]=dict(threshold_driver_sensor=groups[step]['sensors'][sid],
                threshold_driver_operation=OPERATIONS[op],threshold_driver_raw_score=float(trace[tid,sid,op]),
                threshold_driver_scale=float(normalized[step][sid,op]))
        maxima[run]=max(maxima.get(run,score),score);window_scores.setdefault(step,[]).append(score)
        if config.gate_scope=='center_views_v3':
            normalized_trace=trace/normalized[step][None]
            feature=np.stack([normalized_trace[...,:-1].max(axis=(0,2)),normalized_trace[...,-1].max(axis=0)],-1)
            center_maxima[run]=np.maximum(center_maxima.get(run,feature),feature)
    for step,group in groups.items():
        values=np.sort([v for (s,_),v in maxima.items() if s==step]);n=len(values)
        rank=math.ceil((n+1)*(1-config.nominal_window_alert_rate))
        ready=group['train_runs']>=config.min_train_runs and n>=config.min_validation_runs
        calibrated_threshold=float(values[min(rank,n)-1]) if n else None
        threshold=calibrated_threshold*config.threshold_multiplier if n else None
        driver=next((drivers[k] for k,v in maxima.items() if k[0]==step and v==calibrated_threshold),{})
        # Increasing observed tail bands; ties legitimately skip grades.
        bands=(np.maximum.accumulate([calibrated_threshold,*[float(np.quantile(values,q,method='higher')) for q in (.975,.99,1.)]])*config.threshold_multiplier).tolist() if n else None
        group.update(**driver,supported=ready,window_threshold=threshold,ordinal_edges=bands,
            calibrated_window_threshold=calibrated_threshold,threshold_multiplier=config.threshold_multiplier,
            requested_order_statistic=rank,used_order_statistic=min(rank,n) if n else None,
            validation_runs=n,validation_windows=len(window_scores.get(step,[])),
            control_rate_resolution=1/n if n else None,insufficient_controls_for_nominal_rate=rank>n,
            validation_window_alert_rate=float(np.mean(np.asarray(window_scores[step])>threshold)) if n else None,
            calibration_unit='maximum of sampled window scores per independent validation run')
        if config.gate_scope=='center_views_v3':
            features=np.sort([v for (s,_),v in center_maxima.items() if s==step],axis=0)
            center_rank=math.ceil((n+1)*(1-config.nominal_window_alert_rate/2))
            group.update(center_thresholds=(features[min(center_rank,n)-1]*config.threshold_multiplier).tolist() if n else None,
                calibrated_center_thresholds=features[min(center_rank,n)-1].tolist() if n else None,
                center_ordinal_edges=((np.maximum.accumulate(np.stack([features[min(center_rank,n)-1],
                    *[np.quantile(features,q,axis=0,method='higher') for q in (.975,.99,1.)]]),axis=0)*config.threshold_multiplier).transpose(1,2,0).tolist() if n else None),
                center_gate_adopted=False,center_gate_tail_supported=center_rank<=n,
                center_requested_order_statistic=center_rank,center_used_order_statistic=min(center_rank,n) if n else None,
                center_threshold_meaning='per sensor: self-operation maximum / relationship maximum; half tail budget per view')
    return {'schema':SCHEMA,'identity':identity,'config':asdict(config),'operations':list(OPERATIONS),
        'groups':groups,'test_used_for_fitting':False,'normal_labels_verified':False,
        'threshold_source':'Train sensor/operation scales + untouched Validation run-max full-window scores',
        'population_false_alarm_guarantee':False,'ordinal_meaning':'empirical validation rarity bands; not physical severity'}


def fit_window_policy(train_records,validation_records,identity,config=WindowConfig()):
    """Legacy unchanged; center v3 uses disjoint calibration/audit Validation runs."""
    from dataclasses import replace
    train=list(train_records);validation=list(validation_records);config.validate()
    if config.gate_scope=='position_rank_v4':
        from .position_rank_policy import fit_position_policy
        return fit_position_policy(train,validation,identity,config)
    if config.gate_scope=='full_window':return _fit_window_policy_core(train,validation,identity,config)
    ids=sorted({str(r['run_id']) for r in validation})
    order=np.random.default_rng(config.seed).permutation(len(ids));cut=len(ids)//2
    calibration_ids={ids[i] for i in order[:cut]}
    calibration=[r for r in validation if str(r['run_id']) in calibration_ids]
    audit=[r for r in validation if str(r['run_id']) not in calibration_ids]
    candidate=_fit_window_policy_core(train,calibration,identity,config)
    legacy_config=replace(config,gate_scope='full_window',noise_scale_floor=False)
    baseline=_fit_window_policy_core(train,calibration,identity,legacy_config)
    fallback=_fit_window_policy_core(train,validation,identity,legacy_config)
    observed={}
    for row in audit:
        trace=_record(row,'validation',config);step=row['step']
        # The core checks calibration references; audit references must match too.
        train_ids={str(r['run_id']) for r in train}|{str(v) for r in train for v in r['reference_run_ids']}
        if str(row['run_id']) in train_ids or not set(map(str,row['reference_run_ids']))<=train_ids:
            raise ValueError('Audit references/partitions must remain TRAIN-only')
        c=candidate['groups'][step];b=baseline['groups'][step]
        if c['center_thresholds'] is None:continue
        ratios=trace/np.asarray(c['scales'])[None]
        threshold=np.asarray(c['center_thresholds'])
        limits=np.repeat(threshold[:,0,None],len(OPERATIONS),axis=1);limits[:,-1]=threshold[:,1]
        # UI selects ONE center. OR windows/operations within that center,
        # never OR unrelated centers before deciding whether to adopt it.
        ca=(ratios>limits[None]).any(axis=(0,2))
        ba=(trace/np.asarray(b['scales'])[None]>b['window_threshold']).any(axis=(0,2))
        key=(step,str(row['run_id']));old=observed.get(key,(np.zeros_like(ca),np.zeros_like(ba)))
        observed[key]=(old[0] | ca,old[1] | ba)
    for step,group in candidate['groups'].items():
        pairs=[v for (s,_),v in observed.items() if s==step]
        size=len(group['sensors'])
        c_hits=np.sum([c for c,_ in pairs],axis=0,dtype=int) if pairs else np.zeros(size,dtype=int)
        b_hits=np.sum([b for _,b in pairs],axis=0,dtype=int) if pairs else np.zeros(size,dtype=int)
        new_hits=np.sum([c & ~b for c,b in pairs],axis=0,dtype=int) if pairs else np.zeros(size,dtype=int)
        mask=((c_hits==0)&(new_hits==0)&bool(group['supported'] and len(pairs)>=config.min_validation_runs))
        adopted=bool(mask.any())
        new=sum(bool((c & ~b).any()) for c,b in pairs)
        diagnostics=dict(center_thresholds=group['center_thresholds'],center_ordinal_edges=group['center_ordinal_edges'],center_gate_adopted=adopted,
            calibrated_center_thresholds=group['calibrated_center_thresholds'],
            center_adoption_mask=mask.tolist(),
            center_audit_alerts=c_hits.tolist(),center_baseline_audit_alerts=b_hits.tolist(),
            center_new_audit_alert_runs=new_hits.tolist(),
            center_gate_tail_supported=group['center_gate_tail_supported'],audit_runs=len(pairs),
            center_requested_order_statistic=group['center_requested_order_statistic'],
            center_used_order_statistic=group['center_used_order_statistic'],
            candidate_audit_alerts=sum(bool(c.any()) for c,_ in pairs),baseline_audit_alerts=sum(bool(b.any()) for _,b in pairs),
            new_audit_alert_runs=new,gate_selection='per UI center: zero combined self/pair untouched audit-run alerts and no new alerts; no fault-normal claim',
            calibration_validation_runs=group['validation_runs'],
            fallback_validation_runs=fallback['groups'][step]['validation_runs'],
            candidate_scales=group['scales'],candidate_window_threshold=group['window_threshold'])
        # Base group always retains a valid legacy full-window policy. Center
        # inference explicitly selects candidate scales only for accepted centers.
        group.update(fallback['groups'][step])
        group.update(diagnostics)
    candidate.update(threshold_source='TRAIN noise-aware scales; disjoint Validation calibration/audit; center views with guarded legacy fallback',
        gate_contract='center_views_v3',validation_audit_used_for_selection=True,calibration_audit_overlap=False)
    return candidate


def merged_intervals(mask,gap,minimum):
    intervals=segments(mask);merged=[]
    for lo,hi in intervals:
        if merged and lo-merged[-1][1]<=gap:merged[-1][1]=hi
        else:merged.append([lo,hi])
    return [tuple(iv) for iv in merged if iv[1]-iv[0]>=minimum]


class WindowPolicy:
    def __init__(self,payload):
        if payload.get('schema')!=SCHEMA or payload.get('operations')!=list(OPERATIONS):raise ValueError('Unsupported window policy')
        self.payload=payload;self.config=WindowConfig(**payload['config']);self.config.validate()

    def analyze(self,current,reference,residual,step,sensors,*,selected=None,scopes=None,center_sensor=None,cohort=None,position=None):
        if self.config.gate_scope=='position_rank_v4':
            from .position_rank_policy import analyze_position
            return analyze_position(self,current,reference,residual,step,sensors,selected=selected,scopes=scopes,
                center_sensor=center_sensor,cohort=cohort,position=position)
        group=self.payload['groups'].get(step)
        if group is None or group['sensors']!=list(sensors):raise ValueError('Window policy sensor/Step mismatch')
        selected=list(sensors) if selected is None else list(selected)
        scopes=['whole','early','late'] if scopes is None else list(scopes)
        if not selected or len(set(selected))!=len(selected) or any(s not in sensors for s in selected):raise ValueError('Known unique sensors required')
        if not scopes or len(set(scopes))!=len(scopes) or any(s not in ('whole','early','late') for s in scopes):raise ValueError('Known unique scopes required')
        traces,aux=operation_traces(current,reference,residual,self.config)
        center_gate=center_sensor is not None and group.get('center_gate_adopted',False)
        if center_gate and 'center_adoption_mask' in group:
            center_gate=bool(group['center_adoption_mask'][list(sensors).index(center_sensor)]) if center_sensor in sensors else False
        scales=np.asarray(group.get('candidate_scales',group['scales']) if center_gate else group['scales'])
        ratios=traces/scales[None];n=len(traces);candidates=[]
        threshold=group['window_threshold']
        if center_sensor is not None and (center_sensor not in sensors or selected!=[center_sensor]):
            raise ValueError('Center gate requires exactly the selected UI center')
        for sensor in selected:
            sid=list(sensors).index(sensor)
            for scope in scopes:
                lo,hi=(0,n) if scope=='whole' else (0,max(2,n//3)) if scope=='early' else (2*n//3,n)
                for op,name in enumerate(OPERATIONS):
                    threshold=(group['center_thresholds'][sid][int(name=='relationship')] if center_gate else group['window_threshold'])
                    values=ratios[lo:hi,sid,op];peak=int(values.argmax())+lo;score=float(values.max())
                    changed=bool(group['supported'] and score>threshold)
                    # Localization uses the operation's own Train-normalized trace;
                    # it is conditional evidence, not a pointwise FPR guarantee.
                    minimum=1 if name=='local_change' else self.config.min_duration
                    point_threshold=threshold if center_gate else 1.
                    intervals=[(a+lo,b+lo) for a,b in merged_intervals(values>point_threshold,0 if minimum==1 else self.config.merge_gap,minimum)]
                    detected=next((list(iv) for iv in intervals if iv[0]<=peak<iv[1]),None)
                    related=[];partner_scores={}
                    if name=='relationship' and len(sensors)>1:
                        partner=int(aux['relationship'][peak,sid].argmax());related=[sensor,sensors[partner]]
                        partner_scores={other:float(aux['relationship'][lo:hi,sid,j].max()/scales[sid,op])
                                        for j,other in enumerate(sensors) if j!=sid}
                    candidates.append({'sensor':sensor,'time_scope':scope,'scope_rows':[lo,hi],
                        'statistic':name,'statistic_index':op,'normalized_score':score,'window_threshold':threshold,
                        'raw_score':float(traces[lo:hi,sid,op].max()),'normalization_scale':float(scales[sid,op]),
                        'threshold_multiplier':self.config.threshold_multiplier,
                        'unrelaxed_threshold':(group.get('calibrated_center_thresholds',group['center_thresholds'])[sid][int(name=='relationship')]
                            if center_gate else group.get('calibrated_window_threshold',group['window_threshold'])),
                        'active_gate':'center views' if center_gate else 'legacy full-window',
                        'changed':changed,'detected_rows':detected if changed else None,'related_sensors':related,
                        'related_sensor_scores':partner_scores,
                        'estimated_lag_rows':int(aux['lag_rows'][peak,sid]) if name=='delay' else None,
                        'ordinal_index':int(np.sum(score>np.asarray(group['center_ordinal_edges'][sid][int(name=='relationship')] if center_gate else group['ordinal_edges']))) if changed else 0,
                        'pointwise':{'peak_row':peak,'scope_rows':[lo,hi],'exceedance_intervals':intervals if changed else [],
                            'threshold_normalized':point_threshold,'local_width':aux['local_width'],
                            'coordinate_system':'prepared window row; half-open; not seconds'},
                        'localization_basis':name+' Train-normalized local trace; not learned attention'})
        candidates.sort(key=lambda e:-e['normalized_score']);events=[e for e in candidates if e['changed']]
        return {'ranked_evidence':candidates,'events':events,'winner':events[0] if events else None,
                'accepted':bool(group['supported']),'changed':bool(events) if group['supported'] else None,
                'change_decision_source':('center-view calibrated numerical evidence v3; independent control audit passed' if center_gate else
                    'legacy full-window gate retained: center v3 audit not accepted' if center_sensor is not None and self.config.gate_scope=='center_views_v3' else 'calibrated numerical full-window evidence'),
                'calibration':{k:v for k,v in group.items() if k not in ('sensors','scales')},
                'maximum_score':candidates[0]['normalized_score']}
