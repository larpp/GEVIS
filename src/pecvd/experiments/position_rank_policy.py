"""Context-conditional empirical evidence; no labels, neural forwards or I/O.

Rank scores are NOT p-values: beyond TRAIN support an explicit log excess
preserves magnitude ordering. Validation calibrates that complete scoring rule.
"""
from dataclasses import asdict
import json, math
import numpy as np
from .window_policy import SCHEMA, OPERATIONS, _record, operation_traces, merged_intervals, active_operations, CORE_KINDS


def context_key(cohort, position):
    if cohort is None or position is None:
        raise ValueError('Explicit cohort and position required for position-rank criteria')
    if not np.isfinite(float(position)) or int(position)!=float(position):
        raise ValueError('Finite integer position required')
    return json.dumps([str(cohort),int(position)],ensure_ascii=False,separators=(',',':'))


def rank_score(values, bank, scale):
    """Strict empirical rank (ties equal), plus explicit upper-support excess.

    Below/equal constant TRAIN scores remain zero; nonzero unseen differences
    retain ordering. Scale comes only from TRAIN, never the queried window.
    """
    bank=np.asarray(bank,float);values=np.asarray(values,float)
    if bank.ndim!=1 or not len(bank) or not np.isfinite(bank).all() or np.any(np.diff(bank)<0):
        raise ValueError('Sorted nonempty finite TRAIN distribution required')
    if not np.isfinite(scale) or scale<=0 or not np.isfinite(values).all():
        raise ValueError('Finite score and positive TRAIN tail scale required')
    rank=np.searchsorted(bank,values,side='left')/len(bank)
    return rank+np.log1p(np.maximum(values-bank[-1],0.)/scale)


def _limits(values, rate, views=2):
    n=len(values);rank=math.ceil((n+1)*(1-rate/views))
    if not n:return None,None,rank
    ordered=np.sort(values,axis=0);bound=ordered[min(rank,n)-1]
    edges=np.maximum.accumulate(np.stack([bound,*[np.quantile(ordered,q,axis=0,method='higher') for q in (.975,.99,1.)]]),axis=0)
    return bound,edges,rank


def fit_position_policy(train,validation,identity,config):
    config.validate()
    if config.threshold_multiplier!=1.:raise ValueError('Position-rank policy forbids arbitrary threshold multipliers')
    train=list(train);validation=list(validation)
    train_ids={str(r['run_id']) for r in train}|{str(v) for r in train for v in r['reference_run_ids']}
    if train_ids & {str(r['run_id']) for r in validation}:raise ValueError('Train/Validation run overlap')
    groups={};features={}
    for row in train:
        trace=_record(row,'train',config)
        if not set(map(str,row['reference_run_ids']))<=train_ids:raise ValueError('References must be TRAIN only')
        step=row['step'];names=list(row['sensors']);key=context_key(row.get('cohort'),row.get('position'))
        group=groups.setdefault(step,{'sensors':names,'contexts':{}})
        if group['sensors']!=names:raise ValueError('Sensor order changed')
        k=(step,key,str(row['run_id']));peak=trace.max(0)
        features[k]=np.maximum(features.get(k,peak),peak)
    for step,g in groups.items():
        for key in sorted({k[1] for k in features if k[0]==step}):
            samples=np.asarray([v for (s,c,_),v in features.items() if s==step and c==key])
            bank=np.sort(samples,axis=0)
            # Data-derived spread; truly constant channels use TRAIN standard
            # amplitude units, not epsilon. Squared residuals retain squared units.
            spread=np.ptp(samples,axis=0)
            point=np.quantile(samples[:,:,OPERATIONS.index('local_change')],.95,axis=0,method='higher')
            noise=np.where(point>0,point,1.)
            floor=np.repeat(noise[:,None],len(OPERATIONS),axis=1)
            floor[:,OPERATIONS.index('slope')]/=config.local_width
            floor[:,OPERATIONS.index('residual')]=noise**2
            floor[:,OPERATIONS.index('relationship')]=1.  # correlation difference units
            scale=np.where(spread>0,spread,floor)
            available=[r.get('available_context_runs') for r in train if r['step']==step and context_key(r.get('cohort'),r.get('position'))==key and r.get('available_context_runs') is not None]
            g['contexts'][key]={'train_runs':len(bank),'available_train_runs':max(available) if available else None,
                'bank':bank.tolist(),'tail_scales':scale.tolist()}
    scores={};window_scores={}
    for row in validation:
        trace=_record(row,'validation',config)
        if not set(map(str,row['reference_run_ids']))<=train_ids:raise ValueError('Validation references must be TRAIN only')
        step=row['step'];key=context_key(row.get('cohort'),row.get('position'))
        if step not in groups or list(row['sensors'])!=groups[step]['sensors']:raise ValueError('Validation sensor/Step mismatch')
        context=groups[step]['contexts'].get(key)
        if context is None:continue  # explicit unsupported context; no cross-position pooling
        converted=convert(trace,context)
        peak=np.stack([converted[:,:,:-1].max(axis=(0,2)),converted[:,:,-1].max(0)],-1)
        k=(step,key,str(row['run_id']));scores[k]=np.maximum(scores.get(k,peak),peak)
        window_scores.setdefault((step,key),[]).append(peak)
    for step,g in groups.items():
        for key,c in g['contexts'].items():
            values=[v for (s,k,_),v in scores.items() if s==step and k==key]
            bound,edges,rank=_limits(values,config.nominal_window_alert_rate,1 if config.operation_profile=='core' else 2)
            available=[r.get('available_context_runs') for r in validation if r['step']==step and context_key(r.get('cohort'),r.get('position'))==key and r.get('available_context_runs') is not None]
            c['available_validation_runs']=max(available) if available else None
            reasons=[]
            for split,count,minimum in (('train',c['train_runs'],config.min_train_runs),('validation',len(values),config.min_validation_runs)):
                if count<minimum:
                    total=c.get('available_'+split+'_runs')
                    reasons.append(split+('_sampling_budget_shortfall' if total is not None and total>=minimum else '_insufficient_context_support'))
            c.update(validation_runs=len(values),supported=c['train_runs']>=config.min_train_runs and len(values)>=config.min_validation_runs,
                support_failure=';'.join(reasons) if reasons else None,
                thresholds=bound.tolist() if bound is not None else None,
                ordinal_edges=edges.transpose(1,2,0).tolist() if edges is not None else None,
                requested_order_statistic=rank,used_order_statistic=min(rank,len(values)) if values else None,
                insufficient_tail_support=rank>len(values),
                requested_control_alert_budget=config.nominal_window_alert_rate,
                minimum_validation_runs_for_budget=math.ceil((1 if config.operation_profile=='core' else 2)/config.nominal_window_alert_rate)-1,
                control_budget_resolved=rank<=len(values),
                validation_window_alert_rate=((np.asarray(window_scores[(step,key)])>bound[None]).any(-1).mean(0).tolist() if values else None))
        n=len({str(r['run_id']) for r in validation if r['step']==step})
        g.update(supported=any(c['supported'] for c in g['contexts'].values()),
            train_runs=len({str(r['run_id']) for r in train if r['step']==step}),validation_runs=n,
            scales=np.ones((len(g['sensors']),len(OPERATIONS))).tolist(),window_threshold=None,ordinal_edges=None,
            insufficient_controls_for_nominal_rate=any(c['insufficient_tail_support'] for c in g['contexts'].values()),
            control_rate_resolution=1/n if n else None)
    return {'schema':SCHEMA,'identity':identity,'config':asdict(config),'operations':list(OPERATIONS),'groups':groups,
        'gate_contract':'position_rank_v4','test_used_for_fitting':False,'normal_labels_verified':False,
        'active_operations':list(active_operations(config)),
        'supported_kinds':list(CORE_KINDS) if config.operation_profile=='core' else [*CORE_KINDS,'stuck','delay','relationship'],
        'threshold_source':'TRAIN cohort/position/sensor/operation rank + explicit tail excess; same-context Validation per-center self/pair run maxima',
        'population_false_alarm_guarantee':False,'validation_audit_used_for_selection':False,
        'meaning':'experimental conditional evidence, not a learned detector or certified fault probability'}


def convert(trace,context):
    bank=np.asarray(context['bank']);scale=np.asarray(context['tail_scales']);out=np.empty_like(trace,dtype=float)
    for s in range(trace.shape[1]):
        for op in range(trace.shape[2]):out[:,s,op]=rank_score(trace[:,s,op],bank[:,s,op],scale[s,op])
    return out


def analyze_position(policy,current,reference,residual,step,sensors,*,selected=None,scopes=None,center_sensor=None,cohort=None,position=None):
    g=policy.payload['groups'].get(step)
    if g is None or g['sensors']!=list(sensors):raise ValueError('Window policy sensor/Step mismatch')
    selected=list(sensors) if selected is None else list(selected);scopes=['whole','early','late'] if scopes is None else list(scopes)
    if not selected or len(set(selected))!=len(selected) or any(s not in sensors for s in selected):raise ValueError('Known unique sensors required')
    if not scopes or len(set(scopes))!=len(scopes) or any(s not in ('whole','early','late') for s in scopes):raise ValueError('Known scopes required')
    if center_sensor is not None and selected!=[center_sensor]:raise ValueError('Select exactly the UI center')
    key=context_key(cohort,position) if cohort is not None and position is not None else None
    c=g['contexts'].get(key);supported=bool(c and c['supported'])
    trace,aux=operation_traces(current,reference,residual,policy.config);n=len(trace)
    values=convert(trace,c) if c else np.zeros_like(trace);rows=[]
    for sensor in selected:
        sid=list(sensors).index(sensor)
        for scope in scopes:
            lo,hi=(0,n) if scope=='whole' else (0,max(2,n//3)) if scope=='early' else (2*n//3,n)
            for op,name in enumerate(OPERATIONS):
                if name not in active_operations(policy.config):continue
                relation=int(name=='relationship');threshold=c['thresholds'][sid][relation] if c and c['thresholds'] is not None else None
                v=values[lo:hi,sid,op];peak=int(v.argmax())+lo;score=float(v.max())
                changed=bool(supported and score>threshold)
                minimum=1 if name=='local_change' else policy.config.min_duration
                intervals=merged_intervals(v>threshold,0 if minimum==1 else policy.config.merge_gap,minimum) if changed else []
                intervals=[(a+lo,b+lo) for a,b in intervals]
                detected=next((list(iv) for iv in intervals if iv[0]<=peak<iv[1]),None)
                partner_scores={}
                if name=='relationship' and c:
                    bank=np.asarray(c['bank'])[:,sid,op];scale=c['tail_scales'][sid][op]
                    partner_scores={other:float(rank_score(aux['relationship'][lo:hi,sid,j],bank,scale).max()) for j,other in enumerate(sensors) if j!=sid}
                partner=max(partner_scores,key=partner_scores.get) if partner_scores else None
                rows.append(dict(sensor=sensor,time_scope=scope,scope_rows=[lo,hi],statistic=name,statistic_index=op,
                    normalized_score=score,window_threshold=threshold,raw_score=float(trace[lo:hi,sid,op].max()),
                    normalization_scale=None,threshold_multiplier=1.,unrelaxed_threshold=threshold,
                    active_gate='position rank v4',context_key=key,changed=changed,detected_rows=detected,
                    related_sensors=[sensor,partner] if partner else [],related_sensor_scores=partner_scores,
                    estimated_lag_rows=int(aux['lag_rows'][peak,sid]) if name=='delay' else None,
                    ordinal_index=int(np.sum(score>np.asarray(c['ordinal_edges'][sid][relation]))) if changed else 0,
                    pointwise=dict(peak_row=peak,scope_rows=[lo,hi],exceedance_intervals=intervals,
                        threshold_normalized=threshold,local_width=aux['local_width'],coordinate_system='prepared window row; half-open; not seconds'),
                    localization_basis='same-context TRAIN rank/tail transform of '+name+'; not learned attention',
                    score_meaning='empirical rank plus above-TRAIN-support excess; NOT a probability'))
    rows.sort(key=lambda r:-r['normalized_score']);events=[r for r in rows if r['changed']]
    return dict(ranked_evidence=rows,events=events,winner=events[0] if events else None,accepted=supported,
        disabled_operations=[name for name in OPERATIONS if name not in active_operations(policy.config)],
        changed=bool(events) if supported else None,maximum_score=rows[0]['normalized_score'],
        change_decision_source='position-conditioned empirical rank v4; no legacy fallback',
        calibration=dict(context_key=key,supported=supported,train_runs=c['train_runs'] if c else 0,
            validation_runs=c['validation_runs'] if c else 0,
            insufficient_controls_for_nominal_rate=c['insufficient_tail_support'] if c else True,
            unsupported_reason=None if supported else 'missing_or_insufficient_same_context_runs'))
