"""Blind discovery evaluation; intervention truth is used ONLY after inference."""
import time,os
from pathlib import Path
import numpy as np
import pandas as pd
from .discovery import review_changes,CONTRACT,contract
from .evaluation import load_cached_pairs,intervention_cases,interval_iou
from .reporting import ExperimentReport,ready_runtime,_metadata
from .execution_contract import execution_contract


def interval_union_iou(intervals,truth):
    """Half-open row intervals; merge duplicates before scoring fragmentation."""
    if truth is None or truth[1]<=truth[0]:return 0.
    merged=[]
    for lo,hi in sorted((int(a),int(b)) for a,b in intervals if b>a):
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    overlap=sum(max(0,min(hi,truth[1])-max(lo,truth[0])) for lo,hi in merged)
    union=sum(hi-lo for lo,hi in merged)+truth[1]-truth[0]-overlap
    return overlap/union if union else 0.


def localization_scores(result,target,truth_interval):
    """Scoring ONLY, never detector input. Keep old best-target diagnostic explicit."""
    events=[e for e in result['events'] if e['sensor']==target or e.get('partner_sensor')==target]
    intervals=[iv for e in events for iv in e.get('localization_intervals',e['pointwise'].get('exceedance_intervals',[]))]
    winner=result['winner'];predicted=winner.get('detected_rows') if winner else None
    winner_iou=interval_iou(predicted,truth_interval) if winner and (winner['sensor']==target or winner.get('partner_sensor')==target) and predicted else 0.
    numeric=winner.get('numeric_detected_rows',predicted) if winner else None
    correct=bool(winner and (winner['sensor']==target or winner.get('partner_sensor')==target))
    unavailable=result.get('interval_source','').startswith('unavailable:')
    return {'numeric_winner_interval_iou':interval_iou(numeric,truth_interval) if correct and numeric else 0.,
            'localization_status':'NOT_PREPARED' if unavailable else 'EVALUATED',
            'localization_source':winner.get('localization_basis','numeric') if winner else 'no alert',
            'target_interval_iou':np.nan if unavailable else max((interval_iou(iv,truth_interval) for iv in intervals),default=0.),
            'target_union_iou':np.nan if unavailable else interval_union_iou(intervals,truth_interval),
            'winner_interval_iou':np.nan if unavailable else winner_iou,'winner_interval_available':bool(predicted)}


def localization_contract():
    return pd.DataFrame([
        {'metric':'mean_winner_interval_iou','meaning':'single displayed winner; wrong sensor or absent interval scores zero','attention_metric':False},
        {'metric':'mean_target_union_iou','meaning':'union of all detected intervals on true target sensor; diagnostic, not single answer','attention_metric':False},
        {'metric':'mean_target_interval_iou','meaning':'historical BEST-overlap interval on true target sensor; optimistic diagnostic, not displayed answer','attention_metric':False},
        {'metric':'raw_attention_sensor_accuracy','meaning':'unfiltered learned top sensor versus injected sensor; independent of numerical candidate filtering','attention_metric':True},
    ])


def summarize_discovery(details,group_by=('kind',)):
    frame=pd.DataFrame(details);rows=[]
    groups=frame.groupby(list(group_by),sort=False,dropna=False) if group_by else [('all',frame)]
    for key,group in groups:
        if not isinstance(key,tuple):key=(key,)
        controls=group[~group.injected];injected=group[group.injected & group.effective.eq(True)]
        # Unsupported positives remain misses; unsupported controls are unknown,
        # not true negatives. These are synthetic-intervention task metrics.
        tp=int(injected.alert.eq(True).sum());fn=len(injected)-tp
        supported_controls=controls[controls.accepted.eq(True)]
        fp=int(supported_controls.alert.eq(True).sum());tn=len(supported_controls)-fp
        ratio=lambda a,b:a/b if b else np.nan
        row=dict(zip(group_by,key)) if group_by else {}
        row.update(tp=tp,fn=fn,fp=fp,tn=tn,
            precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
            f1=ratio(2*tp,2*tp+fp+fn),
            unsupported_control_cases=len(controls)-len(supported_controls),
            metric_target='synthetic_effective_change_vs_untouched')
        if 'analysis_mode' in group:row['analysis_mode']=group.analysis_mode.iloc[0]
        if 'task' in group:row['task']=group.task.iloc[0]
        row.update(windows=len(group),effective_cases=len(injected),accepted_share=float(group.accepted.mean()),
            unsupported_effective_cases=int((~injected.accepted).sum()),
            detection_recall_when_accepted=(float(injected.loc[injected.accepted,'alert'].mean()) if injected.accepted.any() else np.nan),
            evaluation_status='EVALUATED' if len(injected) else 'NOT_EVALUATED_NO_EFFECTIVE_INJECTIONS',
            detection_recall=float(injected.alert.mean()) if len(injected) else np.nan,
            target_detection_recall=float(injected.target_alert.mean()) if len(injected) else np.nan,
            new_target_detection_recall=float(injected.new_target_alert.mean()) if len(injected) else np.nan,
            top_sensor_accuracy=float(injected.top_sensor_correct.mean()) if len(injected) else np.nan,
            mean_target_interval_iou=float(injected.target_interval_iou.mean()) if len(injected) else np.nan,
            control_windows=len(controls),untouched_alert_rate=float(controls.alert.mean()) if len(controls) else np.nan,
            typed_consistency=float(group.typed_consistent.mean()),
            mean_ms_per_window=float(group.latency_ms.mean()))
        for field in ('winner_interval_iou','target_union_iou'):
            if field in injected:row['mean_'+field]=float(injected[field].mean()) if len(injected) else np.nan
        if 'numeric_winner_interval_iou' in injected:
            row['mean_numeric_winner_interval_iou']=float(injected.numeric_winner_interval_iou.mean()) if len(injected) else np.nan
            if 'winner_interval_iou' in injected:
                comparable=injected.dropna(subset=['numeric_winner_interval_iou','winner_interval_iou'])
                row['paired_interval_iou_gain']=float((comparable.winner_interval_iou-comparable.numeric_winner_interval_iou).mean()) if len(comparable) else np.nan
                row['paired_interval_cases']=len(comparable)
        if 'winner_interval_iou' in injected:
            detected=injected[injected.alert.eq(True)]
            row['winner_interval_iou_when_alerted']=float(detected.winner_interval_iou.mean()) if len(detected) else np.nan
        if 'winner_interval_available' in injected:
            row['winner_interval_coverage']=float(injected.winner_interval_available.mean()) if len(injected) else np.nan
        if 'related_sensor_hit' in injected:
            pairs=injected[injected.kind.eq('relationship')]
            row['relationship_pair_member_recall']=float(pairs.related_sensor_hit.mean()) if len(pairs) else np.nan
        if 'raw_attention_sensor_correct' in injected:
            row['raw_attention_sensor_accuracy']=float(injected.raw_attention_sensor_correct.mean()) if len(injected) else np.nan
        if 'attention_correct_but_numeric_gate_rejected' in injected:
            row['attention_correct_but_gate_rejected_cases']=int(injected.attention_correct_but_numeric_gate_rejected.sum())
        if 'numeric_only_sensor_correct' in injected:
            row['numeric_only_sensor_accuracy']=float(injected.numeric_only_sensor_correct.mean()) if len(injected) else np.nan
            row['sensor_accuracy_gain_over_numeric']=row['top_sensor_accuracy']-row['numeric_only_sensor_accuracy']
        if 'no_sensor_when_unchanged' in group:
            unchanged=group[group.accepted & ~group.alert]
            row['no_sensor_when_unchanged']=float(unchanged.no_sensor_when_unchanged.mean()) if len(unchanged) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_discovery_experiment(root,*,runtime=None,steps=None,pairs_per_step=2,split='test',
                             kinds=None,
                             strengths=(.5,2.),max_cases=7,seed=42,evaluate_centers=False,analysis_mode='auto'):
    """V14 evaluates UI centers; explicit blind mode retains historical full scans."""
    if split not in ('validation','test'):raise ValueError('Held-out split required')
    if max_cases<1:raise ValueError('Positive bounded case count required')
    started=time.perf_counter();runtime=runtime or ready_runtime(root)
    selected=list(runtime.encoders) if steps is None else list(steps)
    if not selected or any(s not in runtime.encoders for s in selected):raise ValueError('Known trained Steps required')
    module=runtime.module
    policy=getattr(module,'DISCOVERY_POLICY',None)
    core=getattr(getattr(policy,'config',None),'operation_profile','all')=='core'
    supported_kinds=('level_shift','drift','spike','variance') if core else ('level_shift','drift','spike','variance','stuck','delay','relationship')
    kinds=supported_kinds if kinds is None else tuple(kinds)
    if any(kind not in supported_kinds for kind in kinds):raise ValueError('Requested change kind is disabled by the active operation profile')
    from .window_policy import MODE
    calibrated=getattr(module,'DISCOVERY_POLICY_MODE',None)==MODE
    if analysis_mode not in ('auto','center','blind'):raise ValueError('Unknown analysis mode')
    if analysis_mode=='auto':
        analysis_mode='center' if calibrated and os.environ.get('GEVIS_DISCOVERY_TRAINING')=='center_relation_v14' else 'blind'
    if calibrated and getattr(module,'DISCOVERY_POLICY',None) is None:
        from .discovery_calibration import policy_path
        selected_path=getattr(module,'DISCOVERY_POLICY_PATH',None)
        if selected_path is None:
            selected_path=policy_path(module.BACKBONE_PATH) if hasattr(module,'BACKBONE_PATH') else '<unknown runtime path>'
        raise RuntimeError(f'Discovery policy was not loaded: {selected_path}. '
            'Run STEP 3 window-evidence-build (prepare_comparison), then rerun this cell '
            'with a fresh runtime. Do not retrain STEP 1/2.')
    # Factory/adapters always receive the runtime's exact encoder metadata.
    if not hasattr(module,'STEP_ENCODERS'): raise RuntimeError('Runtime missing STEP_ENCODERS')
    if analysis_mode=='center':
        if not calibrated:raise ValueError('Center evaluation requires calibrated UI runtime')
        return _run_center_experiment(root,runtime,selected,pairs_per_step,split,kinds,strengths,max_cases,seed,started)
    records=[];sampling=[];center_records=[]
    for index,step in enumerate(selected):
        meta=runtime.encoders[step][1];sensors=meta['sensor_columns']
        if calibrated:
            from .frozen_transfer import load_discovery_pairs
            pairs=load_discovery_pairs(runtime,step,split=split,limit=pairs_per_step,seed=seed)
        else:pairs=load_cached_pairs(runtime,step,split=split,limit=pairs_per_step,seed=seed)
        sampling.append({'step':step,'pairs':len(pairs),'split':split})
        for p,(current,reference) in enumerate(pairs):
            print(f'Blind discovery Step {index+1}/{len(selected)}, pair {p+1}/{len(pairs)}',flush=True)
            control=None
            for spec,frame,truth in intervention_cases(current,sensors,np.asarray(meta['std']),kinds=kinds,
                    strengths=strengths,max_cases=max_cases,seed=seed+p):
                # No spec, truth, expected sensor or expected scope enters this call.
                result=review_changes(module,frame,reference,step,module.ReferenceInfo('held-out aligned reference',1))
                if spec is None:control=result
                target=sensors[spec.sensor] if spec is not None else None
                if evaluate_centers and calibrated:
                    # Select before consulting injection truth, never select the
                    # injected sensor to make a center-local result look correct.
                    center=sensors[p%len(sensors)]
                    focused=review_changes(module,frame,reference,step,module.ReferenceInfo('held-out aligned reference',1),center_sensor=center)
                    center_records.append({'step':step,'center_sensor':center,'injected':spec is not None,
                        'effective':truth['effective'] if truth else None,
                        'direct_center_injection':bool(spec and target==center),
                        'self_alert':focused['self_analysis']['changed'],
                        'relationship_alert':focused['relationship_analysis']['changed'],
                        'related_sensors':focused['relationship_analysis']['related_sensors'],
                        'partner_recovery':bool(spec and target!=center and target in focused['relationship_analysis']['related_sensors']),
                        'meaning':'center selected independently of intervention; untouched is unlabelled; no causal truth'})
                events=result['events'];winner=result['winner']
                numeric_winner=max(events,key=lambda e:e['normalized_score']) if result['accepted'] and events else None
                target_events=[e for e in events if e['sensor']==target]
                old_target=[e for e in control['events'] if e['sensor']==target] if control else []
                localization=(localization_scores(result,target,truth['event_interval']) if truth and truth['effective'] else
                    {k:None for k in ('target_interval_iou','target_union_iou','winner_interval_iou','winner_interval_available')})
                answers={a['answer_type']:a['answer'] for a in result['answers']}
                consistent=(result['changed'] is None and all(v is None for v in answers.values())) or (
                    result['changed'] is False and answers['CHOICE']=='No clear change' and answers['ORDINAL']=='No change') or (
                    result['changed'] is True and answers['CHOICE']!='No clear change' and answers['ORDINAL']!='No change')
                records.append({'step':step,'split':split,'pair_id':current.attrs.get('evaluation_pair_id',str(p)),
                    'kind':spec.kind if spec else 'untouched','injected':spec is not None,
                    'strength':spec.strength if spec else None,
                    'evaluation_regime':'in_distribution_kind' if spec and spec.kind in ('level_shift','drift','spike','variance') else 'untrained_kind' if spec else 'unlabelled_control',
                    'effective':truth['effective'] if truth else None,'accepted':result['accepted'],
                    'alert':result['changed'] is True,'target_alert':bool(target_events) if spec else None,
                    'new_target_alert':bool(target_events) and not old_target if spec else None,
                    'top_sensor_correct':winner['sensor']==target if winner and spec else False if spec else None,
                    'raw_attention_sensor_correct':result.get('attention_top_sensor')==target if spec else None,
                    'numeric_only_sensor_correct':bool(numeric_winner and numeric_winner['sensor']==target) if spec else None,
                    'no_sensor_when_unchanged':winner is None if result['changed'] is False else None,
                    **localization,'typed_consistent':bool(consistent),
                    'criterion_coverage':result['criterion_coverage'],'latency_ms':result['latency_ms'],
                    'expected_sensor':target,'predicted_sensor':winner['sensor'] if winner else None,
                    'related_sensor_hit':bool(winner and target in winner.get('related_sensors',[])) if spec else None,
                    'answers':answers,'events':events,'missing_criteria':result['missing_criteria'],
                    'winner':winner,'event_truth_used_for_inference':False,
                    'discovery_contract':result['contract'],'warnings':result['warnings'],
                    **_gate_diagnostics(result,target),
                    'calibration':result.get('calibration')})
    details=pd.DataFrame(records)
    if details.empty:raise ValueError('No discovery evaluation windows')
    # Overall alert rate is deliberately distinct from target-localized detection.
    summary=summarize_discovery(details,())
    by_step=summarize_discovery(details,('step',))
    by_kind=summarize_discovery(details[details.injected],('kind',))
    controls=details[~details.injected]
    by_kind['control_windows']=len(controls)
    by_kind['untouched_alert_rate']=float(controls.alert.mean()) if len(controls) else np.nan
    errors=details[details.injected & details.effective.eq(True) & (~details.target_alert.eq(True) | ~details.top_sensor_correct.eq(True))]
    diagnostics={'by_kind':by_kind,'contract':pd.DataFrame([contract()]),'localization_metrics':localization_contract()}
    if center_records:
        diagnostics['center_analysis']=pd.DataFrame(center_records)
    diagnostics['by_kind_and_strength']=summarize_discovery(details[details.injected],('kind','strength'))
    diagnostics['by_training_regime']=summarize_discovery(details[details.injected],('evaluation_regime',))
    if calibrated:
        from .discovery_calibration import calibration_summary
        diagnostics['window_calibration']=calibration_summary(module.DISCOVERY_POLICY)
        diagnostics['contract']=pd.DataFrame([{'schema':result['contract'],
            'threshold_source':module.DISCOVERY_POLICY.payload['threshold_source'],
            'reference_pool':'TRAIN only; same protocol for calibration and Test',
            'localization':'operation-specific rolling traces; not attention','test_used_for_fitting':False}])
    return ExperimentReport('Blind shared-evidence change discovery',summary,by_step,details,errors,
        diagnostics=diagnostics,
        metadata={**_metadata(root,runtime,started),'analysis_mode':'blind','execution_contract':execution_contract(runtime,'blind_discovery'),'discovery_contract':result['contract'],'split':split,
            'sampling':sampling,'truth_used_for_inference':False,'baseline_comparison':'separate EXP 3, not measured here'})


def _gate_diagnostics(result,target):
    """Scoring after inference: expose the numeric gate independently of attention."""
    accepted=result.get('accepted',True)  # historical diagnostic-only fixtures omit this field
    candidates=[r for r in result['ranked_evidence'] if r['sensor']==target]
    if result.get('center_sensor') and target!=result['center_sensor']:
        candidates=[{**r,'normalized_score':r['related_sensor_scores'][target],
                     'changed':bool(accepted and r.get('window_threshold') is not None and r['related_sensor_scores'][target]>r['window_threshold'])}
                    for r in result['ranked_evidence'] if target in r.get('related_sensor_scores',{})]
    # Center v3 has distinct self/pair bounds. Raw-score maximum alone can
    # report a rejected relation while another operation actually triggered.
    best=max(candidates,key=lambda r:(bool(r['changed']),r['normalized_score']-(r.get('window_threshold') if r.get('window_threshold') is not None else float('inf'))),default=None)
    score=best['normalized_score'] if best else np.nan
    threshold=best.get('window_threshold',1.) if best else np.nan
    return {'target_max_numeric_score':score,'target_numeric_threshold':threshold,
            'target_score':score,'target_threshold':threshold,
            'target_score_excess':score-threshold if threshold is not None else np.nan,
            'target_score_threshold_ratio':score/threshold if threshold is not None and threshold>0 else np.nan,
            'target_gate_status':('UNSUPPORTED_CONTEXT_OR_CRITERIA' if not accepted else
                'TARGET_NOT_ANALYZED' if not candidates else 'ALERT' if any(r['changed'] for r in candidates) else 'BELOW_THRESHOLD'),
            'attention_correct_but_numeric_gate_rejected':bool(accepted and target is not None and candidates and
                result.get('attention_top_sensor')==target and not any(r['changed'] for r in candidates)),
            'change_decision_source':result.get('change_decision_source','calibrated numerical evidence; attention ranks candidates')}


def _run_center_experiment(root,runtime,steps,pairs_per_step,split,kinds,strengths,max_cases,seed,started):
    """UI task first. Select center and perturbation coordinates BEFORE generation.

    Self changes and context interventions are different tasks. A selected center
    makes center identification given input, so never report it as learned accuracy.
    Context interventions do not automatically establish a true pair change.
    """
    from .frozen_transfer import load_discovery_pairs
    module=runtime.module;rows=[];sampling=[];operation_rows=[]
    for step in steps:
        meta=runtime.encoders[step][1];sensors=list(meta['sensor_columns'])
        pairs=load_discovery_pairs(runtime,step,split=split,limit=pairs_per_step,seed=seed)
        sampling.append({'step':step,'pairs':len(pairs),'split':split})
        for p,(current,reference) in enumerate(pairs):
            center=sensors[p%len(sensors)]
            before_evidence=None
            designs=[('self',center)]
            if len(sensors)>1 and getattr(module.DISCOVERY_POLICY.config,'operation_profile','all')!='core':
                designs.append(('context',sensors[(p+1)%len(sensors)]))
            for task,injected_sensor in designs:
                for spec,frame,truth in intervention_cases(current,sensors,np.asarray(meta['std']),
                        kinds=kinds,strengths=strengths,max_cases=max_cases,seed=seed+p,target_sensor=injected_sensor):
                    if task=='context' and spec is None:continue
                    result=review_changes(module,frame,reference,step,module.ReferenceInfo('held-out aligned reference',1),
                                          center_sensor=center)
                    if task=='self' and spec is None:
                        before_evidence=result.get('ranked_evidence',[])
                    if task=='self' and spec is not None:
                        operation_rows.extend(paired_operation_scores(before_evidence,result.get('ranked_evidence',[]),
                            step=step,pair_id=current.attrs.get('evaluation_pair_id',str(p)),center=center,
                            kind=spec.kind,strength=spec.strength,effective=bool(truth['effective'])))
                    own=result['self_analysis'];relation=result['relationship_analysis']
                    partner_answer=result['answers'][-1];target=injected_sensor if spec else None
                    view=own if task=='self' else relation
                    events=view['events'];winner=view['winner']
                    self_result={**result,'events':events,'winner':winner}
                    loc=localization_scores(self_result,target,truth['event_interval']) if truth and truth['effective'] else {
                        k:None for k in ('target_interval_iou','target_union_iou','winner_interval_iou','winner_interval_available')}
                    rows.append({'step':step,'split':split,'pair_id':current.attrs.get('evaluation_pair_id',str(p)),
                        'analysis_mode':'center','task':task,'center_sensor':center,'expected_sensor':target,
                        'kind':spec.kind if spec else 'untouched','strength':spec.strength if spec else None,
                        'injected':spec is not None,'effective':truth['effective'] if truth else None,
                        'ineffective_reason':truth.get('ineffective_reason') if truth else None,
                        'relationship_evaluable':truth.get('relationship_evaluable') if truth else None,
                        'accepted':view['accepted'],'alert':view['changed'] is True,
                        'unrelaxed_alert':(bool(any(r['normalized_score']>r['unrelaxed_threshold'] for r in result['ranked_evidence']
                            if (r['statistic']=='relationship')==(task=='context'))) if view['accepted'] and
                            all('unrelaxed_threshold' in r for r in result['ranked_evidence']) else None),
                        'ui_combined_alert':result['changed'] is True,
                        'criterion_coverage':result['criterion_coverage'],
                        'predicted_sensor':(winner.get('partner_sensor') or winner['sensor']) if winner else None,
                        'target_alert':bool(events) if spec and task=='self' else None,
                        'new_target_alert':None,'top_sensor_correct':np.nan,'numeric_only_sensor_correct':np.nan,
                        'raw_attention_sensor_correct':result.get('attention_top_sensor')==target if spec else None,
                        'partner_top1_correct':partner_answer['answer']==target if spec and task=='context' else None,
                        'partner_recovery':target in relation['related_sensors'] if spec and task=='context' else None,
                        'self_alert':own['changed'],'relationship_alert':relation['changed'],
                        'related_sensors':relation['related_sensors'],'winner':winner,'events':events,
                        'typed_consistent':all(a['evidence'] is result['winner'] for a in result['answers'][:3]),
                        'no_sensor_when_unchanged':winner is None if view['changed'] is False else None,
                        'latency_ms':result['latency_ms'],'answers':{a['answer_type']:a['answer'] for a in result['answers']},
                        'event_truth_used_for_inference':False,**loc,**_gate_diagnostics(result,target)})
    details=pd.DataFrame(rows)
    if details.empty:raise ValueError('No centered evaluation windows')
    own=details[details.task.eq('self')];context=details[details.task.eq('context')]
    summary=summarize_discovery(own,());by_step=summarize_discovery(own,('step',))
    diagnostics={'center_analysis':details,'by_kind':summarize_discovery(own[own.injected],('kind',)),
        'by_kind_and_strength':summarize_discovery(own[own.injected],('kind','strength')),
        'gate_bottlenecks':details[details.injected][['step','task','kind','strength','center_sensor','expected_sensor',
            'raw_attention_sensor_correct','target_score','target_threshold','target_score_excess','change_decision_source',
            'target_score_threshold_ratio','attention_correct_but_numeric_gate_rejected']],
        'context_interventions':context,
        'intervention_validity':details[details.injected].groupby(['step','task','kind'],sort=False).agg(
            generated=('effective','size'),numeric_effective=('effective',lambda v:int(v.eq(True).sum())),
            flat_target_noops=('ineffective_reason',lambda v:int(v.eq('constant_target_segment').sum())),
            relationship_evaluable=('relationship_evaluable',lambda v:int(v.eq(True).sum()))).reset_index(),
        'localization_metrics':localization_contract(),
        'contract':pd.DataFrame([{'analysis_mode':'center','headline_task':'selected sensor self-change',
            'sensor_accuracy':'not applicable: center is UI input; see context partner ranking',
            'context_truth':'changed context sensor, not a verified changed relationship',
            'change_decision_source':result.get('change_decision_source','fixed calibrated numeric gate; STEP2 localizes, does not train detection')}])}
    if not context.empty:
        effective=context[context.effective.eq(True)]
        diagnostics['partner_ranking']=effective.groupby(['step','kind'],sort=False).agg(
            cases=('partner_top1_correct','size'),partner_top1_accuracy=('partner_top1_correct','mean'),
            partner_recovery=('partner_recovery','mean'),relationship_alert_rate=('alert','mean')).reset_index()
    if getattr(getattr(module.DISCOVERY_POLICY,'config',None),'operation_profile','all')!='core':
        diagnostics.update(dynamic_challenges(runtime,steps,split,pairs_per_step,seed))
    diagnostics['operation_score_changes']=pd.DataFrame(operation_rows)
    diagnostics['paired_change_diagnosis']=paired_change_diagnosis(diagnostics['operation_score_changes'])
    paired=diagnostics['paired_change_diagnosis']
    diagnostics['paired_change_summary']=(paired.groupby(['step','diagnosis'],sort=False).size().rename('cases').reset_index()
        if not paired.empty else pd.DataFrame())
    challenge=diagnostics.get('dynamic_challenge',pd.DataFrame()).copy()
    if not challenge.empty:
        challenge['support_status']=np.where(challenge.effective_cases.eq(0),'NOT_VALIDATED_NO_CASES',
            np.where(challenge.mechanism_recall.fillna(0).eq(0),'EXPERIMENTAL_NO_MECHANISM_RECOVERY','EXPERIMENTAL_REQUIRES_VALIDATION'))
    diagnostics['experimental_support']=challenge
    errors=own[own.injected & own.effective.eq(True) & ~own.alert]
    return ExperimentReport('UI-centered self and relationship evaluation',summary,by_step,details,errors,
        diagnostics=diagnostics,metadata={**_metadata(root,runtime,started),'analysis_mode':'center',
            'execution_contract':execution_contract(runtime,'center_discovery'),'discovery_contract':result['contract'],
            'split':split,'sampling':sampling,'truth_used_for_inference':False,
            'center_selection':'pair index before generation; independent of model predictions',
            'headline_metrics':'self task only; context task reported separately',
            'operation_profile':getattr(module.DISCOVERY_POLICY.config,'operation_profile','all'),
            'evaluated_kinds':list(kinds),'scope_change_not_model_performance_gain':True})


def paired_operation_scores(before,after,*,step,pair_id,center,kind,strength,effective):
    """Matched operation deltas AFTER inference; no truth goes to the detector.

    Compare the same operation/scope, never subtract different winning scores.
    Scores diagnose the numerical path, NOT latent representation quality.
    """
    if before is None:return []
    lookup={(e['sensor'],e['statistic'],e['time_scope']):e for e in before}
    rows=[]
    for event in after:
        if event['sensor']!=center or event['time_scope']!='whole':continue
        old=lookup.get((center,event['statistic'],'whole'))
        if old is None:continue
        threshold=event.get('window_threshold')
        score=event['normalized_score'];previous=old['normalized_score']
        route_matches=(event.get('active_gate')==old.get('active_gate') and
            threshold==old.get('window_threshold') and event.get('normalization_scale')==old.get('normalization_scale'))
        status=('NOT_EVALUATED_NUMERIC_NOOP' if not effective else
            'INCOMPARABLE_ROUTE_CHANGED' if not route_matches else
            'ALREADY_ABOVE_THRESHOLD' if old['changed'] else
            'CROSSED_THRESHOLD' if event['changed'] else
            'SCORE_INCREASED_BELOW_THRESHOLD' if score>previous else 'NO_SCORE_INCREASE')
        rows.append(dict(step=step,pair_id=pair_id,center_sensor=center,kind=kind,strength=strength,
            effective=effective,operation=event['statistic'],before_score=previous,after_score=score,
            score_delta=score-previous,threshold=threshold,
            before_alert=bool(old['changed']),after_alert=bool(event['changed']),
            raw_before_score=old.get('raw_score'),raw_after_score=event.get('raw_score'),
            normalization_scale=event.get('normalization_scale'),active_gate=event.get('active_gate'),
            threshold_multiplier=event.get('threshold_multiplier',1.),unrelaxed_threshold=event.get('unrelaxed_threshold',threshold),
            diagnosis=status,meaning='paired numerical operation response; not learned detector accuracy'))
    return rows


def paired_change_diagnosis(operations):
    """One row per intervention, preserving baseline alerts and no-op coverage."""
    if operations.empty:return pd.DataFrame()
    rows=[]
    keys=['step','pair_id','center_sensor','kind','strength']
    for key,group in operations.groupby(keys,sort=False,dropna=False):
        own=group[group.operation.ne('relationship')]
        relation=group[group.operation.eq('relationship')]
        effective=bool(group.effective.iloc[0]);before=bool(own.before_alert.any());after=bool(own.after_alert.any())
        increased=bool(own.score_delta.gt(0).any())
        diagnosis=('NOT_EVALUATED_NUMERIC_NOOP' if not effective else
            'INCOMPARABLE_ROUTE_CHANGED' if group.diagnosis.eq('INCOMPARABLE_ROUTE_CHANGED').any() else
            'ALREADY_ALERTING_BEFORE_INJECTION' if before else 'NEW_ALERT_AFTER_INJECTION' if after else
            'SCORE_RESPONSE_BLOCKED_BY_GATE' if increased else 'NO_NUMERICAL_SCORE_INCREASE')
        rows.append(dict(zip(keys,key),effective=effective,before_alert=before,after_alert=after,
            relationship_before_alert=bool(relation.before_alert.any()),relationship_after_alert=bool(relation.after_alert.any()),
            any_operation_score_increased=increased,diagnosis=diagnosis,
            active_gate=group.active_gate.iloc[0],
            warning='No score increase does not identify PatchTST/Cross-Attention as the cause'))
    return pd.DataFrame(rows)


def dynamic_challenges(runtime,steps,split,pairs_per_step,seed):
    """Separate observed-varying-input challenge, NEVER replace fixed UI cases.

    Center selection uses input activity before injection, not predictions or
    successful outcomes. Full-interval perturbations avoid random flat subspans.
    No synthetic waveform is substituted for confidential real observations.
    """
    from .frozen_transfer import load_discovery_pairs
    from .injection import Injection,inject
    module=runtime.module;rows=[];controls=[]
    for step in steps:
        meta=runtime.encoders[step][1];names=list(meta['sensor_columns'])
        pairs=load_discovery_pairs(runtime,step,split=split,limit=pairs_per_step,seed=seed)
        for index,(current,reference) in enumerate(pairs):
            values=current[names].to_numpy(float);active=np.flatnonzero(np.ptp(values,axis=0)>0)
            pair=current.attrs.get('evaluation_pair_id',str(index))
            if len(active):
                sid=int(active[index%len(active)]);center=names[sid]
                original=review_changes(module,current,reference,step,module.ReferenceInfo('held-out aligned TRAIN reference',1),center_sensor=center)
                controls.append({'step':step,'pair_id':pair,'center_sensor':center,'untouched_alert':original['changed'] is True})
            for kind in ('stuck','delay','relationship'):
                # Stuck amplitude is not a strength parameter; don't duplicate
                # the same full-interval intervention under two strength labels.
                for strength in ((2.,) if kind=='stuck' else (.5,2.)):
                    base={'step':step,'pair_id':pair,'kind':kind,'strength':strength,
                        'evaluation_design':'varying_input_full_interval_v1','case_source':'observed held-out input; center selected before injection'}
                    if not len(active) or (kind=='relationship' and len(active)<2):
                        rows.append(dict(base,status='NOT_EVALUATED_NO_VARYING_SIGNALS',generated=False,effective=False,hit=False));continue
                    changed,truth=inject(values,Injection(kind,sid,0,len(values),strength,seed+index),np.asarray(meta['std']))
                    frame=current.astype({name:'float64' for name in names});frame.loc[:,names]=changed
                    result=review_changes(module,frame,reference,step,module.ReferenceInfo('held-out aligned TRAIN reference',1),center_sensor=center)
                    applicable=truth['effective'] and (kind!='relationship' or truth['relationship_evaluable'])
                    view=result['relationship_analysis'] if kind=='relationship' else result['self_analysis']
                    events=view.get('events')
                    mechanism_hit=(any(e.get('statistic')==kind for e in events) if events is not None else None)
                    rows.append(dict(base,center_sensor=center,generated=True,effective=bool(applicable),
                        numeric_effective=truth['effective'],relationship_evaluable=truth['relationship_evaluable'],
                        status='EVALUATED' if applicable else 'NOT_EVALUATED_NO_PAIR_CHANGE' if truth['effective'] else 'NOT_EVALUATED_NUMERIC_NOOP',
                        hit=bool(view['changed']) if applicable else False,
                        mechanism_hit=mechanism_hit if applicable else None,
                        actual_lag_rows=min(len(values)-1,max(1,round(strength))) if kind=='delay' else None))
    details=pd.DataFrame(rows);summary=[]
    if not details.empty:
        for key,group in details.groupby(['step','kind','strength'],sort=False):
            effective=group[group.effective.eq(True)]
            summary.append(dict(zip(('step','kind','strength'),key),generated=int(group.generated.sum()),
                effective_cases=len(effective),detection_recall=float(effective.hit.mean()) if len(effective) else np.nan,
                mechanism_recall=(float(effective.mechanism_hit.dropna().mean())
                    if len(effective) and 'mechanism_hit' in effective and effective.mechanism_hit.notna().any() else np.nan),
                mechanism_meaning='matching operation event, not attention accuracy or physical cause',
                status='EVALUATED' if len(effective) else 'NOT_EVALUATED',
                evaluation_design='varying_input_full_interval_v1'))
    return {'dynamic_challenge':pd.DataFrame(summary),'dynamic_challenge_details':details,
            'dynamic_challenge_controls':pd.DataFrame(controls)}
