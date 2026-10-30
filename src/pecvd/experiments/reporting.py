"""Run complete actual-model experiments and return explicit final reports.

No fixture fallback, automatic export or GEVIS training. EXP 4 explicitly prepares
external CSV inputs OFFLINE before frozen inference; other experiments use caches.
Merlion baseline fitting uses only the cached TRAIN split.
"""
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import os
import time
import numpy as np
import pandas as pd
from .evaluation import (load_experiment_runtime, load_cached_pairs, evaluate_questions,
                         summarize_questions, evaluate_injections, source_identity)
from .injection import KINDS


@dataclass
class ExperimentReport:
    name: str
    summary: pd.DataFrame
    by_step: pd.DataFrame = field(default_factory=pd.DataFrame)
    details: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    prerequisites: pd.DataFrame = field(default_factory=pd.DataFrame)
    integrations: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        # Evaluation-only extension; source-bound Fusion training stays unchanged.
        if self.metadata.get('decision_system') == 'fusion_answers':
            from .fusion_metrics import refresh_report
            refresh_report(self)
        # Translate presentation only, AFTER every prediction/metric is computed.
        # Internal trained criteria and source-bound inference code stay intact.
        from pecvd.answer_display import english_frame
        for key in ('summary','by_step','details','errors','prerequisites','integrations'):
            setattr(self,key,english_frame(getattr(self,key)))
        self.diagnostics={key:english_frame(value) for key,value in self.diagnostics.items()}


def ready_runtime(root):
    # Only local pretrained files; do not attempt network fallback from a notebook.
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    print('Loading frozen STEP 1/2 artifacts (original CSV is not read)...',flush=True)
    return load_experiment_runtime(root)


def _steps(runtime,steps):
    selected=list(runtime.encoders) if steps is None else list(steps)
    if not selected or any(step not in runtime.encoders for step in selected):
        raise ValueError('Select at least one trained Step')
    return selected


def predict_fusion_original(runtime, *, steps=None, pairs_per_step=2, seed=42):
    """Report frozen original-input answers without inventing accuracy.

    Keep output assembly outside source-bound training so serialization fixes do
    not invalidate trained checkpoints. The answer already contains its center.
    """
    from .discovery_calibration import load_discovery_pairs
    from .fusion_answers import require_head, review
    module = runtime.module
    require_head(module)
    rows = []
    for step in _steps(runtime, steps):
        names = list(runtime.encoders[step][1]['sensor_columns'])
        for i, (current, reference) in enumerate(load_discovery_pairs(
                runtime, step, split='test', limit=pairs_per_step, seed=seed)):
            center = names[i % len(names)]
            result = review(module, current, reference, step, None, center_sensor=center)
            rows.extend(dict(answer, step=step, pair_index=i, center_sensor=center)
                        for answer in result['answers'])
    details = pd.DataFrame(rows)
    return ExperimentReport('Original cross-run Fusion answers without independent truth',
        pd.DataFrame([dict(questions=len(details), accuracy_available=False,
            metric_target='unlabelled original cross-run predictions')]),
        details=details, metadata=dict(model_evaluated=True, decision_system='fusion_answers',
            rule_fallback=False, original_cross_run_fault_accuracy_available=False))


def ready_answer_runtime(root, runtime=None):
    """Require the same source-bound full GEVIS as EXP4, never a numeric fallback."""
    runtime=runtime or ready_runtime(root)
    model=runtime.module.GEVIS_MODEL
    override=os.environ.get('GEVIS_ANSWER_CHECKPOINT')
    if override or not getattr(model,'answer_criteria',None):
        from .frozen_transfer import load_frozen_answer_model
        path=Path(override or Path(runtime.module.BACKBONE_PATH).parent/'gevis_answer_model.pt')
        runtime.module.GEVIS_MODEL=load_frozen_answer_model(runtime,path)
    if not getattr(runtime.module.GEVIS_MODEL,'answer_criteria',None):
        raise ValueError('STEP3 checkpoint must declare its exact trained answer options')
    return runtime


def predict_window_answers(runtime,current,reference,step,*,centers,question_variants=None):
    """Encode once, then answer for explicitly selected centers; no truth enters."""
    import torch
    from pecvd.gevis_fast import answer_probabilities
    from pecvd.answer_display import QUESTIONS
    from .questions import center_descriptions
    m=runtime.module;model=m.GEVIS_MODEL.to(m.DEVICE).eval()
    options=getattr(model,'answer_criteria',None)
    if not options:raise ValueError('A trained full GEVIS checkpoint is required')
    variants=(list(question_variants) if question_variants is not None else
        [(k,k,QUESTIONS[k]) for k in ('BOOLEAN','CHOICE','ORDINAL')])
    if not variants or len(variants)>32 or any(len(v)!=3 or v[1] not in options for v in variants):
        raise ValueError('Bounded (question_id,answer_type,English text) variants required')
    encoder,meta=runtime.encoders[step];sensors=list(meta['sensor_columns'])
    centers=list(centers)
    if not centers or any(c not in sensors for c in centers):raise ValueError('Known center sensors required')
    rows=[];encoder.to(m.DEVICE).eval()
    try:
        a,_=m.prepare_window(current,meta);b,_=m.prepare_window(reference,meta)
        with torch.no_grad():state=m.numeric_state(encoder,a.to(m.DEVICE),b.to(m.DEVICE))
        patches=state.shape[1]//len(sensors)
        for center in centers:
            descriptions=m.numeric_token_descriptions(step,meta,patches,len(variants))
            descriptions=[center_descriptions(d,sensors,patches,center) for d in descriptions]
            kinds=[v[1] for v in variants];criteria=[list(options[k]) for k in kinds]
            probabilities=answer_probabilities(model,state.expand(len(variants),-1,-1),
                [v[2] for v in variants],kinds,criteria,descriptions)
            for (qid,kind,text),probability,opts in zip(variants,probabilities,criteria):
                values=probability.detach().cpu();index=int(values.argmax())
                rows.append(dict(center_sensor=center,question_id=qid,answer_type=kind,
                    question=text,answer=opts[index],probabilities=dict(zip(opts,map(float,values))),
                    max_model_probability=float(values.max()),
                    change_probability=float(values[1]) if kind=='BOOLEAN' else None,
                    alert=index==1 if kind=='BOOLEAN' else None,
                    score_source='trained_answer_decoder_softmax',
                    meaning='uncalibrated model class probabilities; not equipment-fault probability'))
                rows[-1]['question_scope']=('trained_fixed_prompt' if text==QUESTIONS.get(kind)
                                            else 'question_contrast_probe; semantics not validated')
    finally:encoder.cpu()
    return rows


def run_answer_experiment(root,*,runtime=None,steps=None,pairs_per_step=2,seed=42,
                          inject=False,kinds=None,strengths=(.5,2.),max_cases=8,
                          question_variants=None):
    """Shared frozen-decoder runner for EXP1/2/5; no discovery redirect or fitting."""
    if not 1<=pairs_per_step<=32 or not 1<=max_cases<=64:raise ValueError('Bounded pair/case budgets required')
    core=('level_shift','drift','spike','variance');kinds=core if kinds is None else tuple(kinds)
    if any(k not in core for k in kinds):raise ValueError('Core answer audit supports level_shift/drift/spike/variance')
    runtime=ready_answer_runtime(root,runtime);records=[]
    from .evaluation import intervention_cases,binary_metrics
    for step in _steps(runtime,steps):
        meta=runtime.encoders[step][1];sensors=list(meta['sensor_columns'])
        pairs=load_cached_pairs(runtime,step,split='test',limit=pairs_per_step,seed=seed)
        for index,(current,reference) in enumerate(pairs):
            center=sensors[index%len(sensors)]  # Chosen BEFORE generating interventions.
            cases=(intervention_cases(current,sensors,np.asarray(meta['std']),kinds=kinds,
                strengths=strengths,max_cases=max_cases,seed=seed+index,target_sensor=center)
                if inject else [(None,current,None)])
            for case,(spec,frame,truth) in enumerate(cases):
                rows=predict_window_answers(runtime,frame,reference,step,centers=[center],
                                            question_variants=question_variants)
                for row in rows:
                    row.update(step=step,pair_index=index,case_index=case,injected=spec is not None,
                        kind=spec.kind if spec else 'untouched',strength=spec.strength if spec else None,
                        effective=truth['effective'] if truth else None,truth_used_for_inference=False)
                records.extend(rows)
            print(f'Frozen answer audit: {step}, pair {index+1}/{len(pairs)}',flush=True)
    details=pd.DataFrame(records)
    if details.empty:raise ValueError('No held-out windows available')
    def summarize(group):
        if not inject:
            return dict(status='EVALUATED',questions=len(group),accuracy_available=False,
                        metric_target='original_input_answers_without_independent_truth')
        boolean=group[group.answer_type.eq('BOOLEAN')]
        effective=boolean[boolean.injected & boolean.effective.eq(True)]
        controls=boolean[~boolean.injected]
        scored=pd.concat([controls,effective])
        metrics=binary_metrics(scored.injected.to_numpy(),scored.alert.astype(bool).to_numpy())
        return dict(status='EVALUATED',**metrics,effective_cases=len(effective),
            evaluation_status='EVALUATED' if len(effective) else 'NOT_EVALUATED_NO_EFFECTIVE_INJECTIONS',
            control_windows=len(controls),untouched_alert_rate=float(controls.alert.mean()) if len(controls) else np.nan,
            metric_target='synthetic_added_change_vs_unlabelled_untouched',fault_accuracy_available=False)
    summary=pd.DataFrame([summarize(details)])
    by_step=pd.DataFrame([dict(step=step,**summarize(g)) for step,g in details.groupby('step',sort=False)])
    errors=details[details.injected & details.effective.eq(True) & details.answer_type.eq('BOOLEAN') & ~details.alert.eq(True)] if inject else pd.DataFrame()
    diagnostics={}
    if inject:
        diagnostics['by_kind_and_strength']=pd.DataFrame([dict(kind=k,strength=s,**summarize(
            pd.concat([details[~details.injected],g]))) for (k,s),g in details[details.injected].groupby(['kind','strength'],sort=False)])
    return ExperimentReport('Frozen GEVIS learned-answer audit',summary,by_step,details,errors,
        diagnostics=diagnostics,metadata=dict(status='COMPLETED',model_evaluated=True,
            prediction_mode='learned_answers',weights_updated=False,injections_used=inject,
            numeric_fallback_used=False,probability_threshold=.5 if inject else None,
            backbone_sha256=runtime.module.GEVIS_CHECKPOINT.get('_artifact_sha256'),
            answer_checkpoint=str(os.environ.get('GEVIS_ANSWER_CHECKPOINT') or
                Path(runtime.module.BACKBONE_PATH).parent/'gevis_answer_model.pt')))


def _progress(label,done,total):
    print(f'{label}: {done}/{total}',flush=True)


def _metadata(root,runtime,started):
    return {'status':'COMPLETED','model_evaluated':True,'elapsed_seconds':time.perf_counter()-started,
            'decision_policy':getattr(runtime.module,'DECISION_POLICY_MODE','legacy'),
            'checkpoint_sha256':runtime.module.GEVIS_CHECKPOINT.get('_artifact_sha256'),
            'source_identity':source_identity(root),'weights_updated':False,
            'result_meaning':'held-out comparison and controlled-intervention measurements; not verified equipment-fault performance'}


def run_question_experiment(root,*,steps=None,pairs_per_step=2,question_limit=120,batch_size=32,seed=42,runtime=None,
                            split='test',routing_mode=None):
    if split not in ('validation','test'): raise ValueError('Only held-out validation/test question evaluation')
    if routing_mode not in (None,'learned','explicit_text'): raise ValueError('Unknown routing mode')
    started=time.perf_counter(); runtime=runtime or ready_runtime(root)
    if runtime.module.GEVIS_CHECKPOINT.get('schema')=='gevis.backbone.generic.v10':
        from .discovery_reporting import run_discovery_experiment
        return run_discovery_experiment(root,runtime=runtime,steps=steps,
            pairs_per_step=pairs_per_step,max_cases=question_limit,seed=seed,split=split)
    selected=_steps(runtime,steps); records=[]; coverage=[]
    for index,step in enumerate(selected):
        pairs=load_cached_pairs(runtime,step,split=split,limit=pairs_per_step,seed=seed)
        for pair_index,(current,reference) in enumerate(pairs):
            print(f'Question audit Step {index+1}/{len(selected)}, pair {pair_index+1}/{len(pairs)}',flush=True)
            rows=evaluate_questions(runtime,current,reference,step,limit=question_limit,batch_size=batch_size,
                progress=lambda done,total:_progress('  questions',done,total),routing_mode=routing_mode)
            for row in rows:
                row.update(step=step,pair_id=current.attrs.get('evaluation_pair_id',str(pair_index)))
            records.extend(rows)
        coverage.append({'step':step,'test_pairs':len(pairs),'tested_runs':len({c.attrs.get('evaluation_run_id',str(i)) for i,(c,_) in enumerate(pairs)})})
    details=pd.DataFrame(records); summary,support=summarize_questions(records)
    by_step=[]
    for step,group in details.groupby('step',sort=False):
        result,_=summarize_questions(group.to_dict('records')); result.insert(0,'step',step); by_step.append(result)
    failed=details.supported & (~details.joint_query_correct | ~details.end_to_end_correct.eq(True))
    errors=details[failed].copy()
    errors['failure_reason']=[', '.join(name for name,ok in (
        ('sensor',row.sensor_correct),('scope',row.scope_correct),
        ('question meaning',row.joint_query_correct),('reference criterion',getattr(row,'reference_supported',True)),
        ('answer',row.answer_correct),('withheld',row.accepted)) if not ok)
        for row in errors.itertuples()]
    from .execution_contract import execution_contract
    metadata={**_metadata(root,runtime,started),'execution_contract':execution_contract(runtime,'named_question'),**support,'sampling':coverage,'split':split,'routing_mode':routing_mode or 'learned',
        'configuration':{'seed':seed,'pairs_per_step':pairs_per_step,'question_limit':question_limit,'batch_size':batch_size}}
    from .evaluation import summarize_reference_paths
    diagnostics={}
    from .routing_audit import gate_report
    diagnostics['gate_blockers']=gate_report(details)
    if getattr(runtime.module,'DECISION_POLICY_MODE','legacy') in ('reference_quantiles_v1','reference_quantiles_v2'):
        diagnostics['reference_policy_paths']=summarize_reference_paths(records)
        support_rows=[]; warning_rows=[]
        for kind,group in details[details.supported].groupby('answer_type',sort=False):
            support_rows.append({'answer_type':kind,'questions':len(group),
                'target_criterion_supported':int(group.reference_supported.sum()),
                'predicted_criterion_supported':int(group.predicted_criterion_supported.sum()),
                'accepted':int(group.accepted.sum()),
                'ordinal_tied_boundary_questions':int(group.ordinal_tied_boundaries.eq(True).sum())})
            for warning in sorted({w for items in group.warnings for w in items}):
                warning_rows.append({'answer_type':kind,'reason':warning,'questions':sum(warning in items for items in group.warnings)})
        diagnostics['criterion_support_and_acceptance']=pd.DataFrame(support_rows)
        diagnostics['withholding_reasons']=pd.DataFrame(warning_rows)
    return ExperimentReport('EXP 1: actual question understanding',summary,pd.concat(by_step,ignore_index=True),details,errors,metadata=metadata,diagnostics=diagnostics)


def injection_summary(details,group_by=('kind',)):
    """Only effective interventions enter recall; report withheld outputs explicitly."""
    frame=pd.DataFrame(details)
    if frame.empty: raise ValueError('No actual-model intervention results')
    controls=frame[~frame.injected]
    control_rate=float(controls.alert.mean()) if len(controls) else np.nan
    injected=frame[frame.injected]
    rows=[]
    groups=injected.groupby(list(group_by),dropna=False,sort=False) if group_by else [((),injected)]
    for keys,group in groups:
        if not isinstance(keys,tuple): keys=(keys,)
        effective=group[group.effective.eq(True)]
        group_controls=controls
        for key,value in zip(group_by,keys):
            if key!='kind' and key in group_controls: group_controls=group_controls[group_controls[key].eq(value)]
        rows.append({**dict(zip(group_by,keys)),'injected_cases':len(group),'effective_cases':len(effective),
            'detected_cases':int(effective.alert.sum()),
            'detection_recall':float(effective.alert.mean()) if len(effective) else np.nan,
            'candidate_detection_recall':float(effective.candidate_alert.mean()) if len(effective) else np.nan,
            'top_sensor_accuracy':pd.to_numeric(effective.top_sensor_correct).mean(),
            'mean_interval_iou':effective.best_alert_interval_iou.mean(),
            'accepted_question_share':group.accepted_question_share.mean(),
            'control_windows':len(group_controls),'untouched_alert_rate':float(group_controls.alert.mean()) if len(group_controls) else np.nan,
            'mean_ms_per_window':group.latency_ms.mean()})
        for field in ('oracle_declared_alert','target_candidate_alert','target_accepted_alert','new_target_alert','top_delta_sensor_correct','pointwise_target_iou','learned_evidence_sensor_correct','learned_evidence_patch_hits_event'):
            rows[-1][field+'_mean']=pd.to_numeric(effective[field]).mean() if field in effective else np.nan
    return pd.DataFrame(rows)


def run_injection_experiment(root,*,steps=None,pairs_per_step=2,kinds=KINDS,strengths=(.5,2.),
                             max_cases=7,batch_size=32,max_questions=1000,seed=42,runtime=None):
    started=time.perf_counter(); runtime=runtime or ready_runtime(root)
    selected=_steps(runtime,steps); tables=[]
    for index,step in enumerate(selected):
        pairs=load_cached_pairs(runtime,step,limit=pairs_per_step,seed=seed)
        for pair_index,(current,reference) in enumerate(pairs):
            print(f'Intervention audit Step {index+1}/{len(selected)}, pair {pair_index+1}/{len(pairs)}',flush=True)
            result=evaluate_injections(runtime,current,reference,step,np.asarray(runtime.encoders[step][1]['std']),
                kinds=kinds,strengths=strengths,max_cases=max_cases,max_questions=max_questions,batch_size=batch_size,
                seed=seed+pair_index,progress=lambda done,total:_progress('  windows',done,total))
            result['step']=step; result['pair_id']=current.attrs.get('evaluation_pair_id',str(pair_index)); tables.append(result)
    details=pd.concat(tables,ignore_index=True)
    by_step=injection_summary(details,('step','kind'))
    errors=details[details.injected & details.effective.eq(True) & (~details.alert | ~details.top_sensor_correct.eq(True))]
    stages=details[details.injected & details.effective.eq(True)]
    diagnostics={}
    if 'diagnostic_stage' in stages:
        diagnostics['failure_stages']=stages.groupby(['kind','diagnostic_stage'],dropna=False).size().rename('cases').reset_index()
        gates={}
        for counts in stages.gate_warning_counts.dropna():
            for name,value in counts.items(): gates[name]=gates.get(name,0)+value
        diagnostics['withholding_reasons']=pd.DataFrame([{'gate':name,'blocked_target_questions':value} for name,value in gates.items()])
    from .execution_contract import execution_contract
    return ExperimentReport('EXP 2: actual GEVIS intervention detection',injection_summary(details),by_step,details,errors,diagnostics=diagnostics,
        metadata={**_metadata(root,runtime,started),'execution_contract':execution_contract(runtime,'named_question'),'configuration':{'seed':seed,'pairs_per_step':pairs_per_step,
            'kinds':list(kinds),'strengths':list(strengths),'max_cases':max_cases,'batch_size':batch_size}})


def optional_integrations():
    from .merlion_backend import merlion_backend
    backend=merlion_backend()
    rows=[{'integration':'Merlion IsolationForest','installed':backend['ready'],
           'status':'AVAILABLE' if backend['ready'] else backend['reason'],
           'backend':backend['backend'],'version':backend.get('merlion_version')}]
    for name,module in (('TimeEval metrics','timeeval'),('GutenTAG generator','gutenTAG'),('tsaug augmentation','tsaug')):
        present=importlib.util.find_spec(module) is not None
        rows.append({'integration':name,'installed':present,'status':'AVAILABLE' if present else 'NOT_INSTALLED'})
    return pd.DataFrame(rows)


def run_baseline_experiment(root,*,steps=None,validation_pairs=4,test_pairs=2,max_cases=4,batch_size=32,
                            max_questions=1000,use_merlion=True,seed=42,runtime=None,use_temporal=True,temporal_epochs=10,
                            use_learned_answers=False):
    from .baselines import compare_frozen_runtime
    started=time.perf_counter(); integrations=optional_integrations()
    present=bool(integrations.loc[integrations.integration.eq('Merlion IsolationForest'),'installed'].iloc[0])
    if use_merlion not in ('auto',True,False): raise ValueError('use_merlion must be auto, True or False')
    if use_merlion is True and not present:
        raise ImportError('EXP 3 requires a compatible Merlion worker. Run python scripts/setup_merlion.py from the project root, then restart the kernel.')
    enabled=present if use_merlion=='auto' else use_merlion
    runtime=runtime or ready_runtime(root)
    if use_learned_answers:runtime=ready_answer_runtime(root,runtime)
    elif getattr(runtime.module,'DISCOVERY_POLICY_MODE',None)=='window_calibrated_v2':
        from .gevis_system import ready_gevis_runtime
        runtime=ready_gevis_runtime(root,runtime)
    selected=_steps(runtime,steps); tables=[]; details=[]; operating=[]; timing=[]
    for index,step in enumerate(selected):
        print(f'Paired baseline comparison Step {index+1}/{len(selected)}',flush=True)
        result,raw=compare_frozen_runtime(runtime,step,validation_limit=validation_pairs,test_limit=test_pairs,
            max_cases=max_cases,max_questions=max_questions,batch_size=batch_size,use_merlion=enabled,seed=seed,
            progress=lambda done,total:_progress('  windows',done,total),use_temporal=use_temporal,temporal_epochs=temporal_epochs,
            use_learned_answers=use_learned_answers)
        for key,collection in (('operating_points',operating),('temporal_fit',timing)):
            part=result.attrs.get(key,pd.DataFrame()).copy()
            if not part.empty: part['step']=step; collection.append(part)
        result.attrs.clear()
        result['step']=step; raw['step']=step; tables.append(result); details.append(raw)
    by_step=pd.concat(tables,ignore_index=True); raw=pd.concat(details,ignore_index=True)
    rows=[]
    for method,group in by_step.groupby('method',sort=False):
        sums={key:int(group[key].sum()) for key in ('tp','fp','fn','tn')}
        tp,fp,fn,tn=(sums[k] for k in ('tp','fp','fn','tn'))
        ratio=lambda a,b:a/b if b else np.nan
        rows.append({'method':method,'status':'EVALUATED',**sums,'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
            'untouched_alert_rate':ratio(fp,fp+tn),'f1':ratio(2*tp,2*tp+fp+fn),
            'test_windows':tp+fp+fn+tn})
        if method=='GEVIS_shared_evidence_discovery':
            withheld=int(group.prediction_abstentions.sum())
            rows[-1].update(prediction_abstentions=withheld,answer_coverage=1-withheld/(tp+fp+fn+tn),
                threshold_source=group.threshold_source.iloc[0])
        if method=='GEVIS_learned_answer':
            rows[-1].update(threshold_source='same Validation controls as other scored baselines',
                score_source='trained_answer_decoder_softmax; full sensor max',
                metric_target='synthetic_added_change_vs_unlabelled_untouched')
    if not enabled:
        rows.append({'method':'Merlion_IsolationForest','status':'NOT_INSTALLED' if not present else 'DISABLED'})
    integrations.loc[integrations.integration.eq('Merlion IsolationForest') & enabled,'status']='EVALUATED'
    diagnostics={'nominal_control_operating_points':pd.concat(operating,ignore_index=True) if operating else pd.DataFrame(),
                 'temporal_training_cost':pd.concat(timing,ignore_index=True) if timing else pd.DataFrame(),
                 'gevis_window_latency':raw.groupby('step').agg(mean_scan_ms=('gevis_scan_ms','mean'),max_scan_ms=('gevis_scan_ms','max')).reset_index()}
    if 'GEVIS_learned_answer_ms' in raw:
        diagnostics['learned_answer_window_latency']=raw.groupby('step').agg(
            mean_window_ms=('GEVIS_learned_answer_ms','mean'),max_window_ms=('GEVIS_learned_answer_ms','max')).reset_index()
    from .execution_contract import execution_contract
    diagnostics['execution_paths']=pd.DataFrame([execution_contract(runtime,'named_question'),execution_contract(runtime,'blind_discovery')])
    if use_learned_answers:
        neural=execution_contract(runtime,'learned_answers')
        neural.update(sensor_source='every sensor queried; maximum Yes probability; no injected target supplied',
                      evaluated_method='GEVIS_learned_answer')
        diagnostics['learned_answer_execution_path']=pd.DataFrame([neural])
    return ExperimentReport('EXP 3: paired actual-model comparison',pd.DataFrame(rows),by_step,raw,diagnostics=diagnostics,
        integrations=integrations,metadata={**_metadata(root,runtime,started),
            'learned_answer_evaluated':use_learned_answers,'numeric_fallback_used':False,
            'configuration':{'seed':seed,'validation_pairs':validation_pairs,'test_pairs':test_pairs,
                             'max_cases':max_cases,'batch_size':batch_size,'merlion_enabled':enabled,
                             'temporal_enabled':use_temporal,'temporal_epochs':temporal_epochs,
                             'learned_answer_enabled':use_learned_answers,
                             'merlion_fit_split':'train' if enabled else None,
                             'merlion_version':integrations.loc[integrations.integration.eq('Merlion IsolationForest')].get('version',pd.Series([None])).iloc[0]}})


def transfer_scan_budget(runtime, max_questions=None):
    """Plan a full scan, not unlimited execution or silent sensor subsampling."""
    from .questions import STATISTICS,SCOPES
    rows=[{'step':step,'sensors':len(meta['sensor_columns']),
           'questions_per_window':len(meta['sensor_columns'])*len(STATISTICS)*len(SCOPES)}
          for step,(_,meta) in runtime.encoders.items()]
    if not rows or any(row['sensors']<1 for row in rows): raise ValueError('Nonempty external sensor scope required')
    required=max(row['questions_per_window'] for row in rows)
    if max_questions is not None:
        if isinstance(max_questions,bool) or not isinstance(max_questions,int) or max_questions<1:
            raise ValueError('max_detection_questions must be None or a positive integer')
        if required>max_questions:
            raise ValueError(f'Full external scan needs {required} questions per window; '
                             f'MAX_DETECTION_QUESTIONS_PER_WINDOW={max_questions} is too small. '
                             'Set it to None (exact full-sensor budget) or raise the explicit ceiling. No sensors were skipped.')
    plan=pd.DataFrame(rows)
    plan['scan_budget']=required if max_questions is None else max_questions
    return int(plan.scan_budget.max()),plan


def run_transfer_experiment(root,*,process_config=None,backbone_path=None,pairs_per_step=2,question_limit=120,batch_size=32,seed=42,evaluate_detection=True,source_step=None,source_runtime=None,prepared_frame=None,max_detection_questions=None,prediction_mode='discovery',answer_checkpoint=None,question_texts=None,include_injections=True,quality_options=None):
    # Missing prerequisites produce a visible final status, never a synthetic success.
    if prediction_mode not in ('discovery','learned_answers'):raise ValueError('Unknown transfer prediction mode')
    config=process_config or os.environ.get('GEVIS_EXTERNAL_CONFIG')
    backbone=backbone_path or os.environ.get('GEVIS_SOURCE_BACKBONE')
    if config and not Path(config).is_absolute(): config=Path(root)/config
    if backbone and not Path(backbone).is_absolute(): backbone=Path(root)/backbone
    from .process_inputs import candidate_backbone_path,load_process_frame
    if not config and (Path(root)/'.cache/external/peald/process.json').is_file(): config=Path(root)/'.cache/external/peald/process.json'
    from pecvd.process_config import load_process_config
    from pecvd.gevis_backbone import TARGET_BINDING_VERSION
    source_config=Path(root)/'config/processes/pecvd.json'
    if not backbone:
        from pecvd.model_variants import active_config,active_run_tag
        backbone=candidate_backbone_path(root,active_config(load_process_config(source_config)),active_run_tag())
    required=pd.DataFrame([
        {'item':'EXTERNAL_CONFIG','status':'READY' if config and Path(config).is_file() else 'MISSING',
         'needed':'New dataset process configuration JSON (not the PECVD example)'},
        {'item':'FROZEN_SOURCE_BACKBONE','status':'READY' if source_runtime is not None or Path(backbone).is_file() else 'MISSING',
         'needed':'PECVD-trained STEP 2 backbone and its original STEP 1 manifest; never PEALD-trained weights'},
    ])
    if required.status.eq('MISSING').any():
        summary=pd.DataFrame([{'experiment':'cross-dataset GEVIS','status':'NOT_EVALUATED',
            'reason':'External input mapping and frozen PECVD source weights are required; no external training','model_evaluated':False}])
        return ExperimentReport('EXP 4: external dataset evaluation',summary,prerequisites=required,
            metadata={'status':'NOT_EVALUATED','model_evaluated':False})
    path=Path(config).resolve(); root=Path(root).resolve()
    if path==(root/'config/processes/pecvd.json').resolve():
        raise ValueError('PECVD config is the current dataset, not an external transfer test')
    source_runtime=source_runtime or load_experiment_runtime(root,process_config=source_config,backbone_path=backbone)
    learned_model=None
    if prediction_mode=='learned_answers':
        from .frozen_transfer import load_frozen_answer_model
        candidate=answer_checkpoint or os.environ.get('GEVIS_ANSWER_CHECKPOINT') or Path(backbone).parent/'gevis_answer_model.pt'
        candidate=Path(candidate)
        if not candidate.is_absolute():candidate=root/candidate
        try:learned_model=load_frozen_answer_model(source_runtime,candidate)
        except (FileNotFoundError,ValueError,RuntimeError) as exc:
            expected_width=getattr(getattr(getattr(source_runtime.module,'GEVIS_MODEL',None),'numeric',None),'in_features',None)
            return ExperimentReport('EXP 4: learned question-answer inference',pd.DataFrame([{
                'status':'NOT_EVALUATED','reason':str(exc),'model_evaluated':False,
                'expected_answer_checkpoint':str(candidate),'expected_numeric_dim':expected_width,
                'next_action':'Provide a trained answer checkpoint bound to this backbone; absent weights require STEP3 head training, not a dimension rename'}]),
                prerequisites=pd.concat([required,pd.DataFrame([{'item':'COMPATIBLE_STEP3_ANSWER_CHECKPOINT',
                    'status':'MISSING_OR_INCOMPATIBLE','needed':'trained decoder bound to the exact active CVD backbone'}])],ignore_index=True),
                metadata={'status':'NOT_EVALUATED','model_evaluated':False,'weights_updated':False,
                    'prediction_mode':prediction_mode,'numeric_fallback_used':False,
                    'expected_answer_checkpoint':str(candidate),'expected_numeric_dim':expected_width})
    external_config=load_process_config(path)
    # Explicit OFFLINE preparation. Runtime inference below never reads CSV.
    print('Offline external input preparation; source weights remain frozen...',flush=True)
    frame=prepared_frame if prepared_frame is not None else load_process_frame(root,external_config)
    quality_tables={}
    if quality_options is not None:
        from .process_inputs import screen_external_quality
        frame,quality_tables=screen_external_quality(frame,external_config,**quality_options)
        count=frame[external_config.columns.run].nunique()
        a=max(1,int(.7*count));b=max(a+1,a+int(.15*count))
        if min(a,b-a,count-b)<2:
            return ExperimentReport('EXP 4: external quality preflight',pd.DataFrame([dict(
                status='NOT_EVALUATED',reason='Too few independent runs after explicit quality exclusions',
                model_evaluated=False)]),prerequisites=required,diagnostics=quality_tables,
                metadata=dict(status='NOT_EVALUATED',model_evaluated=False,weights_updated=False))
    from .frozen_transfer import bind_external_runtime,NativeTransferUnsupported
    try:
        bind_kwargs=dict(source_step=source_step,limit=pairs_per_step,seed=seed)
        if prediction_mode=='learned_answers':bind_kwargs['prepare_criteria']=False
        runtime,support=bind_external_runtime(root,source_runtime,external_config,frame,**bind_kwargs)
    except NativeTransferUnsupported as exc:
        from .execution_contract import execution_contract
        source_contract=execution_contract(source_runtime,'named_question')
        source_contract.update(execution_path='not_evaluated',step1_encoder_executed=False,
            step2_cross_attention_executed=False,sensor_source='unavailable for this external sensor schema',
            interval_source='not evaluated')
        return ExperimentReport('EXP 4: frozen source transfer',pd.DataFrame([{
            'status':'NOT_EVALUATED','reason':str(exc),'model_evaluated':False}]),
            prerequisites=required,diagnostics={**quality_tables,'source_artifact_contract':pd.DataFrame([
                source_contract])},
            metadata={'status':'NOT_EVALUATED','model_evaluated':False,'weights_updated':False,
                      'reason':'saved fixed-sensor encoder does not support native unseen-sensor inputs'})
    if not runtime.encoders:
        return ExperimentReport('EXP 4: frozen source transfer',pd.DataFrame([{
            'status':'NOT_EVALUATED','reason':'No complete independent external pairs at selected source context length'}]),
            prerequisites=required,diagnostics={**quality_tables,'external_phase_support':support},
            metadata={'status':'NOT_EVALUATED','model_evaluated':False,**runtime.transfer_metadata})
    if prediction_mode=='learned_answers':
        from .frozen_transfer import predict_external_answers
        runtime.module.GEVIS_MODEL=learned_model;runtime.trained_answer_decoder_loaded=True
        runtime.module.DISCOVERY_POLICY=None;runtime.module.REFERENCE_POLICY=None
        runtime.module.DECISION_POLICY_MODE='learned_answers'
        details=predict_external_answers(runtime,pairs_per_step=pairs_per_step,question_texts=question_texts)
        if details.empty:
            return ExperimentReport('EXP 4: learned question-answer inference',pd.DataFrame([{
                'status':'NOT_EVALUATED','reason':'No untouched external evaluation windows'}]),
                prerequisites=required,metadata={'status':'NOT_EVALUATED','model_evaluated':False})
        summary=details.groupby('answer_type',sort=False).size().rename('predictions').reset_index()
        summary['score_source']='trained_answer_decoder_softmax'
        summary['accuracy_available']=False
        by_step=details.groupby(['step','answer_type'],sort=False).size().rename('predictions').reset_index()
        return ExperimentReport('EXP 4: original PEALD question-answer predictions',summary,by_step,details,
            prerequisites=required,diagnostics={**quality_tables,'external_phase_support':support},metadata={
                'status':'COMPLETED','model_evaluated':True,'weights_updated':False,
                **runtime.transfer_metadata,'prediction_mode':prediction_mode,
                'answer_checkpoint_sha256':getattr(learned_model,'answer_checkpoint_sha256',None),
                'injections_used':False,'numeric_fallback_used':False,
                'score_meaning':'learned class softmax, not calibrated correctness or equipment-fault probability'})
    if getattr(getattr(runtime,'module',None),'GEVIS_CHECKPOINT',{}).get('schema')=='gevis.backbone.generic.v10':
        # Generic models evaluate calibrated discovery directly, not a named
        # question redirect followed by a second, incompatible injection scan.
        if not evaluate_detection:
            return ExperimentReport('EXP 4: frozen source transfer',pd.DataFrame([{
                'status':'NOT_EVALUATED','reason':'Generic discovery evaluation explicitly disabled'}]),
                prerequisites=required,diagnostics={'external_phase_support':support},
                metadata={'status':'NOT_EVALUATED','model_evaluated':False,**runtime.transfer_metadata})
        from .discovery_reporting import run_discovery_experiment
        from .gevis_system import run_gevis_predictions,transfer_prediction_diagnostics,SYSTEM
        from .threshold_comparison import position_calibration_table
        original=run_gevis_predictions(root,runtime=runtime,pairs_per_step=pairs_per_step,seed=seed)
        original.diagnostics.update(quality_tables)
        original.diagnostics.update(transfer_prediction_diagnostics(original.details,
            policy=runtime.module.DISCOVERY_POLICY))
        if not include_injections:
            original.metadata.update(runtime.transfer_metadata,prediction_mode='discovery',gevis_system=SYSTEM)
            original.diagnostics['external_phase_support']=support
            return original
        policy=runtime.module.DISCOVERY_POLICY
        mode='center' if policy.config.gate_scope=='position_rank_v4' else 'blind'
        report=run_discovery_experiment(root,runtime=runtime,pairs_per_step=pairs_per_step,
            max_cases=min(question_limit,8),seed=seed,analysis_mode=mode)
        report.name='EXP 4: frozen weights + external reference/Validation calibration'
        report.prerequisites=required;report.metadata.update(runtime.transfer_metadata)
        report.metadata.update(gevis_system=SYSTEM,prediction_mode='discovery',learned_answer_head_is_detector=False)
        report.diagnostics['original_input_predictions']=original.details
        report.diagnostics.update(quality_tables)
        report.diagnostics.update(transfer_prediction_diagnostics(original.details,report.details,
            policy=policy))
        report.metadata.update(external_input_mode='independent held-out windows; no online updates',
            external_window_calibration='external reference and Validation only; no source thresholds or Test fitting')
        report.diagnostics['external_phase_support']=support
        report.diagnostics['external_position_calibration']=position_calibration_table({
            round(1-policy.config.nominal_window_alert_rate,2):policy})
        return report
    scan_budget,scan_plan=(transfer_scan_budget(runtime,max_detection_questions) if evaluate_detection else (None,pd.DataFrame()))
    if evaluate_detection:
        print(f'External detection preflight: full-sensor budget {scan_budget} questions/window, batch_size={batch_size}',flush=True)
    report=run_question_experiment(root,pairs_per_step=pairs_per_step,question_limit=question_limit,
                                   batch_size=batch_size,seed=seed,runtime=runtime)
    report.name='EXP 4: frozen PECVD weights on external inputs'
    report.prerequisites=required
    report.metadata.update(runtime.transfer_metadata)
    report.diagnostics['external_phase_support']=support
    report.metadata['external_input_mode']='independent held-out windows; no online updates; not chronological-stream validation'
    if evaluate_detection:
        report.diagnostics['external_detection_scan_plan']=scan_plan
        detection=run_injection_experiment(root,pairs_per_step=pairs_per_step,batch_size=batch_size,
            seed=seed,runtime=runtime,max_questions=scan_budget)
        report.diagnostics['external_controlled_detection']=detection.summary
        report.diagnostics['external_detection_stages']=detection.diagnostics.get('failure_stages',pd.DataFrame())
    return report


def show_report(report,display):
    """Always finish with status and a numeric table or an explicit unavailable table."""
    print(report.name,flush=True)
    print('Status:',report.metadata['status'],'| Actual model evaluated:',report.metadata['model_evaluated'])
    if report.metadata.get('execution_contract'):
        print('Executed components (loaded weights do not imply use in every path)')
        display(pd.DataFrame([report.metadata['execution_contract']]))
    if 'unsupported_rejection_rate' in report.metadata:
        print('Unsupported-question rejection rate:',report.metadata['unsupported_rejection_rate'])
    if report.metadata.get('sampling'):
        print('Held-out sample coverage'); display(pd.DataFrame(report.metadata['sampling']))
    if not report.prerequisites.empty:
        print('Required inputs'); display(report.prerequisites)
    if not report.integrations.empty:
        print('Optional integrations'); display(report.integrations)
    for name,table in report.diagnostics.items():
        if name=='evaluation_summary':continue  # Printed once, last and compact.
        if not table.empty: print(name); display(table)
    if not report.by_step.empty:
        print('Per-Step results'); display(report.by_step)
    if not report.errors.empty:
        print('First error/withheld cases (full table: REPORT.errors)')
        keys=[k for k in ('step','pair_id','answer_type','question','expected_sensor','predicted_sensor','expected_scope',
            'predicted_scope','oracle_proxy','candidate','accepted','failure_reason','kind','strength','alert','top_sensor_correct') if k in report.errors]
        display(report.errors[keys].head(12))
    if 'answer_scores' in report.details:
        print('Example answer distributions (full distributions: DETAILS)')
        samples=report.details[report.details.supported].groupby('answer_type',sort=False).head(1)
        scores=[{'answer_type':row.answer_type,'candidate':row.candidate,'accepted':row.accepted,
                 'option':option,'score':score}
                for row in samples.itertuples() for option,score in row.answer_scores.items()]
        display(pd.DataFrame(scores))
    print('FINAL SUMMARY')
    final=report.diagnostics.get('evaluation_summary',report.summary)
    with pd.option_context('display.max_columns',None,'display.width',180):
        display(final.fillna('N/A') if 'evaluation_summary' in report.diagnostics else final)
    if 'evaluation_summary' in report.diagnostics:
        print('Original P/R/F1: N/A without independent truth. Latency: measured window review, excluding offline preparation; not guaranteed warm-only.')
    print('Scores measure question/controlled-change performance. They do not establish real equipment-fault accuracy.')
