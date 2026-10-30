"""Explicit GEVIS dispatch: contextual Fusion answers or preserved numeric comparator.

No answer-head substitution. Calibration and localization fitting are separate
offline preparation steps; predictions never fit on their current input.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SYSTEM='gevis.hybrid.position_rank.v1'


def ready_gevis_runtime(root,runtime=None,*,prepare=True):
    from .reporting import ready_runtime
    from .pipeline_settings import configure_pipeline
    runtime=runtime or ready_runtime(root)
    m=runtime.module
    if getattr(m,'DECISION_SYSTEM','hybrid')=='fusion_answers':
        from .fusion_answers import require_head
        require_head(m)
        return runtime
    if getattr(m,'DISCOVERY_POLICY_MODE',None)!='window_calibrated_v2':
        raise ValueError('Unified GEVIS requires the calibrated hybrid decision path')
    if getattr(m,'DISCOVERY_POLICY',None) is None:
        if not prepare or hasattr(runtime,'external_pairs'):
            raise RuntimeError('Prepare process-specific TRAIN/Validation criteria before inference')
        from .threshold_comparison import prepare_runtime_comparison
        prepare_runtime_comparison(runtime,configure_pipeline(root))
    if m.DISCOVERY_POLICY.config.gate_scope!='position_rank_v4':
        raise ValueError('Unified GEVIS requires position_rank_v4')
    return runtime


def run_gevis_experiment(root,*,runtime=None,**kwargs):
    """Same detector and localization used by STEP4 and all injection reports."""
    from .discovery_reporting import run_discovery_experiment
    runtime=ready_gevis_runtime(root,runtime)
    if getattr(runtime.module,'DECISION_SYSTEM','hybrid')=='fusion_answers':
        from .fusion_answers import evaluate
        return evaluate(runtime,**kwargs)
    report=run_discovery_experiment(root,runtime=runtime,**kwargs)
    report.metadata.update(gevis_system=SYSTEM,decision_path='hybrid',
        learned_answer_head_is_detector=False)
    return report


def run_gevis_predictions(root,*,runtime=None,steps=None,pairs_per_step=2,seed=42,
                          question_variants=None):
    """Untouched input predictions; no ground truth and no accuracy claims.

    Named operation probes explicitly select an existing numerical operation;
    they do not claim that English paraphrase understanding was learned.
    """
    from .reporting import ExperimentReport,_steps,ready_runtime
    runtime=runtime or ready_runtime(root)
    if getattr(runtime.module,'DECISION_SYSTEM','hybrid')=='fusion_answers':
        from .reporting import predict_fusion_original
        if question_variants is not None:raise ValueError('Use unchanged fixed questions for the Fusion experiment')
        return predict_fusion_original(runtime,steps=steps,pairs_per_step=pairs_per_step,seed=seed)
    from .frozen_transfer import load_discovery_pairs
    from .discovery import review_changes
    from pecvd.answer_display import QUESTIONS
    runtime=ready_gevis_runtime(root,runtime,prepare=not hasattr(runtime,'external_pairs'))
    m=runtime.module;records=[]
    variants=question_variants or [('any_change','BOOLEAN',QUESTIONS['BOOLEAN'])]
    operations={'mean_change':'level','variability_increase':'variability','spike':'local_change'}
    for step in _steps(runtime,steps):
        sensors=list(runtime.encoders[step][1]['sensor_columns'])
        for i,(current,reference) in enumerate(load_discovery_pairs(runtime,step,split='test',limit=pairs_per_step,seed=seed)):
            center=sensors[i%len(sensors)]
            result=review_changes(m,current,reference,step,m.ReferenceInfo('held-out TRAIN reference',1),center_sensor=center)
            for qid,kind,text in variants:
                if kind!='BOOLEAN':raise ValueError('Operation probes currently require BOOLEAN')
                if qid not in ('any_change','BOOLEAN',*operations):raise ValueError('Unsupported operation probe')
                candidates=result['ranked_evidence']
                if qid in operations:candidates=[r for r in candidates if r['statistic']==operations[qid]]
                accepted=result['accepted'] and bool(candidates)
                winner=max(candidates,key=lambda r:r['normalized_score']) if candidates else None
                alert=bool(accepted and winner['changed'])
                margin=(winner['normalized_score']-winner['window_threshold']) if accepted else None
                # Display-only monotone scores, never fed back into the detector.
                yes=float(1/(1+np.exp(-np.clip(margin,-60,60)))) if margin is not None else None
                records.append(dict(step=step,pair_index=i,pair_id=current.attrs.get('evaluation_pair_id'),
                    center_sensor=center,question_id=qid,
                    question=text,answer=('Yes' if alert else 'No') if accepted else None,
                    truth_used_for_inference=False,
                    prediction=('ALERT' if alert else 'NO_ALERT') if accepted else 'UNSUPPORTED',
                    accepted=accepted,alert=alert,probabilities={'No':1-yes,'Yes':yes} if yes is not None else {},
                    context_key=result.get('calibration',{}).get('context_key'),
                    calibration=result.get('calibration',{}),
                    missing_criteria=result.get('missing_criteria',[]),
                    latency_ms=result.get('latency_ms'),
                    target_score=winner['normalized_score'] if winner else None,
                    target_threshold=winner['window_threshold'] if winner else None,
                    question_scope='fixed change question' if qid not in operations else 'explicit numerical operation probe; not learned language validation',
                    score_source='hybrid numerical evidence; display scores NOT calibrated probabilities',
                    winner=result['winner']))
    details=pd.DataFrame(records)
    if details.empty:raise ValueError('No held-out input predictions')
    def summary(frame):
        return dict(status='EVALUATED',questions=len(frame),accuracy_available=False,
            metric_target='original_input_answers_without_independent_truth',
            answer_coverage=float(frame.accepted.mean()),
            mean_ms_per_window=_mean_latency(_original_rows(frame)))
    report=ExperimentReport('GEVIS unified hybrid predictions',pd.DataFrame([summary(details)]),
        pd.DataFrame([dict(step=s,**summary(g)) for s,g in details.groupby('step',sort=False)]),details,
        metadata=dict(status='COMPLETED',model_evaluated=True,weights_updated=False,
            gevis_system=SYSTEM,decision_path='hybrid',injections_used=False,
            learned_answer_head_is_detector=False,fault_probability_calibrated=False))
    report.diagnostics.update(transfer_prediction_diagnostics(details))
    return report


def _original_rows(frame):
    frame=frame.copy()
    if 'task' in frame:frame=frame[frame.task.eq('self')]
    if 'question_id' in frame:frame=frame[frame.question_id.isin(['any_change','BOOLEAN'])]
    if 'injected' in frame:frame=frame[frame.injected.eq(False)]
    return frame


def _mean_latency(frame):
    return (float(pd.to_numeric(frame.latency_ms,errors='coerce').mean())
        if 'latency_ms' in frame and len(frame) else np.nan)


def evaluation_summary(report,*,process):
    """Compact original versus controlled evaluation; never executes the model.

    Original unlabeled inputs have coverage/alerts/latency, not invented F1.
    Synthetic metrics come from the report's unchanged control-inclusive summary.
    """
    if getattr(report,'metadata',{}).get('decision_system')=='fusion_answers':
        return report.summary.assign(process=process,evaluation='Fusion answer task; controlled or unlabelled as declared')
    details=report.details
    original=report.diagnostics.get('original_input_predictions')
    if original is None:original=details
    original=_original_rows(original)
    rows=[]
    if not original.empty:
        accepted=original.accepted.eq(True);alert=accepted & original.alert.eq(True)
        rows.append(dict(process=process,evaluation='Original / untouched',windows=len(original),
            precision=np.nan,recall=np.nan,f1=np.nan,mean_ms_per_window=_mean_latency(original),
            alerts=int(alert.sum()),no_alerts=int((accepted&~alert).sum()),
            unsupported=int((~accepted).sum()),answer_coverage=float(accepted.mean()),
            alert_rate=float(alert.mean()),metrics_status='NO_INDEPENDENT_TRUTH'))
    if 'injected' in details and details.injected.eq(True).any():
        own=details[details.task.eq('self')] if 'task' in details else details
        # Preserve the existing metric definition, including abstention handling.
        for metric in report.summary.to_dict('records'):
            rows.append(dict(process=process,evaluation='Synthetic changes + untouched controls',
                windows=metric.get('windows',len(own)),
                precision=metric.get('precision',np.nan),recall=metric.get('recall',np.nan),
                f1=metric.get('f1',np.nan),mean_ms_per_window=metric.get('mean_ms_per_window',_mean_latency(own)),
                alerts=int((own.accepted.eq(True)&own.alert.eq(True)).sum()),
                no_alerts=int((own.accepted.eq(True)&~own.alert.eq(True)).sum()),
                unsupported=int((~own.accepted.eq(True)).sum()),answer_coverage=float(own.accepted.eq(True).mean()),
                alert_rate=float((own.accepted.eq(True)&own.alert.eq(True)).mean()),
                metrics_status='SYNTHETIC_TRUTH_NOT_REAL_FAULT_LABELS'))
    if not rows:
        return pd.DataFrame([dict(process=process,evaluation='NOT_EVALUATED',
            metrics_status='No original or synthetic predictions available',mean_ms_per_window=np.nan)])
    return pd.DataFrame(rows)


def transfer_prediction_diagnostics(original, injections=None, *, policy=None):
    """Report existing outputs only: no forwards, fitting, file I/O or label inference.

    Miss stages are observable checkpoints, not proven causal explanations.
    Attention and interval outcomes are orthogonal to binary detection.
    """
    original=_original_rows(original)
    # One row per actual input/center, not one per wording/operation probe.
    if 'question_id' in original:
        original=original[original.question_id.isin(['any_change','BOOLEAN'])].copy()
    tables={}
    def summarize(frame):
        accepted=frame.accepted.eq(True)
        alerts=accepted & frame.alert.eq(True)
        return dict(windows=len(frame),alerts=int(alerts.sum()),
            no_alerts=int((accepted & ~frame.alert.eq(True)).sum()),
            unsupported=int((~accepted).sum()),answer_coverage=float(accepted.mean()),
            alert_rate_all_inputs=float(alerts.mean()),
            alert_rate_when_accepted=float(alerts[accepted].mean()) if accepted.any() else np.nan,
            mean_ms_per_window=_mean_latency(frame),
            accuracy_available=False,metric_target='original_input_predictions_without_independent_truth')
    if not original.empty:
        tables['original_input_summary']=pd.DataFrame([summarize(original)])
        tables['original_input_by_step']=pd.DataFrame([
            dict(step=s,**summarize(g)) for s,g in original.groupby('step',sort=False)])
        # Support reasons are recoverable from saved criteria; do not guess them
        # from a false alert or a low numerical score.
        reasons=[]
        groups=getattr(policy,'payload',{}).get('groups',{})
        for row in original.to_dict('records'):
            context=groups.get(row['step'],{}).get('contexts',{}).get(row.get('context_key'))
            reasons.append(dict(step=row['step'],pair_index=row.get('pair_index'),
                center_sensor=row.get('center_sensor'),prediction=row.get('prediction'),
                target_score=row.get('target_score'),target_threshold=row.get('target_threshold'),
                train_runs=context.get('train_runs') if context else None,
                validation_runs=context.get('validation_runs') if context else None,
                support_failure=(context.get('support_failure') if context else 'context_metadata_unavailable')
                    if not row['accepted'] else None,
                control_budget_resolved=context.get('control_budget_resolved') if context else None))
        tables['original_input_support']=pd.DataFrame(reasons)
    if injections is None or injections.empty or 'injected' not in injections:
        return tables
    frame=injections.copy()
    if 'task' in frame:frame=frame[frame.task.eq('self')].copy()
    effective=frame[frame.injected.eq(True) & frame.effective.eq(True)].copy()
    controls=frame[frame.injected.eq(False)]
    if not controls.empty:
        tables['injection_control_summary']=pd.DataFrame([{**summarize(controls),
            'metric_target':'unlabelled_untouched_injection_controls'}])
    if effective.empty:return tables
    def stage(row):
        if not row.get('accepted',False):return 'UNSUPPORTED'
        if row.get('alert',False):return 'DETECTED'
        status=row.get('target_gate_status')
        if status=='BELOW_THRESHOLD':return 'BELOW_THRESHOLD'
        if status=='TARGET_NOT_ANALYZED':return 'TARGET_NOT_ANALYZED'
        return 'MISS_UNRESOLVED'
    effective['failure_stage']=[stage(r) for r in effective.to_dict('records')]
    tables['injection_outcome_stages']=effective.groupby('failure_stage',sort=False).size().rename('cases').reset_index()
    tables['injection_outcome_by_step']=effective.groupby(['step','failure_stage'],sort=False).size().rename('cases').reset_index()
    keys=[k for k in ('step','pair_id','center_sensor','kind','strength','accepted','alert',
        'failure_stage','target_gate_status','target_score','target_threshold','target_score_excess',
        'raw_attention_sensor_correct','attention_correct_but_numeric_gate_rejected',
        'winner_interval_iou','winner_interval_available') if k in effective]
    tables['injected_case_diagnosis']=effective[keys].copy()
    tables['injection_miss_cases']=effective.loc[~effective.alert.eq(True),keys].copy()
    tables['diagnosis_limits']=pd.DataFrame([dict(
        meaning='Stages partition effective injections; attention/interval checks are separate, not causal proof.',
        original_accuracy_available=False,test_used_for_threshold_selection=False,
        note='Do not tune thresholds on these Test outcomes; unlabelled controls are not verified normal.')])
    return tables
