"""Paired nominal 95/99 comparison. No changed truth, test fitting or probability claim."""
from dataclasses import replace
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
import torch
from .window_policy import WindowConfig,WindowPolicy,fit_window_policy
from .discovery_calibration import cached_comparisons,residual_trace,identity_for,load_policy
from .pipeline_settings import comparison_policy_path

QUANTILES=(.90,.95,.99)

def configured_quantiles(config):
    return tuple(dict.fromkeys((*QUANTILES,round(1-config.nominal_window_alert_rate,12))))

def comparison_identity():
    return {'schema':'gevis.threshold_comparison.v1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'quantiles':list(QUANTILES),'truth':'all effective injections unchanged; untouched controls unlabelled',
            'scale_quantile':.95,'calibration':'all sampled Validation run maxima; Validation evaluation overlaps calibration'}

def fit_comparison(train,validation,identity_factory,config=WindowConfig()):
    """Same numerical scale and same controls; vary only nominal final tail rate."""
    if config.scale_quantile!=.95:raise ValueError('Keep TRAIN scale fixed at .95 for threshold-only comparison')
    train=list(train);validation=list(validation);policies={}
    for q in configured_quantiles(config):
        cfg=replace(config,nominal_window_alert_rate=1-q)
        payload=fit_window_policy(train,validation,identity_factory(cfg),cfg)
        payload['threshold_comparison']=comparison_identity()
        policies[q]=WindowPolicy(payload)
    if config.gate_scope=='center_views_v3':
        for q,policy in policies.items():
            for step,g in policy.payload['groups'].items():
                g['joint_quantile_adoption']=g['center_gate_adopted']
        # Switching the normalization between the two rows would not be a
        # threshold-only comparison. Adopt the center candidate jointly or not.
        for step in policies[QUANTILES[0]].payload['groups']:
            if not all(np.array_equal(policies[QUANTILES[0]].payload['groups'][step]['scales'],p.payload['groups'][step]['scales']) for p in policies.values()):
                raise RuntimeError('Threshold comparison normalization changed')
            groups=[p.payload['groups'][step] for p in policies.values()]
            if all('center_adoption_mask' in g for g in groups):
                joint=np.logical_and.reduce([np.asarray(g['center_adoption_mask'],bool)&g['center_gate_adopted'] for g in groups])
                for g in groups:
                    g['center_adoption_mask']=joint.tolist()
                    g['center_gate_adopted']=bool(joint.any())
                    g['joint_quantile_adoption']=bool(joint.any())
                if not all(np.array_equal(groups[0]['candidate_scales'],g['candidate_scales']) for g in groups):
                    raise RuntimeError('Center candidate normalization changed between quantiles')
    return policies

def comparison_table(policies):
    rows=[]
    for step in next(iter(policies.values())).payload['groups']:
        bounds=[p.payload['groups'][step]['window_threshold'] for p in policies.values()]
        for q,policy in policies.items():
            g=policy.payload['groups'][step]
            position_rank=policy.config.gate_scope=='position_rank_v4'
            identical=(all(p.payload['groups'][step]['contexts'][key]['thresholds']==g['contexts'][key]['thresholds']
                for p in policies.values() for key in g['contexts']) if position_rank else all(v==bounds[0] for v in bounds))
            rows.append(dict(step=step,nominal_quantile=q,window_threshold=g['window_threshold'],
                unrelaxed_window_threshold=g.get('calibrated_window_threshold',g['window_threshold']),
                threshold_multiplier=policy.config.threshold_multiplier,
                threshold_is_unmodified_empirical_quantile=policy.config.threshold_multiplier==1.,
                validation_runs=g['validation_runs'],control_rate_resolution=g['control_rate_resolution'],
                requested_order_statistic=g.get('requested_order_statistic'),used_order_statistic=g.get('used_order_statistic'),
                center_calibration_runs=g.get('calibration_validation_runs'),
                center_used_order_statistic=g.get('center_used_order_statistic'),
                insufficient_tail_support=g['insufficient_controls_for_nominal_rate'],
                thresholds_identical=identical,verified_fault_probability=False,
                threshold_driver_sensor=g.get('threshold_driver_sensor'),threshold_driver_operation=g.get('threshold_driver_operation'),
                threshold_driver_raw_score=g.get('threshold_driver_raw_score'),threshold_driver_scale=g.get('threshold_driver_scale'),
                gate_scope=policy.config.gate_scope,center_gate_adopted=None if position_rank else g.get('center_gate_adopted',False),
                audit_runs=g.get('audit_runs'),new_audit_alert_runs=g.get('new_audit_alert_runs'),
                threshold_usage=('same cohort/position center bounds; no legacy fallback' if position_rank else
                    'accepted centers use view bounds; rejected centers use legacy' if g.get('center_gate_adopted') else 'full-window bound'),
                center_threshold_min=float(np.min(g['center_thresholds'])) if g.get('center_gate_adopted') else None,
                center_threshold_max=float(np.max(g['center_thresholds'])) if g.get('center_gate_adopted') else None))
    return pd.DataFrame(rows)


def center_audit_table(policies):
    """Actual selected-center acceptance, not misleading all-center OR counts."""
    rows=[]
    for q,policy in policies.items():
        for step,g in policy.payload['groups'].items():
            for sid,sensor in enumerate(g['sensors']):
                if 'center_adoption_mask' not in g:continue
                adopted=bool(g['center_adoption_mask'][sid])
                rows.append(dict(step=step,sensor=sensor,nominal_quantile=q,
                    center_gate_adopted=adopted,audit_runs=g['audit_runs'],
                    candidate_audit_alerts=g['center_audit_alerts'][sid],
                    baseline_audit_alerts=g['center_baseline_audit_alerts'][sid],
                    new_audit_alert_runs=g['center_new_audit_alert_runs'][sid],
                    active_self_threshold=g['center_thresholds'][sid][0] if adopted else g['window_threshold'],
                    active_relationship_threshold=g['center_thresholds'][sid][1] if adopted else g['window_threshold'],
                    active_gate='center views' if adopted else 'legacy fallback',
                    audit_scope='one UI center; combined self/pair; sampled windows per run',
                    verified_fault_probability=False))
    return pd.DataFrame(rows)


def position_calibration_table(policies):
    """Per-context bounds and independent support; do not show banks/rows."""
    rows=[]
    for q,policy in policies.items():
        if policy.config.gate_scope!='position_rank_v4':continue
        for step,g in policy.payload['groups'].items():
            for context,c in g['contexts'].items():
                for sid,sensor in enumerate(g['sensors']):
                    rows.append(dict(step=step,context=context,sensor=sensor,nominal_quantile=q,
                        train_runs=c['train_runs'],validation_runs=c['validation_runs'],supported=c['supported'],
                        available_train_runs=c.get('available_train_runs'),available_validation_runs=c.get('available_validation_runs'),
                        support_failure=c.get('support_failure'),
                        self_threshold=c['thresholds'][sid][0] if c['thresholds'] is not None else None,
                        relationship_threshold=c['thresholds'][sid][1] if c['thresholds'] is not None and policy.config.operation_profile!='core' else None,
                        operation_profile=policy.config.operation_profile,active_views=1 if policy.config.operation_profile=='core' else 2,
                        untouched_calibration_alert_rate=c['validation_window_alert_rate'][sid] if c['validation_window_alert_rate'] is not None else None,
                        insufficient_tail_support=c['insufficient_tail_support'],used_order_statistic=c['used_order_statistic'],
                        requested_control_alert_budget=c.get('requested_control_alert_budget',policy.config.nominal_window_alert_rate),
                        minimum_validation_runs_for_budget=c.get('minimum_validation_runs_for_budget'),
                        control_budget_resolved=c.get('control_budget_resolved',not c['insufficient_tail_support']),
                        score_meaning='rank plus TRAIN-support excess; not probability',legacy_fallback=False))
    return pd.DataFrame(rows)

def load_comparison(encoder_root,backbone_path,config=WindowConfig()):
    policies={}
    for q in configured_quantiles(config):
        path=comparison_policy_path(backbone_path,q)
        if not path.is_file():raise FileNotFoundError('Run STEP 3 threshold-comparison preparation first')
        policy=load_policy(path,identity_for(encoder_root,backbone_path,replace(config,nominal_window_alert_rate=1-q)))
        if policy.payload.get('threshold_comparison')!=comparison_identity():raise ValueError('Comparison source/contract changed; use a new artifact version')
        policies[q]=policy
    return policies

@torch.no_grad()
def prepare_comparison(encoders,encoder_root,backbone_path,device,numeric_state,*,config=WindowConfig()):
    config.validate()  # Fail before cache I/O or any encoder execution.
    quantiles=configured_quantiles(config)
    paths={q:comparison_policy_path(backbone_path,q) for q in quantiles}
    if all(p.is_file() for p in paths.values()):return load_comparison(encoder_root,backbone_path,config)
    source=comparison_identity();identities={q:identity_for(encoder_root,backbone_path,replace(config,nominal_window_alert_rate=1-q)) for q in quantiles}
    partitions={'train':[],'validation':[]}
    for step,(encoder,meta) in encoders.items():
        encoder.to(device).eval()
        try:
            for split in partitions:
                print(f'Threshold comparison: {step} / {split}',flush=True)
                for row in cached_comparisons(encoder_root,step,split=split,config=config,calibration=True):
                    a=torch.tensor(row['current'][None],dtype=torch.float32,device=device)
                    b=torch.tensor(row['reference'][None],dtype=torch.float32,device=device)
                    state=numeric_state(encoder,a,b)
                    partitions[split].append(dict(row,step=step,sensors=list(meta['sensor_columns']),
                        residual=residual_trace(state,len(meta['sensor_columns']),len(row['current']))))
        finally:encoder.cpu()
    policies=fit_comparison(partitions['train'],partitions['validation'],
        lambda cfg:identities[min(quantiles,key=lambda q:abs((1-q)-cfg.nominal_window_alert_rate))],config)
    if source!=comparison_identity():raise RuntimeError('Comparison source changed during preparation')
    for q,policy in policies.items():
        if identity_for(encoder_root,backbone_path,policy.config)!=identities[q]:raise RuntimeError('Artifact inputs changed during preparation')
    for q,policy in policies.items():
        path=paths[q]
        if path.exists():
            existing=load_policy(path,identities[q])
            if existing.payload!=policy.payload:raise ValueError('Existing comparison differs; never overwrite')
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x',encoding='utf-8') as stream:json.dump(policy.payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return policies

def prepare_runtime_comparison(runtime,settings,*,seed=42):
    """Explicit STEP4 preparation from TRAIN/Validation caches, never Test.

    Uses the same configuration and strict no-overwrite preparation as STEP3.
    Does not train weights, change thresholds or substitute a neural decision.
    """
    if hasattr(runtime,'external_pairs'):
        raise ValueError('External calibration requires its own preparation path')
    q=float(settings['active_quantile'])
    if not 0<q<1:raise ValueError('Active quantile must lie inside (0,1)')
    config=WindowConfig(gate_scope=settings.get('gate_profile','full_window'),
        nominal_window_alert_rate=settings.get('control_alert_budget',1-q),
        operation_profile=settings.get('operation_profile','all'),
        noise_scale_floor=settings.get('gate_profile')=='center_views_v3',
        threshold_multiplier=settings.get('threshold_multiplier',1.),
        max_calibration_windows_per_step=settings.get('max_calibration_windows_per_step',2048),seed=seed)
    config.validate()
    module=runtime.module
    print('Preparing missing numerical criteria from TRAIN/Validation caches only; frozen weights, no Test fitting.',flush=True)
    policies=prepare_comparison(runtime.encoders,module.ENCODER_ROOT,module.BACKBONE_PATH,
        module.DEVICE,module.numeric_state,config=config)
    from .pipeline_settings import activate_prepared_policy
    from .window_policy import MODE
    path=activate_prepared_policy(module.BACKBONE_PATH,q)
    module.DISCOVERY_POLICY=policies[q]
    module.DISCOVERY_POLICY_PATH=path
    module.DISCOVERY_POLICY_MODE=MODE
    return policies


def run_threshold_comparison(root,*,runtime=None,split='validation',pairs_per_step=2,max_cases=14,seed=42,steps=None):
    from .reporting import ready_runtime,ExperimentReport
    from .discovery_reporting import run_discovery_experiment
    runtime=ready_runtime(root) if runtime is None else runtime
    module=runtime.module;original=module.DISCOVERY_POLICY
    if original is None:raise RuntimeError('Prepare STEP 3 before comparison')
    policies=load_comparison(module.ENCODER_ROOT,module.BACKBONE_PATH,replace(original.config,nominal_window_alert_rate=.05))
    summaries=[];by_step=[];by_kind=[];reports={};dynamic=[];paired=[];operations=[];support=[];paired_summary=[];challenge_controls=[];relaxation=[]
    learned_path=os.environ.pop('GEVIS_CENTER_DECISION_PATH',None)
    try:
        for q,policy in policies.items():
            module.DISCOVERY_POLICY=policy
            report=run_discovery_experiment(root,runtime=runtime,split=split,pairs_per_step=pairs_per_step,max_cases=max_cases,seed=seed,steps=steps)
            reports[q]=report
            if 'unrelaxed_alert' in report.details:
                own=report.details[report.details.task.eq('self')]
                for step,group in own.groupby('step',sort=False):
                    controls=group[~group.injected];positive=group[group.injected & group.effective.eq(True)]
                    relaxation.append(dict(step=step,nominal_quantile=q,threshold_multiplier=policy.config.threshold_multiplier,
                        control_windows=len(controls),effective_cases=len(positive),
                        untouched_rate_before_relaxation=controls.unrelaxed_alert.mean(),untouched_rate_after_relaxation=controls.alert.mean(),
                        recall_before_relaxation=positive.unrelaxed_alert.mean(),recall_after_relaxation=positive.alert.mean(),
                        meaning='same frozen numerical scores; controls are not verified normal labels'))
            row=report.summary.copy();active=report.details[report.details.injected & report.details.effective.eq(True) & report.details.alert]
            row['sensor_accuracy_when_alerted']=float(active.top_sensor_correct.mean()) if len(active) else np.nan
            row['alerted_injected_cases']=len(active);row['nominal_quantile']=q
            row['calibration_overlap']=split=='validation';summaries.append(row)
            by_step.append(report.by_step.assign(nominal_quantile=q))
            by_kind.append(report.diagnostics['by_kind_and_strength'].assign(nominal_quantile=q))
            if 'dynamic_challenge' in report.diagnostics:dynamic.append(report.diagnostics['dynamic_challenge'].assign(nominal_quantile=q))
            for key,rows in (('paired_change_diagnosis',paired),('operation_score_changes',operations),('experimental_support',support),
                             ('paired_change_summary',paired_summary),('dynamic_challenge_controls',challenge_controls)):
                if key in report.diagnostics:rows.append(report.diagnostics[key].assign(nominal_quantile=q))
        columns=['step','pair_id','kind','strength','expected_sensor','effective','raw_attention_sensor_correct']
        if not all(next(iter(reports.values())).details[columns].equals(r.details[columns]) for r in reports.values()):
            raise RuntimeError('Paired cases or unfiltered attention changed; comparison invalid')
    finally:
        module.DISCOVERY_POLICY=original
        if learned_path is not None:os.environ['GEVIS_CENTER_DECISION_PATH']=learned_path
    return ExperimentReport('Paired 90/95/99 window threshold comparison',pd.concat(summaries,ignore_index=True),
        pd.concat(by_step,ignore_index=True),diagnostics={'thresholds':comparison_table(policies),
        'center_gate_audit':center_audit_table(policies),
        'position_calibration':position_calibration_table(policies),
        'paired_change_diagnosis':pd.concat(paired,ignore_index=True) if paired else pd.DataFrame(),
        'operation_score_changes':pd.concat(operations,ignore_index=True) if operations else pd.DataFrame(),
        'experimental_support':pd.concat(support,ignore_index=True) if support else pd.DataFrame(),
        'paired_change_summary':pd.concat(paired_summary,ignore_index=True) if paired_summary else pd.DataFrame(),
        'dynamic_challenge_controls':pd.concat(challenge_controls,ignore_index=True) if challenge_controls else pd.DataFrame(),
        'relaxation_effect':pd.DataFrame(relaxation),
        'by_kind_and_strength':pd.concat(by_kind,ignore_index=True),
        'dynamic_challenge':pd.concat(dynamic,ignore_index=True) if dynamic else pd.DataFrame()},
        metadata={'same_cases':True,'raw_attention_unchanged':True,'weights_updated':False,
                  'test_used_for_fitting':False,'split':split,'calibration_overlap':split=='validation',
                  'learned_head_bypassed_for_threshold_only_comparison':True,
                  'verified_fault_probability':False,'meaning':'All injected truths retained; no retrospective relabeling.'})
