"""Explicit notebook profile; no weights/data are written or loaded here."""
import json,os
from pathlib import Path

PROFILES=('reference_v11','contrastive_v12','fixed_questions_v13','center_relation_v14')

def configure_pipeline(root):
    root=Path(root)
    settings=json.loads((root/'config/experiments/gevis_pipeline.json').read_text(encoding='utf-8'))
    if settings.get('decision_system','hybrid') not in ('hybrid','fusion_answers'):
        raise ValueError('Unknown decision system: select fusion_answers or the explicit hybrid comparator')
    os.environ['GEVIS_DECISION_SYSTEM']=settings.get('decision_system','hybrid')
    if 'model_variant' in settings:os.environ['GEVIS_MODEL_VARIANT']=settings['model_variant']
    os.environ['GEVIS_GATE_PROFILE']=settings.get('gate_profile','full_window')
    os.environ['GEVIS_OPERATION_PROFILE']=settings.get('operation_profile','all')
    multiplier=float(settings.get('threshold_multiplier',1.))
    if not 0<multiplier<=1:raise ValueError('Threshold multiplier must lie inside (0,1]')
    os.environ['GEVIS_THRESHOLD_MULTIPLIER']=str(multiplier)
    profile=settings['training_profile']
    budget=float(settings.get('control_alert_budget',1-float(settings['active_quantile'])))
    if not 0<budget<1:raise ValueError('Control alert budget must lie inside (0,1)')
    q=round(1-budget,12);settings['active_quantile']=q
    settings['control_alert_budget']=budget
    os.environ['GEVIS_CONTROL_ALERT_BUDGET']=str(budget)
    if profile not in PROFILES:raise ValueError('Known training profile required')
    os.environ['GEVIS_INTERVAL_MODE']=settings.get('interval_mode','numeric')
    if settings.get('comparison_quantiles') not in ([.95,.99],[.90,.95,.99]):raise ValueError('Known paired comparison quantiles required')
    os.environ['GEVIS_DISCOVERY_TRAINING']=profile
    from pecvd.model_variants import active_config,active_run_tag
    from pecvd.process_config import load_process_config
    from .process_inputs import configured_process_path,candidate_backbone_path
    config=active_config(load_process_config(configured_process_path(root)))
    backbone=candidate_backbone_path(root,config,active_run_tag())
    os.environ['GEVIS_DISCOVERY_MODE']='window_calibrated_v2'
    os.environ['GEVIS_DISCOVERY_POLICY_PATH']=str(comparison_policy_path(backbone,q))
    mode=settings.get('center_decision_mode','off')
    if mode not in ('off','validated'):raise ValueError('Unknown learned center decision mode')
    if mode=='validated' and settings.get('gate_profile')!='position_rank_v4':
        from .center_decision import candidate_path
        candidate=candidate_path(backbone)
        if candidate.is_file():os.environ['GEVIS_CENTER_DECISION_PATH']=str(candidate)
        else:os.environ.pop('GEVIS_CENTER_DECISION_PATH',None)
    else:os.environ.pop('GEVIS_CENTER_DECISION_PATH',None)
    return dict(settings,backbone_run=active_run_tag(),decision_meaning=(
        'fixed question + selected-center Fusion -> trained answer logits; no numerical fallback'
        if settings.get('decision_system')=='fusion_answers' else 'preserved numerical comparator'),threshold_meaning=
        ('same-context TRAIN rank/tail score with Validation center bounds; not a probability' if settings.get('gate_profile')=='position_rank_v4' else
         f'empirical run-max bound x {multiplier:g}; not a nominal error guarantee or anomaly probability'))

def comparison_policy_path(backbone_path,quantile):
    if not 0<float(quantile)<1:raise ValueError('Quantile must lie inside (0,1)')
    suffix='_v1'
    profile=os.environ.get('GEVIS_GATE_PROFILE')
    if profile in ('center_views_v3','position_rank_v4'):
        import hashlib
        h=hashlib.sha256()
        from .discovery_calibration import policy_source_paths,numeric_runtime_source_digest
        for path in (*policy_source_paths(),Path(__file__),Path(__file__).with_name('threshold_comparison.py')):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        # Only executable numeric source, never saved notebook outputs. The
        # artifact directory identifies the project root for all shared callers.
        for root in Path(backbone_path).resolve().parents:
            if (root/'notebooks/STEP_3.1_run_gevis_inference.ipynb').is_file():
                h.update(str(numeric_runtime_source_digest(root)).encode())
                settings=root/'config/experiments/gevis_pipeline.json'
                if settings.is_file():h.update(settings.read_bytes())
                break
        h.update(os.environ.get('GEVIS_THRESHOLD_MULTIPLIER','1.0').encode())
        h.update(os.environ.get('GEVIS_OPERATION_PROFILE','all').encode())
        suffix=('_position_rank_v4_' if profile=='position_rank_v4' else '_center_v3_')+h.hexdigest()[:10]
    label=f'{quantile*100:.10g}'.replace('.','p')
    return Path(backbone_path).parent/f'window_evidence_compare_q{label}{suffix}.json'


def activate_prepared_policy(backbone_path, quantile):
    """Bind inference to the policy actually prepared for this notebook backbone.

    Never fit here, silently choose another quantile, or overwrite artifacts.
    Runtime loading still verifies the complete policy identity.
    """
    path=comparison_policy_path(backbone_path,quantile).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing prepared STEP 3 policy: {path}. '
            'Run the window-evidence-build cell (prepare_comparison) before '
            'discovery evaluation. STEP 1/2 retraining is not required.')
    os.environ['GEVIS_DISCOVERY_MODE']='window_calibrated_v2'
    os.environ['GEVIS_DISCOVERY_POLICY_PATH']=str(path)
    return path
