"""Describe what executed, separately from which checkpoints were loaded."""
PATHS={
    'learned_answers':{'step1_encoder_executed':True,'step2_cross_attention_executed':True,
        'sensor_source':'explicit center with all context sensors retained',
        'change_decision_source':'trained STEP3 decoder softmax; no numerical discovery fallback',
        'interval_source':'input window only; no learned event-boundary decoder',
        'fault_probability_calibrated':False},
    'center_discovery':{'step1_encoder_executed':True,'step2_cross_attention_executed':True,
        'sensor_source':'UI center input; learned attention weighted pair evidence ranks partners',
        'change_decision_source':'calibrated numerical evidence, not trained STEP2 detection',
        'interval_source':'operation traces; learned attention does not decode event boundaries'},
    'named_question':{'step1_encoder_executed':True,'step2_cross_attention_executed':True,
        'sensor_source':'learned question-conditioned sensor selection',
        'interval_source':'numeric pointwise evidence; semantic scope is whole/early/late, not an interval decoder'},
    'blind_discovery':{'step1_encoder_executed':True,'step2_cross_attention_executed':True,
        'sensor_source':'sensor-free generic-question Cross-Attention, constrained by calibrated numerical evidence',
        'change_decision_source':'calibrated numerical evidence, not trained STEP2 detection',
        'interval_source':'winning operation numerical trace; learned patch is a separate diagnostic'},
    'mixed_baselines':{'step1_encoder_executed':True,'step2_cross_attention_executed':True,
        'sensor_source':'method-dependent; named scan and blind discovery are distinct rows',
        'interval_source':'not a common localization comparison'},
}


def execution_contract(runtime,path):
    if path not in PATHS:raise ValueError('Unknown execution path')
    m=runtime.module;checkpoint=getattr(m,'GEVIS_CHECKPOINT',{})
    policy=getattr(m,'REFERENCE_POLICY',None)
    detail=dict(PATHS[path])
    if path=='center_discovery':
        import os
        selected=os.environ.get('GEVIS_CENTER_DECISION_PATH')
        if selected:
            detail.update(change_decision_source='learned STEP3 head calibrated on separate Validation controls',
                sensor_source='UI center plus learned partner logits; frozen STEP2 attention is an input',
                interval_source='learned temporal head conditional on learned alert',center_decision_path=str(selected))
    if path=='blind_discovery' and getattr(m,'DISCOVERY_POLICY_MODE',None)=='window_calibrated_v2':
        detail.update(sensor_source='generic-question Cross-Attention among Validation-calibrated events',
            interval_source='same winning operation local trace; learned patch is separate',
            window_policy_path=str(getattr(m,'DISCOVERY_POLICY_PATH','not exposed')))
    if path in ('center_discovery','blind_discovery') and getattr(getattr(getattr(m,'DISCOVERY_POLICY',None),'config',None),'gate_scope',None)=='position_rank_v4':
        detail.pop('center_decision_path',None)
        detail.update(change_decision_source='TRAIN context/operation rank plus tail excess; same-context Validation center bounds; no legacy fallback',
            interval_source='same rank-transformed operation traces; no additional neural forward',
            gate_profile='position_rank_v4',fault_probability_calibrated=False)
    if path in ('center_discovery','blind_discovery'):
        if getattr(getattr(m,'DISCOVERY_POLICY',None),'decision_kind',None)=='direct_gmm':
            detail.update(change_decision_source='TRAIN two-component GMM posterior argmax; no q99 veto',
                gate_profile='direct_gmm',fault_probability_calibrated=False,
                interval_source='GMM is a window decision; no numerical event boundary')
        from .gevis_system import SYSTEM
        detail.update(gevis_system=SYSTEM,learned_answer_head_is_detector=False)
        if getattr(m,'INTERVAL_MODE','numeric')=='learned':
            detail['interval_source']=('frozen PatchTST latent temporal segmentation; numeric intervals retained for comparison'
                if getattr(m,'TEMPORAL_LOCALIZER',None) else 'UNAVAILABLE: prepare STEP3 temporal localizer; no numeric substitution')
    return {'execution_path':path,**detail,
        'loaded_backbone_sha256':checkpoint.get('_artifact_sha256'),
        'backbone_path':str(getattr(m,'BACKBONE_PATH','not exposed')),
        'encoder_root':str(getattr(m,'ENCODER_ROOT','not exposed')),
        'encoder_backends':sorted({meta.get('encoder_backend','legacy') for _,meta in runtime.encoders.values()}),
        'reference_policy_path':str(getattr(m,'REFERENCE_POLICY_PATH','not exposed')),
        'reference_manifest_sha256':policy.payload.get('identity',{}).get('manifest_sha256') if policy else None,
        'backbone_loaded_does_not_imply_executed':True}
