"""One explicit variant switch shared by all notebook/runtime consumers."""
from dataclasses import replace
import os

PRETRAINED_RUN='real_step1_pretrained_reference_v11'
ENCODER_SUFFIX='_hf_patchtst_v1'
OFFICIAL_RUN='real_step1_official_reference_v11'
OFFICIAL_SUFFIX='_hf_patchtst_official_domain_v2'
OFFICIAL_BASE_SUFFIX='_hf_patchtst_official_v1'
HYBRID_SUFFIX='_hf_patchtst_joint_hybrid_v1'
HYBRID_RUN='real_step1_joint_hybrid_reference_v11'
PORTABLE_SUFFIX='_hf_patchtst_portable_joint_v1'
PORTABLE_RUN='real_step1_portable_reference_v11'
TEMPORAL_SUFFIX='_native_temporal_blind_v2'
TEMPORAL_RUN='real_step1_native_temporal_blind_v2'
MULTISCALE_SUFFIX='_native_multiscale_blind_v3'
MULTISCALE_RUN='real_step1_native_multiscale_blind_v3'

def selected_variant():
    value=os.environ.get('GEVIS_MODEL_VARIANT','hybrid')
    if value not in ('legacy','pretrained','official','hybrid','portable','temporal','multiscale'): raise ValueError('Unknown GEVIS_MODEL_VARIANT')
    return value

def active_config(config):
    if selected_variant() in ('temporal','multiscale'):
        multiscale=selected_variant()=='multiscale'
        suffix=MULTISCALE_SUFFIX if multiscale else TEMPORAL_SUFFIX
        steps=[]
        for step in config.steps:
            # Aim for finer patches in every Step without exceeding 64 tokens.
            candidates=[p for p in range(2,9) if step.window_length%p==0 and 8<=step.window_length//p<=64]
            if not candidates:raise ValueError('Temporal candidate needs 8..64 complete native patches')
            patch=min(candidates,key=lambda p:(abs(p-4),p)) if multiscale else max(candidates)
            steps.append(replace(step,patch_length=patch))
        base=config.artifacts.encoder
        for previous in (MULTISCALE_SUFFIX,TEMPORAL_SUFFIX,PORTABLE_SUFFIX,HYBRID_SUFFIX,OFFICIAL_SUFFIX,OFFICIAL_BASE_SUFFIX,ENCODER_SUFFIX):base=base.removesuffix(previous)
        return replace(config,steps=tuple(steps),artifacts=replace(config.artifacts,encoder=base+suffix))
    if selected_variant()=='portable':
        if config.artifacts.encoder.endswith(PORTABLE_SUFFIX):return config
        base=config.artifacts.encoder
        for suffix in (PORTABLE_SUFFIX,HYBRID_SUFFIX,OFFICIAL_SUFFIX,OFFICIAL_BASE_SUFFIX,ENCODER_SUFFIX):base=base.removesuffix(suffix)
        return replace(config,artifacts=replace(config.artifacts,encoder=base+PORTABLE_SUFFIX))
    if selected_variant()=='hybrid':
        if config.artifacts.encoder.endswith(HYBRID_SUFFIX):return config
        base=config.artifacts.encoder
        for suffix in (PORTABLE_SUFFIX,OFFICIAL_SUFFIX,OFFICIAL_BASE_SUFFIX,ENCODER_SUFFIX):base=base.removesuffix(suffix)
        return replace(config,artifacts=replace(config.artifacts,encoder=base+HYBRID_SUFFIX))
    if selected_variant()=='official':
        if config.artifacts.encoder.endswith(OFFICIAL_SUFFIX):return config
        steps=[]
        for step in config.steps:
            length=step.window_length//12*12
            if length<24:raise ValueError('Official PatchTST needs at least two native 12-sample patches per window')
            steps.append(replace(step,window_length=length,patch_length=12,stride=min(step.stride,length)))
        base=config.artifacts.encoder.removesuffix(OFFICIAL_BASE_SUFFIX).removesuffix(ENCODER_SUFFIX)
        return replace(config,steps=tuple(steps),artifacts=replace(config.artifacts,encoder=base+OFFICIAL_SUFFIX))
    if selected_variant()=='legacy' or config.artifacts.encoder.endswith(ENCODER_SUFFIX): return config
    return replace(config,artifacts=replace(config.artifacts,encoder=config.artifacts.encoder+ENCODER_SUFFIX))

def active_run_tag():
    profile=os.environ.get('GEVIS_DISCOVERY_TRAINING','reference_v11')
    if profile not in ('reference_v11','contrastive_v12','fixed_questions_v13','center_relation_v14'):raise ValueError('Unknown discovery training profile')
    if profile in ('contrastive_v12','fixed_questions_v13','center_relation_v14'):
        base={'multiscale':MULTISCALE_RUN,'temporal':TEMPORAL_RUN,'portable':PORTABLE_RUN,'hybrid':HYBRID_RUN,'official':OFFICIAL_RUN,
              'pretrained':PRETRAINED_RUN,'legacy':'real_step1_reference_discovery_v11'}[selected_variant()]
        return base+'_'+profile
    if selected_variant()=='multiscale':return MULTISCALE_RUN
    if selected_variant()=='temporal':return TEMPORAL_RUN
    if selected_variant()=='portable':return PORTABLE_RUN
    if selected_variant()=='hybrid':return HYBRID_RUN
    if selected_variant()=='official':return OFFICIAL_RUN
    return PRETRAINED_RUN if selected_variant()=='pretrained' else 'real_step1_reference_discovery_v11'
