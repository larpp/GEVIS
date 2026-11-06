"""Historical training program; explicit execution only. Never import to load weights."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

def display(value):
    print(value)

# Shared settings: edit config/experiments/gevis_pipeline.json, then restart kernels.
from pathlib import Path
import sys, os
PIPELINE_ROOT=next(p for p in (Path.cwd().resolve(),*Path.cwd().resolve().parents) if (p/'src'/'pecvd').is_dir())
if str(PIPELINE_ROOT/'src') not in sys.path: sys.path.insert(0,str(PIPELINE_ROOT/'src'))
import importlib
for _name in ("pecvd.model_variants", "pecvd.experiments.questions", "pecvd.experiments.pipeline_settings"):
    importlib.reload(importlib.import_module(_name))
from pecvd.experiments.pipeline_settings import configure_pipeline
PIPELINE_SETTINGS=configure_pipeline(PIPELINE_ROOT)
print({k:PIPELINE_SETTINGS[k] for k in ('model_variant','training_profile','backbone_revision','decision_system') if k in PIPELINE_SETTINGS})

from pecvd.experiments.questions import PRIMARY_GOAL_VERSION
from pecvd.experiments.fusion_answers import SYSTEM as GEVIS_SYSTEM, KINDS, QUESTIONS
if PIPELINE_SETTINGS.get('decision_system')!='fusion_answers':
    raise RuntimeError('Active notebooks require decision_system=fusion_answers.')
print({'question_contract':PRIMARY_GOAL_VERSION,'gevis_system':GEVIS_SYSTEM,'decision_path':'learned Fusion readout',
       'questions':{k:QUESTIONS[k] for k in KINDS},'answer_types':KINDS,
       'training':'STEP3 TRAIN readout; Validation selection; frozen STEP1/2',
       'interval_mode':'unavailable in this answer path',
       'probabilities':'softmax scores; not calibrated fault probabilities'})



from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict
import copy, hashlib, json, math, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from pecvd.process_config import load_process_config
from pecvd.experiments.process_inputs import configured_process_path,step_description
PROCESS_CONFIG_PATH=configured_process_path(ROOT)
import os, importlib
# Explicit notebook profile. Change this line only for an intentional legacy comparison.
NOTEBOOK_MODEL_VARIANT='hybrid'
NOTEBOOK_MODEL_VARIANT=globals().get('PIPELINE_SETTINGS',{}).get('model_variant',NOTEBOOK_MODEL_VARIANT)
os.environ['GEVIS_MODEL_VARIANT']=NOTEBOOK_MODEL_VARIANT
import pecvd.model_variants as model_variants
importlib.reload(model_variants)  # A warm kernel may still hold the previous default.
from pecvd.model_variants import active_config,active_run_tag,selected_variant
import pecvd.official_patchtst as official_patchtst
import pecvd.patchtst_adaptation as patchtst_adaptation
importlib.reload(official_patchtst)
importlib.reload(patchtst_adaptation)
import pecvd.hybrid_patchtst as hybrid_patchtst
import pecvd.patchtst_training as patchtst_training
importlib.reload(hybrid_patchtst)
importlib.reload(patchtst_training)
PROCESS_CONFIG=active_config(load_process_config(PROCESS_CONFIG_PATH))
CONFIGURED_MODEL_VARIANT=selected_variant()
DATA_PATH=ROOT/PROCESS_CONFIG.data_path
ARTIFACT_VERSION=PROCESS_CONFIG.artifacts.encoder
ARTIFACT_ROOT=ROOT/'artifacts'/ARTIFACT_VERSION
LEGACY_ENCODER_ROOT=ROOT/'artifacts'/load_process_config(PROCESS_CONFIG_PATH).artifacts.encoder
RUN_COLUMN=PROCESS_CONFIG.columns.run; STEP_COLUMN=PROCESS_CONFIG.columns.step; ORDER_COLUMN=PROCESS_CONFIG.columns.order
STEPS=PROCESS_CONFIG.step_specs

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def save_torch(obj,path,overwrite=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not overwrite: raise FileExistsError(f'{path} exists; set OVERWRITE_ARTIFACTS=True intentionally')
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); tmp.replace(path)

def save_json(obj,path,overwrite=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not overwrite: raise FileExistsError(f'{path} exists; set OVERWRITE_ARTIFACTS=True intentionally')
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)

def derive_step_label(frame):
    if STEP_COLUMN in frame: return frame
    prefix=PROCESS_CONFIG.columns.step_one_hot_prefix; excluded=set(PROCESS_CONFIG.columns.step_one_hot_excluded)
    cols=[x for x in frame if x.startswith(prefix) and x not in excluded]
    if not cols: raise KeyError('step_label and step_name_* are both missing')
    values=frame[cols].fillna(0).to_numpy(); ix=values.argmax(1); strength=values.max(1)
    out=frame.copy(); labels=np.array([x.removeprefix(prefix) for x in cols],object)[ix]
    out[STEP_COLUMN]=np.where(strength>0,labels,None); return out

def select_sensor_columns(frame):
    from pecvd.experiments.process_inputs import configured_sensors
    explicit=configured_sensors(PROCESS_CONFIG,frame)
    if explicit is not None: return explicit
    forbidden={RUN_COLUMN,STEP_COLUMN,ORDER_COLUMN,'row_in_step','local_sample_id','global_sample_id','local_process_id','global_process_id','sample_order','sample_order_within_timestamp','timestamp_group_size','exact_value_group_size','is_exact_duplicate','step_elapsed_seconds','step_row_count','elapsed_seconds','elapsed_from_declared_start_seconds','elapsed_from_first_sample_seconds','data_split','step_segment_id','step_number','step_progress','previous_step_label','previous_step_number','previous_step_duration','previous_step_row_count','previous_step_end_order'}
    prefixes=('step_name_','source_group_','previous_step_','pm_cycle_','recipe_','label_','target_')
    cols=[x for x in frame.select_dtypes(include=np.number) if x not in forbidden and not x.startswith(prefixes)]
    if not cols: raise ValueError('No physical numeric sensor columns remain')
    return cols

def split_run_ids(frame,seed=SEED):
    ids=np.array(sorted(frame[RUN_COLUMN].dropna().unique().tolist()),dtype=object); np.random.default_rng(seed).shuffle(ids)
    n=len(ids); a=max(1,int(.70*n)); b=max(a+1,a+int(.15*n))
    result={'train':ids[:a].tolist(),'validation':ids[a:b].tolist(),'test':ids[b:].tolist()}
    sets={k:set(v) for k,v in result.items()}; assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    return result

from pecvd.legacy_patchtst import PatchEncoder  # Explicit legacy comparison only.
import pecvd.temporal_patchtst as temporal_patchtst
importlib.reload(temporal_patchtst)
import pecvd.multiscale_patchtst as multiscale_patchtst
importlib.reload(multiscale_patchtst)
if selected_variant()=='temporal': PatchEncoder=temporal_patchtst.TemporalPatchEncoder
elif selected_variant()=='multiscale': PatchEncoder=multiscale_patchtst.MultiscalePatchEncoder

def deterministic_mask(batch, tokens, ratio, seed, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    mask = (
        torch.rand(
            (batch, tokens),
            generator=generator,
            device=device,
        )
        < ratio
    )

    # Only for samples with no selected mask
    # mask one additional random token.
    empty_rows = ~mask.any(dim=1)

    if empty_rows.any():
        row_indices = empty_rows.nonzero(as_tuple=True)[0]

        fallback_tokens = torch.randint(
            low=0,
            high=tokens,
            size=(len(row_indices),),
            generator=generator,
            device=device,
        )

        mask[row_indices, fallback_tokens] = True

    return mask

def masked_metrics(model,x,ratio=.25,seed=0):
    target,_=model.patchify(x); mask=deterministic_mask(len(x),target.shape[1],ratio,seed,x.device)
    pred=model(x,mask); den=mask.sum().clamp_min(1)*target.shape[-1]
    mse=((pred-target).pow(2)*mask[...,None]).sum()/den
    zero=(target.pow(2)*mask[...,None]).sum()/den
    return mse,zero,mask,pred,target

def checkerboard_mask(model,batch,patches,parity,device):
    sensor=torch.arange(model.n_sensors,device=device)[:,None]; patch=torch.arange(patches,device=device)[None]
    return ((sensor+patch)%2==parity).reshape(1,-1).expand(batch,-1)

def reconstruction_mask(model,batch,patches,seed,device):
    # Mix random masks with the same blind checkerboard pattern used at evaluation.
    selector=np.random.default_rng(seed).random()
    if selector<CONFIG.checkerboard_probability:
        return checkerboard_mask(model,batch,patches,seed%2,device)
    return deterministic_mask(batch,model.n_sensors*patches,CONFIG.mask_ratio,seed,device)

def blind_reconstruction(model,x):
    target,patches=model.patchify(x); result=torch.zeros_like(target)
    if hasattr(model,'blind_prediction'):
        result=model.blind_prediction(x)
    elif hasattr(model,'pretraining_loss'):
        result=patchtst_adaptation.blind_prediction(model,x)
    else:
        for parity in (0,1):
            mask=checkerboard_mask(model,len(x),patches,parity,x.device); result=torch.where(mask[...,None],model(x,mask),result)
    latent=model.tokens(x).reshape(len(x),model.n_sensors,patches,model.d_model)
    reconstruction=result.reshape(len(x),model.n_sensors,patches,model.patch_len); residual=(reconstruction-target.reshape_as(reconstruction)).pow(2).mean(-1)
    return reconstruction,latent,residual

def rich_features(model,x):
    reconstruction,latent,residual=blind_reconstruction(model,x); patch,patches=model.patchify(x); patch=patch.reshape(len(x),model.n_sensors,patches,model.patch_len)
    statistics=torch.stack([patch.mean(-1),patch.std(-1,unbiased=False),(patch[...,-1]-patch[...,0])/max(1,model.patch_len-1)],-1)
    centered=patch-patch.mean(-1,keepdim=True); normalized=centered/centered.square().sum(-1,keepdim=True).clamp_min(1e-4).sqrt()
    relationship=normalized.permute(0,2,1,3)@normalized.permute(0,2,3,1)
    return {'latent':latent,'reconstruction':reconstruction,'residual':residual,'statistics':statistics,'relationship':relationship}

@dataclass(frozen=True)
class EncoderConfig:
    epochs:int=400 if selected_variant()=='multiscale' else 300 if selected_variant() in ('hybrid','portable','temporal','multiscale') else 80
    min_epochs:int=40
    warmup_epochs:int=5
    reuse_legacy_checkpoint:bool=selected_variant() not in ('temporal','multiscale')
    batch_size:int=32
    learning_rate:float=1e-4
    min_learning_rate:float=2e-6
    adam_eps:float=1e-8
    weight_decay:float=1e-4
    mask_ratio:float=.50
    checkerboard_probability:float=.50
    validation_interval:int=5
    patience:int=12
    scheduler_patience:int=3
    min_delta:float=1e-6
    d_model:int=128 if selected_variant()=='official' else 96
    nhead:int=16 if selected_variant()=='official' else 8
    layers:int=3
    dropout:float=.2 if selected_variant()=='official' else .05
    pretrained_last_layers:int|None=None if selected_variant() in ('official','hybrid','portable') else 1
    pretrained_learning_rate:float=1e-5
    variability_weight:float=.25
    peak_weight:float=.10
    clean_weight:float=0. if selected_variant()=='official' else .5
    slope_weight:float=.25
    relationship_weight:float=.10
    max_grad_norm:float=.5
    use_amp:bool=True
    amp_dtype:str='bfloat16'
    fused_adamw:bool=True
    num_workers:int=0
    overwrite_artifacts:bool=False

CONFIG=EncoderConfig()
SESSION_SIGNATURE='STEP1_GEVIS_HYBRID_REUSE_V6'
AMP_DTYPE={'bfloat16':torch.bfloat16,'float16':torch.float16}[CONFIG.amp_dtype]
# Existing complete bundle: reuse without touching weights/caches/manifest.
REUSE_SAVED_ENCODERS=(ARTIFACT_ROOT/'encoder_manifest.json').is_file()
RUN_TRAINING=not REUSE_SAVED_ENCODERS
SAVE_WINDOWS=True
print({'model_variant':CONFIGURED_MODEL_VARIANT,'device':str(DEVICE),'run_training':RUN_TRAINING,'reuse_saved_encoders':REUSE_SAVED_ENCODERS,'automatic_checkpoint_save':True,'save_windows':SAVE_WINDOWS,'artifact_root':str(ARTIFACT_ROOT)})

SOURCE_PREFIX='source_group_'; PM_PREFIX='pm_cycle_'

def one_hot_label(frame,prefix,excluded=()):
    columns=[c for c in frame.columns if c.startswith(prefix) and c not in excluded]
    if not columns: raise KeyError(f'No {prefix}* columns')
    values=frame[columns].fillna(0).to_numpy(); index=values.argmax(1); strength=values.max(1)
    labels=np.array([c.removeprefix(prefix) for c in columns],dtype=object)[index]
    return np.where(strength>0,labels,'unknown')

def annotate_process_structure(frame):
    out=derive_step_label(frame)
    out['_source_group']=one_hot_label(out,SOURCE_PREFIX,('source_group_nan',))
    out['_pm_cycle']=one_hot_label(out,PM_PREFIX)
    out['_cohort_key']=out['_source_group'].astype(str)+'|pm='+out['_pm_cycle'].astype(str)
    consistency=out.groupby(RUN_COLUMN).agg(source_n=('_source_group','nunique'),pm_n=('_pm_cycle','nunique'),cohort_n=('_cohort_key','nunique'),step_n=(STEP_COLUMN,'nunique'))
    if (consistency[['source_n','pm_n','cohort_n']]!=1).any().any(): raise ValueError('Source/PM cohort changes inside a process')
    expected=set(STEPS); observed=out.groupby(RUN_COLUMN)[STEP_COLUMN].agg(lambda x:set(x.dropna()))
    if not PROCESS_CONFIG.source_columns and not observed.map(lambda x:x==expected).all(): raise ValueError(f'Every process must contain the configured {len(STEPS)} steps')
    return out

def split_run_ids(frame,seed=SEED):
    rng=np.random.default_rng(seed); result={'train':[],'validation':[],'test':[]}
    run_table=frame[[RUN_COLUMN,'_cohort_key']].drop_duplicates()
    if run_table.groupby(RUN_COLUMN)._cohort_key.nunique().max()!=1: raise ValueError('A process belongs to multiple cohorts')
    for _,group in run_table.groupby('_cohort_key',sort=True):
        ids=np.array(sorted(group[RUN_COLUMN].tolist()),dtype=object); rng.shuffle(ids); n=len(ids); n_train=max(1,int(.70*n)); n_val=max(1,int(round(.15*n)))
        if n-n_train-n_val<1: n_val=max(1,n_val-1)
        result['train']+=ids[:n_train].tolist(); result['validation']+=ids[n_train:n_train+n_val].tolist(); result['test']+=ids[n_train+n_val:].tolist()
    sets={k:set(v) for k,v in result.items()}
    if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']: raise ValueError('Run split overlap')
    split_map={run:split for split,ids in result.items() for run in ids}; coverage=run_table.assign(split=run_table[RUN_COLUMN].map(split_map)).groupby(['_cohort_key','split']).size().unstack(fill_value=0)
    if (coverage==0).any().any(): raise ValueError('Every cohort needs train/validation/test processes')
    return result


if not REUSE_SAVED_ENCODERS:
    if not DATA_PATH.exists(): raise FileNotFoundError(DATA_PATH)
    from pecvd.experiments.process_inputs import load_process_frame
    frame=annotate_process_structure(load_process_frame(ROOT,PROCESS_CONFIG))
    for name in (RUN_COLUMN,STEP_COLUMN,ORDER_COLUMN):
        if name not in frame: raise KeyError(name)
    sensor_columns=select_sensor_columns(frame); sensor_descriptors={name:name for name in sensor_columns}; run_split=split_run_ids(frame)
    split_map={run:split for split,ids in run_split.items() for run in ids}
    run_structure=frame[[RUN_COLUMN,'_source_group','_pm_cycle','_cohort_key']].drop_duplicates()
    run_structure['split']=run_structure[RUN_COLUMN].map(split_map)
    cohort_split=run_structure.groupby(['_cohort_key','split']).size().unstack(fill_value=0)
    step_lengths=frame.groupby([RUN_COLUMN,STEP_COLUMN]).size().groupby(STEP_COLUMN).agg(['min','median','max'])
    print({'rows':len(frame),'runs':frame[RUN_COLUMN].nunique(),'split_runs':{k:len(v) for k,v in run_split.items()},'sensor_count':len(sensor_columns),'cohort_count':run_structure._cohort_key.nunique(),'source_sha256_prefix':sha256_file(DATA_PATH)[:12]})
    display(cohort_split); display(step_lengths); display(pd.DataFrame({'sensor':sensor_columns}))
else:
    from pecvd.encoder_reuse import load_saved_encoders
    _backend={'multiscale':'gevis_native_multiscale_blind_v3','temporal':'gevis_native_temporal_blind_v2','hybrid':'hf_patchtst_legacy_joint_v1','portable':'hf_patchtst_portable_joint_v1','official':'hf_patchtst_official_ssl_v1','pretrained':'hf_patchtst_rich_ssl_v1','legacy':'legacy'}[CONFIGURED_MODEL_VARIANT]
    SAVED_MANIFEST,SAVED_ENCODERS=load_saved_encoders(ARTIFACT_ROOT,PROCESS_CONFIG,backend=_backend,source_path=DATA_PATH)
    sensor_columns=SAVED_MANIFEST['sensor_columns'];sensor_descriptors=SAVED_MANIFEST['sensor_descriptors']
    run_split=SAVED_MANIFEST['run_split'];cohort_split=pd.DataFrame(SAVED_MANIFEST['cohort_split_counts'])
    print('Reusing saved STEP 1: exact weights and manifest retained; no training or CSV parsing.')


if not REUSE_SAVED_ENCODERS:
    def make_windows(step,split):
        length,_,stride=STEPS[step]; allowed=set(run_split[split]); part=frame[frame[RUN_COLUMN].isin(allowed)&frame[STEP_COLUMN].eq(step)]
        windows=[]; metadata=[]
        for run,group in part.groupby([RUN_COLUMN,'_gevis_segment_id'] if '_gevis_segment_id' in part else RUN_COLUMN,sort=True):
            if isinstance(run,tuple): run=run[0]
            group=group.sort_values(ORDER_COLUMN); values=group[sensor_columns].to_numpy(np.float32); orders=group[ORDER_COLUMN].to_numpy(); total=len(group)
            source=str(group['_source_group'].iloc[0]); pm=str(group['_pm_cycle'].iloc[0]); cohort=str(group['_cohort_key'].iloc[0]); denominator=max(1,total-length)
            for start in range(0,max(0,total-length+1),stride):
                window=values[start:start+length]
                if not np.isfinite(window).all(): continue
                ratio=float(group['_gevis_position_ratio'].iloc[start]) if '_gevis_position_ratio' in group else start/denominator; windows.append(window); metadata.append({'process_id':str(run),'source_group':source,'pm_cycle':pm,'cohort_key':cohort,'window_start':int(orders[start]),'window_end':int(orders[start+length-1]),'position_ratio':float(ratio),'position_bin':min(9,int(ratio*10))})
        array=np.stack(windows) if windows else np.empty((0,length,len(sensor_columns)),np.float32)
        return array,pd.DataFrame(metadata)

    raw_windows={}; window_metadata={}; rows=[]
    for step in STEPS:
        raw_windows[step]={}; window_metadata[step]={}
        for split in ('train','validation','test'):
            raw_windows[step][split],window_metadata[step][split]=make_windows(step,split); meta=window_metadata[step][split]
            rows.append({'step':step,'split':split,'windows':len(meta),'processes':meta.process_id.nunique() if len(meta) else 0,'cohorts':meta.cohort_key.nunique() if len(meta) else 0,'position_bins':meta.position_bin.nunique() if len(meta) else 0})
    window_table=pd.DataFrame(rows); display(window_table)
    if (window_table.query("split in ['train','validation']").windows==0).any(): raise ValueError('A step has no train/validation windows')
else:
    print('Saved window caches verified; no regeneration.')


if globals().get('SESSION_SIGNATURE')!='STEP1_GEVIS_HYBRID_REUSE_V6': raise RuntimeError('Stale STEP1 kernel. Restart and run from the first cell.')
def run_epoch(model,array,optimizer=None,scaler=None,seed=0):
    from pecvd import multiscale_patchtst
    training=optimizer is not None; model.train(training)
    sampler=multiscale_patchtst.training_sampler(model,len(array),seed,training)
    loader=DataLoader(torch.from_numpy(array),batch_size=CONFIG.batch_size,shuffle=training and sampler is None,sampler=sampler,num_workers=CONFIG.num_workers); total=0.; count=0; max_grad_norm=None; component_totals={}
    for j,x in enumerate(loader):
        x=x.to(DEVICE); target,patches=model.patchify(x); mask=reconstruction_mask(model,len(x),patches,seed+j,x.device)
        with torch.set_grad_enabled(training):
            if hasattr(model,'blind_prediction'):
                with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=CONFIG.use_amp and DEVICE.type=='cuda'):
                    if not training: mask=checkerboard_mask(model,len(x),patches,0,x.device)
                    objective=multiscale_patchtst.multiscale_objective if getattr(model,'reconstruction_contract',None)==multiscale_patchtst.CONTRACT else temporal_patchtst.blind_objective
                    loss,components=objective(model,x,mask,CONFIG)
                for key,value in components.items(): component_totals[key]=component_totals.get(key,0.)+float(value.detach())*len(x)
            elif hasattr(model,'pretraining_loss'):
                with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=CONFIG.use_amp and DEVICE.type=='cuda'):
                    loss,components=patchtst_adaptation.adaptation_objective(model,x,
                        patchtst_adaptation.DetailWeights(variability=CONFIG.variability_weight,slope=CONFIG.slope_weight,
                            peak=CONFIG.peak_weight,relationship=CONFIG.relationship_weight))
                for key,value in components.items(): component_totals[key]=component_totals.get(key,0.)+float(value.detach())*len(x)
            elif hasattr(model,'legacy'):
                with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=CONFIG.use_amp and DEVICE.type=='cuda'):
                    loss,components=patchtst_training.legacy_rich_objective(model,x,mask,CONFIG)
                for key,value in components.items(): component_totals[key]=component_totals.get(key,0.)+float(value.detach())*len(x)
            else:
                with torch.autocast(device_type=DEVICE.type,dtype=AMP_DTYPE,enabled=CONFIG.use_amp and DEVICE.type=='cuda'):
                    pred=model(x,mask); clean=model(x)
                pred=pred.float(); clean=clean.float(); target=target.float(); den=mask.sum().clamp_min(1)*target.shape[-1]
                masked=((pred-target).pow(2)*mask[...,None]).sum()/den; visible=(clean-target).pow(2).mean()
                slope=((clean[...,1:]-clean[...,:-1])-(target[...,1:]-target[...,:-1])).pow(2).mean()
                clean4=clean.reshape(len(x),model.n_sensors,patches,model.patch_len); target4=target.reshape_as(clean4)
                def relationship(z):
                    z=z-z.mean(-1,keepdim=True); z=z/z.square().sum(-1,keepdim=True).clamp_min(1e-4).sqrt(); return z.permute(0,2,1,3)@z.permute(0,2,3,1)
                relation=(relationship(clean4)-relationship(target4)).pow(2).mean()
                from pecvd.pretrained_patchtst import detail_loss
                variability,peak=detail_loss(clean,target)
                loss=masked+CONFIG.clean_weight*visible+CONFIG.slope_weight*slope+CONFIG.relationship_weight*relation+CONFIG.variability_weight*variability+CONFIG.peak_weight*peak
            if not torch.isfinite(loss): return float('nan'),max_grad_norm
            if training:
                optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(optimizer); grad_norm=nn.utils.clip_grad_norm_(model.parameters(),CONFIG.max_grad_norm)
                if not torch.isfinite(grad_norm): optimizer.zero_grad(set_to_none=True); return float('nan'),float('nan')
                max_grad_norm=max(float(grad_norm),max_grad_norm or 0.); scaler.step(optimizer); scaler.update()
        total+=loss.item()*len(x); count+=len(x)
    model.last_epoch_components={key:value/max(1,count) for key,value in component_totals.items()}
    return total/max(1,count),max_grad_norm

@torch.no_grad()
def evaluate_masked(model,array,seed):
    model.eval(); loader=DataLoader(torch.from_numpy(array),batch_size=CONFIG.batch_size,shuffle=False); mse_sum=0.; zero_sum=0.; count=0
    for j,x in enumerate(loader):
        x=x.to(DEVICE); target,_=model.patchify(x); prediction=(patchtst_adaptation.blind_prediction(model,x) if hasattr(model,'pretraining_loss') else rich_features(model,x)['reconstruction'].reshape_as(target)); mse=(prediction-target).pow(2).mean(); zero=target.pow(2).mean(); mse_sum+=mse.item()*len(x); zero_sum+=zero.item()*len(x); count+=len(x)
    return mse_sum/max(1,count),zero_sum/max(1,count)

def train_one_step(step):
    seed_everything(SEED); train=raw_windows[step]['train']; val=raw_windows[step]['validation']; test=raw_windows[step]['test']; mean=train.mean((0,1),keepdims=True); std=train.std((0,1),keepdims=True).clip(1e-6); norm=lambda x:((x-mean)/std).astype(np.float32); train_n,val_n,test_n=map(norm,(train,val,test)); length,patch,stride=STEPS[step]
    from pecvd.pretrained_patchtst import PretrainedPatchEncoder
    if selected_variant() in ('hybrid','portable'):
        import os
        public_path=Path(os.environ.get('GEVIS_PATCHTST_PATH',str(ROOT/'.cache/pretrained_patchtst')))
        encoder_class=hybrid_patchtst.PortableJointPatchEncoder if selected_variant()=='portable' else hybrid_patchtst.HybridPatchEncoder
        model=encoder_class.from_public(public_path,len(sensor_columns),patch,length,
            d_model=CONFIG.d_model,nhead=CONFIG.nhead,layers=CONFIG.layers,dropout=CONFIG.dropout)
        if CONFIG.reuse_legacy_checkpoint:
            legacy_path=LEGACY_ENCODER_ROOT/'encoders'/f'{step}.pt'
            if not legacy_path.is_file(): raise FileNotFoundError('Original encoder missing. Set reuse_legacy_checkpoint=False only to intentionally train a new local branch.')
            legacy=torch.load(legacy_path,map_location='cpu',weights_only=False)
            model.load_legacy(legacy,sensor_columns=sensor_columns,mean=mean,std=std,
                run_split={k:list(map(str,v)) for k,v in run_split.items()},source_sha256=sha256_file(DATA_PATH))
            model.pretrained_identity['legacy_checkpoint_sha256']=sha256_file(legacy_path)
        else:
            model.pretrained_identity['legacy_initialization']='new local joint branch; no legacy warm-start'
        model.configure_finetuning(CONFIG.pretrained_last_layers)
        pretrained_ids={id(p) for p in model.pretrained.parameters()}
        groups=[{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids],'lr':CONFIG.learning_rate},
                {'params':[p for p in model.pretrained.parameters() if p.requires_grad],'lr':CONFIG.pretrained_learning_rate}]
    elif selected_variant()=='official':
        import os
        from pecvd.official_patchtst import OfficialPatchEncoder
        if patch!=12 or CONFIG.d_model!=128: raise ValueError('Official PatchTST keeps public patch width 12 and hidden width 128')
        public_path=Path(os.environ.get('GEVIS_PATCHTST_PATH',str(ROOT/'.cache/pretrained_patchtst')))
        model=OfficialPatchEncoder.from_public(public_path,len(sensor_columns),length,mask_ratio=CONFIG.mask_ratio)
        model.configure_finetuning(CONFIG.pretrained_last_layers)
        pretrained_ids={id(p) for p in model.pretrained.parameters()}
        groups=[{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids],'lr':CONFIG.learning_rate},
                {'params':[p for p in model.pretrained.parameters() if p.requires_grad],'lr':CONFIG.pretrained_learning_rate}]
    elif selected_variant()=='pretrained':
        import os
        public_path=Path(os.environ.get('GEVIS_PATCHTST_PATH',str(ROOT/'.cache/pretrained_patchtst')))
        model=PretrainedPatchEncoder.from_public(public_path,len(sensor_columns),patch,CONFIG.d_model)
        model.configure_finetuning(CONFIG.pretrained_last_layers)
        pretrained_ids={id(p) for p in model.pretrained.parameters()}
        groups=[{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in pretrained_ids],'lr':CONFIG.learning_rate},
                {'params':[p for p in model.pretrained.parameters() if p.requires_grad],'lr':CONFIG.pretrained_learning_rate}]
    else:
        model=PatchEncoder(len(sensor_columns),patch,CONFIG.d_model,CONFIG.nhead,CONFIG.layers,CONFIG.dropout)
        groups=model.parameters()
    if selected_variant()=='multiscale':
        model.window_sample_weights=multiscale_patchtst.balanced_window_weights(window_metadata[step]['train'])
    model=model.to(DEVICE)
    optimizer=torch.optim.AdamW(groups,lr=CONFIG.learning_rate,weight_decay=CONFIG.weight_decay,eps=CONFIG.adam_eps,fused=CONFIG.fused_adamw and DEVICE.type=='cuda'); scaler=torch.amp.GradScaler('cuda',enabled=False); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=CONFIG.scheduler_patience,min_lr=CONFIG.min_learning_rate); history=[]; best=math.inf; best_state=None; stale=0; stop_reason='max_epochs'; nonfinite_epoch=None; started=time.perf_counter()
    selection=patchtst_training.ValidationSelection(CONFIG.min_epochs,CONFIG.patience,CONFIG.min_delta)
    base_rates=[group['lr'] for group in optimizer.param_groups]
    if selected_variant() in ('hybrid','portable','temporal','multiscale'):
        initial_mse,_=evaluate_masked(model,val_n,SEED+10_000)
        selection.observe(initial_mse,0); best=initial_mse; best_state=copy.deepcopy(model.state_dict())
        history.append({'epoch':0,'train_loss':float('nan'),'train_mse':float('nan'),'validation_loss':float('nan'),
            'validation_mse':initial_mse,'validation_blind_mse':initial_mse,'public_gain':float(model.public_gain.tanh().detach()) if hasattr(model,'public_gain') else None})
        print(f'{step:8s} initial validation blind MSE={initial_mse:.5f}; best-state selection includes this initial model')
    for epoch in range(1,CONFIG.epochs+1):
        if selected_variant() in ('hybrid','portable','temporal','multiscale'): patchtst_training.apply_warmup(optimizer,base_rates,epoch,CONFIG.warmup_epochs)
        tr,grad_norm=run_epoch(model,train_n,optimizer,scaler,SEED+epoch*100)
        train_components=dict(getattr(model,'last_epoch_components',{}))
        if not np.isfinite(tr) or grad_norm is None or not np.isfinite(grad_norm): stop_reason='nonfinite_train'; nonfinite_epoch=epoch; print(f'{step:8s} stopped: non-finite train state at epoch {epoch:03d}'); break
        if epoch!=1 and epoch%CONFIG.validation_interval and epoch!=CONFIG.epochs: continue
        validation_loss,_=run_epoch(model,val_n,None,None,SEED+10_000)
        validation_components=dict(getattr(model,'last_epoch_components',{}))
        va=validation_components.get('blind_mse',validation_loss)  # SAME blind MSE as final evaluation.
        if selected_variant() in ('hybrid','portable','temporal','multiscale'):
            va,_=evaluate_masked(model,val_n,SEED+10_000)
            validation_components['blind_mse']=va
        if not np.isfinite(va): stop_reason='nonfinite_validation'; nonfinite_epoch=epoch; print(f'{step:8s} stopped: non-finite validation state at epoch {epoch:03d}'); break
        if selected_variant() not in ('hybrid','portable','temporal','multiscale') or epoch>=CONFIG.warmup_epochs: scheduler.step(va)
        history.append({'epoch':epoch,'train_loss':tr,'train_mse':train_components.get('blind_mse',train_components.get('masked_mse',tr)),'validation_loss':validation_loss,'validation_mse':va,**{'train_'+k:v for k,v in train_components.items()},**{'validation_'+k:v for k,v in validation_components.items()},'gradient_norm':grad_norm,'learning_rate':float(optimizer.param_groups[0]['lr']),'head_learning_rate':float(optimizer.param_groups[0]['lr']),'encoder_learning_rate':float(optimizer.param_groups[1]['lr']) if hasattr(model,'pretraining_loss') or hasattr(model,'legacy') else None}); print(f'{step:8s} epoch {epoch:03d} train_loss={tr:.5f} val_blind_mse={va:.5f} grad={grad_norm:.3f} lr={optimizer.param_groups[0]["lr"]:.2e}')
        if selected_variant() in ('hybrid','portable','temporal','multiscale'):
            history[-1]['public_gain']=float(model.public_gain.tanh().detach()) if hasattr(model,'public_gain') else None
            improved,should_stop=selection.observe(va,epoch)
            if improved:best=va;best_state=copy.deepcopy(model.state_dict())
        else:
            if va<best-CONFIG.min_delta: best=va; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            should_stop=stale>=CONFIG.patience
        if should_stop: stop_reason='early_stopping'; break
    if best_state is None: raise RuntimeError(f'{step}: no finite validation checkpoint was produced')
    model.load_state_dict(best_state); model.eval(); test_mse,zero=evaluate_masked(model,test_n,SEED+20_000) if len(test_n) else (float('nan'),float('nan')); relative=test_mse/zero if zero>0 else float('nan')
    checkpoint={'schema':'gevis.encoder.rich_ssl.v1','artifact_version':ARTIFACT_VERSION,'step':step,'step_descriptor':step_description(PROCESS_CONFIG,step),'sensor_columns':sensor_columns,'sensor_descriptors':sensor_descriptors,'token_descriptor_contract':'configured step + sensor descriptor + relative patch position','feature_contract':['latent','blind_reconstruction','residual','patch_mean','patch_std','patch_slope','sensor_relationship'],'self_supervised_objectives':['mixed_random_checkerboard_masked_reconstruction','visible_reconstruction','patch_slope_preservation','sensor_relationship_preservation'],'step_config':{'window_length':length,'patch_length':patch,'stride':stride,'position_bins':10},'window_length':length,'patch_length':patch,'model_config':{'d_model':CONFIG.d_model,'nhead':CONFIG.nhead,'layers':CONFIG.layers,'dropout':CONFIG.dropout},'parameter_count':sum(p.numel() for p in model.parameters()),'model_state':{k:v.cpu() for k,v in best_state.items()},'mean':mean.reshape(-1),'std':std.reshape(-1),'run_split':{k:list(map(str,v)) for k,v in run_split.items()},'cohort_schema':{'source_prefix':SOURCE_PREFIX,'pm_prefix':PM_PREFIX,'key':'source_group|pm=pm_cycle'},'source_sha256':sha256_file(DATA_PATH),'seed':SEED,'mask_strategy':'mixed_random_checkerboard_train_checkerboard_blind_evaluation','config':asdict(CONFIG),'history':history,'training_status':{'stop_reason':stop_reason,'nonfinite_epoch':nonfinite_epoch,'best_epoch':min(history,key=lambda x:x['validation_mse'])['epoch']},'test_metrics':{'blind_mse':test_mse,'zero_baseline_mse':zero,'relative_mse':relative},'elapsed_seconds':time.perf_counter()-started}
    if hasattr(model,'pretrained_identity'):
        checkpoint.update(encoder_backend=('hf_patchtst_official_ssl_v1' if selected_variant()=='official' else 'hf_patchtst_rich_ssl_v1'),pretrained_config=model.hf_config,pretrained_identity=model.pretrained_identity)
        checkpoint['self_supervised_objectives']+=['patch_variability_preservation','local_peak_reconstruction']
        if selected_variant()=='official':
            checkpoint['self_supervised_objectives']=['checkerboard_blind_mse','blind_variability_preservation','blind_temporal_difference_preservation','blind_patch_peak_error','blind_sensor_relationship_preservation']
            checkpoint['mask_strategy']='two complementary checkerboard blind passes in train/validation/test; hidden samples excluded before scaling'
            checkpoint['architecture_contract']='official Transformers PatchTST forward/head with domain losses; no custom latent projection/statistics network/interpolation'
            checkpoint['adaptation_contract']={'version':patchtst_adaptation.CONTRACT,'selection':'validation blind_mse; never test',
                'encoder_scope':'all learned encoder parameters' if CONFIG.pretrained_last_layers is None else f'last {CONFIG.pretrained_last_layers} layers',
                'encoder_learning_rate':CONFIG.pretrained_learning_rate,'head_learning_rate':CONFIG.learning_rate,
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'weights':{'blind_mse':1.,'variability':CONFIG.variability_weight,'slope':CONFIG.slope_weight,'peak':CONFIG.peak_weight,'relationship':CONFIG.relationship_weight},
                'relationship_role':'loss only; official encoder remains channel-independent','fault_labels':False}

    if selected_variant() in ('hybrid','portable'):
        checkpoint.update(encoder_backend=(hybrid_patchtst.PORTABLE_BACKEND if selected_variant()=='portable' else hybrid_patchtst.BACKEND),pretrained_config=model.hf_config,pretrained_identity=model.pretrained_identity)
        checkpoint['self_supervised_objectives']=['mixed_random_checkerboard_masked_reconstruction','visible_reconstruction','patch_slope_preservation','sensor_relationship_preservation','patch_variability_preservation','local_peak_reconstruction']
        checkpoint['architecture_contract']=('shared trainable sensor token + native joint sensor/time attention for source and external inputs; public residual' if selected_variant()=='portable' else 'original sensor-ID joint architecture + public residual')
        checkpoint['native_sensor_transfer']=bool(getattr(model,'supports_native_sensor_transfer',False))
        checkpoint['adaptation_contract']={'version':'gevis.patchtst.legacy_joint.v1','selection':'validation blind_mse including epoch 0; never test',
            'encoder_scope':'all public learned parameters' if CONFIG.pretrained_last_layers is None else f'last {CONFIG.pretrained_last_layers} public layers',
            'encoder_learning_rate':CONFIG.pretrained_learning_rate,'head_learning_rate':CONFIG.learning_rate,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'weights':{'masked_mse':1.,'visible_mse':CONFIG.clean_weight,'variability':CONFIG.variability_weight,'slope':CONFIG.slope_weight,'peak':CONFIG.peak_weight,'relationship':CONFIG.relationship_weight},
            'public_gain':float(model.public_gain.tanh().detach()),'legacy_warmstart':CONFIG.reuse_legacy_checkpoint,
            'fault_labels':False,'masking':'original mixed random/checkerboard training; fixed two-pass blind validation/test'}
        checkpoint['training_status'].update(initial_validation_mse=history[0]['validation_mse'],best_validation_mse=selection.best,
            improved_over_initial=selection.best_epoch>0,max_epochs=CONFIG.epochs,min_epochs=CONFIG.min_epochs,
            patience_validation_checks=CONFIG.patience,validation_interval=CONFIG.validation_interval,warmup_epochs=CONFIG.warmup_epochs,
            best_epoch=selection.best_epoch)

    if selected_variant() in ('temporal','multiscale'):
        checkpoint.update(encoder_backend=temporal_patchtst.BACKEND,reconstruction_contract=temporal_patchtst.CONTRACT,native_sensor_transfer=True,architecture_contract='native temporal + gated cross-sensor axial attention; trained from scratch', numeric_bridge='paired_temporal_blind_408_v2')
        checkpoint['self_supervised_objectives']=['complementary_hidden_mse','blind_full_window_slope','blind_sensor_relationship','blind_variability','blind_peak']
        checkpoint['training_status'].update(initial_validation_mse=history[0]['validation_mse'],best_validation_mse=selection.best,improved_over_initial=selection.best_epoch>0,max_epochs=CONFIG.epochs,min_epochs=CONFIG.min_epochs,patience_validation_checks=CONFIG.patience,validation_interval=CONFIG.validation_interval,warmup_epochs=CONFIG.warmup_epochs,best_epoch=selection.best_epoch)
        checkpoint['mask_strategy']='complementary hidden predictions; fixed checkerboard Validation/Test/runtime'
        checkpoint['adaptation_contract']={'encoder_scope':'all native temporal parameters; no public warm start','trainable_parameters':sum(p.numel() for p in model.parameters()),'legacy_warmstart':False,'selection':'validation blind_mse only'}
    if selected_variant()=='multiscale':
        checkpoint.update(encoder_backend=multiscale_patchtst.BACKEND,reconstruction_contract=multiscale_patchtst.CONTRACT,numeric_bridge='paired_relational_multiscale_440_v3',architecture_contract='fine native patches; local and coarse global temporal/sensor Attention with gated fusion')
        checkpoint['model_config'].update(local_patches=model.local_patches,coarse_factor=model.coarse_factor)
        checkpoint['self_supervised_objectives']+=['short_half_quarter_blind_crops','lagged_sensor_relationship_preservation']
        checkpoint['training_sampler']='equal TRAIN run mass, then equal occupied position-bin mass; same epoch window count'
        checkpoint['multiscale_training']={'crop_divisors':[2,4],'crop_weight':.5,'lag_weight':.25*CONFIG.relationship_weight,'full_context_retained':True,'selection':'fixed full-context Validation blind MSE','test_selection':False}
    save_torch(checkpoint,ARTIFACT_ROOT/'encoders'/f'{step}.pt',CONFIG.overwrite_artifacts)
    print('Saved encoder:',ARTIFACT_ROOT/'encoders'/f'{step}.pt')
    if SAVE_WINDOWS:
        path=ARTIFACT_ROOT/'windows'/f'{step}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and not CONFIG.overwrite_artifacts: raise FileExistsError(path)
        payload={}
        for split_name,array in [('train',train_n),('validation',val_n),('test',test_n)]:
            payload[split_name]=array; meta=window_metadata[step][split_name]
            for column in ('process_id','source_group','pm_cycle','cohort_key','window_start','window_end','position_ratio','position_bin'):
                values=meta[column].to_numpy(); payload[f'{split_name}_{column}']=values.astype(str) if values.dtype==object else values
        np.savez_compressed(path,**payload)
    return checkpoint

def preflight_training():
    if selected_variant()!=CONFIGURED_MODEL_VARIANT:
        raise RuntimeError('Model variant changed after configuration. Restart kernel and run from the first cell.')
    expected=ROOT/'artifacts'/active_config(load_process_config(PROCESS_CONFIG_PATH)).artifacts.encoder
    if ARTIFACT_ROOT.resolve()!=expected.resolve():
        raise RuntimeError('Artifact root does not match the selected model. Rerun configuration; do not enable overwrite.')
    if CONFIGURED_MODEL_VARIANT in ('temporal','multiscale') and CONFIG.d_model!=96:
        raise ValueError('Temporal v2 bridge requires encoder width 96; do not change only one stage.')
    if CONFIGURED_MODEL_VARIANT=='official' and (CONFIG.d_model!=128 or any(spec[1]!=12 for spec in STEPS.values())):
        raise RuntimeError('Official PatchTST requires hidden width 128 and native patch width 12. Rerun configuration.')
    if CONFIGURED_MODEL_VARIANT=='official' and CONFIG.mask_ratio!=.5:
        raise ValueError('Domain-v2 uses complementary half-grid masks; mask_ratio must be .5')
    if CONFIGURED_MODEL_VARIANT in ('hybrid','portable'):
        if not 0<=CONFIG.min_epochs<=CONFIG.epochs or CONFIG.warmup_epochs>CONFIG.epochs:
            raise ValueError('Invalid min/max/warmup epoch budget')
        if CONFIG.reuse_legacy_checkpoint:
            missing=[step for step in STEPS if not (LEGACY_ENCODER_ROOT/'encoders'/f'{step}.pt').is_file()]
            if missing:raise FileNotFoundError('Original legacy encoders missing; check LEGACY_ENCODER_ROOT or intentionally disable reuse_legacy_checkpoint before training.')
    targets=[ARTIFACT_ROOT/'encoders'/f'{step}.pt' for step in STEPS]
    if SAVE_WINDOWS: targets += [ARTIFACT_ROOT/'windows'/f'{step}.npz' for step in STEPS]
    targets += [ARTIFACT_ROOT/'encoder_manifest.json']
    collisions=[p for p in targets if p.exists()]
    if collisions and not CONFIG.overwrite_artifacts:
        raise FileExistsError(f'{len(collisions)} output file(s) already exist; training has NOT started. First: {collisions[0]}. Preserve existing artifacts or explicitly authorize overwrite.')

encoder_checkpoints={}
if RUN_TRAINING:
    preflight_training()
    for step in STEPS: encoder_checkpoints[step]=train_one_step(step)
elif REUSE_SAVED_ENCODERS:
    encoder_checkpoints=SAVED_ENCODERS
    print('Loaded completed STEP 1 checkpoints; no optimizer steps or writes.')
else: print('RUN_TRAINING=False: schema/window audit only')

def selected_checkpoint_row(checkpoint):
    history=checkpoint['history']
    recorded=checkpoint.get('training_status',{}).get('best_epoch')
    return next((row for row in history if row['epoch']==recorded),
                min(history,key=lambda row:row['validation_mse']))

if encoder_checkpoints:
    summary=pd.DataFrame([{'step':s,'parameters':c['parameter_count'],'best_epoch':selected_checkpoint_row(c)['epoch'],**c['test_metrics'],'elapsed_seconds':c['elapsed_seconds']} for s,c in encoder_checkpoints.items()])
    summary['beats_zero_baseline']=summary.relative_mse<1
    summary['baseline_improvement_pct']=100*(1-summary.relative_mse)
    summary['reconstruction_gate']=np.where(summary.relative_mse<.8,'PASS',np.where(summary.relative_mse<1,'WATCH','FAIL'))
    display(summary.sort_values('relative_mse'))
    display(summary[['step','blind_mse','zero_baseline_mse','relative_mse','baseline_improvement_pct','reconstruction_gate']].sort_values('relative_mse'))
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ax,(step,ck) in zip(axes.flat,encoder_checkpoints.items()):
        h=pd.DataFrame(ck['history']).sort_values('epoch')
        ax.plot(h.epoch,h.train_mse,color='tab:blue',alpha=.75,label='Train: mixed masks, training mode')
        ax.plot(h.epoch,h.validation_mse,color='tab:orange',label='Validation: fixed blind masks, eval mode')
        best=h.loc[h.epoch.eq(selected_checkpoint_row(ck)['epoch'])].iloc[0]
        ax.scatter([best.epoch],[best.validation_mse],color='red',zorder=3,label=f'Selected epoch: {int(best.epoch)}')
        ax.set_title(step);ax.set_xlabel('epoch');ax.set_ylabel('Reconstruction MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    plt.show()
    adaptation_rows=[]
    for step,ck in encoder_checkpoints.items():
        if 'adaptation_contract' not in ck: continue
        best_row=selected_checkpoint_row(ck)
        adaptation_rows.append({'step':step,'best_epoch':best_row['epoch'],
            **{k:v for k,v in best_row.items() if k.startswith('validation_')},
            'trainable_parameters':ck['adaptation_contract']['trainable_parameters'],
            'encoder_scope':ck['adaptation_contract']['encoder_scope']})
    if adaptation_rows:
        print('Validation loss components at the selected epoch (not fault accuracy)')
        display(pd.DataFrame(adaptation_rows))
    convergence=[]
    for step,ck in encoder_checkpoints.items():
        history=ck.get('history',[])
        if not history: continue
        selected=selected_checkpoint_row(ck)
        initial=next((row['validation_mse'] for row in history if row['epoch']==0),None)
        convergence.append({'step':step,'encoder_backend':ck.get('encoder_backend','unknown'),
            **ck.get('training_status',{}),'best_epoch':selected['epoch'],
            'epochs_executed':max(row['epoch'] for row in history),
            'initial_validation_mse':initial,'best_validation_mse':selected['validation_mse'],
            'initial_baseline_available':initial is not None,
            'improved_over_initial':selected['validation_mse']<initial if initial is not None else None})
    if convergence:
        print('Validation-selected convergence for every backend; no Test selection')
        display(pd.DataFrame(convergence))
        if any(row['best_epoch']==0 for row in convergence):
            print('Warning: initial epoch selected; no measured improvement over initial weights.')


if encoder_checkpoints and not REUSE_SAVED_ENCODERS:
    manifest={'schema':'gevis.encoder_manifest.rich_ssl.v1','artifact_version':ARTIFACT_VERSION,'created_by':'STEP_1_train_patchtst_encoder.ipynb','source_sha256':sha256_file(DATA_PATH),'seed':SEED,'sensor_columns':sensor_columns,'sensor_descriptors':sensor_descriptors,'token_descriptor_contract':'configured step + sensor descriptor + relative patch position','feature_contract':next(iter(encoder_checkpoints.values()))['feature_contract'],'self_supervised_objectives':next(iter(encoder_checkpoints.values()))['self_supervised_objectives'],'cohort_schema':{'source_prefix':SOURCE_PREFIX,'pm_prefix':PM_PREFIX,'key':'source_group|pm=pm_cycle','position_bins':10},'run_split':{k:list(map(str,v)) for k,v in run_split.items()},'cohort_split_counts':cohort_split.to_dict(),'step_checkpoints':{s:{'path':f'encoders/{s}.pt','sha256':sha256_file(ARTIFACT_ROOT/'encoders'/f'{s}.pt'),'step_descriptor':ck['step_descriptor'],'step_config':ck['step_config'],'test_metrics':ck['test_metrics']} for s,ck in encoder_checkpoints.items()}}
    save_json(manifest,ARTIFACT_ROOT/'encoder_manifest.json',CONFIG.overwrite_artifacts); print('Saved:',ARTIFACT_ROOT/'encoder_manifest.json')
if REUSE_SAVED_ENCODERS: print('Existing manifest preserved byte-for-byte for STEP 2 bridge identity.')


# Diagnostic only: reuse saved STEP 1 weights; no training or CSV read here.
from pecvd.experiments.encoder_audit import audit_encoder
manifest=json.loads((ARTIFACT_ROOT/'encoder_manifest.json').read_text(encoding='utf-8'))
ENCODER_SENSITIVITY=[]
for phase,item in manifest['step_checkpoints'].items():
    ck=torch.load(ARTIFACT_ROOT/item['path'],map_location='cpu',weights_only=False)
    from pecvd.encoder_loading import load_encoder  # Refresh local factories before strict offline reconstruction.
    frozen=load_encoder(ck,PatchEncoder).to(DEVICE)
    frozen.requires_grad_(False).eval()
    with np.load(ARTIFACT_ROOT/'windows'/f'{phase}.npz',allow_pickle=False) as cache:
        sample=torch.tensor(cache['validation'][:4],device=DEVICE)
    ENCODER_SENSITIVITY.append({'step':phase,**audit_encoder(frozen,sample,seed=SEED)})
    frozen.cpu()
display(pd.DataFrame(ENCODER_SENSITIVITY))
