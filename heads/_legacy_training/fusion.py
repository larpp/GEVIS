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


from pathlib import Path
from dataclasses import dataclass,asdict
import copy,hashlib,importlib,json,math,random,sys,time
import numpy as np,pandas as pd,torch
import matplotlib.pyplot as plt
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import pecvd.gevis_backbone as gevis_backbone
gevis_backbone=importlib.reload(gevis_backbone)  # Refresh the current shared source from disk.
EXPECTED_SOURCE_SCHEMA='gevis.backbone_source.v9'
if gevis_backbone.BACKBONE_SOURCE_SCHEMA!=EXPECTED_SOURCE_SCHEMA: raise RuntimeError('STEP2 and gevis_backbone.py source schemas differ')
QUESTION_BANK=gevis_backbone.QUESTION_BANK; SEMANTIC_QUESTION_BANK=gevis_backbone.SEMANTIC_QUESTION_BANK
BackbonePretrainHeads=gevis_backbone.BackbonePretrainHeads; GEVISBackbone=gevis_backbone.GEVISBackbone
evaluate_cross_attention=gevis_backbone.evaluate_cross_attention; evaluate_text=gevis_backbone.evaluate_text; evaluate_semantics=gevis_backbone.evaluate_semantics
train_cross_attention=gevis_backbone.train_cross_attention; train_text_encoder=gevis_backbone.train_text_encoder
SOURCE_MODULE_PATH=Path(gevis_backbone.__file__).resolve()
SOURCE_MODULE_SHA256=hashlib.sha256(SOURCE_MODULE_PATH.read_bytes()).hexdigest()
from pecvd.encoder_loading import backbone_source_matches
EXPECTED_ARTIFACT_VERSION='gevis_backbone_e5_v1'
ARTIFACT_VERSION=EXPECTED_ARTIFACT_VERSION
CURRICULUM_TAG=gevis_backbone.SEMANTIC_CURRICULUM_VERSION
import importlib
import pecvd.model_variants as model_variants
importlib.reload(model_variants)
from pecvd.model_variants import active_run_tag
RUN_TAG=active_run_tag()
TRAINING_PROFILE=PIPELINE_SETTINGS['training_profile']
from pecvd.experiments import sensor_discrimination
importlib.reload(sensor_discrimination)

from pecvd.process_config import load_process_config
from pecvd.experiments.process_inputs import configured_process_path,candidate_backbone_path
PROCESS_CONFIG_PATH=configured_process_path(ROOT)
from pecvd.model_variants import active_config,active_run_tag,selected_variant
PROCESS_CONFIG=active_config(load_process_config(PROCESS_CONFIG_PATH))
TEXT_REUSE_PATH=ROOT/'artifacts'/(ARTIFACT_VERSION+'_'+CURRICULUM_TAG+'_real_step1_binding_v2')/'gevis_backbone.pt'
TEXT_REUSE_CANDIDATES=[candidate_backbone_path(ROOT,load_process_config(PROCESS_CONFIG_PATH),gevis_backbone.TARGET_BINDING_VERSION),TEXT_REUSE_PATH]
TEXT_REUSE_CANDIDATES.insert(0,candidate_backbone_path(ROOT,PROCESS_CONFIG,RUN_TAG.removesuffix('_contrastive_v12').removesuffix('_fixed_questions_v13').removesuffix('_center_relation_v14')))
TEXT_REUSE_PATH=next((path for path in TEXT_REUSE_CANDIDATES if path.is_file()),TEXT_REUSE_PATH)  # Prefer completed v4 text; never overwrite either source.
# Previous revision is a text-only initialization source; it is never accepted as the completed current backbone.
TEXT_REUSE_CANDIDATES.insert(0,candidate_backbone_path(ROOT,PROCESS_CONFIG,RUN_TAG,revision=''))
TEXT_REUSE_PATH=next((path for path in TEXT_REUSE_CANDIDATES if path.is_file()),TEXT_REUSE_PATH)
ARTIFACT_ROOT=candidate_backbone_path(ROOT,PROCESS_CONFIG,RUN_TAG).parent
SESSION_SIGNATURE='STEP2_GEVIS_GENERIC_DISCOVERY_V10'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
@dataclass(frozen=True)
class Config:
    d_model:int=192
    numeric_dim:int=440 if selected_variant()=='multiscale' else 408 if selected_variant()=='temporal' else (520 if selected_variant()=='official' else 392)  # 4 official 128-D latent blocks + 8 FDC features
    dropout:float=.1
    text_encoder_name:str=gevis_backbone.DEFAULT_TEXT_ENCODER
    text_encoder_revision:str=gevis_backbone.DEFAULT_TEXT_ENCODER_REVISION
    max_text_epochs:int=60
    text_batch_size:int=32
    semantic_steps_per_epoch:int=72
    compare_existing_backbone:bool=False
    text_patience:int=12
    text_learning_rate:float=2e-4
    cross_epochs:int=40
    reuse_text_checkpoint:bool=True
    reuse_fusion_weights:bool=False
    discrimination_margin:float=.5
    discrimination_hard_weight:float=.25
    discrimination_pair_weight:float=.25
    numeric_evidence_loss_weight:float=1.0  # Primary sensor-free discovery objective.
    fusion_mode:str=gevis_backbone.REFERENCE_FUSION_MODE
    cross_batch_size:int=4
    max_cross_train_windows_per_step:int=128
    max_cross_validation_windows_per_step:int=64
    cross_patience:int=8
    cross_learning_rate:float=5e-5
    cross_min_learning_rate:float=1e-6
    cross_scheduler_patience:int=2
    cross_joint_loss_weight:float=.25
    cross_paraphrase_weight:float=.10
    cross_feature_loss_weight:float=.15
    cross_sensor_loss_weight:float=1.0
    cross_scope_loss_weight:float=.5
    cross_sensor_margin_weight:float=.25
    cross_scope_contrast_weight:float=.15
    cross_log_margin:float=.5
    grounding_steps:int=0  # Deferred: this path derives binary targets from synthetic numeric deltas.
    overwrite_artifacts:bool=False
CONFIG=Config()
REUSE_SAVED_BACKBONE=(ARTIFACT_ROOT/'gevis_backbone.pt').is_file() and not CONFIG.overwrite_artifacts
if REUSE_SAVED_BACKBONE:
    from pecvd.encoder_reuse import sha256
    SAVED_BACKBONE=torch.load(ARTIFACT_ROOT/'gevis_backbone.pt',map_location='cpu',weights_only=False)
    _manifest=ROOT/'artifacts'/PROCESS_CONFIG.artifacts.encoder/'encoder_manifest.json'
    if (SAVED_BACKBONE.get('schema')!='gevis.backbone.generic.v10' or
        SAVED_BACKBONE.get('binding_version')!=RUN_TAG or
        SAVED_BACKBONE.get('numeric_dim')!=CONFIG.numeric_dim or
        SAVED_BACKBONE.get('step1_bridge',{}).get('manifest_sha256')!=sha256(_manifest)):
        raise ValueError('Saved STEP 2 does not match this STEP 1/profile; no files changed')
    if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14') and SAVED_BACKBONE.get('sensor_discrimination_source_sha256')!=sensor_discrimination.source_identity():
        raise ValueError('Candidate training source changed; use a new run namespace, never reuse stale weights')
    if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14'):
        _loss_keys=('discrimination_margin','discrimination_hard_weight','discrimination_pair_weight')
        if (SAVED_BACKBONE.get('training_profile')!=TRAINING_PROFILE or
            not backbone_source_matches(SAVED_BACKBONE,SOURCE_MODULE_PATH) or
            any(SAVED_BACKBONE.get('config',{}).get(k)!=getattr(CONFIG,k) for k in _loss_keys)):
            raise ValueError('Candidate source/loss configuration changed; use a new run namespace')
    _payload=json.loads(_manifest.read_text(encoding='utf-8'))
    for _step,_item in _payload['step_checkpoints'].items():
        if sha256(_manifest.parent/_item['path'])!=_item['sha256']:
            raise ValueError('STEP 1 encoder hash mismatch; saved STEP 2 not reused')
    print('Reusing completed STEP 2 with exact STEP 1 bridge; no training or overwrite.')
print({'device':str(DEVICE),'artifact':ARTIFACT_VERSION,'source_schema':EXPECTED_SOURCE_SCHEMA,'linked_module':str(SOURCE_MODULE_PATH),'module_sha256':SOURCE_MODULE_SHA256[:12],'questions':sum(map(len,QUESTION_BANK.values())),'max_text_epochs':CONFIG.max_text_epochs,'cross_epochs':CONFIG.cross_epochs,'cross_source':'frozen real STEP 1 train/validation states'})
train_grounding=gevis_backbone.train_grounding; evaluate_grounding=gevis_backbone.evaluate_grounding


question_rows=[{'split':split,'question':item.text,'answer_type':item.answer_type or 'UNSUPPORTED','supported':item.supported} for split,items in QUESTION_BANK.items() for item in items]
QUESTION_TABLE=pd.DataFrame(question_rows)
QUESTION_SUMMARY=QUESTION_TABLE.groupby(['split','answer_type']).size().rename('questions').reset_index()
display(QUESTION_SUMMARY)
display(QUESTION_TABLE)  # Inspect all training/validation/evaluation questions.

if not REUSE_SAVED_BACKBONE:
    if globals().get('SESSION_SIGNATURE')!='STEP2_GEVIS_GENERIC_DISCOVERY_V10': raise RuntimeError('Stale STEP2 kernel. Restart and run from the first cell.')
    if ARTIFACT_VERSION!=EXPECTED_ARTIFACT_VERSION: raise RuntimeError('Stale artifact variables remain. Restart the kernel.')
    backbone=GEVISBackbone(numeric_dim=CONFIG.numeric_dim,d_model=CONFIG.d_model,dropout=CONFIG.dropout,text_encoder_name=CONFIG.text_encoder_name,text_encoder_revision=CONFIG.text_encoder_revision).to(DEVICE)
    pretrain_heads=BackbonePretrainHeads(CONFIG.d_model).to(DEVICE)
    TEXT_REUSE_SHA256=None
    if getattr(CONFIG,'reuse_text_checkpoint',False):
        if not TEXT_REUSE_PATH.is_file():
            raise FileNotFoundError('Existing E5 backbone missing. Set TEXT_REUSE_PATH to your completed STEP 2 checkpoint.')
        print({'selected_text_checkpoint':str(TEXT_REUSE_PATH),'text_reuse':'frozen text source only'})
        prior=torch.load(TEXT_REUSE_PATH,map_location='cpu',weights_only=False)
        # A legacy v9 checkpoint may initialize only frozen text/semantic weights.
        # Generic v10 numeric/fusion weights are trained and saved in a new namespace.
        if prior.get('schema') not in ('gevis.backbone.v9','gevis.backbone.generic.v10'):
            raise ValueError('Existing STEP 2 checkpoint is not a supported frozen-text source')
        expected={'d_model':CONFIG.d_model,
                  'text_encoder_name':CONFIG.text_encoder_name,'text_encoder_revision':CONFIG.text_encoder_revision,
                  'semantic_curriculum':CURRICULUM_TAG}
        if any(prior.get(key)!=value for key,value in expected.items()):
            raise ValueError('Existing STEP 2 checkpoint differs from the frozen text/architecture contract')
        _manifest=ROOT/'artifacts'/PROCESS_CONFIG.artifacts.encoder/'encoder_manifest.json'
        FUSION_REUSED=bool(prior.get('schema')=='gevis.backbone.generic.v10' and getattr(CONFIG,'reuse_fusion_weights',False) and _manifest.is_file() and prior.get('step1_bridge',{}).get('manifest_sha256')==hashlib.sha256(_manifest.read_bytes()).hexdigest())
        if FUSION_REUSED:
            if prior.get('numeric_dim')!=CONFIG.numeric_dim: raise ValueError('Warm-start numeric width mismatch')
            backbone.load_state_dict(prior['backbone_state'])
        else:
            backbone.text.load_state_dict({key.removeprefix('text.'):value for key,value in prior['backbone_state'].items() if key.startswith('text.')})
        pretrain_heads.load_state_dict(prior['pretrain_heads_state'])
        TEXT_HISTORY=[]; BEST_TEXT_EPOCH=prior.get('best_text_epoch')
        TEXT_REUSE_SHA256=hashlib.sha256(TEXT_REUSE_PATH.read_bytes()).hexdigest()
        REUSED_STEP1_BRIDGE=prior.get('step1_bridge',{})
        backbone.fusion_mode=prior.get('fusion_mode','legacy') if FUSION_REUSED else 'legacy'
        del prior
        print({'text_training':'skipped; existing E5/projection/semantic heads reused','fusion_start':'existing STEP 2' if FUSION_REUSED else 'new numeric/fusion weights; text retained'})
    else:
        TEXT_HISTORY,BEST_TEXT_EPOCH=train_text_encoder(backbone,pretrain_heads,DEVICE,epochs=CONFIG.max_text_epochs,batch_size=CONFIG.text_batch_size,learning_rate=CONFIG.text_learning_rate,patience=CONFIG.text_patience,seed=SEED,steps_per_epoch=CONFIG.semantic_steps_per_epoch)
    TEXT_METRICS=pd.DataFrame([evaluate_text(backbone,pretrain_heads,list(QUESTION_BANK[split]),split,DEVICE) for split in ('train','validation','test')])
    SEMANTIC_METRICS=pd.DataFrame([evaluate_semantics(backbone,pretrain_heads,list(SEMANTIC_QUESTION_BANK[split]),split,DEVICE) for split in ('train','validation','test','audit')])
    SEMANTIC_BY_TYPE=[]; SEMANTIC_DIAGNOSTICS={'confusion':[],'contrast_pairs':[]}
    for split in ('validation','test','audit'):
        rows=list(SEMANTIC_QUESTION_BANK[split])
        for kind in gevis_backbone.ANSWER_TYPES:
            result=evaluate_semantics(backbone,pretrain_heads,[item for item in rows if item.answer_type==kind],split,DEVICE)
            SEMANTIC_BY_TYPE.append({'answer_type':kind,**result})
        details=gevis_backbone.semantic_diagnostics(backbone,pretrain_heads,rows,split,DEVICE)
        for key in SEMANTIC_DIAGNOSTICS: SEMANTIC_DIAGNOSTICS[key]+=details[key]
    print({'backbone_parameters':sum(p.numel() for p in backbone.parameters()),'pretrain_head_parameters':sum(p.numel() for p in pretrain_heads.parameters()),'best_text_epoch':BEST_TEXT_EPOCH})
    display(TEXT_METRICS); display(SEMANTIC_METRICS)
    display(pd.DataFrame(SEMANTIC_BY_TYPE)); display(pd.DataFrame(SEMANTIC_DIAGNOSTICS['contrast_pairs']))
    display(pd.DataFrame(SEMANTIC_DIAGNOSTICS['confusion']))
    # Evaluate old and new weights on exactly the same expanded held-out bank.
    # Existing baseline weights are read only; no baseline retraining or overwrite.
    SEMANTIC_FIXED_COMPARISON=[{'model':'fdc_semantic_scale_v2',**row} for row in SEMANTIC_BY_TYPE]
    baseline_path=ROOT/'artifacts'/'gevis_backbone_v4'/'gevis_backbone.pt'
    BASELINE_COMPARISON_STATUS='disabled'
    if CONFIG.compare_existing_backbone:
        if not baseline_path.exists():
            BASELINE_COMPARISON_STATUS='baseline_missing'
        else:
            old_ck=torch.load(baseline_path,map_location='cpu',weights_only=False)
            if old_ck.get('schema')!='gevis.backbone.v4' or old_ck.get('d_model')!=CONFIG.d_model or old_ck.get('numeric_dim')!=CONFIG.numeric_dim:
                raise ValueError('Existing baseline architecture mismatch; disable baseline comparison explicitly')
            old_model=GEVISBackbone(CONFIG.numeric_dim,CONFIG.d_model,0.,text_encoder_name='byte').to(DEVICE)
            old_heads=BackbonePretrainHeads(CONFIG.d_model).to(DEVICE)
            old_model.load_state_dict(old_ck['backbone_state']); old_heads.load_state_dict(old_ck['pretrain_heads_state'])
            for split in ('validation','test','audit'):
                for kind in gevis_backbone.ANSWER_TYPES:
                    rows=[item for item in SEMANTIC_QUESTION_BANK[split] if item.answer_type==kind]
                    SEMANTIC_FIXED_COMPARISON.append({'model':'existing_baseline','answer_type':kind,**evaluate_semantics(old_model,old_heads,rows,split,DEVICE)})
            BASELINE_COMPARISON_STATUS='same_questions_read_only_baseline'
            del old_model,old_heads,old_ck
    print({'baseline_comparison':BASELINE_COMPARISON_STATUS,'selection':'validation only; test/audit comparisons never select weights'})
    display(pd.DataFrame(SEMANTIC_FIXED_COMPARISON))

else:
    print({'text_training':'skipped; compatible saved backbone reused','text_source_sha256':SAVED_BACKBONE.get('text_reuse_sha256'),'backbone_reused':True})
    display(pd.DataFrame(SAVED_BACKBONE['text_metrics']))
    display(pd.DataFrame(SAVED_BACKBONE['semantic_metrics']))


if not REUSE_SAVED_BACKBONE:
    if TEXT_HISTORY:
        text_history=pd.DataFrame(TEXT_HISTORY)
        ax=text_history.plot(x='epoch',y=['loss','validation_score'],secondary_y='validation_score',figsize=(8,4),title='Text Encoder training and held-out score')
        ax.set_ylabel('training loss'); ax.right_ax.set_ylabel('validation score'); plt.show()
    else:
        print('Text training skipped: existing E5/projection/semantic weights reused unchanged.')

else:
    print('Saved STEP 2 histories are historical; no new training performed.')


if not REUSE_SAVED_BACKBONE:
    if globals().get('SESSION_SIGNATURE')!='STEP2_GEVIS_GENERIC_DISCOVERY_V10': raise RuntimeError('Stale STEP2 kernel. Restart and run from the first cell.')
    from collections import defaultdict
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    from pecvd.process_config import load_process_config
    from pecvd.model_variants import active_config,active_run_tag,selected_variant
    PROCESS_CONFIG=active_config(load_process_config(PROCESS_CONFIG_PATH)); ENCODER_ROOT=ROOT/'artifacts'/PROCESS_CONFIG.artifacts.encoder
    class FrozenPatchEncoder(nn.Module):
        def __init__(self,n_sensors,patch_len,d_model=64,nhead=4,layers=2,dropout=.1):
            super().__init__(); self.n_sensors=n_sensors; self.patch_len=patch_len; self.d_model=d_model; self.input=nn.Linear(patch_len,d_model); self.sensor_emb=nn.Embedding(n_sensors,d_model); self.patch_emb=nn.Embedding(64,d_model); self.mask_token=nn.Parameter(torch.zeros(d_model)); block=nn.TransformerEncoderLayer(d_model,nhead,4*d_model,dropout,batch_first=True,norm_first=True,activation='gelu'); self.encoder=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(d_model); self.decoder=nn.Linear(d_model,patch_len)
        def patchify(self,x):
            b,t,s=x.shape; p=t//self.patch_len; return x[:,:p*self.patch_len].reshape(b,p,self.patch_len,s).permute(0,3,1,2).reshape(b,s*p,self.patch_len),p
        def tokens(self,x,mask=None):
            z,p=self.patchify(x); sid=torch.arange(self.n_sensors,device=x.device).repeat_interleave(p); pid=torch.arange(p,device=x.device).repeat(self.n_sensors); content=self.input(z); content=torch.where(mask[...,None],self.mask_token[None,None],content) if mask is not None else content; return self.norm(self.encoder(content+self.sensor_emb(sid)[None]+self.patch_emb(pid)[None]))
        def forward(self,x,mask=None): return self.decoder(self.tokens(x,mask))
    def file_sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest_path=ENCODER_ROOT/'encoder_manifest.json'
    if not manifest_path.exists(): raise FileNotFoundError('Run STEP_1_train_patchtst_encoder.ipynb first')
    encoder_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if encoder_manifest.get('schema')!='gevis.encoder_manifest.rich_ssl.v1': raise ValueError('Expected STEP 1 rich-SSL encoder manifest v1')
    if globals().get('FUSION_REUSED',False) and REUSED_STEP1_BRIDGE.get('manifest_sha256')!=file_sha256(manifest_path):
        raise ValueError('Warm-start STEP 2 used a different STEP 1 manifest')
    step1_encoders={}; step1_windows={}; step1_provenance={}
    for step,item in encoder_manifest['step_checkpoints'].items():
        path=ENCODER_ROOT/item['path']
        if file_sha256(path)!=item['sha256']: raise ValueError(f'STEP 1 encoder hash mismatch: {step}')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('schema')!='gevis.encoder.rich_ssl.v1': raise ValueError(f'Unsupported STEP 1 encoder: {step}')
        from pecvd.encoder_loading import load_encoder  # Refresh local factories before strict offline reconstruction.
        cfg=ck['model_config']; model=load_encoder(ck,FrozenPatchEncoder); model.requires_grad_(False).eval(); step1_encoders[step]=(model,ck)
        with np.load(ENCODER_ROOT/'windows'/f'{step}.npz') as z:
            step1_windows[step]={split:z[split] for split in ('train','validation')}
            step1_provenance[step]={split:{column:z[f'{split}_{column}'] for column in ('process_id','cohort_key','position_bin')} for split in ('train','validation')}
    encoder_dims={int(ck['model_config']['d_model']) for _,ck in step1_encoders.values()}
    if len(encoder_dims)!=1 or 4*encoder_dims.pop()+(56 if selected_variant()=='multiscale' else 24 if selected_variant()=='temporal' else 8)!=CONFIG.numeric_dim: raise ValueError('STEP 1 latent width does not match STEP 2 numeric_dim')
    class RealStep1Pairs(Dataset):
        def __init__(self,array,meta,seed,limit):
            self.x=array; self.meta=meta; self.seed=seed; groups=defaultdict(list)
            for i,position in enumerate(meta['position_bin']): groups[(int(position),str(meta['cohort_key'][i]))].append(i)
            valid=[]; self.candidates={}
            for i,(position,process) in enumerate(zip(meta['position_bin'],meta['process_id'])):
                pool=[j for j in groups[(int(position),str(meta['cohort_key'][i]))] if str(meta['process_id'][j])!=str(process)]
                if pool: valid.append(i); self.candidates[i]=pool
            if limit and len(valid)>limit: valid=np.random.default_rng(seed).choice(valid,size=limit,replace=False).tolist()
            if not valid: raise ValueError('No real cross-run STEP 1 pairs')
            self.indices=valid
        def __len__(self): return len(self.indices)
        def __getitem__(self,item):
            index=self.indices[item]; rng=np.random.default_rng(self.seed+item); pool=np.asarray(self.candidates[index]); selected=rng.choice(pool,size=min(16,len(pool)),replace=False); reference=np.median(self.x[selected],axis=0).astype(np.float32); return torch.from_numpy(self.x[index]),torch.from_numpy(reference)
    def make_real_loaders(split):
        limit=CONFIG.max_cross_train_windows_per_step if split=='train' else CONFIG.max_cross_validation_windows_per_step
        if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14'):
            return {step:DataLoader(sensor_discrimination.TrainReferencePairs(
                step1_windows[step][split],step1_provenance[step][split],
                step1_windows[step]['train'],step1_provenance[step]['train'],
                SEED+1000*i+(0 if split=='train' else 100000),limit,split=split),
                CONFIG.cross_batch_size,shuffle=split=='train',num_workers=0)
                for i,step in enumerate(step1_encoders)}
        return {step:DataLoader(RealStep1Pairs(step1_windows[step][split],step1_provenance[step][split],SEED+1000*i+(0 if split=='train' else 100000),limit),CONFIG.cross_batch_size,shuffle=split=='train',num_workers=0) for i,step in enumerate(step1_encoders)}
    def real_numeric_state(encoder,current,reference):
        from pecvd.temporal_bridge import supports, numeric_state as temporal_state
        if supports(encoder): return temporal_state(encoder,current,reference)
        with torch.no_grad():
            a=encoder.tokens(current); b=encoder.tokens(reference); cp,_=encoder.patchify(current); rp,_=encoder.patchify(reference); cr=encoder(current); rr=encoder(reference); fdc=torch.stack([cp.mean(-1),rp.mean(-1),(cp.mean(-1)-rp.mean(-1)).abs(),cp.std(-1,unbiased=False),rp.std(-1,unbiased=False),(cp-rp).abs().mean(-1),(cr-cp).pow(2).mean(-1),(rr-rp).pow(2).mean(-1)],-1)
        return torch.cat([a,b,a-b,(a-b).abs(),fdc],-1)
    TRAIN_REAL_LOADERS=make_real_loaders('train'); VALIDATION_REAL_LOADERS=make_real_loaders('validation')
    backbone.text.requires_grad_(False)
    FROZEN_TEXT_STATE={key:value.detach().cpu().clone() for key,value in backbone.text.state_dict().items()}
    FROZEN_SEMANTIC_STATE={key:value.detach().cpu().clone() for key,value in pretrain_heads.semantic_heads.state_dict().items()}
    # Localization loss trains Q/K attention and numeric/descriptor adapters, not the unused answer/value output.
    cross_parameters=list(backbone.numeric_input_norm.parameters())+list(backbone.numeric.parameters())+list(backbone.numeric_key_norm.parameters())+list(backbone.descriptor_norm.parameters())+list(backbone.fusion_gate.parameters())+[backbone.cross.in_proj_weight,backbone.cross.in_proj_bias]
    cross_optimizer=torch.optim.AdamW(cross_parameters,lr=CONFIG.cross_learning_rate,weight_decay=1e-4)
    cross_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(cross_optimizer,mode='max',factor=.5,patience=CONFIG.cross_scheduler_patience,min_lr=CONFIG.cross_min_learning_rate)
    def real_cross_epoch(loaders,training,epoch):
        backbone.train(training); backbone.text.eval(); totals=defaultdict(float); counts=defaultdict(int); count=0
        context=torch.enable_grad() if training else torch.no_grad()
        with context:
            for step,(encoder,meta) in step1_encoders.items():
                encoder=encoder.to(DEVICE).eval(); phase=meta.get('step_descriptor',step); sensors=[meta.get('sensor_descriptors',{}).get(name,name) for name in meta['sensor_columns']]
                for batch_index,(current,reference) in enumerate(loaders[step]):
                    changed,source,targets=gevis_backbone.reference_evidence_pairs(current.to(DEVICE),reference.to(DEVICE),encoder.patch_len,seed=SEED+epoch*10000+batch_index,temporal=TRAINING_PROFILE in ('fixed_questions_v13','center_relation_v14'))
                    relation_batch=TRAINING_PROFILE=='center_relation_v14' and batch_index%4==3
                    if relation_batch:
                        changed,source,targets=sensor_discrimination.center_relation_pairs(current.to(DEVICE),reference.to(DEVICE),encoder.patch_len,seed=SEED+epoch*10000+batch_index)
                    state=real_numeric_state(encoder,changed,source); n=len(changed)
                    kinds=(['BOOLEAN']*n if TRAINING_PROFILE in ('fixed_questions_v13','center_relation_v14') else [list(gevis_backbone.ANSWER_TYPES)[(batch_index+i//2)%3] for i in range(n)])  # Match the shared inference query/type exactly.
                    questions=gevis_backbone.evidence_questions(phase,n)
                    descriptions=gevis_backbone.token_descriptions(phase,sensors,targets['patch_count'])
                    batch_descriptions=[descriptions]*n
                    if TRAINING_PROFILE=='center_relation_v14' and (relation_batch or batch_index%4==1):
                        from pecvd.experiments.questions import center_descriptions
                        centers=targets.get('center_sensor',targets['sensor'])
                        batch_descriptions=[center_descriptions(descriptions,meta['sensor_columns'],targets['patch_count'],meta['sensor_columns'][int(center)]) for center in centers]
                    _,attention=backbone(state,questions,kinds,batch_descriptions)
                    if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14'):
                        loss,metrics=sensor_discrimination.discrimination_loss(attention,targets,
                            margin=CONFIG.discrimination_margin,hard_weight=CONFIG.discrimination_hard_weight,
                            pair_weight=CONFIG.discrimination_pair_weight)
                    else:
                        loss,metrics=gevis_backbone.numeric_evidence_loss(attention,targets)
                    if training: cross_optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(cross_parameters,1.); cross_optimizer.step()
                    count+=n; totals['loss']+=float(loss.detach())*n
                    for key,value in metrics.items(): totals[key]+=float(value)*n
                    for group,mask in [('cross_run',targets['cross_run']),('self_reference',~targets['cross_run']),('relation',torch.full_like(targets['cross_run'],relation_batch and 'center_sensor' in targets))]:
                        size=int(mask.sum())
                        if size:
                            subset={key:(value[mask] if isinstance(value,torch.Tensor) else value) for key,value in targets.items()}
                            _,group_metrics=gevis_backbone.numeric_evidence_loss(attention.detach()[mask],subset)
                            counts[group]+=size
                            for key,value in group_metrics.items(): totals[group+'_'+key]+=float(value)*size
                encoder.cpu()
        result={key:value/max(1,count) for key,value in totals.items()}|{'state_pairs':count,'question_pairs':count}
        for group,size in counts.items():
            for key in ('evidence_sensor_accuracy','evidence_token_accuracy'):result[group+'_'+key]=totals[group+'_'+key]/size
        return result
    BASELINE_FUSION_MODE=getattr(backbone,'fusion_mode','legacy')
    BASELINE_BINDING_METRICS=real_cross_epoch(VALIDATION_REAL_LOADERS,False,1000)
    backbone.fusion_mode=getattr(CONFIG,'fusion_mode',gevis_backbone.TARGET_FUSION_MODE)
    ANCHOR_INITIAL_METRICS=real_cross_epoch(VALIDATION_REAL_LOADERS,False,1000)
    CROSS_HISTORY=[]; best_score=-math.inf; best_cross_state=None; stale=0; BEST_CROSS_EPOCH=None
    for epoch in range(1,CONFIG.cross_epochs+1):
        train_metrics=real_cross_epoch(TRAIN_REAL_LOADERS,True,epoch)
        validation_metrics=real_cross_epoch(VALIDATION_REAL_LOADERS,False,1000)
        score=.7*validation_metrics['cross_run_evidence_sensor_accuracy']+.3*validation_metrics['cross_run_evidence_token_accuracy']
        if TRAINING_PROFILE=='center_relation_v14':
            if 'relation_evidence_sensor_accuracy' not in validation_metrics:
                raise RuntimeError('No relationship validation cases: require at least 3 sensors and 4 batches for this profile')
            score=.5*validation_metrics['cross_run_evidence_sensor_accuracy']+.2*validation_metrics['cross_run_evidence_token_accuracy']+.3*validation_metrics['relation_evidence_sensor_accuracy']
        # Select only on sensor-free numeric-evidence localization.
        if not math.isfinite(score) or not math.isfinite(validation_metrics['loss']):
            raise RuntimeError('Nonfinite cross-attention validation; no checkpoint saved')
        cross_scheduler.step(score)
        CROSS_HISTORY.append({'epoch':epoch,'validation_selection_score':score,
            'learning_rate':float(cross_optimizer.param_groups[0]['lr']),
            **{'train_'+k:v for k,v in train_metrics.items()},**{'validation_'+k:v for k,v in validation_metrics.items()}})
        print(CROSS_HISTORY[-1],flush=True)
        if score>best_score+1e-6:
            best_score=score; stale=0; BEST_CROSS_EPOCH=epoch
            best_cross_state={'backbone':copy.deepcopy(backbone.state_dict()),'feature_head':copy.deepcopy(pretrain_heads.feature_head.state_dict())}
        else: stale+=1
        if stale>=CONFIG.cross_patience: break
    if best_cross_state is None: raise RuntimeError('Real STEP 1 cross-attention training produced no checkpoint')
    backbone.load_state_dict(best_cross_state['backbone'])
    pretrain_heads.feature_head.load_state_dict(best_cross_state['feature_head'])
    if not all(torch.equal(value.detach().cpu(),FROZEN_TEXT_STATE[key]) for key,value in backbone.text.state_dict().items()):
        raise RuntimeError('Frozen text weights changed during binding training')
    if not all(torch.equal(value.detach().cpu(),FROZEN_SEMANTIC_STATE[key]) for key,value in pretrain_heads.semantic_heads.state_dict().items()):
        raise RuntimeError('Frozen semantic heads changed during binding training')
    CROSS_ATTENTION_METRICS={'split':'validation','source':'frozen real STEP 1 cross-run states',
        **real_cross_epoch(VALIDATION_REAL_LOADERS,False,1000)}
    BINDING_COMPARISON=pd.DataFrame([
        {'model':'start_'+BASELINE_FUSION_MODE,'split':'validation',**BASELINE_BINDING_METRICS},
        {'model':'anchor_before_training','split':'validation',**ANCHOR_INITIAL_METRICS},
        {'model':RUN_TAG,**CROSS_ATTENTION_METRICS}])
    display(BINDING_COMPARISON)
    STEP1_BRIDGE={'encoder_artifact_version':PROCESS_CONFIG.artifacts.encoder,'manifest_sha256':file_sha256(manifest_path),'encoder_hashes':{step:item['sha256'] for step,item in encoder_manifest['step_checkpoints'].items()},'selection_split':'validation','test_used':False}
    display(pd.DataFrame(CROSS_HISTORY)); print(CROSS_ATTENTION_METRICS); print({'step1_bridge':'exact frozen manifest and encoder hashes recorded in checkpoint','selection_split':STEP1_BRIDGE['selection_split'],'test_used':STEP1_BRIDGE['test_used']})
    GROUNDING_HISTORY=[]; GROUNDING_METRICS=[]

else:
    display(pd.DataFrame(SAVED_BACKBONE['cross_history']))
    print('Saved Cross-Attention metrics; no new evaluation performed.')


if not REUSE_SAVED_BACKBONE:
    if globals().get('SESSION_SIGNATURE')!='STEP2_GEVIS_GENERIC_DISCOVERY_V10': raise RuntimeError('Stale STEP2 kernel. Restart and run from the first cell.')
    if ARTIFACT_VERSION!='gevis_backbone_e5_v1': raise RuntimeError(f'Invalid artifact version: {ARTIFACT_VERSION}. Restart and run from the first cell.')
    ARTIFACT_ROOT=candidate_backbone_path(ROOT,PROCESS_CONFIG,RUN_TAG).parent
    ARTIFACT_ROOT.mkdir(parents=True,exist_ok=True); path=ARTIFACT_ROOT/'gevis_backbone.pt'
    if path.exists() and not CONFIG.overwrite_artifacts: raise FileExistsError(f'{path} exists; set overwrite_artifacts=True to replace it')
    question_payload={'typed':{split:[asdict(item) for item in items] for split,items in QUESTION_BANK.items()},'semantic':{split:[asdict(item) for item in items] for split,items in SEMANTIC_QUESTION_BANK.items()}}
    question_hash=hashlib.sha256(json.dumps(question_payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    checkpoint={'schema':'gevis.backbone.generic.v10','artifact_version':ARTIFACT_VERSION,'source_schema':EXPECTED_SOURCE_SCHEMA,'source_module_sha256':SOURCE_MODULE_SHA256,'d_model':CONFIG.d_model,'numeric_dim':CONFIG.numeric_dim,'text_encoder_name':CONFIG.text_encoder_name,'text_encoder_revision':CONFIG.text_encoder_revision,'text_encoder_frozen':True,'question_language':'English','question_bank_sha256':question_hash,'question_count':sum(map(len,QUESTION_BANK.values())),'backbone_state':{k:v.cpu() for k,v in backbone.state_dict().items()},'pretrain_heads_state':{k:v.cpu() for k,v in pretrain_heads.state_dict().items()},'best_text_epoch':BEST_TEXT_EPOCH,'text_metrics':TEXT_METRICS.to_dict('records'),'semantic_metrics':SEMANTIC_METRICS.to_dict('records'),'semantic_by_type':SEMANTIC_BY_TYPE,'semantic_fixed_comparison':SEMANTIC_FIXED_COMPARISON,'baseline_comparison_status':BASELINE_COMPARISON_STATUS,'semantic_diagnostics':SEMANTIC_DIAGNOSTICS,'semantic_curriculum':CURRICULUM_TAG,'step1_bridge':STEP1_BRIDGE,'grounding_metrics':GROUNDING_METRICS,'cross_attention_metrics':CROSS_ATTENTION_METRICS,'binding_version':RUN_TAG,'fusion_mode':backbone.fusion_mode,'text_reuse_sha256':TEXT_REUSE_SHA256,'best_cross_epoch':BEST_CROSS_EPOCH,'binding_comparison':BINDING_COMPARISON.to_dict('records'),'cross_history':CROSS_HISTORY,'config':asdict(CONFIG)}
    checkpoint.update(schema='gevis.backbone.generic.v10',curriculum_contract='Sensor-free generic questions; controlled numeric changes supply sensor and patch targets; Cross-Attention discovers the changed sensor across all signals; untouched test split',numeric_evidence_curriculum='primary training and checkpoint-selection objective; no named-sensor question target',limitations=['Questions are restricted to the declared generic English wording bank.','STEP 1 encoders and E5 text encoder are frozen; Cross-Attention and numeric adapters adapt.','Synthetic differences supervise localization; STEP3 learns typed answers from TRAIN pairs, not equipment fault labels.','Localization is supporting evidence, not causal proof.'])
    checkpoint.update(discovery_training_version=gevis_backbone.REFERENCE_DISCOVERY_VERSION,curriculum_contract='Paired cross-run and self references; independent sensor, signed strength and variable duration; numeric-first generic attention',numeric_evidence_curriculum='injected sensor and affected-patch mass; cross-run validation selection; no verified fault labels')
    checkpoint.update(training_profile=TRAINING_PROFILE,
        sensor_discrimination_source_sha256=sensor_discrimination.source_identity() if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14') else None,
        localization_reference_contract='TRAIN references for both fitting and Validation' if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14') else 'historical within-partition references',
        discrimination_loss_contract='numeric evidence + strongest competing sensor log-margin + paired moved-sensor log-margin' if TRAINING_PROFILE in ('contrastive_v12','fixed_questions_v13','center_relation_v14') else 'numeric evidence only')
    checkpoint.update(question_contract=PRIMARY_GOAL_VERSION, localization_prompt=gevis_backbone.evidence_questions('unused',1)[0], temporal_curriculum=TRAINING_PROFILE in ('fixed_questions_v13','center_relation_v14'), output_tasks=('BOOLEAN','CHOICE','ORDINAL'),localization_support_task='CHOICE_SENSOR; not a STEP3 classification head')
    checkpoint.update(center_relation_curriculum=TRAINING_PROFILE=='center_relation_v14',center_input='numeric token metadata, never question text',relation_supervision='synthetic anchor/partner temporal reversal; not causal or fault labels')
    checkpoint['numeric_bridge']='paired_relational_multiscale_440_v3' if CONFIG.numeric_dim==440 else 'paired_temporal_blind_408_v2' if CONFIG.numeric_dim==408 else 'historical_8'
    torch.save(checkpoint,path); print({'saved':str(path),'schema':checkpoint['schema'],'parameters':sum(p.numel() for p in backbone.parameters())+sum(p.numel() for p in pretrain_heads.parameters())})
else:
    checkpoint=SAVED_BACKBONE
    print('Existing STEP 2 file preserved byte-for-byte; continue to STEP 3.')

