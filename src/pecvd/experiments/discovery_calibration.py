"""Artifact-only preparation and loading for the window discovery policy."""
from dataclasses import asdict
import hashlib,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from .window_policy import WindowConfig,WindowPolicy,fit_window_policy,MODE


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def policy_path(backbone_path):
    return Path(os.environ.get('GEVIS_DISCOVERY_POLICY_PATH',str(Path(backbone_path).parent/'window_evidence_v2.json')))


def policy_source_paths():
    """Single source dependency list for artifact namespace and load identity."""
    here=Path(__file__)
    return (here,here.with_name('window_policy.py'),here.with_name('position_rank_policy.py'),
            here.parents[1]/'temporal_bridge.py',here.parents[1]/'temporal_patchtst.py',
            here.parents[1]/'multiscale_patchtst.py',here.parents[1]/'relational_features.py')


def numeric_runtime_source_digest(root):
    notebook=Path(root)/'notebooks/STEP_3.1_run_gevis_inference.ipynb'
    if not notebook.is_file():return None
    doc=json.loads(notebook.read_text(encoding='utf-8'))
    return hashlib.sha256(''.join(next((c['source'] for c in doc['cells'] if c.get('id')=='03-types'),[])).encode()).hexdigest()


def identity_for(encoder_root,backbone_path,config):
    root=Path(encoder_root);manifest=json.loads((root/'encoder_manifest.json').read_text(encoding='utf-8'))
    numeric_source=numeric_runtime_source_digest(root.parent.parent)
    return {'config':asdict(config),'backbone_sha256':digest(backbone_path),
        'numeric_runtime_source_sha256':numeric_source,
        'manifest_sha256':digest(root/'encoder_manifest.json'),
        'encoder_sha256':{s:digest(root/item['path']) for s,item in manifest['step_checkpoints'].items()},
        'cache_sha256':{s:digest(root/'windows'/f'{s}.npz') for s in manifest['step_checkpoints']},
        'source_sha256':{p.name:digest(p) for p in policy_source_paths()}}


def load_policy(path,identity):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('identity')!=identity:raise ValueError('Window policy identity changed; prepare a NEW policy path in STEP 3, never overwrite old results')
    return WindowPolicy(payload)


def attach_policy(module):
    module.DISCOVERY_POLICY_MODE=os.environ.get('GEVIS_DISCOVERY_MODE',MODE)
    if module.DISCOVERY_POLICY_MODE not in (MODE,'legacy_marginal_v1'):raise ValueError('Unknown discovery mode')
    module.DISCOVERY_POLICY=None
    if module.DISCOVERY_POLICY_MODE!=MODE:return
    module.DISCOVERY_POLICY_PATH=policy_path(module.BACKBONE_PATH)
    if not module.DISCOVERY_POLICY_PATH.is_file():return  # Review explicitly withholds; never falls back.
    config=WindowConfig(**json.loads(module.DISCOVERY_POLICY_PATH.read_text(encoding='utf-8'))['config'])
    module.DISCOVERY_POLICY=load_policy(module.DISCOVERY_POLICY_PATH,identity_for(module.ENCODER_ROOT,module.BACKBONE_PATH,config))


def _ordered_indices(runs,seed,limit):
    """Round robin distinct runs before adding windows from the same run."""
    pools={}
    for i in np.random.default_rng(seed).permutation(len(runs)):pools.setdefault(str(runs[i]),[]).append(int(i))
    result=[]
    while any(pools.values()) and len(result)<limit:
        for values in pools.values():
            if values and len(result)<limit:result.append(values.pop(0))
    return result


def _coverage_indices(runs,cohorts,positions,seed,limit):
    """Stratified independent run/context coverage BEFORE repeated windows."""
    strata={}
    for i in np.random.default_rng(seed).permutation(len(runs)):
        key=(str(cohorts[i]),int(positions[i]))
        strata.setdefault(key,{}).setdefault(str(runs[i]),[]).append(int(i))
    pools={key:[values[0] for values in by_run.values()] for key,by_run in strata.items()}
    result=[]
    while any(pools.values()) and len(result)<limit:
        for values in pools.values():
            if values and len(result)<limit:result.append(values.pop(0))
    chosen=set(result)
    for i in _ordered_indices(runs,seed,len(runs)):
        if len(result)>=limit:break
        if i not in chosen:result.append(i)
    return result


def cached_comparisons(encoder_root,step,*,split,config=WindowConfig(),calibration=False):
    """Prepared standardized arrays; references ALWAYS TRAIN, including Test.

    Only requested current split and Train arrays are read. Other splits' run
    identities alone are checked for leakage. No Test values enter preparation.
    """
    if split not in ('train','validation','test'):raise ValueError('Known split required')
    with np.load(Path(encoder_root)/'windows'/f'{step}.npz',allow_pickle=False) as cache:
        ids={s:cache[f'{s}_process_id'].astype(str) for s in ('train','validation','test')}
        if any(set(ids[a])&set(ids[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))):
            raise ValueError('Run overlap in cached partitions')
        train=cache['train'];current=train if split=='train' else cache[split]
        cohorts=cache[f'{split}_cohort_key'].astype(str);positions=cache[f'{split}_position_bin']
        tc=cache['train_cohort_key'].astype(str);tp=cache['train_position_bin']
    coverage=bool(calibration and config.gate_scope=='position_rank_v4')
    available={}
    if coverage:
        reference_runs={}
        for run,c,p in zip(ids['train'],tc,tp):reference_runs.setdefault((str(c),int(p)),set()).add(str(run))
        eligible=np.array([i for i,(run,c,p) in enumerate(zip(ids[split],cohorts,positions))
            if reference_runs.get((str(c),int(p)),set())-{str(run)}],dtype=int)
        indices=[int(eligible[j]) for j in _coverage_indices(ids[split][eligible],cohorts[eligible],positions[eligible],
            config.seed,config.max_calibration_windows_per_step)]
        for i in eligible:available.setdefault((str(cohorts[i]),int(positions[i])),set()).add(str(ids[split][i]))
    else:indices=_ordered_indices(ids[split],config.seed,config.max_windows_per_step)
    for i in indices:
        matches=np.flatnonzero((tc==cohorts[i])&(tp==positions[i])&(ids['train']!=ids[split][i]))
        references={}
        for j in matches:references.setdefault(str(ids['train'][j]),int(j))
        chosen=list(references.values())[:config.max_references]
        if not chosen:continue
        yield {'split':split,'run_id':str(ids[split][i]),'reference_run_ids':list(references)[:config.max_references],
               'current':np.array(current[i],copy=True),'reference':np.median(train[chosen],axis=0).astype('float32'),
               'cohort':str(cohorts[i]),'position':int(positions[i]),'cache_index':int(i),
               'available_context_runs':len(available[(str(cohorts[i]),int(positions[i]))]) if coverage else None,
               'sampling_contract':'run/cohort/position coverage first' if coverage else 'original evaluation case schedule'}


def residual_trace(state,sensors,length):
    """The STEP 1 bridge's last two features are current/reference patch MSE."""
    if isinstance(state,torch.Tensor):state=state.detach().cpu().numpy()
    state=np.asarray(state)
    if state.ndim!=3 or len(state)!=1 or state.shape[1]%sensors or state.shape[-1]<8:
        raise ValueError('Expected frozen STEP 1 numeric bridge')
    patches=state.shape[1]//sensors
    if length%patches:raise ValueError('Native patch grid mismatch')
    value=np.abs(state[0,:,-2]-state[0,:,-1]).reshape(sensors,patches).T
    return np.repeat(value,length//patches,axis=0)


@torch.no_grad()
def prepare_policy(encoders,encoder_root,backbone_path,device,numeric_state,*,config=WindowConfig(),path=None):
    """Fit policy only, never model weights or source CSVs. Create-only save."""
    config.validate();path=Path(path) if path is not None else policy_path(backbone_path)
    identity=identity_for(encoder_root,backbone_path,config)
    if path.is_file():return load_policy(path,identity)
    partitions={'train':[],'validation':[]}
    for step,(encoder,meta) in encoders.items():
        encoder=encoder.to(device).eval()
        try:
            for split in partitions:
                print(f'Window evidence preparation: {step} / {split} (bounded caches only)',flush=True)
                for index,row in enumerate(cached_comparisons(encoder_root,step,split=split,config=config,calibration=True),1):
                    a=torch.tensor(row['current'][None],dtype=torch.float32,device=device)
                    b=torch.tensor(row['reference'][None],dtype=torch.float32,device=device)
                    state=numeric_state(encoder,a,b)
                    partitions[split].append({**row,'step':step,'sensors':list(meta['sensor_columns']),
                        'residual':residual_trace(state,len(meta['sensor_columns']),len(row['current']))})
                    if index%32==0:print(f'  prepared windows: {index}',flush=True)
        finally:encoder.cpu()
    payload=fit_window_policy(partitions['train'],partitions['validation'],identity,config)
    if set(payload['groups'])!=set(encoders):raise ValueError('Missing matched Train comparisons; no window policy saved')
    if identity_for(encoder_root,backbone_path,config)!=identity:raise ValueError('Model/cache/code changed during preparation; no policy saved')
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation: old thresholds/results remain recoverable, even on a race.
    with path.open('x',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return WindowPolicy(payload)


def calibration_summary(policy):
    return pd.DataFrame([{'step':step,**{k:v for k,v in group.items() if k not in ('sensors','scales','ordinal_edges','center_thresholds','center_ordinal_edges','candidate_scales','legacy_fallback','contexts')}}
                         for step,group in policy.payload['groups'].items()])


def load_discovery_pairs(runtime,step,*,split='test',limit=2,seed=42):
    if split not in ('validation','test') or not 1<=limit<=32:raise ValueError('Bounded held-out pairs required')
    module=runtime.module
    if hasattr(runtime,'external_pairs'):raise ValueError('External window calibration must be prepared separately; source thresholds cannot be reused silently')
    policy=getattr(module,'DISCOVERY_POLICY',None)
    config=policy.config if policy is not None else WindowConfig(seed=seed)
    meta=runtime.encoders[step][1];pairs=[]
    for row in cached_comparisons(module.ENCODER_ROOT,step,split=split,config=config):
        frames=[]
        for value in (row['current'],row['reference']):
            frame=pd.DataFrame(value*np.asarray(meta['std'])+np.asarray(meta['mean']),columns=meta['sensor_columns'])
            frame[module.ORDER_COLUMN]=np.arange(len(value))
            frame.attrs.update(cohort_key=row['cohort'],position_bin=row['position'],
                evaluation_pair_id=hashlib.sha256(f"{split}/{step}/{row['cache_index']}".encode()).hexdigest()[:16],
                evaluation_run_id=hashlib.sha256(row['run_id'].encode()).hexdigest()[:16],reference_split='train')
            frames.append(frame)
        pairs.append(tuple(frames))
        if len(pairs)>=limit:break
    if not pairs:raise ValueError('No held-out comparisons matching TRAIN reference cohort/position')
    return pairs
