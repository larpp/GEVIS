"""Frozen PatchTST latent-to-row segmentation, NEVER an alert detector.

TRAIN synthetic masks supervise a shared temporal head. Validation selects its
snapshot and segmentation cutoff. Test and runtime masks never enter fitting.
Numeric detection, support, sensor ranking and type are left unchanged.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy,hashlib,json
import numpy as np
import torch
from torch import nn

SCHEMA='gevis.temporal_localizer.v1'


@dataclass(frozen=True)
class LocalizerConfig:
    width:int=32
    epochs:int=12
    learning_rate:float=.001
    max_windows_per_step:int=32
    cases_per_window:int=4
    seed:int=42

    def validate(self):
        if not (1<=self.width<=256 and 1<=self.epochs<=100 and 0<self.learning_rate<1
                and 1<=self.max_windows_per_step<=256 and 1<=self.cases_per_window<=8):
            raise ValueError('Bounded temporal localization configuration required')


class TemporalHead(nn.Module):
    def __init__(self,features,rows_per_patch,width=32):
        super().__init__()
        if features<1 or rows_per_patch<1:raise ValueError('Positive latent feature/grid dimensions required')
        self.features=features;self.rows_per_patch=rows_per_patch
        self.network=nn.Sequential(nn.LayerNorm(features),nn.Linear(features,width),nn.GELU())
        self.temporal=nn.Conv1d(width,width,3,padding=1)
        self.output=nn.Linear(width,rows_per_patch)

    def forward(self,tokens,rows_per_patch=None):
        if tokens.ndim!=3 or tokens.shape[-1]!=self.features:
            raise ValueError('Expected sensor/batch by patch by latent features')
        h=self.network(tokens)
        h=h+torch.nn.functional.gelu(self.temporal(h.transpose(1,2)).transpose(1,2))
        output=self.output(h)
        grid=rows_per_patch or self.rows_per_patch
        if grid!=self.rows_per_patch:
            output=nn.functional.interpolate(output.reshape(-1,1,self.rows_per_patch),
                size=grid,mode='linear',align_corners=False).reshape(len(tokens),tokens.shape[1],grid)
        return output.flatten(1)


def latent_grid(state,sensors,length,*,latent_width=None):
    """Use only the four learned latent blocks; exclude handcrafted features."""
    if state.ndim!=3 or len(state)!=1 or state.shape[1]%sensors or state.shape[-1]<=8:
        raise ValueError('Expected current/reference PatchTST latent bridge')
    patches=state.shape[1]//sensors
    if length%patches:raise ValueError('Unsupported nonuniform row-to-patch grid')
    features=state.shape[-1]-8 if latent_width is None else int(latent_width)
    if features<4 or features%4 or features>state.shape[-1]-8:
        raise ValueError('Expected four equal latent blocks before the numeric features')
    return state[0,:,:features].reshape(sensors,patches,features),length//patches


def _iou(mask,target):
    union=(mask|target).sum()
    return float((mask&target).sum()/union) if union else 1.


def fit_localizer(train,validation,config=LocalizerConfig(),*,device='cpu',train_run_ids=None,validation_run_ids=None):
    """Invented or cached latent records, with disjoint independent run IDs."""
    config.validate();train=list(train);validation=list(validation)
    if not train or not validation:raise ValueError('TRAIN and Validation localization records required')
    observed_train={str(r['run_id']) for r in train}|{str(v) for r in train for v in r.get('reference_run_ids',[])}
    tids=observed_train if train_run_ids is None else set(map(str,train_run_ids))
    vids={str(r['run_id']) for r in validation} if validation_run_ids is None else set(map(str,validation_run_ids))
    if not observed_train<=tids or not {str(r['run_id']) for r in validation}<=vids:
        raise ValueError('Localization records outside declared partitions')
    if tids&vids:raise ValueError('Localization Train/Validation run overlap')
    for split,rows in (('train',train),('validation',validation)):
        for r in rows:
            if r.get('split')!=split:raise ValueError('Test never fits localization')
            if str(r['run_id']) in set(map(str,r.get('reference_run_ids',[]))):
                raise ValueError('Self-run localization reference is forbidden')
            if not set(map(str,r.get('reference_run_ids',[])))<=tids:
                raise ValueError('Localization references must be TRAIN only')
            x=torch.as_tensor(r['tokens']);y=torch.as_tensor(r['mask'])
            if x.ndim!=3 or y.ndim!=2 or x.shape[0]!=y.shape[0] or not torch.isfinite(x).all():
                raise ValueError('Finite aligned latent tokens and row masks required')
            if not torch.isfinite(y).all() or ((y<0)|(y>1)).any():raise ValueError('Binary mask required')
        if not any(np.asarray(r['mask']).any() for r in rows):
            raise ValueError('Effective localization changes required in both partitions')
    features=train[0]['tokens'].shape[-1]
    grid=16  # normalized within-patch coordinates, independent of process patch length
    # Do not modify the caller's RNG state or any encoder/model weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed);head=TemporalHead(features,grid,config.width).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=config.learning_rate)
    best=None;best_loss=float('inf');updates=0
    def loss_on(r,update=False):
        x=torch.as_tensor(r['tokens'],dtype=torch.float32,device=device).detach()
        y=torch.as_tensor(r['mask'],dtype=torch.float32,device=device)
        if y.shape[-1]%x.shape[1]:raise ValueError('Unsupported temporal grid')
        logits=head(x,y.shape[-1]//x.shape[1])
        if logits.shape!=y.shape:raise ValueError('Localization row grid changed')
        probability=logits.sigmoid()
        positive=y.sum(1)>0
        weights=torch.ones(len(y),device=device)/len(y)
        if positive.any() and (~positive).any():
            weights=torch.where(positive,.5/positive.sum(),.5/(~positive).sum())
        bce=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none').mean(1)
        dice=1-(2*(probability*y).sum(1)+1)/(probability.sum(1)+y.sum(1)+1)
        return ((bce+dice)*weights).sum()
    for epoch in range(config.epochs):
        head.train()
        for r in train:
            optimizer.zero_grad(set_to_none=True);loss=loss_on(r)
            if not torch.isfinite(loss):raise ValueError('Nonfinite localization training loss')
            loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step();updates+=1
        head.eval()
        with torch.no_grad():loss=float(torch.stack([loss_on(r) for r in validation]).mean())
        if loss<best_loss:best_loss=loss;best=copy.deepcopy(head.state_dict())
    head.load_state_dict(best);head.eval()
    scores=[]
    with torch.no_grad():
        predictions=[head(torch.as_tensor(r['tokens'],dtype=torch.float32,device=device),
            r['mask'].shape[-1]//r['tokens'].shape[1]).sigmoid().cpu().numpy() for r in validation]
    for cutoff in (.3,.5,.7):
        measures=[]
        for p,r in zip(predictions,validation):
            target=np.asarray(r['mask'],bool);active=target.any(1)
            per_sensor=np.array([_iou(a>=cutoff,b) for a,b in zip(p,target)])
            measures.append(np.mean([per_sensor[g].mean() for g in (active,~active) if g.any()]))
        score=np.mean(measures)
        scores.append((float(score),cutoff))
    cutoff=max(scores,key=lambda pair:(pair[0],pair[1]))[1]
    return head.cpu(),cutoff,dict(training_updates=updates,fit_split='train',selection_split='validation',
        test_used=False,supervision='synthetic changed-sample masks; not fault labels',
        detection_unchanged=True,validation_loss=best_loss)


def localizer_identity(module,config=LocalizerConfig()):
    from .discovery_calibration import identity_for
    from .window_policy import WindowConfig
    return dict(schema=SCHEMA,config=asdict(config),
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bridge=identity_for(module.ENCODER_ROOT,module.BACKBONE_PATH,WindowConfig()))


def localizer_path(module,config=LocalizerConfig()):
    identity=localizer_identity(module,config)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    return Path(module.BACKBONE_PATH).parent/f'temporal_localizer_{fingerprint}.pt'


def save_localizer(head,cutoff,evidence,path,identity,config=LocalizerConfig()):
    if not np.isfinite(cutoff) or not 0<cutoff<1:raise ValueError('Finite segmentation cutoff required')
    if not evidence.get('training_updates') or evidence.get('test_used') or not evidence.get('detection_unchanged'):
        raise ValueError('Trained localization evidence without Test selection required')
    payload=dict(schema=SCHEMA,identity=identity,config=asdict(config),features=head.features,
        rows_per_patch=head.rows_per_patch,cutoff=cutoff,evidence=evidence,
        state={k:v.detach().cpu() for k,v in head.state_dict().items()})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:torch.save(payload,stream)


def load_localizer(path,identity):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!=SCHEMA or payload.get('identity')!=identity:
        raise ValueError('Temporal localizer identity mismatch')
    config=LocalizerConfig(**payload['config']);config.validate()
    if not np.isfinite(payload['cutoff']) or not 0<payload['cutoff']<1:
        raise ValueError('Finite segmentation cutoff required')
    if any(not torch.isfinite(t).all() for t in payload['state'].values()):
        raise ValueError('Nonfinite temporal localizer weights')
    if not payload['evidence'].get('training_updates') or payload['evidence'].get('test_used'):
        raise ValueError('Temporal localizer lacks valid training evidence')
    head=TemporalHead(payload['features'],payload['rows_per_patch'],config.width)
    head.load_state_dict(payload['state'],strict=True)
    return head.requires_grad_(False).eval(),float(payload['cutoff'])


def attach_localizer(module):
    """Runtime never trains. Missing learned intervals stay explicitly missing."""
    import os
    mode=os.environ.get('GEVIS_INTERVAL_MODE','numeric')
    if mode not in ('numeric','learned'):raise ValueError('Unknown temporal localization mode')
    module.INTERVAL_MODE=mode;module.TEMPORAL_LOCALIZER=None
    if mode=='numeric':return
    path=localizer_path(module)
    module.TEMPORAL_LOCALIZER_PATH=path
    if path.is_file():module.TEMPORAL_LOCALIZER=load_localizer(path,localizer_identity(module))


@torch.no_grad()
def apply_localization(module,result,state,sensors,length,*,latent_width=None):
    """Modify interval fields only. Never create/remove/reorder an event."""
    from .window_policy import merged_intervals
    if getattr(module,'INTERVAL_MODE','numeric')!='learned':return
    localizer=getattr(module,'TEMPORAL_LOCALIZER',None)
    result['interval_source']='learned PatchTST latent segmentation' if localizer else 'unavailable: prepare STEP3 temporal localizer'
    probabilities=None
    if localizer is not None and result['events']:
        tokens,grid=latent_grid(state,len(sensors),length,latent_width=latent_width);head,cutoff=localizer
        head.to(state.device).eval();probabilities=head(tokens,grid).sigmoid().cpu().numpy()
    for row in result['ranked_evidence']:
        row['numeric_detected_rows']=copy.deepcopy(row.get('detected_rows'))
        row['numeric_exceedance_intervals']=copy.deepcopy(row['pointwise'].get('exceedance_intervals',[]))
        intervals=[]
        if probabilities is not None and row['changed']:
            sid=list(sensors).index(row['sensor']);lo,hi=row['scope_rows']
            intervals=[(a+lo,b+lo) for a,b in merged_intervals(probabilities[sid,lo:hi]>=cutoff,0,1)]
            peak=int(probabilities[sid,lo:hi].argmax())+lo
            row['detected_rows']=next((list(iv) for iv in intervals if iv[0]<=peak<iv[1]),None)
        else:row['detected_rows']=None
        row['localization_intervals']=intervals
        row['localization_basis']=result['interval_source']
    # events and winner reference these same evidence dictionaries.


@torch.no_grad()
def cached_localization_records(module,split,config=LocalizerConfig()):
    """Offline frozen feature extraction. Deliberately disallows Test."""
    if split not in ('train','validation'):raise ValueError('Test never prepares localization')
    from .discovery_calibration import cached_comparisons
    from .window_policy import WindowConfig
    from .injection import Injection,inject
    cfg=WindowConfig(max_windows_per_step=config.max_windows_per_step,seed=config.seed)
    for step,(encoder,meta) in module.STEP_ENCODERS.items():
        encoder.to(module.DEVICE).eval();names=list(meta['sensor_columns'])
        try:
            for i,row in enumerate(cached_comparisons(module.ENCODER_ROOT,step,split=split,config=cfg)):
                current=row['current'];reference=row['reference'];n=len(current)
                variants=[(current,np.zeros_like(current,dtype=bool))]
                rng=np.random.default_rng(config.seed+i)
                for j in range(config.cases_per_window):
                    duration=int(rng.integers(1,n+1));start=int(rng.integers(0,n-duration+1))
                    spec=Injection(('level_shift','drift','spike','variance')[j%4],(i+j)%len(names),start,start+duration,
                        (.5,2.)[j%2],config.seed+i*8+j)
                    changed,truth=inject(current,spec,np.ones(len(names)))
                    if truth['effective']:variants.append((changed,truth['changed_mask']))
                for changed,mask in variants:
                    a=torch.as_tensor(changed[None],dtype=torch.float32,device=module.DEVICE)
                    b=torch.as_tensor(reference[None],dtype=torch.float32,device=module.DEVICE)
                    tokens,_=latent_grid(module.numeric_state(encoder,a,b),len(names),n,latent_width=4*encoder.d_model)
                    yield dict(split=split,run_id=row['run_id'],reference_run_ids=row['reference_run_ids'],
                        tokens=tokens.detach().cpu().numpy(),mask=mask.T.astype('float32'))
        finally:encoder.cpu()


def prepare_localizer(module,config=LocalizerConfig()):
    """STEP3-only preparation. Exact artifact reuse or create; no overwrite."""
    config.validate();path=localizer_path(module,config);identity=localizer_identity(module,config)
    if path.is_file():return load_localizer(path,identity)
    print('Preparing temporal localization: frozen PatchTST, TRAIN masks, disjoint Validation; never Test.',flush=True)
    train=list(cached_localization_records(module,'train',config))
    validation=list(cached_localization_records(module,'validation',config))
    # Validate against the complete declared run populations, not just the
    # bounded feature sample (an unsampled TRAIN reference is still valid).
    populations={'train':set(),'validation':set()}
    for step in module.STEP_ENCODERS:
        with np.load(Path(module.ENCODER_ROOT)/'windows'/f'{step}.npz',allow_pickle=False) as cache:
            for split in populations:populations[split].update(map(str,cache[f'{split}_process_id']))
    head,cutoff,evidence=fit_localizer(train,validation,config,device=module.DEVICE,
        train_run_ids=populations['train'],validation_run_ids=populations['validation'])
    if identity!=localizer_identity(module,config):raise ValueError('Localization sources changed during preparation')
    save_localizer(head,cutoff,evidence,path,identity,config)
    return head.requires_grad_(False).eval(),cutoff
