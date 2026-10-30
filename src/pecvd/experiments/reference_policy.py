"""Reference-derived criteria, not verified-normal labels or fault probability.

Fit on TRAIN cross-run comparisons only. A repeated run contributes one maximum
per condition/direction, not one independent sample per overlapping window.
No familywise false-alarm guarantee is inferred from marginal quantiles.
"""
from dataclasses import dataclass,asdict
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from .questions import STATISTICS,SCOPES

SCHEMA='gevis.reference_policy.v1'
OPERATION_VERSION='five_operations_local_peak.v1'


@dataclass(frozen=True)
class ReferenceConfig:
    quantiles:tuple=(.95,.975,.99,.995)
    min_runs:int=20
    max_windows_per_step:int=4096
    max_references:int=16
    seed:int=42

    def validate(self):
        if len(self.quantiles)!=4 or not 0<self.quantiles[0]<self.quantiles[1]<self.quantiles[2]<self.quantiles[3]<1:
            raise ValueError('Four explicit increasing reference quantiles in (0,1) required')
        if self.min_runs<2 or self.max_windows_per_step<1 or self.max_references<1:
            raise ValueError('Positive bounded calibration settings required')


def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def policy_features(current,reference,state,patch_count):
    from .operations import semantic_comparison_features
    values=semantic_comparison_features(current,reference,state,patch_count).clone()
    for scope,(lo,hi) in enumerate(((0,current.shape[1]),(0,max(2,current.shape[1]//3)),(2*current.shape[1]//3,current.shape[1]))):
        values[:,scope,:,3]=(current[:,lo:hi]-reference[:,lo:hi]).abs().amax(1)
    return values


def cohort_token(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20] if value is not None else '*'


def context_from_rows(rows):
    return {'cohort':rows.attrs.get('cohort_key'), 'position':rows.attrs.get('position_bin')}


def _keys(step,sensor,scope,cohort,position):
    cohort=cohort_token(cohort); position=str(position) if position is not None else '*'
    # Fallback never pools distinct physical sensors, Steps, scopes or directions.
    return list(dict.fromkeys([(step,sensor,scope,cohort,position),
                              (step,sensor,scope,cohort,'*'),(step,sensor,scope,'*','*')]))


def fit_records(records,identity,config=ReferenceConfig()):
    config.validate(); grouped={}
    for record in records:
        if record['split'] not in ('train','reference'): raise ValueError('Validation/test cannot fit reference thresholds')
        values=np.asarray(record['features'],float); sensors=record['sensors']
        if values.shape!=(3,len(sensors),5) or not np.isfinite(values).all(): raise ValueError('Finite scope/sensor/operation calibration values required')
        for scope in range(3):
            for i,sensor in enumerate(sensors):
                for key in _keys(record['step'],sensor,SCOPES[scope],record.get('cohort'),record.get('position')):
                    for direction,array in (('increase',np.maximum(values[scope,i],0)),
                                            ('decrease',np.maximum(-values[scope,i],0)),('either',np.abs(values[scope,i]))):
                        runs=grouped.setdefault((*key,direction),{})
                        run=cohort_token(record['run_id'])
                        runs[run]=np.maximum(runs[run],array) if run in runs else array.copy()
    groups=[]
    for key,runs in sorted(grouped.items()):
        samples=np.stack(list(runs.values())); n=len(samples)
        thresholds=np.quantile(samples,config.quantiles,axis=0,method='higher').T
        # Require an observed-run tail resolution for every boundary individually.
        ready=[n>=max(config.min_runs,int(math.ceil(1/(1-q)-1e-9))) for q in config.quantiles]
        groups.append({'key':list(key),'n_runs':n,'thresholds':thresholds.tolist(),
                       'quantile_supported':ready,'tail_resolution':1/n})
    return {'schema':SCHEMA,'operation_version':OPERATION_VERSION,'identity':identity,
            'config':asdict(config),'groups':groups,'fit_split':'train_or_external_reference_only',
            'unit':'per-run maximum within matched group',
            'meaning':'unusual relative to unlabelled reference; not verified normal/fault',
            'multiple_testing':'marginal thresholds; validation window/run alert rates must be reported'}


class ReferencePolicy:
    def __init__(self,payload):
        if payload.get('schema')!=SCHEMA or payload.get('operation_version')!=OPERATION_VERSION:
            raise ValueError('Reference policy schema/operation mismatch')
        self.payload=payload; self.groups={tuple(g['key']):g for g in payload['groups']}

    def lookup(self,step,sensor,scope,direction,context=None):
        context=context or {}
        for key in _keys(step,sensor,scope,context.get('cohort'),context.get('position')):
            group=self.groups.get((*key,direction))
            if group and group['quantile_supported'][0]:
                return group,('cohort_position' if key[-1]!='*' else 'cohort' if key[-2]!='*' else 'step_sensor_scope')
        return None,'insufficient_reference_runs_or_tail_resolution'

    def decide(self,features,step,sensor,scope,statistic='level',direction='either',context=None,temperature=.75):
        values=np.asarray(features,float)
        if values.shape!=(5,) or not np.isfinite(values).all() or temperature<=0: raise ValueError('Five finite operations and positive temperature required')
        scope=SCOPES[scope] if isinstance(scope,(int,np.integer)) else scope
        statistic=STATISTICS[statistic] if isinstance(statistic,(int,np.integer)) else statistic
        if scope not in SCOPES or statistic not in STATISTICS or direction not in ('increase','decrease','either'):
            raise ValueError('Unsupported reference operation query')
        group,level=self.lookup(step,sensor,scope,direction,context)
        absolute,abs_level=self.lookup(step,sensor,scope,'either',context)
        slot=STATISTICS.index(statistic)
        directed=float(values[slot] if direction=='increase' else -values[slot] if direction=='decrease' else abs(values[slot]))
        support={'BOOLEAN':False,'CHOICE':False,'ORDINAL':False}; labels={k:None for k in support}; probability={k:None for k in support}
        threshold=float(group['thresholds'][slot][0]) if group else None
        def softmax(logits):
            logits=np.asarray(logits,float)/temperature; exp=np.exp(logits-logits.max()); return (exp/exp.sum()).tolist()
        if group and threshold>0:
            ratio=directed/threshold; support['BOOLEAN']=True; labels['BOOLEAN']=int(directed>threshold)
            probability['BOOLEAN']=softmax([1-ratio,ratio-1])
        elif group: level='zero_reference_scale; cannot assign normalized answer score'
        largest=None; ratios=None
        if absolute:
            edges=np.asarray(absolute['thresholds'],float)
            if np.all(edges[:,0]>0):
                ratios=np.abs(values)/edges[:,0]; largest=float(ratios.max()); winner=int(ratios.argmax())
                support['CHOICE']=True; labels['CHOICE']=winner+1 if largest>1 else 0
                probability['CHOICE']=softmax(np.r_[1-largest,ratios-1])
                if all(absolute['quantile_supported']) and np.all(np.diff(edges,axis=1)>0):
                    grade=int((np.abs(values)[:,None]>edges).sum(1).max())
                    support['ORDINAL']=True; labels['ORDINAL']=grade
                    probability['ORDINAL']=softmax(-(np.arange(5)-grade)**2)
        return {'labels':labels,'probabilities':probability,'supported':support,
                'requested_value':directed,'threshold':threshold,'threshold_quantile':self.payload['config']['quantiles'][0],
                'reference_runs':group['n_runs'] if group else 0,'fallback':level,
                'absolute_reference_runs':absolute['n_runs'] if absolute else 0,'absolute_fallback':abs_level,
                'thresholds':absolute['thresholds'] if absolute else None,
                'normalized_feature_scores':ratios.tolist() if ratios is not None else None,
                'boolean_any':bool(largest>1) if largest is not None else None,
                'ordinal_meaning':'reference quantile rarity band, not physical fault severity',
                'reason':None if all(support.values()) else 'insufficient support, nonpositive scale or unresolved ordinal quantiles'}

    def summary(self):
        import pandas as pd
        return pd.DataFrame([{'step':key[0],'sensor':key[1],'scope':key[2],'direction':key[-1],
            'group_level':'cohort_position' if key[-2]!='*' else 'cohort' if key[-3]!='*' else 'step_sensor_scope',
            'reference_runs':g['n_runs'],'base_quantile_supported':g['quantile_supported'][0],
            'ordinal_quantiles_supported':all(g['quantile_supported']),'tail_resolution':g['tail_resolution']}
            for key,g in self.groups.items()])


def save_policy(policy,path):
    path=Path(path)
    if path.exists(): raise FileExistsError('Reference policy exists; choose a new policy path rather than overwrite')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream: json.dump(policy,stream,ensure_ascii=False,allow_nan=False)


def policy_pointwise_evidence(current,reference,sensor,scope,criterion,*,original_scale=1.):
    """Observed peaks with the SAME calibrated local-peak criterion, if known."""
    from .evidence import pointwise_evidence
    edges=criterion.get('thresholds') if criterion is not None else None
    threshold=float(edges[3][0]) if edges is not None else None
    ready=threshold is not None and threshold>0
    result=pointwise_evidence(current,reference,sensor,scope,
        threshold=threshold if ready else 1.,original_scale=original_scale)
    result={**result,'criterion_supported':ready,'threshold_basis':'reference local-peak quantile' if ready else 'unavailable; peak only'}
    if not ready:
        result.update(threshold_train_std=None,exceedance_intervals=[],exceedance_rows=None,longest_exceedance_rows=None)
    return result


def load_policy(path,identity):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('identity')!=identity: raise ValueError('Reference policy identity mismatch; create a new policy for this model/cache/config')
    return ReferencePolicy(payload)


def runtime_decision(module,features,step,sensor,scope,statistic,direction,rows):
    """None means explicit legacy mode, not unsupported reference calibration."""
    if getattr(module,'DECISION_POLICY_MODE','legacy')=='legacy': return None
    if module.DECISION_POLICY_MODE not in ('reference_quantiles_v1','reference_quantiles_v2'): raise ValueError('Unknown decision policy')
    if not hasattr(module,'REFERENCE_POLICY'): raise RuntimeError('Reference policy missing; run STEP 3 criterion preparation first')
    def index(value): return int(value.item()) if isinstance(value,torch.Tensor) else int(value)
    return module.REFERENCE_POLICY.decide(np.asarray(features.detach().cpu() if isinstance(features,torch.Tensor) else features),
        step,module.STEP_ENCODERS[step][1]['sensor_columns'][index(sensor)],index(scope),index(statistic),
        ('increase','decrease','either')[index(direction)],context_from_rows(rows))


def identity_for(root,encoder_root,backbone_path,config):
    config.validate(); root=Path(root); encoder_root=Path(encoder_root)
    manifest=json.loads((encoder_root/'encoder_manifest.json').read_text(encoding='utf-8'))
    return {'manifest_sha256':digest_file(encoder_root/'encoder_manifest.json'),
            'backbone_sha256':digest_file(backbone_path),'config':json.loads(json.dumps(asdict(config))),
            'operation_version':OPERATION_VERSION,
            'policy_source_sha256':digest_file(Path(__file__)),
            'operation_source_sha256':digest_file(Path(__file__).with_name('operations.py')),
            'cache_sha256':{s:digest_file(encoder_root/'windows'/f'{s}.npz') for s in manifest['step_checkpoints']}}


@torch.no_grad()
def cached_records(encoders,encoder_root,device,numeric_state,config=ReferenceConfig(),*,split='train'):
    """TRAIN fitting / validation reporting. Loads only the requested cache split."""
    if split not in ('train','validation'): raise ValueError('Test never enters criterion preparation/validation')
    from collections import defaultdict
    for step,(encoder,meta) in encoders.items():
        print(f'Reference criteria {split}: preparing one Step (limit {config.max_windows_per_step} windows)',flush=True)
        encoder.to(device).eval()
        try:
            with np.load(Path(encoder_root)/'windows'/f'{step}.npz',allow_pickle=False) as cache:
                x=cache[split]; runs=cache[f'{split}_process_id'].astype(str)
                cohorts=cache[f'{split}_cohort_key'].astype(str); positions=cache[f'{split}_position_bin']
                train_runs=set(cache['train_process_id'].astype(str)); val_runs=set(cache['validation_process_id'].astype(str)); test_runs=set(cache['test_process_id'].astype(str))
                if train_runs&val_runs or train_runs&test_runs or val_runs&test_runs: raise ValueError('Run leakage in cached partitions')
            pools=defaultdict(list)
            for i,key in enumerate(zip(cohorts,positions)): pools[key].append(i)
            indices=np.random.default_rng(config.seed).permutation(len(x))
            count=0
            for i in indices:
                pool=[j for j in pools[(cohorts[i],positions[i])] if runs[j]!=runs[i]]
                # One reference window per OTHER run avoids over-weighting dense runs.
                unique={}
                for j in pool: unique.setdefault(runs[j],j)
                chosen=list(unique.values())[:config.max_references]
                if not chosen: continue
                a=torch.as_tensor(x[i:i+1],device=device); b=torch.as_tensor(np.median(x[chosen],axis=0)[None].astype(np.float32),device=device)
                state=numeric_state(encoder,a,b); features=policy_features(a,b,state,state.shape[1]//len(meta['sensor_columns']))[0].cpu().numpy()
                yield {'split':split,'step':step,'run_id':runs[i],'cohort':cohorts[i],'position':int(positions[i]),
                       'sensors':meta['sensor_columns'],'features':features}
                count+=1
                if count%64==0: print(f'  reference comparisons: {count}',flush=True)
                if count>=config.max_windows_per_step: break
        finally: encoder.cpu()


def validation_alert_rates(policy,records):
    """Unlabelled control rates, not verified-normal FPR; no threshold selection."""
    import pandas as pd
    windows=[]; by_run={}
    for r in records:
        if r['split']!='validation': raise ValueError('Validation-only control report required')
        alerts=[]; supported=0; total=0
        for scope in range(3):
            for i,sensor in enumerate(r['sensors']):
                for stat in STATISTICS:
                    decision=policy.decide(r['features'][scope,i],r['step'],sensor,scope,stat,'either',r)
                    total+=1
                    if decision['supported']['BOOLEAN']: alerts.append(bool(decision['labels']['BOOLEAN'])); supported+=1
        full=supported==total
        alert=any(alerts) if full else None
        windows.append({'step':r['step'],'supported':full,'alert':alert,'query_support':supported/total})
        values=by_run.setdefault((r['step'],r['run_id']),[]); values.append(alert)
    out=[]
    pooled={}
    for (_,run),values in by_run.items(): pooled.setdefault(run,[]).extend(values)
    for step in [*dict.fromkeys(w['step'] for w in windows),'ALL_EVALUATED_STEPS']:
        selected=[w for w in windows if step=='ALL_EVALUATED_STEPS' or w['step']==step]
        ready=[w for w in selected if w['supported']]
        runs=list(pooled.values()) if step=='ALL_EVALUATED_STEPS' else [v for (s,_),v in by_run.items() if s==step]
        complete=[v for v in runs if all(a is not None for a in v)]
        out.append({'step':step,'validation_windows':len(selected),'fully_supported_windows':len(ready),
            'untouched_window_alert_rate':np.mean([w['alert'] for w in ready]) if ready else None,
            'validation_runs':len(runs),'fully_supported_runs':len(complete),
            'untouched_run_alert_rate':np.mean([any(v) for v in complete]) if complete else None,
            'meaning':'observed sampled windows/Steps only; not full-run coverage, normal labels or FPR guarantee'})
    return pd.DataFrame(out)
