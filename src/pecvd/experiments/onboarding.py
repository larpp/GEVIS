"""Local PEALD structural onboarding, not fault labelling or model evaluation.

The preset is enabled only for an exact set of recognizable equipment columns.
Unrecognized datasets fail closed rather than guessing run identity or sensors.
Generated mappings, descriptions and profiles stay under ignored .cache.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .process_inputs import phase_key


def prepare_peald(root, source='data/merge_peald.csv', *, seed=42, chunk_size=20000):
    root=Path(root).resolve(); source=Path(source)
    if source.is_absolute() or '..' in source.parts: raise ValueError('Project-relative external CSV required')
    path=root/source
    destination=root/'.cache/external/peald/process.json'
    if destination.exists(): raise FileExistsError('Local PEALD mapping exists; inspect it before replacing')
    mapping={'run':'ActivePjobID','step':'PPExecStepName','order':'Timestamp',
             'cohort':'PMStoredProcessRecipeName','step_context':'PPExecname'}
    with path.open(encoding='utf-8-sig',newline='') as stream: header=next(csv.reader(stream))
    if not set(mapping.values())<=set(header): raise ValueError('PEALD metadata preset is not applicable')
    meta=pd.read_csv(path,usecols=list(mapping.values()),dtype='string')
    valid=meta[mapping['run']].notna()&meta[mapping['step']].notna()
    valid &= meta[mapping['run']].str.strip().ne('')&meta[mapping['step']].str.strip().ne('')
    meta=meta.loc[valid].copy()
    stamp=pd.to_datetime(meta[mapping['order']],format='mixed',errors='coerce')
    if stamp.isna().any(): raise ValueError('Unparseable PEALD timestamps')
    if not stamp.is_monotonic_increasing: raise ValueError('Input ordering needs explicit review before automatic onboarding')
    ids=meta[mapping['run']]
    blocks=ids[ids.ne(ids.shift()).fillna(True)]
    if blocks.duplicated().any(): raise ValueError('Job ID reappears across disjoint blocks; run boundary is ambiguous')
    if meta.groupby(mapping['run'])[mapping['cohort']].nunique().gt(1).any():
        raise ValueError('Stored process recipe changes inside a job')
    runs=np.array(sorted(ids.unique()),dtype=object); np.random.default_rng(seed).shuffle(runs)
    n=len(runs); a=max(1,int(.70*n)); b=max(a+1,a+int(.15*n))
    splits={key:set(values) for key,values in (('train',runs[:a]),('validation',runs[a:b]),('test',runs[b:]))}
    if min(map(len,splits.values()))<2: raise ValueError('Not enough independent jobs for cross-run validation/test references')
    meta['_phase']=[phase_key(s,c) for s,c in zip(meta[mapping['step']],meta[mapping['step_context']].fillna(''))]
    meta['_segment']=(ids.ne(ids.shift())|meta['_phase'].ne(meta['_phase'].shift())).cumsum()
    segments=meta.groupby([mapping['run'],'_phase','_segment'],dropna=False).agg(
        length=('_phase','size'),cohort=(mapping['cohort'],'first')).reset_index()
    steps=[]; readiness=[]
    for phase,group in segments.groupby('_phase',sort=True):
        train=group[group[mapping['run']].isin(splits['train'])]
        length=min(64,int(train.length.quantile(.25))//4*4) if len(train) else 0
        eligible=group[group.length>=max(4,length)]
        peers=[]
        for split,run_ids in splits.items():
            counts=eligible[eligible[mapping['run']].isin(run_ids)].groupby('cohort')[mapping['run']].nunique()
            peers.append(set(counts[counts>=2].index))
        ready=length>=4 and bool(set.intersection(*peers))
        readiness.append({'phase':phase,'ready':ready,'reason':'ready' if ready else 'short_segments_or_insufficient_matched_run_support'})
        if not ready: continue
        patch=next(p for p in (8,4,2,1) if length%p==0 and length//p>=2)
        sample=meta[meta._phase.eq(phase)].iloc[0]
        description=str(sample[mapping['step_context']])+'; '+str(sample[mapping['step']])
        steps.append({'name':phase,'description':description,'window_length':length,'patch_length':patch,'stride':patch})
    if not steps: raise ValueError('No PEALD phase has enough independent matched train/validation/test runs')
    candidates=[name for name in header if ((name.startswith(('Temp_','MFC','AUX','VG')) and '_Set_' not in name)
                or name in ('O2Density_Monitor_Value','APC Valve Value (Angle)'))]
    count=np.zeros(len(candidates)); mean=count.copy(); m2=count.copy(); total=0
    for chunk in pd.read_csv(path,usecols=[mapping['run'],mapping['step'],*candidates],
                             dtype={mapping['run']:'string',mapping['step']:'string'},chunksize=chunk_size,low_memory=False):
        selected=chunk[mapping['run']].isin(splits['train'])&chunk[mapping['step']].notna()&chunk[mapping['step']].str.strip().ne('')
        values=chunk.loc[selected,candidates].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        total+=len(values)
        if not len(values): continue
        finite=np.isfinite(values); size=finite.sum(0); cleaned=np.where(finite,values,0.)
        batch_mean=cleaned.sum(0)/np.maximum(size,1)
        batch_m2=(np.where(finite,values-batch_mean,0.)**2).sum(0)
        delta=batch_mean-mean; combined=count+size
        m2+=batch_m2+delta**2*count*size/np.maximum(combined,1)
        mean+=delta*size/np.maximum(combined,1); count=combined
    sensors=[name for i,name in enumerate(candidates) if count[i]>=.98*total and count[i]>1 and m2[i]/count[i]>1e-12]
    if not sensors: raise ValueError('No sufficiently observed nonconstant TRAIN measurement channels')
    payload={'schema':'gevis.process_config.v1','process_family':'PEALD','data_path':str(Path(source).as_posix()),
        'columns':{'run':'process_key','step':'step_label','order':'row_in_process'},'source_columns':mapping,
        'sensor_columns':sensors,'artifacts':{'encoder':'gevis_peald_rich_ssl_v1','decision':'gevis_peald_evidence_v3','legacy_decision':'gevis_peald_legacy'},
        'steps':steps,'onboarding':{'seed':seed,'run_identity_basis':'contiguous ActivePjobID; not certified wafer identity',
            'sensor_selection':'recognized measurement names; >=98% finite and nonconstant on TRAIN only',
            'sensor_support':[{'sensor':name,'ready':name in sensors,
                'reason':'ready' if name in sensors else 'insufficient_finite_TRAIN_values_or_constant_channel_needs_physical_scale'} for name in candidates],
            'phase_support':readiness,'unprepared_phases_are_not_evaluated':True}}
    from pecvd.process_config import load_process_config
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    load_process_config(destination)
    return destination
