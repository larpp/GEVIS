"""External evaluation with SOURCE weights; no external model fitting.

Reference-run normalization is input calibration, not strict source-only scaling.
The channel-independent adapter is an explicit inference change: mean source
sensor embedding, per-channel temporal attention, unchanged decoder parameters.
It must not be presented as native architecture or demonstrated generalization.
"""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn


class NativeTransferUnsupported(ValueError):
    """The saved source forward cannot represent unseen physical sensor IDs."""


class FrozenChannelAdapter(nn.Module):
    def __init__(self, source, n_sensors):
        super().__init__()
        if n_sensors<1: raise ValueError('At least one external physical sensor required')
        self.source=source.requires_grad_(False).eval()
        self.n_sensors=n_sensors
        self.patch_len=source.patch_len
        self.d_model=source.d_model

    @property
    def decoder(self): return self.source.decoder

    def patchify(self,x):
        b,t,s=x.shape
        if s!=self.n_sensors or t%self.patch_len:
            raise ValueError('External window must match sensor count and source patch grid')
        p=t//self.patch_len
        return x.reshape(b,p,self.patch_len,s).permute(0,3,1,2).reshape(b,s*p,self.patch_len),p

    def tokens(self,x,mask=None):
        if hasattr(self.source,'channel_independent_tokens'):
            if x.shape[-1]!=self.n_sensors: raise ValueError('External sensor grid mismatch')
            return self.source.channel_independent_tokens(x,mask)
        z,p=self.patchify(x); b=len(x)
        if p>self.source.patch_emb.num_embeddings: raise ValueError('Source position capacity exceeded')
        content=self.source.input(z).reshape(b*self.n_sensors,p,-1)
        if mask is not None:
            content=torch.where(mask.reshape(b*self.n_sensors,p,1),self.source.mask_token,content)
        # No arbitrary assignment of new physical sensors to old sensor IDs.
        neutral=self.source.sensor_emb.weight.mean(0)
        h=content+neutral+self.source.patch_emb(torch.arange(p,device=x.device))[None]
        return self.source.norm(self.source.encoder(h)).reshape(b,self.n_sensors*p,-1)

    def forward(self,x,mask=None):
        if hasattr(self.source,'channel_independent_forward'):
            return self.source.channel_independent_forward(x,mask)
        return self.decoder(self.tokens(x,mask))


def external_pairs(frame,config,window_length,*,limit=2,seed=42):
    """Independent run split; references never include validation/test runs.

    Fixed row windows are never concatenated across Step occurrences. No temporal
    resampling, interpolation, target labels or external weight training.
    """
    run,step,order=config.columns.run,config.columns.step,config.columns.order
    from .process_inputs import align_step_occurrences
    if window_length<1 or limit<1:raise ValueError('Positive native window length and evaluation limit required')
    frame=align_step_occurrences(frame,run=run,step=step,order=order)
    sensors=list(config.sensor_columns)
    ids=np.asarray(sorted(frame[run].unique()),dtype=object)
    np.random.default_rng(seed).shuffle(ids)
    a=max(1,int(.7*len(ids))); b=max(a+1,a+int(.15*len(ids)))
    splits={'reference':set(ids[:a]),'validation':set(ids[a:b]),'test':set(ids[b:])}
    if any(len(v)<2 for v in splits.values()): raise ValueError('Need at least two independent runs per reference/validation/test split')
    result={}; support=[]
    cohort_columns=[c for c in frame if c.startswith('source_group_')]
    for phase in config.steps:
        rows=frame[frame[step].eq(phase.name)]
        short_segments=0;invalid_windows=0;tail_rows=0
        windows={key:[] for key in splits}
        for _,group in rows.groupby([run,'_gevis_segment_id'],sort=True):
            group=group.sort_values(order)
            split=next(key for key,values in splits.items() if group[run].iloc[0] in values)
            # Complete windows within one occurrence, not just its front.
            if len(group)<window_length:
                short_segments+=1;tail_rows+=len(group);continue
            tail_rows+=len(group)%window_length
            for start in range(0,len(group)-window_length+1,window_length):
                window=group.iloc[start:start+window_length].copy()
                if not np.isfinite(window[sensors].to_numpy(float)).all():
                    invalid_windows+=1;continue
                if any(window[c].nunique(dropna=False)!=1 for c in cohort_columns):
                    raise ValueError('External recipe/cohort changes inside a window')
                # Same phase occurrence AND native offset, not whole-run progress.
                # Exact offsets avoid merging different windows in one coarse bin.
                cohort=(*tuple(window[cohort_columns].iloc[0]),
                    'occurrence',int(window['_gevis_step_occurrence'].iloc[0]),'offset',start)
                position=int(np.clip(float(window['_gevis_step_position_ratio'].iloc[0])*10,0,9))
                window.attrs.update(step_occurrence=int(window['_gevis_step_occurrence'].iloc[0]),
                    native_offset=start,preprocessing_contract='step_occurrence_native_offset_v2')
                windows[split].append((window,cohort,position))
        audit=dict(short_segments=short_segments,nonfinite_windows=invalid_windows,
            unused_tail_rows=tail_rows,alignment='Step occurrence + native row offset + local position',
            **{key+'_runs':len({str(w[run].iloc[0]) for w,_,_ in pool}) for key,pool in windows.items()})
        ref=windows['reference']
        if not ref:
            support.append({'step':phase.name,'status':'NOT_EVALUATED','reason':'no complete reference windows at source context length',**audit})
            continue
        values=np.concatenate([w[sensors].to_numpy(float) for w,_,_ in ref])
        mean=values.mean(0); scale=values.std(0); scale=np.where(scale>1e-6,scale,1.)
        audit['constant_or_near_constant_train_sensors']=int((values.std(0)<=1e-6).sum())
        paired={}
        paired['reference']=ref
        for split in ('validation','test'):
            pairs=[]; seen=set()
            for index in np.random.default_rng(seed).permutation(len(windows[split])):
                current,cohort,position=windows[split][index]
                identity=str(current[run].iloc[0])
                dedup=(identity,str(cohort),position) if split=='validation' else identity
                if dedup in seen: continue
                by_run={}
                for w,c,p in ref:
                    if c==cohort and p==position:by_run.setdefault(str(w[run].iloc[0]),w)
                matches=list(by_run.values())[:16]
                if not matches: continue
                reference=matches[0].copy()
                reference[sensors]=np.median(np.stack([w[sensors].to_numpy(float) for w in matches]),axis=0)
                current=current.copy()
                # Comparable coordinates, not a claim of equal physical duration.
                for w in (current,reference):
                    w[order]=np.arange(window_length)
                    w.attrs['cohort_key']=str(cohort)
                    w.attrs['position_bin']=position
                    w.attrs['evaluation_run_id']=hashlib.sha256(identity.encode()).hexdigest()[:16]
                    w.attrs['evaluation_pair_id']=hashlib.sha256((split+'/'+phase.name+'/'+identity+'/'+str(cohort)+'/'+str(position)).encode()).hexdigest()[:16]
                    w.attrs['reference_split']='train'
                    w.attrs['reference_run_ids']=[hashlib.sha256(str(wref[run].iloc[0]).encode()).hexdigest()[:16]
                        for wref in matches]
                pairs.append((current,reference)); seen.add(dedup)
                if len(pairs)>=(2048 if split=='validation' else limit): break
            if split=='validation':paired['calibration_validation']=pairs
            evaluation=[];evaluation_runs=set()
            for pair in pairs:
                rid=pair[0].attrs['evaluation_run_id']
                if rid in evaluation_runs:continue
                evaluation.append(pair);evaluation_runs.add(rid)
                if len(evaluation)>=limit:break
            paired[split]=evaluation
        ready=bool(paired['test'] and paired['validation'])
        support.append({'step':phase.name,'status':'READY' if ready else 'NOT_EVALUATED',
                        'reason':'matched reference Step occurrence/native offset/local position' if ready else 'no independent matched evaluation pairs',**audit})
        if ready: result[phase.name]=(paired,mean,scale)
    return result,pd.DataFrame(support)


def bind_external_runtime(root,source_runtime,config,frame,*,source_step=None,limit=2,seed=42,allow_legacy_adapter=False,prepare_criteria=True):
    """Source Step chosen BEFORE results, never selected by external accuracy."""
    from pecvd.gevis_ui import notebook_runtime_sources
    from .process_inputs import step_description
    source_step=source_step or next(iter(source_runtime.encoders))
    if source_step not in source_runtime.encoders: raise ValueError('Unknown source encoder Step')
    source,source_meta=source_runtime.encoders[source_step]
    native=bool(getattr(source,'supports_native_sensor_transfer',False))
    if not native and not allow_legacy_adapter:
        raise NativeTransferUnsupported('Frozen transfer is NOT_EVALUATED: this source encoder has fixed sensor IDs. '
                         'PEALD tokens must not be assigned arbitrary PECVD IDs or lose joint attention. '
                         'The retained HYBRID weights are unchanged; native unseen-sensor transfer is not implemented '
                         'for this checkpoint. No PEALD training or automatic portable replacement was performed.')
    pairs,support=external_pairs(frame,config,int(source_meta['window_length']),limit=limit,seed=seed)
    module=ModuleType('_gevis_frozen_transfer')
    module.__dict__.update(source_runtime.module.__dict__)
    module.ORDER_COLUMN=config.columns.order
    module.STEP_ENCODERS={}
    for step,(_,mean,scale) in pairs.items():
        meta={**source_meta,'sensor_columns':list(config.sensor_columns),'mean':mean,'std':scale,
              'step_descriptor':step_description(config,step),
              'sensor_descriptors':dict(config.sensor_descriptors) if getattr(config,'sensor_descriptors',None) else {s:s for s in config.sensor_columns}}
        # Native portable models are the SAME object, parameters and forward;
        # only process input metadata/scalers/reference criteria differ.
        module.STEP_ENCODERS[step]=(source.requires_grad_(False).eval() if native else FrozenChannelAdapter(source,len(config.sensor_columns)).eval(),meta)
    module.GEVIS_MODEL.requires_grad_(False).eval()
    sources=notebook_runtime_sources(Path(root)/'notebooks/STEP_3.1_run_gevis_inference.ipynb')
    exec(compile(sources['03-function'],'external:frozen-functions','exec'),module.__dict__)
    runtime=SimpleNamespace(module=module,encoders=module.STEP_ENCODERS,
        transfer_quality_options=frame.attrs.get('external_quality_options',{}))
    runtime.external_pairs={s:value[0] for s,value in pairs.items()}
    if prepare_criteria and getattr(module,'DISCOVERY_POLICY_MODE',None)=='window_calibrated_v2':
        prepare_external_window_policy(runtime,pairs,config)
    if prepare_criteria and getattr(module,'DISCOVERY_POLICY_MODE',None)!='window_calibrated_v2' and getattr(module,'DECISION_POLICY_MODE','legacy') in ('reference_quantiles_v1','reference_quantiles_v2'):
        from .reference_policy import policy_features
        if module.DECISION_POLICY_MODE=='reference_quantiles_v2':
            from .reference_policy_v2 import ReferencePolicy,ReferenceConfig,fit_records
        else:
            from .reference_policy import ReferencePolicy,ReferenceConfig,fit_records
        config_rule=ReferenceConfig(**source_runtime.module.REFERENCE_POLICY.payload['config'])
        @torch.no_grad()
        def reference_records():
            for phase,(paired,mean,scale) in pairs.items():
                encoder,meta=runtime.encoders[phase]; encoder.to(module.DEVICE).eval()
                try:
                    for current,cohort,position in paired['reference'][:config_rule.max_windows_per_step]:
                        run=str(current[config.columns.run].iloc[0])
                        refs={str(w[config.columns.run].iloc[0]):w for w,c,p in paired['reference']
                              if c==cohort and p==position and str(w[config.columns.run].iloc[0])!=run}
                        if not refs: continue
                        a=torch.tensor(((current[meta['sensor_columns']].to_numpy(float)-mean)/scale).astype('float32')[None],device=module.DEVICE)
                        b=torch.tensor(((np.median(np.stack([w[meta['sensor_columns']].to_numpy(float) for w in list(refs.values())[:config_rule.max_references]]),axis=0)-mean)/scale).astype('float32')[None],device=module.DEVICE)
                        state=module.numeric_state(encoder,a,b)
                        features=policy_features(a,b,state,state.shape[1]//len(meta['sensor_columns']))[0].cpu().numpy()
                        yield {'split':'reference','step':phase,'run_id':run,'cohort':str(cohort),'position':position,
                               'sensors':meta['sensor_columns'],'features':features}
                finally: encoder.cpu()
        module.REFERENCE_POLICY=ReferencePolicy(fit_records(reference_records(),
            {'source_backbone_sha256':module.GEVIS_CHECKPOINT['_artifact_sha256'],
             'scope':'external independent reference runs only; no external weights updated'},config_rule))
    runtime.transfer_metadata={'source_encoder_step':source_step,'weights_updated':False,
        'transfer_condition':('same native portable joint encoder and frozen GEVIS weights' if native else 'legacy comparison only; channel-independent mean-source-ID adapter'),
        'normalization':'external reference runs only; no test normalization fit',
        'native_inference':native,
        'sensor_relationship_limit':('joint sensor/time attention preserved; external quality unverified' if native else 'channel mixing disabled; not native inference'),
        'window_alignment':'Step occurrence/native row offset/local Step position; no physical sub-cycle or equal-duration guarantee',
        'preprocessing_contract':'step_occurrence_native_offset_v2'}
    runtime.transfer_metadata['external_quality_options']=frame.attrs.get('external_quality_options',{})
    return runtime,support


def load_frozen_answer_model(source_runtime,path):
    """Load only a trained decoder bound to this exact source backbone.

    A constructed runtime contains randomly initialized decoder layers until a
    complete compatible answer checkpoint is loaded. Never execute those layers.
    """
    import copy
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Compatible STEP3 answer checkpoint is missing; source backbone alone has no trained answer decoder')
    expected=source_runtime.module.GEVIS_CHECKPOINT.get('_artifact_sha256')
    model=copy.deepcopy(source_runtime.module.GEVIS_MODEL).cpu()
    from pecvd.gevis_fast import load_answer_checkpoint
    load_answer_checkpoint(model,path,backbone_sha256=expected)
    from .discovery_calibration import digest
    model.answer_checkpoint_sha256=digest(path)
    return model


@torch.no_grad()
def predict_external_answers(runtime,*,steps=None,pairs_per_step=2,question_texts=None):
    """Actual learned decoder on untouched windows. No thresholding or injections."""
    from pecvd.answer_display import QUESTIONS,label
    from pecvd.gevis_fast import answer_probabilities
    from .questions import center_descriptions
    if not getattr(runtime,'trained_answer_decoder_loaded',False):
        raise ValueError('A verified trained answer decoder is required before prediction')
    texts=dict(QUESTIONS if question_texts is None else question_texts)
    kinds=[k for k in ('BOOLEAN','CHOICE','ORDINAL') if k in texts]
    if not kinds or any(not isinstance(texts[k],str) or not texts[k].strip() for k in kinds):
        raise ValueError('Supply nonempty English questions keyed by BOOLEAN/CHOICE/ORDINAL')
    if not 1<=pairs_per_step<=32:raise ValueError('Bounded pair count 1..32 required')
    m=runtime.module;model=m.GEVIS_MODEL.to(m.DEVICE).eval();rows=[]
    for step in (list(runtime.encoders) if steps is None else list(steps)):
        encoder,meta=runtime.encoders[step];sensors=list(meta['sensor_columns']);encoder.to(m.DEVICE).eval()
        try:
            for index,(current,reference) in enumerate(runtime.external_pairs[step]['test'][:pairs_per_step]):
                center=sensors[index%len(sensors)]
                a,_=m.prepare_window(current,meta);b,_=m.prepare_window(reference,meta)
                state=m.numeric_state(encoder,a.to(m.DEVICE),b.to(m.DEVICE))
                patches=state.shape[1]//len(sensors)
                descriptions=m.numeric_token_descriptions(step,meta,patches,len(kinds))
                descriptions=[center_descriptions(d,sensors,patches,center) for d in descriptions]
                trained_options=getattr(model,'answer_criteria',None) or m.CRITERIA
                criteria=[list(trained_options[k]) for k in kinds]
                probabilities=answer_probabilities(model,state.expand(len(kinds),-1,-1),
                    [texts[k] for k in kinds],kinds,criteria,descriptions)
                for k,values,options in zip(kinds,probabilities,criteria):
                    probability=values.detach().cpu();chosen=int(probability.argmax())
                    rows.append(dict(step=step,pair_index=index,center_sensor=center,question=texts[k],
                        answer_type=k,answer=label(options[chosen]),
                        probabilities={label(option):float(p) for option,p in zip(options,probability)},
                        max_model_probability=float(probability.max()),injected=False,
                        score_source='trained_answer_decoder_softmax',
                        meaning='uncalibrated model class probabilities; not equipment-fault probability'))
        finally:encoder.cpu()
    return pd.DataFrame(rows)


@torch.no_grad()
def prepare_external_window_policy(runtime,pairs,process_config):
    """Independent external criteria only, in memory; no weight/policy writes."""
    from .window_policy import WindowPolicy,fit_window_policy
    from .discovery_calibration import residual_trace
    from .discovery_calibration import _coverage_indices
    m=runtime.module;source_policy=getattr(m,'DISCOVERY_POLICY',None)
    if source_policy is None:
        import os
        from .window_policy import WindowConfig
        cfg=WindowConfig(gate_scope=os.environ.get('GEVIS_GATE_PROFILE','position_rank_v4'),
            operation_profile=os.environ.get('GEVIS_OPERATION_PROFILE','core'),
            nominal_window_alert_rate=float(os.environ.get('GEVIS_CONTROL_ALERT_BUDGET','.01')))
    else:cfg=source_policy.config
    if cfg.gate_scope!='position_rank_v4':
        raise NativeTransferUnsupported('External window calibration currently requires position_rank_v4; legacy source gates are not reused')
    partitions={'train':[],'validation':[]}
    run_column=process_config.columns.run
    for step,(paired,mean,scale) in pairs.items():
        encoder,meta=runtime.encoders[step];sensors=list(meta['sensor_columns'])
        encoder.to(m.DEVICE).eval()
        def record(current,reference,split,run_id,refs,cohort,position):
            a=((current[sensors].to_numpy(float)-mean)/scale).astype('float32')
            b=((reference[sensors].to_numpy(float)-mean)/scale).astype('float32')
            state=m.numeric_state(encoder,torch.tensor(a[None],device=m.DEVICE),torch.tensor(b[None],device=m.DEVICE))
            return dict(step=step,sensors=sensors,split=split,run_id=run_id,reference_run_ids=refs,
                cohort=str(cohort),position=position,current=a,reference=b,
                residual=residual_trace(state,len(sensors),len(a)))
        try:
            reference_rows=paired['reference']
            available={}
            for frame,cohort,position in reference_rows:
                available.setdefault((str(cohort),int(position)),set()).add(str(frame[run_column].iloc[0]))
            indices=_coverage_indices([str(f[run_column].iloc[0]) for f,_,_ in reference_rows],
                [str(c) for _,c,_ in reference_rows],[p for _,_,p in reference_rows],cfg.seed,cfg.max_calibration_windows_per_step)
            for index in indices:
                current,cohort,position=reference_rows[index]
                run=str(current[run_column].iloc[0]);refs={}
                for frame,c,p in reference_rows:
                    rid=str(frame[run_column].iloc[0])
                    if c==cohort and p==position and rid!=run:refs.setdefault(rid,frame)
                selected=list(refs.items())[:cfg.max_references]
                if not selected:continue
                reference=selected[0][1].copy()
                reference[sensors]=np.median(np.stack([f[sensors].to_numpy(float) for _,f in selected]),axis=0)
                rid=lambda value:hashlib.sha256(str(value).encode()).hexdigest()[:16]
                item=record(current,reference,'train',rid(run),[rid(r) for r,_ in selected],cohort,position)
                item['available_context_runs']=len(available[(str(cohort),int(position))])
                partitions['train'].append(item)
            validation=paired['calibration_validation'];available={}
            for current,_ in validation:
                attrs=current.attrs
                available.setdefault((attrs['cohort_key'],attrs['position_bin']),set()).add(attrs['evaluation_run_id'])
            indices=_coverage_indices([c.attrs['evaluation_run_id'] for c,_ in validation],
                [c.attrs['cohort_key'] for c,_ in validation],[c.attrs['position_bin'] for c,_ in validation],cfg.seed,cfg.max_calibration_windows_per_step)
            for index in indices:
                current,reference=validation[index]
                attrs=current.attrs
                item=record(current,reference,'validation',attrs['evaluation_run_id'],
                    attrs['reference_run_ids'],attrs['cohort_key'],attrs['position_bin'])
                item['available_context_runs']=len(available[(attrs['cohort_key'],attrs['position_bin'])])
                partitions['validation'].append(item)
        finally:encoder.cpu()
    from .discovery_calibration import digest
    identity={'scope':'external_window_reference_validation_only',
        'source_backbone_sha256':m.GEVIS_CHECKPOINT.get('_artifact_sha256'),
        'preprocessing_contract':'step_occurrence_native_offset_v2',
        'process_config_sha256':getattr(process_config,'sha256',None),
        'quality_options':getattr(runtime,'transfer_quality_options',{}),
        'preprocessing_source_sha256':{p.name:digest(p) for p in (
            Path(__file__),Path(__file__).with_name('process_inputs.py'))},
        'weights_updated':False,'test_used_for_calibration':False}
    policy=WindowPolicy(fit_window_policy(partitions['train'],partitions['validation'],identity,cfg))
    m.DISCOVERY_POLICY=policy;m.DISCOVERY_POLICY_PATH=None
    # position_rank_v4 already bypasses any source learned-head selection.
    runtime.external_window_calibrated=True


def load_discovery_pairs(runtime,step,*,split='test',limit=2,seed=42):
    """Dispatch only explicitly calibrated external runtimes; preserve source guard."""
    if not hasattr(runtime,'external_pairs'):
        from .discovery_calibration import load_discovery_pairs as source_pairs
        return source_pairs(runtime,step,split=split,limit=limit,seed=seed)
    if split not in ('validation','test') or not 1<=limit<=32:raise ValueError('Bounded held-out pairs required')
    policy=getattr(runtime.module,'DISCOVERY_POLICY',None)
    if not getattr(runtime,'external_window_calibrated',False) or policy is None or policy.payload.get('identity',{}).get('scope')!='external_window_reference_validation_only':
        raise ValueError('External window calibration must be prepared separately; source thresholds cannot be reused silently')
    return runtime.external_pairs[step][split][:limit]
