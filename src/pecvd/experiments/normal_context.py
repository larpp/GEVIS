"""Normal-only, context-conditioned patch-statistic quantiles for GEVIS.

This model never receives the measurement it predicts as an input. It uses
Step/recipe/sensor and already elapsed native time. Learned intervals are
features for the typed head, not rule-generated anomaly answers or guaranteed
prediction coverage. No TRAIN waveform is fetched by inference.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RECIPE='normal_context_ordered_quantiles_v1'
QUANTILES=(.05,.5,.95)
FEATURES=8
EVIDENCE_DIM=32
MIN_RUN_GROUPS=2


def _json(value):
    return json.dumps(value,sort_keys=True,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))


def sha(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _key(step,recipe,sensor):return (str(step),'' if recipe is None else str(recipe),str(sensor))


def request_identity(runtime,steps,config):
    preparation=runtime.prepared.identity
    return dict(recipe=RECIPE,code_sha256=digest(__file__),steps=list(steps),
        source_files=preparation.get('source_files',{}),train_groups=list(runtime.prepared.split_runs['train']),
        delivery_reservation=preparation.get('delivery_metadata_identity'),
        reserved_groups=preparation.get('reserved_delivery_groups',[]),
        native_code=preparation.get('native_preparation_sha256'),
        metadata={s:runtime.encoders[s][1] for s in steps},
        patches={s:int(runtime.encoders[s][0].patch_len) for s in steps},
        config=dict(seed=config.seed,epochs=config.normal_epochs,width=config.normal_width,
            max_seconds=config.normal_max_seconds,batch_size=config.normal_batch_size,
            windows_per_run=config.normal_windows_per_run,learning_rate=config.normal_learning_rate))


def patch_stats(values,patch):
    values=np.asarray(values,np.float32)
    if values.ndim!=2 or values.shape[0]%patch or patch<2 or not np.isfinite(values).all():
        raise ValueError('Complete finite normal patch grid required')
    count=len(values)//patch
    grid=values.reshape(count,patch,values.shape[1]).transpose(2,0,1)
    differences=np.diff(grid,axis=-1)
    return np.stack([grid.mean(-1),grid.std(-1),grid.min(-1),grid.max(-1),grid[...,0],grid[...,-1],
        differences.mean(-1),differences.std(-1)],-1).astype(np.float32)


def prepare_data(runtime,steps,config):
    """Original TRAIN only; bounded windows per Run/Step/recipe, all observed sensors."""
    train_groups=set(runtime.prepared.split_runs['train'])
    forbidden=set(runtime.prepared.split_runs['validation'])|set(runtime.prepared.split_runs['test'])
    if train_groups&forbidden:raise ValueError('Normal TRAIN groups overlap held-out groups')
    grouped=defaultdict(list);available_times=defaultdict(set);available_groups=defaultdict(set)
    for step in steps:
        encoder,meta=runtime.encoders[step];patch=int(encoder.patch_len);cadence=float(meta['cadence_seconds'])
        for window in runtime.prepared.windows[step]['train']:
            if window['run_id'] not in train_groups or window.get('split','train')!='train' or window.get('injected',False):
                raise ValueError('Normal profile cannot fit Validation/Test/synthetic examples')
            names=list(window.get('sensor_columns',meta['sensor_columns']))
            if not names or any(n not in meta['sensor_columns'] for n in names):raise ValueError('Unknown normal sensor grid')
            values=np.asarray(window['current']);recipe=window['cohort'][0];offset=int(window['cohort'][-1])
            if values.shape!=(meta['window_length'],len(names)) or not np.isfinite(values).all():
                raise ValueError('Invalid original normal window')
            ticks=offset+np.arange(values.shape[0]//patch)*patch
            for tick in ticks:available_times[(step,str(recipe),int(tick))].add(window['run_id'])
            for name in names:available_groups[_key(step,recipe,name)].add(window['run_id'])
            grouped[(step,str(recipe),window['run_id'])].append(window)
    if not grouped:raise ValueError('No original TRAIN windows for normal profile')
    rng=np.random.default_rng(config.seed);selected={};covered=defaultdict(set)
    def include(step,run,window):
        pair=(step,run,window['pair_id'])
        if pair in selected:return
        selected[pair]=window
        patch=int(runtime.encoders[step][0].patch_len)
        for tick in int(window['cohort'][-1])+np.arange(len(window['current'])//patch)*patch:
            covered[(step,str(window['cohort'][0]),int(tick))].add(run)
    for group,pool in sorted(grouped.items()):
        step,_,run=group;pool=sorted(pool,key=lambda w:w['pair_id'])
        for index in rng.permutation(len(pool))[:config.normal_windows_per_run]:include(step,run,pool[index])
    # Add actual normal windows to cover elapsed patch positions from >=2 Runs.
    # This is not a claim that an interval between observed positions is learned.
    for group,pool in sorted(grouped.items()):
        step,recipe,run=group;patch=int(runtime.encoders[step][0].patch_len)
        for window in sorted(pool,key=lambda w:(w['cohort'][-1],w['pair_id'])):
            keys=[(step,recipe,int(t)) for t in int(window['cohort'][-1])+np.arange(len(window['current'])//patch)*patch]
            if any(len(available_times[k])>=MIN_RUN_GROUPS and len(covered[k])<MIN_RUN_GROUPS and run not in covered[k] for k in keys):
                include(step,run,window)
    sampled=[];sample_groups=defaultdict(set);groups_at_time=defaultdict(set)
    values_by_context=defaultdict(list);ticks_by_context=defaultdict(list);runs_by_context=defaultdict(list);bytes_used=0
    for (step,run,pair),window in sorted(selected.items()):
        encoder,meta=runtime.encoders[step];patch=int(encoder.patch_len)
        names=list(window.get('sensor_columns',meta['sensor_columns']))
        statistics=patch_stats(window['current'],patch);recipe=window['cohort'][0]
        ticks=int(window['cohort'][-1])+np.arange(statistics.shape[1])*patch
        sampled.append((step,run,pair))
        for sensor_index,name in enumerate(names):
            key=_key(step,recipe,name)
            values_by_context[key].append(statistics[sensor_index].copy())
            ticks_by_context[key].append(ticks.astype(np.int64));runs_by_context[key].append(run)
            sample_groups[key].add(run)
            for tick in ticks:groups_at_time[(key,int(tick))].add(run)
            bytes_used+=2*(statistics[sensor_index].nbytes+ticks.nbytes+8*len(ticks))
        if bytes_used+len(groups_at_time)*256>1024**3:
            raise MemoryError('Normal-profile preparation exceeds the 1 GiB feature/support budget')
    candidates=sorted(values_by_context);contexts=[];supported_offsets=[];targets=[];elapsed=[];ids=[];support=[];used_groups=set()
    for key in candidates:
        offsets=sorted({int(t) for chunk in ticks_by_context[key] for t in chunk
                        if len(groups_at_time[(key,int(t))])>=MIN_RUN_GROUPS})
        if not offsets:continue
        context_id=len(contexts);contexts.append(key);supported_offsets.append(offsets)
        local_values=[];local_ticks=[];local_groups=set()
        for values,ticks,run in zip(values_by_context[key],ticks_by_context[key],runs_by_context[key]):
            mask=np.isin(ticks,offsets)
            if mask.any():local_values.append(values[mask]);local_ticks.append(ticks[mask]);local_groups.add(run)
        values=np.concatenate(local_values);ticks=np.concatenate(local_ticks)
        times=(ticks*float(runtime.encoders[key[0]][1]['cadence_seconds'])).astype(np.float32)
        targets.append(values);elapsed.append(times);ids.append(np.full(len(times),context_id,np.int64));used_groups.update(local_groups)
        support.append(dict(step=key[0],recipe=key[1],sensor=key[2],patch_targets=len(times),
            available_train_groups=len(available_groups[key]),sampled_train_groups=len(local_groups),
            supported_elapsed_positions=len(offsets),minimum_groups_per_position=MIN_RUN_GROUPS))
    if not contexts:raise ValueError('Normal profile needs actual patch positions from at least two independent TRAIN Run groups')
    targets=np.concatenate(targets);elapsed=np.concatenate(elapsed);ids=np.concatenate(ids)
    transformed=np.arcsinh(targets.astype(np.float64));center=np.median(transformed,axis=0)
    q=np.quantile(transformed,[.25,.75],axis=0);scale=(q[1]-q[0])/1.349
    scale=np.where(scale>0,scale,transformed.std(0));scale=np.where(scale>0,scale,1.)
    scale=np.maximum(scale,np.finfo(np.float32).eps)
    normalized=((transformed-center)/scale).astype(np.float32)
    if not np.isfinite(normalized).all():raise ValueError('Nonfinite normal-only targets')
    data_digest=hashlib.sha256()
    for v in (ids,elapsed,normalized):data_digest.update(str((v.dtype,v.shape)).encode());data_digest.update(v.tobytes())
    spec=dict(recipe=RECIPE,quantiles=list(QUANTILES),width=config.normal_width,embedding_dim=16,
        contexts=[list(k) for k in contexts],target_center=center.astype(np.float32).tolist(),target_scale=scale.astype(np.float32).tolist(),
        supported_offsets=supported_offsets,minimum_groups_per_position=MIN_RUN_GROUPS,
        time_min=[offsets[0]*float(runtime.encoders[k[0]][1]['cadence_seconds']) for k,offsets in zip(contexts,supported_offsets)],
        time_max=[offsets[-1]*float(runtime.encoders[k[0]][1]['cadence_seconds']) for k,offsets in zip(contexts,supported_offsets)],
        patch_length={s:int(runtime.encoders[s][0].patch_len) for s in steps},
        cadence={s:float(runtime.encoders[s][1]['cadence_seconds']) for s in steps},
        train_groups=sorted(used_groups),
        fit_data_sha256=data_digest.hexdigest(),sampled_windows_sha256=sha(sampled),support=support)
    return spec,ids,elapsed,normalized


class NormalProfile(nn.Module):
    def __init__(self,spec):
        super().__init__();validate_spec(spec);self.spec=spec
        self.contexts={tuple(k):i for i,k in enumerate(spec['contexts'])}
        self.supported_offsets=[set(v) for v in spec['supported_offsets']]
        self.register_buffer('target_center',torch.tensor(spec['target_center'],dtype=torch.float32))
        self.register_buffer('target_scale',torch.tensor(spec['target_scale'],dtype=torch.float32))
        self.register_buffer('time_min',torch.tensor(spec['time_min'],dtype=torch.float32))
        self.register_buffer('time_max',torch.tensor(spec['time_max'],dtype=torch.float32))
        self.register_buffer('fit_updates',torch.tensor(0,dtype=torch.int64))
        width=int(spec['width']);emb=int(spec['embedding_dim'])
        self.embedding=nn.Embedding(len(self.contexts),emb)
        self.network=nn.Sequential(nn.Linear(emb+2,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,FEATURES*3))

    def validate_fitted(self):
        if int(self.fit_updates)<1:raise ValueError('Normal profile has no fitted state')
        for name in ('target_center','target_scale','time_min','time_max'):
            expected=torch.tensor(self.spec[name],dtype=torch.float32)
            if not torch.equal(getattr(self,name).detach().cpu(),expected):
                raise ValueError('Normal model state differs from saved scalar/time specification')

    def predict(self,ids,times):
        maximum=self.time_max[ids].clamp_min(1.)
        coordinates=torch.stack([times/maximum,torch.log1p(times)/torch.log1p(maximum)],-1)
        output=self.network(torch.cat([self.embedding(ids),coordinates],-1)).reshape(-1,FEATURES,3)
        median=output[...,0]
        return torch.stack([median-F.softplus(output[...,1]),median,median+F.softplus(output[...,2])],-1)

    def support(self,step,recipe,names,offset_samples,patches):
        problems=[]
        if step not in self.spec['patch_length']:return [dict(sensor=n,reason='untrained_normal_step') for n in names]
        patch=self.spec['patch_length'][step];cadence=self.spec['cadence'][step]
        for name in names:
            key=_key(step,recipe,name);index=self.contexts.get(key)
            if index is None:problems.append(dict(sensor=name,reason='untrained_normal_context'));continue
            ticks={int(offset_samples+i*patch) for i in range(patches)}
            if not ticks<=self.supported_offsets[index]:
                problems.append(dict(sensor=name,reason='elapsed_patch_position_not_fitted_from_two_TRAIN_Runs'))
        return problems

    @torch.no_grad()
    def evidence(self,statistics,step,recipe,names,offset_samples):
        if int(self.fit_updates)<1:raise ValueError('Normal profile is untrained; no random-profile inference')
        # statistics: selected sensors x patches x eight current-only statistics.
        if statistics.ndim!=3 or statistics.shape[0]!=len(names) or statistics.shape[-1]!=FEATURES:
            raise ValueError('Complete normal-profile statistic grid required')
        patches=statistics.shape[1]
        if self.support(step,recipe,names,offset_samples,patches):raise ValueError('Normal context/time unsupported; withhold the request')
        device=statistics.device;indices=[self.contexts[_key(step,recipe,n)] for n in names]
        ids=torch.tensor(indices,device=device).repeat_interleave(patches)
        times=(offset_samples+torch.arange(patches,device=device)*self.spec['patch_length'][step])*self.spec['cadence'][step]
        times=times.repeat(len(names)).float()
        prediction=self.predict(ids,times).reshape(len(names),patches,FEATURES,3).double()
        observed=(torch.asinh(statistics.double())-self.target_center.double())/self.target_scale.double()
        lower,median,upper=prediction.unbind(-1)
        width=(upper-lower).clamp_min(torch.finfo(torch.float32).eps)
        signed=torch.asinh((observed-median)/width)
        outside=torch.asinh((F.relu(lower-observed)+F.relu(observed-upper))/width)
        result=torch.cat([signed.mean(1),signed.abs().amax(1),outside.amax(1),torch.log1p(width).mean(1)],-1).float()
        if not torch.isfinite(result).all():raise ValueError('Nonfinite normal evidence; no classifier fallback')
        return result


def validate_spec(spec):
    contexts=spec.get('contexts',[])
    if spec.get('recipe')!=RECIPE or spec.get('quantiles')!=list(QUANTILES) or not contexts or len(contexts)!=len({tuple(k) for k in contexts}):
        raise ValueError('Unknown normal-profile schema/context inventory')
    if any(len(k)!=3 or not k[0] or not k[2] for k in contexts):raise ValueError('Invalid normal context key')
    for field in ('target_center','target_scale'):
        a=np.asarray(spec[field],float)
        if a.shape!=(FEATURES,) or not np.isfinite(a).all():raise ValueError('Invalid normal target normalization')
    if (np.asarray(spec['target_scale'])<=0).any():raise ValueError('Positive normal target scale required')
    lo=np.asarray(spec['time_min'],float);hi=np.asarray(spec['time_max'],float)
    if lo.shape!=(len(contexts),) or hi.shape!=lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo<0).any() or (hi<lo).any():
        raise ValueError('Invalid known elapsed-time support')
    if len(set(spec.get('train_groups',[])))<MIN_RUN_GROUPS or not spec.get('fit_data_sha256'):raise ValueError('Normal-only fitting provenance missing')
    positions=spec.get('supported_offsets',[])
    if spec.get('minimum_groups_per_position')!=MIN_RUN_GROUPS or len(positions)!=len(contexts):
        raise ValueError('Missing actual fitted patch-position support')
    for index,offsets in enumerate(positions):
        if not offsets or any(type(t) is not int or t<0 for t in offsets) or offsets!=sorted(set(offsets)):
            raise ValueError('Invalid fitted elapsed positions')
        cadence=spec['cadence'][contexts[index][0]]
        if not np.isclose(lo[index],offsets[0]*cadence) or not np.isclose(hi[index],offsets[-1]*cadence):
            raise ValueError('Normal time range differs from fitted positions')
    for step in {k[0] for k in contexts}:
        if int(spec['patch_length'].get(step,0))<2 or not np.isfinite(spec['cadence'].get(step,np.nan)) or spec['cadence'][step]<=0:
            raise ValueError('Invalid native patch timing')


def content_digest(record):
    checksum=hashlib.sha256(_json({k:v for k,v in record.items() if k not in ('state','content_sha256')}).encode())
    for name,value in sorted(record['state'].items()):
        checksum.update(name.encode());checksum.update(str((value.dtype,tuple(value.shape))).encode())
        checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def fit_or_load(root,runtime,steps,config,request,atomic_writer,*,allow_fit,progress=print):
    path=Path(root)/'.cache/current-normal-profile'/(sha(request)[:24]+'.pt')
    if path.is_file():
        before=digest(path);record=torch.load(path,map_location='cpu',weights_only=False)
        if sha(record.get('request'))!=sha(request) or record.get('content_sha256')!=content_digest(record):
            raise ValueError('Normal profile cache identity/content changed')
        profile=NormalProfile(record['spec']);profile.load_state_dict(record['state'],strict=True);profile.validate_fitted()
        if int(profile.fit_updates)<1 or record['training'].get('optimizer_updates')!=int(profile.fit_updates):raise ValueError('Missing normal-profile fitting evidence')
        if not set(profile.spec['train_groups'])<=set(request['train_groups']):raise ValueError('Normal cache contains non-TRAIN groups')
        if digest(path)!=before:raise ValueError('Normal-profile cache changed while loading')
        if progress:progress('Reused normal-only context profile; no normal refit',flush=True)
        return profile.requires_grad_(False).eval(),record['training']
    if not allow_fit:raise FileNotFoundError('Normal context profile missing; train STEP3, not evaluation')
    if progress:progress('Stage 1: learning normal context from original TRAIN only; no encoder call',flush=True)
    spec,ids,times,targets=prepare_data(runtime,steps,config)
    device=runtime.module.DEVICE;torch.manual_seed(config.seed)
    profile=NormalProfile(spec).to(device)
    optimizer=torch.optim.AdamW(profile.parameters(),lr=config.normal_learning_rate)
    rng=np.random.default_rng(config.seed);history=[];updates=0;started=time.perf_counter();stop='epochs_completed'
    tau=torch.tensor(QUANTILES,device=device).view(1,1,3)
    for epoch in range(config.normal_epochs):
        order=rng.permutation(len(ids));total=0.;count=0
        for start in range(0,len(ids),config.normal_batch_size):
            chosen=order[start:start+config.normal_batch_size]
            context=torch.as_tensor(ids[chosen],device=device);elapsed=torch.as_tensor(times[chosen],device=device)
            target=torch.as_tensor(targets[chosen],device=device)
            prediction=profile.predict(context,elapsed);error=target[...,None]-prediction
            loss=torch.maximum(tau*error,(tau-1)*error).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite normal-profile objective; no cache save')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(profile.parameters(),1.);optimizer.step();updates+=1
            total+=float(loss.detach())*len(chosen);count+=len(chosen)
        history.append(dict(epoch=epoch+1,normal_train_pinball_loss=total/count,seconds=time.perf_counter()-started))
        if progress:progress(f'Normal profile epoch {epoch+1}: TRAIN pinball loss={total/count:.5f}; {count} original patch targets',flush=True)
        # At least a complete epoch: never publish random/unvisited context embeddings.
        if time.perf_counter()-started>=config.normal_max_seconds:stop='wall_time_budget_after_complete_epoch';break
    if updates<1:raise ValueError('No fitted normal profile')
    profile.fit_updates.fill_(updates)
    training=dict(recipe=RECIPE,optimizer_updates=updates,history=history,stop_reason=stop,
        fit_seconds=time.perf_counter()-started,patch_targets=len(ids),sampled_train_groups=len(spec['train_groups']),
        objective='normal-only ordered pinball quantiles; no current target input; no Validation selection',
        fit_data_sha256=spec['fit_data_sha256'],support=spec['support'])
    if digest(__file__)!=request['code_sha256']:raise ValueError('Normal-profile code changed during fitting')
    record=dict(request=request,spec=spec,state={k:v.detach().cpu() for k,v in profile.state_dict().items()},training=training)
    record['content_sha256']=content_digest(record);atomic_writer(record,path)
    if progress:progress('Normal-only profile saved before encoder/head training',flush=True)
    return profile.requires_grad_(False).eval(),training


@torch.no_grad()
def describe_originals(profile,runtime,examples):
    """Held-out nominal coverage diagnostic only; no fitting/calibration/selection."""
    device=next(profile.parameters()).device;inside=np.zeros(FEATURES,np.int64)
    errors=np.zeros(FEATURES,float);widths=np.zeros(FEATURES,float);count=0
    for case in examples:
        if case['injected']:continue
        step=case['step'];meta=runtime.encoders[step][1]
        names=case.get('sensor_columns',meta['sensor_columns']);patch=profile.spec['patch_length'][step]
        values=patch_stats(case['current'],patch);patches=values.shape[1]
        if profile.support(step,case.get('recipe_key'),names,case['offset_samples'],patches):continue
        ids=torch.tensor([profile.contexts[_key(step,case.get('recipe_key'),n)] for n in names],device=device).repeat_interleave(patches)
        times=(case['offset_samples']+torch.arange(patches,device=device)*patch)*profile.spec['cadence'][step]
        predicted=profile.predict(ids,times.repeat(len(names)).float()).double()
        target=(torch.asinh(torch.as_tensor(values.reshape(-1,FEATURES),device=device).double())-profile.target_center.double())/profile.target_scale.double()
        lower,median,upper=predicted.unbind(-1)
        inside+=((target>=lower)&(target<=upper)).sum(0).cpu().numpy()
        errors+=(target-median).abs().sum(0).cpu().numpy();widths+=(upper-lower).sum(0).cpu().numpy();count+=len(target)
    labels=('mean','std','min','max','first','last','difference_mean','difference_std')
    return [dict(statistic=label,observed_patches=count,inside_05_95_fraction=float(inside[i]/count) if count else None,
        mean_absolute_median_error_scaled=float(errors[i]/count) if count else None,
        mean_interval_width_scaled=float(widths[i]/count) if count else None) for i,label in enumerate(labels)]
