"""Native PEALD Main/SubCycle units for current-only learning and incoming data.

Reads all data_complete CSVs locally. No source rewrite, interpolation, clipping,
external truth read or waveform-reference construction. Templates/scales/routes
are TRAIN-only. Complete-folder and original-normality are not fault certificates.
"""
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd

from . import peald_training as p

COLUMNS=dict(run='ActivePjobID',order='Timestamp',cohort='PMStoredProcessRecipeName',
    execution='PPExecname',parent_step='ProcessRecipeStepName',parent_sequence='ProcessRecipeStepSeqNo',
    step='PPExecStepName',substep='PPExecStepName',substep_sequence='PPExecStepSeqNo',
    substep_id='PPExecStepID',loop='SubRecipeLoopMoniterValue',step_context='PPExecname')
META=set(COLUMNS.values())|{'ProcessRecipeStepID','Unnamed: 0','ProcessRecipeEndRemainTime',
    'ProcessRecipeStepRemainTime','ProcessRecipeTotalTime','ProcessRecipeStepTime','SubRecipeLoopSettingValue','PMstatus'}


def text(value):
    return '' if pd.isna(value) else str(value).strip()


def token(value):
    value=text(value)
    try:
        number=float(value)
        if math.isfinite(number) and number.is_integer():return str(int(number))
    except ValueError:pass
    return value


def route_key(kind,execution,parent_sequence,parent_step):
    return kind+'|'+p._hash((execution,token(parent_sequence),parent_step))


def units(frame,*,file_id='incoming'):
    if not set(COLUMNS.values())<=set(frame):raise ValueError('PEALD native hierarchy metadata columns missing')
    values={key:frame[name].map(token if key in ('parent_sequence','substep_sequence','loop') else text) for key,name in COLUMNS.items()}
    stamps=pd.to_datetime(frame[COLUMNS['order']],format='mixed',errors='coerce')
    sub=values['execution'].str.contains('SubRecipe',case=False,regex=False)
    main=values['execution'].str.contains('MainRecipe',case=False,regex=False)
    kinds=np.where(sub,'subcycle',np.where(main,'main','unsupported'))
    parent=list(zip(kinds,values['execution'],values['parent_sequence'],values['parent_step']))
    keys=[(values['run'].iloc[i],values['cohort'].iloc[i],parent[i],values['loop'].iloc[i] if sub.iloc[i] else '') for i in range(len(frame))]
    blocks=np.cumsum([True]+[keys[i]!=keys[i-1] or frame.index[i]!=frame.index[i-1]+1 for i in range(1,len(keys))]) if len(frame) else []
    run_blocks=(values['run'].ne(values['run'].shift())|pd.Series(frame.index,index=frame.index).diff().ne(1)).cumsum()
    out=[]
    for _,index in pd.Series(np.arange(len(frame))).groupby(blocks,sort=False):
        ids=index.to_numpy();first=int(ids[0]);kind,execution,parent_seq,parent_name=parent[first]
        current=frame.iloc[ids]
        substeps=[];last=None
        for j in ids:
            pair=(values['substep_sequence'].iloc[j],values['substep'].iloc[j])
            if pair!=last:substeps.append(pair);last=pair
        times=stamps.iloc[ids]
        out.append(dict(frame=current,times=times,rows=current.index.to_numpy(),kind=kind,
            raw_run=values['run'].iloc[first],recipe=values['cohort'].iloc[first],execution=execution,
            parent_sequence=parent_seq,parent_step=parent_name,loop=values['loop'].iloc[first] if kind=='subcycle' else '',
            substeps=tuple(substeps),route=route_key(kind,execution,parent_seq,parent_name),
            record=(file_id,int(run_blocks.iloc[first])),file_id=file_id,
            metadata_complete=bool(kind!='unsupported' and values['run'].iloc[first] and values['cohort'].iloc[first] and parent_name and parent_seq and
                (kind!='subcycle' or (values['loop'].iloc[first] and all(values[k].iloc[ids].ne('').all() for k in ('substep','substep_sequence','substep_id')))))))
    return out


def timestamps_ok(unit,cadence,tolerance):
    times=unit['times']
    if times.isna().any():return False
    delta=times.diff().dt.total_seconds().iloc[1:].to_numpy()
    return bool(np.all(np.isfinite(delta)&(delta>0)&(np.abs(delta-cadence)<=tolerance)))


def starts(length,width):
    if length<width:return []
    values=list(range(0,length-width+1,width))
    if values[-1]!=length-width:values.append(length-width)
    return values


def _record_groups(collection):
    groups=defaultdict(list)
    for unit in collection:groups[unit['record']].append(unit)
    return [sorted(v,key=lambda u:int(u['rows'][0])) for v in groups.values()]


def delivery_metadata_identity(package):
    import hashlib,json
    package=Path(package);path=package/'inputs/test.csv';before=p.digest(path)
    frame=pd.read_csv(path,usecols=[COLUMNS['run'],COLUMNS['order']],dtype='string')
    text=frame.fillna('').to_csv(index=False)
    bounds=package/'inputs/run_boundaries.csv';boundary_digest=p.digest(bounds) if bounds.is_file() else None
    identity=dict(run_time_sha256=hashlib.sha256(text.encode()).hexdigest(),boundaries_sha256=boundary_digest)
    if p.digest(path)!=before:raise ValueError('Delivery inputs changed while reserving source membership')
    return frame,identity


def _recipe_ok(recipe,suffix):return recipe==suffix or recipe.endswith('_'+suffix)


def prepare(root,*,config,patch_length,max_window_length=64,delivery_package,sensor_columns,progress=print,max_memory_mb=4096):
    root=Path(root);directory=root/'data/mt_nit/data_complete'
    paths=sorted(directory.glob('*.csv'))
    if not paths:raise FileNotFoundError('data/mt_nit/data_complete contains no CSV files')
    source_files={path.relative_to(root).as_posix():p.digest(path) for path in paths}
    whitelist=set(sensor_columns)-META
    if not whitelist:raise ValueError('Explicit physical measurement sensor whitelist required')
    all_units=[];estimated_bytes=0
    for i,path in enumerate(paths):
        header=pd.read_csv(path,nrows=0).columns.tolist()
        if not set(COLUMNS.values())<=set(header):raise ValueError('Native hierarchy columns missing from a data_complete file')
        meta=pd.read_csv(path,usecols=list(set(COLUMNS.values())),dtype='string')
        recipe=meta[COLUMNS['cohort']].fillna('').str.strip()
        times=pd.to_datetime(meta[COLUMNS['order']],format='mixed',errors='coerce')
        eligible=(recipe.eq(config.recipe_suffix)|recipe.str.endswith('_'+config.recipe_suffix))&~times.dt.month.isin(config.excluded_months)
        if not eligible.any():continue
        selected=list(set(COLUMNS.values()))+[name for name in header if name in whitelist]
        frame=pd.read_csv(path,usecols=selected,dtype={name:'string' for name in set(COLUMNS.values())},low_memory=False)
        for name in frame.columns:
            if name in whitelist and pd.api.types.is_numeric_dtype(frame[name]):frame[name]=frame[name].astype(np.float32)
        # Retain whole file units here; per-Run recipe/month policy below avoids joining fragments.
        estimated_bytes+=2*int(frame.memory_usage(deep=True).sum())
        if estimated_bytes>max_memory_mb*1024**2:
            raise MemoryError('Native preparation memory budget reached before fitting. Increase max_memory_mb only with sufficient local RAM.')
        all_units.extend(units(frame,file_id=source_files[path.relative_to(root).as_posix()]))
        if progress and (i==0 or (i+1)%10==0 or i+1==len(paths)):
            progress(f'Native data_complete preparation: {i+1}/{len(paths)} files; no fitting yet',flush=True)
    occurrences=defaultdict(set)
    for unit in all_units:occurrences[(unit['file_id'],unit['raw_run'])].add(unit['record'])
    ambiguous={key for key,value in occurrences.items() if len(value)>1}
    by_record=defaultdict(list)
    for unit in all_units:
        if (unit['file_id'],unit['raw_run']) not in ambiguous:by_record[unit['record']].append(unit)
    records=[]
    for record,group in by_record.items():
        times=pd.concat([u['times'] for u in group]).dropna()
        if times.empty or not group[0]['raw_run']:continue
        if any(not _recipe_ok(u['recipe'],config.recipe_suffix) for u in group):continue
        if any(int(month) in config.excluded_months for month in times.dt.month.unique()):continue
        if any(u['kind']=='main' and u['times'].isna().any() for u in group):continue
        records.append(dict(record=record,raw_run=group[0]['raw_run'],start=times.min(),end=times.max(),units=group))
    # Same ID + overlapping source intervals are conservative aliases, never separate splits.
    by_run=defaultdict(list)
    for record in records:by_run[record['raw_run']].append(record)
    groups=[]
    for raw_run,items in by_run.items():
        cluster=[];end=None
        for item in sorted(items,key=lambda v:v['start']):
            if end is not None and item['start']>end:
                groups.append((raw_run,cluster));cluster=[]
            cluster.append(item);end=max(end,item['end']) if cluster[:-1] else item['end']
        if cluster:groups.append((raw_run,cluster))
    # Only input identities/timestamps reserve the delivered originals. No evaluator access.
    delivery,delivery_identity=delivery_metadata_identity(delivery_package)
    delivered_times=pd.to_datetime(delivery[COLUMNS['order']],format='mixed',errors='coerce')
    reserved=defaultdict(list)
    boundary_path=Path(delivery_package)/'inputs/run_boundaries.csv'
    if boundary_path.is_file():
        from .ald_delivery_evaluation import delivery_boundaries
        ranges=delivery_boundaries(pd.read_csv(boundary_path,dtype={'run_id':str}),len(delivery))
        parts=[delivery.iloc[v['start']:v['end']] for v in ranges]
    else:parts=[delivery.loc[index] for index in delivery.groupby(COLUMNS['run'],sort=False).groups.values()]
    for part in parts:
        ids=part[COLUMNS['run']].astype('string').str.strip()
        if ids.isna().any() or ids.eq('').any() or ids.nunique()!=1:raise ValueError('Delivery source Run identity is ambiguous')
        raw_run=ids.iloc[0];times=delivered_times.loc[part.index].dropna()
        reserved[text(raw_run)].append((times.min(),times.max()) if len(times) else (None,None))
    grouped=[];reserved_groups=[]
    for raw_run,cluster in groups:
        start=min(x['start'] for x in cluster);end=max(x['end'] for x in cluster)
        identity=p._hash((raw_run,start.isoformat(),end.isoformat()))
        if any(a is None or not (end<a or start>b) for a,b in reserved.get(raw_run,[])):
            reserved_groups.append(identity);continue
        group=[u for item in cluster for u in item['units']]
        for u in group:u['group']=identity
        grouped.append(dict(group=identity,start=start,end=end,units=group))
    grouped.sort(key=lambda x:(x['start'],x['group']));a=int(.7*len(grouped));b=a+int(.15*len(grouped))
    partitions=dict(train=grouped[:a],validation=grouped[a:b],test=grouped[b:])
    if min(map(len,partitions.values()))<2:raise ValueError('Need two original Run groups per split after native policy/delivery exclusion')
    vstart=min(g['start'] for g in partitions['validation']);tstart=min(g['start'] for g in partitions['test'])
    partitions['train']=[g for g in partitions['train'] if g['end']<vstart]
    partitions['validation']=[g for g in partitions['validation'] if g['end']<tstart]
    if min(map(len,partitions.values()))<2:raise ValueError('Chronological boundary purging leaves insufficient Run support')
    dataset={split:[u for g in values for u in g['units']] for split,values in partitions.items()}
    intervals=[]
    for unit in dataset['train']:
        dt=unit['times'].diff().dt.total_seconds().dropna().to_numpy();intervals.extend(dt[dt>0])
    if not intervals:raise ValueError('No finite native TRAIN cadence')
    cadence=float(np.median(intervals));tolerance=min(.5*cadence,max(.05*cadence,8*float(np.median(np.abs(np.asarray(intervals)-cadence)))))
    if cadence+tolerance>config.max_interval_seconds:tolerance=max(0.,config.max_interval_seconds-cadence)
    if tolerance<=0:raise ValueError('TRAIN cadence exceeds the configured native timestamp-gap limit')
    # A bad Main timestamp/interval excludes its whole original Run group. A bad SubCycle excludes that unit only.
    invalid_main={u['group'] for split in dataset for u in dataset[split] if u['kind']=='main' and not timestamps_ok(u,cadence,tolerance)}
    for collection in dataset.values():
        for group in _record_groups(collection):
            for previous,next_unit in zip(group,group[1:]):
                if previous['kind']==next_unit['kind']=='main' and previous['rows'][-1]+1==next_unit['rows'][0]:
                    delta=(next_unit['times'].iloc[0]-previous['times'].iloc[-1]).total_seconds()
                    if not math.isfinite(delta) or delta<=0 or abs(delta-cadence)>tolerance:invalid_main.add(previous['group'])
    dataset={split:[u for u in values if u['group'] not in invalid_main] for split,values in dataset.items()}
    route_groups=defaultdict(list)
    for u in dataset['train']:
        if u['metadata_complete'] and timestamps_ok(u,cadence,tolerance):route_groups[u['route']].append(u)
    windows={};metadata={};support=[];sensor_audit=[];templates=[];quality=[]
    for route,candidates in route_groups.items():
        sample=candidates[0];template=None
        if sample['kind']=='subcycle':
            counts=Counter(u['substeps'] for u in candidates)
            template=max(counts,key=lambda seq:(counts[seq],len(seq),seq))
            if len({u['group'] for u in candidates if u['substeps']==template})<2:
                support.append(dict(step=route,status='UNSUPPORTED',reason='subcycle_template_needs_two_TRAIN_Run_groups'));continue
            candidates=[u for u in candidates if u['substeps']==template]
            templates.append(dict(step=route,source='TRAIN_most_common_observed_order_not_independent_completeness_certificate',substeps=template))
        available=set().union(*(set(u['frame'].columns)&whitelist for u in candidates))
        names=[];means=[];scales=[]
        for name in sorted(available):
            if name.startswith('_'):continue
            parts=[];groups=set();numeric=True
            for u in candidates:
                if name not in u['frame']:continue
                raw=u['frame'][name];values=pd.to_numeric(raw,errors='coerce')
                if (raw.notna()&values.isna()).any():numeric=False;break
                finite=values.to_numpy(dtype=float);finite=finite[np.isfinite(finite)]
                if len(finite):parts.append(finite);groups.add(u['group'])
            if not numeric or not parts or len(groups)<2:continue
            values=np.concatenate(parts)
            if np.ptp(values)<=0:
                sensor_audit.append(dict(step=route,sensor=name,status='constant_TRAIN_channel'));continue
            q=np.quantile(values,[.25,.75]);scale=float((q[1]-q[0])/1.349)
            if scale<=1e-6:scale=float(values.std())
            if not math.isfinite(scale) or scale<=0:continue
            names.append(name);means.append(float(np.median(values)));scales.append(scale)
        if not names:
            support.append(dict(step=route,status='UNSUPPORTED',reason='no_common_numeric_variable_TRAIN_sensors'));continue
        long=[len(u['frame']) for u in candidates if len(u['frame'])>=2*patch_length]
        width=min(max_window_length,int(np.quantile(long,.25)))//patch_length*patch_length if long else 0
        if width<2*patch_length:
            support.append(dict(step=route,status='UNSUPPORTED',reason='native_unit_shorter_than_two_encoder_patches'));continue
        mean=np.asarray(means,np.float32);std=np.asarray(scales,np.float32)
        meta=dict(sensor_columns=names,window_length=width,mean=mean.astype(np.float32),std=std.astype(np.float32),
            cadence_seconds=cadence,tolerance_seconds=tolerance,allowed_recipe_keys=sorted({p._hash(u['recipe']) for u in candidates}),
            native_kind=sample['kind'],execution=sample['execution'],parent_sequence=sample['parent_sequence'],parent_step=sample['parent_step'],
            expected_substeps=template,step_descriptor='ALD '+sample['parent_step']+' '+sample['kind'],sensor_descriptors={s:s for s in names})
        pools={split:[] for split in dataset}
        for split,collection in dataset.items():
            seen=set()
            for u in collection:
                if u['route']!=route:continue
                if p._hash(u['recipe']) not in meta['allowed_recipe_keys']:
                    quality.append(dict(step=route,split=split,reason='recipe_absent_from_TRAIN'));continue
                if not valid_unit(u,meta):
                    quality.append(dict(step=route,split=split,reason='native_timestamp_substep_or_sensor_failure'));continue
                for start in starts(len(u['frame']),width):
                    window=dict(u,frame=u['frame'].iloc[start:start+width])
                    observed=observed_sensors(window,meta)
                    if not observed:continue
                    values=window['frame'][observed].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
                    sensor_ids=[names.index(v) for v in observed]
                    times=u['times'].iloc[start:start+width];positions=u['rows'][start:start+width]
                    key=(u['group'],times.iloc[0].isoformat(),times.iloc[-1].isoformat(),route)
                    if key in seen:continue
                    seen.add(key)
                    tick=int(round((times.iloc[0]-u['times'].iloc[0]).total_seconds()/cadence))
                    current=((values-meta['mean'][sensor_ids])/meta['std'][sensor_ids]).astype(np.float32)
                    pools[split].append(dict(current=current,sensor_columns=observed,run_id=u['group'],lineage_id=p._hash(u['raw_run']),pair_id=p._hash(key),step=route,split=split,
                        cohort=(p._hash(u['recipe']),token(u['loop']),start,tick),position=0.,input_row_start=int(positions[0]),input_row_end=int(positions[-1])))
        if not pools['train'] or not pools['validation']:
            support.append(dict(step=route,status='UNSUPPORTED',reason='no_native_TRAIN_or_Validation_windows'));continue
        windows[route]=pools;metadata[route]=meta
        support.append(dict(step=route,status='READY',native_kind=sample['kind'],parent_step=sample['parent_step'],
            sensors=len(names),window_length=width,**{k+'_windows':len(v) for k,v in pools.items()}))
    for path in paths:
        if p.digest(path)!=source_files[path.relative_to(root).as_posix()]:raise ValueError('Native source file changed during preparation')
    split_runs={k:sorted({u['group'] for u in values}) for k,values in dataset.items()}
    if delivery_metadata_identity(delivery_package)[1]!=delivery_identity:raise ValueError('Delivery source reservation metadata changed during native preparation')
    identity=dict(source_directory='data/mt_nit/data_complete',source_files=source_files,split_runs=split_runs,
        delivery_metadata_identity=delivery_identity,
        reserved_delivery_groups=sorted(reserved_groups),split_policy='chronological_Run_start_with_boundary_purge',
        native_preparation_sha256=p.digest(__file__),native_policy=dict(recipe_suffix=config.recipe_suffix,excluded_months=config.excluded_months),
        patch_length=patch_length)
    tables=dict(support=pd.DataFrame(support),sensors=pd.DataFrame(sensor_audit),templates=pd.DataFrame(templates),
        native_quality=pd.DataFrame(quality),cadence=pd.DataFrame([dict(step=s,cadence_seconds=cadence,tolerance_seconds=tolerance) for s in windows]),
        current_input_policy=pd.DataFrame([dict(source='data_complete_all_files',recipe_suffix=config.recipe_suffix,
            split_policy='chronological_Run_with_boundary_purge',delivery_excluded_before_scaling=True,
            normality_is_dataset_premise=True,main_timestamp_failure='exclude_Run',subcycle_failure='exclude_SubCycle')]))
    prepared=p.Prepared(windows,metadata,tables,split_runs,identity)
    prepared.process=SimpleNamespace(source_columns=COLUMNS)
    return prepared


def observed_sensors(unit,meta):
    result=[]
    for name in meta['sensor_columns']:
        if name in unit['frame']:
            values=pd.to_numeric(unit['frame'][name],errors='coerce').to_numpy(dtype=float)
            if np.isfinite(values).all():result.append(name)
    return result


def valid_unit(unit,meta):
    if not unit['metadata_complete'] or not timestamps_ok(unit,meta['cadence_seconds'],meta['tolerance_seconds']):return False
    if meta.get('expected_substeps') is not None and unit['substeps']!=tuple(tuple(v) for v in meta['expected_substeps']):return False
    return True


def incoming_windows(frame,runtime):
    observed_units=units(frame);rejected=[];cases=[]
    bad_main_runs=set()
    for u in observed_units:
        if u['kind']!='main':continue
        meta=runtime.encoders.get(u['route'],(None,None))[1]
        if u['times'].isna().any() or (meta is not None and not timestamps_ok(u,meta['cadence_seconds'],meta['tolerance_seconds'])):
            bad_main_runs.add(u['raw_run'])
    for u in observed_units:
        route=u['route'];meta=runtime.encoders.get(route,(None,None))[1]
        reason=None
        if u['raw_run'] in bad_main_runs:reason='Main_timestamp_failure_excludes_Run'
        elif meta is None:reason='unsupported_native_context'
        elif p._hash(u['recipe']) not in meta['allowed_recipe_keys']:reason='unsupported_recipe'
        elif not valid_unit(u,meta):reason='invalid_or_incomplete_native_unit'
        if reason:
            rejected.append(dict(step=route,source_row_start=int(u['rows'][0]),source_row_end=int(u['rows'][-1])+1,rows=len(u['rows']),reason=reason));continue
        width=meta['window_length']
        for start in starts(len(u['frame']),width):
            window=dict(u,frame=u['frame'].iloc[start:start+width]);names=observed_sensors(window,meta)
            if not names:continue
            values=window['frame'][names].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
            positions=u['rows'][start:start+width]
            tick=int(round((u['times'].iloc[start]-u['times'].iloc[0]).total_seconds()/meta['cadence_seconds']))
            cases.append(dict(step=route,current=values,centers=names,sensor_columns=names,
                recipe_key=p._hash(u['recipe']),offset_samples=tick,run_id=p._hash(u['raw_run']),
                row_start=int(positions[0]),row_end=int(positions[-1])+1))
        if len(u['frame'])<width:rejected.append(dict(step=route,source_row_start=int(u['rows'][0]),source_row_end=int(u['rows'][-1])+1,rows=len(u['frame']),reason='native_unit_too_short_for_trained_window'))
    cases.sort(key=lambda c:(c['row_start'],c['row_end'],c['step']))
    return cases,rejected


class Receiver:
    """Raw native rows -> current-only replies; no reference lookup.

    Main complete windows emit immediately. SubCycles emit after the cycle is
    complete and its substep order is validated. Boundary tail windows are
    emitted only at that observed boundary, never before it is known. Main
    timestamp failure invalidates the Run; early Main results are provisional.
    """
    def __init__(self,session,*,on_result=None,on_quality=None,max_cycle_rows=4096):
        from collections import deque
        self.session=session;self.on_result=on_result;self.on_quality=on_quality;self.max_cycle_rows=max_cycle_rows
        self._audit=deque(maxlen=256);self._rows=[];self._key=None;self._unit_start=None;self._previous=None
        self._count=0;self._last_emitted=0;self._discard_unit=False;self._seen=0;self._raw_run=None;self.invalid_run=False;self._last_main=None
        first_meta=next(iter(session.metadata.values()))
        self._main_cadence=float(first_meta['cadence_seconds']);self._main_tolerance=float(first_meta['tolerance_seconds'])

    def _quality(self,step,start,end,reason):
        item=dict(step=step,source_run=p._hash(self._raw_run) if self._raw_run else None,source_row_start=start,source_row_end=end,rows=max(0,end-start),reason=reason)
        self._audit.append(item)
        if self.on_quality:self.on_quality(item)

    def drain_audit(self):
        values=list(self._audit);self._audit.clear();return values

    def _emit(self,rows):
        if not rows or self.invalid_run:return None
        step=self._key[2];meta=self.session.metadata[step]
        frame=pd.DataFrame(rows).set_index('_row_index');unit=dict(frame=frame)
        names=observed_sensors(unit,meta)
        if not names:
            self._quality(step,int(frame.index[0]),int(frame.index[-1])+1,'no_finite_supported_sensor_window');return None
        values=frame[names].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
        timestamp=float(rows[0]['_seconds']);offset=max(0,int(round((timestamp-self._unit_start)/meta['cadence_seconds'])))
        extra = dict(loop_key=self._key[3], lineage_id=p._hash(self._raw_run)) if getattr(self.session,'supports_reference_context',False) else {}
        reply=self.session.process_window(current=values,step=step,run_id=p._hash(self._raw_run),
            recipe_key=p._hash(self._key[1]),sensor_columns=names,offset_samples=offset,**extra)
        reply.update(row_start=int(frame.index[0]),row_end=int(frame.index[-1])+1,run_id=p._hash(self._raw_run),
            native_kind=meta['native_kind'],parent_step=meta['parent_step'],quality_provisional=True)
        if self.on_result:self.on_result(reply)
        return reply

    def _finish_unit(self):
        replies=[]
        if self._key is None or not self._rows or self._discard_unit:return replies
        step=self._key[2];meta=self.session.metadata.get(step)
        if meta is not None and not self.invalid_run:
            if meta['native_kind']=='subcycle':
                frame=pd.DataFrame(self._rows).set_index('_row_index');group=units(frame)
                if len(group)!=1 or not valid_unit(group[0],meta):
                    self._quality(step,int(frame.index[0]),int(frame.index[-1])+1,'incomplete_or_invalid_SubCycle')
                else:
                    offsets=starts(len(self._rows),meta['window_length'])
                    for start in offsets:
                        result=self._emit(self._rows[start:start+meta['window_length']])
                        if result is not None:replies.append(result)
                    if not offsets:self._quality(step,int(frame.index[0]),int(frame.index[-1])+1,'native_unit_too_short')
            elif self._count>=meta['window_length'] and self._last_emitted!=self._count:
                result=self._emit(self._rows[-meta['window_length']:])
                if result is not None:replies.append(result)
            elif self._count<meta['window_length']:
                self._quality(step,self._rows[0]['_row_index'],self._rows[-1]['_row_index']+1,'native_unit_too_short')
        self._rows=[];return replies

    def push(self,record,*,row_index=None):
        row=dict(record);index=self._seen if row_index is None else int(row_index);self._seen+=1
        raw_run=text(row.get(COLUMNS['run']));recipe=text(row.get(COLUMNS['cohort']))
        execution=text(row.get(COLUMNS['execution']));parent=text(row.get(COLUMNS['parent_step']))
        kind='subcycle' if 'subrecipe' in execution.lower() else ('main' if 'mainrecipe' in execution.lower() else 'unsupported')
        step=route_key(kind,execution,row.get(COLUMNS['parent_sequence']),parent)
        loop=token(row.get(COLUMNS['loop'])) if kind=='subcycle' else ''
        key=(raw_run,recipe,step,loop);replies=[]
        if not raw_run or not recipe:
            self.invalid_run=True;self._rows=[];self._discard_unit=True
            self._quality(step,index,index+1,'missing_native_Run_identity');return replies
        if key!=self._key:
            replies.extend(self._finish_unit())
            if raw_run!=self._raw_run:
                self.invalid_run=False;self._last_main=None
            self._raw_run=raw_run;self._key=key;self._rows=[];self._unit_start=None;self._previous=None;self._count=0;self._last_emitted=0;self._discard_unit=False
        meta=self.session.metadata.get(step)
        if kind=='main':
            raw_stamp=row.get('_seconds')
            if raw_stamp is None:
                parsed=pd.to_datetime(row.get(COLUMNS['order']),format='mixed',errors='coerce')
                raw_stamp=parsed.value/1e9 if not pd.isna(parsed) else float('nan')
            raw_stamp=float(raw_stamp)
            invalid_main=not math.isfinite(raw_stamp)
            if self._last_main is not None and self._last_main[0]+1==index:
                delta=raw_stamp-self._last_main[1]
                invalid_main=invalid_main or not (math.isfinite(delta) and delta>0 and abs(delta-self._main_cadence)<=self._main_tolerance)
            self._last_main=(index,raw_stamp)
            if invalid_main:
                self.invalid_run=True;self._quality(step,index,index+1,'Main_timestamp_failure_invalidates_Run');return replies
        if not raw_run or not recipe or meta is None or p._hash(recipe) not in meta.get('allowed_recipe_keys',[]):
            self._quality(step,index,index+1,'unsupported_or_missing_native_context');return replies
        timestamp=row.get('_seconds')
        if timestamp is None:
            stamp=pd.to_datetime(row.get(COLUMNS['order']),format='mixed',errors='coerce')
            timestamp=stamp.value/1e9 if not pd.isna(stamp) else float('nan')
        timestamp=float(timestamp);row['_seconds']=timestamp;row['_row_index']=index
        bad=not math.isfinite(timestamp)
        if self._previous is not None:
            delta=timestamp-self._previous
            bad=bad or not (math.isfinite(delta) and delta>0 and abs(delta-meta['cadence_seconds'])<=meta['tolerance_seconds'])
        if bad:
            if kind=='main':self.invalid_run=True
            self._quality(step,index,index+1,'Main_timestamp_failure_invalidates_Run' if kind=='main' else 'invalid_SubCycle_timestamp')
            # Retain invalid SubCycle metadata until its boundary; it cannot be emitted.
            if kind=='subcycle':self._rows=[];self._discard_unit=True
            return replies
        if self.invalid_run or self._discard_unit:return replies
        if self._unit_start is None:self._unit_start=timestamp
        self._previous=timestamp;self._count+=1;self._rows.append(row)
        width=meta['window_length']
        if kind=='main':
            if len(self._rows)>width:self._rows=self._rows[-width:]
            if self._count%width==0:
                result=self._emit(self._rows[-width:]);self._last_emitted=self._count
                if result is not None:replies.append(result)
        elif len(self._rows)>self.max_cycle_rows:
            self._quality(step,self._rows[0]['_row_index'],index+1,'SubCycle_buffer_capacity_exceeded')
            self._rows=[];self._previous=None
            # Do not restart a partial cycle as complete; mark until context changes.
            self._discard_unit=True
        return replies

    def finish(self):
        replies=self._finish_unit();self._key=None;self._rows=[]
        return replies
