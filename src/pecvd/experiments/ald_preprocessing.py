"""Versioned ALD experiment preprocessing; raw source files stay immutable.

The native chronological splits and hierarchy below are kept
in this experiment module so existing content-bound CVD/ALD packages still load.
"""
from collections import Counter, defaultdict
from copy import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd

from . import peald_training as p
from . import ald_input_store as input_store
from .current_input import Config as _LegacyNativeConfig
from .peald_current_data import (COLUMNS, META, units, token, timestamps_ok,
    starts, _record_groups, _recipe_ok, valid_unit,
    observed_sensors)

LINEAGE_POLICY = 'source_Run_group_with_native_time_interval_v1'


@dataclass(frozen=True)
class NativeConfig(_LegacyNativeConfig):
    """data_complete defaults: all months and recipes, with native quality checks."""
    excluded_months: tuple = ()
    recipe_suffix: str = ''

    def validate(self):
        if not isinstance(self.recipe_suffix, str):
            raise ValueError('Recipe suffix must be a string; empty means all recipes')
        _LegacyNativeConfig.validate(replace(self, recipe_suffix=self.recipe_suffix or 'all_recipes'))


@dataclass(frozen=True)
class PreprocessingConfig:
    schema_columns: int | None = None
    max_window_length: int = 64
    encode_valves: bool = True
    drop_columns: tuple = ()
    context_columns: tuple = ()

    def validate(self):
        if self.schema_columns is not None and (type(self.schema_columns) is not int or self.schema_columns < 1):
            raise ValueError('CSV column count must be None (all files) or a positive integer')
        if type(self.max_window_length) is not int or self.max_window_length < 2:
            raise ValueError('Native window length must be at least two')
        if type(self.encode_valves) is not bool:
            raise ValueError('encode_valves must be Boolean')
        if set(self.drop_columns) & set(COLUMNS.values()):
            raise ValueError('Native hierarchy metadata cannot be dropped')


def schema_inventory(directory, policy=PreprocessingConfig()):
    """Use common column names across every file by default, without imputation."""
    policy.validate()
    headers = {p: tuple(pd.read_csv(p, nrows=0).columns) for p in sorted(Path(directory).glob('*.csv'))}
    if not headers:
        raise FileNotFoundError('data_complete contains no CSV files')
    if policy.schema_columns is None:
        common = tuple(sorted(set.intersection(*(set(h) for h in headers.values()))))
        if not common:
            raise ValueError('ALD files have no common columns')
        rows = [dict(file=p.name, columns=len(h), included=True, used_columns=len(common),
                     reason='accepted_common_columns') for p, h in headers.items()]
        return common, pd.DataFrame(rows)
    candidates = Counter(tuple(sorted(h)) for h in headers.values() if len(h) == policy.schema_columns)
    if not candidates:
        raise ValueError(f'No {policy.schema_columns}-column ALD CSV schema found')
    canonical = min(candidates, key=lambda h: (-candidates[h], h))
    rows = [dict(file=p.name, columns=len(h), included=tuple(sorted(h)) == canonical,
        reason='accepted_schema' if tuple(sorted(h)) == canonical else 'schema_mismatch') for p, h in headers.items()]
    return canonical, pd.DataFrame(rows)


def sensor_candidates(header, policy):
    excluded = META | {'ActiveCjobID', 'ActivePjobID'} | set(policy.drop_columns)
    return [c for c in header if c not in excluded and not c.startswith(('Unnamed:', '_', 'ASCII('))
            and (policy.encode_valves or not c.startswith('ValveAct_'))]


def is_context_sensor(name, policy=PreprocessingConfig()):
    return name.startswith('ValveAct_') or name in policy.context_columns


def sensor_roles(prepared):
    """Separate observed input context from sensors that receive questions."""
    return pd.DataFrame([dict(step=step, sensor=name,
        role='context_only' if name in meta.get('context_sensor_columns', ()) or name.startswith('ValveAct_') else 'prediction_target')
        for step, meta in prepared.metadata.items() for name in meta['sensor_columns']])


def canonicalize_run_lineages(prepared, *, progress=None):
    """Repair old native window IDs without copying/recomputing sensor arrays.

    A raw equipment Job ID can be reused months later. run_id already identifies
    the original Run interval and groups overlapping aliases together. The same
    identity must be used for reference/fit/heldout independence checks.
    """
    if prepared.identity.get('source_directory') != 'data/mt_nit/data_complete':
        return prepared
    if 'delivery_metadata_identity' in prepared.identity or prepared.identity.get('reserved_delivery_groups'):
        raise ValueError('Prepare from data_complete only before repairing Run identities')
    partitions = {s: set(prepared.split_runs[s]) for s in ('train', 'validation', 'test')}
    if any(partitions[a] & partitions[b] for a, b in
           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Canonical source Run appears in more than one partition')
    result = copy(prepared); result.windows = {}; changed = 0; counts = []
    for step, pools in prepared.windows.items():
        result.windows[step] = {}
        for split, windows in pools.items():
            repaired = []
            for window in windows:
                run_id = window['run_id']
                if run_id not in partitions[split] or window.get('split', split) != split:
                    raise ValueError('Native window does not belong to its declared source Run partition')
                changed += window.get('lineage_id') != run_id
                repaired.append(dict(window, lineage_id=run_id))
            result.windows[step][split] = repaired
            counts.append(dict(step=step, split=split, windows=len(repaired),
                               independent_runs=len({w['run_id'] for w in repaired})))
    result.identity = dict(prepared.identity, lineage_policy=LINEAGE_POLICY)
    result.tables = dict(prepared.tables, run_lineages=pd.DataFrame(counts))
    if changed and progress:
        progress(f'Fixed canonical source Run IDs on {changed} windows; existing normalized arrays reused.', flush=True)
    return result


def encode_frame(frame, sensors, *, strict=True):
    """Same deterministic scalar encoding for training, CSV and live rows.

    Arbitrary text never reaches numpy/torch. Unknown states/missing values are
    invalid observations, rather than being filled with a normal valve state.
    """
    result = frame.copy()
    for name in sensors:
        if name not in result:
            if strict:
                raise ValueError(f'Missing ALD sensor column: {name}')
            continue
        raw = result[name]
        if name.startswith('ValveAct_'):
            normalized = raw.astype('string').str.strip().str.upper()
            values = normalized.map({'CLOSE': 0., 'CLOSED': 0., 'OPEN': 1., '0': 0., '1': 1., '0.0': 0., '1.0': 1.})
        else:
            values = pd.to_numeric(raw, errors='coerce')
        array = values.to_numpy(dtype=np.float32, na_value=np.nan)
        if strict and not np.isfinite(array).all():
            raise ValueError(f'Invalid ALD numeric/valve value in column: {name}')
        result[name] = array
    return result


def transform_live(frame, metadata):
    names = metadata['sensor_columns']
    encoded = encode_frame(frame, names)
    return ((encoded[names].to_numpy(np.float32) - np.asarray(metadata['mean'])) /
            np.asarray(metadata['std'])).astype(np.float32)


def prepare(root, *, config, patch_length, policy=PreprocessingConfig(), progress=print, max_memory_mb=4096):
    config.validate();policy.validate()
    if not math.isfinite(max_memory_mb) or max_memory_mb <= 0:
        raise ValueError('Finite positive ALD working memory budget required')
    max_window_length=policy.max_window_length
    if max_window_length < 2*patch_length:
        raise ValueError('Native window needs two complete encoder patches')
    root=Path(root);directory=root/'data/mt_nit/data_complete'
    canonical,file_audit=schema_inventory(directory,policy)
    paths=[directory/name for name in file_audit.loc[file_audit.included,'file']]
    if not paths:raise FileNotFoundError('data/mt_nit/data_complete contains no CSV files')
    source_files={path.relative_to(root).as_posix():p.digest(path) for path in paths}
    whitelist=set(sensor_candidates(canonical,policy))
    if not whitelist:raise ValueError('Explicit physical measurement sensor whitelist required')
    all_units=[]
    cache_root=root/'.cache/ald-native-inputs'
    for i,path in enumerate(paths):
        header=pd.read_csv(path,nrows=0).columns.tolist()
        if not set(COLUMNS.values())<=set(header):raise ValueError('Native hierarchy columns missing from a data_complete file')
        meta=pd.read_csv(path,usecols=list(set(COLUMNS.values())),dtype='string')
        recipe=meta[COLUMNS['cohort']].fillna('').str.strip()
        times=pd.to_datetime(meta[COLUMNS['order']],format='mixed',errors='coerce')
        recipe_eligible=(recipe.eq(config.recipe_suffix)|recipe.str.endswith('_'+config.recipe_suffix)) if config.recipe_suffix else recipe.ne('')
        eligible=recipe_eligible&~times.dt.month.isin(config.excluded_months)
        if not eligible.any():continue
        selected=list(set(COLUMNS.values()))+[name for name in header if name in whitelist]
        frame=pd.read_csv(path,usecols=selected,dtype={name:'string' for name in set(COLUMNS.values())},low_memory=False)
        # Preserve unknown text for TRAIN rejection; only known valve states are encoded.
        frame=encode_frame(frame,[name for name in whitelist if name in frame and name.startswith('ValveAct_')],strict=False)
        for name in frame.columns:
            if name in whitelist and pd.api.types.is_numeric_dtype(frame[name]):frame[name]=frame[name].astype(np.float32)
        # Bound the current file's working set, rather than accumulating every
        # file's strings and DataFrames for the entire preparation lifetime.
        if 2*int(frame.memory_usage(deep=True).sum())>max_memory_mb*1024**2:
            raise MemoryError('One ALD file exceeds the preparation working memory budget: '+path.name)
        file_sha256=source_files[path.relative_to(root).as_posix()]
        native_units=units(frame,file_id=file_sha256)
        stored=input_store.stage_frame(frame,whitelist,cache_root,source_sha256=file_sha256)
        for unit in native_units:
            unit['frame']=stored.iloc[unit['rows']]
        all_units.extend(native_units)
        del frame,meta,native_units,stored
        if progress and (i==0 or (i+1)%10==0 or i+1==len(paths)):
            progress(f'Native data_complete preparation: {i+1}/{len(paths)} files; sensor columns on disk; no fitting yet',flush=True)
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
        if config.recipe_suffix and any(not _recipe_ok(u['recipe'],config.recipe_suffix) for u in group):continue
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
    # Partition using data_complete only. External inputs, labels, guides and
    # original membership lists do not influence fitting or model selection.
    grouped=[]
    for raw_run,cluster in groups:
        start=min(x['start'] for x in cluster);end=max(x['end'] for x in cluster)
        identity=p._hash((raw_run,start.isoformat(),end.isoformat()))
        group=[u for item in cluster for u in item['units']]
        for u in group:u['group']=identity
        grouped.append(dict(group=identity,start=start,end=end,units=group))
    grouped.sort(key=lambda x:(x['start'],x['group']));a=int(.7*len(grouped));b=a+int(.15*len(grouped))
    partitions=dict(train=grouped[:a],validation=grouped[a:b],test=grouped[b:])
    if min(map(len,partitions.values()))<2:raise ValueError('Need two original Run groups per split after native quality checks')
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
        if progress:progress(f'Prepare native route {len(support)+1}/{len(route_groups)}: {candidates[0]["parent_step"]}',flush=True)
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
            if not numeric or not parts or len(groups)<2:
                sensor_audit.append(dict(step=route,sensor=name,status='text_empty_or_insufficient_TRAIN_support'));continue
            values=np.concatenate(parts)
            if np.ptp(values)<=0:
                sensor_audit.append(dict(step=route,sensor=name,status='constant_TRAIN_channel'));continue
            q=np.quantile(values,[.25,.75]);scale=float((q[1]-q[0])/1.349)
            if scale<=1e-6:scale=float(values.std())
            if not math.isfinite(scale) or scale<=0:continue
            # Binary states retain their 0/1 meaning instead of robust scaling.
            names.append(name);means.append(0. if name.startswith('ValveAct_') else float(np.median(values)));scales.append(1. if name.startswith('ValveAct_') else scale)
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
        meta['context_sensor_columns']=[name for name in names if is_context_sensor(name,policy)]
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
                    pools[split].append(dict(current=current,sensor_columns=observed,run_id=u['group'],lineage_id=u['group'],pair_id=p._hash(key),step=route,split=split,
                        cohort=(p._hash(u['recipe']),token(u['loop']),start,tick),position=0.,input_row_start=int(positions[0]),input_row_end=int(positions[-1])))
        if not pools['train'] or not pools['validation']:
            support.append(dict(step=route,status='UNSUPPORTED',reason='no_native_TRAIN_or_Validation_windows'));continue
        windows[route]=pools;metadata[route]=meta
        support.append(dict(step=route,status='READY',native_kind=sample['kind'],parent_step=sample['parent_step'],
            sensors=len(names),window_length=width,**{k+'_windows':len(v) for k,v in pools.items()}))
    for path in paths:
        if p.digest(path)!=source_files[path.relative_to(root).as_posix()]:raise ValueError('Native source file changed during preparation')
    split_runs={k:sorted({u['group'] for u in values}) for k,values in dataset.items()}
    identity=dict(source_directory='data/mt_nit/data_complete',source_files=source_files,split_runs=split_runs,
        split_policy='chronological_Run_start_with_boundary_purge',
        split_input_scope='data_complete_only',
        lineage_policy=LINEAGE_POLICY,
        native_preparation_sha256=p.digest(__file__),input_store_sha256=p.digest(input_store.__file__),
        native_policy=dict(recipe_suffix=config.recipe_suffix,excluded_months=config.excluded_months),
        patch_length=patch_length,preprocessing=asdict(policy),native_config=asdict(config),canonical_columns=list(canonical),
        file_schema_audit=file_audit.to_dict('records'))
    tables=dict(files=file_audit,support=pd.DataFrame(support),sensors=pd.DataFrame(sensor_audit),templates=pd.DataFrame(templates),
        native_quality=pd.DataFrame(quality),cadence=pd.DataFrame([dict(step=s,cadence_seconds=cadence,tolerance_seconds=tolerance) for s in windows]),
        current_input_policy=pd.DataFrame([dict(source='data_complete_all_files',recipe_suffix=config.recipe_suffix,
            split_policy='chronological_Run_with_boundary_purge',split_input_scope='data_complete_only',
            normality_is_dataset_premise=True,main_timestamp_failure='exclude_Run',subcycle_failure='exclude_SubCycle')]))
    prepared=p.Prepared(windows,metadata,tables,split_runs,identity)
    prepared.process=SimpleNamespace(source_columns=COLUMNS)
    return prepared

