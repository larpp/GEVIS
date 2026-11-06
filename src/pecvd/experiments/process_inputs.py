"""Explicit process input and artifact namespaces, no source data side effects."""
import os
import hashlib
import json
import re
from pathlib import Path
from pecvd.process_config import load_process_config


def configured_process_path(root):
    choice=os.environ.get('GEVIS_PROCESS_CONFIG')
    path=Path(choice) if choice else Path(root)/'config/processes/pecvd.json'
    if choice and not path.is_absolute(): path=Path(root)/path
    return path.resolve()


def candidate_backbone_path(root,config,run_tag,*,revision=None):
    """One explicit training revision for every STEP/runtime consumer.

    Missing revision preserves historical paths. An explicit empty revision is
    reserved for reviewed, read-only historical/text-source lookup. No artifact
    discovery, rename, overwrite or relaxed checkpoint identity is performed.
    """
    root=Path(root)
    if revision is None:
        settings=root/'config/experiments/gevis_pipeline.json'
        revision=json.loads(settings.read_text(encoding='utf-8')).get('backbone_revision','') if settings.is_file() else ''
    if not isinstance(revision,str) or (revision and not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',revision)):
        raise ValueError('backbone_revision must be empty or a bounded lowercase namespace token')
    default=(root/'config/processes/pecvd.json').resolve()
    suffix='' if config.source_path.resolve()==default else '_'+config.artifacts.encoder
    name='gevis_backbone_e5_v1_fdc_semantic_step1_bridge_all_types_v1_'+run_tag+('_'+revision if revision else '')+suffix
    return root/'artifacts'/name/'gevis_backbone.pt'


def external_readiness(root, config=None, backbone=None):
    """CSV existence is NOT preprocessing, transfer, or detection success."""
    root=Path(root)
    path=Path(config) if config else root/'.cache/external/peald/process.json'
    if not path.is_file():
        return {'source_present':(root/'data/merge_peald.csv').is_file(),'config_ready':False,
                'backbone_ready':False,'model_evaluated':False,
                'next':'Prepare external input mapping; retain PECVD source STEP 1/2 weights without external training.'}
    cfg=load_process_config(path)
    from pecvd.gevis_backbone import TARGET_BINDING_VERSION
    source_path=root/'config/processes/pecvd.json'
    source=load_process_config(source_path) if source_path.is_file() else None
    from pecvd.model_variants import active_config,active_run_tag
    weights=Path(backbone) if backbone else (candidate_backbone_path(root,active_config(source),active_run_tag()) if source else None)
    return {'source_present':(root/cfg.data_path).is_file(),'config_ready':True,
            'backbone_ready':bool(weights and weights.is_file()),'model_evaluated':False,
            'next':'Run EXP 4 with frozen PECVD source weights; external reference normalization only.'}


def configured_sensors(config,frame):
    """External sources must explicitly exclude numeric identifiers/targets."""
    columns=list(config.sensor_columns)
    if not columns: return None
    import pandas as pd
    if any(name not in frame or not pd.api.types.is_numeric_dtype(frame[name]) for name in columns):
        raise ValueError('Configured physical sensor columns are absent or nonnumeric')
    return columns


def phase_key(step,context=''):
    value=str(context).strip()+'|'+str(step).strip()
    return 'phase_'+hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]


def step_description(config,step):
    return next((s.description or s.name for s in config.steps if s.name==step),step)


def align_step_occurrences(frame, *, run, step, order):
    """Structural alignment only: preserve sensor values and native row grid.

    Repeated same-Step occurrences are not independent runs or silently pooled.
    Their ordinal position is structural, NOT verified physical sub-cycle truth.
    """
    import numpy as np
    if frame.empty:return frame.copy()
    required=[run,step,order,'_gevis_segment_id']
    if any(c not in frame for c in required):raise ValueError('Missing external segment metadata')
    result=frame.sort_values([run,order],kind='stable').copy()
    if result.duplicated([run,order]).any():raise ValueError('Duplicate run/order coordinates')
    boundary=result[run].ne(result[run].shift()) | result['_gevis_segment_id'].ne(result['_gevis_segment_id'].shift())
    if result.loc[boundary].duplicated([run,'_gevis_segment_id']).any():
        raise ValueError('Segment identity reappears across disjoint intervals')
    segments=result.drop_duplicates([run,'_gevis_segment_id'])[[run,step,'_gevis_segment_id']].copy()
    segments['_gevis_step_occurrence']=segments.groupby([run,step],sort=False).cumcount()
    lookup=segments.set_index([run,'_gevis_segment_id'])['_gevis_step_occurrence']
    keys=list(zip(result[run],result['_gevis_segment_id']))
    result['_gevis_step_occurrence']=[int(lookup.loc[k]) for k in keys]
    group=result.groupby([run,'_gevis_segment_id'],sort=False)
    if group[step].nunique().gt(1).any():raise ValueError('A segment crosses a Step boundary')
    result['_gevis_step_row']=group.cumcount()
    result['_gevis_step_position_ratio']=result['_gevis_step_row']/np.maximum(group[step].transform('size')-1,1)
    return result


def screen_external_quality(frame,config,*,suspect_months=(),excluded_months=(),
                            minimum_step_rows=None,exclude_file_edge_runs=False):
    """Review hints are not fault labels. Exclusions remove whole runs only.

    Dataset-edge Steps may be censored, but an edge alone does not prove it.
    Explicit Step minima must come from process knowledge, not Test outcomes.
    """
    import pandas as pd
    suspect_months=tuple(suspect_months);excluded_months=tuple(excluded_months)
    if any(isinstance(m,bool) or not isinstance(m,int) or not 1<=m<=12 for m in (*suspect_months,*excluded_months)):
        raise ValueError('Calendar months must be integer values from 1 to 12')
    minimum_step_rows=dict(minimum_step_rows or {})
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in minimum_step_rows.values()):
        raise ValueError('Step minimum row counts must be positive integers')
    run,step=config.columns.run,config.columns.step
    out=frame.copy();reasons={};tables={}
    def mark(runs,reason):
        for rid in runs:reasons.setdefault(rid,set()).add(reason)
    has_time='_gevis_time' in out
    if (suspect_months or excluded_months or exclude_file_edge_runs) and not has_time:
        raise ValueError('Calendar/edge screening needs original parsed timestamps')
    if has_time:
        stamp=pd.to_datetime(out['_gevis_time'],errors='coerce')
        if stamp.isna().any():raise ValueError('Quality screening requires valid timestamps')
        month=stamp.dt.month
        out['_gevis_suspect_month']=month.isin(suspect_months)
        excluded_runs=set(out.loc[month.isin(excluded_months),run])
        mark(excluded_runs,'explicit_calendar_exclusion')
        periods=stamp.dt.strftime('%Y-%m')
        tables['external_quality_by_month']=pd.DataFrame([
            dict(period=period,rows=len(g),runs=g[run].nunique(),
                suspect_month=bool(g['_gevis_suspect_month'].any()),
                explicitly_excluded_month=int(period[-2:]) in excluded_months)
            for period,g in out.groupby(periods,sort=True)])
        # Only edges of the available dataset, not every legitimate run boundary.
        edge_keys=set(zip(out.loc[stamp.eq(stamp.min())|stamp.eq(stamp.max()),run],
                         out.loc[stamp.eq(stamp.min())|stamp.eq(stamp.max()),'_gevis_segment_id']))
    else:edge_keys=set()
    segments=[]
    raw_step=getattr(config,'source_columns',{}).get('step')
    for (rid,sid),g in out.groupby([run,'_gevis_segment_id'],sort=False):
        phase=g[step].iloc[0];label=str(g[raw_step].iloc[0]).strip() if raw_step else phase
        minimum=minimum_step_rows.get(phase,minimum_step_rows.get(label))
        too_short=minimum is not None and len(g)<minimum
        edge=(rid,sid) in edge_keys
        if too_short:mark([rid],'explicit_step_minimum_not_met')
        if edge and exclude_file_edge_runs:mark([rid],'explicit_file_edge_exclusion')
        segments.append(dict(step=phase,rows=len(g),dataset_edge_candidate=edge,
            below_declared_minimum=too_short))
    if segments:
        tables['external_step_quality']=pd.DataFrame(segments).groupby('step',sort=False).agg(
            segments=('rows','size'),minimum_observed_rows=('rows','min'),
            maximum_observed_rows=('rows','max'),dataset_edge_candidates=('dataset_edge_candidate','sum'),
            below_declared_minimum=('below_declared_minimum','sum')).reset_index()
    remove=out[run].isin(reasons)
    tables['external_quality_summary']=pd.DataFrame([dict(
        input_runs=out[run].nunique(),excluded_runs=len(reasons),
        input_rows=len(out),excluded_rows=int(remove.sum()),retained_rows=int((~remove).sum()),
        suspect_rows=int(out['_gevis_suspect_month'].sum()) if has_time else 0,
        exclusion_unit='whole run; no fragment stitching',
        note='Suspect months and dataset edges are review hints, not confirmed corruption.')])
    tables['external_quality_exclusions']=pd.DataFrame([
        dict(reason=reason,runs=sum(reason in rs for rs in reasons.values()))
        for reason in sorted({r for rs in reasons.values() for r in rs})])
    out=out.loc[~remove].copy()
    out.attrs['external_quality_options']=dict(suspect_months=list(suspect_months),
        excluded_months=list(excluded_months),minimum_step_rows=minimum_step_rows,
        exclude_file_edge_runs=exclude_file_edge_runs)
    return out,tables


def load_process_frame(root,config):
    """Offline preparation ONLY. Runtime/experiments continue to use caches."""
    import numpy as np
    import pandas as pd
    mapping=config.source_columns
    if not mapping: return pd.read_csv(Path(root)/config.data_path)
    usecols=list(dict.fromkeys([*mapping.values(),*config.sensor_columns]))
    frame=pd.read_csv(Path(root)/config.data_path,usecols=usecols,dtype={name:'string' for name in mapping.values()},low_memory=False)
    valid=frame[mapping['run']].notna() & frame[mapping['step']].notna()
    valid &= frame[mapping['run']].str.strip().ne('') & frame[mapping['step']].str.strip().ne('')
    frame=frame.loc[valid].copy()
    stamp=pd.to_datetime(frame[mapping['order']],format='mixed',errors='coerce')
    if stamp.isna().any(): raise ValueError('External timestamps are ambiguous or invalid; no silent reordering')
    if frame.assign(_stamp=stamp).duplicated([mapping['run'],'_stamp']).any():
        raise ValueError('Duplicate external timestamps within a run; resolve sampling identity before inference')
    frame['_gevis_time']=stamp
    frame['_gevis_input_row']=np.arange(len(frame))
    frame[config.columns.run]=frame[mapping['run']].str.strip().astype(str)
    context=frame[mapping['step_context']].fillna('') if mapping.get('step_context') else ['']*len(frame)
    frame[config.columns.step]=[phase_key(s,c) for s,c in zip(frame[mapping['step']],context)]
    frame=frame.sort_values([config.columns.run,'_gevis_time','_gevis_input_row'],kind='stable')
    frame[config.columns.order]=frame.groupby(config.columns.run).cumcount()
    changed=frame[config.columns.run].ne(frame[config.columns.run].shift()) | frame[config.columns.step].ne(frame[config.columns.step].shift())
    frame['_gevis_segment_id']=changed.cumsum()
    sizes=frame.groupby(config.columns.run)[config.columns.order].transform('max').clip(lower=1)
    frame['_gevis_position_ratio']=frame[config.columns.order]/sizes
    for sensor in config.sensor_columns: frame[sensor]=pd.to_numeric(frame[sensor],errors='coerce')
    if mapping.get('cohort'):
        cohort=frame[mapping['cohort']].fillna('unknown').map(lambda x:hashlib.sha256(str(x).encode()).hexdigest()[:12])
        if frame.assign(_cohort=cohort).groupby(config.columns.run)._cohort.nunique().gt(1).any():
            raise ValueError('Recipe changes inside a declared run; grouping needs revision')
        for value in cohort.unique(): frame['source_group_'+value]=cohort.eq(value).astype('int8')
    return align_step_occurrences(frame,run=config.columns.run,step=config.columns.step,order=config.columns.order)
