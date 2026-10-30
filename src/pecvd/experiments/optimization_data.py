"""Run/reference boundaries for isolated optimization; no artifact writes."""
import numpy as np


def restrict_training_references(rows,values,ids,cohorts,positions,allowed_runs,*,max_references):
    """Rebuild TRAIN references from fitting runs only, retaining current inputs.

    The original TRAIN-fitted scaler is frozen preprocessing, not a newly fitted
    internal-fold scaler. No claim of pretraining-disjoint internal selection.
    """
    values=np.asarray(values);ids=np.asarray(ids).astype(str)
    cohorts=np.asarray(cohorts).astype(str);positions=np.asarray(positions)
    allowed=set(allowed_runs)
    if values.ndim!=3 or len(values)!=len(ids) or cohorts.shape!=ids.shape or positions.shape!=ids.shape or max_references<1:
        raise ValueError('Aligned TRAIN cache and positive reference budget required')
    result=[]
    for row in rows:
        if row['split']!='train' or row['run_id'] not in set(ids):raise ValueError('TRAIN current rows required')
        matches=np.flatnonzero((cohorts==row['cohort'])&(positions==row['position'])&(ids!=row['run_id']))
        chosen={}
        for j in matches:
            if ids[j] in allowed:chosen.setdefault(ids[j],int(j))
        selected=list(chosen)[:max_references]
        if not selected:raise ValueError('No fitting-run reference support; do not drop cases silently')
        reference=np.median(values[[chosen[key] for key in selected]],axis=0).astype('float32')
        result.append(dict(row,reference=reference,reference_run_ids=selected))
    return result


def attach_reference_context(rows,values,ids,cohorts,positions):
    """Summarize the exact existing TRAIN reference members, never change pool."""
    from .optimization_models import local_waveform_statistics
    ids=np.asarray(ids).astype(str);cohorts=np.asarray(cohorts).astype(str)
    lookup={};stats={}
    for j,(run,cohort,position) in enumerate(zip(ids,cohorts,positions)):
        lookup.setdefault((run,cohort,int(position)),j)
    result=[]
    for row in rows:
        if row['run_id'] in row['reference_run_ids']:raise ValueError('Self reference forbidden')
        members=[lookup[(run,row['cohort'],row['position'])] for run in row['reference_run_ids']]
        if not members:raise ValueError('Exact reference members required')
        if not np.array_equal(np.median(values[members],axis=0).astype('float32'),row['reference']):
            raise ValueError('Reference pool reconstruction mismatch')
        for j in members:
            if j not in stats:stats[j]=local_waveform_statistics(values[j])
        summary=np.median(np.stack([stats[j] for j in members]),axis=0).astype('float32')
        result.append(dict(row,reference_context=summary))
    return result


def collect_protocol_rows(runtime,config):
    from .discovery_calibration import cached_comparisons
    from .optimization_models import split_runs
    module=runtime.module;prepared={};train=[];validation=[]
    for step in runtime.encoders:
        prepared[step]={}
        for split in ('train','validation'):
            rows=[];seen=set();limit=config[split+'_runs_per_step']
            for row in cached_comparisons(module.ENCODER_ROOT,step,split=split,config=module.DISCOVERY_POLICY.config):
                if row['run_id'] in seen:continue
                seen.add(row['run_id']);rows.append(dict(row,step=step))
                if len(seen)==limit:break
            if len(seen)!=limit:raise ValueError('Insufficient independent current runs for fixed protocol')
            prepared[step][split]=rows
            (train if split=='train' else validation).extend(rows)
    if {r['run_id'] for r in train}&{r['run_id'] for r in validation}:raise ValueError('Main split overlap')
    selection,fit=split_runs(train,config['train_selection_fraction'],config['seed']+1)
    selection_ids={r['run_id'] for r in selection};fit_ids={r['run_id'] for r in fit}
    old_overlap=sum(bool(selection_ids&set(r['reference_run_ids'])) for r in fit)
    for step,groups in prepared.items():
        with np.load(module.ENCODER_ROOT/'windows'/f'{step}.npz',allow_pickle=False) as cache:
            groups['train']=restrict_training_references(groups['train'],cache['train'],cache['train_process_id'],
                cache['train_cohort_key'],cache['train_position_bin'],fit_ids,
                max_references=module.DISCOVERY_POLICY.config.max_references)
            if 'reference_context' in config.get('feature_views',[]):
                for split in ('train','validation'):
                    groups[split]=attach_reference_context(groups[split],cache['train'],cache['train_process_id'],
                        cache['train_cohort_key'],cache['train_position_bin'])
        if any(selection_ids&set(r['reference_run_ids']) for r in groups['train']):raise RuntimeError('Selection reference leakage')
    audit=dict(fit_windows=len(fit),selection_windows=len(selection),
        prior_fit_windows_using_selection_reference=old_overlap,
        restricted_train_windows_using_selection_reference=0,
        validation_inputs_unchanged=True,original_train_preprocessing_frozen=True)
    return prepared,audit
