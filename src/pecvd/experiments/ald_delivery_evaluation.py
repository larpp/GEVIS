"""Frozen ALD Run replay, then evaluator-only membership/observation diagnostics.

Uses the canonical model unchanged. No fitting, oracle references, threshold
selection, input export or whole-Run probability. All results stay in memory.
"""
from pathlib import Path
import json
import hashlib
import time

import numpy as np
import pandas as pd

from . import external_test as external, fusion_answers as fa
from . import online_inference as online
from .reporting import ExperimentReport

VERSION = 'gevis.ald_delivery_replay.normal_train.v2'
RUN_RULE = 'any_BOOLEAN_Yes_over_all_answered_supported_sensor_windows'
COORDS = ['case_id', 'step', 'row_start', 'row_end', 'center_sensor']


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def delivery_boundaries(frame, total_rows):
    required = ['run_id', 'input_data_start_row_1', 'input_data_end_row_1']
    if not set(required) <= set(frame) or frame.empty:
        raise ValueError('Run boundary inventory is missing')
    if frame.run_id.isna().any() or frame.run_id.astype(str).duplicated().any():
        raise ValueError('Unique nonempty Run IDs required')
    out = []; previous = 0
    for row in frame.to_dict('records'):
        values = [float(row[c]) for c in required[1:]]
        if not all(np.isfinite(v) and v.is_integer() for v in values):
            raise ValueError('Integer 1-based inclusive Run bounds required')
        start, end = int(values[0])-1, int(values[1])
        if start != previous or end <= start or end > total_rows:
            raise ValueError('Run bounds must partition all input rows in order')
        if 'run_rows' in row and float(row['run_rows']) != end-start:
            raise ValueError('Run row count differs from bounds')
        out.append(dict(run_id=str(row['run_id']), start=start, end=end)); previous=end
    if previous != total_rows:raise ValueError('Rows remain outside Run bounds')
    return out


def discover_package(root):
    """Inspect input presence only; evaluator files never select the package."""
    found = [p for base in (Path(root)/'notebooks', Path(root)/'data') for p in base.glob('test*')
             if (p/'inputs/test.csv').is_file() and (p/'inputs/run_boundaries.csv').is_file()]
    if len(found) != 1:
        raise ValueError('Set DELIVERY_PACKAGE explicitly: expected one input/boundaries package')
    return found[0]


def _inputs(package):
    package = Path(package)
    return {name: external.peald.digest(package/'inputs'/name)
            for name in ('test.csv', 'run_boundaries.csv')}


def _seal(report):
    return _digest(dict(metadata={k: v for k, v in report.metadata.items() if k != 'prediction_sha256'},
        details=report.details.to_dict('records'),
        tables={k: v.to_dict('records') for k, v in report.diagnostics.items() if isinstance(v, pd.DataFrame)}))


def decisions(details):
    """Three output slots; No-gated types are explicitly not applicable."""
    if details.empty:
        return pd.DataFrame(columns=COORDS+['BOOLEAN', 'CHOICE', 'ORDINAL', 'p_yes', 'contradiction'])
    if details.duplicated(COORDS+['answer_type']).any():
        raise ValueError('Duplicate typed sensor/window prediction')
    records = []
    for coord, group in details.groupby(COORDS, sort=False, dropna=False):
        if set(group.answer_type) != set(fa.KINDS) or len(group) != 3:
            raise ValueError('Every answered sensor/window needs all three typed answers')
        row = dict(zip(COORDS, coord))
        gate=int(group.loc[group.answer_type.eq('BOOLEAN'),'prediction'].iloc[0])
        for answer in group.to_dict('records'):
            kind = answer['answer_type']; criteria = fa.ANSWER_CRITERIA[kind]
            if answer.get('evaluated') is False:
                if (kind=='BOOLEAN' or gate!=0 or answer.get('status')!='NOT_APPLICABLE'
                        or not pd.isna(answer.get('prediction')) or answer.get('answer') is not None
                        or answer.get('probabilities') is not None):
                    raise ValueError('Only BOOLEAN No permits empty not-applicable typed outputs')
                row[kind]=None;row[kind+'_index']=None;continue
            probs = answer['probabilities']
            if set(probs) != set(criteria):
                raise ValueError('Probability criteria differ from the saved head contract')
            values = np.asarray([probs[c] for c in criteria], float)
            if (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                    or not np.isclose(values.sum(), 1., atol=1e-5)):
                raise ValueError('Invalid head probability vector')
            index = int(values.argmax())
            if answer['prediction'] != index or answer['answer'] != criteria[index]:
                raise ValueError('Head probability argmax and displayed answer differ')
            row[kind] = criteria[index]
            row[kind+'_index'] = index
            if kind == 'BOOLEAN':row['p_yes'] = float(values[1])
        if gate==0 and row['CHOICE_index'] is None and row['ORDINAL_index'] is None:
            row['contradiction']=False;records.append(row);continue
        row['contradiction'] = bool((row['BOOLEAN_index'] == 0) !=
            (row['CHOICE_index'] == 0 and row['ORDINAL_index'] == 0) or
            (row['BOOLEAN_index'] == 1 and (row['CHOICE_index'] == 0 or row['ORDINAL_index'] == 0)))
        records.append(row)
    return pd.DataFrame(records)


def replay(contract, package, *, center_batch_size=8, expected_runs=20, progress=print, session=None):
    """Only inputs + boundaries enter this phase. No evaluator directory access."""
    if contract.domain != 'PEALD' or not hasattr(contract.runtime, 'prepared'):
        raise ValueError('Load this copy existing PEALD contract first')
    runtime = contract.runtime
    external.fast.require_runtime(runtime)
    training = runtime.module.FUSION_ANSWER_TRAINING or {}
    from . import decision_training as decision
    if training.get('checkpoint_binding')==decision.SCHEMA and training.get('input_mode')==decision.INPUT_MODE:
        from . import decision_replay
        if session is not None:raise ValueError('Detailed replay constructs its native-context session')
        return decision_replay.replay_delivery(contract,package,center_batch_size=center_batch_size,
            expected_runs=expected_runs,progress=progress)
    if training.get('input_mode') == 'current_only':
        from . import current_input
        if session is not None:
            raise ValueError('Current-only replay constructs its own bounded streaming session')
        return current_input.replay_delivery(contract, package, center_batch_size=center_batch_size, expected_runs=expected_runs, progress=progress)
    if (training.get('checkpoint_binding') != 'ALD_evaluate_saved_head' or
            training.get('target_domain') != 'PEALD' or training.get('optimizer_updates', 0) < 1):
        raise ValueError('Restore the ALD_evaluate saved PEALD checkpoint before external replay')
    package = Path(package).resolve(); identity = _inputs(package)
    model_before = external._fingerprint(runtime)
    started = time.perf_counter()
    frame = external.read_input(package/'inputs/test.csv', contract)
    boundaries = delivery_boundaries(pd.read_csv(package/'inputs/run_boundaries.csv', dtype={'run_id': str}), len(frame))
    if expected_runs is not None and len(boundaries) != expected_runs:
        raise ValueError('Unexpected delivery Run count; check input package')
    input_read_ms = 1000*(time.perf_counter()-started)
    runner = session or online.InferenceSession(runtime, center_batch_size=center_batch_size)
    if runner.runtime is not runtime:raise ValueError('Session belongs to another model')
    rows = []; coverage = []; requests = []; exclusions = []
    for order, boundary in enumerate(boundaries):
        rid = boundary['run_id']; start = boundary['start']; end = boundary['end']
        # Preserve original per-Run coordinates without joining consecutive Runs.
        part = frame.iloc[start:end].reset_index(drop=True)
        part.attrs = dict(frame.attrs)
        started = time.perf_counter()
        cases, rejected = external.windows(part, contract, rid)
        cases.sort(key=lambda w: (w['row_start'], w['row_end'], w['step']))
        preparation_ms = 1000*(time.perf_counter()-started)
        for item in rejected:
            item = dict(item, case_id=rid)
            for key in ('source_row_start', 'source_row_end'):
                if key in item:item[key] += start
            exclusions.append(item)
        observed = np.zeros(end-start, dtype=bool); answered = 0; yes = 0
        overlap = external._overlap(runtime, [c['run_id'] for c in cases])
        if progress:progress(f'Run {order+1}/{len(boundaries)}: {len(cases)} native windows; all supported sensors', flush=True)
        for i, case in enumerate(cases):
            result = runner.process_window(current=case['raw_current'], step=case['step'],
                cohort=case['cohort'], run_id=case['run_id'], centers=case['centers'], position=case['position'])
            a = start+case['row_start']; b = start+case['row_end']
            request_id = f'{rid}/{case["step"]}/{a}:{b}'
            requests.append(dict(case_id=rid, request_id=request_id, step=case['step'],
                row_start=a, row_end=b, status=result['status'], reason=result['reason'],
                selected_sensors=len(case['centers']), answers=len(result['answers']),
                reference_count=result.get('reference_count', 0), **result['timings']))
            if result['status'] == 'OK':
                names = {r['center_sensor'] for r in result['answers']}
                if names != set(case['centers']) or len(result['answers']) != 3*len(names):
                    raise ValueError('Incomplete selected-sensor typed answer set')
                observed[case['row_start']:case['row_end']] = True; answered += 1
                rows.extend(dict(r, case_id=rid, run_id=rid, source_group=case['run_id'],
                    step=case['step'], row_start=a, row_end=b, pair_id=request_id)
                    for r in result['answers'])
                yes += sum(r['answer_type'] == 'BOOLEAN' and r['prediction'] == 1 for r in result['answers'])
            else:
                if result['answers']:raise ValueError('Failed request must not carry predictions')
                exclusions.append(dict(case_id=rid, step=case['step'], source_row_start=a,
                    source_row_end=b, rows=b-a, reason=result['reason']))
            if progress and ((i+1) % 50 == 0 or i+1 == len(cases)):
                progress(f'Run {order+1}/{len(boundaries)}: {i+1}/{len(cases)} windows completed', flush=True)
        nboolean = sum(1 for r in rows if r['case_id'] == rid and r['answer_type'] == 'BOOLEAN')
        complete_rows = bool(observed.all() and answered == len(cases) and not rejected)
        coverage.append(dict(case_id=rid, run_order=order, input_start=start, input_end=end,
            input_rows=end-start, available_windows=len(cases), answered_windows=answered,
            sensor_windows=nboolean, yes_sensor_windows=yes,
            observed_prediction=int(yes > 0) if nboolean else np.nan,
            # Fixed diagnostic membership on the observed model scope only.
            # Never selectively score positives while withholding partial negatives.
            run_prediction=int(yes > 0) if nboolean else np.nan,
            scope_status='COMPLETE_SUPPORTED_ROWS' if complete_rows else ('PARTIAL_SUPPORTED_ROWS' if nboolean else 'UNSCORED'),
            whole_run_negative_certified=False,
            input_row_coverage=float(observed.mean()), quality_exclusion_records=len(rejected),
            overlap_status=overlap, preparation_ms=preparation_ms))
    if _inputs(package) != identity:raise ValueError('Delivery inputs changed during replay')
    if external._fingerprint(runtime) != model_before:raise RuntimeError('Frozen model changed during replay')
    details = pd.DataFrame(rows)
    joint = decisions(details)
    latency, by_step = online.latency_tables(requests)
    report = ExperimentReport('ALD delivery replay: frozen head and independent TRAIN references',
        pd.DataFrame([dict(runs=len(boundaries), answered_runs=sum(r['sensor_windows'] > 0 for r in coverage),
            questions=len(rows), input_read_ms=input_read_ms, accuracy_available=False)]), details=details,
        diagnostics=dict(coverage=pd.DataFrame(coverage), exclusions=pd.DataFrame(exclusions),
            requests=pd.DataFrame(requests), decisions=joint, request_latency=latency, request_latency_by_step=by_step),
        metadata=dict(version=VERSION, decision_system=fa.SYSTEM, package=str(package), input_identity=identity,
            model_sha256=model_before, checkpoint=str(runtime.module.FUSION_ANSWER_PATH),
            head_training_reference_mode=training['reference_mode'],
            reference_contract=training['reference_contract'],
            original_runs_normal_is_dataset_premise=bool(training.get('original_runs_normal')),
            training_deployment_reference_aligned=training.get('reference_mode') == 'normal_train',
            checkpoint_selection=training['checkpoint_selection'],
            input_window_policy='unchanged local external.windows; unsupported phases remain unobserved',
            run_rule=RUN_RULE, training=False, selection=False, automatic_export=False,
            independent_test_certified=False, real_fault_truth=False, original_used_for_prediction=False,
            timing_scope='per-window model request; disk/Run segmentation and waiting excluded; preparation separate',
            limitation='Run any-Yes is a fixed diagnostic aggregation, not a learned Run classifier or probability. '
                'All observed-scope negatives enter diagnostic scoring, including partial observations. '
                'They are not full-Run clearance. Row coverage is not all-sensor cell coverage.'))
    report.metadata['prediction_sha256'] = _seal(report)
    return report


def _binary(rows, column, *, scope):
    scored = rows.loc[rows[column].notna()]
    t = scored.run_label.to_numpy(int); p = scored[column].to_numpy(int)
    tn = int(((t == 0) & (p == 0)).sum()); fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum()); tp = int(((t == 1) & (p == 1)).sum())
    ratio = lambda a, b: a/b if b else np.nan
    return dict(scope=scope, total_runs=len(rows), scored_runs=len(scored), unscored_runs=len(rows)-len(scored),
        partial_runs=int(rows.scope_status.eq('PARTIAL_SUPPORTED_ROWS').sum()), tn=tn, fp=fp, fn=fn, tp=tp,
        precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn), f1=ratio(2*tp,2*tp+fp+fn),
        false_positive_rate=ratio(fp,fp+tn), accuracy=ratio(tp+tn,len(scored)))


def score_saved(report, package, *, expected_injected=8):
    """Posthoc Run membership only. Never pass events/labels to the model."""
    package = Path(package).resolve()
    if (report.metadata.get('version') != VERSION or report.metadata.get('package') != str(package)
            or _seal(report) != report.metadata.get('prediction_sha256')):
        raise ValueError('Complete frozen predictions from this package are required before scoring')
    if _inputs(package) != report.metadata['input_identity']:raise ValueError('Delivery input identity changed')
    files = {name: external.peald.digest(package/'evaluator'/name)
             for name in ('run_labels.csv', 'events.json', 'changed_cells.csv')}
    labels = pd.read_csv(package/'evaluator/run_labels.csv', dtype={'run_id': str, 'event_id': str})
    runs = report.diagnostics['coverage'].copy()
    if not {'run_id', 'run_label'} <= set(labels) or labels.run_id.isna().any() or labels.run_id.duplicated().any():
        raise ValueError('Unique Run truth inventory required')
    if set(labels.run_id) != set(runs.case_id) or not labels.run_label.isin([0, 1]).all():
        raise ValueError('Run truth must match the whole input inventory exactly')
    if expected_injected is not None and labels.run_label.sum() != expected_injected:
        raise ValueError('Unexpected injected Run count')
    runs = runs.merge(labels, left_on='case_id', right_on='run_id', validate='one_to_one', sort=False)
    for key, target in [('input_data_start_row_1','input_start'), ('input_data_end_row_1','input_end'), ('run_rows','input_rows')]:
        if key not in runs:raise ValueError('Run labels need original input bounds')
        wanted = runs[target]+1 if key.endswith('start_row_1') else runs[target]
        if not pd.to_numeric(runs[key], errors='coerce').eq(wanted).all():raise ValueError('Label/input Run bounds differ')
    for column, outcome in [('run_prediction','outcome'), ('observed_prediction','observed_outcome')]:
        runs[outcome] = [f'{"T" if t == p else "F"}{"P" if p == 1 else "N"}' if pd.notna(p)
                         else 'UNSCORED' for t,p in zip(runs.run_label,runs[column])]
    events = json.loads((package/'evaluator/events.json').read_text(encoding='utf-8-sig'))['events']
    if len({e['event_id'] for e in events}) != len(events):raise ValueError('Unique event IDs required')
    positive_ids = set(runs.loc[runs.run_label.eq(1), 'run_id'])
    if {e['run_id'] for e in events} != positive_ids or len(events) != len(positive_ids):
        raise ValueError('Expected one event per injected Run and no control events')
    cells = pd.read_csv(package/'evaluator/changed_cells.csv',
        usecols=['run_id','event_id','input_data_row_1','sensor'], dtype={'run_id':str,'event_id':str,'sensor':str})
    if cells.empty or cells[['run_id','event_id','sensor']].isna().any().any() or cells.duplicated(['input_data_row_1','sensor']).any():
        raise ValueError('Unique changed-cell inventory required')
    cells['row'] = pd.to_numeric(cells.input_data_row_1, errors='coerce')-1
    if not np.isfinite(cells.row).all() or (cells.row != np.floor(cells.row)).any():raise ValueError('Integer changed-cell rows required')
    if set(cells.event_id) != {e['event_id'] for e in events}:raise ValueError('Changed-cell event inventory differs')
    joint = report.diagnostics['decisions']; observation=[]; responses=[]
    lookup = runs.set_index('run_id')
    for e in events:
        rid = e['run_id']; row = lookup.loc[rid]; eid = e['event_id']; sensor = e['sensor']
        lo = int(e['input_data_start_row_1'])-1; hi = int(e['input_data_end_row_1'])
        if not row.input_start <= lo < hi <= row.input_end:raise ValueError('Event outside its Run')
        if str(row.event_id) != str(eid):raise ValueError('Run label and event identity differ')
        changed = cells.loc[cells.event_id.eq(eid)]
        if (not changed.run_id.eq(rid).all() or not changed.sensor.eq(sensor).all()
                or not changed.row.between(lo,hi-1).all() or len(changed) != int(e['changed_cells'])):
            raise ValueError('Changed cells do not match the event sensor/span/count')
        g = joint.loc[joint.case_id.eq(rid) & joint.center_sensor.eq(sensor)]
        overlap = g.loc[g.row_start.lt(hi) & g.row_end.gt(lo)]
        seen = sum(bool(((g.row_start <= r) & (g.row_end > r)).any()) for r in changed.row)
        observation.append(dict(run_id=rid,event_id=eid,sensor=sensor,injection_type=e['type_id'],
            changed_cells=len(changed),observed_changed_cells=seen,observed_fraction=seen/len(changed),
            answered_overlapping_windows=len(overlap),event_window_any_yes=bool(overlap.BOOLEAN_index.eq(1).any()) if len(overlap) else None,
            observation_status='OBSERVED' if seen == len(changed) else ('PARTIAL' if seen else 'UNOBSERVED')))
        responses.extend(dict(item, event_id=eid, injection_type=e['type_id'],
            typed_truth_status='NOT_SCORED: event type/level is not reviewed question truth')
            for item in overlap.to_dict('records'))
    if files != {name: external.peald.digest(package/'evaluator'/name) for name in files}:
        raise ValueError('Evaluator files changed during scoring')
    metrics = pd.DataFrame([_binary(runs,'run_prediction',
        scope='fixed any-Yes Run membership on observed supported scope; partial negatives included')])
    return dict(runs=runs.sort_values('run_order'), run_metrics=metrics,
        events=pd.DataFrame(observation), event_responses=pd.DataFrame(responses),
        truth_identity=files, prediction_sha256=report.metadata['prediction_sha256'],
        typed_accuracy_available=False, independent_test_certified=False)


def score_reviewed_answers(report, labels, *, truth_source):
    if _seal(report) != report.metadata.get('prediction_sha256'):raise ValueError('Predictions changed')
    return external.score_reviewed_labels(report, labels, truth_source=truth_source)


def plot_runs(scored):
    """Injection membership versus fixed observed-scope aggregation; no clearance."""
    import matplotlib.pyplot as plt
    runs = scored['runs']; x = np.arange(len(runs))
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={'width_ratios':[3,1]})
    axes[0].scatter(x-.12, runs.run_label, marker='s', facecolors='none', edgecolors='black',label='Injection truth')
    valid = runs.run_prediction.notna()
    axes[0].scatter(x[valid]+.12,runs.loc[valid,'run_prediction'],marker='x',label='Run decision')
    partial = valid & runs.scope_status.eq('PARTIAL_SUPPORTED_ROWS')
    axes[0].scatter(x[partial]+.12,runs.loc[partial,'run_prediction'],s=65,marker='o',facecolors='none',edgecolors='orange',label='Partial scope')
    missing = runs.observed_prediction.isna()
    axes[0].scatter(x[missing],np.full(int(missing.sum()),-.15),marker='v',color='grey',label='Unscored')
    axes[0].set(xticks=x,xticklabels=runs.run_id,yticks=[0,1],ylim=(-.3,1.2),title='Sequential Run replay; partial negatives are not clearance')
    axes[0].tick_params(axis='x',rotation=90);axes[0].legend(fontsize=8)
    metric=scored['run_metrics'].iloc[0];matrix=np.array([[metric.tn,metric.fp],[metric.fn,metric.tp]],int)
    axes[1].imshow(matrix,cmap='Blues')
    for (a,b),value in np.ndenumerate(matrix):axes[1].text(b,a,str(value),ha='center',va='center',color='black')
    axes[1].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted membership',ylabel='Injection truth',
        title=f'Observed-scope Run membership | unscored={int(metric.unscored_runs)}')
    fig.tight_layout();return fig


def plot_sensors(report):
    import matplotlib.pyplot as plt
    order=report.diagnostics['coverage'].case_id.tolist();g=report.diagnostics['decisions']
    if g.empty:return None
    heat=g.pivot_table(index='case_id',columns='center_sensor',values='BOOLEAN_index',aggfunc='mean').reindex(order)
    fig,ax=plt.subplots(figsize=(max(12,.22*len(heat.columns)),7))
    cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('lightgrey')
    im=ax.imshow(np.ma.masked_invalid(heat.to_numpy()),aspect='auto',vmin=0,vmax=1,cmap=cmap)
    ax.set(xticks=np.arange(len(heat.columns)),xticklabels=heat.columns,yticks=np.arange(len(order)),yticklabels=order,
        title='BOOLEAN Yes fraction per Run/sensor across answered windows; grey = no answers')
    ax.tick_params(axis='x',rotation=90,labelsize=8);fig.colorbar(im,ax=ax,label='Yes window fraction')
    fig.tight_layout();return fig


def plot_timeline(report, run_id, *, sensor=None):
    """Saved answers only, without another neural forward or truth-based selection."""
    import matplotlib.pyplot as plt
    g=report.diagnostics['decisions']
    g=g.loc[g.case_id.eq(run_id)]
    if sensor is not None:g=g.loc[g.center_sensor.eq(sensor)]
    if g.empty:raise ValueError('No saved answers for selected Run/sensor')
    fig,axes=plt.subplots(3,1,figsize=(14,7),sharex=True)
    x=(g.row_start+g.row_end-1)/2+1
    for ax,kind in zip(axes,('BOOLEAN','CHOICE','ORDINAL')):
        values=g.p_yes if kind=='BOOLEAN' else g[kind+'_index']
        ax.scatter(x,values,s=9,alpha=.6)
        ax.set_ylabel('P(Yes)' if kind=='BOOLEAN' else kind)
        if kind!='BOOLEAN':
            criteria=fa.ANSWER_CRITERIA[kind]
            ax.set_yticks(np.arange(len(criteria)),criteria)
    axes[0].set_title(f'{run_id} | {sensor or "all supported sensors"} | saved window answers, no new inference')
    axes[-1].set_xlabel('Input data row (1-based); marker at model-window center')
    fig.tight_layout();return fig
