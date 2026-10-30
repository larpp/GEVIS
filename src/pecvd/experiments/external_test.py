"""Read-only supplied CSV Test and unlabelled new-input review.

No injection, fitting, Test normalization, model selection or automatic export.
An original-paired comparison is a synthetic change task, never fault truth.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from . import fast_experiments as fast, fusion_answers as fa, peald_training as peald
from . import inference_benchmark as timing
from . import online_inference as online
from .process_inputs import phase_key
from .reporting import ExperimentReport


@dataclass
class InputContract:
    runtime: object
    process: object
    domain: str = 'PEALD'


def load_peald(root, *, steps=None):
    """Restore the exact existing EXP6 DEP model; never fit missing weights."""
    from pecvd.process_config import load_process_config
    root = Path(root)
    source = fast.load_runtime(root)
    process = load_process_config(root/'.cache/external/peald/process.json')
    config = peald.PealdConfig(**json.loads((root/'config/experiments/peald_fast.json').read_text(encoding='utf-8')))
    prepared = peald.prepare(root, process, patch_length=source.encoders['DEP.'][0].patch_len, config=config)
    target = peald.runtime_for(source, prepared, source_step='DEP.')
    peald.train_or_load(root, target, steps=steps, config=config, allow_training=False)
    return InputContract(target, process)


def inventory(directory):
    paths = sorted(Path(directory).glob('*.csv'))
    if not paths:
        raise FileNotFoundError('No input CSVs; select the local supplied or new-input directory')
    return paths


def _columns(contract):
    m = contract.process.source_columns
    if not {'run', 'step', 'order'} <= set(m):
        raise ValueError('Explicit raw run, Step and timestamp mapping required')
    names = list(dict.fromkeys(n for _, meta in contract.runtime.encoders.values() for n in meta['sensor_columns']))
    return list(dict.fromkeys(m.values())), names


def read_input(path, contract):
    keys, sensors = _columns(contract)
    frame = pd.read_csv(path, usecols=keys+sensors, dtype={k:'string' for k in keys}, low_memory=False)
    for sensor in sensors:
        frame[sensor] = pd.to_numeric(frame[sensor], errors='coerce')
    if frame.empty:
        raise ValueError('Empty input CSV')
    return frame


def _keys(frame, keys):
    return list(frame[keys].astype('string').fillna('').itertuples(index=False, name=None))


def original_lookup(paths, original_path, contract, *, chunk_size=20000):
    """Read only requested source coordinates; duplicate keys are ambiguous."""
    keys, sensors = _columns(contract)
    wanted = set()
    for path in paths:
        frame = pd.read_csv(path, usecols=keys, dtype='string')
        wanted.update(_keys(frame, keys))
    lookup = {}; ambiguous = set(); offset = 0
    before = peald.digest(original_path)
    for chunk in pd.read_csv(original_path, usecols=keys+sensors,
            dtype={k:'string' for k in keys}, chunksize=chunk_size, low_memory=False):
        for local, key in enumerate(_keys(chunk, keys)):
            if key not in wanted:
                continue
            if key in lookup:
                ambiguous.add(key)
                continue
            lookup[key] = (offset+local, chunk.iloc[local][sensors].to_numpy(dtype=float))
        offset += len(chunk)
    if peald.digest(original_path) != before:
        raise ValueError('Original changed during alignment')
    return dict(rows=lookup, ambiguous=ambiguous, source_sha256=before)


def align_original(frame, lookup, contract):
    keys, sensors = _columns(contract)
    coordinates = _keys(frame, keys)
    if len(set(coordinates)) != len(coordinates):
        raise ValueError('Duplicate input coordinates; no ambiguous source join')
    if any(k in lookup['ambiguous'] for k in coordinates):
        raise ValueError('Ambiguous original coordinates; no automatic source choice')
    if any(k not in lookup['rows'] for k in coordinates):
        raise ValueError('Original coordinates unavailable; no synthetic truth invented')
    values = np.stack([lookup['rows'][k][1] for k in coordinates])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite original values; paired truth unavailable')
    reference = frame.copy()
    reference[sensors] = values
    group = hashlib.sha256(repr(coordinates).encode()).hexdigest()[:20]
    source_rows = [lookup['rows'][k][0] for k in coordinates]
    return reference, group, source_rows


def _overlap(runtime, run_ids):
    training = getattr(runtime.module, 'FUSION_ANSWER_TRAINING', None) or {}
    splits = getattr(getattr(runtime, 'prepared', None), 'split_runs', {})
    # Preprocessing may use more TRAIN runs than the bounded head-fit sample.
    train = set(training.get('train_runs', [])) | set(splits.get('train', []))
    validation = set(training.get('validation_runs', [])) | set(splits.get('validation', []))
    if set(run_ids) & (train | validation):
        return 'overlaps_fit_or_selection_runs'
    return 'no_registered_head_fit_overlap' if train and validation else 'unknown'


def windows(frame, contract, case_id, *, centers=None, stride=None):
    """Use frozen native Step lengths/scales/cadence; never bridge a boundary."""
    runtime = contract.runtime
    fast.require_runtime(runtime)
    m = contract.process.source_columns
    stamps = pd.to_datetime(frame[m['order']], format='mixed', errors='coerce')
    runs = frame[m['run']].astype('string').fillna('').str.strip()
    steps = frame[m['step']].astype('string').fillna('').str.strip()
    contexts = frame[m['step_context']].fillna('') if m.get('step_context') else ['']*len(frame)
    phases = pd.Series([phase_key(s, c) for s, c in zip(steps, contexts)], index=frame.index)
    recipes = frame[m['cohort']].astype('string').fillna('').str.strip() if m.get('cohort') else pd.Series('explicit_single_recipe', index=frame.index)
    cuts = runs.ne(runs.shift()) | phases.ne(phases.shift()) | recipes.ne(recipes.shift())
    pieces = cuts.cumsum()
    run_blocks = runs.loc[runs.ne(runs.shift())]
    ambiguous = set(run_blocks[run_blocks.duplicated(keep=False)])
    changed_recipe = recipes.groupby(runs).nunique()
    bad_runs = ambiguous | set(changed_recipe[changed_recipe > 1].index)
    backwards = stamps.groupby(runs).diff().dt.total_seconds().le(0)
    bad_runs |= set(runs.loc[backwards])
    cadence_table = runtime.prepared.tables['cadence'].set_index('step')
    cases = []; excluded = []
    for _, group in frame.groupby(pieces, sort=False):
        indices = group.index.to_numpy(); first = indices[0]; phase = phases.loc[first]
        if runs.loc[first] in bad_runs:
            excluded.append(dict(case_id=case_id, step=phase, reason='ambiguous_recipe_run_or_time', rows=len(group))); continue
        if not runs.loc[first] or not steps.loc[first] or not recipes.loc[first]:
            excluded.append(dict(case_id=case_id, step=phase, reason='missing_input_identity', rows=len(group))); continue
        if phase not in runtime.encoders:
            excluded.append(dict(case_id=case_id, step=phase, reason='unsupported_phase', rows=len(group))); continue
        try:
            fast._steps(runtime, [phase])
        except ValueError:
            excluded.append(dict(case_id=case_id, step=phase, reason='untrained_target_head', rows=len(group))); continue
        encoder, meta = runtime.encoders[phase]
        length = int(meta['window_length']); names = list(meta['sensor_columns'])
        chosen = names if centers is None else [n for n in names if n in centers]
        if not chosen:
            excluded.append(dict(case_id=case_id, step=phase, reason='no_selected_sensor', rows=len(group))); continue
        if length < 2*encoder.patch_len or length % encoder.patch_len:
            raise ValueError('Frozen native window/patch contract invalid')
        mean = np.asarray(meta['mean']); scale = np.asarray(meta['std'])
        if mean.shape != (len(names),) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Frozen TRAIN normalization invalid')
        # Match EXP6's raw float32 cast before TRAIN normalization. Preserve
        # frame's original precision separately for exact changed-cell truth.
        values = group[names].to_numpy(dtype=np.float32)
        valid = np.isfinite(values).all(1) & stamps.loc[indices].notna().to_numpy()
        dt = stamps.loc[indices].diff().dt.total_seconds().to_numpy()
        cadence = float(meta['cadence_seconds']); tolerance = float(cadence_table.loc[phase, 'tolerance_seconds'])
        bad_time = np.isfinite(dt) & (dt <= 0)
        if bad_time.any():
            excluded.append(dict(case_id=case_id, step=phase, reason='nonmonotonic_occurrence_time', rows=len(group))); continue
        break_before = np.ones(len(group), dtype=bool)
        break_before[1:] = ~valid[:-1] | ~valid[1:] | (np.abs(dt[1:]-cadence) > tolerance)
        coverage = np.zeros(len(group), dtype=bool)
        for block in np.unique(np.cumsum(break_before)):
            loc = np.flatnonzero((np.cumsum(break_before) == block) & valid)
            advance = length if stride is None else stride
            if not isinstance(advance, int) or not 1 <= advance <= length:
                raise ValueError('Explicit stride must be an integer in 1..frozen window length')
            for start in range(0, len(loc)-length+1, advance):
                local = loc[start:start+length]; rows = indices[local]
                offset = int(local[0])
                tick = int(round((stamps.loc[rows[0]]-stamps.loc[first]).total_seconds()/cadence)) if pd.notna(stamps.loc[first]) else offset
                cases.append(dict(case_id=case_id, pair_id=f'{case_id}/{int(rows[0])}:{int(rows[-1])+1}',
                    step=phase, current=((values[local]-mean)/scale).astype(np.float32),
                    centers=chosen, row_start=int(rows[0]), row_end=int(rows[-1])+1,
                    position=offset/max(1, len(group)-1), run_id=peald._hash(runs.loc[first]),
                    cohort=(peald._hash(recipes.loc[first]), offset, tick),
                    names=names, mean=mean, scale=scale, raw_current=values[local].copy(),
                    file_edge_context_review=bool(first == 0 or indices[-1] == len(frame)-1)))
                coverage[local] = True
        if (~valid).any():
            excluded.append(dict(case_id=case_id, step=phase, reason='invalid_sensor_or_timestamp_rows', rows=int((~valid).sum())))
        if (valid & ~coverage).any():
            excluded.append(dict(case_id=case_id, step=phase, reason='short_piece_or_unused_tail', rows=int((valid & ~coverage).sum())))
    return cases, excluded


def train_reference(runtime, case):
    """Frozen TRAIN bank by recipe, Step and native offset; no normality claim."""
    matches = {}
    recipe, offset, tick = case['cohort']
    for row in runtime.prepared.windows[case['step']]['train']:
        key = row['cohort']
        if key[0] == recipe and key[2:] == (offset, tick) and row['run_id'] != case['run_id']:
            matches.setdefault(row['run_id'], row['current'])
    refs = list(matches.values())[:16]
    return np.median(np.stack(refs), axis=0) if refs else None


def _fingerprint(runtime):
    h = hashlib.sha256(); seen = set()
    models = [runtime.module.GEVIS_MODEL, fa.require_head(runtime.module)]
    models += [e for e, _ in runtime.encoders.values()]
    models += list(getattr(runtime, 'step_heads', {}).values())
    for model in models:
        if id(model) in seen:
            continue
        seen.add(id(model))
        for key, value in model.state_dict().items():
            h.update(key.encode()); h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _paired_metrics(rows):
    boolean = rows.loc[rows.answer_type.eq('BOOLEAN') & rows.paired_boolean_label.notna()].copy()
    if boolean.empty:
        return pd.DataFrame(), {}
    boolean['label'] = boolean.paired_boolean_label.astype(int)
    scored = fast._report('Supplied CSV versus exact original: synthetic change only', boolean.to_dict('records'))
    metric = scored.diagnostics['boolean_metrics']
    metric['fpr_available'] = metric.negative_support.gt(0)
    metric.loc[~metric.fpr_available, 'false_positive_rate'] = np.nan
    return scored.summary, scored.diagnostics


def evaluate(contract, paths, *, reference_mode='train_bank', original_path=None,
             score_original=False, centers=None, stride=None, center_batch_size=8,
             progress=None, session=None, max_windows_per_file=None):
    """Replay native requests; optional original truth is attached after forward."""
    if reference_mode not in ('paired_original', 'train_bank'):
        raise ValueError('Choose train_bank or explicit paired_original diagnostic')
    if max_windows_per_file is not None and (type(max_windows_per_file) is not int or max_windows_per_file < 1):
        raise ValueError('max_windows_per_file must be a positive integer or None')
    paths = [Path(p) for p in paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError('Unique nonempty input paths required')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Unique case IDs required')
    if centers is not None and (not centers or len(set(centers)) != len(centers)):
        raise ValueError('Unique nonempty selected sensors required')
    if not 1 <= center_batch_size <= 32:
        raise ValueError('Bounded center batch size in 1..32 required')
    wants_truth = score_original or reference_mode == 'paired_original'
    if wants_truth and original_path is None:
        raise ValueError('Explicit unmodified original CSV required for scoring')
    runtime = contract.runtime; fast.require_runtime(runtime)
    model_before = _fingerprint(runtime)
    runner = session or online.InferenceSession(runtime, center_batch_size=center_batch_size)
    if runner.runtime is not runtime:
        raise ValueError('Session belongs to a different model runtime')
    originals = None
    if progress is not None:
        progress('Frozen model and TRAIN reference index ready; starting file replay', flush=True)
    if reference_mode == 'paired_original':
        originals = original_lookup(paths, original_path, contract)
    rows = []; coverage = []; excluded = []; regions = []; file_rows = []; requests = []
    for file_index, path in enumerate(paths):
        if progress is not None:
            progress(f'File {file_index+1}/{len(paths)}: reading and preparing native windows', flush=True)
        before = peald.digest(path); started = time.perf_counter(); frame = read_input(path, contract)
        cases, rejected = windows(frame, contract, path.stem, centers=centers, stride=stride)
        available_windows = len(cases)
        if max_windows_per_file is not None:
            cases = cases[:max_windows_per_file]
        excluded.extend(rejected)
        preparation_ms = 1000*(time.perf_counter()-started)
        overlap = _overlap(runtime, [c['run_id'] for c in cases])
        reference = None; source_group = None; source_status = 'not_requested'
        if reference_mode == 'paired_original':
            try:
                reference, source_group, _ = align_original(frame, originals, contract)
            except ValueError as error:
                coverage.append(dict(case_id=path.stem, status='NOT_EVALUATED',
                    reason=str(error), input_rows=len(frame), windows=0))
                continue
        first_answer_index = len(rows); completed = 0
        for case_index, case in enumerate(cases):
            if reference_mode == 'paired_original':
                a, b = case['row_start'], case['row_end']
                raw = reference.iloc[a:b][case['names']].to_numpy(dtype=np.float32)
                pair = dict(case, reference=((raw-case['mean'])/case['scale']).astype(np.float32))
                timing.synchronize(runtime, case['step']); start = time.perf_counter()
                answers, stages = online.predict_pair(runtime, pair, case['centers'], center_batch_size=center_batch_size)
                timing.synchronize(runtime, case['step'])
                result = dict(status='OK', reason=None, answers=answers,
                    timings=dict(stages, request_ms=1000*(time.perf_counter()-start),
                                 normalization_ms=0., reference_ms=0.))
            else:
                result = runner.process_window(current=case['raw_current'], step=case['step'],
                    cohort=case['cohort'], run_id=case['run_id'], centers=case['centers'],
                    position=case['position'])
            requests.append(dict(request_id=case['pair_id'], case_id=path.stem,
                step=case['step'], status=result['status'], reason=result['reason'],
                selected_sensors=result.get('selected_sensors', len(case['centers']) if result['status'] == 'OK' else 0),
                answers=len(result['answers']),
                **result['timings']))
            if result['status'] != 'OK':
                excluded.append(dict(case_id=path.stem, step=case['step'],
                    reason=result['reason'], rows=case['row_end']-case['row_start']))
            else:
                completed += 1
            rows.extend(dict(answer, case_id=path.stem, pair_id=case['pair_id'],
                step=case['step'], row_start=case['row_start'], row_end=case['row_end'],
                run_id=case['run_id'], source_group=None, overlap_status=overlap,
                file_edge_context_review=case['file_edge_context_review'],
                paired_boolean_label=None) for answer in result['answers'])
            if progress is not None and (case_index == 0 or (case_index+1) % 25 == 0 or case_index+1 == len(cases)):
                progress(f'File {file_index+1}/{len(paths)}: windows {case_index+1}/{len(cases)}, answered {completed}', flush=True)
        # Exact originals are now evaluator-only for the default request path.
        # Missing truth never removes predictions or changes the TRAIN reference.
        if wants_truth:
            if originals is None:
                if progress is not None:
                    progress('Predictions ready for first file; building evaluator-only original index', flush=True)
                originals = original_lookup(paths, original_path, contract)
            try:
                reference, source_group, _ = align_original(frame, originals, contract)
                source_status = 'exact_original_aligned'
            except ValueError as error:
                source_status = str(error)
        case_answers = [r for r in rows[first_answer_index:] if r['answer_type'] == 'BOOLEAN']
        if reference is not None:
            _, sensors = _columns(contract)
            difference = frame[sensors].to_numpy(float) != reference[sensors].to_numpy(float)
            difference[~np.isfinite(frame[sensors].to_numpy(float)).all(1)] = False
            for row in rows[first_answer_index:]:
                row['source_group'] = source_group
                if row['answer_type'] == 'BOOLEAN':
                    row['paired_boolean_label'] = int(difference[row['row_start']:row['row_end'], sensors.index(row['center_sensor'])].any())
            by_sensor = {}
            for row in case_answers:
                by_sensor.setdefault(row['center_sensor'], []).append(row)
            for sid, sensor in enumerate(sensors):
                indices = np.flatnonzero(difference[:, sid])
                for block in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1):
                    if len(block):
                        relevant = [r for r in by_sensor.get(sensor, [])
                            if r['row_start'] < block[-1]+1 and r['row_end'] > block[0]]
                        regions.append(dict(case_id=path.stem, source_group=source_group,
                            sensor=sensor, row_start=int(block[0]), row_end=int(block[-1])+1,
                            changed_rows=len(block), evaluated=bool(relevant),
                            hit=any(r['prediction'] == 1 for r in relevant)))
        coverage.append(dict(case_id=path.stem, source_group=source_group,
            status='EVALUATED' if completed else 'NOT_EVALUATED', input_rows=len(frame),
            windows=completed, available_windows=available_windows,
            windows_not_requested_by_limit=available_windows-len(cases),
            overlap_status=overlap, preparation_ms=preparation_ms,
            truth_status=source_status))
        file_rows.append(dict(case_id=path.stem, input_sha256=before))
        if peald.digest(path) != before:
            raise ValueError('Input changed during evaluation')
    if _fingerprint(runtime) != model_before:
        raise RuntimeError('Frozen model weights changed; reject results')
    if originals is not None and peald.digest(original_path) != originals['source_sha256']:
        raise ValueError('Original changed during evaluation')
    details = pd.DataFrame(rows)
    summary = pd.DataFrame([dict(domain=contract.domain, files=len(paths), question_rows=len(rows), accuracy_available=False)])
    latency, by_step = online.latency_tables(requests, domain=contract.domain)
    diagnostics = dict(coverage=pd.DataFrame(coverage), exclusions=pd.DataFrame(excluded),
        changed_regions=pd.DataFrame(regions), input_identity=pd.DataFrame(file_rows),
        requests=pd.DataFrame(requests), request_latency=latency, request_latency_by_step=by_step)
    if rows:
        diagnostics['answer_consistency'] = fast._consistency(rows)
        if details.paired_boolean_label.notna().any():
            summary, metrics = _paired_metrics(details)
            diagnostics.update(metrics); summary['domain'] = contract.domain
            for field, name in [('case_id', 'boolean_by_file'), ('source_group', 'boolean_by_source_group'), ('overlap_status', 'boolean_by_overlap')]:
                tables = []
                for key, group in details.groupby(field, sort=False):
                    _, metric = _paired_metrics(group)
                    if metric:
                        tables.append(metric['boolean_metrics'].assign(**{field:key}))
                diagnostics[name] = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
            if reference_mode == 'train_bank':
                target = 'synthetic injection sensitivity against frozen TRAIN reference; untouched is not certified normal'
                summary['metric_target'] = target
                for key, table in list(diagnostics.items()):
                    if isinstance(table, pd.DataFrame):
                        diagnostics[key] = table.rename(columns={'false_positive_rate':'unchanged_alarm_rate',
                            'fp':'unchanged_alarms','tn':'unchanged_no_alarm'})
                        if 'metric_target' in diagnostics[key]:
                            diagnostics[key]['metric_target'] = target
                        if 'control' in diagnostics[key]:
                            diagnostics[key]['control'] = diagnostics[key]['control'].replace(
                                'all_known_unchanged_centers', 'injection_untouched_centers')
                if 'boolean_false_positives_by_step' in diagnostics:
                    diagnostics['unchanged_alarms_by_step'] = diagnostics.pop('boolean_false_positives_by_step')
        else:
            details = details.drop(columns='paired_boolean_label')
    diagnostics['denominators'] = pd.DataFrame([dict(domain=contract.domain, input_files=len(paths),
        evaluated_files=sum(c['status'] == 'EVALUATED' for c in coverage),
        exact_original_groups=len({c.get('source_group') for c in coverage if c.get('source_group')}),
        model_windows=sum(c.get('windows', 0) for c in coverage), question_rows=len(rows),
        labelled_boolean_questions=sum(r['answer_type'] == 'BOOLEAN' and r['paired_boolean_label'] is not None for r in rows),
        type_strength_labels_available=False, independent_test_certified=False)])
    if progress is not None:
        progress(f'Finished: {len(paths)} files, {len(requests)} input windows, {len(rows)} answers', flush=True)
    return ExperimentReport('Frozen TRAIN-reference replay; evaluator truth attached after prediction',
        summary, details=details, diagnostics=diagnostics,
        metadata=dict(domain=contract.domain, decision_system=fa.SYSTEM, model_evaluated=bool(rows),
            max_windows_per_file=max_windows_per_file,
            evaluation_scope='bounded_preview' if max_windows_per_file is not None else 'selected_files_all_windows',
            reference_mode=reference_mode, inference_recipe=online.RECIPE,
            original_used_for_prediction=reference_mode == 'paired_original',
            truth_used_after_prediction=wants_truth and reference_mode == 'train_bank',
            weights_preserved=True, model_sha256=model_before, automatic_export=False,
            test_used_for_fitting=False, test_used_for_selection=False, independent_test_certified=False,
            real_fault_truth=False, external_type_strength_scored=False, cycle_KPI_verified=False,
            processing_unit='one complete native Step model window; all selected sensors and three answers each',
            timing_scope='request rows in diagnostics.requests; normalization/reference/encoder/Fusion/head/result included; startup, disk, segmentation and transport excluded',
            limitation='Synthetic injection masks do not prove actual fault truth. Unchanged alarms under TRAIN reference are not verified fault false positives. Original-paired mode is an optional oracle diagnostic. No chronological independence or physical-cycle/Edge KPI is certified.'))


def score_reviewed_labels(report, labels, *, truth_source):
    """Join independently reviewed answer labels after prediction, never refit."""
    keys = ['case_id', 'row_start', 'row_end', 'center_sensor', 'answer_type']
    if not isinstance(truth_source, str) or not truth_source.strip():
        raise ValueError('Explicit independent truth provenance required')
    if not set(keys+['label']) <= set(labels) or labels.empty or labels.duplicated(keys).any():
        raise ValueError('Unique nonempty question-level reviewed labels required')
    if report.details.empty or report.details.duplicated(keys).any():
        raise ValueError('Unique saved predictions required before scoring')
    joined = labels[keys+['label']].merge(report.details, on=keys, how='left', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Reviewed label has no exact predicted question coordinate')
    result = fast._report('Independently reviewed saved new-input predictions', joined.drop(columns='_merge').to_dict('records'))
    target = 'independently reviewed fixed-question answer; not certified equipment fault performance'
    for table in [result.summary, result.by_step, *result.diagnostics.values()]:
        if isinstance(table, pd.DataFrame) and 'metric_target' in table:
            table['metric_target'] = target
    result.metadata.update(truth_source=truth_source, labelled_questions=len(joined),
        metric_target=target, unlabelled_questions=len(report.details)-len(joined), automatic_export=False)
    return result
