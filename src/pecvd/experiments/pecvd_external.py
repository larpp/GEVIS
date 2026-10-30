"""PECVD source-format CSV replay; labels are joined only after inference."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import external_test as external, fast_experiments as fast
from . import online_inference as online
from .reporting import ExperimentReport


def _onehot(frame, prefix, excluded=()):
    names = [c for c in frame if c.startswith(prefix) and c not in excluded]
    if not names:
        raise ValueError(f'PECVD input requires {prefix} metadata, not PEALD raw CSV')
    values = frame[names].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    if not np.isin(values, [0, 1]).all() or not (values.sum(1) == 1).all():
        raise ValueError(f'Ambiguous or missing {prefix} one-hot metadata')
    return np.array([n.removeprefix(prefix) for n in names])[values.argmax(1)]


def read_frame(path, runtime, config):
    header = pd.read_csv(path, nrows=0).columns
    sensors = list(dict.fromkeys(n for _, m in runtime.encoders.values() for n in m['sensor_columns']))
    run, step, order = config.columns.run, config.columns.step, config.columns.order
    required = [run, order, *sensors]
    missing = [n for n in required if n not in header]
    if missing:
        raise ValueError('PECVD source-format columns missing: ' + ', '.join(missing))
    metadata = [n for n in header if n.startswith(('source_group_', 'pm_cycle_', config.columns.step_one_hot_prefix))]
    columns = list(dict.fromkeys(required + metadata + ([step] if step in header else [])))
    frame = pd.read_csv(path, usecols=columns, dtype={run: 'string'}, low_memory=False)
    if frame.empty:
        raise ValueError('Empty PECVD input')
    if step not in frame:
        frame[step] = _onehot(frame, config.columns.step_one_hot_prefix, config.columns.step_one_hot_excluded)
    if frame[[run, step]].isna().any().any() or frame[run].str.strip().eq('').any():
        raise ValueError('Missing PECVD run/Step identity')
    frame[order] = pd.to_numeric(frame[order], errors='coerce')
    if (not np.isfinite(frame[order]).all() or frame[order].ne(np.floor(frame[order])).any()
            or frame.duplicated([run, order]).any()):
        raise ValueError('Invalid or duplicate PECVD run/order coordinates')
    if frame.groupby(run, sort=False)[order].diff().le(0).any():
        raise ValueError('PECVD input must preserve increasing run order; no automatic sort')
    blocks = frame.loc[frame[run].ne(frame[run].shift()), run]
    if blocks.duplicated().any():
        raise ValueError('PECVD run reappears across disjoint file intervals')
    frame['_cohort'] = _onehot(frame, 'source_group_', ('source_group_nan',)).astype(str)
    frame['_cohort'] += '|pm=' + _onehot(frame, 'pm_cycle_').astype(str)
    if frame.groupby(run)['_cohort'].nunique().gt(1).any():
        raise ValueError('PECVD source/PM context changes inside a run')
    for name in sensors:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    return frame


def windows(frame, runtime, config, case_id, *, steps=None, centers=None, stride=None):
    selected = fast._steps(runtime, steps)
    run, step, order = config.columns.run, config.columns.step, config.columns.order
    cuts = frame[run].ne(frame[run].shift()) | frame[step].ne(frame[step].shift())
    # Missing native rows split occurrences; never stitch across a gap.
    cuts |= frame.groupby(run, sort=False)[order].diff().ne(1)
    cases, exclusions = [], []
    for _, group in frame.groupby(cuts.cumsum(), sort=False):
        phase = group[step].iloc[0]
        if phase not in selected:
            exclusions.append(dict(case_id=case_id, step=phase, reason='unsupported_or_unselected_step', rows=len(group)))
            continue
        encoder, meta = runtime.encoders[phase]
        length = int(meta['window_length']); names = list(meta['sensor_columns'])
        chosen = names if centers is None else [n for n in centers if n in names]
        if not chosen:
            raise ValueError('No model-supported selected sensors')
        advance = length if stride is None else stride
        if not isinstance(advance, int) or not 1 <= advance <= length:
            raise ValueError('Stride must be in 1..native window length')
        if length % encoder.patch_len or length < 2 * encoder.patch_len:
            raise ValueError('Invalid frozen native window/patch contract')
        values = group[names].to_numpy(dtype=np.float32)
        covered = np.zeros(len(group), dtype=bool)
        for start in range(0, len(group)-length+1, advance):
            raw = values[start:start+length]
            if not np.isfinite(raw).all():
                exclusions.append(dict(case_id=case_id, step=phase, reason='nonfinite_window', rows=length))
                continue
            ratio = start / max(1, len(group)-length)
            position_bin = min(9, int(ratio * 10))
            a = int(group.index[start]); b = int(group.index[start+length-1])+1
            cases.append(dict(case_id=case_id, pair_id=f'{case_id}/{a}:{b}', step=phase,
                raw_current=raw, centers=chosen, run_id=str(group[run].iloc[0]),
                cohort=(str(group['_cohort'].iloc[0]), position_bin), position=position_bin / 9,
                row_start=a, row_end=b))
            covered[start:start+length] = True
        if (~covered).any():
            exclusions.append(dict(case_id=case_id, step=phase, reason='uncovered_short_tail_or_invalid_rows', rows=int((~covered).sum())))
    return cases, exclusions


def evaluate(runtime, config, paths, *, original_path=None, steps=None, centers=None,
             stride=None, center_batch_size=8, progress=None, max_windows_per_file=None):
    fast.require_runtime(runtime)
    if max_windows_per_file is not None and (type(max_windows_per_file) is not int or max_windows_per_file < 1):
        raise ValueError('max_windows_per_file must be a positive integer or None')
    if config.process_family != 'PECVD' or hasattr(runtime, 'prepared'):
        raise ValueError('EXP9 requires source PECVD runtime/config, not PEALD transfer weights')
    if original_path is not None and not Path(original_path).is_file():
        raise FileNotFoundError('EXP9 evaluator original CSV is missing')
    paths = [Path(p) for p in paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths) or len({p.stem for p in paths}) != len(paths):
        raise ValueError('Nonempty unique PECVD input paths/case IDs required')
    before = external._fingerprint(runtime)
    session = online.InferenceSession(runtime, center_batch_size=center_batch_size)
    rows, requests, coverage, excluded = [], [], [], []
    original = None; original_hash = None
    run, step, order = config.columns.run, config.columns.step, config.columns.order
    for index, path in enumerate(paths):
        if progress:
            progress(f'PECVD file {index+1}/{len(paths)}: prepare and replay', flush=True)
        digest = external.peald.digest(path)
        frame = read_frame(path, runtime, config)
        cases, rejects = windows(frame, runtime, config, path.stem, steps=steps, centers=centers, stride=stride)
        available_windows = len(cases)
        if max_windows_per_file is not None:
            cases = cases[:max_windows_per_file]
        excluded.extend(rejects); first = len(rows)
        overlap = external._overlap(runtime, [c['run_id'] for c in cases])
        for i, case in enumerate(cases):
            result = session.process_window(current=case['raw_current'], step=case['step'],
                cohort=case['cohort'], run_id=case['run_id'], centers=case['centers'], position=case['position'])
            requests.append(dict(case_id=path.stem, step=case['step'], request_id=case['pair_id'],
                status=result['status'], reason=result['reason'], answers=len(result['answers']),
                selected_sensors=len(case['centers']), **result['timings']))
            rows.extend(dict(a, case_id=path.stem, pair_id=case['pair_id'], step=case['step'],
                row_start=case['row_start'], row_end=case['row_end'], run_id=case['run_id'],
                overlap_status=overlap, paired_boolean_label=None) for a in result['answers'])
            if progress and (i == 0 or (i+1) % 25 == 0 or i+1 == len(cases)):
                progress(f'PECVD file {index+1}/{len(paths)}: windows {i+1}/{len(cases)}', flush=True)
        truth_status = 'not_requested'
        # Original values cannot influence any forward call or TRAIN reference.
        if original_path is not None:
            if original is None:
                original_hash = external.peald.digest(original_path)
                original = read_frame(original_path, runtime, config)
            keys = [run, step, order, '_cohort']
            if original.duplicated(keys).any():
                raise ValueError('Ambiguous original run/Step/order coordinates')
            source = original.set_index(keys)
            coords = pd.MultiIndex.from_frame(frame[keys])
            if coords.isin(source.index).all():
                aligned = source.reindex(coords)
                truth_status = 'exact_original_aligned'
                for row in rows[first:]:
                    if row['answer_type'] != 'BOOLEAN':
                        continue
                    a, b, sensor = row['row_start'], row['row_end'], row['center_sensor']
                    x = frame[sensor].iloc[a:b].to_numpy(float)
                    y = aligned[sensor].iloc[a:b].to_numpy(float)
                    if np.isfinite(x).all() and np.isfinite(y).all():
                        row['paired_boolean_label'] = int((x != y).any())
            else:
                truth_status = 'original_coordinates_unavailable'
        coverage.append(dict(case_id=path.stem, windows=len(cases), answers=len(rows)-first,
            available_windows=available_windows, windows_not_requested_by_limit=available_windows-len(cases),
            input_rows=len(frame), overlap_status=overlap, truth_status=truth_status,
            file_edge_context_review=True))
        if external.peald.digest(path) != digest:
            raise ValueError('PECVD input changed during evaluation')
    if external._fingerprint(runtime) != before:
        raise RuntimeError('Model weights changed during EXP9')
    if original_hash and external.peald.digest(original_path) != original_hash:
        raise ValueError('Original changed during evaluation')
    details = pd.DataFrame(rows)
    latency, by_step = online.latency_tables(requests, domain='PECVD')
    summary = pd.DataFrame([dict(domain='PECVD', accuracy_available=False, question_rows=len(rows))])
    diagnostics = dict(requests=pd.DataFrame(requests), request_latency=latency,
        request_latency_by_step=by_step, coverage=pd.DataFrame(coverage), exclusions=pd.DataFrame(excluded))
    diagnostics['denominators'] = pd.DataFrame([dict(domain='PECVD', input_files=len(paths),
        files_with_answers=sum(c['answers'] > 0 for c in coverage), attempted_windows=len(requests),
        answered_windows=sum(r['status'] == 'OK' for r in requests), question_rows=len(rows),
        labelled_boolean_questions=sum(r['answer_type'] == 'BOOLEAN' and r['paired_boolean_label'] is not None for r in rows),
        independent_test_certified=False)])
    if rows:
        diagnostics['answer_consistency'] = fast._consistency(rows)
        if details.paired_boolean_label.notna().any():
            summary, metrics = external._paired_metrics(details)
            summary['metric_target'] = 'injection sensitivity against frozen PECVD TRAIN reference; unchanged is not certified normal'
            for key, table in metrics.items():
                diagnostics[key] = table.rename(columns={'fp':'unchanged_alarms', 'tn':'unchanged_no_alarm',
                    'false_positive_rate':'unchanged_alarm_rate'}) if isinstance(table, pd.DataFrame) else table
    return ExperimentReport('PECVD external input TRAIN-reference replay', summary, details=details,
        diagnostics=diagnostics, metadata=dict(domain='PECVD', model_sha256=before,
            max_windows_per_file=max_windows_per_file,
            evaluation_scope='bounded_preview' if max_windows_per_file is not None else 'selected_files_all_windows',
            reference_mode='train_bank', original_used_for_prediction=False, weights_preserved=True,
            independent_test_certified=False, truth_used_after_prediction=original_path is not None,
            ordinal_accuracy_available=False, automatic_export=False, cycle_KPI_verified=False,
            timing_scope='native complete-window request; excludes disk, startup, segmentation and transport',
            context_limit='Position uses supplied contiguous Step extent; truncated Step boundaries require review.'))
