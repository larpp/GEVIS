"""Evaluator-only event tables from saved predictions; no model or data writes."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(directory, *, columns=None):
    """Canonical columns are documented in the EXP7 guide; never guess identities."""
    directory = Path(directory)
    tables = {}
    for name in ('cases', 'events'):
        path = directory / (name + '.csv')
        if not path.is_file():
            raise FileNotFoundError(f'Missing evaluator truth: {path}')
        tables[name] = pd.read_csv(path).rename(columns=(columns or {}).get(name, {}))
    return tables['cases'], tables['events']


def score_saved(report, cases, events, *, choice_map=None):
    """Join held-out generation truth after forward, with explicit CHOICE mapping.

    Event hit means at least one overlapping BOOLEAN Yes, not full-span detection.
    ORDINAL values are reported as responses, never equated to external severity.
    """
    required = {'case_id', 'source_group', 'type_id', 'strength', 'is_control'}
    if not required <= set(cases):
        raise ValueError('cases requires columns: ' + ', '.join(sorted(required)))
    if not {'case_id', 'sensor', 'row_start', 'row_end'} <= set(events):
        raise ValueError('events requires case_id, sensor, row_start, row_end (exclusive)')
    if cases.case_id.isna().any() or cases.case_id.duplicated().any():
        raise ValueError('Each case_id must have exactly one generation record')
    cases = cases.copy()
    flags = cases.is_control.astype(str).str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False})
    if flags.isna().any() or cases.source_group.isna().any():
        raise ValueError('Explicit control flags and source groups required')
    cases['is_control'] = flags
    if cases.loc[~flags, ['type_id', 'strength']].isna().any().any():
        raise ValueError('Injected cases need generation type and strength')
    if not set(events.case_id) <= set(cases.case_id):
        raise ValueError('Event case_id missing from cases')
    if events.case_id.isin(cases.loc[flags, 'case_id']).any():
        raise ValueError('Control cases cannot contain injection events')
    bounds = events[['row_start', 'row_end']].to_numpy(dtype=float)
    if (not np.isfinite(bounds).all() or (bounds != np.floor(bounds)).any()
            or (bounds[:, 0] < 0).any() or (bounds[:, 1] <= bounds[:, 0]).any()):
        raise ValueError('Event coordinates must be nonnegative half-open row intervals')
    details = report.details
    keys = ['case_id', 'row_start', 'row_end', 'step', 'center_sensor', 'answer_type']
    if not set(keys + ['prediction']) <= set(details):
        raise ValueError('Saved predictions lack exact question coordinates')
    if details.duplicated(keys).any():
        raise ValueError('Duplicate saved question coordinates')
    if not set(details.case_id) <= set(cases.case_id):
        raise ValueError('Generation truth must identify every predicted case')
    choice_map = choice_map or {}
    if any(not isinstance(v, int) or v not in range(1, 6) for v in choice_map.values()):
        raise ValueError('Explicit CHOICE map values must be supported changed classes 1..5')
    lookup = cases.set_index('case_id')
    groups = dict(tuple(details.groupby(['case_id', 'center_sensor'], sort=False)))
    records = []
    for event_index, event in events.reset_index(drop=True).iterrows():
        meta = lookup.loc[event.case_id]
        group = groups.get((event.case_id, event.sensor), details.iloc[:0])
        span = group.loc[(group.row_start < event.row_end) & (group.row_end > event.row_start)]
        boolean = span.loc[span.answer_type.eq('BOOLEAN')]
        actual = boolean.iloc[:0]
        if 'paired_boolean_label' in boolean:
            actual = boolean.loc[boolean.paired_boolean_label.eq(1)]
        choice = span.loc[span.answer_type.eq('CHOICE')]
        ordinal = span.loc[span.answer_type.eq('ORDINAL')]
        target = choice_map.get(meta.type_id)
        # Only windows confirmed changed by exact-original masks can carry type truth.
        coords = actual[['row_start', 'row_end', 'step']]
        typed = choice.merge(coords, on=['row_start', 'row_end', 'step'])
        valid_type = target is not None and not typed.empty
        records.append(dict(event_id=event.get('event_id', event_index), case_id=event.case_id,
            source_group=meta.source_group, type_id=meta.type_id, strength=meta.strength,
            evaluated=not boolean.empty, overlapping_windows=len(boolean),
            hit=bool(boolean.prediction.eq(1).any()) if len(boolean) else None,
            actual_changed_windows=len(actual),
            changed_window_hits=int(actual.prediction.eq(1).sum()),
            choice_status='scored' if valid_type else ('unmapped_type' if target is None else 'no_changed_window'),
            choice_questions=len(typed) if valid_type else 0,
            choice_correct=int(typed.prediction.eq(target).sum()) if valid_type else 0,
            ordinal_mean=float(ordinal.prediction.mean()) if len(ordinal) else np.nan))
    event_table = pd.DataFrame(records)
    summaries = []
    if len(event_table):
        for identity, group in event_table.groupby(['type_id', 'strength'], sort=False):
            covered = group.loc[group.evaluated]
            ntype = int(group.choice_questions.sum())
            summaries.append(dict(type_id=identity[0], strength=identity[1],
                source_groups=group.source_group.nunique(), events=len(group),
                evaluated_events=len(covered), uncovered_events=len(group)-len(covered),
                event_hits=int(covered.hit.sum()),
                event_hit_rate=float(covered.hit.mean()) if len(covered) else np.nan,
                choice_questions=ntype,
                choice_accuracy=float(group.choice_correct.sum()/ntype) if ntype else np.nan,
                ordinal_mean_response=group.ordinal_mean.mean()))
    boolean = details.loc[details.answer_type.eq('BOOLEAN')].copy()
    boolean = boolean.drop(columns=['source_group'], errors='ignore').merge(
        cases[['case_id', 'source_group', 'is_control']], on='case_id', validate='many_to_one')
    match = ['source_group', 'step', 'row_start', 'row_end', 'center_sensor']
    controls = boolean.loc[boolean.is_control, match + ['prediction']]
    if controls.duplicated(match).any():
        raise ValueError('Ambiguous control predictions for one source/window/sensor')
    paired = boolean.loc[~boolean.is_control].merge(
        controls.rename(columns={'prediction': 'control_prediction'}),
        on=match, how='left', validate='many_to_one')
    comparisons = []
    for case_id, group in paired.groupby('case_id', sort=False):
        matched = group.loc[group.control_prediction.notna()]
        comparisons.append(dict(case_id=case_id, matched_questions=len(matched),
            unmatched_questions=len(group)-len(matched),
            control_alarm_rate=matched.control_prediction.mean(),
            injected_alarm_rate=matched.prediction.mean(),
            additional_alarm_rate=((matched.prediction == 1) & (matched.control_prediction == 0)).mean(),
            net_alarm_rate_change=(matched.prediction-matched.control_prediction).mean()))
    return dict(events=event_table, by_type_strength=pd.DataFrame(summaries),
        control_comparison=pd.DataFrame(comparisons),
        truth_coverage=pd.DataFrame([dict(generation_cases=len(cases),
            predicted_cases=details.case_id.nunique(), cases_without_predictions=len(set(cases.case_id)-set(details.case_id)),
            generation_events=len(events), ordinal_accuracy_available=False,
            limitation='Generation event spans; changed-cell masks remain separate. Event hits do not imply full-span recall.')] ))
