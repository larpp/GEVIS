"""Three-answer Fast assessment from one saved report, never inference or fitting."""
import numpy as np
import pandas as pd


KINDS = ('BOOLEAN', 'CHOICE', 'ORDINAL')


def _row(table, kind):
    if not isinstance(table, pd.DataFrame) or 'answer_type' not in table:
        return {}
    rows = table.loc[table.answer_type.eq(kind)]
    if len(rows) > 1:
        raise ValueError('Choose one model/variant/evaluation scope before assessment')
    return rows.iloc[0].to_dict() if len(rows) else {}


def assess(report, *, evidence_scope):
    """No cross-report quality/timing merge or final product acceptance claim.

    All answer families remain required. The project's F1 goal is a reference;
    its per-family averaging/acceptance definitions require a fixed protocol.
    """
    if not isinstance(evidence_scope, str) or not evidence_scope.strip():
        raise ValueError('Name the actual evaluation scope')
    diagnostics = report.diagnostics
    quality = []
    for kind in KINDS:
        summary = _row(report.summary, kind)
        changed = _row(diagnostics.get('changed_only_metrics'), kind)
        binary = _row(diagnostics.get('boolean_metrics'), kind) if kind == 'BOOLEAN' else {}
        ordinal = _row(diagnostics.get('ordinal_metrics'), kind) if kind == 'ORDINAL' else {}
        metric = 'positive_f1' if kind == 'BOOLEAN' else 'macro_f1'
        score = binary.get('f1', np.nan) if kind == 'BOOLEAN' else summary.get('macro_f1', np.nan)
        questions = int(summary.get('questions', binary.get('questions', 0)))
        available = bool(questions and np.isfinite(score))
        status = 'measured' if available else 'truth_or_scored_metric_unavailable'
        if available and (changed.get('changed_questions') == 0 or
                          (kind == 'BOOLEAN' and (binary.get('positive_support', 0) == 0 or
                                                 binary.get('negative_support', 0) == 0))):
            available = False
            status = 'insufficient_changed_or_negative_truth'
        quality.append(dict(answer_type=kind, required=True, questions=questions,
            comparison_metric=metric, f1=score, metric_available=available,
            project_f1_reference=0.96,
            meets_reference=bool(score >= .96) if available else None,
            answer_accuracy=summary.get('answer_accuracy', np.nan),
            changed_questions=changed.get('changed_questions', np.nan),
            changed_accuracy=changed.get('changed_accuracy', np.nan),
            ordinal_mae=ordinal.get('mean_absolute_step_error', np.nan),
            within_one_step_accuracy=ordinal.get('within_one_step_accuracy', np.nan),
            status=status, evidence_scope=evidence_scope,
            acceptance_status='per_answer_acceptance_definition_required'))
    requests = diagnostics.get('requests', pd.DataFrame())
    timing = dict(attempted_requests=len(requests), measured_requests=0,
        failed_or_unmeasured_requests=len(requests), mean_ms=np.nan, p95_ms=np.nan,
        p99_ms=np.nan, fraction_over_50ms=np.nan, mean_within_50ms=None,
        project_mean_goal_ms=50., three_answers_per_sensor_count_check=None,
        evidence_scope=evidence_scope, acceptance_status='cycle_and_Edge_boundary_not_certified')
    if not requests.empty and {'status', 'request_ms'} <= set(requests):
        ok = requests.loc[requests.status.eq('OK')].copy()
        values = pd.to_numeric(ok.request_ms, errors='coerce')
        valid = np.isfinite(values) & values.ge(0)
        measured = values.loc[valid]
        timing.update(measured_requests=len(measured),
                      failed_or_unmeasured_requests=len(requests)-len(measured))
        if len(measured):
            timing.update(mean_ms=float(measured.mean()), p95_ms=float(measured.quantile(.95)),
                p99_ms=float(measured.quantile(.99)),
                fraction_over_50ms=float(measured.gt(50).mean()),
                mean_within_50ms=bool(measured.mean() <= 50))
        if len(ok) and {'answers', 'selected_sensors'} <= set(ok):
            timing['three_answers_per_sensor_count_check'] = bool(
                ok.selected_sensors.gt(0).all() and ok.answers.eq(3*ok.selected_sensors).all())
    complete = timing['three_answers_per_sensor_count_check']
    timing['three_answer_mean_reference_met'] = (
        bool(complete and timing['mean_within_50ms'])
        if complete is not None and timing['mean_within_50ms'] is not None else None)
    metadata = report.metadata
    scope = pd.DataFrame([dict(evidence_scope=evidence_scope,
        domain=metadata.get('domain', 'see report provenance'),
        required_answers='BOOLEAN, CHOICE, ORDINAL in the same Fast request',
        measured_scope=metadata.get('timing_scope', 'not supplied'),
        cycle_equivalence='model window to physical cycle mapping requires validation',
        quality_and_latency_same_protocol='requires fixed input/output/device/independent-truth protocol',
        precision_scope='future continuous-value forecasting, not CHOICE/ORDINAL',
        overall_acceptance='not_certified')])
    return dict(quality=pd.DataFrame(quality), request_latency=pd.DataFrame([timing]), scope=scope)


def show(report, display, *, evidence_scope):
    tables = assess(report, evidence_scope=evidence_scope)
    for name, table in tables.items():
        print('Fast three-answer assessment:', name)
        display(table)
    return tables


def compact(report, display, *, evidence_scope, predictions=0, show_details=False):
    """Purpose-first notebook view; no extra evaluation, plots or inference."""
    tables = assess(report, evidence_scope=evidence_scope)
    quality = tables['quality']
    if quality.questions.gt(0).any():
        display(quality[['answer_type', 'questions', 'comparison_metric', 'f1',
            'answer_accuracy', 'changed_accuracy', 'ordinal_mae', 'status']])
    timing = tables['request_latency']
    if timing.attempted_requests.gt(0).any():
        display(timing[['attempted_requests', 'measured_requests', 'failed_or_unmeasured_requests',
            'mean_ms', 'p95_ms', 'p99_ms', 'fraction_over_50ms', 'three_answers_per_sensor_count_check']])
    if predictions and not report.details.empty and 'answer_type' in report.details:
        columns = [c for c in ('case_id', 'step', 'center_sensor', 'answer_type', 'answer') if c in report.details]
        display(report.details[columns].head(predictions))
    if not quality.questions.gt(0).any():
        print('Prediction/timing only: no accuracy or F1 without independent answer truth.')
    if show_details:
        display(tables['scope'])
        for name, table in report.diagnostics.items():
            if isinstance(table, pd.DataFrame):
                print(name); display(table.head(50))
    return tables
