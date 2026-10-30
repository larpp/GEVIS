"""Score stored Fusion answers only; never fit or change model predictions."""
import numpy as np
import pandas as pd

from pecvd.answer_display import FUSION_ANSWER_CRITERIA as ANSWER_CRITERIA


METRIC_TARGET = 'known paired perturbation answer; not actual equipment fault'
METRIC_POLICY = (
    'Macro averages include classes present in truth or predictions; classes absent '
    'from both are excluded. Undefined class ratios use zero; supports are shown. '
    'Ordinal MAE is synthetic strength-step error, not physical severity error.'
)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.


def negative_metrics(frame, **identity):
    """Only independently labelled unchanged selected-center examples count."""
    negative = frame.loc[frame.label.eq(0)]
    fp = int(negative.prediction.ne(0).sum())
    support = len(negative)
    return dict(identity, negative_support=support, fp=fp, tn=support - fp,
                false_positive_rate=fp / support if support else float('nan'),
                fpr_available=bool(support), metric_target=METRIC_TARGET)


def _score(group, kind):
    options = ANSWER_CRITERIA[kind]
    raw = group[['label', 'prediction']].to_numpy(dtype=float)
    if (not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()
            or (raw < 0).any() or (raw >= len(options)).any()):
        raise ValueError('Answer labels/predictions must be finite supported class indices')
    truth, prediction = raw.astype(np.int64).T
    matrix = np.zeros((len(options), len(options)), dtype=np.int64)
    np.add.at(matrix, (truth, prediction), 1)
    support, predicted = matrix.sum(1), matrix.sum(0)
    classes = []
    for i, answer in enumerate(options):
        tp = int(matrix[i, i])
        fp, fn = int(predicted[i] - tp), int(support[i] - tp)
        classes.append(dict(answer_type=kind, class_index=i, answer=answer,
            support=int(support[i]), predicted=int(predicted[i]), tp=tp, fp=fp, fn=fn,
            precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
            f1=_ratio(2 * tp, 2 * tp + fp + fn),
            included_in_macro=bool(support[i] + predicted[i])))
    active = [c for c in classes if c['included_in_macro']]
    summary = dict(answer_type=kind, questions=len(group),
        answer_accuracy=float(np.trace(matrix) / len(group)),
        majority_baseline_accuracy=float(support.max() / len(group)),
        macro_precision=float(np.mean([c['precision'] for c in active])),
        macro_recall=float(np.mean([c['recall'] for c in active])),
        macro_f1=float(np.mean([c['f1'] for c in active])),
        weighted_f1=float(sum(c['support'] * c['f1'] for c in classes) / len(group)),
        evaluated_classes=len(active), metric_target=METRIC_TARGET)
    confusion = [dict(answer_type=kind, true_class=i, true_answer=options[i],
        predicted_class=j, predicted_answer=options[j], count=int(matrix[i, j]))
        for i in range(len(options)) for j in range(len(options))]
    return summary, classes, confusion, matrix, truth, prediction


def refresh_report(report):
    """Recompute from saved predictions; preserve unlabelled cross-run reports."""
    if report.metadata.get('decision_system') != 'fusion_answers':
        return report
    details = report.details
    if 'label' not in details:
        return report
    if details.empty or not {'answer_type', 'prediction'} <= set(details):
        raise ValueError('Nonempty typed answer predictions required for scoring')
    if not set(details.answer_type) <= set(ANSWER_CRITERIA):
        raise ValueError('Unknown Fusion answer type')
    summaries, classes, confusion, binary, ordinal, changed_rows = [], [], [], [], [], []
    for kind, group in details.groupby('answer_type', sort=False):
        summary, rows, counts, matrix, truth, prediction = _score(group, kind)
        summaries.append(summary); classes.extend(rows); confusion.extend(counts)
        changed = truth != 0
        changed_rows.append(dict(answer_type=kind, changed_questions=int(changed.sum()),
            changed_accuracy=float((truth[changed] == prediction[changed]).mean()) if changed.any() else float('nan'),
            available=bool(changed.any()), metric_target=METRIC_TARGET))
        if kind == 'BOOLEAN':
            tn, fp, fn, tp = (int(x) for x in matrix.ravel())
            binary.append(dict(answer_type=kind, questions=len(group), tp=tp, fp=fp,
                fn=fn, tn=tn, positive_support=tp + fn, negative_support=tn + fp,
                precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                false_positive_rate=fp/(fp+tn) if fp+tn else float('nan'),
                fpr_available=bool(fp+tn), f1=_ratio(2*tp, 2*tp+fp+fn)))
        if kind == 'ORDINAL':
            error = np.abs(truth - prediction)
            ordinal.append(dict(answer_type=kind, questions=len(group),
                mean_absolute_step_error=float(error.mean()),
                within_one_step_accuracy=float((error <= 1).mean()),
                maximum_step_error=int(error.max()),
                metric_target='synthetic perturbation strength steps; not physical severity'))
    report.summary = pd.DataFrame(summaries)
    if 'step' in details:
        report.by_step = pd.DataFrame([dict(step=step, **_score(group, kind)[0])
            for (step, kind), group in details.groupby(['step', 'answer_type'], sort=False)])
    report.diagnostics.update(per_class=pd.DataFrame(classes), confusion=pd.DataFrame(confusion),
        boolean_metrics=pd.DataFrame(binary), ordinal_metrics=pd.DataFrame(ordinal),
        changed_only_metrics=pd.DataFrame(changed_rows))
    boolean_rows = details.loc[details.answer_type.eq('BOOLEAN')]
    by_step = []
    if 'step' in boolean_rows:
        by_step = [negative_metrics(g, step=step) for step, g in boolean_rows.groupby('step', sort=False)]
    controls = []
    if not boolean_rows.empty:
        controls.append(negative_metrics(boolean_rows, control='all_known_unchanged_centers'))
        if 'kind' in boolean_rows:
            controls.append(negative_metrics(boolean_rows.loc[boolean_rows.kind.eq('equal')], control='identical_reference'))
            if 'center_is_perturbed' in boolean_rows:
                controls.append(negative_metrics(boolean_rows.loc[boolean_rows.kind.ne('equal') &
                    boolean_rows.center_is_perturbed.eq(False)], control='other_sensor_changed'))
    report.diagnostics.update(boolean_false_positives_by_step=pd.DataFrame(by_step),
                              control_false_positives=pd.DataFrame(controls))
    report.metadata.update(answer_metrics_schema='gevis.fusion_answer_metrics.v1',
        answer_metric_policy=METRIC_POLICY)
    return report


def display_report(report, display):
    """Refresh reports retained in an existing notebook kernel and show typed tables."""
    refresh_report(report)
    for title, table in [('Answer classification metrics', report.summary),
                         ('Changed-center answer accuracy', report.diagnostics['changed_only_metrics']),
                         ('BOOLEAN detection metrics', report.diagnostics['boolean_metrics']),
                         ('Per-class answer metrics', report.diagnostics['per_class']),
                         ('ORDINAL strength-step error', report.diagnostics['ordinal_metrics']),
                         ('Metrics by Step and answer type', report.by_step)]:
        if not table.empty:
            print(title)
            display(table)
    print(METRIC_POLICY)
    for key in ('boolean_false_positives_by_step', 'control_false_positives'):
        table = report.diagnostics[key]
        if not table.empty:
            print(key)
            display(table)
