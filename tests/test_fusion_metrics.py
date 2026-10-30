"""Scoring regression with invented class indices, never private model outputs."""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from pecvd.experiments.fusion_metrics import display_report, refresh_report
from pecvd.experiments.reporting import ExperimentReport


def fixture():
    records = []
    for kind, truth, prediction in [
            ('BOOLEAN', [0, 0, 1, 1], [0, 1, 1, 1]),
            ('CHOICE', [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]),
            ('ORDINAL', [0, 1, 4], [0, 3, 1])]:
        records.extend(dict(answer_type=kind, label=y, prediction=p,
                            step='phase-a' if i == 0 else 'phase-b')
                       for i, (y, p) in enumerate(zip(truth, prediction)))
    return ExperimentReport('Synthetic Fusion answers', pd.DataFrame(),
        details=pd.DataFrame(records), metadata={'decision_system': 'fusion_answers'})


class FusionMetricTests(unittest.TestCase):
    def test_multiclass_macro_is_hand_computed_from_pooled_confusion(self):
        report = fixture()
        choice = report.summary.set_index('answer_type').loc['CHOICE']
        self.assertAlmostEqual(choice.answer_accuracy, 4 / 6)
        self.assertAlmostEqual(choice.macro_precision, (1 + .5 + 2 / 3) / 3)
        self.assertAlmostEqual(choice.macro_recall, (.5 + .5 + 1) / 3)
        self.assertAlmostEqual(choice.macro_f1, (2 / 3 + .5 + .8) / 3)
        self.assertAlmostEqual(choice.weighted_f1, choice.macro_f1)
        self.assertEqual(choice.evaluated_classes, 3)
        self.assertFalse(report.summary.isna().any().any())
        self.assertNotIn('tp', report.summary)
        self.assertNotIn('mean_absolute_step_error', report.summary)
        classes = report.diagnostics['per_class']
        absent = classes[(classes.answer_type == 'CHOICE') & (classes.class_index >= 3)]
        self.assertFalse(absent.included_in_macro.any())
        self.assertEqual(int(absent.support.sum()), 0)
        self.assertFalse(classes.isna().any().any())
        # Overall counts are pooled, not a mean that overweights the short Step.
        first = report.by_step.query("step == 'phase-a' and answer_type == 'CHOICE'")
        self.assertEqual(float(first.answer_accuracy.iloc[0]), 1.)

    def test_boolean_confusion_and_ordinal_error_are_separate(self):
        report = fixture()
        boolean = report.diagnostics['boolean_metrics'].iloc[0]
        self.assertEqual([boolean.tp, boolean.fp, boolean.fn, boolean.tn], [2, 1, 0, 1])
        self.assertAlmostEqual(boolean.precision, 2 / 3)
        self.assertEqual(boolean.recall, 1.)
        self.assertEqual(boolean.false_positive_rate, .5)
        self.assertEqual(boolean.f1, .8)
        ordinal = report.diagnostics['ordinal_metrics'].iloc[0]
        self.assertAlmostEqual(ordinal.mean_absolute_step_error, 5 / 3)
        self.assertAlmostEqual(ordinal.within_one_step_accuracy, 1 / 3)
        self.assertEqual(ordinal.maximum_step_error, 3)
        self.assertEqual(int(report.diagnostics['confusion']['count'].sum()), 13)

    def test_predicted_only_class_is_penalized_and_zero_division_is_explicit(self):
        report = fixture()
        report.details = pd.DataFrame([dict(answer_type='CHOICE', label=0, prediction=5)])
        refresh_report(report)
        self.assertEqual(report.summary.evaluated_classes.iloc[0], 2)
        self.assertEqual(report.summary.macro_f1.iloc[0], 0.)
        self.assertIn('Undefined class ratios use zero', report.metadata['answer_metric_policy'])
        row = report.diagnostics['per_class'].query('class_index == 5').iloc[0]
        self.assertTrue(row.included_in_macro)
        self.assertEqual(row.support, 0)
        self.assertEqual(row.fp, 1)

    def test_unlabelled_original_reports_do_not_gain_accuracy(self):
        summary = pd.DataFrame([dict(questions=1, accuracy_available=False)])
        report = ExperimentReport('Unlabelled', summary, details=pd.DataFrame([
            dict(answer_type='BOOLEAN', prediction=1)]),
            metadata={'decision_system': 'fusion_answers'})
        pd.testing.assert_frame_equal(report.summary, summary)
        self.assertNotIn('per_class', report.diagnostics)

    def test_display_refreshes_existing_report_without_altering_predictions(self):
        report = fixture()
        original = report.details.copy(deep=True)
        report.summary = pd.DataFrame([dict(legacy='old summary')])
        shown = []
        with contextlib.redirect_stdout(io.StringIO()):
            display_report(report, shown.append)
        self.assertEqual(len(shown), 8)
        self.assertIn('macro_f1', shown[0])
        self.assertAlmostEqual(report.diagnostics['changed_only_metrics'].set_index('answer_type')
                               .loc['CHOICE', 'changed_accuracy'], 3/4)
        pd.testing.assert_frame_equal(report.details, original)

    def test_false_positives_use_known_negative_denominators_and_step_groups(self):
        report = fixture()
        by_step = report.diagnostics['boolean_false_positives_by_step'].set_index('step')
        self.assertEqual(by_step.loc['phase-a', 'false_positive_rate'], 0.)
        self.assertEqual(by_step.loc['phase-b', 'false_positive_rate'], 1.)
        self.assertEqual(by_step.negative_support.sum(), 2)
        self.assertEqual(by_step.fp.sum(), 1)
        rows = pd.DataFrame([
            dict(answer_type='BOOLEAN', label=0, prediction=1, step='a', kind='equal', center_is_perturbed=False),
            dict(answer_type='BOOLEAN', label=0, prediction=0, step='a', kind='drift', center_is_perturbed=False),
            dict(answer_type='BOOLEAN', label=1, prediction=1, step='b', kind='drift', center_is_perturbed=True)])
        report.details = rows
        refresh_report(report)
        controls = report.diagnostics['control_false_positives'].set_index('control')
        self.assertEqual(controls.loc['identical_reference', 'false_positive_rate'], 1.)
        self.assertEqual(controls.loc['other_sensor_changed', 'false_positive_rate'], 0.)
        empty = report.diagnostics['boolean_false_positives_by_step'].set_index('step').loc['b']
        self.assertFalse(empty.fpr_available)
        self.assertTrue(np.isnan(empty.false_positive_rate))

    def test_invalid_class_indices_refuse_misleading_metrics(self):
        for prediction in (np.nan, .5, -1, 6):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                ExperimentReport('Invalid', pd.DataFrame(), details=pd.DataFrame([
                    dict(answer_type='CHOICE', label=0, prediction=prediction)]),
                    metadata={'decision_system': 'fusion_answers'})

