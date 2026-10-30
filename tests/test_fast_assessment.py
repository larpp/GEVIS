"""Do not hide unscored answer families or confuse partial timing with acceptance."""
from types import SimpleNamespace as NS
import copy
import contextlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd

from pecvd.experiments.fast_assessment import assess


def report():
    return NS(summary=pd.DataFrame([
        dict(answer_type=k, questions=10, macro_f1=s, answer_accuracy=s)
        for k, s in [('BOOLEAN', .99), ('CHOICE', .8), ('ORDINAL', .7)]]),
        diagnostics=dict(boolean_metrics=pd.DataFrame([dict(answer_type='BOOLEAN', questions=10,
            f1=.98, positive_support=5, negative_support=5)]),
            changed_only_metrics=pd.DataFrame([dict(answer_type=k, changed_questions=5, changed_accuracy=.6)
                for k in ('BOOLEAN', 'CHOICE', 'ORDINAL')]),
            ordinal_metrics=pd.DataFrame([dict(answer_type='ORDINAL', mean_absolute_step_error=.3,
                within_one_step_accuracy=.9)]),
            requests=pd.DataFrame([dict(status='OK', request_ms=40., answers=6, selected_sensors=2),
                dict(status='OK', request_ms=60., answers=6, selected_sensors=2),
                dict(status='NOT_EVALUATED', request_ms=1., answers=0, selected_sensors=0)])),
        metadata=dict(domain='PECVD', timing_scope='synthetic test requests'))


class FastAssessmentTests(unittest.TestCase):
    def test_all_answers_required_boolean_positive_f1_and_full_request_goal(self):
        source = report(); before = copy.deepcopy(source)
        tables = assess(source, evidence_scope='synthetic_fixture')
        quality = tables['quality'].set_index('answer_type')
        self.assertTrue(quality.required.all())
        self.assertEqual(quality.loc['BOOLEAN', 'f1'], .98)
        self.assertEqual(quality.loc['CHOICE', 'comparison_metric'], 'macro_f1')
        self.assertFalse(quality.loc['ORDINAL', 'meets_reference'])
        self.assertEqual(quality.loc['ORDINAL', 'ordinal_mae'], .3)
        timing = tables['request_latency'].iloc[0]
        self.assertEqual(timing.mean_ms, 50.)
        self.assertEqual(timing.fraction_over_50ms, .5)
        self.assertEqual(timing.failed_or_unmeasured_requests, 1)
        self.assertTrue(timing.three_answers_per_sensor_count_check)
        self.assertTrue(timing.three_answer_mean_reference_met)
        self.assertEqual(tables['scope'].iloc[0].overall_acceptance, 'not_certified')
        pd.testing.assert_frame_equal(source.summary, before.summary)
        pd.testing.assert_frame_equal(source.diagnostics['requests'], before.diagnostics['requests'])

    def test_boolean_only_scoring_keeps_choice_and_ordinal_unscored(self):
        source = report()
        source.summary = source.summary.iloc[:1]
        quality = assess(source, evidence_scope='external_injection')['quality'].set_index('answer_type')
        self.assertFalse(quality.loc['CHOICE', 'metric_available'])
        self.assertTrue(pd.isna(quality.loc['CHOICE', 'meets_reference']))
        self.assertEqual(len(quality), 3)

    def test_control_only_score_does_not_claim_complete_quality(self):
        source = report()
        source.diagnostics['changed_only_metrics']['changed_questions'] = 0
        quality = assess(source, evidence_scope='control_only')['quality']
        self.assertFalse(quality.metric_available.any())

    def test_no_timing_does_not_inherit_fast_time_from_other_report(self):
        source = report(); source.diagnostics.pop('requests')
        timing = assess(source, evidence_scope='paired_diagnostic')['request_latency'].iloc[0]
        self.assertTrue(pd.isna(timing.mean_ms))
        self.assertIsNone(timing.mean_within_50ms)

    def test_one_answer_per_sensor_cannot_pass_three_answer_count(self):
        source = report(); source.diagnostics['requests']['answers'] = 2
        timing = assess(source, evidence_scope='partial_request')['request_latency'].iloc[0]
        self.assertFalse(timing.three_answers_per_sensor_count_check)
        self.assertFalse(timing.three_answer_mean_reference_met)

    def test_variant_merge_requires_explicit_single_scope(self):
        source = report(); source.summary = pd.concat([source.summary, source.summary])
        with self.assertRaises(ValueError): assess(source, evidence_scope='mixed_variants')

    def test_compact_view_has_bounded_predictions_and_no_fake_quality(self):
        from pecvd.experiments.fast_assessment import compact
        source=report();source.summary=pd.DataFrame();source.diagnostics.pop('boolean_metrics')
        source.details=pd.DataFrame([dict(answer_type='BOOLEAN',answer='Yes')]*100)
        shown=[]
        with contextlib.redirect_stdout(io.StringIO()):
            compact(source,shown.append,evidence_scope='preview',predictions=18)
        self.assertTrue(any(len(table)==18 for table in shown))
        self.assertFalse(any('f1' in table for table in shown))


if __name__ == '__main__': unittest.main()
