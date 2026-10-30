"""Synthetic evaluation joins must not manufacture labels or independence."""
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest

import pandas as pd

from pecvd.experiments.generated_test_scoring import score_saved


class GeneratedScoringTests(unittest.TestCase):
    def setUp(self):
        self.cases = pd.DataFrame([
            dict(case_id='control', source_group='source', type_id=None, strength=None, is_control=True),
            dict(case_id='weak', source_group='source', type_id='T01', strength='weak', is_control=False),
            dict(case_id='strong', source_group='source', type_id='T07', strength='strong', is_control=False)])
        self.events = pd.DataFrame([
            dict(case_id='weak', sensor='sensor', row_start=2, row_end=4),
            dict(case_id='strong', sensor='sensor', row_start=12, row_end=14),
            dict(case_id='strong', sensor='unsupported', row_start=2, row_end=4)])
        rows = []
        for case, alarm in [('control', 0), ('weak', 1), ('strong', 1)]:
            for kind, prediction in [('BOOLEAN', alarm), ('CHOICE', 1), ('ORDINAL', 2)]:
                rows.append(dict(case_id=case, row_start=0, row_end=10, step='step',
                    center_sensor='sensor', answer_type=kind, prediction=prediction,
                    paired_boolean_label=int(case != 'control')))
        self.report = SimpleNamespace(details=pd.DataFrame(rows))

    def test_event_coverage_type_truth_and_control_delta_are_distinct(self):
        before = self.report.details.copy(deep=True)
        result = score_saved(self.report, self.cases, self.events, choice_map={'T01': 1})
        events = result['events']
        self.assertTrue(events.iloc[0].hit)
        self.assertEqual(events.iloc[0].choice_correct, 1)
        self.assertEqual(events.iloc[1].choice_status, 'unmapped_type')
        self.assertEqual(events.evaluated.tolist(), [True, False, False])
        self.assertTrue(events.iloc[1:].hit.isna().all())
        self.assertTrue(result['control_comparison'].additional_alarm_rate.eq(1).all())
        self.assertFalse(result['truth_coverage'].iloc[0].ordinal_accuracy_available)
        pd.testing.assert_frame_equal(before, self.report.details)

    def test_missing_control_preserves_unmatched_denominator(self):
        report = SimpleNamespace(details=self.report.details.loc[self.report.details.case_id.ne('control')])
        result = score_saved(report, self.cases, self.events)
        self.assertTrue(result['control_comparison'].matched_questions.eq(0).all())
        self.assertTrue(result['control_comparison'].additional_alarm_rate.isna().all())

    def test_no_exact_changed_mask_means_no_type_accuracy(self):
        report = SimpleNamespace(details=self.report.details.drop(columns='paired_boolean_label'))
        result = score_saved(report, self.cases, self.events, choice_map={'T01': 1})
        self.assertEqual(result['events'].iloc[0].choice_questions, 0)

    def test_invalid_or_ambiguous_generation_truth_is_rejected(self):
        with self.assertRaises(ValueError):
            score_saved(self.report, pd.concat([self.cases, self.cases.iloc[:1]]), self.events)
        events = self.events.copy(); events.loc[0, 'row_end'] = 2
        with self.assertRaises(ValueError):
            score_saved(self.report, self.cases, events)
        with self.assertRaises(ValueError):
            score_saved(self.report, self.cases, self.events, choice_map={'T01': 9})

    def test_empty_events_keep_control_report(self):
        result = score_saved(self.report, self.cases, self.events.iloc[:0])
        self.assertTrue(result['by_type_strength'].empty)
        self.assertEqual(len(result['control_comparison']), 2)

    def test_external_notebooks_do_not_autorun_generated_scoring(self):
        root=Path(__file__).resolve().parents[1]
        for name in ('CVD_predict_external_csv.ipynb','ALD_predict_external_csv.ipynb'):
            notebook=json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
            code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
            self.assertNotIn('score_saved(',code)
            self.assertNotIn('score_reviewed_labels(',code)
            self.assertNotIn('original_path=ORIGINAL_PATH',code)


if __name__ == '__main__':
    unittest.main()
