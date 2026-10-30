"""Verify UI inputs, experimental targets and numeric gates remain distinct."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_shared_discovery import fixture
from tests.test_window_policy import records
from pecvd.experiments.window_policy import MODE, WindowConfig, WindowPolicy, fit_window_policy
from pecvd.experiments.discovery import review_changes
from pecvd.experiments.discovery_reporting import run_discovery_experiment, _gate_diagnostics
from pecvd.experiments.evaluation import intervention_cases


def centered_fixture():
    runtime,current,reference,_=fixture();m=runtime.module
    train=records('train',2);validation=records('validation',2,2)
    for r in train+validation:r['step']='invented_phase'
    m.DISCOVERY_POLICY_MODE=MODE
    m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,validation,{},
        WindowConfig(min_train_runs=1,min_validation_runs=1)))
    return runtime,current,reference


class CenteredEvaluationTests(unittest.TestCase):
    def test_preselected_flat_target_is_not_silently_moved_to_active_channel(self):
        _, current, _, _ = fixture()
        current['b'] = np.arange(len(current), dtype=float)
        cases=list(intervention_cases(current,['a','b'],np.ones(2),kinds=('delay',),
            strengths=(2.,),target_sensor='a',max_cases=1))
        spec, _, truth=cases[-1]
        self.assertEqual(spec.sensor,0)
        self.assertFalse(truth['effective'])

    def test_v14_primary_report_calls_ui_center_path_and_separates_tasks(self):
        runtime,current,reference=centered_fixture()
        with patch.dict(os.environ,{'GEVIS_DISCOVERY_TRAINING':'center_relation_v14'}), \
             patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(current,reference)]), \
             patch('pecvd.experiments.discovery_reporting.review_changes',wraps=review_changes) as calls, \
             contextlib.redirect_stdout(io.StringIO()):
            report=run_discovery_experiment('.',runtime=runtime,kinds=('spike',),strengths=(2.,),max_cases=1)
        self.assertEqual(report.metadata['analysis_mode'],'center')
        self.assertEqual(len(calls.call_args_list),3) # one control, self and context
        for call in calls.call_args_list:
            self.assertEqual(call.kwargs,{'center_sensor':'a'})
            self.assertEqual(len(call.args),5)
        self.assertEqual(set(report.details.task),{'self','context'})
        self.assertEqual(report.summary.iloc[0].windows,2)
        metrics=report.summary.iloc[0]
        denominator=2*metrics.tp+metrics.fp+metrics.fn
        if denominator:self.assertAlmostEqual(metrics.f1,2*metrics.tp/denominator)
        self.assertEqual(metrics.metric_target,'synthetic_effective_change_vs_untouched')
        self.assertTrue(report.details.top_sensor_correct.isna().all())
        self.assertEqual(set(report.diagnostics['gate_bottlenecks'].task),{'self','context'})
        self.assertIn('partner_ranking',report.diagnostics)
        self.assertFalse(report.metadata['truth_used_for_inference'])

    def test_paired_threshold_comparison_uses_center_primary_and_restores_policy(self):
        from pecvd.experiments.threshold_comparison import run_threshold_comparison
        runtime,current,reference=centered_fixture();m=runtime.module;original=m.DISCOVERY_POLICY
        m.ENCODER_ROOT='synthetic';m.BACKBONE_PATH='synthetic'
        with patch.dict(os.environ,{'GEVIS_DISCOVERY_TRAINING':'center_relation_v14'}), \
             patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(current,reference)]), \
             patch('pecvd.experiments.threshold_comparison.load_comparison',return_value={.95:original,.99:original}):
            report=run_threshold_comparison('.',runtime=runtime,max_cases=1)
        self.assertEqual(list(report.summary.analysis_mode),['center','center'])
        self.assertEqual(list(report.summary.task),['self','self'])
        self.assertIs(m.DISCOVERY_POLICY,original)
        self.assertTrue(report.metadata['same_cases'])

    def test_gate_report_distinguishes_correct_attention_from_rejected_change(self):
        result={'attention_top_sensor':'a','ranked_evidence':[
            {'sensor':'a','normalized_score':2.,'window_threshold':10.,'changed':False}]}
        row=_gate_diagnostics(result,'a')
        self.assertTrue(row['attention_correct_but_numeric_gate_rejected'])
        self.assertAlmostEqual(row['target_score_threshold_ratio'],.2)

    def test_context_gate_reads_partner_pair_score_in_center_row(self):
        result={'center_sensor':'a','attention_top_sensor':'b','ranked_evidence':[
            {'sensor':'a','normalized_score':9.,'window_threshold':10.,'changed':False,
             'related_sensor_scores':{'b':2.}}]}
        row=_gate_diagnostics(result,'b')
        self.assertTrue(row['attention_correct_but_numeric_gate_rejected'])
        self.assertAlmostEqual(row['target_score_threshold_ratio'],.2)


if __name__=='__main__':unittest.main()
