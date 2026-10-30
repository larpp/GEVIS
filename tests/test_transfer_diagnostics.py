"""Invented outputs only; no private files, model loading or neural execution."""
from tests.legacy_notebooks import legacy_exp_book
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import pandas as pd
from pecvd.experiments.gevis_system import transfer_prediction_diagnostics,evaluation_summary,run_gevis_predictions


def original():
    return pd.DataFrame([
        dict(step='phase',pair_index=i,center_sensor='sensor',question_id='any_change',
             accepted=accepted,alert=alert,context_key='context',
             prediction='UNSUPPORTED' if not accepted else 'ALERT' if alert else 'NO_ALERT')
        for i,(accepted,alert) in enumerate([(True,True),(True,False),(False,False)])])


class TransferDiagnosticsTests(unittest.TestCase):
    def test_compact_summary_preserves_injection_metrics_and_original_latency(self):
        frame=original();frame['latency_ms']=[10.,20.,30.]
        controls=frame.drop(columns='question_id').copy();controls['injected']=False
        controls['effective']=None
        change=controls.iloc[:1].copy();change['injected']=True;change['effective']=True
        report=SimpleNamespace(details=pd.concat([controls,change],ignore_index=True),
            summary=pd.DataFrame([dict(precision=.75,recall=.5,f1=.6,mean_ms_per_window=18.,windows=4)]),
            diagnostics={'original_input_predictions':frame})
        table=evaluation_summary(report,process='PEALD')
        self.assertEqual(len(table),2)
        self.assertEqual(table.iloc[0].mean_ms_per_window,20.)
        self.assertTrue(pd.isna(table.iloc[0].recall))
        self.assertEqual(table.iloc[0].unsupported,1)
        self.assertEqual(table.iloc[1].f1,.6)
        self.assertEqual(table.iloc[1].mean_ms_per_window,18.)
        self.assertEqual(report.summary.iloc[0].f1,.6)

    def test_pecvd_original_summary_from_injection_controls_no_extra_forward(self):
        frame=original().drop(columns='question_id');frame['injected']=False
        frame['latency_ms']=[5.,7.,9.]
        report=SimpleNamespace(details=frame,diagnostics={},summary=pd.DataFrame())
        table=evaluation_summary(report,process='PECVD')
        self.assertEqual(len(table),1);self.assertEqual(table.iloc[0].windows,3)
        self.assertEqual(table.iloc[0].mean_ms_per_window,7.)
        self.assertTrue(pd.isna(table.iloc[0].precision))

    def test_original_prediction_latency_recorded_once_per_window_with_stub_only(self):
        from unittest.mock import patch
        frame=pd.DataFrame({'sensor':[0.,1.]});frame.attrs['evaluation_pair_id']='synthetic'
        candidate=dict(sensor='sensor',statistic='level',normalized_score=.2,window_threshold=.5,changed=False)
        result=dict(accepted=True,ranked_evidence=[candidate],winner=None,latency_ms=12.)
        module=SimpleNamespace(ReferenceInfo=lambda *args:None)
        runtime=SimpleNamespace(module=module,encoders={'phase':(None,{'sensor_columns':['sensor']})})
        with patch('pecvd.experiments.gevis_system.ready_gevis_runtime',return_value=runtime), \
             patch('pecvd.experiments.frozen_transfer.load_discovery_pairs',return_value=[(frame,frame)]), \
             patch('pecvd.experiments.discovery.review_changes',return_value=result) as review:
            report=run_gevis_predictions('.',runtime=runtime,question_variants=[
                ('any_change','BOOLEAN','Any change?'),('mean_change','BOOLEAN','Mean change?')])
        review.assert_called_once()
        self.assertEqual(report.summary.iloc[0].mean_ms_per_window,12.)
        self.assertEqual(report.diagnostics['original_input_summary'].iloc[0].windows,1)
        self.assertTrue(report.details.latency_ms.eq(12.).all())

    def test_retired_final_cells_show_process_and_speed_without_model_execution(self):
        import importlib
        from pecvd.experiments.reporting import ExperimentReport,show_report
        root=Path(__file__).resolve().parents[1]
        for file,process in [('01_question_understanding.ipynb','PECVD'),
            ('02_anomaly_injection.ipynb','PECVD'),('04_dataset_transfer.ipynb','PEALD')]:
            details=original();details['latency_ms']=[1.,2.,3.]
            report=ExperimentReport('synthetic',pd.DataFrame([{'mean_ms_per_window':2.}]),
                details=details,metadata={'status':'COMPLETED','model_evaluated':True})
            shown=[]
            env=dict(REPORT=report,importlib=importlib,show_report=show_report,display=shown.append)
            doc=legacy_exp_book(file)
            final=next(c for c in doc['cells'] if c.get('id')=='final-summary')
            exec(compile(''.join(final['source']),'synthetic-final-summary','exec'),env)
            self.assertEqual(env['SUMMARY'].iloc[0].process,process)
            self.assertEqual(env['SUMMARY'].iloc[0].mean_ms_per_window,2.)
            self.assertEqual(shown[-1].iloc[0].precision,'N/A')

    def test_original_is_coverage_not_accuracy_and_unsupported_not_negative(self):
        frame=original();before=frame.copy(deep=True)
        policy=SimpleNamespace(payload={'groups':{'phase':{'contexts':{'context':dict(
            train_runs=7,validation_runs=2,support_failure='insufficient_runs',control_budget_resolved=False)}}}})
        tables=transfer_prediction_diagnostics(frame,policy=policy)
        s=tables['original_input_summary'].iloc[0]
        self.assertEqual((s.windows,s.alerts,s.no_alerts,s.unsupported),(3,1,1,1))
        self.assertAlmostEqual(s.alert_rate_all_inputs,1/3)
        self.assertAlmostEqual(s.alert_rate_when_accepted,.5)
        self.assertFalse(s.accuracy_available)
        self.assertNotIn('f1',s.index)
        self.assertEqual(tables['original_input_support'].iloc[2].support_failure,'insufficient_runs')
        pd.testing.assert_frame_equal(frame,before)

    def test_effective_outcomes_partition_and_attention_is_not_miss_cause(self):
        rows=[]
        for accepted,alert,status in [(False,False,'UNSUPPORTED_CONTEXT_OR_CRITERIA'),
            (True,False,'BELOW_THRESHOLD'),(True,False,'TARGET_NOT_ANALYZED'),
            (True,False,'OTHER'),(True,True,'ALERT')]:
            rows.append(dict(step='phase',task='self',injected=True,effective=True,
                accepted=accepted,alert=alert,target_gate_status=status,
                raw_attention_sensor_correct=False))
        rows.append(dict(step='phase',task='self',injected=True,effective=False,accepted=True,alert=False))
        rows.append(dict(step='phase',task='self',injected=False,effective=None,accepted=False,alert=False))
        frame=pd.DataFrame(rows);before=frame.copy(deep=True)
        tables=transfer_prediction_diagnostics(original(),frame)
        self.assertEqual(tables['injection_outcome_stages'].cases.sum(),5)
        self.assertEqual(len(tables['injection_miss_cases']),4)
        self.assertEqual(set(tables['injection_outcome_stages'].failure_stage),{
            'UNSUPPORTED','BELOW_THRESHOLD','TARGET_NOT_ANALYZED','MISS_UNRESOLVED','DETECTED'})
        self.assertEqual(tables['injection_control_summary'].iloc[0].unsupported,1)
        pd.testing.assert_frame_equal(frame,before)

    def test_no_double_count_of_question_variants_and_no_injection_mode(self):
        frame=original();variant=frame.copy();variant['question_id']='spike'
        tables=transfer_prediction_diagnostics(pd.concat([frame,variant],ignore_index=True))
        self.assertEqual(tables['original_input_summary'].iloc[0].windows,3)
        self.assertNotIn('injection_outcome_stages',tables)
        self.assertEqual(transfer_prediction_diagnostics(pd.DataFrame()),{})

    def test_retired_diagnostic_fixture_reuses_existing_report(self):
        root=Path(__file__).resolve().parents[1]
        doc=legacy_exp_book('04_dataset_transfer.ipynb')
        cell=next(c for c in doc['cells'] if c.get('id')=='prediction-and-miss-diagnosis')
        report=SimpleNamespace(details=pd.DataFrame(),diagnostics={'original_input_predictions':original()})
        env=dict(REPORT=report,INCLUDE_INJECTION_EVALUATION=True,pd=pd,display=lambda x:None)
        exec(compile(''.join(cell['source']),'EXP4:diagnosis','exec'),env)
        self.assertEqual(report.diagnostics['original_input_summary'].iloc[0].windows,3)


if __name__=='__main__':unittest.main()
