import contextlib,io,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests.test_gevis_experiments import fake_runtime
from tests.test_reference_policy import records
from pecvd.experiments.reference_policy_v2 import ReferencePolicy,fit_records
from pecvd.experiments.discovery import review_changes,CONTRACT
from pecvd.experiments.questions import DISCOVERY_QUESTIONS,discovery_questions

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    runtime,pairs=fake_runtime();module=runtime.module
    module.DECISION_POLICY_MODE='reference_quantiles_v2';module.STEP_ENCODERS=runtime.encoders
    # Eight independent flat reference runs: numerical tolerance, not missing support.
    source=[{**r,'step':'invented_phase','sensors':['a','b'],'features':np.zeros((3,2,5))} for r in records(8)]
    module.REFERENCE_POLICY=ReferencePolicy(fit_records(source,{}))
    current,reference=pairs(runtime,'invented_phase',limit=1)[0]
    current.loc[:,['a','b']]=0.;reference.loc[:,['a','b']]=0.
    return runtime,current,reference,pairs


class SharedDiscoveryTests(unittest.TestCase):
    def test_default_discovery_questions_never_name_a_target_sensor(self):
        from types import SimpleNamespace
        questions=discovery_questions(lambda text,kind,criteria:SimpleNamespace(
            text=text,answer_type=kind,criteria=criteria))
        self.assertEqual([q.answer_type for q in questions],['BOOLEAN','CHOICE','ORDINAL'])
        for question in questions:
            self.assertEqual(question.text,DISCOVERY_QUESTIONS[question.answer_type])
            self.assertNotIn('{sensor}',question.text)
            self.assertNotRegex(question.text,r'\bsensor\s+[A-Za-z0-9_]')

    def test_ui_review_uses_shared_evidence_not_a_guessed_sensor_question(self):
        from pecvd.gevis_ui import GevisRuntime
        from types import SimpleNamespace
        runtime,current,reference,_=fixture();m=runtime.module
        m.review_changes=lambda *args,**kwargs:review_changes(m,*args,**kwargs)
        m.aggregate_reference_windows=lambda rows,meta:rows[0]
        m.GEVIS_CHECKPOINT={'schema':'gevis.question_model.v15'}
        real=GevisRuntime(m)
        current.loc[8,'b']=5.
        with patch.object(GevisRuntime,'grounded_questions',side_effect=AssertionError('Do not guess sensor in question')):
            answers=real.ask_all_questions(current,(reference,),'invented_phase')
            selected=real.ask_selected_question(current,(reference,),'invented_phase','CHOICE')
            self.assertEqual(answers[0]['answer'],'Yes')
        self.assertTrue(all(a['evidence']['sensor']=='b' for a in answers if a['answer_type']!='CHOICE_SENSOR'))
        self.assertEqual(answers[-1]['typed_family'],'CHOICE')
        self.assertEqual(selected['evidence']['sensor'],'b')
        # Aggregated UI cards also use one shared median-reference decision.
        with patch('pecvd.gevis_ui.window_position_ratio',return_value=0.), \
             patch('pecvd.gevis_ui.matching_position_window',side_effect=lambda frame,*args:frame):
            cards,used=real.aggregate_typed_probabilities(current,(reference,),'invented_phase',1)
        self.assertEqual(used,1)
        self.assertEqual(cards[cards.selected].shape[0],4)

    def test_partial_support_can_report_supported_positive_but_never_negative(self):
        runtime,current,reference,_=fixture();m=runtime.module
        data=[{**r,'step':'invented_phase','sensors':['a'],'features':np.zeros((3,1,5))} for r in records(8)]
        m.REFERENCE_POLICY=ReferencePolicy(fit_records(data,{}))
        info=m.ReferenceInfo('test',1)
        result=review_changes(m,current,reference,'invented_phase',info)
        self.assertIsNone(result['changed']);self.assertFalse(result['complete_coverage'])
        current.loc[8,'a']=5.
        result=review_changes(m,current,reference,'invented_phase',info)
        self.assertTrue(result['changed']);self.assertFalse(result['complete_coverage'])
        self.assertIn('incomplete_reference_coverage',result['warnings'])

    def test_baseline_adds_discovery_on_identical_windows(self):
        from pecvd.experiments.baselines import compare_frozen_runtime
        runtime,current,reference,_=fixture()
        def pairs(*args,limit=2,**kwargs):return [(current,reference)]*limit
        with patch('pecvd.experiments.evaluation.load_cached_pairs',side_effect=pairs):
            results,raw=compare_frozen_runtime(runtime,'invented_phase',validation_limit=2,test_limit=1,
                max_cases=4,use_merlion=False,use_temporal=False)
        row=results[results.method.eq('GEVIS_shared_evidence_discovery')].iloc[0]
        self.assertEqual(row.test_windows,5)
        self.assertEqual(len(raw),7)
        self.assertEqual(row.prediction_abstentions,0)

    def test_no_change_and_spike_share_exact_typed_evidence_without_questions(self):
        runtime,current,reference,_=fixture();m=runtime.module;info=m.ReferenceInfo('test',1)
        with patch.object(m,'ask_gevis',side_effect=AssertionError('No named queries')):
            result=review_changes(m,current,reference,'invented_phase',info)
            self.assertFalse(result['changed']);self.assertTrue(result['complete_coverage'])
            self.assertEqual([r['answer'] for r in result['answers'][:3]],['No','No clear change','No change'])
            self.assertTrue(result['answers'][-1]['reference_only'])
            self.assertEqual(set(result['answers'][-1]['probabilities']),{'a','b'})
            current.loc[8,'b']=5.
            result=review_changes(m,current,reference,'invented_phase',info)
        self.assertTrue(result['changed']);self.assertEqual(result['winner']['sensor'],'b')
        self.assertEqual([row['question'] for row in result['answers'][:3]],
                         [DISCOVERY_QUESTIONS[kind] for kind in ('BOOLEAN','CHOICE','ORDINAL')])
        self.assertIn((8,9),result['winner']['pointwise']['exceedance_intervals'])
        self.assertTrue(all(a['evidence'] is result['winner'] for a in result['answers'][:3]))
        self.assertEqual(result['answers'][0]['answer'],'Yes')
        self.assertEqual(result['contract'],CONTRACT)
        self.assertEqual(len(result['ranked_evidence']),2*3*5)
        self.assertEqual(result['criterion_coverage'],1.)
        self.assertFalse(runtime.calls)

    def test_selection_bounds_and_missing_support_never_fabricate_no_change(self):
        runtime,current,reference,_=fixture();m=runtime.module;info=m.ReferenceInfo('test',1)
        current.loc[8,'b']=5.
        self.assertFalse(review_changes(m,current,reference,'invented_phase',info,sensors=['a'])['changed'])
        choice=review_changes(m,current,reference,'invented_phase',info,sensors=['a'])['answers'][-1]
        self.assertEqual(choice['probabilities'],{'a':1.})
        self.assertEqual(choice['answer'],'a')
        self.assertTrue(choice['reference_only'])
        self.assertFalse(review_changes(m,current,reference,'invented_phase',info,scopes=['early'])['changed'])
        with self.assertRaises(ValueError):review_changes(m,current,reference,'invented_phase',info,sensors=['unknown'])
        m.REFERENCE_POLICY=ReferencePolicy(fit_records([],{}))
        result=review_changes(m,current,reference,'invented_phase',info)
        self.assertIsNone(result['changed']);self.assertFalse(result['accepted'])
        self.assertTrue(all(a['answer'] is None for a in result['answers']))

    def test_blind_report_never_passes_event_metadata_to_inference(self):
        from pecvd.experiments.discovery_reporting import run_discovery_experiment
        runtime,current,reference,pairs=fixture()
        def windows(*args,**kwargs):return [(current,reference)]
        with patch('pecvd.experiments.discovery_reporting.load_cached_pairs',side_effect=windows), \
             patch('pecvd.experiments.discovery_reporting.review_changes',wraps=review_changes) as calls, \
             contextlib.redirect_stdout(io.StringIO()):
            report=run_discovery_experiment(ROOT,runtime=runtime,max_cases=4)
        self.assertTrue(report.metadata['model_evaluated'])
        self.assertFalse(report.metadata['truth_used_for_inference'])
        self.assertEqual(len(report.details),5)
        self.assertTrue(report.details.typed_consistent.all())
        for call in calls.call_args_list:
            self.assertEqual(len(call.args),5);self.assertFalse(call.kwargs)

    def test_step4_default_runs_discovery_not_named_query_or_explicit_matching(self):
        from tests.test_step4_reference_primary import sources
        runtime,current,reference,pairs=fixture();env={'display':lambda *a:None,'RUN_GMM_COMPARISON':False}
        from tests.test_threshold_comparison import notebook_comparison_fixture
        with patch.dict('os.environ',{}), \
             patch('pecvd.experiments.threshold_comparison.run_threshold_comparison',side_effect=notebook_comparison_fixture), \
             patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
             patch('importlib.reload',side_effect=lambda module:module), \
             patch('pecvd.experiments.evaluation.load_experiment_runtime',return_value=runtime), \
             patch('pecvd.experiments.discovery_reporting.load_cached_pairs',return_value=[(current,reference)]), \
             patch('pecvd.experiments.reporting.run_question_experiment',side_effect=AssertionError('No named audit by default')), \
             patch('pandas.read_csv',side_effect=AssertionError('No CSV reads')), \
             contextlib.redirect_stdout(io.StringIO()):
            for source in sources().values():
                # This regression intentionally exercises the retained legacy
                # blind runtime; the calibrated UI path has its own execution test.
                if 'DISCOVERY_REPORT=run_discovery_experiment(' in source:env['EVALUATION_MODE']='blind'
                exec(compile(source,'STEP4:blind-default','exec'),env)
        self.assertFalse(env['RUN_TYPED_QUESTION_AUDIT'])
        self.assertEqual(env['DISCOVERY_REPORT'].metadata['discovery_contract'],CONTRACT)
        self.assertFalse(runtime.calls)


if __name__=='__main__':unittest.main()
