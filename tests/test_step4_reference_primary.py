"""Retired reference comparator regressions use source-only legacy fixtures."""
import ast,contextlib,hashlib,io,json,os
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from pecvd.experiments.reference_policy_v2 import ReferencePolicy,fit_records
from tests.test_reference_policy import records
from tests.test_gevis_experiments import fake_runtime

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/CVD_evaluate.ipynb'
LEGACY_FIXTURE=ROOT/'tests/fixtures/step4_legacy_validation.json'


def sources():
    return {c['id']:''.join(c['source']) for c in json.loads(LEGACY_FIXTURE.read_text(encoding='utf-8'))['cells'] if c.get('source_group')=='retired-reference-v2'}


class Step4LegacyReferenceTests(unittest.TestCase):
    def test_discovery_cell_prepares_only_missing_policy_before_evaluation(self):
        from types import SimpleNamespace
        class ReachedEvaluation(Exception):pass
        for policy in (None,object()):
            runtime=SimpleNamespace(module=SimpleNamespace(DISCOVERY_POLICY=policy,DISCOVERY_POLICY_MODE='window_calibrated_v2'))
            env=dict(RUN_DISCOVERY_EVALUATION=True,REFERENCE_RUNTIME=runtime,ROOT=ROOT,
                PIPELINE_SETTINGS={'active_quantile':.99},SEED=42,SELECTED_STEPS=None,
                TEST_PAIRS_PER_STEP=2,EVALUATION_MODE='center',importlib=__import__('importlib'))
            with patch('importlib.reload',side_effect=lambda module:module), \
                 patch('pecvd.experiments.threshold_comparison.prepare_runtime_comparison') as prepare, \
                 patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',side_effect=ReachedEvaluation) as evaluate:
                with self.assertRaises(ReachedEvaluation):
                    exec(sources()['shared-discovery-evaluate'],env)
                self.assertEqual(prepare.call_count,int(policy is None))
                evaluate.assert_called_once()
        with patch('importlib.reload',side_effect=lambda module:module), \
             patch('pecvd.experiments.threshold_comparison.prepare_runtime_comparison',side_effect=ValueError('identity mismatch')), \
             patch('pecvd.experiments.discovery_reporting.run_discovery_experiment') as evaluate:
            env['REFERENCE_RUNTIME'].module.DISCOVERY_POLICY=None
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                exec(sources()['shared-discovery-evaluate'],env)
            evaluate.assert_not_called()

    def test_bootstrap_recovers_helpers_absent_from_cached_module(self):
        import pecvd.experiments.routing_audit as routing
        import pecvd.experiments.evaluation as evaluation
        import pecvd.experiments.reporting as reporting
        prefix=sources()['reference-load'].split('from pecvd.experiments.evaluation import',1)[0]
        with patch.object(routing,'refresh_evaluation_modules',None), \
             patch.object(routing,'require_routing_runtime',None):
            del routing.refresh_evaluation_modules
            del routing.require_routing_runtime
            env={}
            exec(compile(prefix,'STEP4:actual-warm-bootstrap','exec'),env)
            self.assertTrue(callable(routing.refresh_evaluation_modules))
            self.assertTrue(callable(env['require_routing_runtime']))
            self.assertIs(reporting.evaluate_questions,evaluation.evaluate_questions)

    def test_gate_cell_handles_old_report_before_key_access(self):
        from types import SimpleNamespace
        source=sources()['routing-gate-audit']
        old=SimpleNamespace(details=pd.DataFrame([{'supported':True}]),diagnostics={})
        with self.assertRaisesRegex(RuntimeError,'REPORT'):
            exec(source,{'REPORT':old,'display':lambda *a:None,'RUN_TYPED_QUESTION_AUDIT':True})
        report=SimpleNamespace(details=pd.DataFrame([{'step':'phase','answer_type':'BOOLEAN',
            'supported':True,'gate_checks':{'sensor_binding':{'passed':False}}}]),diagnostics={})
        env={'REPORT':report,'display':lambda *a:None,'RUN_TYPED_QUESTION_AUDIT':True}
        with contextlib.redirect_stdout(io.StringIO()): exec(source,env)
        self.assertEqual(env['GATE_BLOCKERS'].blocked_questions.iloc[0],1)

    def test_secondary_named_audit_executes_current_runtime_and_final_measurements(self):
        runtime,pairs=fake_runtime()
        runtime.module.DECISION_POLICY_MODE='reference_quantiles_v2'
        runtime.module.STEP_ENCODERS=runtime.encoders
        payload=fit_records(({**r,'step':'invented_phase','sensors':['a','b']} for r in records(600)),{'synthetic':True})
        runtime.module.REFERENCE_POLICY=ReferencePolicy(payload)
        shown=[]; env={'display':shown.append}; log=io.StringIO(); splits=[]
        def load(*args,**kwargs):
            splits.append(kwargs.get('split','test')); return pairs(*args,**kwargs)
        with patch.dict(os.environ,{'GEVIS_DECISION_POLICY':'reference_quantiles_v2'}), \
             patch('importlib.reload',side_effect=lambda module:module), \
             patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
             patch('pecvd.experiments.evaluation.load_experiment_runtime',return_value=runtime), \
             patch('pecvd.experiments.evaluation.load_cached_pairs',side_effect=load), \
             patch('pecvd.experiments.reporting.load_cached_pairs',side_effect=load), \
             patch('pecvd.experiments.reference_policy_v2.fit_records',side_effect=AssertionError('Must not fit in STEP 4')), \
             patch('pandas.read_csv',side_effect=AssertionError('Must not read CSV')), \
             contextlib.redirect_stdout(log):
            for id,source in sources().items():
                exec(compile(source,'STEP4:synthetic-runall','exec'),env)
                if id=='reference-config':
                    env['RUN_TYPED_QUESTION_AUDIT']=True
                    env['RUN_DISCOVERY_EVALUATION']=False
        self.assertTrue(runtime.calls)
        self.assertEqual(set(splits),{'test','validation'})
        self.assertEqual(env['REPORT'].metadata['split'],'test')
        self.assertEqual(env['VALIDATION_REPORT'].metadata['split'],'validation')
        self.assertEqual(set(env['ROUTING_COMPARISON'].routing_mode),{'learned','explicit_text'})
        self.assertFalse(env['VALIDATION_GATE_COMPARISON'].parameters_applied.any())
        self.assertEqual(env['REPORT'].metadata['decision_policy'],'reference_quantiles_v2')
        self.assertTrue(env['REPORT'].metadata['model_evaluated'])
        self.assertEqual(set(env['AGGREGATE'].execution_path),{'oracle','semantic_only','grounding_only','full_gevis'})
        self.assertEqual(set(env['SUMMARY'].answer_type),{'BOOLEAN','CHOICE','ORDINAL'})
        self.assertTrue(env['CONFUSION'].predicted_class.eq('Withheld: unsupported predicted criterion').any())
        pd.testing.assert_frame_equal(shown[-1],env['SUMMARY'][shown[-1].columns])
        self.assertIn('FINAL SUMMARY — CURRENT REFERENCE POLICY',log.getvalue())
        self.assertIn('No Validation alert-rate record',log.getvalue())
        self.assertNotIn('ljr_contract',env)
        self.assertFalse(env['REPORT'].metadata['weights_updated'])

    def test_legacy_environment_and_runtime_fail_closed(self):
        source=sources()
        with patch.dict(os.environ,{'GEVIS_DECISION_POLICY':'legacy'}):
            with self.assertRaisesRegex(ValueError,'legacy'):
                exec(source['reference-config'],{})
        runtime,_=fake_runtime(); runtime.module.DECISION_POLICY_MODE='legacy'
        env={'ROOT':ROOT,'EXPERIMENT_MODE':'reference_quantiles_v2','pd':pd}
        with patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
             patch('importlib.reload',side_effect=lambda module:module), \
             patch('pecvd.experiments.evaluation.load_experiment_runtime',return_value=runtime):
            with self.assertRaisesRegex(RuntimeError,'reference policy mode'):
                exec(source['reference-load'],env)

    def test_missing_criterion_is_not_silently_skipped(self):
        env={'ROOT':ROOT,'EXPERIMENT_MODE':'reference_quantiles_v2','pd':pd}
        with patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
             patch('importlib.reload',side_effect=lambda module:module), \
             patch('pecvd.experiments.evaluation.load_experiment_runtime',side_effect=FileNotFoundError('Run STEP 3 criterion preparation')):
            with self.assertRaisesRegex(FileNotFoundError,'STEP 3'):
                exec(sources()['reference-load'],env)

    def test_active_notebook_is_current_only(self):
        doc=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        source='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
        for marker in ('ljr_contract','GEVIS_CE_LJR','evaluate_predicted_paths','structured_answer_probabilities',"get('EXPERIMENT_MODE','typed')"):
            self.assertNotIn(marker,source)
        for c in doc['cells']:
            if c['cell_type']=='code': ast.parse(''.join(c['source']))


if __name__=='__main__': unittest.main()
