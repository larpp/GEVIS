from tests.legacy_notebooks import legacy_exp_book
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pecvd.experiments import gmm_comparison as gmm
from pecvd.experiments.position_rank_policy import context_key
from pecvd.experiments.window_policy import WindowConfig, WindowPolicy, fit_window_policy, MODE

ROOT = Path(__file__).resolve().parents[1]


def records(split='train', count=24):
    rng = np.random.default_rng(17 if split == 'train' else 18)
    result = []
    for i in range(count):
        amplitude = rng.uniform(.02,.08) if i < count//2 else rng.uniform(1.5,2.)
        current = np.full((16,2),amplitude) + rng.normal(0,.003,(16,2))
        result.append(dict(split=split,run_id=f'{split}-{i}',reference_run_ids=['train-reference'],
            step='invented_phase',sensors=['a','b'],cohort='recipe',position=0,
            current=current,reference=np.zeros_like(current),residual=np.zeros_like(current)))
    return result


@unittest.skipUnless(importlib.util.find_spec('sklearn'),'optional direct GMM fitting package unavailable')
class DirectGMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = WindowConfig(gate_scope='position_rank_v4',operation_profile='core',nominal_window_alert_rate=.01)
        cls.train = records(); cls.validation = records('validation',8)
        cls.baseline = WindowPolicy(fit_window_policy(cls.train,cls.validation,{},cls.config))
        cls.bundle = gmm.fit_models(cls.train,cls.config)

    def test_frozen_json_prediction_and_no_percentile_veto(self):
        baseline = WindowPolicy(copy.deepcopy(self.baseline.payload))
        for c in baseline.payload['groups']['invented_phase']['contexts'].values():
            c['thresholds'] = [[1e20,1e20],[1e20,1e20]]
        before = copy.deepcopy(baseline.payload)
        policy = gmm.GMMPolicy(baseline,json.loads(json.dumps(self.bundle,allow_nan=False)))
        r = self.train[-1]
        kwargs = dict(selected=['a'],center_sensor='a',cohort='recipe',position=0)
        self.assertFalse(baseline.analyze(r['current'],r['reference'],r['residual'],r['step'],r['sensors'],**kwargs)['changed'])
        result = policy.analyze(r['current'],r['reference'],r['residual'],r['step'],r['sensors'],**kwargs)
        self.assertTrue(result['accepted']); self.assertTrue(result['changed'])
        self.assertEqual(baseline.payload,before)
        self.assertIsNone(result['winner']['detected_rows'])
        self.assertIn('no q99 veto',result['change_decision_source'])

    def test_numpy_responsibility_matches_genuine_sklearn(self):
        from sklearn.mixture import GaussianMixture
        model = next(iter(self.bundle['models']['invented_phase'].values()))['a']
        fit = GaussianMixture(2,covariance_type='diag')
        fit.weights_ = np.array(model['weights']); fit.means_ = np.array(model['centers'])
        fit.precisions_cholesky_ = 1/np.sqrt(model['variances'])
        raw = np.array([[.02,.01,.01,.02,0.],[2.,.1,.1,1.,0.]])
        x = (np.log1p(raw)-model['mean'])/model['scale']
        np.testing.assert_allclose(gmm._probability(model,raw),fit.predict_proba(x),rtol=1e-10)

    def test_train_only_unique_run_support_and_degenerate_abstention(self):
        with self.assertRaisesRegex(ValueError,'TRAIN only'):
            gmm.fit_models(self.validation,self.config)
        with self.assertRaisesRegex(ValueError,'Independent TRAIN reference'):
            gmm.fit_models([{**self.train[0],'reference_run_ids':['train-0']}],self.config)
        repeated = gmm.fit_models([self.train[0]]*20,self.config)
        self.assertFalse(repeated['models']);self.assertEqual(repeated['diagnostics'][0]['train_runs'],1)
        flat = copy.deepcopy(self.train)
        for row in flat:row['current'][:]=0
        bundle = gmm.fit_models(flat,self.config)
        self.assertFalse(bundle['models'])
        policy = gmm.GMMPolicy(self.baseline,bundle);r = flat[0]
        result = policy.analyze(r['current'],r['reference'],r['residual'],r['step'],r['sensors'],cohort='recipe',position=0)
        self.assertFalse(result['accepted']);self.assertIsNone(result['changed'])

    def test_unknown_context_and_incompatible_config_abstain_or_fail(self):
        policy=gmm.GMMPolicy(self.baseline,self.bundle);r=self.train[0]
        result=policy.analyze(r['current'],r['reference'],r['residual'],r['step'],r['sensors'],cohort='unknown',position=0)
        self.assertFalse(result['accepted']);self.assertFalse(result['events'])
        with self.assertRaisesRegex(ValueError,'complete windows'):
            policy.analyze(r['current'],r['reference'],r['residual'],r['step'],r['sensors'],cohort='recipe',position=0,scopes=['early'])
        with self.assertRaisesRegex(ValueError,'configuration mismatch'):
            gmm.GMMPolicy(SimpleNamespace(config=replace(self.config,local_width=7)),self.bundle)

    def test_explicit_preparation_train_only_create_only_load_never_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            module=SimpleNamespace(DISCOVERY_POLICY=self.baseline,ENCODER_ROOT=Path(folder),
                BACKBONE_PATH=Path(folder)/'backbone.pt',DEVICE='cpu',
                numeric_state=lambda *a:torch.zeros((1,8,8)))
            runtime=SimpleNamespace(module=module,encoders={'invented_phase':(torch.nn.Linear(1,1),{'sensor_columns':['a','b']})})
            with patch('pecvd.experiments.discovery_calibration.identity_for',return_value={}), \
                 patch('pecvd.experiments.discovery_calibration.cached_comparisons',return_value=self.train) as read, \
                 contextlib.redirect_stdout(io.StringIO()):
                policy=gmm.prepare_gmm(runtime)
                self.assertEqual(read.call_args.kwargs['split'],'train')
                with patch.object(gmm,'fit_models',side_effect=AssertionError('No inference fit')):
                    restored=gmm.load_gmm(module);gmm.prepare_gmm(runtime)
                self.assertEqual(restored.bundle,policy.bundle)
                read.assert_called_once()
            path=next(Path(folder).glob('direct_gmm_*.json'));data=json.loads(path.read_text(encoding='utf-8'))
            data['identity']={};path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):gmm.load_gmm(module)

    def runtime_fixture(self):
        from tests.test_shared_discovery import fixture
        runtime,current,reference,_=fixture();m=runtime.module
        m.DISCOVERY_POLICY=WindowPolicy(copy.deepcopy(self.baseline.payload));m.DISCOVERY_POLICY_MODE=MODE
        for frame in (current,reference):frame.attrs.update(cohort_key='recipe',position_bin=0,reference_split='train',evaluation_pair_id='synthetic-pair')
        return runtime,current,reference

    def test_actual_center_runtime_paired_interventions_and_contract(self):
        runtime,current,reference=self.runtime_fixture();m=runtime.module;original=m.DISCOVERY_POLICY
        policy=gmm.GMMPolicy(original,self.bundle)
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(current,reference)]), \
             patch.object(m,'numeric_state',wraps=m.numeric_state) as encoder_calls, \
             patch.object(m.GEVIS_MODEL,'forward',wraps=m.GEVIS_MODEL.forward) as attention_calls, \
             contextlib.redirect_stdout(io.StringIO()):
            report=gmm.run_boundary_comparison(ROOT,runtime=runtime,gmm_policy=policy,
                split='test',analysis_mode='center',max_cases=4,pairs_per_step=1)
        self.assertIs(m.DISCOVERY_POLICY,original)
        self.assertEqual(report.summary.boundary.tolist(),['q99','gmm_direct'])
        self.assertTrue(report.metadata['paired_cases_verified'])
        self.assertGreater(encoder_calls.call_count,0);self.assertEqual(encoder_calls.call_count,attention_calls.call_count)
        self.assertTrue({'recall','f1','unsupported_effective_cases','untouched_alert_rate'} <= set(report.summary))
        contract=report.metadata['execution_contracts']['gmm_direct']
        self.assertIn('GMM',contract['change_decision_source'])

    def test_failure_restores_default_and_inference_never_prepares(self):
        runtime,current,reference=self.runtime_fixture();original=runtime.module.DISCOVERY_POLICY
        policy=gmm.GMMPolicy(original,self.bundle)
        with patch('pecvd.experiments.gevis_system.run_gevis_experiment',side_effect=[SimpleNamespace(),RuntimeError('synthetic failure')]):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                gmm.run_boundary_comparison(ROOT,runtime=runtime,gmm_policy=policy)
        self.assertIs(runtime.module.DISCOVERY_POLICY,original)
        with patch.object(gmm,'load_gmm',return_value=policy), \
             patch('pecvd.experiments.discovery.review_changes',side_effect=['q99 result','GMM result']) as review, \
             patch.object(gmm,'prepare_gmm',side_effect=AssertionError('Never fit')):
            result=gmm.compare_pair(runtime.module,current,reference,'invented_phase',None)
        self.assertEqual(result['gmm_direct'],'GMM result');self.assertEqual(review.call_count,2)
        self.assertIs(runtime.module.DISCOVERY_POLICY,original)

    def test_mismatched_evaluation_cases_are_rejected(self):
        import pandas as pd
        runtime,_,_=self.runtime_fixture();original=runtime.module.DISCOVERY_POLICY
        policy=gmm.GMMPolicy(original,self.bundle)
        first=SimpleNamespace(details=pd.DataFrame([dict(step='phase',pair_id='first',accepted=True)]))
        second=SimpleNamespace(metadata={},details=pd.DataFrame([dict(step='phase',pair_id='different',accepted=True)]))
        with patch('pecvd.experiments.gevis_system.run_gevis_experiment',return_value=second):
            with self.assertRaisesRegex(ValueError,'schedules differ'):
                gmm.run_boundary_comparison(ROOT,runtime=runtime,gmm_policy=policy,baseline_report=first)
        self.assertIs(runtime.module.DISCOVERY_POLICY,original)

    def test_flat_preparation_reports_failure_without_saving_or_mutating_q99(self):
        flat=copy.deepcopy(self.train)
        for row in flat:row['current'][:]=0
        with tempfile.TemporaryDirectory() as folder:
            module=SimpleNamespace(DISCOVERY_POLICY=self.baseline,ENCODER_ROOT=Path(folder),
                BACKBONE_PATH=Path(folder)/'backbone.pt',DEVICE='cpu',numeric_state=lambda *a:torch.zeros((1,8,8)))
            runtime=SimpleNamespace(module=module,encoders={'invented_phase':(torch.nn.Linear(1,1),{'sensor_columns':['a','b']})})
            with patch('pecvd.experiments.discovery_calibration.identity_for',return_value={}), \
                 patch('pecvd.experiments.discovery_calibration.cached_comparisons',return_value=flat), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(gmm.GMMNotIdentifiable) as error:gmm.prepare_gmm(runtime)
            self.assertTrue(all(r['reason']=='constant_features' for r in error.exception.diagnostics))
            self.assertFalse(list(Path(folder).glob('direct_gmm_*.json')))
            self.assertIs(module.DISCOVERY_POLICY,self.baseline)

    def test_retired_notebook_fixtures_call_comparison_with_existing_baseline(self):
        for name,flag,baseline in [('CVD_evaluate.ipynb','RUN_DISCOVERY_EVALUATION','DISCOVERY_REPORT'),
                                   ('test/02_anomaly_injection.ipynb','RUN_GMM_COMPARISON','REPORT')]:
            path=(ROOT/'tests/fixtures/step4_legacy_validation.json' if name.startswith('CVD_evaluate') else ROOT/'notebooks'/name)
            doc=(legacy_exp_book(Path(name).name) if name.startswith('test/') else json.loads(path.read_text(encoding='utf-8')))
            source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='boundary-comparison'))
            fake=SimpleNamespace(summary=None,by_step=None,diagnostics={'common_support':None,'gmm_fit_support':None,'by_kind':None})
            sentinel=object();env={flag:True,baseline:sentinel,'ROOT':ROOT,'REFERENCE_RUNTIME':None,
                'SELECTED_STEPS':None,'TEST_PAIRS_PER_STEP':2,'SEED':42,'EVALUATION_MODE':'center','display':lambda *a:None,
                'RUNTIME':SimpleNamespace(module=SimpleNamespace()),'KINDS':None,'STRENGTHS':(.5,2.),'MAX_CASES_PER_PAIR':8}
            with patch.object(gmm,'run_boundary_comparison',return_value=fake) as run,contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source,name,'exec'),env)
            self.assertIs(run.call_args.kwargs['baseline_report'],sentinel)
            self.assertEqual(run.call_args.kwargs['split'],'validation' if 'STEP_3_' in name else 'test')
