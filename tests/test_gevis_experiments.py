from contextlib import ExitStack
from tests.legacy_notebooks import legacy_exp_book
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from pecvd.experiments.questions import question_bank, runtime_questions, CRITERIA
from pecvd.experiments.injection import Injection, inject, synthetic_fixture, KINDS
from pecvd.experiments.operations import semantic_comparison_features
from pecvd.experiments.evaluation import binary_metrics, interval_iou, summarize_questions
from pecvd.experiments.baselines import compare_baselines
from pecvd.experiments.transfer import validate_transfer_pack

ROOT=Path(__file__).resolve().parents[1]


def fake_runtime():
    """Invented runtime, for exercising the actual orchestrators without weights."""
    import pandas as pd
    from pecvd import gevis_fast
    class Encoder:
        def to(self,*args): return self
        def eval(self): return self
        def cpu(self): return self
    class SyntheticAttention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.numeric=torch.nn.Linear(8,8)
            self.answer_criteria=gevis_fast.DEFAULT_ANSWER_CRITERIA
            self.heads=torch.nn.ModuleDict({k:torch.nn.Linear(8,len(v)) for k,v in self.answer_criteria.items()})
        def forward(self,state,questions,kinds,criteria,descriptions,return_fusion=False):
            return state.mean(1),state.mean(1),torch.full(state.shape[:2],1/state.shape[1],device=state.device)
        def decode(self,fusion,kinds,criteria):
            return [self.heads[k](fusion[0][i]) for i,k in enumerate(kinds)]
    def prepare(frame,meta):
        return torch.tensor(frame[['a','b']].to_numpy(),dtype=torch.float32)[None],{}
    calls=[]
    def ask(current,reference,step,questions,info,*,routing_mode='learned'):
        calls.append(len(questions))
        changed=float(np.max(np.abs(current[['a','b']].to_numpy()-reference[['a','b']].to_numpy())))
        score=float(1/(1+np.exp(-(changed-1))))
        answers=[]
        for q in questions:
            from pecvd.experiments.routing_audit import explicit_slots,acceptance_gates
            resolved=explicit_slots(q.text,['a','b']) if routing_mode=='explicit_text' else {}
            accepted=not any(word in q.text for word in ('prove','remaining lifetime','certainly'))
            choices=CRITERIA[q.answer_type]; selected=int(score>.5)
            candidate=choices[selected]
            probabilities={k:(score if i==selected else (1-score)/(len(choices)-1)) for i,k in enumerate(choices)}
            if q.answer_type=='BOOLEAN': probabilities={choices[0]:1-score,choices[1]:score}
            gates=acceptance_gates(criterion_ready=True,criteria_match=True,support_confidence=.9 if accepted else 0.,
                comparable=True,answer_confidence=.9,semantic_scores={'time_scope':.9},sensor_confidence=.8,
                required_semantics=('time_scope',),confidence_floor=.55,semantic_floor=.60,grounding_floor=.30,
                support_floor=.50,score_gate=False,explicit_sensor=bool(resolved.get('sensor')),explicit_scope=bool(resolved.get('time_scope')))
            answers.append({'predicted_query':{'sensor':resolved.get('sensor') or 'a','time_scope':resolved.get('time_scope') or 'whole','statistic':'level','direction':'increase'},
                'gate_checks':gates,'routing':{'mode':routing_mode},
                'accepted':accepted,'candidate_answer':candidate,'answer':candidate if accepted else None,
                'probabilities':probabilities,'semantic_scores':{'statistic':{'level':1.}},
                'sensor_weights':{'a':.8,'b':.2},'coverage_warnings':[] if accepted else ['unsupported_question'],
                'evidence':{'sensor':'a','scope_rows':[0,len(current)]}})
        return answers
    module=SimpleNamespace(DEVICE='cpu',ORDER_COLUMN='row_order',prepare_window=prepare,numeric_state=lambda *args:torch.zeros(1,6,8),
        semantic_comparison_features=semantic_comparison_features,gevis_fast=gevis_fast,
        CRITERIA=CRITERIA,
        GevisQuestion=lambda text,kind=None,criteria=None,answer_type=None:SimpleNamespace(text=text,answer_type=answer_type or kind,criteria=criteria),
        ReferenceInfo=lambda text,count:SimpleNamespace(description=text,process_count=count),ask_gevis=ask,
        GEVIS_CHECKPOINT={'_artifact_sha256':'synthetic-test-only'},BACKBONE_PATH=Path('synthetic/backbone.pt'),
        GEVIS_MODEL=SyntheticAttention(),
        numeric_token_descriptions=lambda step,meta,patches,batch:[['fixture']*(len(meta['sensor_columns'])*patches)]*batch)
    runtime=SimpleNamespace(module=module,encoders={'invented_phase':(Encoder(),{'sensor_columns':['a','b'],
        'std':np.ones(2),'mean':np.zeros(2),'window_length':12})},calls=calls)
    def pairs(runtime,step,split='test',limit=2,**kwargs):
        rng=np.random.default_rng(42)
        result=[]
        for i in range(limit):
            current=pd.DataFrame(rng.normal(0,.02,(12,2)),columns=['a','b'])
            current['row_order']=np.arange(len(current))
            reference=current.copy()
            for frame in (current,reference):
                frame.attrs['evaluation_pair_id']=f'{split}-{i}'; frame.attrs['evaluation_run_id']=f'{split}-{i}'
            result.append((current,reference))
        return result
    return runtime,pairs


class ExperimentTests(unittest.TestCase):
    def test_bank_covers_slots_targets_types_and_pairs(self):
        bank=question_bank('invented',['a','b','c'],limit=120)
        self.assertEqual(len(bank),102)
        self.assertEqual({c.time_scope for c in bank},{'whole','early','late'})
        self.assertEqual({c.sensor for c in bank},{None})
        self.assertEqual(bank,question_bank('invented',['different'],limit=120))
        self.assertEqual(len({c.text for c in bank}),len(bank))
        self.assertEqual({c.answer_type for c in bank},set(CRITERIA))
        self.assertEqual({c.statistic for c in bank if c.answer_type=='BOOLEAN'},
                         {'level','variability','slope','local_change','residual'})
        for a,b in zip(bank[::2],bank[1::2]):
            self.assertEqual(a.family,b.family)
            self.assertNotEqual(a.text,b.text)
        self.assertFalse(set(c.text for c in bank)&set(c.text for c in question_bank('invented',['a'],split='train')))

    def test_default_runtime_api(self):
        factory=lambda text,kind,criteria:(text,kind,criteria)
        self.assertEqual(len(runtime_questions('phase','sensor',factory)),3)
        self.assertEqual(len(runtime_questions('phase','sensor',factory,bank=True,limit=20)),20)

    def test_intervention_is_copy_only_and_truth_is_exact(self):
        x,_,s=synthetic_fixture()
        saved=x.copy()
        for kind in KINDS:
            spec=Injection(kind,1,64,96,2,42)
            y,truth=inject(x,spec,s)
            np.testing.assert_array_equal(x,saved)
            np.testing.assert_array_equal(y[:64],x[:64])
            np.testing.assert_array_equal(y[:,[0,2]],x[:,[0,2]])
            np.testing.assert_array_equal(truth['changed_mask'],y!=x)
            np.testing.assert_array_equal(y,inject(x,spec,s)[0])
        with self.assertRaises(ValueError): inject(x,Injection('spike',0,-1,10),s)

    def test_float32_interventions_do_not_emit_values_in_dtype_warnings(self):
        import warnings
        import pandas as pd
        from pecvd.experiments.evaluation import intervention_cases
        x,_,scale=synthetic_fixture(length=24,sensors=2)
        frame=pd.DataFrame(x.astype('float32'),columns=['a','b'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',FutureWarning)
            cases=list(intervention_cases(frame,['a','b'],scale,max_cases=7))
        self.assertEqual(len(cases),8)
        np.testing.assert_array_equal(frame.to_numpy(),x.astype('float32'))

    def test_signed_mse_difference_not_latent_difference(self):
        x=torch.zeros(2,12,2); reference=x.clone(); x[:,8:,0]=3
        state=torch.zeros(2,6,10); state[...,-2]=5; state[...,-1]=2
        features=semantic_comparison_features(x,reference,state,3)
        self.assertEqual(features.shape,(2,3,2,5))
        self.assertAlmostEqual(float(features[0,0,0,0]),1.)
        self.assertAlmostEqual(float(features[0,2,0,0]),3.)
        self.assertTrue(torch.all(features[...,-1]==3))

    def test_empty_classes_do_not_become_perfect_scores(self):
        result=binary_metrics([False,False],[False,False])
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['precision'])
        self.assertEqual(interval_iou((0,5),(3,8)),.25)

    def test_coverage_and_unsupported_rejection(self):
        rows=[]
        for i in range(2):
            rows.append(dict(answer_type='BOOLEAN',supported=True,accepted=i==0,answer_correct=i==0,
                sensor_correct=True,scope_correct=True,statistic_correct=True,direction_correct=True,
                family='paired',candidate='yes' if i==0 else 'no'))
        rows.append(dict(rows[0],supported=False,accepted=False))
        frame,support=summarize_questions(rows)
        self.assertEqual(frame.iloc[0].coverage,.5)
        self.assertEqual(frame.iloc[0].accepted_proxy_accuracy,1.)
        self.assertEqual(frame.iloc[0].paraphrase_consistency,0.)
        self.assertEqual(support['unsupported_rejection_rate'],1.)

    def test_transfer_leakage_and_explicit_no_transfer_claim(self):
        rng=np.random.default_rng(1)
        pack={'dataset_id':'fixture','sensors':['a','b']}
        for split in ('train','validation','test'):
            pack[split]=rng.normal(size=(2,16,2)); pack[split+'_run_ids']=[split+'0',split+'1']
        self.assertFalse(validate_transfer_pack(pack)['gevis_transfer_verified'])
        pack['test_run_ids']=pack['train_run_ids']
        with self.assertRaises(ValueError): validate_transfer_pack(pack)

    def test_baseline_threshold_does_not_depend_on_test(self):
        rng=np.random.default_rng(2)
        train=rng.normal(size=(40,2)); val=rng.normal(size=(20,2)); test=rng.normal(size=(20,2))
        args=(train,val,test,np.zeros_like(test),np.ones(2),np.arange(20)>10)
        first=compare_baselines(*args)
        second=compare_baselines(train,val,test*100,np.zeros_like(test),np.ones(2),args[-1])
        self.assertEqual(first[0]['threshold'],second[0]['threshold'])

    def test_paired_runtime_comparison_uses_same_control_and_injection(self):
        import pandas as pd
        from pecvd.experiments.baselines import compare_frozen_runtime
        current=pd.DataFrame({'a':np.linspace(0,.1,12)})
        reference=current.copy()
        class Encoder:
            def to(self,*args): return self
            def eval(self): return self
        def prepare(frame,meta): return torch.tensor(frame[['a']].to_numpy(),dtype=torch.float32)[None],{}
        def features(a,b,state,patches):
            value=(a-b).mean().abs()
            return torch.ones(1,3,1,5)*value
        module=SimpleNamespace(DEVICE='cpu',prepare_window=prepare,
            numeric_state=lambda *args:torch.zeros(1,3,8),semantic_comparison_features=features)
        runtime=SimpleNamespace(module=module,encoders={'phase':(Encoder(),{'sensor_columns':['a'],'std':[1.]})})
        def scanner(runtime,frame,ref,*args,**kwargs):
            score=float((frame.a-ref.a).abs().max())
            return [{'score':score,'alert':score>1}]
        with patch('pecvd.experiments.evaluation.load_cached_pairs',return_value=[(current,reference)]*3), \
             patch('pecvd.experiments.evaluation.runtime_scan',side_effect=scanner):
            results,details=compare_frozen_runtime(runtime,'phase')
        self.assertEqual(len(results),4)
        held=details[details.split=='test']
        self.assertEqual(list(held.injected),[False,True,True,True,True]*3)
        self.assertTrue(all(results.fn==0))

    def test_retired_exp_fixtures_execute_legacy_core_and_show_final_results(self):
        for path in (ROOT/'notebooks').glob('STEP_*.ipynb'):
            doc=json.loads(path.read_text(encoding='utf-8'))
            for c in doc['cells']:
                if c['cell_type']=='code': ast.parse(''.join(c['source']))
        for legacy_name in ('01_question_understanding.ipynb', '02_anomaly_injection.ipynb', '03_baseline_comparison.ipynb', '04_dataset_transfer.ipynb'):
            path=Path(legacy_name)
            doc=legacy_exp_book(path.name)
            runtime,pairs=fake_runtime(); shown=[]
            if path.name.startswith(('01_','02_')):
                from pecvd.experiments.reference_policy_v2 import ReferencePolicy,fit_records
                from tests.test_reference_policy import records
                runtime.module.DECISION_POLICY_MODE='reference_quantiles_v2'
                runtime.module.STEP_ENCODERS=runtime.encoders
                runtime.module.REFERENCE_POLICY=ReferencePolicy(fit_records(
                    ({**r,'step':'invented_phase','sensors':['a','b']} for r in records(30)),{}))
            if path.name.startswith(('01_','02_')):
                from tests.test_window_policy import records as window_records
                from pecvd.experiments.window_policy import WindowPolicy,WindowConfig,fit_window_policy,MODE
                train=window_records('train',2);valid=window_records('validation',2,2)
                for row in train+valid:row['step']='invented_phase'
                runtime.module.DISCOVERY_POLICY_MODE=MODE
                runtime.module.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,valid,{},
                    WindowConfig(min_train_runs=1,min_validation_runs=1)))
            # This legacy full-window fixture tests the retained detector, not
            # the separate core/q99 comparison covered by test_gmm_comparison.
            env={'display':lambda value:shown.append(value),'__name__':'__main__','RUN_GMM_COMPARISON':False}
            source='\n'.join(''.join(c['source']) for c in doc['cells'])
            self.assertNotIn('RUN_PRIVATE',source)
            self.assertNotIn('synthetic_fixture',source)
            from tests.test_threshold_comparison import notebook_comparison_fixture
            with ExitStack() as legacy_guards:
                legacy_guards.enter_context(patch('pecvd.experiments.fast_experiments.load_runtime', side_effect=AssertionError('Legacy regression must not load the active private model')))
                legacy_guards.enter_context(patch('pandas.read_csv', side_effect=AssertionError('No private CSV reads')))
                legacy_guards.enter_context(patch('torch.load', side_effect=AssertionError('No private checkpoint reads')))
                legacy_guards.enter_context(patch('numpy.load', side_effect=AssertionError('No private cache reads')))
                legacy_guards.enter_context(patch('pecvd.experiments.threshold_comparison.run_threshold_comparison', side_effect=notebook_comparison_fixture))
                legacy_guards.enter_context(patch('pecvd.experiments.gevis_system.ready_gevis_runtime', return_value=runtime))
                legacy_guards.enter_context(patch('pecvd.experiments.discovery_calibration.load_discovery_pairs', side_effect=pairs))
                legacy_guards.enter_context(patch('pecvd.experiments.reporting.ready_runtime', return_value=runtime))
                legacy_guards.enter_context(patch('pecvd.experiments.discovery_reporting.ready_runtime', return_value=runtime))
                legacy_guards.enter_context(patch('pecvd.experiments.discovery_reporting.load_cached_pairs', side_effect=pairs))
                legacy_guards.enter_context(patch('pecvd.experiments.reporting.load_cached_pairs', side_effect=pairs))
                legacy_guards.enter_context(patch('pecvd.experiments.evaluation.load_cached_pairs', side_effect=pairs))
                legacy_guards.enter_context(patch('pecvd.experiments.reporting.optional_integrations', return_value=__import__('pandas').DataFrame([{'integration': 'Merlion IsolationForest', 'installed': True, 'status': 'AVAILABLE', 'version': 'test-only'}])))
                legacy_guards.enter_context(patch('pecvd.experiments.merlion_backend.require_merlion', return_value={'merlion_version': 'test-only', 'backend': 'synthetic runtime'}))
                legacy_guards.enter_context(patch('pecvd.experiments.merlion_backend.merlion_backend', return_value={'ready': True}))
                legacy_guards.enter_context(patch('pecvd.experiments.baselines.merlion_train_array', return_value=np.zeros((32, 2))))
                legacy_guards.enter_context(patch('pecvd.experiments.baselines.merlion_scores', side_effect=lambda train, test, seed=42: np.abs(test).max(axis=1)))
                legacy_guards.enter_context(patch('pecvd.experiments.baselines.cached_train_windows', return_value=np.zeros((2, 12, 2), dtype=np.float32)))
                legacy_guards.enter_context(patch.dict('os.environ', {'GEVIS_EXTERNAL_CONFIG': 'nonexistent-synthetic-external-config.json', 'GEVIS_SOURCE_BACKBONE': 'nonexistent-synthetic-source.pt'}))
                legacy_guards.enter_context(patch('importlib.reload', side_effect=lambda module: module))
                for c in doc['cells']:
                    if c['cell_type']=='code':
                        # User-run outputs/counts are historical UI state, not
                        # an input to Run All. Source-release sanitizing is tested separately.
                        exec(compile(''.join(c['source']),str(path),'exec'),env)
            # The final report formats unavailable metrics as N/A; this makes
            # a display copy rather than preserving DataFrame object identity.
            expected=env['SUMMARY'].fillna('N/A') if 'evaluation_summary' in env['REPORT'].diagnostics else env['SUMMARY']
            __import__('pandas').testing.assert_frame_equal(shown[-1],expected)
            if path.name.startswith(('02_','03_')):
                self.assertFalse(env['RUN_NUMERICAL_DISCOVERY'])
                self.assertNotIn('THRESHOLD_COMPARISON',env)
            self.assertFalse(env['SUMMARY'].empty)
            report=env['REPORT']
            if path.name.startswith('04'):
                self.assertEqual(report.metadata['status'],'NOT_EVALUATED')
                self.assertFalse(report.metadata['model_evaluated'])
                self.assertTrue(report.prerequisites.status.eq('MISSING').all())
            else:
                self.assertTrue(report.metadata['model_evaluated'])
                if path.name.startswith(('01_','02_')):
                    if path.name.startswith('01_'):
                        self.assertFalse(runtime.calls)  # Actual decoder, never legacy named-query gates.
                        self.assertTrue(env['QUESTION_CONTRAST'].groupby(['step','pair_index']).size().eq(4).all())
                    else:self.assertFalse(runtime.calls)
                    field='truth_used_for_inference' if path.name.startswith('01_') else 'event_truth_used_for_inference'
                    self.assertFalse(report.details[field].any())
                    self.assertFalse(report.metadata['learned_answer_head_is_detector'])
                else:
                    self.assertTrue(runtime.calls)
                    self.assertTrue(any(size>1 for size in runtime.calls))
                self.assertFalse(report.by_step.empty)
                self.assertFalse(report.details.empty)
                if path.name.startswith('03'):
                    self.assertFalse(env['USE_LEARNED_ANSWERS'])
                    self.assertNotIn('GEVIS_learned_answer',set(report.summary.method))
                    self.assertIn('GEVIS_accepted_policy',set(report.summary.method))  # retained legacy fixture
                    selected=report.summary[report.summary.method.eq('Merlion_IsolationForest')]
                    self.assertEqual(list(selected.status),['EVALUATED'])
                    self.assertTrue(env['USE_MERLION'])

    def test_retired_baseline_fixture_refreshes_stale_functions_without_resetting_settings(self):
        import importlib
        from pecvd.experiments import reporting, baselines, merlion_backend
        real_reload=importlib.reload
        doc=legacy_exp_book('03_baseline_comparison.ipynb')
        cell=next(c for c in doc['cells'] if c.get('id')=='runtime-check')
        source=''.join(cell['source'])
        # Execute the real reload prefix, stopping before model evaluation.
        prefix=source.split('REPORT=run_baseline_experiment(',1)[0]
        stale=lambda *args,**kwargs: (_ for _ in ()).throw(ImportError('obsolete in-memory function'))
        env={'SEED':123,'SELECTED_STEPS':['unchanged'],'run_baseline_experiment':stale}
        try:
            reporting.run_baseline_experiment=stale
            baselines.merlion_scores=stale
            merlion_backend.require_merlion=stale
            exec(compile(prefix,'EXP3 runtime reload','exec'),env)
            self.assertIsNot(env['run_baseline_experiment'],stale)
            self.assertIs(env['run_baseline_experiment'],reporting.run_baseline_experiment)
            self.assertIsNot(baselines.merlion_scores,stale)
            self.assertIsNot(merlion_backend.require_merlion,stale)
            self.assertEqual(env['SEED'],123)
            self.assertEqual(env['SELECTED_STEPS'],['unchanged'])
            self.assertIn('compatible Merlion worker',' '.join(str(x) for x in env['run_baseline_experiment'].__code__.co_consts))
        finally:
            for module in (merlion_backend,baselines,reporting): real_reload(module)

    def test_required_merlion_failure_happens_before_loading_gevis(self):
        import pandas as pd
        from pecvd.experiments.reporting import run_baseline_experiment
        with patch('pecvd.experiments.reporting.optional_integrations',return_value=pd.DataFrame([
            {'integration':'Merlion IsolationForest','installed':False,'status':'NOT_INSTALLED'}])), \
             patch('pecvd.experiments.reporting.ready_runtime') as load:
            with self.assertRaisesRegex(ImportError,'setup_merlion.py'):
                run_baseline_experiment(ROOT,use_merlion=True)
            load.assert_not_called()

    def test_merlion_worker_rejects_invalid_input_before_process_execution(self):
        from pecvd.experiments.merlion_backend import worker_scores
        with patch('pecvd.experiments.merlion_backend.require_merlion') as require:
            with self.assertRaises(ValueError): worker_scores(np.zeros((2,2)),np.zeros((2,3)))
            with self.assertRaises(ValueError): worker_scores(np.zeros((2,2)),np.full((2,2),np.nan))
            require.assert_not_called()

    def test_merlion_worker_does_not_expose_stderr_on_failure(self):
        from pecvd.experiments.merlion_backend import worker_scores
        with patch('pecvd.experiments.merlion_backend.require_merlion',return_value={'python':'fixture-python'}), \
             patch('pecvd.experiments.merlion_backend.subprocess.run',return_value=SimpleNamespace(returncode=1,stderr=b'private fixture values',stdout=b'')):
            with self.assertRaises(RuntimeError) as error: worker_scores(np.zeros((3,2)),np.zeros((2,2)))
            self.assertNotIn('private fixture',str(error.exception))

    def test_merlion_pipe_protocol_carries_arrays_without_temp_files(self):
        from io import BytesIO
        from pecvd.experiments.merlion_backend import worker_scores
        response=BytesIO(); np.savez(response,scores=np.array([.1,.2]))
        with patch('pecvd.experiments.merlion_backend.require_merlion',return_value={'python':'fixture-python'}), \
             patch('pecvd.experiments.merlion_backend.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=response.getvalue(),stderr=b'')) as run:
            scores=worker_scores(np.ones((3,2)),np.zeros((2,2)),seed=7)
        np.testing.assert_array_equal(scores,[.1,.2])
        self.assertEqual(run.call_args.kwargs['env']['MPLBACKEND'],'Agg')
        payload=run.call_args.kwargs['input']
        with np.load(BytesIO(payload),allow_pickle=False) as sent:
            self.assertEqual(sent['seed'].item(),7)
            np.testing.assert_array_equal(sent['train'],np.ones((3,2)))
            np.testing.assert_array_equal(sent['test'],np.zeros((2,2)))

    def test_merlion_probe_failure_is_retried_and_success_is_cached(self):
        from pecvd.experiments import merlion_backend as backend
        failure=SimpleNamespace(returncode=1,stdout=b'',stderr=b'MERLION_ERROR:ValueError')
        success=SimpleNamespace(returncode=0,stdout=b'{"ready":true,"merlion_version":"fixture"}',stderr=b'')
        backend._probe.cache_clear()
        try:
            with patch.dict('os.environ',{'MPLBACKEND':'module://matplotlib_inline.backend_inline'}), \
                 patch.object(backend.subprocess,'run',side_effect=[failure,success]) as run:
                self.assertFalse(backend._probe('fixture-python')['ready'])
                self.assertEqual(backend._probe.cache_info().currsize,0)
                self.assertTrue(backend._probe('fixture-python')['ready'])
                self.assertTrue(backend._probe('fixture-python')['ready'])
                self.assertEqual(run.call_count,2)
                self.assertEqual(run.call_args.kwargs['env']['MPLBACKEND'],'Agg')
                self.assertEqual(__import__('os').environ['MPLBACKEND'],'module://matplotlib_inline.backend_inline')
        finally: backend._probe.cache_clear()

    def test_batching_reduces_runtime_calls_and_preserves_answer_count(self):
        from pecvd.experiments.evaluation import ask_batched
        runtime,pairs=fake_runtime(); current,reference=pairs(runtime,'invented_phase',limit=1)[0]
        questions=runtime_questions('invented_phase','a',runtime.module.GevisQuestion,bank=True,limit=10)
        result=ask_batched(runtime,current,reference,'invented_phase',questions,batch_size=4)
        self.assertEqual(len(result),10)
        self.assertEqual(runtime.calls,[4,4,2])

    def test_injection_summary_excludes_ineffective_changes(self):
        from pecvd.experiments.reporting import injection_summary
        rows=[dict(kind='drift',injected=True,effective=True,alert=True,candidate_alert=True,
                   top_sensor_correct=True,best_alert_interval_iou=.5,accepted_question_share=.8,latency_ms=2.),
              dict(kind='drift',injected=True,effective=False,alert=False,candidate_alert=False,
                   top_sensor_correct=False,best_alert_interval_iou=None,accepted_question_share=.8,latency_ms=2.),
              dict(kind='untouched',injected=False,effective=None,alert=False,candidate_alert=False,
                   top_sensor_correct=None,best_alert_interval_iou=None,accepted_question_share=.8,latency_ms=2.)]
        summary=injection_summary(rows)
        self.assertEqual(summary.iloc[0].effective_cases,1)
        self.assertEqual(summary.iloc[0].detection_recall,1.)
        self.assertEqual(summary.iloc[0].untouched_alert_rate,0.)

    def test_current_runtime_cell_ids_are_stable(self):
        for name in ('STEP_3_train_gevis_decision.ipynb','STEP_3.1_run_gevis_inference.ipynb','CVD_evaluate.ipynb'):
            current=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
            by_id={c['id']:c for c in current['cells']}
            self.assertEqual(len(by_id),len(current['cells']))
            for cell in current['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
            if name.startswith('STEP_3.1'):
                self.assertTrue({'03-types','03-load','03-function'}<=by_id.keys())


if __name__=='__main__': unittest.main()
