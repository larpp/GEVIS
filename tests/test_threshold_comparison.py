import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tests.test_window_policy import records
from pecvd.experiments.threshold_comparison import fit_comparison,comparison_table,run_threshold_comparison
from pecvd.experiments.window_policy import WindowConfig

def notebook_comparison_fixture(*args,**kwargs):
    """Orchestration stub only; actual threshold math has independent tests."""
    frame=pd.DataFrame({'nominal_quantile':[.95,.99]})
    return SimpleNamespace(summary=frame,by_step=frame,
        diagnostics={'thresholds':frame,'by_kind_and_strength':frame},metadata={'synthetic_fixture':True})

class ThresholdComparisonTests(unittest.TestCase):
    def test_step4_prepares_with_step3_config_and_attaches_active_quantile(self):
        from pecvd.experiments.threshold_comparison import prepare_runtime_comparison
        from pecvd.experiments.window_policy import MODE
        module=SimpleNamespace(ENCODER_ROOT='encoders',BACKBONE_PATH='backbone.pt',
            DEVICE='cpu',numeric_state=object(),DISCOVERY_POLICY=None)
        runtime=SimpleNamespace(module=module,encoders={'fixture':object()})
        settings={'active_quantile':.99,'gate_profile':'position_rank_v4','operation_profile':'core',
            'threshold_multiplier':1.,'max_calibration_windows_per_step':2048}
        policies={q:object() for q in (.90,.95,.99)}
        with patch('pecvd.experiments.threshold_comparison.prepare_comparison',return_value=policies) as prepare, \
             patch('pecvd.experiments.pipeline_settings.activate_prepared_policy',return_value=Path('new-policy.json')) as activate:
            self.assertIs(prepare_runtime_comparison(runtime,settings),policies)
        from dataclasses import asdict
        self.assertEqual(asdict(prepare.call_args.kwargs['config']),asdict(WindowConfig(gate_scope='position_rank_v4',operation_profile='core',nominal_window_alert_rate=1-.99)))
        self.assertEqual(prepare.call_args.args,(runtime.encoders,'encoders','backbone.pt','cpu',module.numeric_state))
        activate.assert_called_once_with('backbone.pt',.99)
        self.assertIs(module.DISCOVERY_POLICY,policies[.99])
        self.assertEqual(module.DISCOVERY_POLICY_MODE,MODE)
        self.assertEqual(module.DISCOVERY_POLICY_PATH,Path('new-policy.json'))

    def test_step4_preparation_rejects_external_or_invalid_config_before_io(self):
        from pecvd.experiments.threshold_comparison import prepare_runtime_comparison
        with patch('pecvd.experiments.threshold_comparison.prepare_comparison') as prepare:
            with self.assertRaisesRegex(ValueError,'External'):
                prepare_runtime_comparison(SimpleNamespace(external_pairs=[]),{'active_quantile':.99})
            with self.assertRaises(ValueError):
                prepare_runtime_comparison(SimpleNamespace(),{'active_quantile':.99,'gate_profile':'invalid'})
            prepare.assert_not_called()

    def test_namespace_tracks_all_load_identity_sources_and_preserves_old_policy(self):
        from pecvd.experiments.pipeline_settings import comparison_policy_path
        from pecvd.experiments.discovery_calibration import policy_source_paths
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'GEVIS_GATE_PROFILE':'position_rank_v4'}):
            root=Path(temp);backbone=root/'artifacts/model/backbone.pt'
            dependencies=[]
            for actual in policy_source_paths():
                fixture=root/'sources'/actual.name;fixture.parent.mkdir(exist_ok=True)
                fixture.write_text('original synthetic source',encoding='utf-8');dependencies.append(fixture)
            with patch('pecvd.experiments.discovery_calibration.policy_source_paths',return_value=tuple(dependencies)):
                original=comparison_policy_path(backbone,.99)
                original.parent.mkdir(parents=True);original.write_text('preserved policy',encoding='utf-8')
                for dependency in dependencies:
                    dependency.write_text('changed synthetic source',encoding='utf-8')
                    self.assertNotEqual(original,comparison_policy_path(backbone,.99),dependency.name)
                    dependency.write_text('original synthetic source',encoding='utf-8')
                self.assertEqual(original,comparison_policy_path(backbone,.99))
                self.assertEqual(original.read_text(encoding='utf-8'),'preserved policy')

    def test_namespace_tracks_numeric_source_not_notebook_outputs(self):
        from pecvd.experiments.pipeline_settings import comparison_policy_path
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'GEVIS_GATE_PROFILE':'position_rank_v4'}):
            root=Path(temp);notebook=root/'notebooks/STEP_3.1_run_gevis_inference.ipynb'
            notebook.parent.mkdir()
            doc={'cells':[{'id':'03-types','source':['x=1\n'],'outputs':[]}]}
            notebook.write_text(json.dumps(doc),encoding='utf-8')
            backbone=root/'artifacts/model/backbone.pt';before=comparison_policy_path(backbone,.99)
            doc['cells'][0]['outputs']=[{'text':'synthetic output'}]
            notebook.write_text(json.dumps(doc),encoding='utf-8')
            self.assertEqual(before,comparison_policy_path(backbone,.99))
            doc['cells'][0]['source']=['x=2\n'];notebook.write_text(json.dumps(doc),encoding='utf-8')
            self.assertNotEqual(before,comparison_policy_path(backbone,.99))

    def test_invalid_scope_fails_before_cache_identity_or_encoder_execution(self):
        from pecvd.experiments.threshold_comparison import prepare_comparison
        with patch('pecvd.experiments.threshold_comparison.identity_for',side_effect=AssertionError('No artifact reads')):
            with self.assertRaisesRegex(ValueError,"Unknown gate scope: 'invalid'"):
                prepare_comparison({},'unused','unused','cpu',lambda *a:None,config=WindowConfig(gate_scope='invalid'))

    def test_explicit_relaxation_changes_actual_gate_without_rescaling_inputs(self):
        from dataclasses import replace
        from pecvd.experiments.window_policy import fit_window_policy,WindowPolicy,OPERATIONS
        cfg=WindowConfig();train=records('train',12);val=records('validation',16,2)
        original=fit_window_policy(train,val,{},cfg)
        relaxed=fit_window_policy(train,val,{},replace(cfg,threshold_multiplier=.9))
        a=original['groups']['fixture'];b=relaxed['groups']['fixture']
        np.testing.assert_array_equal(a['scales'],b['scales'])
        self.assertAlmostEqual(b['window_threshold'],a['window_threshold']*.9)
        self.assertEqual(b['calibrated_window_threshold'],a['window_threshold'])
        # Controlled boundary fixture: 9.5 crosses relaxed 9, not original 10.
        for payload,bound in ((original,10.),(relaxed,9.)):
            g=payload['groups']['fixture'];g['scales']=np.ones((len(g['sensors']),len(OPERATIONS))).tolist()
            g.update(window_threshold=bound,calibrated_window_threshold=10.,ordinal_edges=[bound,11.,12.,13.])
        shape=(32,len(a['sensors']));current=np.full(shape,9.5);zero=np.zeros(shape)
        self.assertFalse(WindowPolicy(original).analyze(current,zero,zero,'fixture',a['sensors'])['changed'])
        self.assertTrue(WindowPolicy(relaxed).analyze(current,zero,zero,'fixture',a['sensors'])['changed'])
        for invalid in (0.,-1.,1.1,float('nan')):
            with self.assertRaises(ValueError):replace(cfg,threshold_multiplier=invalid).validate()

    def test_center_profile_policy_namespace_binds_relaxation_setting(self):
        from pecvd.experiments.pipeline_settings import comparison_policy_path
        with patch.dict(os.environ,{'GEVIS_GATE_PROFILE':'center_views_v3','GEVIS_THRESHOLD_MULTIPLIER':'1.0'}):
            original=comparison_policy_path('fixture/backbone.pt',.9)
            os.environ['GEVIS_THRESHOLD_MULTIPLIER']='0.9'
            self.assertNotEqual(original,comparison_policy_path('fixture/backbone.pt',.9))

    def test_90_95_99_use_same_scales_and_disclose_sixteen_run_rank(self):
        policies=fit_comparison(records('train',12),records('validation',16,2),lambda c:{})
        self.assertEqual(tuple(policies),(.90,.95,.99))
        groups=[p.payload['groups']['fixture'] for p in policies.values()]
        self.assertEqual([g['used_order_statistic'] for g in groups],[16,16,16])
        self.assertEqual(len({g['window_threshold'] for g in groups}),1)
        for g in groups:
            np.testing.assert_array_equal(g['scales'],groups[0]['scales'])
            self.assertIn(g['threshold_driver_operation'],('level','variability','slope','local_change','residual','stuck','delay','relationship'))
            self.assertGreater(g['threshold_driver_scale'],0)

    def test_paired_diagnosis_compares_same_operation_not_different_winners(self):
        from pecvd.experiments.discovery_reporting import paired_operation_scores,paired_change_diagnosis
        def event(op,score):
            return dict(sensor='center',statistic=op,time_scope='whole',normalized_score=score,
                window_threshold=10.,changed=score>10.,normalization_scale=2.,active_gate='fixture',raw_score=score*2)
        before=[event('residual',9.),event('level',1.)]
        after=[event('level',4.),event('residual',8.)]
        rows=paired_operation_scores(before,after,step='phase',pair_id='pair',center='center',kind='level_shift',strength=.5,effective=True)
        frame=pd.DataFrame(rows)
        self.assertEqual(frame.set_index('operation').loc['level','score_delta'],3.)
        self.assertEqual(paired_change_diagnosis(frame).diagnosis.iloc[0],'SCORE_RESPONSE_BLOCKED_BY_GATE')
        after[0]=event('level',11.)
        rows=paired_operation_scores(before,after,step='phase',pair_id='pair',center='center',kind='level_shift',strength=.5,effective=True)
        self.assertEqual(paired_change_diagnosis(pd.DataFrame(rows)).diagnosis.iloc[0],'NEW_ALERT_AFTER_INJECTION')

    def test_relation_alert_is_not_substituted_for_self_detection(self):
        from pecvd.experiments.discovery_reporting import paired_operation_scores,paired_change_diagnosis
        def event(op,score):
            return dict(sensor='center',statistic=op,time_scope='whole',normalized_score=score,
                window_threshold=10.,changed=score>10.)
        rows=paired_operation_scores([event('level',1.),event('relationship',1.)],
            [event('level',1.),event('relationship',12.)],step='phase',pair_id='pair',center='center',kind='level_shift',strength=.5,effective=True)
        result=paired_change_diagnosis(pd.DataFrame(rows)).iloc[0]
        self.assertFalse(result.after_alert);self.assertTrue(result.relationship_after_alert)
        self.assertEqual(result.diagnosis,'NO_NUMERICAL_SCORE_INCREASE')

    def test_noop_and_preexisting_alert_are_not_new_detection_success(self):
        from pecvd.experiments.discovery_reporting import paired_operation_scores,paired_change_diagnosis
        e=dict(sensor='a',statistic='level',time_scope='whole',normalized_score=11.,window_threshold=10.,changed=True)
        def diagnose(effective):
            rows=paired_operation_scores([e],[e],step='phase',pair_id='p',center='a',kind='stuck',strength=2.,effective=effective)
            return paired_change_diagnosis(pd.DataFrame(rows)).diagnosis.iloc[0]
        self.assertEqual(diagnose(False),'NOT_EVALUATED_NUMERIC_NOOP')
        self.assertEqual(diagnose(True),'ALREADY_ALERTING_BEFORE_INJECTION')

    def test_activation_uses_prepared_backbone_not_stale_environment(self):
        from pecvd.experiments.pipeline_settings import activate_prepared_policy,comparison_policy_path
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'GEVIS_DISCOVERY_POLICY_PATH':'stale.json'}):
            backbone=Path(directory)/'gevis_backbone.pt'
            with self.assertRaisesRegex(FileNotFoundError,'window-evidence-build'):
                activate_prepared_policy(backbone,.99)
            self.assertEqual(os.environ['GEVIS_DISCOVERY_POLICY_PATH'],'stale.json')
            path=comparison_policy_path(backbone,.99)
            path.touch()  # Synthetic presence only; runtime identity checking is separate.
            self.assertEqual(activate_prepared_policy(backbone,.99),path.resolve())
            self.assertEqual(Path(os.environ['GEVIS_DISCOVERY_POLICY_PATH']),path.resolve())
            self.assertEqual(os.environ['GEVIS_DISCOVERY_MODE'],'window_calibrated_v2')

    def test_small_control_count_discloses_identical_thresholds(self):
        policies=fit_comparison(records('train',12),records('validation',8,1),lambda c:{'fixture':True})
        table=comparison_table(policies)
        self.assertTrue(table.thresholds_identical.all());self.assertTrue(table.insufficient_tail_support.all())
        np.testing.assert_array_equal(policies[.95].payload['groups']['fixture']['scales'],policies[.99].payload['groups']['fixture']['scales'])
        self.assertFalse(table.verified_fault_probability.any())

    def test_more_controls_monotonic_and_test_never_fits(self):
        policies=fit_comparison(records('train',12),records('validation',120,2),lambda c:{'fixture':True})
        a=policies[.95].payload['groups']['fixture']['window_threshold'];b=policies[.99].payload['groups']['fixture']['window_threshold']
        self.assertGreaterEqual(b,a)
        with self.assertRaises(ValueError):fit_comparison(records('train',12),records('test',8),lambda c:{})
        with self.assertRaises(ValueError):fit_comparison(records('train',12),records('validation',8),lambda c:{},WindowConfig(scale_quantile=.99))

    def test_runtime_restores_policy_even_on_failure(self):
        policies=fit_comparison(records('train',12),records('validation',8,1),lambda c:{})
        original=policies[.95]
        runtime=SimpleNamespace(module=SimpleNamespace(DISCOVERY_POLICY=original,ENCODER_ROOT=Path('fake'),BACKBONE_PATH=Path('fake')))
        with patch('pecvd.experiments.threshold_comparison.load_comparison',return_value=policies),patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',side_effect=RuntimeError('fixture')):
            with self.assertRaisesRegex(RuntimeError,'fixture'):run_threshold_comparison('.',runtime=runtime)
        self.assertIs(runtime.module.DISCOVERY_POLICY,original)

    def test_profile_namespaces_do_not_replace_original_encoder_or_backbone(self):
        from pecvd.model_variants import active_run_tag,active_config
        from pecvd.process_config import load_process_config
        root=Path(__file__).resolve().parents[1]
        with patch.dict(os.environ,{'GEVIS_MODEL_VARIANT':'hybrid','GEVIS_DISCOVERY_TRAINING':'reference_v11'}):
            before=active_run_tag();encoder=active_config(load_process_config(root/'config/processes/pecvd.json')).artifacts.encoder
            os.environ['GEVIS_DISCOVERY_TRAINING']='contrastive_v12'
            self.assertEqual(active_run_tag(),before+'_contrastive_v12')
            self.assertEqual(encoder,active_config(load_process_config(root/'config/processes/pecvd.json')).artifacts.encoder)
            os.environ['GEVIS_DISCOVERY_TRAINING']='fixed_questions_v13'
            self.assertEqual(active_run_tag(),before+'_fixed_questions_v13')
            os.environ['GEVIS_DISCOVERY_TRAINING']='center_relation_v14'
            self.assertEqual(active_run_tag(),before+'_center_relation_v14')
            self.assertEqual(encoder,active_config(load_process_config(root/'config/processes/pecvd.json')).artifacts.encoder)

    def test_all_notebooks_share_profile_and_only_discovery_runs_threshold_comparison(self):
        from pecvd.experiments.pipeline_audit import NOTEBOOKS
        root=Path(__file__).resolve().parents[1]
        for name in NOTEBOOKS:
            doc=json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
            code=[c for c in doc['cells'] if c['cell_type']=='code']
            if name in ('CVD_predict_external_csv.ipynb','ALD_predict_external_csv.ipynb'):
                # External replay configures the source profile inside its
                # strict load-only loader; EXP7 locks the protocol first.
                source='\n'.join(''.join(c['source']) for c in code)
                self.assertIn('external.load_peald(ROOT' if name.startswith('ALD_') else 'fast.load_runtime(ROOT)',source)
                self.assertIn("reference_mode='train_bank'" if name.startswith('ALD_') else 'original_path=None',source)
                continue
            self.assertEqual(code[0]['id'],'shared-pipeline-settings')
            self.assertIn('PRIMARY_GOAL_VERSION',''.join(code[0]['source']))
            if name=='STEP_3.1_run_gevis_inference.ipynb':
                runtime=''.join(next(c['source'] for c in code if c.get('id')=='03-function'))
                self.assertIn('center_sensor=center_sensor',runtime)
            if name in ('CVD_evaluate.ipynb','CVD_evaluate.ipynb'):
                self.assertFalse(any(c.get('id')=='paired-threshold-comparison' for c in code))
