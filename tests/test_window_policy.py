import contextlib,copy,io,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from pecvd.experiments.window_policy import WindowConfig,WindowPolicy,OPERATIONS,fit_window_policy,operation_traces,merged_intervals,MODE,SCHEMA


def records(split,count,seed=0):
    rng=np.random.default_rng(seed);t=np.linspace(0,4*np.pi,96)
    base=np.stack([np.sin(t),np.sin(t+.3)],1)
    rows=[]
    for i in range(count):
        a=base+rng.normal(0,.03,base.shape);b=base+rng.normal(0,.015,base.shape)
        rows.append({'split':split,'step':'fixture','run_id':f'{split}-{i}',
            'reference_run_ids':['train-0' if split!='train' or i else 'train-1'],
            'sensors':['a','b'],'current':a,'reference':b,'residual':np.abs(a-b)*.1})
    return rows


def fitted():
    cfg=WindowConfig(min_train_runs=8,min_validation_runs=4)
    return WindowPolicy(fit_window_policy(records('train',16),records('validation',12,seed=7),{'fixture':True},cfg))


class WindowPolicyTests(unittest.TestCase):
    def test_center_review_keeps_context_and_separates_unrelated_changes(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery import review_changes
        runtime,current,reference,_=fixture();m=runtime.module
        cfg=WindowConfig(min_train_runs=1,min_validation_runs=1)
        train=records('train',2);val=records('validation',2,2)
        for row in train+val:row['step']='invented_phase'
        m.DISCOVERY_POLICY_MODE=MODE;m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,val,{},cfg))
        current.loc[8,'b']=20.
        result=review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1),center_sensor='a')
        self.assertEqual(result['context_sensors'],['a','b'])
        self.assertFalse(result['self_analysis']['changed'])
        self.assertFalse(result['relationship_analysis']['changed'])
        self.assertTrue(all(row['sensor']=='a' for row in result['ranked_evidence']))
        self.assertNotIn('a',result['answers'][-1]['probabilities'])
        changed=review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1),center_sensor='b')
        self.assertTrue(changed['self_analysis']['changed'])
        self.assertEqual(changed['answers'][0]['center_analysis']['center_sensor'],'b')
        with self.assertRaises(ValueError):review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1),center_sensor='missing')
        # Relation changes can exist while the center's own waveform is identical.
        t=np.linspace(0,4*np.pi,96);reference_pair=np.stack([np.sin(t),np.sin(t)],1)
        changed_pair=reference_pair.copy();changed_pair[:,1]*=-1
        traces,_=operation_traces(changed_pair,reference_pair,np.zeros_like(changed_pair),cfg)
        np.testing.assert_allclose(traces[:,0,:-1],0.,atol=1e-10)
        self.assertGreater(traces[:,0,-1].max(),1.)

    def test_fit_excludes_test_and_detects_run_or_reference_leakage(self):
        train=records('train',8);validation=records('validation',4,1)
        with self.assertRaisesRegex(ValueError,'partition'):fit_window_policy(train,records('test',4),{})
        bad=copy.deepcopy(validation);bad[0]['run_id']='train-0'
        with self.assertRaisesRegex(ValueError,'overlap'):fit_window_policy(train,bad,{})
        bad=copy.deepcopy(validation);bad[0]['reference_run_ids']=['validation-1']
        with self.assertRaisesRegex(ValueError,'Train only'):fit_window_policy(train,bad,{})
        bad=copy.deepcopy(train);bad[0]['reference_run_ids']=['train-0']
        with self.assertRaisesRegex(ValueError,'Self-run'):fit_window_policy(bad,validation,{})

    def test_window_threshold_uses_validation_run_max_not_duplicate_window_count(self):
        train=records('train',12);val=records('validation',6,2)
        cfg=WindowConfig(min_validation_runs=4)
        original=fit_window_policy(train,val,{},cfg)
        repeated=fit_window_policy(train,val*10,{},cfg)
        group=original['groups']['fixture'];other=repeated['groups']['fixture']
        self.assertEqual(group['window_threshold'],other['window_threshold'])
        self.assertEqual(other['validation_runs'],6)
        self.assertAlmostEqual(other['control_rate_resolution'],1/6)
        self.assertTrue(group['insufficient_controls_for_nominal_rate'])
        # Calibration controls report their measured rate, not a promised 5%.
        self.assertEqual(group['validation_window_alert_rate'],0.)
        self.assertFalse(original['population_false_alarm_guarantee'])
        val[0]['residual']+=100
        revised=fit_window_policy(train,val,{},cfg)
        self.assertGreater(revised['groups']['fixture']['window_threshold'],group['window_threshold'])

    def test_operation_specific_variance_trace_spans_event_instead_of_one_peak(self):
        cfg=WindowConfig();rng=np.random.default_rng(3);b=rng.normal(0,.05,(128,2));a=b.copy()
        a[32:96,0]+=np.tile([-1.,1.],32)
        trace,_=operation_traces(a,b,np.zeros_like(a),cfg)
        score=trace[:,0,OPERATIONS.index('variability')]
        intervals=merged_intervals(score>.3,1,2)
        best=max(intervals,key=lambda iv:iv[1]-iv[0])
        intersection=max(0,min(best[1],96)-max(best[0],32))
        self.assertGreater(intersection/(max(best[1],96)-min(best[0],32)),.8)
        self.assertGreater(trace[40:88,0,5].max(),-1.)  # valid dedicated stuck slot

    def test_delay_and_relationship_have_distinct_local_traces_and_pair_metadata(self):
        t=np.arange(128);b=np.stack([np.sin(t*.4),np.sin(t*.4+.1)],1);a=b.copy()
        a[32:96,0]=b[np.arange(32,96)-3,0]
        trace,aux=operation_traces(a,b,np.zeros_like(a))
        self.assertGreater(trace[40:88,0,6].mean(),.3)
        self.assertEqual(int(np.median(aux['lag_rows'][40:88,0])),-3)
        a=b.copy();a[32:96,0]=np.random.default_rng(7).permutation(a[32:96,0])
        trace,aux=operation_traces(a,b,np.zeros_like(a))
        self.assertGreater(trace[40:88,0,7].mean(),.2)
        np.testing.assert_allclose(aux['relationship'][:,0,1],aux['relationship'][:,1,0])

    def test_insufficient_calibration_abstains_and_no_change_is_consistent(self):
        payload=fit_window_policy(records('train',8),records('validation',2),{})
        policy=WindowPolicy(payload);a=np.zeros((96,2))
        result=policy.analyze(a,a,a,'fixture',['a','b'])
        self.assertFalse(result['accepted']);self.assertIsNone(result['changed'])
        policy=fitted();result=policy.analyze(a,a,a,'fixture',['a','b'])
        self.assertFalse(result['changed']);self.assertIsNone(result['winner'])
        with self.assertRaises(ValueError):policy.analyze(a,a,a,'fixture',['b','a'])

    def test_new_runtime_shares_exact_winner_for_all_typed_outputs_and_never_calls_attention(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery import review_changes
        runtime,current,reference,_=fixture();m=runtime.module
        cfg=WindowConfig(min_train_runs=1,min_validation_runs=1)
        train=records('train',2);val=records('validation',2,2)
        for r in train+val:r['step']='invented_phase'
        m.DISCOVERY_POLICY_MODE=MODE;m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,val,{},cfg))
        current.loc[8,'b']=20.
        with patch.object(m,'ask_gevis',side_effect=AssertionError('No learned routing in blind discovery')):
            result=review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1))
        self.assertTrue(result['changed']);self.assertEqual(result['contract'],SCHEMA)
        self.assertEqual(result['winner']['sensor'],'b')
        self.assertTrue(all(a['evidence'] is result['winner'] for a in result['answers'][:3]))
        self.assertEqual(result['answers'][-1]['answer_type'],'CHOICE_SENSOR')
        self.assertEqual(set(result['answers'][-1]['probabilities']),{'a','b'})
        self.assertTrue(all(a['accepted'] for a in result['answers']))
        self.assertEqual(result['answers'][0]['answer'],'Yes')
        self.assertNotEqual(result['answers'][1]['answer'],'No clear change')
        self.assertIn('insufficient_controls_for_nominal_window_rate',result['warnings'])
        m.DISCOVERY_POLICY=None
        with self.assertRaisesRegex(RuntimeError,'STEP 3'):review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1))

    def test_current_metadata_cannot_supply_event_sensor_or_interval(self):
        policy=fitted();r=records('test',1,4)[0];r['current'][30:60,1]+=2.
        before=policy.analyze(r['current'],r['reference'],r['residual'],'fixture',['a','b'])
        r.update(expected_sensor='a',injection_interval=(0,1),kind='delay')
        after=policy.analyze(r['current'],r['reference'],r['residual'],'fixture',['a','b'])
        self.assertEqual(before,after)

    def test_calibrated_report_uses_train_reference_pairs_and_keeps_iou_truth_outside_detector(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery_reporting import run_discovery_experiment
        from pecvd.experiments.discovery import review_changes
        runtime,current,reference,_=fixture();m=runtime.module
        train=records('train',8);val=records('validation',4)
        for r in train+val:r['step']='invented_phase'
        m.DISCOVERY_POLICY_MODE=MODE;m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,val,{}))
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(current,reference)]) as pairs,patch('pecvd.experiments.discovery_reporting.load_cached_pairs',side_effect=AssertionError('Old same-split reference protocol')),patch('pecvd.experiments.discovery_reporting.review_changes',wraps=review_changes) as call,contextlib.redirect_stdout(io.StringIO()):
            report=run_discovery_experiment('.',runtime=runtime,max_cases=2)
        self.assertEqual(pairs.call_args.kwargs['split'],'test')
        self.assertEqual(report.metadata['discovery_contract'],SCHEMA)
        self.assertIn('window_calibration',report.diagnostics)
        self.assertTrue(report.details.typed_consistent.all())
        for args in call.call_args_list:self.assertEqual(len(args.args),5);self.assertFalse(args.kwargs)


class CalibrationCacheTests(unittest.TestCase):
    def test_retired_window_fixture_prepares_create_only_policy_from_synthetic_caches(self):
        import hashlib
        root_repo=Path(__file__).resolve().parents[1]
        notebook=json.loads((root_repo/'tests/fixtures/step3_legacy_diagnostics.json').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in notebook['cells'] if c.get('id')=='window-evidence-build'))
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);eroot=root/'artifacts/encoders';(eroot/'windows').mkdir(parents=True)
            backbone=root/'artifacts/backbone/backbone.pt';backbone.parent.mkdir();backbone.write_bytes(b'synthetic backbone identity')
            encoder_path=eroot/'fixture.pt';encoder_path.write_bytes(b'synthetic encoder identity')
            manifest={'step_checkpoints':{'fixture':{'path':'fixture.pt'}}}
            (eroot/'encoder_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            arrays={};rng=np.random.default_rng(4)
            for split,count in [('train',8),('validation',4),('test',4)]:
                arrays[split]=rng.normal(0,.1,(count,16,2)).astype('float32')
                arrays[split+'_process_id']=np.array([f'{split}-{i}' for i in range(count)])
                arrays[split+'_position_bin']=np.zeros(count,int);arrays[split+'_cohort_key']=np.array(['fixture']*count)
            np.savez(eroot/'windows/fixture.npz',**arrays)
            encoder=torch.nn.Linear(2,2)
            def numeric_state(model,a,b):
                state=torch.zeros(1,8,8,device=a.device)
                state[0,:,-2]=((a-b).square().reshape(1,4,4,2).mean(2).permute(0,2,1).reshape(-1))
                return state
            # Preserve this q95/full-window preparation regression; direct GMM
            # preparation is exercised separately with the canonical core profile.
            env={'RUN_STRUCTURED_EVALUATION':True,'RUN_GMM_COMPARISON':False,'encoders':{'fixture':(encoder,{'sensor_columns':['a','b']})},
                'ENCODER_ROOT':eroot,'backbone_path':backbone,'DEVICE':torch.device('cpu'),'numeric_state':numeric_state,
                'SEED':42,'display':lambda *a:None,'PIPELINE_SETTINGS':{'active_quantile':.95}}
            with patch('pandas.read_csv',side_effect=AssertionError('No CSV parsing')),patch.dict('os.environ',{'GEVIS_DISCOVERY_POLICY_PATH':str(backbone.parent/'window.json')}):
                # Reproduce a notebook kernel that imported the module before
                # activate_prepared_policy was added on disk.
                from pecvd.experiments import pipeline_settings
                del pipeline_settings.activate_prepared_policy
                exec(compile(source,'STEP3:actual-window-build','exec'),env)
                self.assertTrue(callable(pipeline_settings.activate_prepared_policy))
                import os
                self.assertEqual(Path(os.environ['GEVIS_DISCOVERY_POLICY_PATH']),env['WINDOW_POLICY_PATH'])
                self.assertEqual(env['WINDOW_POLICY_PATH'].name,'window_evidence_compare_q95_v1.json')
                from pecvd.experiments.pipeline_settings import comparison_policy_path
                paths=[comparison_policy_path(backbone,q) for q in (.95,.99)]
                before=[p.read_bytes() for p in paths]
                exec(compile(source,'STEP3:actual-window-reuse','exec'),env)
                self.assertEqual(before,[p.read_bytes() for p in paths])
                # Old validator in a warm kernel, new on-disk v4 profile.
                import pecvd.experiments.window_policy as window_module
                actual_validate=window_module.WindowConfig.validate
                def old_validate(config):
                    if config.gate_scope=='position_rank_v4':raise ValueError('Unknown gate scope')
                    return actual_validate(config)
                env['PIPELINE_SETTINGS'].update(gate_profile='position_rank_v4',threshold_multiplier=1.)
                with patch.object(window_module.WindowConfig,'validate',old_validate):
                    with self.assertRaisesRegex(ValueError,'Unknown gate scope'):
                        window_module.WindowConfig(gate_scope='position_rank_v4').validate()
                    exec(compile(source,'STEP3:warm-v4-preparation','exec'),env)
                self.assertEqual(env['WINDOW_CONFIG'].gate_scope,'position_rank_v4')
                self.assertEqual(env['WINDOW_POLICY'].payload['gate_contract'],'position_rank_v4')
                self.assertIn('position_rank_v4',env['WINDOW_POLICY_PATH'].name)
                self.assertEqual(before,[p.read_bytes() for p in paths])
            self.assertEqual(env['WINDOW_POLICY'].payload['groups']['fixture']['validation_runs'],4)
            self.assertFalse(env['WINDOW_POLICY'].payload['test_used_for_fitting'])
            self.assertTrue(all(p.grad is None for p in encoder.parameters()))
            # Actual STEP 3.1 wrapper must pass the same policy to discovery.
            from tests.test_shared_discovery import fixture
            import ast
            runtime,current,reference,_=fixture();m=runtime.module
            rows=records('train',8)+records('validation',4)
            for r in rows:r['step']='invented_phase'
            policy=WindowPolicy(fit_window_policy(rows[:8],rows[8:],{}))
            doc=json.loads((root_repo/'notebooks/STEP_3.1_run_gevis_inference.ipynb').read_text(encoding='utf-8'))
            code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='03-function'))
            fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='review_changes')
            scope={**vars(m),'DISCOVERY_POLICY_MODE':MODE,'DISCOVERY_POLICY':policy}
            exec(compile(ast.Module([fn],type_ignores=[]),'STEP31:actual-review-wrapper','exec'),scope)
            current.loc[8,'b']=20
            result=scope['review_changes'](current,reference,'invented_phase',m.ReferenceInfo('synthetic',1))
            self.assertEqual(result['contract'],SCHEMA)
            self.assertTrue(result['changed'])

    def test_cache_loader_references_train_and_preparation_never_reads_test_values(self):
        from pecvd.experiments.discovery_calibration import cached_comparisons
        values={};shape=(4,16,2)
        for split in ('train','validation','test'):
            values[split]=np.ones(shape)*({'train':1.,'validation':100.,'test':999.}[split])
            values[split+'_process_id']=np.array([f'{split}{i}' for i in range(4)])
            values[split+'_position_bin']=np.zeros(4,int);values[split+'_cohort_key']=np.array(['fixture']*4)
        class Guard:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,key):
                if key=='test':raise AssertionError('Test values entered calibration')
                return values[key]
        with patch('pecvd.experiments.discovery_calibration.np.load',return_value=Guard()):
            result=list(cached_comparisons('.', 'fixture',split='validation'))
        self.assertEqual(len(result),4)
        self.assertTrue(all(np.all(r['reference']==1) for r in result))
        self.assertTrue(all(all(v.startswith('train') for v in r['reference_run_ids']) for r in result))

    def test_identity_load_rejects_changes_and_attach_does_not_fit(self):
        from pecvd.experiments.discovery_calibration import load_policy,attach_policy
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'policy.json';payload=fitted().payload
            path.write_text(json.dumps(payload),encoding='utf-8')
            self.assertEqual(load_policy(path,payload['identity']).payload,payload)
            with self.assertRaises(ValueError):load_policy(path,{'changed':True})
            module=SimpleNamespace(BACKBONE_PATH=Path(folder)/'backbone.pt',ENCODER_ROOT=Path(folder)/'encoders')
            with patch.dict('os.environ',{'GEVIS_DISCOVERY_MODE':MODE,'GEVIS_DISCOVERY_POLICY_PATH':str(Path(folder)/'missing.json')}),patch('pecvd.experiments.discovery_calibration.prepare_policy',side_effect=AssertionError('No runtime fitting')):
                attach_policy(module)
            self.assertIsNone(module.DISCOVERY_POLICY)


if __name__=='__main__':unittest.main()
