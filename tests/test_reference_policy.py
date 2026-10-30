import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from pecvd.experiments.reference_policy import (ReferenceConfig,ReferencePolicy,fit_records,save_policy,
    load_policy,policy_features,cached_records,validation_alert_rates)


def records(n=600,*,split='train',cohort='recipe'):
    rng=np.random.default_rng(13)
    for i in range(n):
        values=rng.normal(0,1,(3,2,5)); values[...,3]=np.abs(values[...,3])
        yield {'split':split,'step':'phase','run_id':f'run_{i}','cohort':cohort,'position':0,
               'sensors':['pressure','flow'],'features':values}


class ReferencePolicyTests(unittest.TestCase):
    def test_step3_first_code_cell_defines_evaluation_switch_in_fresh_namespace(self):
        import sys,types
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        first=next(c for c in doc['cells'] if c['cell_type']=='code')
        self.assertEqual(first.get('id'),'shared-pipeline-settings')
        first=next(c for c in doc['cells'] if c.get('id')=='02-config')
        module=types.ModuleType('step3_order_test')
        with patch.dict(sys.modules,{module.__name__:module}):
            exec(compile(''.join(first['source']),'synthetic:step3-first-cell','exec'),module.__dict__)
        self.assertIs(module.RUN_STRUCTURED_EVALUATION,True)
        self.assertFalse(module.RUN_TRAINING)
        self.assertEqual(module.SESSION_SIGNATURE,'STEP3_GEVIS_PRETRAINED_POLICY_V2')

    def test_active_step3_ends_with_fusion_and_excludes_retired_numerical_cells(self):
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        ids=[c.get('id') for c in doc['cells']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(ids[-1],'fusion-answer-train')
        for dependency in ('02-config','02-load','02-data','02-functions','02-train'):
            self.assertLess(ids.index(dependency),ids.index('fusion-answer-train'))
        retired={'reference-policy-build','window-evidence-build','temporal-localization-build',
                 'learned-center-fit','02-eval','02-structured-diagnostic','boundary-comparison',
                 'boundary-comparison-guide','paired-threshold-comparison'}
        self.assertFalse(set(ids)&retired)
        self.assertNotIn('Default inference remains q99',json.dumps(doc))
        self.assertNotIn('run_boundary_comparison',json.dumps(doc))

    def test_data_derived_directional_thresholds_and_common_typed_contract(self):
        policy=ReferencePolicy(fit_records(records(),{'model':'synthetic'}))
        decision=policy.decide([20.,0,0,0,0],'phase','pressure',0,'level','either',{'cohort':'recipe','position':0})
        self.assertTrue(all(decision['supported'].values()))
        self.assertEqual(decision['labels'],{'BOOLEAN':1,'CHOICE':1,'ORDINAL':4})
        self.assertTrue(decision['boolean_any'])
        self.assertNotEqual(decision['threshold'],1.)
        for kind,p in decision['probabilities'].items():
            self.assertAlmostEqual(sum(p),1.); self.assertEqual(np.argmax(p),decision['labels'][kind])
        other=policy.decide([0.,20,0,0,0],'phase','pressure',0,'level','either')
        self.assertEqual(other['labels']['BOOLEAN'],0)
        self.assertEqual(other['labels']['CHOICE'],2)
        self.assertTrue(other['boolean_any'])
        decrease=policy.decide([-20,0,0,0,0],'phase','pressure',0,'level','decrease')
        increase=policy.decide([-20,0,0,0,0],'phase','pressure',0,'level','increase')
        self.assertEqual(decrease['labels']['BOOLEAN'],1); self.assertEqual(increase['labels']['BOOLEAN'],0)

    def test_repeated_windows_do_not_inflate_independent_run_support(self):
        base=list(records(20)); duplicated=[r for r in base for _ in range(10)]
        policy=ReferencePolicy(fit_records(duplicated,{}))
        group,_=policy.lookup('phase','pressure','whole','either')
        self.assertEqual(group['n_runs'],20)
        decision=policy.decide([10,10,10,10,10],'phase','pressure',0)
        self.assertTrue(decision['supported']['BOOLEAN'])
        self.assertFalse(decision['supported']['ORDINAL'])
        self.assertIsNone(decision['labels']['ORDINAL'])

    def test_fallback_does_not_pool_physical_sensors_or_steps(self):
        data=list(records(10,cohort='a'))+[{**r,'run_id':'b'+r['run_id']} for r in records(10,cohort='b')]
        policy=ReferencePolicy(fit_records(data,{}))
        group,level=policy.lookup('phase','pressure','whole','either',{'cohort':'a','position':0})
        self.assertEqual(level,'step_sensor_scope'); self.assertEqual(group['n_runs'],20)
        for step,sensor in [('other','pressure'),('phase','unseen')]:
            d=policy.decide([1]*5,step,sensor,0)
            self.assertFalse(any(d['supported'].values())); self.assertIsNone(d['threshold'])

    def test_99_percentile_requires_adequate_tail_resolution(self):
        config=ReferenceConfig(quantiles=(.99,.992,.995,.999))
        policy=ReferencePolicy(fit_records(records(30),{},config))
        self.assertFalse(policy.decide([100]*5,'phase','pressure',0)['supported']['BOOLEAN'])
        with self.assertRaises(ValueError): fit_records(records(2,split='test'),{})
        with self.assertRaises(ValueError): fit_records(records(2,split='validation'),{})

    def test_zero_scale_and_duplicate_ordinal_edges_do_not_fabricate_answers(self):
        data=[{**r,'features':np.zeros((3,2,5))} for r in records(600)]
        policy=ReferencePolicy(fit_records(data,{}))
        d=policy.decide([1]*5,'phase','pressure',0)
        self.assertFalse(any(d['supported'].values()))
        self.assertIsNone(d['probabilities']['BOOLEAN'])

    def test_serialized_identity_and_no_overwrite(self):
        payload=fit_records(records(25),{'model':'a'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json'; save_policy(payload,path)
            loaded=load_policy(path,{'model':'a'})
            self.assertEqual(loaded.payload['identity'],{'model':'a'})
            with self.assertRaises(ValueError): load_policy(path,{'model':'b'})
            with self.assertRaises(FileExistsError): save_policy(payload,path)

    def test_peak_operation_is_explicit_and_does_not_dilute_short_event(self):
        a=torch.zeros(1,30,2); b=a.clone(); a[0,3,0]=6
        state=torch.zeros(1,12,8)
        features=policy_features(a,b,state,6)
        self.assertEqual(float(features[0,0,0,3]),6.)
        self.assertEqual(float(features[0,1,0,3]),6.)
        self.assertEqual(float(features[0,2,0,3]),0.)

    def test_pointwise_intervals_use_reference_peak_criterion_or_remain_unavailable(self):
        from pecvd.experiments.reference_policy import policy_pointwise_evidence
        a=np.zeros((30,2)); b=a.copy(); a[3,0]=6
        policy=ReferencePolicy(fit_records(records(600),{}))
        criterion=policy.decide([0,0,0,6,0],'phase','pressure',0)
        result=policy_pointwise_evidence(a,b,0,0,criterion)
        self.assertEqual(result['threshold_train_std'],criterion['thresholds'][3][0])
        self.assertEqual(result['exceedance_intervals'],[(3,4)])
        unknown=policy_pointwise_evidence(a,b,0,0,{'thresholds':None})
        self.assertIsNone(unknown['threshold_train_std'])
        self.assertFalse(unknown['criterion_supported']); self.assertEqual(unknown['exceedance_intervals'],[])

    def test_cache_fit_uses_only_train_and_excludes_reference_self_run(self):
        config=ReferenceConfig(max_windows_per_step=4,max_references=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'windows').mkdir()
            arrays={}
            for split,n,offset in [('train',40,0),('validation',4,1e5),('test',4,1e6)]:
                arrays[split]=np.broadcast_to((np.arange(n)//2+offset)[:,None,None],(n,8,2)).copy().astype('float32')
                arrays[split+'_process_id']=np.array([f'{split}_{i//2}' for i in range(n)])
                arrays[split+'_cohort_key']=np.array(['recipe']*n)
                arrays[split+'_position_bin']=np.zeros(n,dtype=int)
            np.savez(root/'windows/phase.npz',**arrays)
            def state(encoder,a,b):
                self.assertNotEqual(float(a[0,0,0]),float(b[0,0,0]))
                return torch.zeros(1,8,8)
            found=list(cached_records({'phase':(torch.nn.Linear(1,1),{'sensor_columns':['pressure','flow']})},root,'cpu',state,config))
            self.assertEqual(len(found),4)
            self.assertTrue(all(r['split']=='train' and np.abs(r['features']).max()<1e5 for r in found))
            with self.assertRaises(ValueError): list(cached_records({},root,'cpu',state,config,split='test'))

    def test_validation_rates_are_not_fitting_and_report_run_support(self):
        policy=ReferencePolicy(fit_records(records(600),{})); before=json.dumps(policy.payload,sort_keys=True)
        report=validation_alert_rates(policy,list(records(3,split='validation')))
        self.assertEqual(int(report.validation_runs.iloc[0]),3)
        self.assertEqual(int(report.fully_supported_runs.iloc[0]),3)
        self.assertEqual(report.step.iloc[-1],'ALL_EVALUATED_STEPS')
        self.assertEqual(json.dumps(policy.payload,sort_keys=True),before)
        with self.assertRaises(ValueError): validation_alert_rates(policy,records(1))

    def test_path_report_counts_abstentions_and_excludes_unsupported_targets(self):
        from pecvd.experiments.evaluation import summarize_reference_paths
        from pecvd.experiments.questions import CRITERIA
        no,yes=CRITERIA['BOOLEAN']
        def row(target,prediction,ready=True):
            return {'answer_type':'BOOLEAN','supported':True,'reference_supported':ready,
                'oracle_proxy':target,'reference_paths':dict(oracle=target,semantic_only=target,
                    grounding_only=prediction,full_gevis=prediction)}
        report=summarize_reference_paths([row(no,no),row(yes,None),row(None,None,False)])
        full=report[report.execution_path.eq('full_gevis')].iloc[0]
        self.assertEqual(full.questions,3); self.assertEqual(full.reference_supported_questions,2)
        self.assertEqual(full.predicted_criterion_abstentions,1)
        self.assertEqual(full.answer_accuracy,.5)
        self.assertEqual(full.balanced_accuracy_present_classes,.5)
        self.assertEqual(full.confusion_with_abstention,[[1,0,0],[0,0,1]])

    def test_retired_criterion_fixture_fits_train_once_and_reuses_exact_identity(self):
        import pandas as pd
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/'tests/fixtures/step3_legacy_diagnostics.json').read_text(encoding='utf-8'))
        cell=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='reference-policy-build'))
        calls=[]
        def cached(*args,split='train',**kwargs):
            calls.append(split)
            return records(25,split=split)
        with tempfile.TemporaryDirectory() as directory:
            env={'RUN_STRUCTURED_EVALUATION':True,'SESSION_SIGNATURE':'STEP3_GEVIS_PRETRAINED_POLICY_V2','ROOT':root,'BACKBONE_ROOT':Path(directory),
                'ENCODER_ROOT':Path(directory),'backbone_path':Path(directory)/'synthetic.pt',
                'encoders':{},'DEVICE':'cpu','numeric_state':None,'Path':Path,'pd':pd,'display':lambda *a:None}
            with patch.dict('os.environ',{},clear=True), \
                 patch('pecvd.experiments.reference_policy_v2.identity_for',return_value={'synthetic':'exact'}), \
                 patch('pecvd.experiments.reference_policy_v2.cached_records',side_effect=cached):
                exec(compile(cell,'synthetic:criterion-cell','exec'),env)
                saved=env['REFERENCE_PATH'].read_bytes()
                exec(compile(cell,'synthetic:criterion-cell','exec'),env)
                self.assertEqual(calls,['train','validation'])
                self.assertEqual(env['REFERENCE_PATH'].read_bytes(),saved)


if __name__=='__main__': unittest.main()
