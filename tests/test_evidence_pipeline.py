"""Invented data only; no private source/artifact access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from pecvd.experiments.evidence import pointwise_evidence,reuse_window_inputs,prepare_inputs
from pecvd.gevis_backbone import numeric_evidence_pairs,numeric_evidence_loss,evidence_questions
from tests.test_gevis_experiments import fake_runtime


class EvidencePipelineTests(unittest.TestCase):
    def test_single_spike_is_not_diluted_and_coordinates_are_rows(self):
        a=np.zeros((120,2)); b=a.copy(); a[70,1]=5
        result=pointwise_evidence(a,b,1,original_scale=10)
        self.assertEqual(result['peak_row'],70)
        self.assertEqual(result['peak_signed_original_units'],50)
        self.assertEqual(result['exceedance_intervals'],[(70,71)])
        self.assertEqual(result['longest_exceedance_rows'],1)
        self.assertLess(result['mean_absolute_train_std'],1.)
        self.assertIn('not seconds',result['coordinate_system'])

    def test_input_reuse_is_scoped_and_resets_after_exception(self):
        a=np.zeros((6,2)); b=a.copy(); calls=[]
        encoder=object(); meta={'sensor_columns':['a','b']}
        def prepare(x,meta): return torch.tensor(x)[None],{}
        def state(*args): calls.append(1); return torch.zeros(1,6,8)
        def features(*args): return torch.zeros(1,3,2,5)
        def get(): return prepare_inputs(a,b,'phase',encoder,meta,'cpu',prepare,state,features)
        try:
            with reuse_window_inputs():
                self.assertIs(get(),get()); self.assertEqual(len(calls),1)
                raise RuntimeError('invented')
        except RuntimeError: pass
        get(); self.assertEqual(len(calls),2)
        with reuse_window_inputs(): get(); get()
        self.assertEqual(len(calls),3)

    def test_numeric_contrasts_move_without_question_or_input_shortcuts(self):
        x=torch.randn(3,20,4); saved=x.clone()
        changed,reference,targets=numeric_evidence_pairs(x,4,seed=3)
        torch.testing.assert_close(x,saved)
        torch.testing.assert_close(reference[::2],reference[1::2])
        self.assertTrue(torch.all(targets['sensor'][::2]!=targets['sensor'][1::2]))
        self.assertEqual(len(set(evidence_questions('invented',6))),1)
        for i in range(len(changed)):
            different=(changed[i]-reference[i]).abs().sum(0)>0
            self.assertEqual(int(different.sum()),1)
            self.assertTrue(different[targets['sensor'][i]])
        logits=torch.zeros(6,20,requires_grad=True)
        loss,metrics=numeric_evidence_loss(logits.softmax(-1),targets)
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_failure_diagnosis_separates_oracle_routing_and_gates(self):
        from pecvd.experiments.evaluation import evaluate_injections,intervention_cases
        runtime,pairs=fake_runtime(); current,reference=pairs(runtime,'invented_phase',limit=1)[0]
        spec,_,_=list(intervention_cases(current,['a','b'],np.ones(2),kinds=('level_shift',),strengths=(2.,),max_cases=1))[1]
        target=['a','b'][spec.sensor]
        def scan(runtime,current,reference,step,**kwargs):
            value=float((current[target]-reference[target]).abs().max())
            return [{'requested_sensor':target,'requested_statistic':'level','requested_scope':'whole',
                'sensor':target,'interval':[0,12],'score':.95 if value>1 else .1,
                'accepted':False,'alert':False,'candidate_alert':value>1,
                'warnings':['low_sensor_grounding_confidence']}]
        with patch('pecvd.experiments.evaluation.runtime_scan',side_effect=scan):
            result=evaluate_injections(runtime,current,reference,'invented_phase',np.ones(2),
                kinds=('level_shift',),strengths=(2.,),max_cases=1)
        case=result.iloc[1]
        self.assertTrue(case.oracle_declared_alert)
        self.assertTrue(case.target_candidate_alert)
        self.assertFalse(case.target_accepted_alert)
        self.assertEqual(case.diagnostic_stage,'withheld_by_runtime')
        self.assertEqual(case.gate_warning_counts['low_sensor_grounding_confidence'],1)

    def test_unrelated_existing_alert_does_not_become_new_target_detection(self):
        from pecvd.experiments.evaluation import evaluate_injections,intervention_cases
        runtime,pairs=fake_runtime(); current,reference=pairs(runtime,'invented_phase',limit=1)[0]
        spec,_,_=list(intervention_cases(current,['a','b'],np.ones(2),kinds=('level_shift',),strengths=(2.,),max_cases=1))[1]
        other=['a','b'][1-spec.sensor]
        row={'requested_sensor':other,'requested_statistic':'level','requested_scope':'whole','sensor':other,
             'interval':[0,12],'score':.99,'accepted':True,'alert':True,'candidate_alert':True,'warnings':[]}
        with patch('pecvd.experiments.evaluation.runtime_scan',return_value=[row]):
            result=evaluate_injections(runtime,current,reference,'invented_phase',np.ones(2),
                kinds=('level_shift',),strengths=(2.,),max_cases=1)
        case=result.iloc[1]
        self.assertTrue(case.alert)  # Preserved legacy metric, explicitly insufficient.
        self.assertFalse(case.target_accepted_alert)
        self.assertFalse(case.new_target_alert)
        self.assertEqual(case.diagnostic_stage,'routing_or_operation_mismatch')

    def test_strict_grounded_accuracy_does_not_credit_same_answer_wrong_sensor(self):
        from pecvd.experiments.evaluation import summarize_questions
        row=dict(answer_type='BOOLEAN',supported=True,accepted=True,answer_correct=True,
                 sensor_correct=False,scope_correct=True,statistic_correct=True,direction_correct=True,
                 family='invented',joint_query_correct=False,end_to_end_correct=True)
        summary,_=summarize_questions([row])
        self.assertEqual(summary.iloc[0].candidate_proxy_accuracy,1.)
        self.assertEqual(summary.iloc[0].grounded_accepted_accuracy,0.)
        self.assertEqual(summary.iloc[0].accepted_wrong_query_share,1.)

    def test_temporal_baseline_is_deterministic_and_preserves_rng(self):
        from pecvd.experiments.temporal_baseline import temporal_scores
        x=np.random.default_rng(4).normal(size=(3,12,2)).astype(np.float32)
        rng=torch.get_rng_state().clone()
        first,info=temporal_scores(x,x,seed=3,epochs=1,hidden=4)
        torch.testing.assert_close(rng,torch.get_rng_state())
        second,_=temporal_scores(x,x,seed=3,epochs=1,hidden=4)
        np.testing.assert_allclose(first,second)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(info['fit_split'],'train')
        with self.assertRaises(ValueError): temporal_scores(x,x[:,:6],epochs=1)

    def test_class_metrics_pool_counts_instead_of_step_averages(self):
        from pecvd.experiments.evaluation import class_metrics_from_confusion
        a=np.array([[90,0],[10,0]]); b=np.array([[0,1],[0,9]])
        pooled=class_metrics_from_confusion(a+b)
        self.assertEqual(pooled['questions'],110)
        self.assertAlmostEqual(pooled['answer_accuracy'],99/110)
        mean_f1=.5*(class_metrics_from_confusion(a)['macro_f1_present_classes']+class_metrics_from_confusion(b)['macro_f1_present_classes'])
        self.assertNotAlmostEqual(pooled['macro_f1_present_classes'],mean_f1)

    def test_legacy_step4_only_scores_declared_semantic_slots(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        n=json.loads((root/'tests/fixtures/step4_legacy_validation.json').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in n['cells'] if c.get('id')=='v4-eval'))
        self.assertIn("if kind=='BOOLEAN':",source)
        self.assertIn("'statistic_direction_scored_types':'BOOLEAN only'",source)
        self.assertIn('class_metrics_from_confusion(np.stack(group.confusion).sum(0))',source)
        ast.parse(source)
        from types import SimpleNamespace
        from pecvd import gevis_fast
        from pecvd.experiments.questions import CRITERIA
        class Head:
            def __init__(self,size,scope=False): self.size=size;self.scope=scope
            def __call__(self,text):
                result=torch.zeros(len(text),self.size)
                ids=torch.zeros(len(text),dtype=torch.long) if self.scope else (text[:,0]>0).long()
                result[torch.arange(len(text)),ids]=1
                return result
        class Model:
            semantic_heads={'statistic':Head(5),'direction':Head(3),'time_scope':Head(3,True)}
            def eval(self): return self
            def _text(self,texts,device): return torch.tensor([[0 if text=='BOOLEAN' else 1] for text in texts],dtype=torch.float32)
            def __call__(self,state,*args,**kwargs): return None,None,torch.ones(len(state),6)
        class Encoder:
            def to(self,*args): return self
            def eval(self): return self
            def cpu(self): return self
        kinds=('BOOLEAN','CHOICE','ORDINAL')
        contract={'features':torch.zeros(1,3,2,5),'sensor_ids':torch.zeros(1,dtype=torch.long),
            'scope_ids':torch.zeros(1,dtype=torch.long),'statistic_ids':torch.zeros(1,dtype=torch.long),
            'direction_ids':torch.zeros(1,dtype=torch.long),'questions':{kind:[kind] for kind in kinds}}
        ns={'gevis':Model(),'encoders':{'phase':(Encoder(),{})},'DEVICE':'cpu','ANSWER_TYPES':kinds,'CRITERIA':CRITERIA,
            'numeric_state':lambda *a:torch.zeros(1,6,8),'grounded_question_contract':lambda *a:contract,
            'numeric_token_descriptions':lambda *a:[['d']*6],'gevis_fast':gevis_fast,'CONFIG':SimpleNamespace(reward_temperature=.75)}
        env={'torch':torch,'np':np,'pd':pd,'STRUCTURED_NS':ns}
        functions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in ('_path_metrics','evaluate_predicted_paths')]
        exec(compile(ast.Module(body=functions,type_ignores=[]),'STEP4 semantic synthetic evaluation','exec'),env)
        _,quality=env['evaluate_predicted_paths']({'phase':[(torch.zeros(1,12,2),torch.zeros(1,12,2),torch.zeros(1))]})
        self.assertEqual(quality.iloc[0].statistic_semantic_accuracy,1.)
        self.assertEqual(quality.iloc[0].direction_semantic_accuracy,1.)

    def test_operating_points_report_small_control_limit_not_matched_test_rates(self):
        from pecvd.experiments.baselines import compare_frozen_runtime
        runtime,pairs=fake_runtime()
        with patch('pecvd.experiments.evaluation.load_cached_pairs',side_effect=pairs):
            summary,details=compare_frozen_runtime(runtime,'invented_phase',validation_limit=2,test_limit=1)
        operating=summary.attrs['operating_points']
        self.assertTrue(operating.insufficient_controls_for_rate.all())
        self.assertTrue(operating.control_rate_resolution.eq(.5).all())
        self.assertTrue(details.columns.str.contains('scan_ms').any())

    def test_external_mapping_is_explicit_private_and_source_unchanged(self):
        from pecvd.process_config import load_process_config
        from pecvd.experiments.process_inputs import external_readiness,configured_sensors,candidate_backbone_path
        script=Path(__file__).resolve().parents[1]/'scripts/configure_external_process.py'
        spec=importlib.util.spec_from_file_location('external_config_test',script)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'data').mkdir()
            source=root/'data/merge_peald.csv'; source.write_text('run,phase,order,pressure\nr1,A,0,1\n',encoding='utf-8')
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            not_ready=external_readiness(root)
            self.assertTrue(not_ready['source_present']); self.assertFalse(not_ready['model_evaluated'])
            destination=module.configure(root,'data/merge_peald.csv',run='run',step='phase',order='order',sensors=['pressure'],
                steps=[{'name':'A','window_length':12,'patch_length':3,'stride':3}])
            self.assertTrue(destination.is_relative_to(root/'.cache'))
            cfg=load_process_config(destination)
            self.assertEqual(configured_sensors(cfg,pd.DataFrame({'pressure':[1.]})),['pressure'])
            self.assertIn(cfg.artifacts.encoder,str(candidate_backbone_path(root,cfg,'real_step1_evidence_v3')))
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertFalse(external_readiness(root)['backbone_ready'])
            with self.assertRaises(FileExistsError): module.configure(root,'data/merge_peald.csv',run='run',step='phase',order='order',sensors=['pressure'],steps=[])

    def test_peald_auto_mapping_is_train_only_and_windows_never_join_repeated_steps(self):
        import ast
        from pecvd.experiments.onboarding import prepare_peald
        from pecvd.experiments.process_inputs import load_process_frame
        from pecvd.process_config import load_process_config
        ids=np.array([f'job_{i:02}' for i in range(16)],dtype=object)
        shuffled=ids.copy(); np.random.default_rng(42).shuffle(shuffled)
        train=set(shuffled[:11]); held=set(shuffled[11:])
        rows=[]; stamps=pd.date_range('2000-01-01',periods=16*18,freq='s')
        for i,run in enumerate(ids):
            for j in range(18):
                rows.append({'ActivePjobID':run,'PPExecStepName':'short' if 8<=j<10 else 'repeated',
                    'Timestamp':str(stamps[i*18+j]),'PMStoredProcessRecipeName':'recipe','PPExecname':'program',
                    'Temp_Monitor_U':float(j%8),'MFC_constant':0.,
                    'Temp_Monitor_TestOnly':float(j) if run in held else 0.,'PPExecStepSeqNo':j})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'data').mkdir()
            pd.DataFrame(rows).to_csv(root/'data/merge_peald.csv',index=False)
            path=prepare_peald(root,chunk_size=50)
            cfg=load_process_config(path)
            self.assertEqual(cfg.sensor_columns,('Temp_Monitor_U',))
            self.assertEqual(len(cfg.steps),1) # Short steps explicitly reported, not called evaluated.
            payload=json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(any(not row['ready'] for row in payload['onboarding']['phase_support']))
            frame=load_process_frame(root,cfg)
            frame['_source_group']='synthetic';frame['_pm_cycle']='unknown';frame['_cohort_key']='synthetic'
            notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
            source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
            fn=next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name=='make_windows')
            env={'np':np,'pd':pd,'STEPS':cfg.step_specs,'run_split':{'train':list(train)},'frame':frame,
                 'RUN_COLUMN':cfg.columns.run,'STEP_COLUMN':cfg.columns.step,'ORDER_COLUMN':cfg.columns.order,'sensor_columns':list(cfg.sensor_columns)}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'STEP1 repeated-step synthetic windows','exec'),env)
            windows,metadata=env['make_windows'](cfg.steps[0].name,'train')
            self.assertEqual(len(windows),22)
            self.assertTrue((metadata.window_end-metadata.window_start).eq(cfg.steps[0].window_length-1).all())


if __name__=='__main__': unittest.main()
