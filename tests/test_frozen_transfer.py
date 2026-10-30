from tests.legacy_notebooks import legacy_exp_book
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn

from pecvd.experiments.frozen_transfer import FrozenChannelAdapter,external_pairs

ROOT=Path(__file__).resolve().parents[1]


class Step3ExportTests(unittest.TestCase):
    def test_trained_full_gevis_roundtrip_and_no_overwrite(self):
        import hashlib
        import tempfile
        from tests.test_gevis_fast import model_class
        from pecvd.gevis_fast import fit_answer_checkpoint,load_answer_checkpoint,answer_probabilities
        torch.manual_seed(17)
        model=model_class()(numeric_dim=440,d_model=16,dropout=0.)
        criteria={'BOOLEAN':['no','yes'],'CHOICE':['none','level','variance','slope','local','residual'],
                  'ORDINAL':['none','low','medium','high','very high']}
        kinds=list(criteria);tokens=torch.randn(3,4,440)
        with torch.no_grad():
            fusion=model(tokens,['compare']*3,kinds,list(criteria.values()),return_fusion=True)
        cache=[]
        for i,kind in enumerate(kinds):
            rewards=torch.zeros(1,len(criteria[kind]));rewards[0,-1]=1.
            cache.append(dict(answer_type=kind,fusion=tuple(t[i:i+1].detach().clone() for t in fusion),rewards=rewards))
        frozen={k:v.clone() for k,v in model.state_dict().items()
                if not k.startswith(('change_head.','criterion_proj.')) and k!='temperature'}
        with tempfile.TemporaryDirectory() as directory:
            backbone=Path(directory)/'gevis_backbone.pt';backbone.write_bytes(b'synthetic source')
            kwargs=dict(train_run_ids=['train'],validation_run_ids=['validation'],epochs=1,
                        batch_size=2,max_seconds=30,progress=lambda _:None)
            path,history,report=fit_answer_checkpoint(model,cache,cache,criteria,backbone,**kwargs)
            self.assertGreater(report['optimizer_updates'],0);self.assertTrue(history)
            for key,value in frozen.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
            restored=model_class()(numeric_dim=440,d_model=16,dropout=0.)
            load_answer_checkpoint(restored,path,backbone_sha256=hashlib.sha256(backbone.read_bytes()).hexdigest(),criteria=criteria)
            a=answer_probabilities(model,tokens,['compare']*3,kinds,list(criteria.values()))
            b=answer_probabilities(restored,tokens,['compare']*3,kinds,list(criteria.values()))
            for p,q in zip(a,b):torch.testing.assert_close(p,q);self.assertAlmostEqual(float(q.sum()),1.,places=6)
            with self.assertRaises(FileExistsError):fit_answer_checkpoint(model,cache,cache,criteria,backbone,**kwargs)
            with self.assertRaises(ValueError):load_answer_checkpoint(restored,path,backbone_sha256='wrong')
            with self.assertRaises(ValueError):answer_probabilities(model,tokens[...,:392],['compare']*3,kinds,list(criteria.values()))
            reordered=list(criteria.values());reordered[0]=list(reversed(reordered[0]))
            with self.assertRaises(ValueError):answer_probabilities(restored,tokens,['compare']*3,kinds,reordered)

    def test_fit_rejects_overlap_missing_types_and_zero_budget_without_export(self):
        import tempfile
        from tests.test_gevis_fast import model_class
        from pecvd.gevis_fast import fit_answer_checkpoint,DEFAULT_ANSWER_CRITERIA
        model=model_class()(numeric_dim=440,d_model=16,dropout=0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'gevis_backbone.pt';path.write_bytes(b'synthetic')
            with self.assertRaises(ValueError):
                fit_answer_checkpoint(model,[],[],DEFAULT_ANSWER_CRITERIA,path,
                    train_run_ids=['same'],validation_run_ids=['same'])
            with self.assertRaises(ValueError):
                fit_answer_checkpoint(model,[],[],DEFAULT_ANSWER_CRITERIA,path,
                    train_run_ids=['a'],validation_run_ids=['b'],max_seconds=0)
            self.assertFalse((path.parent/'gevis_answer_model.pt').exists())

    def test_notebook_uses_real_fit_not_confirmation_and_train_only_references(self):
        import ast
        from collections import defaultdict
        notebook=json.loads((ROOT/'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        source='\n'.join(''.join(c.get('source',[])) for c in notebook['cells'] if c['cell_type']=='code')
        self.assertIn('RUN_ANSWER_HEAD_TRAINING=False',source)
        self.assertIn('RUN_TEMPORAL_LOCALIZATION_TRAINING=not RUN_FUSION_ANSWER_TRAINING',source)
        self.assertIn('gevis_fast.fit_answer_checkpoint(',source)
        self.assertNotIn('ANSWER_HEAD_TRAINING_VERIFIED',source)
        self.assertIn("reference_array=windows[step]['train']",source)
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='FdcComparisonPairs')
        namespace=dict(Dataset=torch.utils.data.Dataset,np=np,torch=torch,defaultdict=defaultdict)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<synthetic train references>','exec'),namespace)
        data=np.zeros((1,4,1),dtype='float32');ref=np.full((2,4,1),7.,dtype='float32')
        meta=dict(position_bin=[0],cohort_key=['a'],process_id=['validation'])
        reference_meta=dict(position_bin=[0,0],cohort_key=['a','a'],process_id=['train','train'])
        dataset=namespace['FdcComparisonPairs'](data,meta,42,reference_array=ref,reference_meta=reference_meta)
        self.assertEqual(len(dataset.candidates[0]),1)
        self.assertEqual(float(dataset[0][1].mean()),7.)


class LearnedExperimentTests(unittest.TestCase):
    def runtime(self):
        from pecvd.gevis_fast import DEFAULT_ANSWER_CRITERIA
        class Answers(nn.Module):
            def __init__(self):
                super().__init__();self.numeric=nn.Linear(8,8)
                self.heads=nn.ModuleDict({k:nn.Linear(8,len(v)) for k,v in DEFAULT_ANSWER_CRITERIA.items()})
                self.answer_criteria=DEFAULT_ANSWER_CRITERIA;self.seen=[]
            def forward(self,state,questions,kinds,criteria,descriptions,return_fusion=False):
                self.seen.append(list(questions));return state.mean(1)
            def decode(self,fusion,kinds,criteria):
                return [self.heads[k](fusion[i]) for i,k in enumerate(kinds)]
        model=Answers().requires_grad_(False)
        meta=dict(sensor_columns=['a','b'],std=[1.,1.],mean=[0.,0.])
        m=SimpleNamespace(GEVIS_MODEL=model,DEVICE='cpu',BACKBONE_PATH=Path('synthetic/backbone.pt'),
            GEVIS_CHECKPOINT={'schema':'gevis.backbone.generic.v10'},DISCOVERY_POLICY_MODE='window_calibrated_v2',
            DISCOVERY_POLICY=None,DECISION_POLICY_MODE='reference_quantiles_v2',
            prepare_window=lambda frame,meta:(torch.tensor(frame[['a','b']].to_numpy(),dtype=torch.float32)[None],{}),
            numeric_state=lambda encoder,a,b:(a-b).reshape(1,2,8),
            numeric_token_descriptions=lambda step,meta,p,b:[['sensor a','sensor b'] for _ in range(b)],
            semantic_comparison_features=lambda a,b,s,p:torch.zeros(1,3,2,5))
        return SimpleNamespace(module=m,encoders={'phase':(nn.Identity(),meta)})

    def pairs(self):
        current=pd.DataFrame({'a':np.linspace(0,1,8),'b':np.linspace(1,0,8)})
        return [(current,current.copy()),(current.copy(),current.copy())]

    def test_question_contrasts_use_real_decoder_same_input_without_f1(self):
        from pecvd.experiments.reporting import run_answer_experiment
        runtime=self.runtime();variants=[('any','BOOLEAN','Is there a change?'),('mean','BOOLEAN','Has the mean changed?')]
        with patch('pecvd.experiments.reporting.load_cached_pairs',return_value=self.pairs()) as pairs, \
             patch('pecvd.experiments.discovery.review_changes',side_effect=AssertionError('No numerical fallback')):
            report=run_answer_experiment(ROOT,runtime=runtime,pairs_per_step=2,question_variants=variants)
        self.assertNotIn('f1',report.summary)
        self.assertFalse(report.summary.accuracy_available.any())
        self.assertTrue(report.details.score_source.eq('trained_answer_decoder_softmax').all())
        self.assertEqual(len(report.details),4)
        self.assertTrue(all(q==[v[2] for v in variants] for q in runtime.module.GEVIS_MODEL.seen))
        self.assertEqual(pairs.call_args.kwargs['split'],'test')

    def test_missing_answer_weights_stop_before_cache_reads_without_fallback(self):
        from pecvd.experiments.reporting import run_answer_experiment
        runtime=self.runtime();del runtime.module.GEVIS_MODEL.answer_criteria
        with patch.dict('os.environ',{},clear=True), \
             patch('pecvd.experiments.reporting.load_cached_pairs',side_effect=AssertionError('No input read')), \
             patch('pecvd.experiments.discovery.review_changes',side_effect=AssertionError('No fallback')):
            with self.assertRaises(FileNotFoundError):run_answer_experiment(ROOT,runtime=runtime)

    def test_injection_runner_is_learned_argmax_not_numeric_gate(self):
        from pecvd.experiments.reporting import run_answer_experiment
        with patch('pecvd.experiments.reporting.load_cached_pairs',return_value=self.pairs()), \
             patch('pecvd.experiments.discovery.review_changes',side_effect=AssertionError('No numerical fallback')):
            report=run_answer_experiment(ROOT,runtime=self.runtime(),inject=True,max_cases=4)
        self.assertIn('f1',report.summary)
        self.assertFalse(report.metadata['numeric_fallback_used'])
        self.assertEqual(report.metadata['probability_threshold'],.5)
        self.assertFalse(report.details.truth_used_for_inference.any())
        self.assertEqual(set(report.details[report.details.injected].kind),{'level_shift','drift','spike','variance'})

    def test_paired_baseline_adds_neural_method_without_numeric_policy(self):
        from pecvd.experiments.baselines import compare_frozen_runtime
        with patch('pecvd.experiments.evaluation.load_cached_pairs',return_value=self.pairs()), \
             patch('pecvd.experiments.discovery.review_changes',side_effect=AssertionError('Missing numeric policy must not fall back')):
            result,raw=compare_frozen_runtime(self.runtime(),'phase',max_cases=4,
                use_merlion=False,use_temporal=False,use_learned_answers=True)
        learned=result[result.method.eq('GEVIS_learned_answer')].iloc[0]
        held=raw[raw.split.eq('test') & raw.effective]
        self.assertEqual(learned.test_windows,len(held))
        self.assertIn('trained_answer_decoder_softmax',learned.score_source)
        self.assertIn('GEVIS_learned_answer_ms',raw)
        self.assertNotIn('GEVIS_discovery_alert',raw)

    def test_all_exp_notebooks_use_fast_fusion_and_distinct_current_tasks(self):
        expected={'CVD_question_sensitivity.ipynb':'run_question_probes',
                  'CVD_synthetic_change_generalization.ipynb':'run_generalization','CVD_feature_ablation.ipynb':'run_feature_controls'}
        for path in (ROOT/'notebooks/research').glob('*.ipynb'):
            source='\n'.join(''.join(c.get('source',[])) for c in json.loads(path.read_text(encoding='utf-8'))['cells'])
            self.assertNotIn('STEP1 retrain -> STEP2 retrain',source,str(path))
            self.assertNotIn('notebooks/_backups',source,str(path))
            self.assertIn('fusion_answers',source,str(path))
            self.assertIn(expected[path.name],source,str(path))
            self.assertNotIn('USE_LEARNED_ANSWERS=False',source)


class TinySource(nn.Module):
    def __init__(self):
        super().__init__(); self.patch_len=2; self.d_model=4
        self.input=nn.Linear(2,4); self.sensor_emb=nn.Embedding(3,4)
        self.patch_emb=nn.Embedding(64,4); self.mask_token=nn.Parameter(torch.zeros(4))
        block=nn.TransformerEncoderLayer(4,1,8,batch_first=True,dropout=0.)
        self.encoder=nn.TransformerEncoder(block,1); self.norm=nn.LayerNorm(4); self.decoder=nn.Linear(4,2)


def fixture():
    rng=np.random.default_rng(4); rows=[]
    for run in range(30):
        for segment in range(3):
            for t in range(8):
                rows.append({'run':str(run),'step':'external','order':segment*8+t,
                    '_gevis_segment_id':run*3+segment,'_gevis_position_ratio':segment/3+t/24,
                    'source_group_one':1,'alpha':float(rng.normal()),'beta':float(rng.normal())})
    return pd.DataFrame(rows),SimpleNamespace(columns=SimpleNamespace(run='run',step='step',order='order'),
        sensor_columns=('alpha','beta'),steps=[SimpleNamespace(name='external',description='external phase')])


class FrozenTransferTests(unittest.TestCase):
    def test_retired_exp4_fixture_displays_english_core_contract(self):
        import contextlib,io,ast
        doc=legacy_exp_book('04_dataset_transfer.ipynb')
        code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='shared-pipeline-settings'))
        # Execute the actual display statements with synthetic current settings.
        nodes=ast.parse(code).body
        start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.ImportFrom) and n.module=='pecvd.experiments.questions')
        scope={'PIPELINE_SETTINGS':{'operation_profile':'core','gate_profile':'position_rank_v4','training_profile':'center_relation_v14','control_alert_budget':.01}}
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(ast.Module(body=nodes[start:],type_ignores=[]),'exp4-display','exec'),scope)
        text=capture.getvalue()
        self.assertIn('Is there a change in this interval?',text)
        self.assertFalse(any('\uac00'<=ch<='\ud7a3' for ch in text))
        self.assertIn('frozen source weights',text)
        self.assertIn('not used as an alternative detector',text)
        self.assertIn('conditional numerical evidence',text)
        self.assertNotIn('fits learned decision candidate',text)
        self.assertNotIn('AND center-pair relationship change',text)
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None
            for c in doc['cells'] if c['cell_type']=='code'))

    def test_trained_answer_checkpoint_is_bound_and_strict_not_random(self):
        import tempfile
        from pecvd.experiments.frozen_transfer import load_frozen_answer_model
        model=nn.Module();model.change_head=nn.Linear(4,1);model.criterion_proj=nn.Linear(2,2)
        model.temperature=nn.Parameter(torch.tensor(0.))
        runtime=SimpleNamespace(module=SimpleNamespace(GEVIS_MODEL=model,
            GEVIS_CHECKPOINT={'_artifact_sha256':'source-fixture'}))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'answer.pt'
            with self.assertRaises(FileNotFoundError):load_frozen_answer_model(runtime,path)
            payload={'schema':'gevis.typed_answer.v1','backbone_sha256':'source-fixture','model_state':model.state_dict()}
            torch.save(payload,path);loaded=load_frozen_answer_model(runtime,path)
            self.assertIsNot(loaded,model)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            for key,value in model.state_dict().items():torch.testing.assert_close(loaded.state_dict()[key],value)
            payload['backbone_sha256']='wrong';torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'different backbone'):load_frozen_answer_model(runtime,path)
            payload['backbone_sha256']='source-fixture';payload['model_state'].pop('criterion_proj.bias')
            torch.save(payload,path)
            with self.assertRaises(RuntimeError):load_frozen_answer_model(runtime,path)

    def test_original_answer_predictions_use_decoder_softmax_not_numeric_gate(self):
        from pecvd.experiments.frozen_transfer import predict_external_answers
        from pecvd.experiments.questions import CRITERIA
        class AnswerModel(nn.Module):
            def __init__(self):
                super().__init__();self.heads=nn.ModuleDict({k:nn.Linear(8,len(v)) for k,v in CRITERIA.items()})
                self.numeric=nn.Linear(8,8)
                self.calls=0
            def forward(self,state,texts,kinds,criteria,descriptions,return_fusion=False):
                self.calls+=1;self.texts=texts;return state.mean(1)
            def decode(self,fusion,kinds,criteria):
                return [self.heads[k](fusion[i]) for i,k in enumerate(kinds)]
        model=AnswerModel().requires_grad_(False)
        m=SimpleNamespace(GEVIS_MODEL=model,DEVICE='cpu',CRITERIA=CRITERIA,
            prepare_window=lambda frame,meta:(torch.zeros(1,8,2),{}),
            numeric_state=lambda *a:torch.ones(1,4,8),
            numeric_token_descriptions=lambda step,meta,p,b:[['x']*4 for _ in range(b)])
        runtime=SimpleNamespace(module=m,encoders={'phase':(nn.Identity(),{'sensor_columns':['x','y']})},
            external_pairs={'phase':{'test':[(object(),object())]}})
        with self.assertRaisesRegex(ValueError,'verified trained'):predict_external_answers(runtime)
        runtime.trained_answer_decoder_loaded=True
        with patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',side_effect=AssertionError('No discovery')):
            result=predict_external_answers(runtime)
        self.assertEqual(model.calls,1)
        self.assertEqual(len(result),3);self.assertFalse(result.injected.any())
        self.assertTrue(result.score_source.eq('trained_answer_decoder_softmax').all())
        for probabilities in result.probabilities:self.assertAlmostEqual(sum(probabilities.values()),1.,places=6)

    def test_learned_transfer_missing_checkpoint_never_reads_external_data(self):
        import tempfile
        from pecvd.experiments.reporting import run_transfer_experiment
        runtime=SimpleNamespace(module=SimpleNamespace(GEVIS_MODEL=SimpleNamespace(
            numeric=SimpleNamespace(in_features=440))))
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'external.json';config.write_text('{}',encoding='utf-8')
            with patch('pecvd.experiments.process_inputs.load_process_frame',side_effect=AssertionError('No external CSV read')):
                backbone=Path(folder)/'current_backbone/gevis_backbone.pt'
                with patch.dict('os.environ',{},clear=True):
                    report=run_transfer_experiment(ROOT,process_config=config,backbone_path=backbone,
                        source_runtime=runtime,prediction_mode='learned_answers')
        self.assertFalse(report.metadata['model_evaluated'])
        self.assertFalse(report.metadata['numeric_fallback_used'])
        self.assertEqual(report.metadata['status'],'NOT_EVALUATED')
        self.assertEqual(report.metadata['expected_numeric_dim'],440)
        self.assertEqual(Path(report.metadata['expected_answer_checkpoint']),backbone.parent/'gevis_answer_model.pt')
        self.assertNotIn('gevis_fdc_e5_v1',report.metadata['expected_answer_checkpoint'])

    def test_learned_transfer_does_not_prepare_numeric_criteria_or_inject(self):
        import tempfile
        from pecvd.experiments.reporting import run_transfer_experiment
        runtime=SimpleNamespace(module=SimpleNamespace(),encoders={'phase':(nn.Identity(),{})},transfer_metadata={})
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'external.json';config.write_text('{}',encoding='utf-8')
            with patch('pecvd.process_config.load_process_config',return_value=SimpleNamespace()), \
                 patch('pecvd.experiments.frozen_transfer.load_frozen_answer_model',return_value=nn.Identity()), \
                 patch('pecvd.experiments.frozen_transfer.bind_external_runtime',return_value=(runtime,pd.DataFrame())) as bind, \
                 patch('pecvd.experiments.frozen_transfer.predict_external_answers',return_value=pd.DataFrame([{
                     'step':'phase','answer_type':'BOOLEAN','answer':'Yes','probabilities':{'No':.1,'Yes':.9}}])), \
                 patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',side_effect=AssertionError('No numeric discovery')), \
                 patch('pecvd.experiments.reporting.run_injection_experiment',side_effect=AssertionError('No injections')):
                report=run_transfer_experiment(ROOT,process_config=config,backbone_path='unused-source.pt',
                    source_runtime=runtime,prepared_frame=pd.DataFrame(),prediction_mode='learned_answers')
        self.assertFalse(bind.call_args.kwargs['prepare_criteria'])
        self.assertEqual(report.metadata['status'],'COMPLETED')
        self.assertFalse(report.metadata['injections_used'])
        self.assertFalse(report.metadata['numeric_fallback_used'])
        self.assertIsNone(runtime.module.DISCOVERY_POLICY)
        self.assertFalse(report.summary.accuracy_available.any())

    def test_external_window_calibration_is_independent_and_not_eval_limited(self):
        import copy
        from pecvd.experiments.frozen_transfer import prepare_external_window_policy,load_discovery_pairs
        from pecvd.experiments.window_policy import WindowConfig
        frame,config=fixture();packs,_=external_pairs(frame,config,8,limit=1,seed=42)
        paired,mean,scale=packs['external']
        self.assertEqual(len(paired['validation']),1)
        self.assertEqual(len(paired['calibration_validation']),12)
        cfg=WindowConfig(gate_scope='position_rank_v4',operation_profile='core')
        source_policy=SimpleNamespace(config=cfg)
        m=SimpleNamespace(DEVICE='cpu',DISCOVERY_POLICY=source_policy,
            GEVIS_CHECKPOINT={'_artifact_sha256':'synthetic-only'},
            numeric_state=lambda encoder,a,b:torch.zeros(1,4,8))
        runtime=SimpleNamespace(module=m,encoders={'external':(nn.Identity(),{
            'sensor_columns':['alpha','beta'],'mean':mean,'std':scale})},external_pairs={'external':paired})
        with self.assertRaisesRegex(ValueError,'source thresholds'):
            load_discovery_pairs(runtime,'external')
        prepare_external_window_policy(runtime,packs,config)
        before=copy.deepcopy(m.DISCOVERY_POLICY.payload)
        self.assertIsNot(m.DISCOVERY_POLICY,source_policy)
        self.assertFalse(before['test_used_for_fitting'])
        self.assertEqual(before['identity']['scope'],'external_window_reference_validation_only')
        self.assertEqual(len(load_discovery_pairs(runtime,'external',limit=1)),1)
        # Test magnitudes cannot change either scaling or fitted window bounds.
        for current,reference in paired['test']:
            current[['alpha','beta']]+=1e6
        prepare_external_window_policy(runtime,packs,config)
        self.assertEqual(before,m.DISCOVERY_POLICY.payload)
        self.assertTrue(any(c['supported'] for c in before['groups']['external']['contexts'].values()))

    def test_full_external_scan_batches_all_100_sensors(self):
        from pecvd.experiments.reporting import transfer_scan_budget
        from pecvd.experiments.evaluation import runtime_scan
        calls=[]
        def ask(current,reference,step,questions,info):
            calls.append(len(questions))
            return [{'accepted':False,'answer':None,'candidate_answer':'no',
                     'probabilities':{'no':1.,'yes':0.},
                     'evidence':{'sensor':'channel_0','scope_rows':[0,8]},
                     'predicted_query':{'time_scope':'whole','statistic':'level'}} for _ in questions]
        from pecvd.experiments.questions import CRITERIA
        no,yes=CRITERIA['BOOLEAN']
        def localized_ask(*args):
            answers=ask(*args)
            for answer in answers:
                answer['candidate_answer']=no; answer['probabilities']={no:1.,yes:0.}
            return answers
        runtime=SimpleNamespace(encoders={'phase':(None,{'sensor_columns':[f'channel_{i}' for i in range(100)]})},
            module=SimpleNamespace(GevisQuestion=lambda *args:args,ReferenceInfo=lambda *args:args,ask_gevis=localized_ask))
        budget,plan=transfer_scan_budget(runtime)
        self.assertEqual(budget,1500); self.assertEqual(plan.questions_per_window.iloc[0],1500)
        rows=runtime_scan(runtime,pd.DataFrame(),pd.DataFrame(),'phase',max_questions=budget,batch_size=32)
        self.assertEqual(len(rows),1500); self.assertEqual(sum(calls),1500)
        self.assertLessEqual(max(calls),32)
        self.assertEqual(len({r['requested_sensor'] for r in rows}),100)
        with self.assertRaisesRegex(ValueError,'MAX_DETECTION_QUESTIONS_PER_WINDOW'):
            transfer_scan_budget(runtime,1000)

    def test_transfer_budget_preflight_and_forwarding_before_expensive_questions(self):
        import tempfile
        from pecvd.experiments.reporting import run_transfer_experiment,ExperimentReport
        runtime=SimpleNamespace(encoders={'phase':(None,{'sensor_columns':[f'channel_{i}' for i in range(100)]})},transfer_metadata={})
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'external.json'; config.write_text('{}',encoding='utf-8')
            with patch('pecvd.process_config.load_process_config',return_value=SimpleNamespace()), \
                 patch('pecvd.experiments.frozen_transfer.bind_external_runtime',return_value=(runtime,pd.DataFrame())), \
                 patch('pecvd.experiments.reporting.run_question_experiment',side_effect=lambda *a,**k:ExperimentReport('questions',pd.DataFrame([{'ok':True}]),metadata={} )) as questions, \
                 patch('pecvd.experiments.reporting.run_injection_experiment',return_value=ExperimentReport('injections',pd.DataFrame([{'ok':True}]))) as injection:
                kwargs=dict(process_config=config,backbone_path='unused-source.pt',source_runtime=runtime,prepared_frame=pd.DataFrame())
                with self.assertRaisesRegex(ValueError,'1500'):
                    run_transfer_experiment(ROOT,max_detection_questions=1000,**kwargs)
                questions.assert_not_called(); injection.assert_not_called()
                report=run_transfer_experiment(ROOT,**kwargs)
                self.assertEqual(injection.call_args.kwargs['max_questions'],1500)
                self.assertIn('external_detection_scan_plan',report.diagnostics)
                self.assertEqual(questions.call_count,1)
                run_transfer_experiment(ROOT,max_detection_questions=2000,**kwargs)
                self.assertEqual(injection.call_args.kwargs['max_questions'],2000)

    def test_generic_transfer_uses_one_external_discovery_not_legacy_question_scan(self):
        import tempfile
        from pecvd.experiments.reporting import run_transfer_experiment,ExperimentReport
        from pecvd.experiments.window_policy import WindowConfig
        policy=SimpleNamespace(config=WindowConfig(gate_scope='position_rank_v4',operation_profile='core'))
        runtime=SimpleNamespace(module=SimpleNamespace(GEVIS_CHECKPOINT={'schema':'gevis.backbone.generic.v10'},
            DISCOVERY_POLICY=policy),encoders={'phase':(None,{'sensor_columns':['external_channel']})},transfer_metadata={'weights_updated':False})
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'external.json';config.write_text('{}',encoding='utf-8')
            with patch('pecvd.process_config.load_process_config',return_value=SimpleNamespace()), \
                 patch('pecvd.experiments.frozen_transfer.bind_external_runtime',return_value=(runtime,pd.DataFrame())), \
                 patch('pecvd.experiments.reporting.run_question_experiment',side_effect=AssertionError('No named redirect')), \
                 patch('pecvd.experiments.reporting.run_injection_experiment',side_effect=AssertionError('No duplicate legacy scan')), \
             patch('pecvd.experiments.gevis_system.run_gevis_predictions',return_value=ExperimentReport('original',pd.DataFrame(),details=pd.DataFrame())), \
                 patch('pecvd.experiments.threshold_comparison.position_calibration_table',return_value=pd.DataFrame()), \
                 patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',return_value=ExperimentReport(
                     'external',pd.DataFrame([{'ok':True}]),metadata={})) as discovery:
                report=run_transfer_experiment(ROOT,process_config=config,backbone_path='unused-source.pt',
                    source_runtime=runtime,prepared_frame=pd.DataFrame())
            self.assertEqual(discovery.call_count,1)
            self.assertEqual(discovery.call_args.kwargs['analysis_mode'],'center')
            self.assertIs(discovery.call_args.kwargs['runtime'],runtime)
            self.assertIn('external_position_calibration',report.diagnostics)
            self.assertFalse(report.metadata['weights_updated'])

    def test_transfer_budget_rejects_invalid_limits(self):
        from pecvd.experiments.reporting import transfer_scan_budget
        runtime=SimpleNamespace(encoders={'phase':(None,{'sensor_columns':['one']})})
        for limit in (True,0,-1,1.5,'1500'):
            with self.assertRaises(ValueError): transfer_scan_budget(runtime,limit)

    def test_adapter_variable_channels_permutation_and_frozen_weights(self):
        source=TinySource(); before={k:v.clone() for k,v in source.state_dict().items()}
        adapter=FrozenChannelAdapter(source,5).eval(); x=torch.randn(2,8,5)
        result=adapter.tokens(x).reshape(2,5,4,4)
        permutation=[4,0,3,1,2]
        permuted=adapter.tokens(x[:,:,permutation]).reshape(2,5,4,4)
        torch.testing.assert_close(permuted,result[:,permutation])
        self.assertEqual(adapter(x).shape,(2,20,2))
        self.assertFalse(any(p.requires_grad for p in adapter.parameters()))
        for k,v in source.state_dict().items(): torch.testing.assert_close(v,before[k])

    def test_reference_scaling_does_not_use_test_values_or_join_segments(self):
        frame,config=fixture(); pairs,support=external_pairs(frame,config,8,seed=42)
        self.assertEqual(list(support.status),['READY'])
        (_,mean,scale)=pairs['external']
        ids=np.array(sorted(frame.run.unique()),object); np.random.default_rng(42).shuffle(ids)
        changed=frame.copy(); changed.loc[~changed.run.isin(ids[:21]),['alpha','beta']]+=1e5
        newer,_=external_pairs(changed,config,8,seed=42)
        np.testing.assert_equal(newer['external'][1],mean); np.testing.assert_equal(newer['external'][2],scale)
        too_long,support=external_pairs(frame,config,10,seed=42)
        self.assertFalse(too_long); self.assertEqual(list(support.status),['NOT_EVALUATED'])

    def test_actual_runtime_functions_with_external_channel_count(self):
        # Real notebook definitions, no real data/weights/network.
        from pecvd.gevis_ui import notebook_runtime_sources
        from pecvd.experiments.frozen_transfer import bind_external_runtime
        from pecvd import gevis_fast
        from pecvd.gevis_backbone import GEVISBackbone
        from types import ModuleType
        import math,time
        from dataclasses import dataclass,asdict
        env={'__name__':'synthetic_transfer','np':np,'pd':pd,'torch':torch,'nn':nn,'math':math,
             'time':time,'dataclass':dataclass,'asdict':asdict,'DEVICE':torch.device('cpu')}
        # Reuse just GEVIS/Question definitions via existing test-safe cell AST.
        import ast,sys
        module=ModuleType('synthetic_transfer'); module.__dict__.update(env); sys.modules[module.__name__]=module
        code=notebook_runtime_sources(ROOT/'notebooks/STEP_3.1_run_gevis_inference.ipynb')['03-types']
        tree=ast.parse(code)
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and getattr(n,'name','') in
               ('PatchEncoder','ByteTextEncoder','GEVIS','GevisQuestion','ReferenceInfo','numeric_state')]
        module.ANSWER_TYPES={'BOOLEAN':0,'CHOICE':1,'ORDINAL':2}
        module.frozen_text=gevis_fast.frozen_text
        import pecvd.gevis_backbone as gb
        module.gevis_backbone=gb; module.gevis_fast=gevis_fast
        exec(compile(ast.Module(nodes,type_ignores=[]),'synthetic:types','exec'),module.__dict__)
        from pecvd.hybrid_patchtst import PortableJointPatchEncoder
        from transformers import PatchTSTConfig
        public=PatchTSTConfig(context_length=16,patch_length=4,patch_stride=4,num_input_channels=1,
            d_model=8,num_hidden_layers=1,num_attention_heads=1,ffn_dim=16,
            share_embedding=True,channel_attention=False,use_cls_token=True,scaling='std')
        source=PortableJointPatchEncoder(3,2,8,public_config=public.to_dict(),d_model=4,nhead=1,layers=1,dropout=0.)
        model=module.GEVIS(numeric_dim=24,d_model=8,dropout=0.)
        module.GEVIS_MODEL=model; module.GEVIS_CHECKPOINT={'_artifact_sha256':'synthetic'}
        from pecvd.experiments.questions import CRITERIA
        module.CRITERIA=CRITERIA
        runtime=SimpleNamespace(module=module,encoders={'source_phase':(source,{'window_length':8,'sensor_columns':['x','y','z']})})
        frame,config=fixture()
        external,support=bind_external_runtime(ROOT,runtime,config,frame,limit=1)
        self.assertIs(external.encoders['external'][0],source)
        self.assertTrue(external.transfer_metadata['native_inference'])
        from pecvd.experiments.evaluation import load_cached_pairs,ask_batched
        current,reference=load_cached_pairs(external,'external',limit=1)[0]
        q=module.GevisQuestion('Sensor alpha: compare mean over full interval','BOOLEAN',tuple(CRITERIA['BOOLEAN']))
        result=ask_batched(external,current,reference,'external',[q])
        self.assertEqual(len(result),1); self.assertIn(result[0]['predicted_query']['sensor'],config.sensor_columns)
        self.assertIs(external.module.GEVIS_MODEL,model)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        # Missing calibration must withhold, not silently fall back to fixed 1.0.
        from pecvd.experiments.reference_policy import ReferencePolicy,fit_records
        external.module.DECISION_POLICY_MODE='reference_quantiles_v1'
        external.module.REFERENCE_POLICY=ReferencePolicy(fit_records([],{}))
        answer=ask_batched(external,current,reference,'external',[q])[0]
        self.assertIsNone(answer['candidate_answer']); self.assertFalse(answer['accepted'])
        self.assertIn('reference_criterion_unsupported',answer['coverage_warnings'])
        self.assertFalse(answer['reference_criterion']['supported']['BOOLEAN'])

        # Rebinding reference criteria is frozen and does not fit held-out runs.
        runtime.module.DECISION_POLICY_MODE='reference_quantiles_v1'
        runtime.module.REFERENCE_POLICY=external.module.REFERENCE_POLICY
        calibrated,_=bind_external_runtime(ROOT,runtime,config,frame,limit=1)
        payload=calibrated.module.REFERENCE_POLICY.payload
        self.assertTrue(payload['groups'])
        self.assertTrue(all(g['n_runs']==21 for g in payload['groups']))
        self.assertIs(calibrated.module.GEVIS_MODEL,model)
        self.assertEqual(payload['fit_split'],'train_or_external_reference_only')
        from pecvd.experiments.evaluation import evaluate_questions,summarize_reference_paths
        rows=evaluate_questions(calibrated,current,reference,'external',limit=12)
        paths=summarize_reference_paths(rows)
        self.assertTrue(len(paths))
        self.assertEqual(set(paths.execution_path),{'oracle','semantic_only','grounding_only','full_gevis'})
        ordinal=paths[paths.answer_type.eq('ORDINAL')]
        self.assertTrue(ordinal.reference_supported_questions.eq(0).all())
        # V2 uses actual runtime functions: operation scores are NOT correctness
        # gates, while required semantic and sensor gates still apply.
        from pecvd.experiments.reference_policy_v2 import ReferencePolicy as V2Policy,fit_records as fit_v2
        runtime.module.DECISION_POLICY_MODE='reference_quantiles_v2'
        runtime.module.REFERENCE_POLICY=V2Policy(fit_v2([],{}))
        v2,_=bind_external_runtime(ROOT,runtime,config,frame,limit=1)
        with torch.no_grad():
            for parameter in model.support_head.parameters(): parameter.zero_()
            [layer for layer in model.support_head.modules() if isinstance(layer,torch.nn.Linear)][-1].bias.fill_(8)
            for name,head in model.semantic_heads.items():
                for parameter in head.parameters(): parameter.zero_()
                if name=='time_scope': [layer for layer in head.modules() if isinstance(layer,torch.nn.Linear)][-1].bias[0]=8
        questions=[module.GevisQuestion('Sensor alpha, full interval: review the difference',kind,tuple(CRITERIA[kind])) for kind in CRITERIA]
        answers=v2.module.ask_gevis(current,reference,'external',questions,module.ReferenceInfo('synthetic',1),
            confidence_floor=.9999,grounding_floor=0.)
        self.assertFalse(answers[0]['accepted'])  # BOOLEAN needs statistic/direction.
        self.assertIn('low_semantic_confidence',answers[0]['coverage_warnings'])
        self.assertTrue(answers[1]['accepted']); self.assertTrue(answers[2]['accepted'])
        discovered=v2.module.review_changes(current,reference,'external',module.ReferenceInfo('synthetic',1))
        self.assertEqual(discovered['contract'],'gevis.learned_generic_discovery.v2')
        self.assertTrue(discovered['complete_coverage'])
        self.assertLess(answers[2]['confidence']['answer'],.9999)
        blocked=v2.module.ask_gevis(current,reference,'external',questions[1:],module.ReferenceInfo('synthetic',1),grounding_floor=1.01)
        self.assertTrue(all(not r['accepted'] and 'low_sensor_grounding_confidence' in r['coverage_warnings'] for r in blocked))

        # Explicit routing uses text, never declared audit/scoring metadata.
        query=module.GevisQuestion('Sensor beta, late interval: review the difference','CHOICE',tuple(CRITERIA['CHOICE']),
            sensor='alpha',time_scope='early')
        routed=v2.module.ask_gevis(current,reference,'external',[query],module.ReferenceInfo('synthetic',1),
            routing_mode='explicit_text',grounding_floor=1.01)[0]
        self.assertEqual(routed['predicted_query']['sensor'],'beta')
        self.assertEqual(routed['predicted_query']['time_scope'],'late')
        self.assertEqual(routed['routing']['sensor_source'],'explicit_text')
        self.assertTrue(routed['gate_checks']['sensor_binding']['passed'])
        self.assertTrue(routed['gate_checks']['semantic_time_scope']['passed'])
        self.assertEqual(routed['accepted'],all(g['passed'] for g in routed['gate_checks'].values()))
        self.assertNotIn('low_sensor_grounding_confidence',routed['coverage_warnings'])
        self.assertEqual(routed['declared_metadata_audit_only']['sensor'],'alpha')
        ambiguous=module.GevisQuestion('Sensor alpha and beta, early interval and late interval: review',
            'CHOICE',tuple(CRITERIA['CHOICE']))
        result=v2.module.ask_gevis(current,reference,'external',[ambiguous],module.ReferenceInfo('synthetic',1),
            routing_mode='explicit_text',grounding_floor=1.01)[0]
        self.assertEqual(result['routing']['sensor_source'],'cross_attention')
        self.assertEqual(result['routing']['scope_source'],'semantic_head')
        self.assertFalse(result['accepted'])
        self.assertIn('low_sensor_grounding_confidence',result['coverage_warnings'])
        # Explicit binding does not invent criterion support or bypass required meanings.
        v2.module.REFERENCE_POLICY=V2Policy(fit_v2([],{}))
        unsupported=v2.module.ask_gevis(current,reference,'external',[query],module.ReferenceInfo('synthetic',1),
            routing_mode='explicit_text')[0]
        self.assertFalse(unsupported['gate_checks']['reference_criterion']['passed'])
        self.assertIsNone(unsupported['candidate_answer']);self.assertFalse(unsupported['accepted'])



    def test_external_missing_source_report_never_requests_external_training(self):
        from pecvd.experiments.reporting import run_transfer_experiment
        with patch('pecvd.experiments.reporting.load_experiment_runtime') as load:
            report=run_transfer_experiment(ROOT,process_config='missing-synthetic.json',backbone_path='missing-synthetic.pt')
        self.assertFalse(report.metadata['model_evaluated']); load.assert_not_called()
        self.assertIn('FROZEN_SOURCE_BACKBONE',set(report.prerequisites.item))
        self.assertNotIn('EXTERNAL_BACKBONE',set(report.prerequisites.item))



if __name__=='__main__': unittest.main()
