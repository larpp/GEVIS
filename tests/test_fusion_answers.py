import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import fusion_answers as fa

ROOT=Path(__file__).resolve().parents[1]


def runtime():
    from tests.test_gevis_fast import model_class
    from pecvd.gevis_backbone import token_descriptions
    torch.manual_seed(5)
    model=model_class()(numeric_dim=8,d_model=16,dropout=0.).requires_grad_(False).eval()
    def state(encoder,a,b):
        # Real paired numeric interface in a small invented grid, no labels.
        am=a.reshape(1,2,4,2).mean(2).permute(0,2,1)
        bm=b.reshape(1,2,4,2).mean(2).permute(0,2,1)
        d=am-bm
        return torch.stack([am,bm,d,d.abs(),am.square(),bm.square(),d.square(),torch.ones_like(d)],-1).reshape(1,4,8)
    def prepare(rows,meta):return torch.tensor(rows[['a','b']].to_numpy(),dtype=torch.float32)[None],{'position_ratio':.2}
    m=SimpleNamespace(ROOT=ROOT,DECISION_SYSTEM=fa.SYSTEM,DEVICE='cpu',GEVIS_MODEL=model,
        STEP_ENCODERS={'phase':(torch.nn.Identity(),{'sensor_columns':['a','b'],'window_length':8})},
        numeric_state=state,numeric_token_descriptions=lambda step,meta,p,n:[token_descriptions(step,['a','b'],p)]*n,
        prepare_window=prepare,RUN_COLUMN='process_key',STEP_COLUMN='step_label',ORDER_COLUMN='row_in_process',
        ENCODER_ROOT=Path('synthetic'),BACKBONE_PATH=Path('synthetic/backbone.pt'),FUSION_ANSWER_HEAD=None)
    return SimpleNamespace(module=m,encoders=m.STEP_ENCODERS)


def source_rows(split):
    rng=np.random.default_rng(1 if split=='train' else 2)
    return [dict(current=rng.normal(0,.2,(8,2)),reference=np.zeros((8,2)),position=i%4,
        run_id=f'{split}-{i}',split=split,cache_index=i,reference_run_ids=['train-ref'],cohort='recipe') for i in range(4)]


def learnable_cache(split):
    rows=[]
    for i in range(12):
        for kind in fa.KINDS:
            for positive in (0,1):
                x=torch.zeros(10);x[positive]=3.;x[2]=i/12;x[3+fa.KINDS.index(kind)]=1.
                rows.append(dict(features=x,answer_type=kind,label=positive,split=split,
                    run_id=f'{split}-{i}',step='phase',center_sensor=('a','b')[positive]))
    return rows


class FusionAnswerTests(unittest.TestCase):
    def test_actual_fusion_fixed_questions_and_selected_center_features(self):
        r=runtime();m=r.module;a=np.zeros((8,2));a[:,0]=2.;b=np.zeros_like(a)
        with patch.object(m.GEVIS_MODEL,'forward',wraps=m.GEVIS_MODEL.forward) as forward:
            x,tasks=fa.fusion_features(m,a,b,'phase',['a','b'])
        self.assertEqual(forward.call_count,1)
        self.assertEqual(forward.call_args.args[1],[fa.QUESTIONS[k] for k in fa.KINDS]*2)
        self.assertFalse(torch.equal(x[0],x[3]))
        self.assertEqual(tasks[0],('a','BOOLEAN'));self.assertEqual(tasks[3],('b','BOOLEAN'))
        self.assertTrue(all(not p.requires_grad for p in m.GEVIS_MODEL.parameters()))

    def test_targets_change_with_ui_center_on_identical_signal_and_fixed_question(self):
        r=runtime()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons',return_value=source_rows('train')),contextlib.redirect_stdout(io.StringIO()):
            rows=fa.controlled_examples(r.module,'train')
        booleans=[e for e in rows if e['answer_type']=='BOOLEAN' and e['injected']]
        pairs={}
        for e in booleans:pairs.setdefault(e['pair_id'],[]).append(e)
        self.assertTrue(pairs)
        for group in pairs.values():
            self.assertEqual({e['label'] for e in group},{0,1})
            self.assertEqual({e['question'] for e in group},{fa.QUESTIONS['BOOLEAN']})
        self.assertTrue(all(e['label']==0 for e in rows if not e['injected']))
        self.assertTrue(all(e['label']==0 for e in rows if e['injected'] and not e['center_is_perturbed']))

    def test_genuine_learning_gradients_best_validation_and_no_backbone_update(self):
        r=runtime();before={k:v.clone() for k,v in r.module.GEVIS_MODEL.state_dict().items()}
        cfg=replace(fa.FusionConfig(),epochs=30,width=16,learning_rate=.02,max_seconds=20.)
        with contextlib.redirect_stdout(io.StringIO()):head,report=fa.fit_readout(learnable_cache('train'),learnable_cache('validation'),cfg)
        self.assertGreater(report['optimizer_updates'],0)
        self.assertEqual(report['validation_loss'],min(row['validation_loss'] for row in report['history']))
        x=torch.stack([e['features'] for e in learnable_cache('validation') if e['answer_type']=='BOOLEAN'])
        predicted=head(x,'BOOLEAN').argmax(-1).tolist()
        expected=[e['label'] for e in learnable_cache('validation') if e['answer_type']=='BOOLEAN']
        self.assertEqual(predicted,expected)
        for k,v in before.items():self.assertTrue(torch.equal(v,r.module.GEVIS_MODEL.state_dict()[k]))
        self.assertFalse(any(p.requires_grad for p in head.parameters()))

    def test_split_overlap_test_fitting_and_single_class_rejected(self):
        train=learnable_cache('train');val=learnable_cache('validation')
        bad=copy.deepcopy(val);bad[0]['run_id']=train[0]['run_id']
        with self.assertRaisesRegex(ValueError,'overlap'):fa.fit_readout(train,bad)
        bad=copy.deepcopy(train);bad[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'TRAIN optimization'):fa.fit_readout(bad,val)
        bad=copy.deepcopy(train)
        for e in bad:e['label']=0
        with self.assertRaisesRegex(ValueError,'two distinct'):fa.fit_readout(bad,val)

    def test_missing_head_never_calls_numerical_fallback(self):
        from pecvd.experiments.discovery import review_changes
        from pecvd.experiments.gevis_system import ready_gevis_runtime
        r=runtime();m=r.module
        m.DISCOVERY_POLICY=SimpleNamespace(analyze=lambda *a:(_ for _ in ()).throw(AssertionError('No rule')))
        with self.assertRaisesRegex(RuntimeError,'no q99/GMM fallback'):ready_gevis_runtime(ROOT,r)
        with self.assertRaisesRegex(RuntimeError,'head missing'):review_changes(m,None,None,'phase',None,center_sensor='a')

    def test_primary_runtime_outputs_actual_head_logits_with_no_rules(self):
        from pecvd.experiments.discovery import review_changes
        r=runtime();m=r.module;a=np.zeros((8,2));a[:,0]=2.;b=np.zeros_like(a)
        x,_=fa.fusion_features(m,a,b,'phase',['a'])
        m.FUSION_ANSWER_HEAD=fa.ContextReadout(x.shape[-1],16).eval()
        # Known head tensors exercise decoding; production requires trained checkpoint evidence.
        with torch.no_grad():
            for p in m.FUSION_ANSWER_HEAD.parameters():p.zero_()
            m.FUSION_ANSWER_HEAD.answers['BOOLEAN'].bias[1]=4.
        current=pd.DataFrame(a,columns=['a','b']);reference=pd.DataFrame(b,columns=['a','b'])
        with patch('pecvd.experiments.window_policy.WindowPolicy.analyze',side_effect=AssertionError('No rule call')):
            result=review_changes(m,current,reference,'phase',None,center_sensor='a')
        self.assertTrue(result['changed']);self.assertEqual(result['answers'][0]['answer'],'Yes')
        self.assertAlmostEqual(result['answers'][0]['probabilities']['Yes'],float(torch.tensor([0.,4.]).softmax(0)[1]),places=6)
        self.assertEqual([a['question'] for a in result['answers']],[fa.QUESTIONS[k] for k in fa.KINDS])
        with self.assertRaisesRegex(ValueError,'analysis center'):review_changes(m,current,reference,'phase',None)

    def test_complete_step_selection_covers_rows_and_never_mixes_runs_or_occurrences(self):
        m=runtime().module
        frame=pd.DataFrame(dict(process_key=['p1']*22+['p2']*8,step_label=['phase']*30,
            row_in_process=list(range(22))+list(range(8)),step_number=[1]*30,a=0.,b=0.))
        reference=pd.DataFrame(dict(process_key=['ref']*24,step_label=['phase']*24,
            row_in_process=range(24),step_number=[1]*24,a=0.,b=0.))
        with patch.object(fa,'review',return_value={'answers':[]}) as review:
            result=fa.review_selected_step(m,frame,reference,'p1','phase','a')
        covered=set()
        for window in result['windows']:covered.update(range(window['start_row'],window['end_row']))
        self.assertEqual(covered,set(range(22)));self.assertIsNone(result['whole_step_decision'])
        for call in review.call_args_list:self.assertEqual(set(call.args[1].process_key),{'p1'})
        frame.loc[11:21,'step_number']=2
        with self.assertRaisesRegex(ValueError,'occurrence explicitly'):fa.review_selected_step(m,frame,reference,'p1','phase','a')

    def test_primary_evaluation_is_head_accuracy_and_center_swap_not_rule_recall(self):
        from pecvd.experiments.gevis_system import run_gevis_experiment
        r=runtime();m=r.module;x,_=fa.fusion_features(m,np.zeros((8,2)),np.zeros((8,2)),'phase',['a'])
        m.FUSION_ANSWER_HEAD=fa.ContextReadout(x.shape[-1],16).eval()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons',return_value=source_rows('test')), \
             patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',side_effect=AssertionError('No old metric path')),contextlib.redirect_stdout(io.StringIO()):
            report=run_gevis_experiment(ROOT,runtime=r,split='test')
        self.assertEqual(set(report.summary.answer_type),set(fa.KINDS))
        self.assertIn('majority_baseline_accuracy',report.summary)
        self.assertIn('macro_f1',report.summary)
        self.assertFalse(report.summary.isna().any().any())
        self.assertEqual(len(report.diagnostics['ordinal_metrics']),1)
        self.assertEqual(len(report.diagnostics['boolean_metrics']),1)
        self.assertGreater(report.diagnostics['center_swap'].pairs.iloc[0],0)
        self.assertFalse(report.metadata['rule_fallback'])
        self.assertFalse(report.metadata['original_cross_run_fault_accuracy_available'])

    def test_step4_original_predictions_assemble_real_answers_with_existing_center(self):
        from pecvd.experiments.gevis_system import run_gevis_predictions
        r=runtime();m=r.module
        a=np.zeros((8,2));a[:,0]=2.;b=np.zeros_like(a)
        x,_=fa.fusion_features(m,a,b,'phase',['a'])
        m.FUSION_ANSWER_HEAD=fa.ContextReadout(x.shape[-1],16).eval()
        before={k:v.clone() for k,v in m.FUSION_ANSWER_HEAD.state_dict().items()}
        current=pd.DataFrame(a,columns=['a','b']);reference=pd.DataFrame(b,columns=['a','b'])
        expected=fa.review(m,current,reference,'phase',None,center_sensor='a')['answers']
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',
                   return_value=[(current,reference),(current,reference)]) as pairs, \
             patch.object(fa,'predict_original',side_effect=AssertionError('No legacy row assembly')), \
             patch.object(fa,'fit_readout',side_effect=AssertionError('No inference training')):
            report=run_gevis_predictions(ROOT,runtime=r,steps=['phase'],pairs_per_step=2,seed=9)
        self.assertEqual(pairs.call_args.kwargs,dict(split='test',limit=2,seed=9))
        self.assertEqual(len(report.details),6)
        self.assertTrue(report.details.columns.is_unique)
        self.assertEqual(report.details.center_sensor.tolist(),['a']*3+['b']*3)
        self.assertEqual(report.details.pair_index.tolist(),[0]*3+[1]*3)
        self.assertEqual(set(report.details.step),{'phase'})
        self.assertEqual(set(report.details.answer_type),set(fa.KINDS))
        self.assertEqual(report.details.iloc[:3].probabilities.tolist(),[a['probabilities'] for a in expected])
        self.assertFalse(report.summary.accuracy_available.iloc[0])
        self.assertNotIn('answer_accuracy',report.summary)
        self.assertNotIn('label',report.details)
        self.assertFalse(report.metadata['original_cross_run_fault_accuracy_available'])
        self.assertFalse(report.metadata['rule_fallback'])
        for k,v in before.items():torch.testing.assert_close(m.FUSION_ANSWER_HEAD.state_dict()[k],v)

    def test_original_prediction_reporting_still_requires_a_head(self):
        from pecvd.experiments.gevis_system import run_gevis_predictions
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',
                   side_effect=AssertionError('Do not load inputs before trained head check')):
            with self.assertRaisesRegex(RuntimeError,'head missing'):
                run_gevis_predictions(ROOT,runtime=runtime())

    def test_checkpoint_actual_training_identity_create_only_reuse_and_no_test_read(self):
        r=runtime();cfg=replace(fa.FusionConfig(),epochs=2,width=16)
        with tempfile.TemporaryDirectory() as folder:
            r.module.BACKBONE_PATH=Path(folder)/'backbone.pt'
            def controlled(module,split,config):return learnable_cache(split)
            with patch.object(fa,'identity',return_value={'synthetic':True}), \
                 patch.object(fa,'controlled_examples',side_effect=controlled) as examples,contextlib.redirect_stdout(io.StringIO()):
                training=fa.prepare(r,cfg)
                self.assertEqual([c.args[1] for c in examples.call_args_list],['train','validation'])
                self.assertGreater(training['optimizer_updates'],0)
                saved=r.module.FUSION_ANSWER_PATH.read_bytes()
                with patch.object(fa,'fit_readout',side_effect=AssertionError('No inference fit')):fa.prepare(r,cfg)
                self.assertEqual(saved,r.module.FUSION_ANSWER_PATH.read_bytes())
                payload=torch.load(r.module.FUSION_ANSWER_PATH,weights_only=False)
                payload['training']['optimizer_updates']=0
                torch.save(payload,r.module.FUSION_ANSWER_PATH)
                with self.assertRaisesRegex(ValueError,'actual answer-head training'):fa.attach(r.module,cfg)

    def test_goal_and_notebook_default_path_is_fixed_question_fusion(self):
        settings=json.loads((ROOT/'config/experiments/gevis_pipeline.json').read_text(encoding='utf-8'))
        self.assertEqual(settings['decision_system'],'fusion_answers')
        for name,ident in [('STEP_3_train_gevis_decision.ipynb','fusion-answer-train'),('CVD_evaluate.ipynb','fusion-answer-test')]:
            doc=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
            self.assertIn(ident,[c.get('id') for c in doc['cells']])
        direction=(ROOT/'.agents/eidos/direction.md').read_text(encoding='utf-8')
        self.assertIn('D0011',direction);self.assertIn('never fall back',direction)

    def test_end_to_end_actual_fusion_cache_fit_checkpoint_reload_and_ui_answer(self):
        r=runtime();m=r.module;cfg=replace(fa.FusionConfig(),epochs=2,width=16)
        def source(*args,**kwargs):return source_rows(kwargs['split'])
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(fa,'identity',return_value={'synthetic_e2e':True}), \
             patch('pecvd.experiments.discovery_calibration.cached_comparisons',side_effect=source), \
             contextlib.redirect_stdout(io.StringIO()):
            m.BACKBONE_PATH=Path(folder)/'backbone.pt'
            training=fa.prepare(r,cfg)
            self.assertGreater(training['optimizer_updates'],0)
            before=copy.deepcopy(m.FUSION_ANSWER_HEAD.state_dict())
            m.FUSION_ANSWER_HEAD=None;fa.attach(m,cfg)
            for k,v in before.items():self.assertTrue(torch.equal(v,m.FUSION_ANSWER_HEAD.state_dict()[k]))
            current=pd.DataFrame(source_rows('test')[0]['current'],columns=['a','b'])
            result=fa.review(m,current,current.copy(),'phase',None,center_sensor='b')
            self.assertEqual(result['center_sensor'],'b')
            self.assertEqual(len(result['answers']),3)
            self.assertTrue(all(a['decision_source']==fa.SCHEMA for a in result['answers']))

    def test_step3_and_step4_execute_fusion_hooks_without_numerical_preparation(self):
        r=runtime();report=SimpleNamespace(summary=None,by_step=None,
            details=pd.DataFrame([dict(answer_type=k,label=0,prediction=0,step='phase') for k in fa.KINDS]),
            errors=None,metadata={'decision_system':'fusion_answers'},
            diagnostics={'center_swap':None,'off_center':None,'confusion':None,'intervention_metrics':None,'delay_metrics':None,'reconstruction_metrics':None})
        doc=json.loads((ROOT/'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='fusion-answer-train'))
        env=dict(RUN_FUSION_ANSWER_TRAINING=True,ROOT=ROOT,gevis=r.module.GEVIS_MODEL,encoders=r.encoders,
            ENCODER_ROOT='synthetic',backbone_path=Path('synthetic.pt'),DEVICE='cpu',
            numeric_state=r.module.numeric_state,numeric_token_descriptions=r.module.numeric_token_descriptions,
            RUN_COLUMN='process_key',STEP_COLUMN='step_label',ORDER_COLUMN='row_in_process',pd=pd,display=lambda *a:None)
        import importlib
        with patch('importlib.reload',side_effect=lambda m:m),patch.object(fa,'prepare',return_value={'history':[],'augmentation_coverage':[]}) as prepare, \
             patch.object(fa,'evaluate',return_value=report),contextlib.redirect_stdout(io.StringIO()):
            env['importlib']=importlib
            exec(compile(source,'STEP3:actual-fusion-hook','exec'),env)
        self.assertIs(prepare.call_args.args[0].module.GEVIS_MODEL,r.module.GEVIS_MODEL)
        doc=json.loads((ROOT/'notebooks/CVD_evaluate.ipynb').read_text(encoding='utf-8'))
        config_source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='reference-config'))
        with patch.dict('os.environ',{'GEVIS_DECISION_POLICY':'legacy'}),contextlib.redirect_stdout(io.StringIO()):
            exec(config_source,{'PIPELINE_SETTINGS':{'decision_system':'fusion_answers'}})
        source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='fusion-answer-test'))
        with patch('pecvd.experiments.gevis_system.run_gevis_experiment',return_value=report) as run, \
             patch('pecvd.experiments.gevis_system.run_gevis_predictions',return_value=report), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(source,dict(RUN_FUSION_ANSWER_EVALUATION=True,ROOT=ROOT,REFERENCE_RUNTIME=r,
                SELECTED_STEPS=None,TEST_PAIRS_PER_STEP=2,TEST_WINDOWS_PER_STEP=32,SEED=42,display=lambda *a:None))
        self.assertEqual(run.call_args.kwargs['split'],'test')

    def test_actual_step31_question_api_uses_ui_center_not_question_metadata(self):
        import ast,sys,types
        r=runtime();m=types.ModuleType('_synthetic_fusion_question_api')
        m.__dict__.update(r.module.__dict__)
        m.torch=torch
        x,_=fa.fusion_features(m,np.zeros((8,2)),np.zeros((8,2)),'phase',['a'])
        m.FUSION_ANSWER_HEAD=fa.ContextReadout(x.shape[-1],16).eval()
        doc=json.loads((ROOT/'notebooks/STEP_3.1_run_gevis_inference.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='03-function'))
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='ask_gevis')
        question=SimpleNamespace(text=fa.QUESTIONS['BOOLEAN'],answer_type='BOOLEAN',criteria=fa.ANSWER_CRITERIA['BOOLEAN'],sensor='b')
        frame=pd.DataFrame(np.zeros((8,2)),columns=['a','b'])
        with patch.dict(sys.modules,{m.__name__:m}):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'STEP3.1:actual-question-api','exec'),m.__dict__)
            result=m.ask_gevis(frame,frame.copy(),'phase',[question],None,center_sensor='a')
        self.assertEqual(result[0]['center_sensor'],'a')
        self.assertEqual(result[0]['question'],fa.QUESTIONS['BOOLEAN'])
        self.assertEqual(result[0]['decision_source'],fa.SCHEMA)
