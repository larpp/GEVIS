from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from pecvd.gevis_backbone import (
    ANSWER_TYPES,
    BACKBONE_SOURCE_SCHEMA,
    BackbonePretrainHeads,
    E5TextEncoder,
    GEVISBackbone,
    QUESTION_BANK,
    SEMANTIC_QUESTION_BANK,
    SEMANTIC_TYPES,
    controlled_numeric_batch,
    make_semantic_questions,
    real_step1_cross_batch,
    grounded_numeric_batch,
    semantic_counterfactual,
    semantic_diagnostics,
    _semantic_loss,
    train_text_encoder,
)
from pecvd.gevis_fast import frozen_text


class GevisBackboneTests(unittest.TestCase):
    def test_new_names_have_every_semantic_combination_and_splits_are_disjoint(self):
        banks = {split: make_semantic_questions(split, variants=2) for split in ('train','validation','test','audit')}
        names = {split: {item.step for item in rows} | {item.sensor for item in rows} for split, rows in banks.items()}
        for split, rows in banks.items():
            for step in {item.step for item in rows}:
                combinations = {(item.statistic,item.direction,item.time_scope) for item in rows if item.step == step}
                self.assertEqual(len(combinations), 45)
            for other in banks:
                if split != other: self.assertFalse(names[split] & names[other])
        self.assertNotEqual({item.step for item in make_semantic_questions('train',seed=1)}, {item.step for item in make_semantic_questions('train',seed=2)})

    def test_numeric_grounding_changes_question_not_numeric_state(self):
        numeric, questions, types, descriptions, labels, masks = grounded_numeric_batch(18,torch.device('cpu'),9,split='audit')
        self.assertEqual(len(questions), 72)
        for row in range(0,len(questions),4):
            for other in range(row+1,row+4): torch.testing.assert_close(numeric[row],numeric[other])
            self.assertNotEqual(questions[row], questions[row+1])
            self.assertEqual(questions[row],questions[row+2])  # Sensor IDs are absent from text.
            torch.testing.assert_close(masks[row],masks[row+1])
            self.assertFalse(torch.equal(masks[row],masks[row+2]))
        self.assertTrue((labels.reshape(-1,2)[:,0] != labels.reshape(-1,2)[:,1]).any())
        self.assertEqual(types, ['BOOLEAN'] * 72)

    def test_notebook_descriptor_reuse_preserves_outputs_and_gradients(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename, cell_id in (
            ('STEP_3_train_gevis_decision.ipynb', '02-config'),
            ('STEP_3.1_run_gevis_inference.ipynb', '03-types'),
        ):
            with self.subTest(notebook=filename):
                notebook = json.loads((root / 'notebooks' / filename).read_text(encoding='utf-8'))
                source = ''.join(next(c['source'] for c in notebook['cells'] if c.get('id') == cell_id))
                tree = ast.parse(source)
                classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {'ByteTextEncoder', 'GEVIS'}]
                namespace = {'torch': torch, 'nn': torch.nn, 'math': math, 'ANSWER_TYPES': ANSWER_TYPES, 'frozen_text': frozen_text}
                exec(compile(ast.Module(body=classes, type_ignores=[]), filename, 'exec'), namespace)
                model = namespace['GEVIS'](numeric_dim=8, d_model=32, dropout=0.0)
                reference = copy.deepcopy(model)
                # Compare against the previous expanded-description computation.
                reference_tree = copy.deepcopy(tree)
                reference_class = next(node for node in reference_tree.body if isinstance(node, ast.ClassDef) and node.name == 'GEVIS')
                forward = next(node for node in reference_class.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
                description_if = next(node for node in forward.body if isinstance(node, ast.If) and 'token_descriptions is not None' in ast.unparse(node.test))
                description_if.body = description_if.body[:1] + ast.parse('flat = [text for items in token_descriptions for text in items]\ndescriptors = self.descriptor_norm(self.text(flat, device).reshape_as(numeric_keys))\ngate = self.fusion_gate(torch.cat([numeric_keys, descriptors], -1))\nkeys = numeric_keys + gate * descriptors').body
                exec(compile(ast.fix_missing_locations(ast.Module(body=[reference_class], type_ignores=[])), filename, 'exec'), namespace)
                reference_model = namespace['GEVIS'](numeric_dim=8, d_model=32, dropout=0.0)
                reference_model.load_state_dict(reference.state_dict())
                numeric = torch.randn(4, 3, 8)
                descriptions = [['sensor A early', 'sensor A late', 'sensor B whole']] * 4
                questions = ['is the level higher?'] * 4
                criteria = [['no', 'yes']] * 4
                with patch.object(model.text, 'forward', wraps=model.text.forward) as calls:
                    logits, attention = model(numeric, questions, ['BOOLEAN'] * 4, criteria, descriptions)
                self.assertEqual(calls.call_args_list[1].args[0], descriptions[0])
                expected_logits, expected_attention = reference_model(numeric, questions, ['BOOLEAN'] * 4, criteria, descriptions)
                torch.testing.assert_close(torch.stack(logits), torch.stack(expected_logits), atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(attention, expected_attention, atol=2e-6, rtol=2e-5)
                torch.stack(logits).square().sum().backward()
                torch.stack(expected_logits).square().sum().backward()
                for (name, parameter), (_, expected) in zip(model.named_parameters(), reference_model.named_parameters()):
                    if parameter.grad is not None:
                        self.assertIsNotNone(expected.grad, name)
                        torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-5, rtol=2e-4, msg=name)
                self.assertGreater(float(model.text.token.weight.grad.abs().sum()), 0.0)
                with self.assertRaises(ValueError):
                    model(numeric, questions, ['BOOLEAN'] * 4, criteria, [['short']] * 4)

    def test_source_schema_and_question_count_are_explicit(self) -> None:
        self.assertEqual(BACKBONE_SOURCE_SCHEMA, "gevis.backbone_source.v9")
        self.assertEqual(sum(map(len, QUESTION_BANK.values())), 112)

    def test_question_splits_include_supported_and_unsupported_examples(self) -> None:
        for split, examples in QUESTION_BANK.items():
            self.assertTrue(any(item.supported for item in examples), split)
            self.assertTrue(any(not item.supported for item in examples), split)
            present = {item.answer_type for item in examples if item.supported}
            self.assertEqual(present, set(ANSWER_TYPES))

    def test_semantic_curriculum_holds_out_process_identifiers(self) -> None:
        train_text = " ".join(item.text for item in SEMANTIC_QUESTION_BANK["train"])
        validation_text = " ".join(item.text for item in SEMANTIC_QUESTION_BANK["validation"])
        self.assertNotIn("unseen_gamma", train_text)
        self.assertNotIn("unseen_delta", train_text + validation_text)
        self.assertNotIn("unseen_omega", train_text + validation_text)
        for examples in SEMANTIC_QUESTION_BANK.values():
            self.assertEqual({item.answer_type for item in examples},set(ANSWER_TYPES))

    def test_active_pretraining_questions_are_english(self) -> None:
        texts = [item.text for rows in QUESTION_BANK.values() for item in rows]
        texts += [item.text for rows in SEMANTIC_QUESTION_BANK.values() for item in rows]
        self.assertTrue(texts)
        self.assertTrue(all(text.isascii() for text in texts))
        self.assertTrue(any("reference" in text.lower() for text in texts))

    def test_e5_encoder_is_frozen_and_projection_trains_without_network(self) -> None:
        class FakeTokenizer:
            def __init__(self): self.seen = []
            def __call__(self, texts, **kwargs):
                self.seen = list(texts)
                ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
                return {'input_ids': ids, 'attention_mask': ids.ne(0).long()}
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.config = SimpleNamespace(hidden_size=4); self.embedding = torch.nn.Embedding(8, 4)
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        tokenizer, model = FakeTokenizer(), FakeModel()
        encoder = E5TextEncoder(6, tokenizer=tokenizer, model=model)
        encoder.train()
        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        output = encoder(["check level", "compare slope"], torch.device('cpu'))
        self.assertEqual(tuple(output.shape), (2, 6))
        self.assertTrue(all(text.startswith('query: ') for text in tokenizer.seen))
        seen=list(tokenizer.seen)
        for bad in (chr(0xac00),chr(0xfffd)):
            with self.assertRaisesRegex(ValueError,'English E5 input required'):
                encoder([bad,'compare slope'],torch.device('cpu'))
        self.assertEqual(tokenizer.seen,seen)
        output.square().mean().backward()
        self.assertIsNotNone(encoder.projection.weight.grad)
        self.assertIsNone(model.embedding.weight.grad)

    def test_each_type_has_all_slots_and_counterfactual_changes_one_only(self):
        rows=make_semantic_questions('train',variants=2)
        for kind in ANSWER_TYPES:
            selected=[item for item in rows if item.answer_type==kind]
            for name in {item.step for item in selected}:
                self.assertEqual(len({(item.statistic,item.direction,item.time_scope) for item in selected if item.step==name}),45)
        for item in rows[:12]:
            for field in SEMANTIC_TYPES:
                changed=semantic_counterfactual(item,field)
                self.assertEqual(sum(getattr(item,k)!=getattr(changed,k) for k in SEMANTIC_TYPES),1)
                self.assertEqual((item.step,item.sensor,item.answer_type),(changed.step,changed.sensor,changed.answer_type))
                self.assertNotEqual(item.text,changed.text)
        banks={split:{item.text for item in make_semantic_questions(split,variants=1)} for split in ('train','validation','test','audit')}
        for split,texts in banks.items():
            for other in banks:
                if split!=other: self.assertFalse(texts & banks[other])

    def test_contrast_loss_gradients_diagnostics_and_bounded_text_fit(self):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone=GEVISBackbone(numeric_dim=264,d_model=32,dropout=0.).to(device)
        heads=BackbonePretrainHeads(32).to(device)
        parameters=sum(p.numel() for p in backbone.parameters())+sum(p.numel() for p in heads.parameters())
        rows=list(make_semantic_questions('train',variants=1))[:9]
        loss,parts=_semantic_loss(backbone,heads,rows,device)
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        for head in heads.semantic_heads.values():
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertEqual(set(parts),{'semantic_'+field+'_loss' for field in SEMANTIC_TYPES})
        small_bank={split:make_semantic_questions(split,variants=1) for split in ('train','validation','test','audit')}
        with patch('pecvd.gevis_backbone.SEMANTIC_QUESTION_BANK',small_bank):
            progress=[]
            history,best=train_text_encoder(backbone,heads,device,epochs=1,batch_size=8,learning_rate=1e-3,patience=2,seed=3,steps_per_epoch=1,progress=progress.append)
        self.assertEqual(best,1); self.assertEqual(len(history),1); self.assertEqual(progress[0]['updates'],1)
        self.assertEqual(parameters,sum(p.numel() for p in backbone.parameters())+sum(p.numel() for p in heads.parameters()))
        details=semantic_diagnostics(backbone,heads,list(small_bank['audit']),'audit',device)
        self.assertEqual(len(details['contrast_pairs']),9)
        self.assertEqual({row['answer_type'] for row in details['contrast_pairs']},set(ANSWER_TYPES))
        for field in SEMANTIC_TYPES:
            self.assertEqual(sum(row['count'] for row in details['confusion'] if row['slot']==field),135)
        # Execute the actual STEP 2 reporting cell on synthetic weights/banks.
        import pandas as pd
        import pecvd.gevis_backbone as helper
        notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in notebook['cells'] if c.get('id')=='04-text-training'))
        from tests.notebook_branch import training_source
        source=training_source(source)
        config=type('SyntheticConfig',(),{'numeric_dim':264,'d_model':32,'dropout':0.,'text_encoder_name':'byte','text_encoder_revision':'synthetic','max_text_epochs':1,
            'text_batch_size':8,'text_learning_rate':1e-3,'text_patience':2,'semantic_steps_per_epoch':1,'compare_existing_backbone':True})()
        ns={'SESSION_SIGNATURE':'STEP2_GEVIS_GENERIC_DISCOVERY_V10',
            'ARTIFACT_VERSION':'gevis_backbone_e5_v1','EXPECTED_ARTIFACT_VERSION':'gevis_backbone_e5_v1',
            'CONFIG':config,'SEED':3,'DEVICE':device,'ROOT':Path('synthetic_root'),
            'torch':torch,'pd':pd,'GEVISBackbone':GEVISBackbone,'BackbonePretrainHeads':BackbonePretrainHeads,
            'gevis_backbone':helper,'QUESTION_BANK':QUESTION_BANK,'SEMANTIC_QUESTION_BANK':small_bank,
            'train_text_encoder':lambda *a,**kw:([{'epoch':1}],1),
            'evaluate_text':helper.evaluate_text,'evaluate_semantics':helper.evaluate_semantics,'display':lambda _:None}
        # Baseline is synthetic too: neither path probing nor torch.load touches
        # a private baseline artifact. Both branches see exactly the same rows.
        baseline={'schema':'gevis.backbone.v4','d_model':32,'numeric_dim':264,
                  'backbone_state':backbone.state_dict(),'pretrain_heads_state':heads.state_dict()}
        with patch.object(Path,'exists',return_value=True),patch.object(torch,'load',return_value=baseline):
            exec(compile(source,'STEP2:synthetic-semantic-report','exec'),ns)
        self.assertEqual(len(ns['SEMANTIC_BY_TYPE']),9)
        self.assertEqual(len(ns['SEMANTIC_FIXED_COMPARISON']),18)
        self.assertEqual(ns['BASELINE_COMPARISON_STATUS'],'same_questions_read_only_baseline')

    def test_same_answer_type_has_minimal_semantic_contrasts(self) -> None:
        examples = SEMANTIC_QUESTION_BANK["train"]
        for field, values in SEMANTIC_TYPES.items():
            self.assertEqual({getattr(item, field) for item in examples}, set(values), field)

    def test_controlled_batch_uses_counterfactual_sensor_time_targets(self) -> None:
        numeric, questions, answer_types, token_descriptions, feature_target, targets = controlled_numeric_batch(
            8, 12, 264, torch.device("cpu"), 7
        )
        self.assertEqual(tuple(numeric.shape), (8, 12, 264))
        self.assertEqual(len(questions), 8)
        self.assertEqual(answer_types, ["CHOICE"] * 8)
        self.assertEqual(len(token_descriptions), 8)
        self.assertEqual(len(token_descriptions[0]), 12)
        self.assertEqual(tuple(feature_target.shape), (8,))
        self.assertEqual(tuple(targets["sensor"].shape), (8,))
        self.assertEqual(tuple(targets["scope"].shape), (8,))
        self.assertEqual(targets["patch_count"], 3)
        torch.testing.assert_close(numeric[0::2], numeric[1::2])
        self.assertTrue(torch.all(targets["sensor"][0::2] != targets["sensor"][1::2]))
        self.assertTrue(all(text.isascii() for row in token_descriptions for text in row))

    def test_real_step1_batch_reuses_state_and_supports_variable_patch_count(self) -> None:
        state = torch.randn(3, 20, 392)
        numeric, questions, kinds, descriptions, feature, targets = real_step1_cross_batch(
            state, "unseen phase", [f"sensor {i}" for i in range(4)], seed=19, split="validation"
        )
        self.assertEqual(tuple(numeric.shape), (18, 20, 392))
        torch.testing.assert_close(numeric[0::2], numeric[1::2])
        torch.testing.assert_close(numeric[:6], numeric[6:12])
        torch.testing.assert_close(numeric[:6], numeric[12:18])
        self.assertTrue(torch.all(targets["sensor"][0::2] != targets["sensor"][1::2]))
        self.assertEqual(tuple(targets["sensor"].shape), (18,))
        self.assertEqual(targets["patch_count"], 5)
        self.assertEqual(kinds, [kind for kind in ANSWER_TYPES for _ in range(6)])
        self.assertEqual(len(descriptions[0]), 20)
        self.assertEqual(tuple(feature.shape), (18,))
        self.assertTrue(all(question.isascii() for question in questions))

    def test_backbone_and_small_heads_have_expected_contract(self) -> None:
        backbone = GEVISBackbone(numeric_dim=264, d_model=32, dropout=0.0)
        heads = BackbonePretrainHeads(d_model=32)
        numeric = torch.zeros((3, 10, 264))
        fused, attention = backbone(
            numeric,
            ["Is a review needed?", "What has changed?", "What is the degree of change?"],
            ["BOOLEAN", "CHOICE", "ORDINAL"],
        )
        self.assertEqual(tuple(fused.shape), (3, 32))
        self.assertEqual(tuple(attention.shape), (3, 10))
        self.assertEqual(tuple(heads.type_head(fused).shape), (3, 3))
        self.assertEqual(tuple(heads.support_head(fused).shape), (3, 1))
        self.assertEqual(tuple(heads.semantic_heads["statistic"](fused).shape), (3, 5))

    def test_refined_binding_batch_has_sensor_and_scope_minimal_pairs(self):
        names=['pressure_A','pressure_B','flow_other']
        for patches in (3,5,8):
            state=torch.randn(2,len(names)*patches,40)
            before=state.clone()
            packed=real_step1_cross_batch(state,'invented phase',names,seed=3,binding_refinement=True)
            numeric,questions,kinds,descriptions,feature,targets=packed
            self.assertEqual(len(numeric),36)
            self.assertEqual(targets['group_size'],6)
            self.assertEqual(kinds,[kind for kind in ANSWER_TYPES for _ in range(12)])
            torch.testing.assert_close(state,before)
            for start in range(0,len(numeric),6):
                self.assertEqual(targets['scope'][start:start+6].tolist(),[0,0,1,1,2,2])
                self.assertNotEqual(int(targets['sensor'][start]),int(targets['sensor'][start+1]))
                self.assertEqual(int(targets['sensor'][start]),int(targets['sensor'][start+4]))
                for row in range(start,start+6): torch.testing.assert_close(numeric[start],numeric[row])
            # First source row deliberately uses the closest sensor descriptor.
            i,j=targets['sensor'][:2].tolist()
            from difflib import SequenceMatcher
            expected=max((k for k in range(3) if k!=i),key=lambda k:SequenceMatcher(None,names[i],names[k]).ratio())
            self.assertEqual(j,expected)

    def test_refined_binding_loss_penalizes_wrong_sensor_scope_and_collapsed_attention(self):
        from pecvd.gevis_backbone import refined_cross_grounding_loss,cross_attention_batch_metrics
        _,_,_,_,_,targets=real_step1_cross_batch(torch.randn(1,15,40),'phase',['a','b','c'],seed=7,binding_refinement=True)
        good=torch.full((18,15),1e-6)
        for row,(sensor,scope) in enumerate(zip(targets['sensor'],targets['scope'])):
            start=int(sensor)*5
            indices=(range(5) if scope==0 else range(2) if scope==1 else range(3,5))
            for index in indices: good[row,start+index]=1.
        good=good/good.sum(-1,keepdim=True)
        wrong_sensor=good.reshape(18,3,5).roll(1,1).reshape(18,15)
        wrong_scope=good.reshape(18,3,5).flip(-1).reshape(18,15)
        correct=refined_cross_grounding_loss(good,targets)
        self.assertLess(float(sum(correct.values())),float(sum(refined_cross_grounding_loss(wrong_sensor,targets).values())))
        self.assertLess(float(sum(correct.values())),float(sum(refined_cross_grounding_loss(wrong_scope,targets).values())))
        self.assertAlmostEqual(cross_attention_batch_metrics(good,targets)['scoped_joint_accuracy'],1.)
        self.assertAlmostEqual(cross_attention_batch_metrics(wrong_sensor,targets)['joint_sensor_scope_accuracy'],0.)
        self.assertAlmostEqual(cross_attention_batch_metrics(good,targets)['counterfactual_scope_shift_accuracy'],1.)
        logits=torch.randn(18,15,requires_grad=True)
        sum(refined_cross_grounding_loss(logits.softmax(-1),targets).values()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_actual_step2_cross_cell_trains_binding_only_on_invented_states(self):
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        import pecvd.gevis_backbone as helper
        root=Path(__file__).resolve().parents[1]
        n=json.loads((root/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in n['cells'] if c.get('id')=='07-cross-training'))
        from tests.notebook_branch import training_source
        source=training_source(source)
        tree=ast.parse(source)
        definitions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('FrozenPatchEncoder','real_numeric_state','real_cross_epoch')]
        backbone=GEVISBackbone(40,32,0.)
        heads=BackbonePretrainHeads(32)
        config=SimpleNamespace(cross_learning_rate=1e-3,cross_scheduler_patience=1,cross_min_learning_rate=1e-6,
            cross_feature_loss_weight=.15,cross_sensor_loss_weight=1.,cross_scope_loss_weight=.5,cross_joint_loss_weight=.25,cross_paraphrase_weight=.1,
            cross_sensor_margin_weight=.25,cross_scope_contrast_weight=.15,cross_log_margin=.5,cross_epochs=2,cross_patience=2)
        ns={'torch':torch,'nn':torch.nn,'np':np,'pd':pd,'copy':copy,'math':math,'defaultdict':defaultdict,
            'backbone':backbone,'pretrain_heads':heads,'DEVICE':torch.device('cpu'),'CONFIG':config,'SEED':3,
            'gevis_backbone':helper,'display':lambda _:None,'file_sha256':lambda p:'invented-manifest',
            'manifest_path':Path('invented-manifest.json'),
            'encoder_manifest':{'step_checkpoints':{'p':{'sha256':'invented'}}},
            'PROCESS_CONFIG':SimpleNamespace(artifacts=SimpleNamespace(encoder='invented')),'RUN_TAG':helper.CROSS_BINDING_VERSION,
            'TRAINING_PROFILE':'reference_v11'}
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'STEP2 synthetic definitions','exec'),ns)
        encoders={}; loaders={}
        for phase,patches in [('short',3),('long',5)]:
            encoder=ns['FrozenPatchEncoder'](3,2,d_model=8,layers=1,dropout=0.)
            encoder.requires_grad_(False).eval()
            encoders[phase]=(encoder,{'sensor_columns':['pressure_a','pressure_b','flow']})
            loaders[phase]=[(torch.randn(1,2*patches,3),torch.randn(1,2*patches,3))]
        ns.update(step1_encoders=encoders,TRAIN_REAL_LOADERS=loaders,VALIDATION_REAL_LOADERS=loaders)
        frozen={phase:copy.deepcopy(model.state_dict()) for phase,(model,_) in encoders.items()}
        numeric_before=backbone.numeric.weight.detach().clone()
        training=source[source.index('backbone.text.requires_grad_(False)'):]
        exec(compile(training,'STEP2 synthetic actual training loop','exec'),ns)
        self.assertEqual(len(ns['CROSS_HISTORY']),2)
        self.assertIn(ns['BEST_CROSS_EPOCH'],(1,2))
        self.assertEqual(ns['CROSS_ATTENTION_METRICS']['state_pairs'],4)
        self.assertFalse(torch.equal(numeric_before,backbone.numeric.weight))
        self.assertFalse(any(p.requires_grad for p in backbone.text.parameters()))
        self.assertTrue(ns['BINDING_COMPARISON'].split.eq('validation').all())
        for phase,(model,_) in encoders.items():
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,frozen[phase][key])
        # Selected state is restored, not merely reported.
        for key,value in backbone.state_dict().items():
            torch.testing.assert_close(value,ns['best_cross_state']['backbone'][key])
        # Exercise the candidate's real notebook loss path as well, not only its
        # standalone formula. Paired inputs must hold query AND answer type fixed.
        from pecvd.experiments import sensor_discrimination
        ns.update(TRAINING_PROFILE='contrastive_v12',sensor_discrimination=sensor_discrimination)
        config.discrimination_margin=.5
        config.discrimination_hard_weight=.25
        config.discrimination_pair_weight=.25
        def check_pair(module,args):
            self.assertEqual(args[1][::2],args[1][1::2])
            self.assertEqual(args[2][::2],args[2][1::2])
        handle=backbone.register_forward_pre_hook(check_pair)
        try:
            exec(compile(training,'STEP2 synthetic candidate training loop','exec'),ns)
        finally:
            handle.remove()
        self.assertIn('train_moved_sensor_loss',ns['CROSS_HISTORY'][0])
        import os
        ns['TRAINING_PROFILE']='fixed_questions_v13'
        def check_fixed(module,args):
            self.assertEqual(set(args[1]),{'Is there a change in this interval?'})
            self.assertEqual(set(args[2]),{'BOOLEAN'})
        handle=backbone.register_forward_pre_hook(check_fixed)
        try:
            with patch.dict(os.environ,{'GEVIS_DISCOVERY_TRAINING':'fixed_questions_v13'}):
                exec(compile(training,'STEP2 synthetic fixed-question training loop','exec'),ns)
        finally:
            handle.remove()
        self.assertIn('train_moved_sensor_loss',ns['CROSS_HISTORY'][0])
        ns['TRAINING_PROFILE']='center_relation_v14'
        ns['TRAIN_REAL_LOADERS']={key:value*4 for key,value in loaders.items()}
        ns['VALIDATION_REAL_LOADERS']=ns['TRAIN_REAL_LOADERS']
        with patch.dict(os.environ,{'GEVIS_DISCOVERY_TRAINING':'center_relation_v14'}):
            exec(compile(training,'STEP2 synthetic center/relation training loop','exec'),ns)
        self.assertIn('train_moved_sensor_loss',ns['CROSS_HISTORY'][0])
        for phase,(model,_) in encoders.items():
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,frozen[phase][key])

    def test_current_binding_candidate_namespaces_are_explicit(self):
        root=Path(__file__).resolve().parents[1]
        for filename in ('STEP_2_train_gevis_backbone.ipynb','STEP_3_train_gevis_decision.ipynb',
                         'STEP_3.1_run_gevis_inference.ipynb','CVD_evaluate.ipynb'):
            n=json.loads((root/'notebooks'/filename).read_text(encoding='utf-8'))
            source='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
            if filename.startswith('STEP_2'):
                self.assertIn('RUN_TAG=active_run_tag()',source)
                self.assertIn('TEXT_REUSE_SHA256',source)
            else: self.assertIn('active_run_tag()',source)
            for c in n['cells']:
                if c['cell_type']=='code': ast.parse(''.join(c['source']))

    def test_step2_reuses_exact_text_checkpoint_without_text_training(self):
        import hashlib
        import tempfile
        import pandas as pd
        import pecvd.gevis_backbone as helper
        root=Path(__file__).resolve().parents[1]
        n=json.loads((root/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in n['cells'] if c.get('id')=='04-text-training'))
        from tests.notebook_branch import training_source
        source=training_source(source)
        initial=GEVISBackbone(40,32,0.)
        heads=BackbonePretrainHeads(32)
        config=SimpleNamespace(numeric_dim=40,d_model=32,dropout=0.,text_encoder_name='byte',text_encoder_revision='invented',
                               reuse_text_checkpoint=True,compare_existing_backbone=False)
        expected={'schema':'gevis.backbone.v9','numeric_dim':40,'d_model':32,
                  'text_encoder_name':'byte','text_encoder_revision':'invented',
                  'semantic_curriculum':helper.SEMANTIC_CURRICULUM_VERSION,
                  'backbone_state':initial.state_dict(),'pretrain_heads_state':heads.state_dict(),
                  'best_text_epoch':3,'step1_bridge':{'manifest_sha256':'invented'}}
        with tempfile.TemporaryDirectory() as temporary:
            manifest=Path(temporary)/'artifacts'/'invented'/'encoder_manifest.json'
            manifest.parent.mkdir(parents=True); manifest.write_text('{}',encoding='utf-8')
            expected['step1_bridge']['manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
            path=Path(temporary)/'baseline.pt'; torch.save(expected,path)
            original_digest=hashlib.sha256(path.read_bytes()).hexdigest()
            ns={'SESSION_SIGNATURE':'STEP2_GEVIS_GENERIC_DISCOVERY_V10',
                'ARTIFACT_VERSION':'gevis_backbone_e5_v1','EXPECTED_ARTIFACT_VERSION':'gevis_backbone_e5_v1',
                'CONFIG':config,'TEXT_REUSE_PATH':path,'CURRICULUM_TAG':helper.SEMANTIC_CURRICULUM_VERSION,
                'PROCESS_CONFIG':SimpleNamespace(artifacts=SimpleNamespace(encoder='invented')),
                'ROOT':Path(temporary),'DEVICE':torch.device('cpu'),'torch':torch,'pd':pd,'hashlib':hashlib,
                'GEVISBackbone':GEVISBackbone,'BackbonePretrainHeads':BackbonePretrainHeads,
                'gevis_backbone':helper,'QUESTION_BANK':QUESTION_BANK,'SEMANTIC_QUESTION_BANK':SEMANTIC_QUESTION_BANK,
                'evaluate_text':lambda *a,**k:{'split':a[3]},'evaluate_semantics':lambda *a,**k:{'split':a[3]},
                'train_text_encoder':lambda *a,**k:self.fail('text must not be retrained'),'display':lambda _:None}
            with patch.object(helper,'semantic_diagnostics',return_value={'confusion':[],'contrast_pairs':[]}):
                exec(compile(source,'STEP2 actual checkpoint reuse','exec'),ns)
            self.assertEqual(ns['TEXT_HISTORY'],[])
            self.assertEqual(ns['TEXT_REUSE_SHA256'],original_digest)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),original_digest)
            self.assertFalse(ns['FUSION_REUSED'])
            for key,value in ns['backbone'].text.state_dict().items(): torch.testing.assert_close(value,initial.text.state_dict()[key])
            for key,value in ns['pretrain_heads'].state_dict().items(): torch.testing.assert_close(value,expected['pretrain_heads_state'][key])
            # A freshly trained STEP 1 changes identity. Keep E5/semantic heads,
            # but do not reuse the incompatible numeric/fusion checkpoint.
            manifest.write_text('{"new_step1":true}',encoding='utf-8')
            with patch.object(helper,'semantic_diagnostics',return_value={'confusion':[],'contrast_pairs':[]}):
                exec(compile(source,'STEP2 changed-manifest text-only reuse','exec'),ns)
            self.assertFalse(ns['FUSION_REUSED'])
            self.assertFalse(torch.equal(ns['backbone'].numeric.weight,initial.numeric.weight))
            for key,value in ns['backbone'].text.state_dict().items(): torch.testing.assert_close(value,initial.text.state_dict()[key])

    def test_token_descriptions_are_validated_and_ground_attention_keys(self) -> None:
        backbone = GEVISBackbone(numeric_dim=8, d_model=32, dropout=0.0)
        numeric = torch.zeros((2, 3, 8))
        descriptions = [[f"Phase new signal {row}_{token}" for token in range(3)] for row in range(2)]
        fused, attention = backbone(numeric, [descriptions[0][1], descriptions[1][2]], ["BOOLEAN"] * 2, descriptions)
        self.assertEqual(tuple(fused.shape), (2, 32))
        self.assertEqual(tuple(attention.shape), (2, 3))
        with self.assertRaises(ValueError):
            backbone(numeric, ["q", "q"], ["BOOLEAN"] * 2, [["short"], ["short"]])


if __name__ == "__main__":
    unittest.main()
