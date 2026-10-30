from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import sys
import types
import unittest

import torch

from pecvd.gevis_backbone import ANSWER_TYPES
from pecvd.gevis_fast import (configure_fast, fit_heads, frozen_text, head_loss,
    pack_boolean_records, configure_boolean_fusion, boolean_fusion_logits,
    BooleanNumericProbe, evaluate_boolean, fit_boolean)
from pecvd.gevis_fast import (typed_targets,structured_answer_probabilities,pack_typed_records,typed_batch,TypedFusionScorer,
    fit_typed_comparison,predict_typed,typed_metrics,choose_boolean_threshold,choose_temperature)

ROOT = Path(__file__).resolve().parents[1]


def model_class():
    nb = json.loads((ROOT / 'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
    source = ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == '02-config'))
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {'ByteTextEncoder', 'GEVIS'}]
    ns = {'torch': torch, 'nn': torch.nn, 'math': math, 'ANSWER_TYPES': ANSWER_TYPES, 'frozen_text': frozen_text}
    exec(compile(ast.Module(body=classes, type_ignores=[]), 'synthetic-fast-model', 'exec'), ns)
    return ns['GEVIS']


class FastGevisTests(unittest.TestCase):
    def test_integrated_wrapper_places_new_layers_on_base_device(self):
        # Meta exercises non-CPU placement even on CPU-only test machines.
        base=model_class()(numeric_dim=264,d_model=32,dropout=0.).to('meta')
        model=TypedFusionScorer(base)
        self.assertEqual({p.device.type for p in model.parameters()},{'meta'})
        self.assertEqual(model.operation.weight.device,base.numeric.weight.device)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable')
    def test_integrated_cuda_prediction_and_paired_training(self):
        base=model_class()(numeric_dim=264,d_model=32,dropout=0.).to('cuda')
        features=torch.tensor([[0.,0.,0.,0.,0.],[2.,0.,0.,0.,0.]]*2)
        stat=torch.zeros(4,dtype=torch.long); direction=torch.full((4,),2,dtype=torch.long)
        labels,rewards,scores=typed_targets(features,stat,direction)
        cache=pack_typed_records([{'states':torch.randn(4,3,264),
            'bank_index':torch.arange(4),'features':scores,'statistic':stat,
            'direction':direction,'scope':torch.zeros(4,dtype=torch.long),
            'step_id':torch.zeros(4,dtype=torch.long),'labels':labels,'rewards':rewards}])
        models,history,report=fit_typed_comparison(base,cache,cache,epochs=1,
            steps_per_epoch=1,batch_size=4,max_seconds=60.,progress=lambda _:None)
        self.assertEqual(report['paired_updates'],1)
        self.assertEqual(len(history),3)
        for model in models.values():
            self.assertEqual({p.device.type for p in model.parameters()},{'cuda'})
            probabilities,_=predict_typed(model,cache)
            self.assertEqual({k:p.shape[1] for k,p in probabilities.items()},
                             {'BOOLEAN':2,'CHOICE':6,'ORDINAL':5})
            self.assertTrue(all(torch.isfinite(p).all() for p in probabilities.values()))

    def test_integrated_targets_and_reward_costs_cover_three_types(self):
        features=torch.tensor([[0.,0.,0.,0.,0.],[2.5,0.,0.,0.,0.],[0.,-3.5,0.,0.,0.]])
        labels,rewards,_=typed_targets(features,torch.tensor([0,0,1]),torch.tensor([2,0,1]),false_alarm_cost=3.,miss_cost=2.)
        self.assertEqual(labels['BOOLEAN'].tolist(),[0,1,1])
        self.assertEqual(labels['CHOICE'].tolist(),[0,1,2])
        self.assertEqual(labels['ORDINAL'].tolist(),[0,2,3])
        self.assertEqual(float(rewards['BOOLEAN'][0,1]),-3.)
        self.assertEqual(float(rewards['BOOLEAN'][1,0]),-2.)

    def test_integrated_paired_training_calibration_and_three_type_shapes(self):
        torch.manual_seed(3)
        base=model_class()(numeric_dim=264,d_model=32,dropout=0.)
        x=torch.randn(8,3,264); features=torch.tensor([[0.,0.,0.,0.,0.],[2.,0.,0.,0.,0.]]*4)
        stat=torch.zeros(8,dtype=torch.long); direction=torch.full((8,),2,dtype=torch.long)
        labels,rewards,scores=typed_targets(features,stat,direction)
        record={'states':x,'bank_index':torch.arange(8),'features':scores,'statistic':stat,'direction':direction,'scope':torch.zeros(8,dtype=torch.long),'step_id':torch.zeros(8,dtype=torch.long),'labels':labels,'rewards':rewards}
        cache=pack_typed_records([record])
        models,history,report=fit_typed_comparison(base,cache,cache,epochs=1,steps_per_epoch=1,batch_size=4,max_seconds=20.,reward_weight=0.,progress=lambda _:None)
        self.assertEqual(report['paired_updates'],1)
        self.assertEqual({row['method'] for row in history},{'GEVIS_CE','GEVIS_CE_LJR','numeric_probe'})
        # Zero LJR weight is a control: matched initial weights/batches produce matched parameters.
        for key,value in models['GEVIS_CE'].state_dict().items():
            torch.testing.assert_close(value,models['GEVIS_CE_LJR'].state_dict()[key])
        probs,_=predict_typed(models['GEVIS_CE'],cache)
        self.assertEqual({k:p.shape[1] for k,p in probs.items()},{'BOOLEAN':2,'CHOICE':6,'ORDINAL':5})
        temp=choose_temperature(probs['BOOLEAN'],labels['BOOLEAN'])
        threshold,status=choose_boolean_threshold(probs['BOOLEAN'],labels['BOOLEAN'],minimum_recall=.5)
        self.assertGreater(temp,0); self.assertTrue(0<threshold<1)
        self.assertEqual(status['status'],'VALIDATION_ONLY')
        self.assertIn('ordinal_mae',typed_metrics(probs['ORDINAL'],labels['ORDINAL'],rewards['ORDINAL'],'ORDINAL'))

    def test_step3_reload_recovers_missing_function_in_stale_module(self):
        import importlib
        import pecvd.gevis_fast as helper
        nb=json.loads((ROOT/'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in nb['cells'] if c.get('id')=='02-config'))
        lines=source.splitlines()
        start=lines.index('import pecvd.gevis_fast as gevis_fast')
        end=lines.index('from pecvd.gevis_fast import frozen_text, configure_fast, fit_heads')+1
        bootstrap='\n'.join(lines[start:end])
        saved=helper.pack_boolean_records
        try:
            del helper.pack_boolean_records
            namespace={'Path':Path,'ROOT':ROOT,'importlib':importlib}
            exec(compile(bootstrap,'Step3:reload-regression','exec'),namespace)
            self.assertTrue(callable(namespace['gevis_fast'].pack_boolean_records))
            self.assertIs(namespace['frozen_text'],namespace['gevis_fast'].frozen_text)
        finally:
            if not hasattr(helper,'pack_boolean_records'): helper.pack_boolean_records=saved
    def test_diagnostic_masks_ragged_patches_and_unfreezes_numeric_fusion(self):
        model = model_class()(numeric_dim=264,d_model=32,dropout=0.)
        configure_boolean_fusion(model)
        a=torch.randn(2,3,264); b=torch.randn(2,5,264)
        cache=pack_boolean_records([{'states':a,'labels':torch.tensor([0,1])},{'states':b,'labels':torch.tensor([0,1])}])
        self.assertEqual(cache['states'].shape,(4,5,264))
        self.assertEqual(cache['lengths'].tolist(),[3,3,5,5])
        logits=boolean_fusion_logits(model,cache['states'],cache['lengths'])
        with torch.no_grad():
            direct=boolean_fusion_logits(model,a,torch.tensor([3,3]))
        torch.testing.assert_close(logits[:2],direct,atol=2e-6,rtol=2e-5)
        torch.nn.functional.cross_entropy(logits,cache['labels']).backward()
        self.assertIsNotNone(model.numeric.weight.grad)
        self.assertIsNotNone(model.cross.in_proj_weight.grad)
        self.assertIsNone(model.text.token.weight.grad)

    def test_balanced_diagnostic_learns_synthetic_signal_without_balancing_evaluation(self):
        torch.manual_seed(4)
        values=torch.cat([torch.linspace(-.3,.3,56),torch.tensor([-2.,-1.5,1.5,2.]*2)])
        states=torch.zeros(64,3,264); states[:,:,256]=values[:,None]
        labels=(values.abs()>1.).long()
        cache=pack_boolean_records([{'states':states,'labels':labels}])
        probe=BooleanNumericProbe()
        fn=lambda model,x,lengths:model(x,lengths)
        before=cache['labels'].clone()
        history,report=fit_boolean(probe,cache,cache,fn,epochs=25,batch_size=32,learning_rate=.03,
                                   max_seconds=20.,patience=25,progress=lambda _:None)
        metrics,_=evaluate_boolean(probe,cache,fn)
        self.assertGreater(metrics['change_recall'],0.)
        self.assertGreater(metrics['balanced_accuracy'],.5)
        self.assertEqual(report['LJR_weight'],0.)
        torch.testing.assert_close(cache['labels'],before)
        self.assertEqual(metrics['change_cases'],8)
        self.assertEqual(metrics['no_change_cases'],56)
        self.assertTrue(history)

    def test_diagnostic_rejects_absent_train_positives_and_reports_missing_test_recall(self):
        probe=BooleanNumericProbe(); fn=lambda model,x,lengths:model(x,lengths)
        cache=pack_boolean_records([{'states':torch.zeros(4,3,264),'labels':torch.zeros(4,dtype=torch.long)}])
        with self.assertRaisesRegex(ValueError,'both classes'):
            fit_boolean(probe,cache,cache,fn)
        metrics,_=evaluate_boolean(probe,cache,fn)
        self.assertTrue(math.isnan(metrics['change_recall']))
        self.assertTrue(math.isnan(metrics['balanced_accuracy']))
        with self.assertRaises(MemoryError):
            pack_boolean_records([{'states':torch.zeros(4,3,264),'labels':torch.zeros(4)}],max_bytes=1)

    def baseline_helpers(self):
        import numpy as np
        from collections import Counter, defaultdict
        nb = json.loads((ROOT / 'tests/fixtures/step4_legacy_validation.json').read_text(encoding='utf-8'))
        nodes = []
        for cell in nb['cells']:
            if cell.get('id') in {'v4-eval', 'v4-data'}:
                nodes.extend(node for node in ast.parse(''.join(cell['source'])).body if isinstance(node, ast.FunctionDef) and node.name in {'selection_metrics', 'boolean_metrics', 'fit_constant_baselines', 'matched_candidates'})
        namespace = {'np': np, 'Counter': Counter, 'defaultdict': defaultdict}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'synthetic-baseline-comparison', 'exec'), namespace)
        return namespace

    def test_high_accuracy_constant_can_miss_every_positive(self):
        import numpy as np
        ns = self.baseline_helpers()
        target = np.array([0] * 98 + [1] * 2)
        rewards = np.eye(2)[target]
        constant = np.zeros(100, dtype=int)
        metrics = ns['selection_metrics'](rewards, constant, 2)
        binary = ns['boolean_metrics'](rewards, constant)
        self.assertEqual(metrics['reward_argmax_accuracy'], .98)
        self.assertEqual(metrics['balanced_accuracy'], .5)
        self.assertEqual(binary['change_recall'], 0.)
        self.assertEqual(binary['false_negative'], 2)
        detector = np.array([0] * 90 + [1] * 10)
        better = ns['selection_metrics'](rewards, detector, 2)
        self.assertLess(better['reward_argmax_accuracy'], metrics['reward_argmax_accuracy'])
        self.assertGreater(better['balanced_accuracy'], metrics['balanced_accuracy'])
        self.assertEqual(ns['boolean_metrics'](rewards, detector)['change_recall'], 1.)

    def test_no_positive_cases_report_undefined_recall(self):
        import numpy as np
        ns = self.baseline_helpers()
        rewards = np.array([[1., 0.]] * 4)
        result = ns['boolean_metrics'](rewards, np.zeros(4, dtype=int))
        self.assertEqual(result['change_cases'], 0)
        self.assertTrue(np.isnan(result['change_recall']))
        self.assertTrue(np.isnan(result['change_precision']))
        self.assertEqual(result['false_alarm_rate'], 0.)

    def test_constants_are_chosen_from_train_not_test(self):
        import numpy as np
        ns = self.baseline_helpers()
        train = np.array([[0., 1.], [0., 1.], [0., 1.], [100., -100.]])
        chosen = ns['fit_constant_baselines'](train)
        self.assertEqual(chosen, {'always_first': 0, 'train_majority': 1, 'train_best_constant_reward': 0})
        test = np.array([[1., 0.]] * 4)
        self.assertEqual(ns['selection_metrics'](test, np.full(4, chosen['train_majority']), 2)['reward_argmax_accuracy'], 0.)
        self.assertEqual(chosen['train_majority'], 1)

    def test_typed_probability_metrics_report_calibration(self):
        labels = torch.tensor([0, 1, 1, 0])
        rewards = torch.nn.functional.one_hot(labels, 2).float()
        perfect = torch.nn.functional.one_hot(labels, 2).float()
        metrics = typed_metrics(perfect, labels, rewards, 'BOOLEAN')
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertEqual(metrics['mean_true_answer_probability'], 1.)
        self.assertEqual(metrics['expected_calibration_error'], 0.)
        self.assertEqual(metrics['confidence_accuracy_gap'], 0.)

        overconfident = torch.tensor([[.99, .01], [.99, .01], [.01, .99], [.01, .99]])
        metrics = typed_metrics(overconfident, labels, rewards, 'BOOLEAN')
        self.assertEqual(metrics['accuracy'], .5)
        self.assertGreater(metrics['expected_calibration_error'], .45)
        self.assertGreater(metrics['confidence_accuracy_gap'], .45)

    def test_structured_answers_execute_requested_operations_without_training(self):
        features=torch.tensor([[2.,.2,.1,.3,.4],[-2.,.2,.1,.3,.4]])
        statistic=torch.tensor([0,0]); direction=torch.tensor([0,1])
        probabilities,scores=structured_answer_probabilities(features,statistic,direction,temperature=.25)
        self.assertEqual(set(probabilities),{'BOOLEAN','CHOICE','ORDINAL'})
        self.assertTrue((probabilities['BOOLEAN'][:,1]>.99).all())
        self.assertTrue((probabilities['CHOICE'][:,1]>.9).all())
        torch.testing.assert_close(probabilities['ORDINAL'].sum(-1),torch.ones(2))
        torch.testing.assert_close(scores,features)

    def test_structured_answer_argmax_matches_declared_target_boundaries(self):
        strengths=torch.tensor([.49,.75,.99,1.01,1.49,1.99,2.01,3.01,4.01])
        features=torch.zeros((len(strengths),5)); features[:,0]=strengths
        statistic=torch.zeros(len(strengths),dtype=torch.long)
        direction=torch.full((len(strengths),),2,dtype=torch.long)
        labels,_,_=typed_targets(features,statistic,direction)
        probabilities,_=structured_answer_probabilities(features,statistic,direction,temperature=.25)
        for kind in ('BOOLEAN','CHOICE','ORDINAL'):
            torch.testing.assert_close(probabilities[kind].argmax(-1),labels[kind])

    def test_step4_references_match_cohort_and_exclude_current_process(self):
        ns = self.baseline_helpers()
        result = ns['matched_candidates']({'process_id': ['a','b','c','d'], 'position_bin': [0,0,0,0], 'cohort_key': ['X','X','Y','Y']})
        self.assertEqual(result, {0:[1], 1:[0], 2:[3], 3:[2]})

    def test_notebook_prepare_fit_and_runtime_on_synthetic_windows(self):
        import numpy as np
        import pandas as pd
        from dataclasses import replace
        from torch.utils.data import DataLoader
        nb = json.loads((ROOT / 'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        module = types.ModuleType('_gevis_fast_synthetic_pipeline')
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)
        ns = module.__dict__
        config_source = ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == '02-config'))
        exec(compile(config_source, '02-config', 'exec'), ns)
        # The real notebook derives this from the loaded rich-SSL checkpoints.
        # This isolated synthetic fixture intentionally keeps the legacy 64-wide encoder.
        ns['ENCODER_D_MODEL'] = 64
        ns['selected_variant']=lambda:'legacy'
        functions_source = ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == '02-functions'))
        exec(compile(functions_source, '02-functions', 'exec'), ns)
        ns['DEVICE'] = torch.device('cpu')
        ns['CONFIG'] = replace(ns['CONFIG'], d_model=32, max_windows_per_step=4, max_validation_windows_per_step=4)
        encoder = ns['PatchEncoder'](2, 3, dropout=0.)
        meta = {'sensor_columns': ['pressure', 'flow'], 'sensor_descriptors': {'pressure': 'pressure', 'flow': 'flow'}, 'step_descriptor': 'new phase', 'window_length': 12, 'patch_length': 3, 'mean': np.zeros(2), 'std': np.ones(2)}
        ns['encoders'] = {'new phase': (encoder, meta)}
        model = ns['GEVIS'](numeric_dim=264, d_model=32, dropout=0.)
        configure_fast(model)
        a, b = torch.randn(4, 12, 2), torch.randn(4, 12, 2)
        loader = DataLoader(torch.utils.data.TensorDataset(a, b, torch.zeros(4, dtype=torch.long)), batch_size=2)
        cache = ns['prepare_fast_cache'](model, {'new phase': loader}, 'train')
        self.assertEqual(len(cache), 6)
        metrics = ns['evaluate_cache'](model, cache)
        self.assertEqual(metrics['CHOICE']['pairs'], 4)
        history, report = fit_heads(model, cache, ns['CRITERIA'], epochs=1, max_seconds=20., progress=lambda _: None)
        self.assertEqual(len(history), 1)
        self.assertLess(report['trainable_parameters'], 60000)
        diagnostic_current=torch.zeros(4,12,2); diagnostic_current[:2,:,0]=3.
        diagnostic_reference=torch.zeros_like(diagnostic_current)
        diagnostic_loader=DataLoader(torch.utils.data.TensorDataset(diagnostic_current,diagnostic_reference,torch.zeros(4,dtype=torch.long)),batch_size=2)
        ns.update({'gevis':model,'RUN_BOOLEAN_DIAGNOSTIC':True,'BOOLEAN_EPOCHS':1,'BOOLEAN_FIT_SECONDS':10.,'SAVE_BOOLEAN_DIAGNOSTIC':False,
                   'train_loaders':{'new phase':diagnostic_loader},'validation_loaders':{'new phase':diagnostic_loader},'test_loaders':{'new phase':diagnostic_loader},'display':lambda _:None})
        legacy=json.loads((ROOT/'tests/fixtures/step3_legacy_diagnostics.json').read_text(encoding='utf-8'))
        self.assertFalse(any(c.get('id') in ('02-boolean-diagnostic','02-integrated-experiment') for c in nb['cells']))
        ns.update(BOOLEAN_PAIR_LIMIT=32,MIN_VALIDATION_RECALL=.8,INTEGRATED_PAIR_LIMIT=16,LJR_WEIGHT=.2,
            LABEL_REWARD_POLICY={'mode':'configured_numeric_proxy','thresholds':[1.]*5,
                'ordinal_edges':[1.,2.,3.,4.],'false_alarm_cost':1.,'miss_cost':1.,'verified_fault_labels':False})
        diagnostic_source=''.join(next(c['source'] for c in legacy['cells'] if c.get('id')=='02-boolean-diagnostic'))
        legacy_functions=''.join(next(c['source'] for c in legacy['cells'] if c.get('id')=='legacy-diagnostic-functions'))
        exec(compile(legacy_functions,'legacy:diagnostic-functions','exec'),ns)
        exec(compile(diagnostic_source,'STEP3:synthetic-boolean-diagnostic','exec'),ns)
        self.assertTrue(ns['BOOLEAN_DIAGNOSTIC_AVAILABLE'])
        self.assertEqual(ns['BOOLEAN_TRAIN']['labels'].sum().item(),2)
        self.assertEqual(ns['BOOLEAN_REPORTS']['GEVIS_fusion']['LJR_weight'],0.)
        self.assertEqual(len(ns['BOOLEAN_RESULTS']),4)
        validator = json.loads((ROOT / 'tests/fixtures/step4_legacy_validation.json').read_text(encoding='utf-8'))
        ns.update({'GEVIS_MODEL': model, 'STEP_ENCODERS': ns['encoders'], 'CHECKPOINT': {'criteria': ns['CRITERIA']}, 'BATCH_SIZE': 2, 'display': lambda _: None,
                   'scenarios': lambda step, split='test': {'real_cross_run': (a.numpy(), b.numpy(), np.zeros(4, dtype=int))}})
        for cell_id in ('v4-eval', 'v4-baseline-comparison'):
            exec(compile(''.join(next(c['source'] for c in validator['cells'] if c.get('id') == cell_id)), cell_id, 'exec'), ns)
        self.assertEqual(len(ns['comparison_summary']), 12)
        self.assertEqual(len(ns['boolean_diagnostics']), 4)
        self.assertEqual(set(ns['comparison_summary']['method']), {'GEVIS_LJR','always_first','train_majority','train_best_constant_reward'})
        ns.update({'RUN_BOOLEAN_DIAGNOSTIC_VALIDATE':True,'DIAG_NS':ns,'DIAG_CHECKPOINT':{'pair_limit':32,'train_majority_class':0,'reports':ns['BOOLEAN_REPORTS']},
                   'BOOLEAN_FUSION':ns['BOOLEAN_MODELS']['GEVIS_fusion'],'NUMERIC_PROBE':ns['BOOLEAN_MODELS']['numeric_probe']})
        diagnostic_eval=''.join(next(c['source'] for c in validator['cells'] if c.get('id')=='v4-boolean-diagnostic'))
        exec(compile(diagnostic_eval,'STEP4:synthetic-boolean-diagnostic','exec'),ns)
        self.assertEqual(len(ns['diagnostic_comparison']),8)
        self.assertTrue((ns['diagnostic_comparison']['change_cases']==2).all())
        ns.update({'RUN_INTEGRATED_EXPERIMENT':True,'INTEGRATED_EPOCHS':1,'INTEGRATED_STEPS_PER_EPOCH':1,'INTEGRATED_FIT_SECONDS':20.,'SAVE_INTEGRATED_EXPERIMENT':False})
        integrated_source=''.join(next(c['source'] for c in legacy['cells'] if c.get('id')=='02-integrated-experiment'))
        exec(compile(integrated_source,'STEP3:synthetic-integrated-comparison','exec'),ns)
        self.assertEqual(set(ns['INTEGRATED_RESULTS']['answer_type']),{'BOOLEAN','CHOICE','ORDINAL'})
        self.assertEqual(ns['INTEGRATED_REPORT']['paired_updates'],1)
        self.assertEqual(len(ns['INTEGRATED_RESULTS']),36)
        ns.update({'EXPERIMENT_MODE':'integrated','INTEGRATED_NS':ns,
                   'INTEGRATED_CK':{'pair_limit':16,'policy':ns['LABEL_REWARD_POLICY'],'calibration':ns['INTEGRATED_CALIBRATION'],
                       'constants':ns['INTEGRATED_CONSTANTS'],'report':ns['INTEGRATED_REPORT'],'preparation_seconds':0.,
                       'backbone_text_metrics':[{'split':'synthetic'}],'backbone_semantic_metrics':[{'split':'synthetic'}],'backbone_grounding_metrics':[{'split':'synthetic'}]}})
        integrated_eval=''.join(next(c['source'] for c in validator['cells'] if c.get('id')=='v4-integrated-comparison'))
        exec(compile(integrated_eval,'STEP4:synthetic-integrated-comparison','exec'),ns)
        self.assertEqual(set(ns['integrated_results']['answer_type']),{'BOOLEAN','CHOICE','ORDINAL'})
        self.assertEqual(len(ns['effect']),12)
        self.assertEqual(set(ns['class_results']['answer_type']),{'BOOLEAN','CHOICE','ORDINAL'})
        self.assertEqual(set(ns['confusion_results']['answer_type']),{'BOOLEAN','CHOICE','ORDINAL'})
        # Every expected class is shown, including absent classes as unverified.
        labels=torch.tensor([0,0,1]); probs=torch.nn.functional.one_hot(torch.tensor([0,1,1]),6).float()
        classes,confusion=ns['typed_class_diagnostics'](probs,labels,'CHOICE')
        self.assertEqual(len(classes),6); self.assertEqual(len(confusion),36)
        self.assertEqual(classes[0]['recall'],.5)
        self.assertTrue(classes[5]['missing_in_split']); self.assertTrue(math.isnan(classes[5]['recall']))
        self.assertEqual(sum(row['count'] for row in confusion),3)
        runtime = json.loads((ROOT / 'notebooks/STEP_3.1_run_gevis_inference.ipynb').read_text(encoding='utf-8'))
        runtime_module = types.ModuleType('_gevis_fast_synthetic_runtime')
        sys.modules[runtime_module.__name__] = runtime_module
        self.addCleanup(sys.modules.pop, runtime_module.__name__, None)
        rns = runtime_module.__dict__
        for cell_id in ('03-types', '03-function'):
            exec(compile(''.join(next(c['source'] for c in runtime['cells'] if c.get('id') == cell_id)), cell_id, 'exec'), rns)
        loaded = rns['GEVIS'](numeric_dim=264, d_model=32, dropout=0.)
        loaded.load_state_dict(model.state_dict()); loaded.requires_grad_(False); loaded.eval()
        # Synthetic parser oracle: isolate runtime gating from untrained text quality.
        for field, target in {'statistic': 0, 'direction': 2, 'time_scope': 0}.items():
            loaded.semantic_heads[field].weight.zero_()
            loaded.semantic_heads[field].bias.zero_()
            loaded.semantic_heads[field].bias[target] = 10.
        for parameter in loaded.support_head.parameters(): parameter.data.zero_()
        loaded.support_head[-1].bias.data.fill_(10.)
        rns.update({'DEVICE': torch.device('cpu'), 'GEVIS_MODEL': loaded, 'STEP_ENCODERS': {'new phase': (encoder, meta)}, 'GEVIS_CHECKPOINT': {'schema': 'gevis.backbone.generic.v10', '_artifact_sha256': 'synthetic'}})
        current = pd.DataFrame(a[0].numpy(), columns=meta['sensor_columns']); current['row_in_process'] = np.arange(12)
        reference = pd.DataFrame(b[0].numpy(), columns=meta['sensor_columns']); reference['row_in_process'] = np.arange(12)
        question = rns['make_grounded_questions']('new phase', 'pressure')[0]
        info = rns['ReferenceInfo']('synthetic', 1)
        answer = rns['ask_gevis'](current, reference, 'new phase', [question], info, confidence_floor=0.,grounding_floor=0.)[0]
        self.assertTrue(answer['accepted'])
        self.assertGreater(answer['batch_inference_ms'], 0.)
        loaded.semantic_heads['direction'].bias.zero_()
        loaded.semantic_heads['direction'].bias[0] = 10.
        changed = rns['ask_gevis'](current, reference, 'new phase', [question], info, confidence_floor=0.,grounding_floor=0.)[0]
        self.assertEqual(changed['predicted_query']['direction'],'increase')
        self.assertTrue(changed['accepted'])
        loaded.semantic_heads['direction'].bias.zero_()
        loaded.semantic_heads['direction'].bias[2] = 10.
        loaded.support_head[-1].bias.data.fill_(-10.)
        unsupported = rns['GevisQuestion']('unknown request', 'BOOLEAN', tuple(ns['CRITERIA']['BOOLEAN']))
        answer = rns['ask_gevis'](current, reference, 'new phase', [unsupported], info, confidence_floor=0.,grounding_floor=0.)[0]
        self.assertIsNone(answer['answer'])
        self.assertIn('unsupported_question_or_criteria',answer['coverage_warnings'])

    def test_frozen_representation_decode_equals_live_and_heads_only_update(self):
        torch.manual_seed(7)
        model = model_class()(numeric_dim=8, d_model=32, dropout=0.)
        params = configure_fast(model)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        numeric = torch.randn(4, 3, 8)
        questions = ['sensor A level increase'] * 4
        descriptions = [['sensor A early', 'sensor A middle', 'sensor A late']] * 4
        criteria = [['no', 'yes']] * 4
        with torch.no_grad():
            fusion = model(numeric, questions, ['BOOLEAN'] * 4, criteria, descriptions, return_fusion=True)
            live, _ = model(numeric, questions, ['BOOLEAN'] * 4, criteria, descriptions)
        cached = model.decode(fusion, ['BOOLEAN'] * 4, criteria)
        torch.testing.assert_close(torch.stack(live), torch.stack(cached))
        optimizer = torch.optim.AdamW(params, lr=.01)
        before = model.change_head[-1].weight.detach().clone()
        loss = head_loss(torch.stack(cached), torch.tensor([[1., -1.]] * 4))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.change_head[-1].weight))
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad, name)
                torch.testing.assert_close(p, frozen[name], msg=name)
        self.assertLess(sum(p.numel() for p in params), sum(p.numel() for p in model.parameters()) / 2)

    def test_supervision_corrects_wrong_saturated_prediction_and_handles_ties(self):
        logits = torch.tensor([[30., -30.]], requires_grad=True)
        loss = head_loss(logits, torch.tensor([[-1., 1.]]))
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0]), .9)
        self.assertLess(float(logits.grad[0, 1]), -.9)
        equal = torch.zeros(1, 2, requires_grad=True)
        head_loss(equal, torch.ones(1, 2)).backward()
        torch.testing.assert_close(equal.grad, torch.zeros_like(equal))

    def test_validation_failure_rolls_back_including_initial_state(self):
        model = model_class()(numeric_dim=8, d_model=32, dropout=0.)
        configure_fast(model)
        before = copy.deepcopy(model.state_dict())
        fusion = (torch.randn(4, 32), torch.randn(4, 32), torch.softmax(torch.randn(4, 3), -1))
        cache = [{'answer_type': 'BOOLEAN', 'fusion': fusion, 'rewards': torch.tensor([[1., -1.]] * 4)}]
        history, report = fit_heads(model, cache, {'BOOLEAN': ['no', 'yes']}, epochs=3, patience=1,
                                    max_seconds=10., validation=lambda _: 1., progress=lambda _: None)
        self.assertEqual(len(history), 1)
        self.assertEqual(report['best_score'], 1.)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_time_budget_preserves_baseline_without_updates(self):
        model = model_class()(numeric_dim=8, d_model=32, dropout=0.)
        fusion = (torch.randn(2, 32), torch.randn(2, 32), torch.ones(2, 3) / 3)
        cache = [{'answer_type': 'BOOLEAN', 'fusion': fusion, 'rewards': torch.ones(2, 2)}]
        history, _ = fit_heads(model, cache, {'BOOLEAN': ['no', 'yes']}, max_seconds=1e-12,
                              validation=lambda _: 0., progress=lambda _: None)
        self.assertEqual(history, [])

    def test_question_targets_independent_of_batch_partition(self):
        nb = json.loads((ROOT / 'notebooks/STEP_3_train_gevis_decision.ipynb').read_text(encoding='utf-8'))
        source = ''.join(next(c['source'] for c in nb['cells'] if c.get('id') == '02-functions'))
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'semantic_comparison_features', 'grounded_question_contract'}]
        ns = {'torch': torch, 'math': math, 'SEMANTIC_NAMES': ('level', 'variability', 'slope', 'local', 'residual'), 'SCOPE_NAMES': ('whole', 'early', 'late'), 'DIRECTION_NAMES': ('up', 'down', 'either'), 'CONFIG': type('Config', (), {'reward_temperature': .75})()}
        import pecvd.gevis_backbone as gevis_backbone
        ns.update({'gevis_backbone': gevis_backbone, 'render_semantic_question': gevis_backbone.render_semantic_question})
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'synthetic-contract', 'exec'), ns)
        a, b = torch.randn(8, 12, 3), torch.randn(8, 12, 3)
        state, pos, ids = torch.randn(8, 9, 8), torch.zeros(8, dtype=torch.long), torch.arange(8)
        meta = {'sensor_columns': ['a', 'b', 'c']}
        contract = ns['grounded_question_contract']
        full = contract(a, b, state, pos, 'new-phase', meta, ids, 'test')
        parts = [contract(a[start:start+4], b[start:start+4], state[start:start+4], pos[start:start+4], 'new-phase', meta, ids[start:start+4], 'test') for start in (0, 4)]
        for answer_type in ANSWER_TYPES:
            torch.testing.assert_close(full['rewards'][answer_type], torch.cat([p['rewards'][answer_type] for p in parts]))
            self.assertEqual(full['questions'][answer_type], sum([p['questions'][answer_type] for p in parts], []))
