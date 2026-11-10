"""Control48 regression tests: invented arrays/models only, no project assets."""
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


class SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.patch_len = 2


class SyntheticFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(3, 4)
    def _text(self, texts, device):
        return torch.ones(len(texts), 4, device=device)


class SyntheticTokenizer:
    def __call__(self, texts, **kwargs):
        return dict(input_ids=torch.tensor([[1, 2, 0]] * len(texts)),
                    attention_mask=torch.tensor([[1, 1, 0]] * len(texts)))


class SyntheticE5(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(8, 4)
        self.devices = []
    def forward(self, input_ids, attention_mask):
        self.devices.append((self.embedding.weight.device, input_ids.device))
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def fixture():
    torch.manual_seed(5)
    names = ['pressure', 'neighbor', 'ValveAct_test']
    meta = dict(sensor_columns=names, mean=[10., 20., 0.], std=[2., 3., 1.],
        window_length=8, cadence_seconds=1., context_sensor_columns=['ValveAct_test'])
    lower = np.tile(np.array([-1., -2., 0.], np.float32), (8, 1))
    upper = np.tile(np.array([1., 2., 1.], np.float32), (8, 1))
    refs = dict(basis_lineages=['basis-a', 'basis-b'], fitting_lineages=['fit-a'],
        records=[dict(key=['phase', 'recipe', 'loop', 0], lower=lower, upper=upper,
                      lineages=['basis-a', 'basis-b'], reference_runs=2)])
    enc = SyntheticEncoder()
    model = SyntheticFusion()
    head = h.ExperimentHead(42, 12)
    head.requires_grad_(False).eval()
    module = SimpleNamespace(GEVIS_MODEL=model, FUSION_ANSWER_HEAD=head, DEVICE=torch.device('cpu'),
        CURRENT_SOURCE_RECIPE_SHA256='invented-source')
    prepared = SimpleNamespace(metadata={'phase': deepcopy(meta)}, identity={},
        split_runs={'train': ['fit-a'], 'validation': ['v-a'], 'test': ['t-a']})
    runtime = SimpleNamespace(module=module, encoders={'phase': (enc, meta)}, prepared=prepared,
        references=refs, reference_bank=d._bank(refs), domain='PEALD', config=d.Config(),
        target_sensors={'phase': ['pressure', 'neighbor']}, source_steps={'phase': 'phase'},
        source_runtime=SimpleNamespace(module=module, encoders={'phase': (enc, meta)}),
        distilation_input_policy='domain')
    return runtime


def cases(split):
    result = []
    for i in range(24):
        positive = i >= 4
        family = 1 + (i - 4) // 4 if positive else 0
        level = 1 + (i - 4) % 4 if positive else 0
        values = np.zeros((8, 3), np.float32)
        if positive:
            values[:, 0] = 1. + level / 2
        result.append(dict(current=values, step='phase', split=split, run_id=split + str(i),
            lineage_id=split + str(i), pair_id=split + str(i), sensor_columns=['pressure', 'neighbor', 'ValveAct_test'],
            centers=['pressure'], targets=np.array([[int(positive), family, level]], np.int64),
            kind='level_shift' if positive else 'equal', injected=positive,
            recipe_key='recipe', loop_key='loop', offset_samples=0, reference_key=('phase', 'recipe', 'loop', 0)))
    return result


def fake_window(runtime, values, record, step, centers, offset, batch_size, *, answer_types=None):
    kinds = fa.KINDS if answer_types is None else answer_types
    tail = student_module.boundary_tail(values, record, centers,
        runtime.encoders[step][1]['sensor_columns'], offset, 1.)
    out = torch.zeros(len(centers), len(kinds), 42)
    for i, center in enumerate(centers):
        sid = runtime.encoders[step][1]['sensor_columns'].index(center)
        for j, kind in enumerate(kinds):
            out[i, j, :32] = float(values[:, sid].mean()) * .1 + fa.KINDS.index(kind) * .05
            out[i, j, -10:] = tail[i]
    return out, [(c, k) for c in centers for k in kinds]


def fake_features(runtime, case, bank):
    return fake_window(runtime, d.case_current(case), bank[tuple(case['reference_key'])],
        case['step'], case['centers'], case['offset_samples'], 8)[0]


class DistilationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Lazy imports keep portable-core collection free of legacy imports.
        global kd, data_module, runtime_module, student_module, mapping, objective, search, h, d, fa
        from heads import distilation as kd
        from heads.distilation import data as data_module, runtime as runtime_module, student as student_module
        from heads.distilation import mapping, objective, search
        from pecvd.experiments import ald_head_experiments as h, decision_training as d, fusion_answers as fa
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_student_and_exact_boundary_tail_match_reference_source_ast(self):
        ref = ast.parse((ROOT / 'notebooks/distilation/references/component_distillation.py.txt').read_text(encoding='utf-8'))
        actual = ast.parse((ROOT / 'src/heads/distilation/student.py').read_text(encoding='utf-8'))
        for name in ('GevisStudent', 'boundary_tail'):
            find = lambda tree: next(n for n in tree.body if getattr(n, 'name', None) == name)
            self.assertEqual(ast.dump(find(ref), include_attributes=False), ast.dump(find(actual), include_attributes=False))

    def test_input_channels_signed_scaled_asinh_order_and_no_second_normalization(self):
        values = np.array([[0., 3.], [2., -1.]], np.float32)
        lo = np.array([[1., 1.], [1., -1.]], np.float32)
        hi = np.array([[3., 2.], [1., 1.]], np.float32)
        x = student_module.inputs(values, dict(lower=lo, upper=hi))
        restored = torch.sinh(x).numpy().transpose(2, 1, 0)
        scale = np.maximum(hi - lo, 1.)
        np.testing.assert_allclose(restored[:, 0], values, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(restored[:, 1], lo, rtol=1e-6)
        np.testing.assert_allclose(restored[:, 2], hi, rtol=1e-6)
        np.testing.assert_allclose(restored[:, 3], (values - lo) / scale, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(restored[:, 4], (values - hi) / scale, rtol=1e-6, atol=1e-7)
        self.assertLess(x[0, 3, 0], 0)
        with self.assertRaises(ValueError):
            student_module.inputs(values, dict(lower=hi, upper=lo))

    def test_student_dimensions_cross_attention_gradient_and_kind_subset_parity(self):
        queries = torch.randn(3, 4)
        model = kd.GevisStudent(queries, 2)
        self.assertEqual(model.width, 48)
        self.assertEqual(len(model.temporal.layers), 2)
        x = torch.randn(3, 5, 8)
        centers = torch.tensor([0, 2]); tail = torch.randn(2, 10)
        features = model(x, centers, tail)
        self.assertEqual(tuple(features.shape), (2, 3, 42))
        torch.testing.assert_close(features[..., -10:], tail[:, None].expand(-1, 3, -1))
        model.eval()
        state = model.encode(x)
        torch.testing.assert_close(model.features(state, centers, tail, ('BOOLEAN',))[:, 0],
            model.features(state, centers, tail)[:, 0], rtol=1e-5, atol=1e-6)
        features[..., :-10].square().mean().backward()
        self.assertIsNotNone(model.cross.in_proj_weight.grad)
        self.assertFalse(model.question_vectors.requires_grad)
        torch.testing.assert_close(model.question_vectors, queries)

    def test_legacy_loss_excludes_tail_and_detaches_teacher(self):
        config = kd.Config()
        target_features = torch.zeros(2, 3, 42, requires_grad=True)
        features = torch.zeros(2, 3, 42, requires_grad=True)
        with torch.no_grad():
            features[..., -10:] = 1000
        labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
        student = {k: torch.zeros(2, size, requires_grad=True) for k, size in zip(fa.KINDS, (2, 6, 5))}
        teacher = {k: v.detach().clone().requires_grad_(True) for k, v in student.items()}
        value, parts = objective.legacy_loss(student, teacher, labels, features, target_features, torch.ones(42), config)
        self.assertEqual(parts['feature'], 0.)
        self.assertAlmostEqual(parts['kl'], 0., places=6)
        expected_ce = np.log(2) + np.log(5) + np.log(4)
        self.assertAlmostEqual(float(value.detach()), .5 * expected_ce, places=5)
        value.backward()
        self.assertIsNone(target_features.grad)
        self.assertTrue(all(t.grad is None for t in teacher.values()))

    def test_fixed_argmax_mapping_preserves_gate_ordinal_unknown_and_original_reply(self):
        array = np.array([[1, i, 3] for i in range(1, 6)] + [[0, 0, 0], [1, -1, -1]])
        mapped = mapping.remap(array)
        np.testing.assert_array_equal(mapped[:, 1], [1, 2, 1, 3, 3, 0, -1])
        np.testing.assert_array_equal(mapped[:, [0, 2]], array[:, [0, 2]])
        answer = dict(answer_type='CHOICE', prediction=5, answer='Delay', evaluated=True,
            probabilities={'local': .4, 'delay': .3, 'level': .3}, detected_types=['Delay'])
        reply = dict(answers=[answer, dict(answer_type='BOOLEAN', prediction=1, evaluated=True)])
        result = mapping.remap_reply(reply)
        self.assertEqual(result['answers'][0]['prediction'], 3)
        self.assertEqual(result['answers'][0]['answer'], '국소 변화·지연')
        self.assertIsNone(result['answers'][0]['probabilities'])
        self.assertEqual(result['answers'][0]['original_five_prediction'], 5)
        self.assertEqual(reply['answers'][0]['prediction'], 5)
        self.assertEqual(result['answers'][1], reply['answers'][1])

    def test_metrics_keep_gate_misses_false_positives_and_unknown_type_mask(self):
        truth = np.array([[1, 5, 1], [1, 4, 2], [1, -1, -1], [0, 0, 0]])
        prediction = np.array([[1, 4, 1], [0, 0, 0], [1, 1, 3], [1, 2, 2]])
        report, support = mapping.metrics(truth, prediction, model='synthetic', split='validation')
        self.assertLess(float(report.loc[report.answer_type.eq('CHOICE'), 'f1'].iloc[0]), 1.)
        self.assertEqual(int(report.loc[report.answer_type.eq('CHOICE'), 'known_targets'].iloc[0]), 3)
        self.assertEqual(support['source_choice_support'][5], 1)

    def test_configuration_and_teacher_mode_reject_head_or_architecture_changes(self):
        with self.assertRaises(ValueError):
            replace(kd.Config(), encoder_width=32).validate()
        with self.assertRaises(ValueError):
            replace(kd.Config(), encoder_layers=1).validate()
        with self.assertRaises(ValueError):
            replace(kd.Config(), alpha=2).validate()
        with self.assertRaisesRegex(ValueError, 'Original-five'):
            runtime_module.validate_original_head(h.ExperimentHead(42, 12, h.ChoiceConfig(mode='grouped')))

    def test_legacy_epoch_selection_is_original_five_before_v7_regrouping(self):
        truth = np.array([[0, 0, 0], [1, 1, 1], [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 2]])
        wrong_five = truth.copy()
        wrong_five[1, 1] = 3
        wrong_five[3, 1] = 5
        np.testing.assert_array_equal(mapping.remap(truth), mapping.remap(wrong_five))
        _, perfect_key = objective.legacy_validation_selection(truth, truth, 0.)
        _, wrong_key = objective.legacy_validation_selection(truth, wrong_five, 0.)
        self.assertGreater(perfect_key, wrong_key)

    def _real_runtime_text_device(self, device):
        from pecvd.gevis_runtime import GEVIS
        from pecvd.gevis_backbone import E5TextEncoder
        teacher = fixture()
        model = GEVIS(numeric_dim=8, d_model=4, dropout=0., text_encoder_name='byte')
        e5 = SyntheticE5()
        model.text = E5TextEncoder(4, tokenizer=SyntheticTokenizer(), model=e5)
        teacher.module.GEVIS_MODEL = model.cpu()
        teacher.module.DEVICE = device
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Reproduce the original CPU-embedding/CUDA-ID crash without assets or
        # downloads, using the real GEVIS._text/frozen_text/E5 forward methods.
        if device.type == 'cuda':
            with self.assertRaises(RuntimeError):
                model._text([fa.QUESTIONS[k] for k in fa.KINDS], device)
            model._text_cache = None
            e5.devices.clear()
        config = replace(kd.Config(), epochs=1, patience=1, max_seconds=30.)
        def population(runtime, split, cfg):
            rows = cases(split)
            return rows, dict(generation=[], support=[], exclusions=[], selected_cases=len(rows),
                lineages=sorted({c['lineage_id'] for c in rows}))
        with tempfile.TemporaryDirectory() as root, \
             patch.object(data_module, 'cases_for', side_effect=population), \
             patch.object(d, '_features', side_effect=fake_features):
            study = kd.Study(root, teacher, config, progress=None)
            self.assertEqual(e5.devices, [(device, device)])
            self.assertTrue(all(p.device == device for p in model.parameters()))
            self.assertTrue(all(p.device == device for p in study.head.parameters()))
            self.assertTrue(all(p.device == device for encoder, _ in teacher.encoders.values() for p in encoder.parameters()))
            ids = torch.tensor([0, 1, 2], device=device)
            with torch.no_grad():
                expected = model._text([fa.QUESTIONS[k] for k in fa.KINDS], device) + model.type_emb(ids)
            torch.testing.assert_close(study.queries, expected.cpu())
            # Warm cached text still returns on the requested device.
            self.assertEqual(len(e5.devices), 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), before[key], rtol=0, atol=0)
            history = study.fit()
            self.assertEqual(len(history), 1)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in study.head.parameters()))

    def test_real_runtime_e5_cold_cache_cpu_device_and_student_fit(self):
        self._real_runtime_text_device(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable for real embedding-device regression')
    def test_real_runtime_e5_cold_cache_cuda_device_and_student_fit(self):
        self._real_runtime_text_device(torch.device('cuda', torch.cuda.current_device()))

    def test_input_contract_rejects_changed_normalization_split_and_sensor_order(self):
        teacher = fixture()
        prepared = deepcopy(teacher.prepared)
        prepared.windows = {'phase': {s: [] for s in ('train', 'validation', 'test')}}
        data_module.attach_prepared(teacher, prepared)
        for field in ('mean', 'sensor_columns'):
            changed = deepcopy(prepared)
            changed.metadata['phase'][field] = [99, 0, 0] if field == 'mean' else ['neighbor', 'pressure', 'ValveAct_test']
            with self.assertRaises(ValueError):
                data_module.attach_prepared(teacher, changed)
        changed = deepcopy(prepared); changed.split_runs['test'] = ['new-test']
        with self.assertRaisesRegex(ValueError, 'partitions differ'):
            data_module.attach_prepared(teacher, changed)

    def test_real_student_fit_freezes_head_query_and_seals_test_then_roundtrip_and_timing(self):
        teacher = fixture()
        before = {k: v.clone() for k, v in teacher.module.FUSION_ANSWER_HEAD.state_dict().items()}
        config = replace(kd.Config(), epochs=2, patience=2, batch_cases=8,
            max_seconds=60., benchmark_windows=2, warmup=1, repeats=2)
        touched = []
        def population(runtime, split, cfg):
            touched.append(split)
            rows = cases(split)
            return rows, dict(generation=[], support=[], exclusions=[], selected_cases=len(rows),
                             lineages=sorted({c['lineage_id'] for c in rows}))
        with tempfile.TemporaryDirectory() as root, \
             patch.object(data_module, 'cases_for', side_effect=population), \
             patch.object(d, '_features', side_effect=fake_features), \
             patch.object(d, '_window_features', side_effect=fake_window):
            study = kd.Study(root, teacher, config, progress=None)
            with self.assertRaisesRegex(ValueError, 'Freeze'):
                study.dataset('test')
            history = study.fit()
            self.assertGreater(len(history), 0)
            self.assertEqual(set(touched), {'train', 'validation'})
            self.assertTrue(study.parity_report().prediction_parity.all())
            validation = study.report('validation')
            timing = study.benchmark('validation')
            self.assertEqual(set(validation['metrics'].answer_type), set(fa.KINDS))
            self.assertEqual(set(timing['summary'].model), {'teacher', 'control48'})
            self.assertTrue(timing['summary'].p95_ms.gt(0).all())
            study.freeze_selection()
            saved = study.save()
            with patch('pecvd.gevis.load_gevis', side_effect=AssertionError('merged load must not load Teacher')):
                merged = kd.load_merged(saved)
            self.assertFalse(hasattr(merged.module, 'GEVIS_MODEL'))
            self.assertTrue(study.merged_parity_report(merged).merged_prediction_parity.all())
            merged_timing = study.benchmark('validation', merged=merged)['summary']
            self.assertTrue(merged_timing.loc[merged_timing.model.eq('control48'), 'merged_package_loaded'].all())
            test = study.evaluate_test()
            self.assertTrue(test['metrics'].split.eq('test').all())
            self.assertEqual(touched.count('test'), 1)
            with self.assertRaisesRegex(ValueError, 'closed'):
                study.fit()
            with self.assertRaises(ValueError):
                study.evaluate_test()
            for key, previous in before.items():
                torch.testing.assert_close(teacher.module.FUSION_ANSWER_HEAD.state_dict()[key], previous, rtol=0, atol=0)
                torch.testing.assert_close(merged.module.FUSION_ANSWER_HEAD.state_dict()[key], previous, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in study.head.parameters()))
            payload = torch.load(saved, weights_only=False)
            payload['state']['head.feature_mean'][0] += 1
            damaged = Path(root) / 'synthetic-tampered.pt'; torch.save(payload, damaged)
            with self.assertRaisesRegex(ValueError, 'content differs'):
                kd.load_merged(damaged)

    def test_student_session_raw_standardized_gate_context_and_support_guards(self):
        teacher = fixture()
        model = kd.GevisStudent(torch.randn(3, 4), 2).eval()
        with torch.no_grad():
            teacher.module.FUSION_ANSWER_HEAD.answers['BOOLEAN'].weight.zero_()
            teacher.module.FUSION_ANSWER_HEAD.answers['BOOLEAN'].bias.copy_(torch.tensor([4., -4.]))
        session = kd.Session(teacher, model)
        values = np.zeros((8, 3), np.float32)
        request = dict(current=values, step='phase', recipe_key='recipe', loop_key='loop',
            offset_samples=0, lineage_id='new-run', sensor_columns=['pressure', 'neighbor', 'ValveAct_test'])
        with patch.object(model, 'encode', wraps=model.encode) as encode:
            reply = session.process_window(**request, standardized=True)
            self.assertEqual(encode.call_count, 1)
        self.assertEqual(reply['status'], 'OK')
        self.assertTrue(all(not a['evaluated'] for a in reply['answers'] if a['answer_type'] != 'BOOLEAN'))
        meta = teacher.encoders['phase'][1]
        raw = dict(request, current=values * np.array(meta['std']) + np.array(meta['mean']))
        self.assertEqual(reply['answers'], session.process_window(**raw)['answers'])
        self.assertEqual(session.process_window(**dict(request, lineage_id='basis-a'), standardized=True)['status'], 'NOT_EVALUATED')
        self.assertEqual(session.process_window(**dict(request, recipe_key='missing'), standardized=True)['status'], 'NOT_EVALUATED')
        with self.assertRaisesRegex(ValueError, 'Context-only'):
            session.process_window(**request, centers=['ValveAct_test'], standardized=True)

    def test_optuna_selection_is_validation_quality_then_actual_p95(self):
        records = [dict(trial=0, min_f1=.97, p95_ms=8., coverage_complete=True),
                   dict(trial=1, min_f1=.98, p95_ms=6., coverage_complete=True),
                   dict(trial=2, min_f1=.95, p95_ms=1., coverage_complete=True)]
        self.assertEqual(search.choose_trial(records, .96), (1, True))
        self.assertEqual(search.choose_trial(records, .99), (1, False))

    def test_verified_live_control48_adoption_requires_original_head_and_queries(self):
        teacher = fixture()
        with tempfile.TemporaryDirectory() as root:
            study = kd.Study(root, teacher, progress=None)
            model = kd.GevisStudent(study.queries, study.patch_length)
            copied = study.adopt_student(model, dict(objective='legacy', selected_epoch=2), study.head)
            self.assertTrue(study.training['adopted_live_student'])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(copied.state_dict()[key], value, rtol=0, atol=0)
            wrong = deepcopy(study.head)
            with torch.no_grad():
                wrong.feature_mean[0] = 99
            with self.assertRaisesRegex(ValueError, 'Head differs'):
                study.adopt_student(model, dict(objective='legacy'), wrong)

    def test_real_optuna_optional_two_trial_synthetic_integration(self):
        import importlib.util
        if importlib.util.find_spec('optuna') is None:
            self.skipTest('optional optuna not installed; run with isolated test dependency path')
        teacher = fixture()
        config = replace(kd.Config(), epochs=1, patience=1, batch_cases=8, max_seconds=30.,
                         benchmark_windows=1, warmup=1, repeats=1)
        touched = []
        def population(runtime, split, cfg):
            touched.append(split)
            rows = cases(split)
            return rows, dict(generation=[], support=[], exclusions=[], selected_cases=len(rows),
                lineages=sorted({c['lineage_id'] for c in rows}))
        with tempfile.TemporaryDirectory() as root, \
             patch.object(data_module, 'cases_for', side_effect=population), \
             patch.object(d, '_features', side_effect=fake_features), \
             patch.object(d, '_window_features', side_effect=fake_window):
            study = kd.Study(root, teacher, config, progress=None)
            study.fit()
            report = search.optimize(study, n_trials=2)
            self.assertEqual(len(report['trials']), 2)
            self.assertEqual(set(touched), {'train', 'validation'})
            self.assertFalse(study.test_opened)
            self.assertIn(report['selected_trial'], (0, 1))
            self.assertEqual(study.student.width, 48)
            self.assertEqual(len(study.student.temporal.layers), 2)
            self.assertEqual(study.training['objective'], 'legacy')
            study.freeze_selection()
            with self.assertRaisesRegex(ValueError, 'closed'):
                search.optimize(study, n_trials=1)

    def test_notebook_sources_compile_without_reading_outputs(self):
        books = list((ROOT / 'notebooks/distilation').glob('*.ipynb'))
        self.assertGreaterEqual(len(books), 1)
        for path in books:
            book = json.loads(path.read_text(encoding='utf-8'))
            for index, cell in enumerate(book['cells']):
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    compile(source, f'{path.name}:{index}', 'exec')
                    self.assertNotIn('matplotlib', source)


if __name__ == '__main__':
    unittest.main()
