"""Batched-runtime parity on invented CPU/CUDA data, no project assets."""
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


def imports():
    global fast, kd, original, h, d, preprocessing
    from heads import control48_fast as fast, distilation as kd
    from heads.distilation import runtime as original
    from pecvd.experiments import ald_head_experiments as h, decision_training as d, ald_preprocessing as preprocessing


def synthetic_runtime(n=17, dimension=8, window=16, device='cpu', mode='positive'):
    torch.manual_seed(42)
    device = torch.device(device)
    names = [f'sensor_{i}' for i in range(n)] + ['ValveAct_context']
    meta = dict(sensor_columns=names, window_length=window, cadence_seconds=1.,
                mean=[0.] * (n + 1), std=[1.] * (n + 1))
    lower = np.full((window, n + 1), -1., np.float32)
    upper = np.ones_like(lower); lower[:, -1] = 0
    refs = dict(basis_lineages=['a', 'b'], records=[dict(key=['phase', 'recipe', 'loop', 0],
        lower=lower, upper=upper, lineages=['a', 'b'], reference_runs=2)])
    class Head(h.ExperimentHead):
        def forward(self, features, kind):
            if mode == 'mixed' and kind == 'BOOLEAN':
                score = features[:, -9] - .5  # exact above-bound fraction
                return torch.stack([-score, score], -1)
            return super().forward(features, kind)
    head = Head(8 * dimension + 10, 96).eval().requires_grad_(False).to(device)
    if mode != 'mixed':
        with torch.no_grad():
            head.answers['BOOLEAN'].weight.zero_()
            head.answers['BOOLEAN'].bias.copy_(torch.tensor([-5., 5.] if mode == 'positive' else [5., -5.], device=device))
    student = kd.GevisStudent(torch.randn(3, dimension), 4).eval().requires_grad_(False).to(device)
    runtime = SimpleNamespace(module=SimpleNamespace(DEVICE=device, FUSION_ANSWER_HEAD=head),
        student=student, encoders={'phase': (None, meta)}, references=refs, reference_bank=d._bank(refs),
        target_sensors={'phase': names[:-1]}, config=d.Config(), distilation_input_policy='domain')
    values = np.random.default_rng(42).uniform(-.5, .5, (window, n + 1)).astype(np.float32)
    values[:, -1] = 0.
    if mode == 'mixed':
        values[:, :n // 2] = 2.
    request = dict(current=values, step='phase', recipe_key='recipe', loop_key='loop', offset_samples=0,
                   lineage_id='invented-run', sensor_columns=names)
    return runtime, request


class FastRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        imports()
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_numeric_normalization_exact_values_and_original_invalid_valve_rules(self):
        runtime, request = synthetic_runtime(n=2)
        meta = runtime.encoders['phase'][1]
        meta['mean'] = [3., 2., .5]; meta['std'] = [2., 3., .5]
        for dtype in (np.float32, np.float64, np.int64):
            value = request['current'].astype(dtype)
            value[:, -1] = np.arange(len(value)) % 2
            expected = preprocessing.transform_live(pd.DataFrame(value, columns=meta['sensor_columns']), meta)
            np.testing.assert_array_equal(fast.normalize_numeric_array(value, meta), expected)
        for bad in (1. - 1e-10, 1e-50, -0., 2., np.nan):
            value = request['current'].astype(np.float64)
            value[0, -1] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                fast.normalize_numeric_array(value, meta)
        value = request['current'].astype(object)
        value[:, -1] = ' OPEN '
        expected = preprocessing.transform_live(pd.DataFrame(value, columns=meta['sensor_columns']), meta)
        np.testing.assert_array_equal(fast.normalize_numeric_array(value, meta), expected)
        with self.assertRaises(ValueError):
            fast.normalize_numeric_array(np.zeros_like(request['current'], dtype=bool), meta)

    def test_cached_feature_grids_match_original_student_all_tasks(self):
        runtime, request = synthetic_runtime(n=9)
        model = runtime.student
        record = runtime.reference_bank.get(('phase', 'recipe', 'loop', 0))
        with torch.no_grad():
            state = model.encode(fast.inputs(request['current'], record))
            ids = torch.tensor([8, 0, 4])
            tail = fast.boundary_tail(request['current'], record, ['sensor_8', 'sensor_0', 'sensor_4'],
                request['sensor_columns'], 0, 1.)
            cache = fast._WindowFeatures(model, state)
            for kinds in (('BOOLEAN',), ('CHOICE', 'ORDINAL'), ('BOOLEAN', 'CHOICE', 'ORDINAL')):
                torch.testing.assert_close(cache.features(ids, tail, kinds), model.features(state, ids, tail, kinds),
                                           rtol=1e-5, atol=1e-6)

    def _session_parity(self, device):
        for mode in ('positive', 'negative', 'mixed'):
            runtime, request = synthetic_runtime(n=17, device=device, mode=mode)
            before = original.head_digest(runtime.module.FUSION_ANSWER_HEAD)
            reference = original.Session(runtime).process_window(**request)
            for batch in (8, 16, 32):
                session = fast.FastSession(runtime, center_batch_size=batch)
                found = session.process_window(**request)
                passed, difference = fast.reply_parity(reference, found)
                self.assertTrue(passed, (device, mode, batch, difference))
                subset = dict(request, centers=['sensor_16', 'sensor_0', 'sensor_7'])
                self.assertTrue(fast.reply_parity(original.Session(runtime).process_window(**subset),
                    session.process_window(**subset))[0])
            self.assertEqual(original.head_digest(runtime.module.FUSION_ANSWER_HEAD), before)

    def test_cpu_original_five_gate_ordinal_probability_and_subset_order_parity(self):
        self._session_parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_original_five_gate_ordinal_probability_and_subset_order_parity(self):
        self._session_parity('cuda')

    def test_head_calls_tail_calls_and_no_cross_request_numeric_cache(self):
        runtime, request = synthetic_runtime(n=17)
        head = runtime.module.FUSION_ANSWER_HEAD
        with patch.object(head, 'forward', wraps=head.forward) as calls:
            original.Session(runtime).process_window(**request)
            self.assertEqual(calls.call_count, 3 * 17)
        session = fast.FastSession(runtime)
        with patch.object(head, 'forward', wraps=head.forward) as calls, \
             patch.object(fast, 'boundary_tail', wraps=fast.boundary_tail) as tail, \
             patch.object(runtime.student, 'encode', wraps=runtime.student.encode) as encoder:
            session.process_window(**request)
            self.assertEqual(calls.call_count, 3)
            self.assertEqual(tail.call_count, 1)
            changed = request['current'].copy(); changed[:, 0] += .2
            session.process_window(**dict(request, current=changed))
            self.assertEqual(encoder.call_count, 2)
            self.assertEqual(tail.call_count, 2)

    def test_quality_guards_and_original_package_code_identity_retained(self):
        runtime, request = synthetic_runtime(n=3)
        session = fast.FastSession(runtime)
        for override in (dict(lineage_id='a'), dict(recipe_key='missing'), dict(sensor_columns=['wrong'])):
            self.assertTrue(fast.reply_parity(original.Session(runtime).process_window(**dict(request, **override)),
                session.process_window(**dict(request, **override)))[0])
        with self.assertRaisesRegex(ValueError, 'Context-only'):
            session.process_window(**dict(request, centers=['ValveAct_context']))
        self.assertNotIn('control48_fast.py', original.code_identity())

    def test_identical_whole_request_benchmark_and_validation_only_batch_selection(self):
        runtime, request = synthetic_runtime(n=4)
        report = fast.benchmark_requests(runtime, [request], batches=(8, 16), repeats=2, warmup=1)
        self.assertTrue(report['parity'].passed.all())
        self.assertEqual(set(report['summary'].runtime), {'reference', 'batch8', 'batch16'})
        self.assertTrue(report['summary'].requests.eq(2).all())
        self.assertTrue(report['summary'].mean_ms.gt(0).all())
        selected = fast.choose_batch(report)
        self.assertIn(selected['center_batch_size'], (8, 16))
        self.assertFalse(selected['weights_changed'])
        bad = dict(report, summary=report['summary'].assign(split='test'))
        with self.assertRaisesRegex(ValueError, 'Validation only'):
            fast.choose_batch(bad)
        with self.assertRaisesRegex(ValueError, 'Raw requests'):
            fast.benchmark_requests(runtime, [dict(request, standardized=True)])

    def test_scored_case_f1_preserved_without_refitting_and_test_not_for_selection(self):
        runtime, request = synthetic_runtime(n=4, mode='mixed')
        case = dict(centers=['sensor_0', 'sensor_3'], targets=np.array([[1, 1, 1], [0, 0, 0]]))
        def make_request(c, *, raw, selected_centers):
            self.assertFalse(raw)
            self.assertTrue(selected_centers)
            return dict(request, centers=c['centers'], standardized=True)
        experiment = SimpleNamespace(datasets={'validation': {'cases': [case]}, 'test': {'cases': [case]}},
            test_done=False, request=make_request)
        report = fast.verify_scored_cases(experiment, runtime, progress=None)
        self.assertTrue(report['parity'].passed.all())
        table = report['metrics'].pivot(index='answer_type', columns='model', values='f1')
        np.testing.assert_array_equal(table.reference_runtime, table.batched_runtime)
        with self.assertRaisesRegex(ValueError, 'already be evaluated'):
            fast.verify_scored_cases(experiment, runtime, split='test', progress=None)

    def test_heldout_requests_reuses_existing_cases_and_no_new_teacher_encoding(self):
        experiment = SimpleNamespace(datasets={}, test_done=False)
        with self.assertRaisesRegex(ValueError, 'will not prepare'):
            fast.heldout_requests(experiment)


if __name__ == '__main__':
    unittest.main()
