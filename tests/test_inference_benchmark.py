"""Real PEALD selected-head dispatch and measured inference scope."""
import copy
import unittest
from unittest.mock import patch

from pecvd.experiments import inference_benchmark as timing, online_inference as online
from tests.test_external_test import ready


class InferenceBenchmarkTests(unittest.TestCase):
    def test_actual_forward_every_step_and_specialist_dispatch(self):
        contract, _, _, phase = ready(); runtime = contract.runtime
        other = 'second_phase'
        runtime.encoders[other] = runtime.encoders[phase]
        runtime.prepared.windows[other] = runtime.prepared.windows[phase]
        runtime.module.FUSION_ANSWER_TRAINING['trained_steps'] += [other]
        runtime.step_heads[phase] = copy.deepcopy(runtime.module.FUSION_ANSWER_HEAD)
        runtime.module.FUSION_ANSWER_TRAINING['step_head_selection'] = [dict(step=phase, selected_head='step')]
        cases = timing.peald_cases(runtime)
        real = online.InferenceSession.process_window
        seen = []
        def record(session, **kwargs):
            seen.append(kwargs['step'])
            return real(session, **kwargs)
        with patch.object(online.InferenceSession, 'process_window', new=record):
            report = timing.benchmark(runtime, cases, repeats=1, warmup=1)
        self.assertEqual(len(seen), 4)
        self.assertEqual(set(report.details.step), {phase, other})
        self.assertTrue(report.details.answers.eq(9).all())
        self.assertTrue(report.details.selected_sensors.eq(3).all())
        self.assertGreater(report.summary.mean_ms_per_window.iloc[0], 0)
        self.assertIn('p99_ms_per_window', report.diagnostics['by_step'])
        self.assertIn('excludes disk/preprocessing', report.metadata['timing_scope'])
        self.assertTrue(report.diagnostics['device_memory'].empty)
        del runtime.step_heads[phase]
        with self.assertRaisesRegex(RuntimeError, 'Step head missing'):
            timing.benchmark(runtime, cases, repeats=1, warmup=0)

    def test_invalid_budgets_are_rejected(self):
        with self.assertRaises(ValueError):
            timing.benchmark(None, [], repeats=1)
