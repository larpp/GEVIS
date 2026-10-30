"""Real model parity, per-request encoding reuse and evaluator isolation."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pecvd.experiments import online_inference as online, external_test as external
from tests.test_external_test import ready


class OnlineInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.raw, cls.frame, cls.phase = ready()

    def setUp(self):
        self.contract = copy.deepcopy(self.__class__.contract)

    def test_cached_numeric_and_grouped_head_match_original_all_center_probabilities(self):
        runtime = self.contract.runtime
        cases, _ = external.windows(self.frame, self.contract, 'case')
        case = cases[0]; case['reference'] = case['current'].copy()
        from pecvd.experiments import fast_experiments as fast, fusion_answers as fa
        expected = []
        for center in case['centers']:
            features, tasks = fa.fusion_features(runtime.module, case['current'], case['reference'],
                case['step'], [center], case['position'])
            expected += fast._decode(fa.require_head(runtime.module), features, tasks, runtime.module.DEVICE)
        original = runtime.module.numeric_state; seen = []
        def numeric(*args):
            seen.append(1)
            return original(*args)
        with patch.object(runtime.module, 'numeric_state', side_effect=numeric):
            actual, _ = online.predict_pair(runtime, case, case['centers'], center_batch_size=1)
            self.assertEqual(len(seen), 1)
            online.predict_pair(runtime, dict(case, current=case['current']+.1), case['centers'], center_batch_size=1)
            self.assertEqual(len(seen), 2)
        for left, right in zip(expected, actual):
            self.assertEqual((left['center_sensor'], left['answer_type'], left['prediction']),
                             (right['center_sensor'], right['answer_type'], right['prediction']))
            np.testing.assert_allclose(list(left['probabilities'].values()), list(right['probabilities'].values()), atol=1e-6, rtol=1e-5)
        batched, _ = online.predict_pair(runtime, case, case['centers'], center_batch_size=8)
        for left, right in zip(expected, batched):
            self.assertEqual(left['prediction'], right['prediction'])
            np.testing.assert_allclose(list(left['probabilities'].values()), list(right['probabilities'].values()), atol=1e-6, rtol=1e-5)

    def test_request_matches_paired_computation_with_actual_frozen_train_reference(self):
        runtime = self.contract.runtime
        case = external.windows(self.frame, self.contract, 'case')[0][0]
        session = online.InferenceSession(runtime, center_batch_size=1)
        with patch('pandas.read_csv', side_effect=AssertionError('No I/O inside request')):
            result = session.process_window(current=case['raw_current'], step=case['step'],
                cohort=case['cohort'], run_id=case['run_id'], position=case['position'])
        reference = external.train_reference(runtime, case)
        expected, _ = online.predict_pair(runtime, dict(case, reference=reference), case['centers'])
        self.assertEqual(result['status'], 'OK')
        self.assertEqual([r['prediction'] for r in result['answers']], [r['prediction'] for r in expected])
        self.assertEqual(len(result['answers']), 9)
        self.assertGreater(result['timings']['request_ms'], 0)
        self.assertGreaterEqual(result['timings']['request_ms'], result['timings']['encoder_ms'])
        # A caller cannot submit an oracle reference or label into this API.
        with self.assertRaises(TypeError):
            session.process_window(current=case['raw_current'], step=case['step'],
                cohort=case['cohort'], run_id=case['run_id'], reference=reference)

    def test_source_truth_only_joins_after_predictions_and_cannot_change_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original = root/'original.csv'; input_path = root/'case.csv'
            self.raw.to_csv(original, index=False)
            self.frame.to_csv(input_path, index=False)
            real = online.InferenceSession.process_window; seen = []
            def record(session, **kwargs):
                seen.append(1)
                return real(session, **kwargs)
            lookup = external.original_lookup
            def evaluate_only(*args, **kwargs):
                self.assertGreater(len(seen), 0)
                return lookup(*args, **kwargs)
            with patch.object(online.InferenceSession, 'process_window', new=record), \
                 patch.object(external, 'original_lookup', side_effect=evaluate_only):
                scored = external.evaluate(self.contract, [input_path], original_path=original,
                    score_original=True)
            unscored = external.evaluate(self.contract, [input_path])
            self.assertEqual(scored.details.prediction.tolist(), unscored.details.prediction.tolist())
            self.assertFalse(scored.metadata['original_used_for_prediction'])
            self.assertTrue(scored.metadata['truth_used_after_prediction'])
            self.assertIn('unchanged_alarm_rate', scored.diagnostics['boolean_metrics'])
            self.assertIn('requests', scored.diagnostics)
            self.assertEqual(len(scored.diagnostics['requests']), 3)
            # A different/missing evaluator mapping cannot drop valid predictions.
            changed = self.raw.copy(); changed['job'] = 'not_the_source'; changed.to_csv(original, index=False)
            unmatched = external.evaluate(self.contract, [input_path], original_path=original, score_original=True)
            self.assertEqual(unmatched.details.prediction.tolist(), unscored.details.prediction.tolist())

    def test_reference_bank_rejects_validation_and_request_never_uses_same_run(self):
        runtime = self.contract.runtime
        row = runtime.prepared.windows[self.phase]['train'][0]
        runtime.prepared.windows[self.phase]['train'] = [row]
        session = online.InferenceSession(runtime)
        result = session.process_window(current=row['current'], standardized=True,
            step=self.phase, cohort=row['cohort'], run_id=row['run_id'])
        self.assertEqual(result['status'], 'NOT_EVALUATED')
        self.assertEqual(result['answers'], [])
        runtime.prepared.windows[self.phase]['train'][0]['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'TRAIN reference'):
            online.InferenceSession(runtime)

    def test_startup_respects_a_head_trained_on_only_selected_steps(self):
        runtime = self.contract.runtime
        runtime.encoders['not_trained'] = runtime.encoders[self.phase]
        runtime.prepared.windows['not_trained'] = runtime.prepared.windows[self.phase]
        session = online.InferenceSession(runtime)
        self.assertEqual(list(session.modules), [self.phase])
        result = session.process_window(current=np.zeros((8, 3)), step='not_trained',
            cohort=('recipe', 0, 0), run_id='unseen')
        self.assertEqual(result['status'], 'NOT_EVALUATED')
        self.assertEqual(result['answers'], [])


if __name__ == '__main__':
    unittest.main()
