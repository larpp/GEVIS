"""Fixed protocols on invented windows; never private inputs or model quality."""
import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import pattern_robustness as p, fast_experiments as fast
from tests.test_fast_experiments import ready, cache

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return json.loads((ROOT / 'config/experiments/fixed_pattern_example.json').read_text(encoding='utf-8'))


def windows(runtime):
    with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache):
        return p.source_windows(runtime, limit=2)


class PatternRobustnessTests(unittest.TestCase):
    def test_fixed_targets_controls_and_compound_ambiguity_preserve_inputs(self):
        runtime = ready(); inputs = windows(runtime)
        original = [row['current'].copy() for row in inputs]
        generated = p.cases(runtime, inputs, plan())
        self.assertEqual(len(generated), 20)
        for case in generated:
            np.testing.assert_array_equal(case['changed_mask'], case['current'] != case['reference'])
            for sensor, labels in case['targets'].items():
                changed = bool(case['changed_mask'][:, ['a', 'b'].index(sensor)].any())
                self.assertEqual(labels['BOOLEAN'], changed)
                if not changed:
                    self.assertEqual(labels, dict(BOOLEAN=0, CHOICE=0, ORDINAL=0))
                elif case['challenge'] == 'level_plus_drift':
                    self.assertEqual(set(labels), {'BOOLEAN'})
        for row, before in zip(inputs, original): np.testing.assert_array_equal(row['current'], before)
        repeated = p.cases(runtime, inputs, plan())
        for a, b in zip(generated, repeated): np.testing.assert_array_equal(a['current'], b['current'])

    def test_nonstage_strength_has_no_invented_ordinal_label_and_flat_delay_is_negative(self):
        runtime = ready(); inputs = windows(runtime)
        config = plan(); config['patterns'] = [config['patterns'][0]]
        config['patterns'][0]['components'][0]['strength'] = .75
        changed = next(c for c in p.cases(runtime, inputs, config) if c['challenge'] != 'control')
        self.assertEqual(set(changed['targets']['a']), {'BOOLEAN', 'CHOICE'})
        config = plan(); config['patterns'] = [config['patterns'][2]]
        for row in inputs: row['current'] = np.ones((8, 2))
        self.assertTrue(all(not c['changed_mask'].any() for c in p.cases(runtime, inputs, config)))

    def test_lock_sensor_span_and_split_errors_are_rejected(self):
        runtime = ready(); inputs = windows(runtime); config = plan()
        digest = p.protocol_digest(config); config['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Freeze'):
            p.compare({'current': runtime}, inputs, config, locked_digest=digest)
        config = plan(); config['patterns'][0]['components'][0]['sensor'] = 99
        with self.assertRaisesRegex(ValueError, 'sensor index'):
            p.cases(runtime, inputs, config)
        config = plan(); config['training_exposure'] = 'unseen'
        with self.assertRaisesRegex(ValueError, 'cannot certify'):
            p.protocol_digest(config)
        inputs[0]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Test'):
            p.cases(runtime, inputs, plan())

    def test_real_comparison_uses_same_cases_and_frozen_parameters(self):
        a = ready(); b = copy.deepcopy(a)
        config = plan(); inputs = windows(a)
        before = p._fingerprint(a)
        report = p.compare({'baseline': a, 'candidate': b}, inputs, config, locked_digest=p.protocol_digest(config))
        self.assertTrue(report.metadata['weights_preserved'])
        self.assertFalse(report.metadata['test_used_for_selection'])
        self.assertEqual(p._fingerprint(a), before)
        left = report.reports['baseline'].details.sort_values(['pair_id', 'center_sensor', 'answer_type']).reset_index(drop=True)
        right = report.reports['candidate'].details.sort_values(['pair_id', 'center_sensor', 'answer_type']).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        excluded = report.reports['baseline'].diagnostics['unscored_answers']
        self.assertEqual(set(excluded.answer_type), {'CHOICE', 'ORDINAL'})
        only_negative = report.summary.query("pattern == 'level_plus_drift' and answer_type != 'BOOLEAN'")
        self.assertTrue(only_negative.changed_questions.eq(0).all())
        self.assertTrue(only_negative.unscored_questions.gt(0).all())

    def test_train_overlap_and_weight_mutation_are_rejected(self):
        runtime = ready(); inputs = windows(runtime); config = plan()
        runtime.module.FUSION_ANSWER_TRAINING = dict(train_runs=[inputs[0]['run_id']], validation_runs=[])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            p.compare({'current': runtime}, inputs, config, locked_digest=p.protocol_digest(config))
        runtime.module.FUSION_ANSWER_TRAINING = dict(trained_steps=[])
        with self.assertRaisesRegex(ValueError, 'no trained answer-head support'):
            p.compare({'current': runtime}, inputs, config, locked_digest=p.protocol_digest(config))
        runtime = ready(); real = fast.score_cases
        def mutate(*args, **kwargs):
            result = real(*args, **kwargs)
            with torch.no_grad(): next(runtime.module.FUSION_ANSWER_HEAD.parameters()).add_(1)
            return result
        with patch.object(fast, 'score_cases', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'weights changed'):
                p.compare({'current': runtime}, inputs, config, locked_digest=p.protocol_digest(config))

    def test_exp7_preview_does_not_run_optional_fixed_pattern_generator(self):
        # The external preview RunAll is covered by test_external_test. Fixed
        # pattern comparisons above remain available as a separate utility.
        notebook = json.loads((ROOT / 'notebooks/ALD_predict_external_csv.ipynb').read_text(encoding='utf-8'))
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        self.assertNotIn('patterns.compare(', source)
        self.assertNotIn('patterns.source_windows(', source)
        self.assertIn('score_original=False', source)
        self.assertIn('max_windows_per_file=MAX_WINDOWS_PER_FILE', source)


if __name__ == '__main__':
    unittest.main()
