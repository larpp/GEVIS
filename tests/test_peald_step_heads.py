"""Step specialization with invented inputs; no private files or weights."""
import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from pecvd.experiments import fusion_answers as fa, peald_step_heads as s, peald_training as p
from tests.test_peald_readout import examples
from tests.test_peald_training import fixture, prepared_runtime


def constant_head(dimension, width, index):
    head = fa.ContextReadout(dimension, width)
    with torch.no_grad():
        for kind in fa.KINDS:
            head.answers[kind].weight.zero_()
            head.answers[kind].bias.fill_(-20.)
            head.answers[kind].bias[index] = 20.
    return head.requires_grad_(False).eval()


class StepHeadTests(unittest.TestCase):
    def test_selection_needs_interpretation_improvement_and_detection_support(self):
        shared = dict(boolean_f1=.98, macro_f1=.8, choice_macro_f1=.7, ordinal_macro_f1=.7)
        good = dict(boolean_f1=.97, macro_f1=.85, choice_macro_f1=.8, ordinal_macro_f1=.8)
        self.assertTrue(s.prefer_step(shared, good, .96))
        self.assertFalse(s.prefer_step(shared, dict(good, boolean_f1=.94), .96))
        self.assertFalse(s.prefer_step(shared, dict(good, ordinal_macro_f1=.5), .96))
        self.assertFalse(s.prefer_step(shared, shared, .96))

    def test_insufficient_run_or_class_support_keeps_shared_without_training(self):
        train = examples('train'); val = examples('validation')
        head = fa.ContextReadout(12, 16)
        with patch.object(s.readout, 'fit_readout', side_effect=AssertionError('No unsupported fitting')):
            result, selection = s.fit_steps(head, train, val, replace(p.PealdConfig(), min_step_train_runs=9), 'cpu')
            self.assertFalse(result)
            self.assertEqual(selection[0]['reason'], 'insufficient_independent_run_support')
            for row in train: row['label'] = 0
            result, selection = s.fit_steps(head, train, val, p.PealdConfig(), 'cpu')
            self.assertEqual(selection[0]['reason'], 'insufficient_step_class_coverage')
            # A Step cannot silently mask an answer learned by the common head,
            # even if that class is absent from its local Validation as well.
            train = [r for r in examples('train') if int(r['pair_id']) % 6 != 5]
            val = [r for r in examples('validation') if int(r['pair_id']) % 6 != 5]
            result, selection = s.fit_steps(head, train, val, p.PealdConfig(), 'cpu')
            self.assertEqual(selection[0]['reason'], 'insufficient_step_class_coverage')

    def test_warm_start_training_preserves_common_state_and_feature_coordinates(self):
        shared = fa.ContextReadout(12, 16)
        with torch.no_grad(): shared.feature_mean.fill_(2.); shared.feature_scale.fill_(3.)
        before = {k: v.clone() for k, v in shared.state_dict().items()}
        config = replace(p.PealdConfig(), width=16, step_epochs=2, step_max_seconds=30)
        with patch.object(s, 'prefer_step', return_value=True), contextlib.redirect_stdout(io.StringIO()):
            heads, selection = s.fit_steps(shared, examples('train'), examples('validation'), config, 'cpu')
        chosen = heads['invented']
        self.assertEqual(selection[0]['selected_head'], 'step')
        torch.testing.assert_close(chosen.feature_mean, shared.feature_mean)
        torch.testing.assert_close(chosen.feature_scale, shared.feature_scale)
        for key, value in before.items(): torch.testing.assert_close(shared.state_dict()[key], value)
        self.assertTrue(any(not torch.equal(chosen.state_dict()[k], v) for k, v in before.items()))
        restored = s.restore(s.dump(heads), selection, 12, 16, ['invented'])
        for key, value in chosen.state_dict().items(): torch.testing.assert_close(restored['invented'].state_dict()[key], value)
        with self.assertRaisesRegex(ValueError, 'missing'):
            s.restore({}, selection, 12, 16, ['invented'])

    def test_routing_is_step_specific_and_does_not_mutate_common_runtime(self):
        common = constant_head(12, 16, 0); local = constant_head(12, 16, 1)
        runtime = NS(module=NS(FUSION_ANSWER_HEAD=common, DEVICE='cpu'),
                     encoders={'a': None, 'b': None}, step_heads={'a': local})
        a = s.for_step(runtime, 'a'); b = s.for_step(runtime, 'b')
        self.assertIs(a.module.FUSION_ANSWER_HEAD, local)
        self.assertIs(b.module.FUSION_ANSWER_HEAD, common)
        self.assertIs(runtime.module.FUSION_ANSWER_HEAD, common)
        with self.assertRaises(ValueError): s.for_step(runtime, 'unknown')
        runtime.module.FUSION_ANSWER_TRAINING = dict(step_head_selection=[dict(step='b', selected_head='step')])
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            s.for_step(runtime, 'b')

    def test_actual_target_inference_evaluation_and_original_use_selected_head(self):
        source, process, raw, config, phase = fixture()
        target = prepared_runtime(source, process, raw, config)
        window = target.prepared.windows[phase]['test'][0]
        features, _ = fa.fusion_features(target.module, window['current'], window['current'], phase, ['p'])
        common = constant_head(features.shape[-1], config.width, 0)
        local = constant_head(features.shape[-1], config.width, 1)
        target.module.FUSION_ANSWER_HEAD = common
        target.module.FUSION_ANSWER_TRAINING = dict(trained_steps=[phase], step_head_selection=[])
        target.step_heads = {phase: local}
        answers = p.predict_arrays(target, window['current'], window['current'], phase, 'p')
        self.assertTrue(all(a['answer_index'] == 1 for a in answers))
        report = p.evaluate(target, config=config)
        self.assertTrue(report.details.prediction.eq(1).all())
        original = p.original(target, config=config)
        self.assertFalse(original.details.empty)
        self.assertTrue(original.details.prediction.eq(1).all())
        self.assertIs(target.module.FUSION_ANSWER_HEAD, common)

    def test_target_checkpoint_retains_selected_step_states_and_rejects_missing_state(self):
        source, process, raw, config, phase = fixture()
        config = replace(config, step_epochs=1)
        target = prepared_runtime(source, process, raw, config)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source.module.BACKBONE_PATH = root / 'backbone'; source.module.BACKBONE_PATH.write_bytes(b'synthetic')
            source.module.FUSION_ANSWER_PATH = root / 'head'; source.module.FUSION_ANSWER_PATH.write_bytes(b'synthetic')
            with patch.object(s, 'prefer_step', return_value=True), contextlib.redirect_stdout(io.StringIO()):
                p.train_or_load(root, target, config=config)
            self.assertIn(phase, target.step_heads)
            fresh = prepared_runtime(source, process, raw, config)
            with patch.object(p, 'examples', side_effect=AssertionError('Reload never fits')):
                p.train_or_load(root, fresh, config=config, allow_training=False)
            for k, v in target.step_heads[phase].state_dict().items():
                torch.testing.assert_close(fresh.step_heads[phase].state_dict()[k], v)
            path = target.module.FUSION_ANSWER_PATH
            payload = torch.load(path, weights_only=False); payload['step_states'] = {}; torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'missing'):
                p.train_or_load(root, fresh, config=config, allow_training=False)


if __name__ == '__main__':
    unittest.main()
