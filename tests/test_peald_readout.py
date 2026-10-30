"""Synthetic-only checks for grouped target-head learning and Validation selection."""
import contextlib
from dataclasses import replace
import io
import unittest

import torch

from pecvd.experiments import fusion_answers as fa, peald_readout as r
from pecvd.experiments.peald_training import PealdConfig


def examples(split):
    rows = []
    for sample in range(30):
        choice = sample % 6
        grade = 1 + sample % 4 if choice else 0
        signal = torch.zeros(12)
        signal[choice] = 1.
        signal[6 + grade] = 1.
        for i, kind in enumerate(fa.KINDS):
            rows.append(dict(features=signal + i * .01, answer_type=kind,
                label=(int(choice > 0), choice, grade)[i], split=split,
                run_id=split + str(sample // 6), step='invented',
                pair_id=str(sample), center_sensor='invented_sensor'))
    return rows


class PealdReadoutTests(unittest.TestCase):
    def test_grouping_uses_pair_and_center_identity_not_row_order(self):
        original = examples('train')
        a, b = r.grouped(original, 'train', 'cpu')
        scrambled = original[::3] + original[1::3] + original[2::3]
        x, y = r.grouped(scrambled, 'train', 'cpu')
        torch.testing.assert_close(a, x)
        torch.testing.assert_close(b, y)
        with self.assertRaisesRegex(ValueError, 'Complete'):
            r.grouped(original[:-1], 'train', 'cpu')
        with self.assertRaisesRegex(ValueError, 'Unique'):
            r.grouped(original + [original[0]], 'train', 'cpu')
        with self.assertRaisesRegex(ValueError, 'Only TRAIN'):
            r.grouped(examples('test'), 'train', 'cpu')
        broken = [dict(row) for row in original]
        broken[0]['label'] = 1
        with self.assertRaisesRegex(ValueError, 'Contradictory'):
            r.grouped(broken, 'train', 'cpu')

    def test_ordinal_loss_penalizes_farther_mistakes(self):
        def distance(pred):
            z = torch.full((1, 5), -20.)
            z[0, pred] = 20.
            return r.auxiliary_losses([torch.zeros(1, 2), torch.zeros(1, 6), z], torch.tensor([1]))[1]
        self.assertLess(distance(1), distance(2))
        self.assertLess(distance(2), distance(4))

    def test_consistency_loss_backpropagates_through_every_question(self):
        logits = [torch.tensor([[0., 2.]], requires_grad=True),
                  torch.tensor([[3., 0., 0., 0., 0., 0.]], requires_grad=True),
                  torch.tensor([[0., 1., 1., 1., 1.]], requires_grad=True)]
        consistency, ordinal = r.auxiliary_losses(logits, torch.tensor([2]))
        consistency.backward()
        for value in logits:
            self.assertGreater(float(value.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(ordinal))
        agreed = [torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2)]
        self.assertEqual(float(r.auxiliary_losses(agreed, torch.tensor([1, 1]))[0]), 0.)

    def test_selection_prioritizes_detection_goal_then_macro_quality(self):
        low = dict(boolean_f1=.95, macro_f1=.99)
        passed = dict(boolean_f1=.96, macro_f1=.7)
        stronger = dict(boolean_f1=.97, macro_f1=.8)
        self.assertGreater(r.selection_key(passed, 2., .96), r.selection_key(low, .1, .96))
        self.assertGreater(r.selection_key(stronger, 3., .96), r.selection_key(passed, .1, .96))

    def test_real_fit_restores_selected_validation_state_and_preserves_argmax(self):
        config = replace(PealdConfig(), epochs=3, width=16, batch_size=30, max_seconds=30)
        train = examples('train'); validation = examples('validation')
        with contextlib.redirect_stdout(io.StringIO()):
            head, result = r.fit_readout(train, validation, config)
        self.assertGreater(result['optimizer_updates'], 0)
        self.assertEqual(result['training_recipe'], r.RECIPE)
        self.assertFalse(set(result['train_runs']) & set(result['validation_runs']))
        x, y = r.grouped(validation, 'validation', 'cpu')
        with torch.no_grad():
            logits = [head(x[:, i], kind) for i, kind in enumerate(fa.KINDS)]
        actual = r.metrics(logits, y)
        for key, value in actual.items():
            self.assertEqual(result['selected_validation'][key], value)
        self.assertEqual(result['validation_boolean_goal_met'], actual['boolean_f1'] >= .96)
        self.assertTrue(all(not p.requires_grad for p in head.parameters()))
        # The saved model is still the same ordinary three-output readout.
        restored = fa.ContextReadout(12, 16)
        restored.load_state_dict(head.state_dict(), strict=True)
        for i, kind in enumerate(fa.KINDS):
            torch.testing.assert_close(restored(x[:, i], kind), logits[i])

    def test_invalid_config_and_split_overlap_are_rejected(self):
        for changes in (dict(consistency_weight=-1), dict(ordinal_weight=float('nan')),
                        dict(boolean_f1_goal=0), dict(boolean_f1_goal=1.1)):
            with self.assertRaises(ValueError):
                replace(PealdConfig(), **changes).validate()
        train = examples('train'); validation = examples('validation')
        validation[0]['run_id'] = train[0]['run_id']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            r.fit_readout(train, validation, PealdConfig())


if __name__ == '__main__':
    unittest.main()
