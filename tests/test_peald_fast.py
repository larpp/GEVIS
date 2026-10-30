"""Invented PEALD runs through native encoder, real Fusion and typed readout."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import peald_fast as peald, fusion_answers as fa
from pecvd.gevis_backbone import token_descriptions
from pecvd.temporal_patchtst import TemporalPatchEncoder
from tests.test_fusion_answers import runtime

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    r = runtime(); m = r.module
    encoder = TemporalPatchEncoder(2, 4, d_model=8, nhead=2, layers=1, dropout=0.).requires_grad_(False).eval()
    m.STEP_ENCODERS = {'DEP.': (encoder, dict(sensor_columns=['a','b'], window_length=8))}
    r.encoders = m.STEP_ENCODERS
    m.numeric_state = lambda e, a, b: e.tokens(a) - e.tokens(b)
    m.numeric_token_descriptions = lambda step, meta, p, n: [token_descriptions(
        meta.get('step_descriptor', step), meta['sensor_columns'], p)] * n
    f, _ = fa.fusion_features(m, np.zeros((8,2)), np.zeros((8,2)), 'DEP.', ['a'])
    m.FUSION_ANSWER_HEAD = fa.ContextReadout(f.shape[-1], 16).requires_grad_(False).eval()
    m.FUSION_ANSWER_TRAINING = dict(trained_steps=['DEP.'])
    config = NS(sensor_columns=['p','q','r'], sensor_descriptors={},
        columns=NS(run='run', step='step', order='row'), steps=[NS(name='target',description='Target phase')])
    rows = [dict(run=f'run{i:02}', step='target', row=t, _gevis_segment_id=i,
        p=np.sin(t/3)+i/10, q=np.cos(t/2), r=t/7) for i in range(24) for t in range(16)]
    return r, config, pd.DataFrame(rows)


class PealdFastTests(unittest.TestCase):
    def test_native_transfer_preserves_all_weights_and_source_support(self):
        source, config, frame = fixture(); before = frame.copy(deep=True)
        modules = [source.encoders['DEP.'][0], source.module.GEVIS_MODEL, source.module.FUSION_ANSWER_HEAD]
        states = [{k:v.clone() for k,v in m.state_dict().items()} for m in modules]
        with patch.object(fa, 'fit_readout', side_effect=AssertionError('No PEALD fitting')):
            target, support = peald.bind(source, config, frame, source_step='DEP.', pairs_per_step=1)
            self.assertEqual(support.status.tolist(), ['READY'])
            self.assertIs(target.encoders['target'][0], modules[0])
            self.assertIs(target.module.GEVIS_MODEL, modules[1])
            self.assertIs(target.module.FUSION_ANSWER_HEAD, modules[2])
            report = peald.controlled(target)
            original = peald.original(target, center_batch_size=1)
        self.assertEqual(set(report.summary.answer_type), set(fa.KINDS))
        self.assertIn('f1', report.diagnostics['boolean_metrics'])
        self.assertEqual(len(original.details), 9)
        self.assertNotIn('label', original.details)
        self.assertFalse(original.summary.accuracy_available.iloc[0])
        self.assertFalse(report.metadata['target_steps_are_source_trained'])
        self.assertEqual(source.module.FUSION_ANSWER_TRAINING['trained_steps'], ['DEP.'])
        self.assertEqual(list(source.encoders), ['DEP.'])
        with self.assertRaisesRegex(RuntimeError, 'no trained Fusion'):
            fa.predict_arrays(target.module, np.zeros((8,3)), np.zeros((8,3)), 'target', 'p')
        for model, saved in zip(modules, states):
            for key, value in saved.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        pd.testing.assert_frame_equal(frame, before)

    def test_run_separation_target_masks_and_no_truth_in_forward(self):
        source, config, frame = fixture()
        target, _ = peald.bind(source, config, frame, source_step='DEP.', pairs_per_step=1)
        seen = []
        real_score = peald.fast.score_cases
        def score(r, cases):
            seen.extend(cases)
            return real_score(r, cases)
        with patch.object(peald.fast, 'score_cases', side_effect=score):
            peald.controlled(target)
        for case in seen:
            np.testing.assert_array_equal(case['changed_mask'], case['current'] != case['reference'])
            for center in case['centers']:
                self.assertEqual(case['targets'][center]['BOOLEAN'], int(case['changed_mask'][:, ['p','q','r'].index(center)].any()))
        self.assertTrue(any(c['kind']=='delay' for c in seen))
        pair = target.external_pairs['target'][0]['test'][0][0]
        self.assertNotIn(pair.attrs['evaluation_run_id'], pair.attrs['reference_run_ids'])
        pair.attrs['reference_run_ids'].append(pair.attrs['evaluation_run_id'])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            peald.original(target)

    def test_unsupported_inputs_fail_without_fallback(self):
        source, config, frame = fixture()
        source.encoders['DEP.'][0].supports_native_sensor_transfer = False
        with patch.object(peald, 'external_pairs', side_effect=AssertionError('Must fail before data preparation')):
            with self.assertRaisesRegex(peald.NativeTransferUnsupported, 'no ID substitution'):
                peald.bind(source, config, frame, source_step='DEP.')
        source.encoders['DEP.'][0].supports_native_sensor_transfer = True
        target, _ = peald.bind(source, config, frame, source_step='DEP.', pairs_per_step=1)
        with self.assertRaisesRegex(ValueError, 'supported PEALD'):
            peald.controlled(target, steps=['not-supported'])
        with self.assertRaisesRegex(ValueError, 'center sensors'):
            peald.original(target, centers=['unknown'])

    def test_active_step_notebooks_have_no_joint_comparison(self):
        for path in (ROOT/'notebooks').rglob('*.ipynb'):
            text = path.read_text(encoding='utf-8')
            self.assertNotIn('fusion_adaptation', text)
            self.assertNotIn('joint_cross_attention', text)
            self.assertNotIn('cross-adaptation', text)
