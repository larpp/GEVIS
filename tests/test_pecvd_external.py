"""Invented source-format CSVs through real common inference, no private files."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import contextlib
import io
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from pecvd.experiments import pecvd_external as subject, fast_experiments as fast
from tests.test_fast_experiments import ready


class PecvdExternalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = ready()
        encoder, meta = self.runtime.encoders['phase']
        encoder.patch_len = 4
        meta.update(mean=np.array([1., 2.]), std=np.array([2., 3.]))
        self.runtime.module.FUSION_ANSWER_TRAINING = dict(trained_steps=['phase'],
            train_runs=['train-a', 'train-b'], validation_runs=['val'])
        self.runtime.module.ENCODER_ROOT = self.root
        (self.root/'windows').mkdir()
        arrays = {}
        for split, ids in [('train', ['train-a', 'train-b']), ('validation', ['val']), ('test', ['test'])]:
            runs = [r for r in ids for _ in range(10)]
            arrays.update({split: np.zeros((len(runs), 8, 2), dtype=np.float32),
                f'{split}_process_id': np.array(runs),
                f'{split}_cohort_key': np.array(['source|pm=one'] * len(runs)),
                f'{split}_position_bin': np.tile(np.arange(10), len(ids))})
        np.savez(self.root/'windows/phase.npz', **arrays)
        self.config = NS(process_family='PECVD', source_columns={}, columns=NS(
            run='process_key', step='step_label', order='row_in_process',
            step_one_hot_prefix='step_name_', step_one_hot_excluded=('step_name_nan',)))
        self.frame = pd.DataFrame(dict(process_key=['new']*16, row_in_process=np.arange(16),
            step_name_phase=[1]*16, source_group_source=[1]*16, pm_cycle_one=[1]*16,
            a=np.arange(16)/10, b=np.arange(16)/20))

    def test_real_forward_truth_join_preserves_answers_and_source_files(self):
        original = self.root/'original.csv'; path = self.root/'case.csv'
        self.frame.to_csv(original, index=False)
        changed = self.frame.copy(); changed.loc[2:3, 'a'] += 3
        changed.to_csv(path, index=False)
        before = path.read_bytes()
        report = subject.evaluate(self.runtime, self.config, [path], original_path=original)
        unscored = subject.evaluate(self.runtime, self.config, [path])
        self.assertEqual(len(report.details), 12)
        self.assertEqual(report.details.prediction.tolist(), unscored.details.prediction.tolist())
        self.assertEqual(report.details.paired_boolean_label.dropna().astype(int).sum(), 1)
        self.assertEqual(report.diagnostics['request_latency'].iloc[0].measured_windows, 2)
        self.assertEqual(report.metadata['domain'], 'PECVD')
        self.assertFalse(report.metadata['original_used_for_prediction'])
        self.assertEqual(path.read_bytes(), before)

    def test_no_matching_train_context_returns_no_fabricated_answers(self):
        frame = self.frame.rename(columns={'source_group_source':'source_group_absent'})
        path = self.root/'case.csv'; frame.to_csv(path, index=False)
        report = subject.evaluate(self.runtime, self.config, [path])
        self.assertTrue(report.details.empty)
        self.assertTrue(report.diagnostics['request_latency'].empty)
        self.assertTrue(report.diagnostics['requests'].reason.eq('no_matching_independent_TRAIN_reference').all())

    def test_preview_bounds_requests_without_original_or_scoring(self):
        path=self.root/'case.csv';self.frame.to_csv(path,index=False)
        result=subject.evaluate(self.runtime,self.config,[path],max_windows_per_file=1)
        self.assertEqual(len(result.diagnostics['requests']),1)
        self.assertEqual(result.diagnostics['coverage'].iloc[0].windows_not_requested_by_limit,1)
        self.assertFalse(result.metadata['truth_used_after_prediction'])
        self.assertFalse(result.details.paired_boolean_label.notna().any())

    def test_boundaries_native_scaling_and_invalid_order(self):
        path = self.root/'case.csv'; self.frame.to_csv(path, index=False)
        frame = subject.read_frame(path, self.runtime, self.config)
        cases, _ = subject.windows(frame, self.runtime, self.config, 'case')
        self.assertEqual([r['cohort'][1] for r in cases], [0, 9])
        np.testing.assert_allclose(cases[0]['raw_current'], self.frame[['a','b']].iloc[:8])
        frame.loc[8:, 'step_label'] = 'unknown'
        cases, excluded = subject.windows(frame, self.runtime, self.config, 'case')
        self.assertEqual(len(cases), 1)
        self.assertTrue(excluded)
        wrong = self.frame.copy(); wrong.loc[3, 'row_in_process'] = 1
        wrong.to_csv(path, index=False)
        with self.assertRaises(ValueError): subject.read_frame(path, self.runtime, self.config)

    def test_actual_notebook_run_all_uses_pecvd_adapter(self):
        path = self.root/'case.csv'; self.frame.to_csv(path, index=False)
        project = Path(__file__).resolve().parents[1]
        notebook = json.loads((project/'notebooks/CVD_predict_external_csv.ipynb').read_text(encoding='utf-8'))
        ns = dict()
        with patch('IPython.display.display'), patch.object(fast, 'load_runtime', return_value=self.runtime), contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code': continue
                source = ''.join(cell['source'])
                if cell['id'] == 'setup':
                    exec(compile(source, 'EXP9:setup', 'exec'), ns)
                    ns.update(DOMAIN='PECVD',INPUT_DIRECTORY=self.root)
                elif cell['id'] == 'load':
                    with patch('pecvd.process_config.load_process_config', return_value=self.config):
                        exec(compile(source, 'EXP9:load', 'exec'), ns)
                else:
                    exec(compile(source, 'EXP9:'+cell['id'], 'exec'), ns)
        self.assertEqual(ns['REPORT'].metadata['domain'], 'PECVD')
        self.assertFalse(ns['REPORT'].details.empty)


if __name__ == '__main__': unittest.main()
