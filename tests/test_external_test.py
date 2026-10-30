"""Invented supplied CSVs, real frozen neural forward, no private input reads."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pecvd.experiments import external_test as external, peald_training as p
from pecvd.experiments import fusion_answers as fa, peald_readout, peald_step_heads
from tests.test_peald_training import fixture, prepared_runtime

ROOT = Path(__file__).resolve().parents[1]


def ready():
    source, process, raw, config, phase = fixture()
    target = prepared_runtime(source, process, raw, config)
    with contextlib.redirect_stdout(io.StringIO()):
        head, training = peald_readout.fit_readout(p.examples(target, 'train', config=config),
            p.examples(target, 'validation', config=config), config)
    target.module.FUSION_ANSWER_HEAD = head
    target.module.FUSION_ANSWER_TRAINING = training
    target.step_heads = {}
    test_run = next(job for job in raw.job.unique() if p._hash(job) in target.prepared.split_runs['test'])
    frame = raw.loc[raw.job.eq(test_run)].reset_index(drop=True)
    return external.InputContract(target, process), raw, frame, phase


class ExternalTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.raw, cls.frame, cls.phase = ready()

    def setUp(self):
        self.contract = copy.deepcopy(self.base)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root/'original.csv'
        self.raw.to_csv(self.original, index=False)
        self.control = self.root/'case_a.csv'
        self.frame.to_csv(self.control, index=False)
        self.injected = self.root/'case_b.csv'
        frame = self.frame.copy(); frame.loc[5, 'p'] += 2
        frame.to_csv(self.injected, index=False)

    def test_actual_forward_paired_truth_is_center_specific_and_no_fit_or_export(self):
        before = external._fingerprint(self.contract.runtime)
        sources = {path:path.read_bytes() for path in (self.original, self.control, self.injected)}
        with patch.object(peald_readout, 'fit_readout', side_effect=AssertionError('No Test fit')), \
             patch.object(p, 'prepare_frame', side_effect=AssertionError('Never fit preprocessing on Test')):
            report = external.evaluate(self.contract, [self.control, self.injected],
                reference_mode='paired_original', original_path=self.original)
        self.assertEqual(set(report.details.answer_type), set(fa.KINDS))
        changed = report.details.query("answer_type == 'BOOLEAN' and paired_boolean_label == 1")
        self.assertEqual(changed.center_sensor.tolist(), ['p'])
        self.assertTrue(changed.case_id.eq('case_b').all())
        self.assertFalse(report.details.loc[~report.details.answer_type.eq('BOOLEAN'), 'paired_boolean_label'].notna().any())
        self.assertEqual(report.diagnostics['coverage'].source_group.nunique(), 1)
        self.assertEqual(report.diagnostics['boolean_metrics'].positive_support.iloc[0], 1)
        self.assertEqual(len(report.diagnostics['changed_regions']), 1)
        self.assertFalse(report.metadata['test_used_for_fitting'])
        self.assertFalse(report.metadata['independent_test_certified'])
        self.assertEqual(before, external._fingerprint(self.contract.runtime))
        self.assertEqual(sources, {path:path.read_bytes() for path in sources})
        self.assertEqual(set(self.root.iterdir()), set(sources))

    def test_alignment_ambiguity_missing_coordinates_and_duplicate_cases_fail_closed(self):
        lookup = external.original_lookup([self.control], self.original, self.contract)
        frame = external.read_input(self.control, self.contract)
        unknown = frame.copy(); unknown.loc[0, 'job'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            external.align_original(unknown, lookup, self.contract)
        duplicate = pd.concat([frame.iloc[[0]], frame], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            external.align_original(duplicate, lookup, self.contract)
        raw = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        raw.to_csv(self.original, index=False)
        all_input = self.root/'all.csv'; self.raw.to_csv(all_input, index=False)
        ambiguous = external.original_lookup([all_input], self.original, self.contract)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            external.align_original(self.raw, ambiguous, self.contract)
        with self.assertRaisesRegex(ValueError, 'Unique nonempty'):
            external.evaluate(self.contract, [self.control, self.control], reference_mode='train_bank')

    def test_no_window_crosses_step_nan_cadence_or_reappearing_run(self):
        frame = self.frame.copy(); frame.loc[8:15, 'phase'] = 'unsupported'
        frame.loc[18, 'p'] = np.nan
        cases, excluded = external.windows(frame, self.contract, 'case', stride=1)
        self.assertTrue(cases)
        self.assertTrue(all(c['row_end'] <= 8 for c in cases))
        self.assertIn('unsupported_phase', {r['reason'] for r in excluded})
        frame = self.frame.copy(); frame.loc[12:, 'stamp'] = (
            pd.to_datetime(frame.loc[12:, 'stamp'])+pd.Timedelta(seconds=30)).astype(str)
        cases, _ = external.windows(frame, self.contract, 'case', stride=1)
        self.assertTrue(cases)
        self.assertTrue(all(not (c['row_start'] < 12 < c['row_end']) for c in cases))
        repeated = pd.concat([self.frame, self.frame.assign(job='different'), self.frame], ignore_index=True)
        cases, excluded = external.windows(repeated, self.contract, 'case')
        self.assertTrue(all(c['run_id'] == p._hash('different') for c in cases))
        self.assertIn('ambiguous_recipe_run_or_time', {r['reason'] for r in excluded})

    def test_frozen_scaling_no_clipping_and_fit_overlap_are_explicit(self):
        frame = self.frame.copy(); frame.loc[0, 'p'] = .123456789012345
        cases, _ = external.windows(frame, self.contract, 'precision')
        meta = self.contract.runtime.encoders[self.phase][1]
        expected = ((frame.iloc[:8][meta['sensor_columns']].to_numpy(dtype=np.float32)
                     -meta['mean'])/meta['std']).astype(np.float32)
        np.testing.assert_array_equal(cases[0]['current'], expected)
        frame = self.frame.copy(); frame.loc[0, 'p'] = 1e9
        cases, _ = external.windows(frame, self.contract, 'case')
        meta = self.contract.runtime.encoders[self.phase][1]
        self.assertAlmostEqual(float(cases[0]['current'][0, 0]),
            float(np.float32((1e9-meta['mean'][0])/meta['std'][0])))
        train_run = next(job for job in self.raw.job.unique() if p._hash(job) in self.contract.runtime.prepared.split_runs['train'])
        self.raw.loc[self.raw.job.eq(train_run)].to_csv(self.control, index=False)
        result = external.evaluate(self.contract, [self.control], reference_mode='paired_original', original_path=self.original)
        self.assertTrue(result.diagnostics['coverage'].overlap_status.eq('overlaps_fit_or_selection_runs').all())
        self.assertFalse(result.metadata['independent_test_certified'])

    def test_unlabelled_predictions_then_exact_review_join_without_refit(self):
        report = external.evaluate(self.contract, [self.injected], reference_mode='train_bank', centers=['p'])
        self.assertFalse(report.details.empty)
        self.assertNotIn('label', report.details)
        self.assertNotIn('paired_boolean_label', report.details)
        self.assertFalse(report.summary.accuracy_available.iloc[0])
        self.assertNotIn('boolean_metrics', report.diagnostics)
        keys = ['case_id', 'row_start', 'row_end', 'center_sensor', 'answer_type']
        labels = report.details[keys].iloc[[0]].copy(); labels['label'] = 0
        result = external.score_reviewed_labels(report, labels, truth_source='Invented independent fixture review')
        self.assertEqual(result.metadata['labelled_questions'], 1)
        self.assertTrue(result.summary.metric_target.str.startswith('independently reviewed').all())
        bad = labels.copy(); bad['row_start'] += 1
        with self.assertRaisesRegex(ValueError, 'exact predicted'):
            external.score_reviewed_labels(report, bad, truth_source='review')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            external.score_reviewed_labels(report, labels, truth_source='')

    def test_no_reference_or_no_native_support_produces_explicit_withholding(self):
        self.contract.runtime.prepared.windows[self.phase]['train'] = []
        report = external.evaluate(self.contract, [self.injected], reference_mode='train_bank')
        self.assertTrue(report.details.empty)
        self.assertFalse(report.metadata['model_evaluated'])
        self.assertTrue(report.diagnostics['exclusions'].reason.str.contains('TRAIN_reference').any())
        short = self.frame.iloc[:3].copy()
        cases, exclusions = external.windows(short, self.contract, 'short')
        self.assertFalse(cases)
        self.assertEqual(exclusions[0]['rows'], 3)

    def test_model_mutation_is_rejected(self):
        real = external.online.predict_pair
        def mutate(*args, **kwargs):
            answers = real(*args, **kwargs)
            import torch
            with torch.no_grad():
                next(self.contract.runtime.module.FUSION_ANSWER_HEAD.parameters()).add_(1)
            return answers
        with patch.object(external.online, 'predict_pair', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'weights changed'):
                external.evaluate(self.contract, [self.injected], reference_mode='train_bank', centers=['p'])

    def test_actual_exp7_exp8_run_all_use_supplied_files_without_new_injection(self):
        self.contract.process.data_path = str(self.original)
        for name in ('ALD_predict_external_csv.ipynb',):
            env = {'display':lambda value:None}
            with patch.object(external, 'load_peald', return_value=self.contract), \
                 patch.object(external, 'inventory', return_value=[self.injected]), \
                 patch.object(external, 'original_lookup', side_effect=AssertionError('No preview scoring')), \
                 patch('pecvd.experiments.injection.inject', side_effect=AssertionError('No new injection')), \
                 patch.object(p, 'train_or_load', side_effect=AssertionError('No notebook fitting')), \
                 contextlib.redirect_stdout(io.StringIO()):
                nb = json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
                for cell in nb['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), name+':'+cell['id'], 'exec'), env)
                        if cell['id'] in ('setup', 'load-frozen-model'):
                            env['WINDOW_STRIDE'] = None
                            env['INPUT_DIRECTORY'] = self.injected.parent
            self.assertFalse(env['REPORT'].details.empty)
            self.assertFalse(env['REPORT'].metadata['test_used_for_fitting'])
            self.assertFalse(env['REPORT'].metadata['truth_used_after_prediction'])

    def test_preview_limit_bounds_actual_forward_requests_and_reports_coverage(self):
        with patch.object(external, 'original_lookup', side_effect=AssertionError('No preview scoring')):
            report=external.evaluate(self.contract,[self.injected],max_windows_per_file=1)
        self.assertEqual(len(report.diagnostics['requests']),1)
        self.assertGreater(report.diagnostics['coverage'].iloc[0].available_windows,1)
        self.assertFalse(report.metadata['truth_used_after_prediction'])
        with self.assertRaises(ValueError):
            external.evaluate(self.contract,[self.injected],max_windows_per_file=0)

    def test_exp7_setup_recovers_a_cached_evaluator_without_score_original(self):
        import inspect
        notebook = json.loads((ROOT/'notebooks/ALD_predict_external_csv.ipynb').read_bytes())
        setup = ''.join(next(c['source'] for c in notebook['cells'] if c.get('id') == 'setup'))
        def old_evaluate(contract, paths, *, reference_mode='train_bank'):
            raise AssertionError('Stale evaluator must not run')
        with patch.object(external, 'evaluate', old_evaluate), \
             patch.object(external, 'load_peald', side_effect=AssertionError('Setup must not load private data')), \
             patch('pathlib.Path.cwd', return_value=ROOT), patch.dict('os.environ', {}):
            namespace = {}
            exec(compile(setup, 'EXP7:lock-protocol', 'exec'), namespace)
            loaded = namespace['external']
            self.assertIn('score_original', inspect.signature(loaded.evaluate).parameters)
            inspect.signature(loaded.evaluate).bind(None, [], reference_mode='train_bank',
                                                   score_original=True, progress=print)

    def test_loader_only_restores_exact_saved_peald_head(self):
        with patch.object(external.fast, 'load_runtime', return_value=self.contract.runtime.source_runtime), \
             patch('pecvd.process_config.load_process_config', return_value=self.contract.process), \
             patch.object(p, 'prepare', return_value=self.contract.runtime.prepared), \
             patch.object(p, 'train_or_load', side_effect=FileNotFoundError('Exact checkpoint missing')) as load:
            with self.assertRaisesRegex(FileNotFoundError, 'Exact checkpoint'):
                external.load_peald(ROOT)
        self.assertFalse(load.call_args.kwargs['allow_training'])


if __name__ == '__main__':
    unittest.main()
