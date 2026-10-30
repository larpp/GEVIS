"""Invented windows through real fast Fusion/heads; no private data or artifacts."""
import ast
import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pecvd.experiments import fast_experiments as fast, fusion_answers as fa
from tests.test_fusion_answers import runtime, source_rows

ROOT = Path(__file__).resolve().parents[1]


def ready():
    r = runtime()
    x, _ = fa.fusion_features(r.module, np.zeros((8, 2)), np.zeros((8, 2)), 'phase', ['a'])
    r.module.FUSION_ANSWER_HEAD = fa.ContextReadout(x.shape[-1], 16).requires_grad_(False).eval()
    return r


def cache(*args, **kwargs):
    return source_rows(kwargs['split'])


class FastExperimentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(fa.FusionConfig(), max_windows_per_step=4, epochs=2, width=16,
                           max_seconds=10., batch_size=64)

    def test_case_targets_are_selected_center_conditional_and_unknown_families_unscored(self):
        r = ready()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache):
            for challenge in fast.CHALLENGES:
                cases = fast.controlled_cases(r, challenge=challenge, config=self.cfg)
                if challenge in ('standard','delay'):
                    self.assertTrue(any(c['kind']=='delay' and c['changed_mask'].any() for c in cases))
                expected = ({'BOOLEAN'} if challenge == 'compound' else
                            {'BOOLEAN', 'CHOICE'} if challenge == 'amplitude' else set(fa.KINDS))
                for case in cases:
                    np.testing.assert_array_equal(case['changed_mask'], case['current'] != case['reference'])
                    for center in case['centers']:
                        sid = ['a', 'b'].index(center)
                        self.assertEqual(set(case['targets'][center]), expected)
                        self.assertEqual(case['targets'][center]['BOOLEAN'], int(case['changed_mask'][:, sid].any()))
                changed = next(c for c in cases if c['kind'] != 'equal')
                self.assertEqual([changed['targets'][c]['BOOLEAN'] for c in changed['centers']], [1, 0])
        with self.assertRaisesRegex(ValueError, 'Unknown controlled'):
            fast.controlled_cases(r, challenge='unsupported')

    def test_question_probes_change_text_only_restore_hooks_and_leave_frozen_weights(self):
        r = ready(); m = r.module
        before = {k: v.clone() for k, v in m.GEVIS_MODEL.state_dict().items()}
        seen = []
        def capture(module, args, output):
            seen.append((list(args[1]), list(args[2])))
        handle = m.GEVIS_MODEL.register_forward_hook(capture)
        try:
            with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache), \
                 contextlib.redirect_stdout(io.StringIO()):
                suite = fast.run_question_probes(r, config=self.cfg)
            self.assertEqual(set(suite.reports), {'original', 'question_withheld', 'mismatched_question'})
            self.assertTrue(any(all(text == 'Question withheld.' for text in texts) for texts, _ in seen))
            self.assertTrue(all(set(kinds) == set(fa.KINDS) for _, kinds in seen))
            for key, value in before.items():
                torch.testing.assert_close(m.GEVIS_MODEL.state_dict()[key], value)
            count = len(m.GEVIS_MODEL._forward_pre_hooks)
            with patch.object(fa, 'fusion_features', side_effect=ValueError('probe failure')):
                with self.assertRaisesRegex(ValueError, 'probe failure'):
                    fast.score_cases(r, [dict(current=None, reference=None, step='phase', centers=['a'], position=0)],
                                     probe='question_withheld')
            self.assertEqual(len(m.GEVIS_MODEL._forward_pre_hooks), count)
        finally:
            handle.remove()

    def test_generalization_executes_real_head_without_fitting_and_excludes_ambiguous_targets(self):
        r = ready()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache), \
             patch.object(fa, 'fit_readout', side_effect=AssertionError('No fitting')), \
             contextlib.redirect_stdout(io.StringIO()):
            suite = fast.run_generalization(r, config=self.cfg)
        self.assertEqual(set(suite.reports), set(fast.CHALLENGES))
        self.assertEqual(set(suite.reports['amplitude'].summary.answer_type), {'BOOLEAN', 'CHOICE'})
        self.assertEqual(set(suite.reports['compound'].summary.answer_type), {'BOOLEAN'})
        self.assertTrue(suite.reports['amplitude'].diagnostics['unscored_answers'].answer_type.eq('ORDINAL').all())
        self.assertFalse(suite.metadata['deployed_weights_updated'])
        self.assertTrue(np.isfinite(suite.summary.macro_f1).all())

    def test_feature_masks_have_equal_dimensions_and_explicit_position_only_control(self):
        for view in fast.FEATURE_VIEWS:
            mask = fast.feature_mask(66, view)
            self.assertEqual(len(mask), 66)
            self.assertEqual(mask[-2:].tolist(), [1., 1.])
        selected = fast.feature_mask(66, 'selected_numeric_only')
        self.assertEqual(selected[:32].sum(), 0)
        self.assertEqual(selected[32:48].sum(), 16)
        self.assertEqual(fast.feature_mask(66, 'position_only').sum(), 2)
        with self.assertRaises(ValueError):
            fast.feature_mask(65, 'full_retrained')

    def test_feature_controls_use_identical_train_validation_and_preserve_deployed_head(self):
        r = ready(); m = r.module
        before = {k: v.clone() for k, v in m.FUSION_ANSWER_HEAD.state_dict().items()}
        real_fit = fa.fit_readout
        fit_inputs = []
        def fit(train, validation, config, device):
            fit_inputs.append(([(e['run_id'], e['answer_type'], e['label']) for e in train],
                               [(e['run_id'], e['answer_type'], e['label']) for e in validation]))
            self.assertTrue(all(e['split'] == 'train' for e in train))
            self.assertTrue(all(e['split'] == 'validation' for e in validation))
            return real_fit(train, validation, config, device)
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache), \
             patch.object(fa, 'fit_readout', side_effect=fit), contextlib.redirect_stdout(io.StringIO()):
            suite = fast.run_feature_controls(r, config=self.cfg)
        self.assertEqual(len(fit_inputs), len(fast.FEATURE_VIEWS))
        self.assertTrue(all(data == fit_inputs[0] for data in fit_inputs))
        self.assertEqual(suite.diagnostics['training'].parameters.nunique(), 1)
        self.assertTrue(suite.diagnostics['training'].optimizer_updates.gt(0).all())
        self.assertIn('deployed_reference', suite.reports)
        for key, value in before.items():
            torch.testing.assert_close(m.FUSION_ANSWER_HEAD.state_dict()[key], value)

    def test_control_run_leakage_is_rejected_before_fitting(self):
        r = ready()
        def leaking(*args, **kwargs):
            rows = source_rows(kwargs['split'])
            for row in rows:
                row['run_id'] = 'same-run'
            return rows
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=leaking), \
             patch.object(fa, 'fit_readout', side_effect=AssertionError('No fitting')), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'run overlap'):
                fast.run_feature_controls(r, config=self.cfg)

    def test_original_review_and_plot_use_real_model_and_do_not_invent_truth(self):
        r = ready()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache):
            report, cases = fast.run_original_review(r, config=self.cfg, center_batch_size=1)
        self.assertFalse(report.summary.accuracy_available.iloc[0])
        self.assertNotIn('label', report.details)
        self.assertEqual(set(report.details.center_sensor), {'a', 'b'})
        before = cases[0]['current'].copy()
        fig, inspection = fast.plot_review(r, cases[0], 'a')
        self.assertFalse(inspection['truth_available'])
        self.assertEqual(inspection['attention'].shape[:2], (3, 2))
        self.assertEqual(len(inspection['answers']), 3)
        np.testing.assert_array_equal(cases[0]['current'], before)
        plt.close(fig)

    def test_benchmark_executes_real_forward_reports_scope_and_cpu_memory_unavailable(self):
        r = ready()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache), \
             patch.object(fa, 'fit_readout', side_effect=AssertionError('No fitting')):
            report = fast.benchmark(r, config=self.cfg, repeats=2, warmup=1)
        self.assertEqual(report.summary.measured_windows.iloc[0], 2)
        self.assertGreater(report.summary.mean_ms_per_window.iloc[0], 0)
        self.assertEqual(report.details.answers.tolist(), [3, 3])
        self.assertTrue(report.diagnostics['device_memory'].empty)
        self.assertIn('excludes disk', report.metadata['timing_scope'])

    def test_missing_or_wrong_model_fails_before_loading_inputs(self):
        for mode in ('fusion_answers', 'wrong'):
            r = runtime(); r.module.DECISION_SYSTEM = mode
            with patch('pecvd.experiments.discovery_calibration.cached_comparisons',
                       side_effect=AssertionError('No private input reads')):
                with self.assertRaises(RuntimeError):
                    fast.run_question_probes(r, config=self.cfg)

    def test_all_three_actual_research_notebooks_run_their_real_fast_experiment_paths(self):
        paths = sorted((ROOT / 'notebooks/research').glob('*.ipynb'))
        self.assertEqual(len(paths), 3)
        for path in paths:
            with self.subTest(notebook=path.name):
                r = ready(); env = {'display': lambda value: None, 'RUN_RECONSTRUCTION_CONTROLS': False}
                r.encoders['phase'][1].update(mean=np.zeros(2), std=np.ones(2))
                @contextlib.contextmanager
                def synthetic_train_cache(path, **kwargs):
                    self.assertEqual(Path(path), Path('synthetic/windows/phase.npz'))
                    yield dict(train=np.stack([np.zeros((8, 2))]*4),
                        train_process_id=np.array(['train-0','train-1','train-2','train-3']),
                        train_cohort_key=np.array(['recipe']*4), train_position_bin=np.arange(4),
                        validation_process_id=np.array(['validation-0']), test_process_id=np.array(['test-0']))
                settings = dict(decision_system='fusion_answers', training_profile='center_relation_v14')
                with patch('importlib.reload', side_effect=lambda module: module), \
                     patch('pecvd.experiments.pipeline_settings.configure_pipeline', return_value=settings), \
                     patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
                     patch('pecvd.experiments.evaluation.load_experiment_runtime', return_value=r), \
                     patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cache), \
                     patch('pandas.read_csv', side_effect=AssertionError('No CSV reads')), \
                     patch('torch.load', side_effect=AssertionError('No private checkpoint reads in synthetic Run All')), \
                     patch('numpy.load', side_effect=synthetic_train_cache), \
                     patch('pecvd.experiments.threshold_comparison.prepare_runtime_comparison',
                           side_effect=AssertionError('No numerical fallback')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    doc = json.loads(path.read_text(encoding='utf-8'))
                    for cell in doc['cells']:
                        if cell['cell_type'] == 'code':
                            exec(compile(''.join(cell['source']), path.name, 'exec'), env)
                            if cell['id'] == 'setup':
                                env['CONFIG'] = self.cfg
                                env['CONTROL_CONFIG'] = self.cfg
                                env['BENCHMARK_CONFIG'] = self.cfg
                                env['REVIEW_CONFIG'] = self.cfg
                                env['BENCHMARK_REPEATS'] = 2; env['WARMUP_WINDOWS'] = 1
                self.assertIn('SUMMARY', env)
                self.assertFalse(env['SUMMARY'].empty)

    def test_active_exp_sources_have_no_retired_model_or_rule_cells_and_clear_outputs(self):
        for path in [*(ROOT / 'notebooks/research').glob('*.ipynb'), ROOT/'notebooks/ALD_evaluate.ipynb', ROOT/'notebooks/CVD_predict_external_csv.ipynb', ROOT/'notebooks/ALD_predict_external_csv.ipynb']:
            doc = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(doc['cells']), len({c['id'] for c in doc['cells']}))
            code = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code')
            for forbidden in ('gmm', 'threshold_comparison', 'run_baseline_experiment',
                              'run_transfer_experiment', 'run_foundation_audit', 'QUESTION_VARIANTS'):
                self.assertNotIn(forbidden, code)
            for cell in doc['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
                    self.assertFalse(cell['outputs'])
                    self.assertIsNone(cell['execution_count'])
