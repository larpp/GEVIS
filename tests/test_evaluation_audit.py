"""Invented clustered records and TRAIN caches; no private-data reads."""
import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import evaluation_audit as audit, online_inference as online
from pecvd.experiments import fusion_answers as fa, inference_benchmark as timing
from tests.test_fusion_answers import runtime, source_rows


class EvaluationAuditTests(unittest.TestCase):
    def test_cluster_intervals_do_not_treat_duplicate_questions_as_independent_runs(self):
        frame = pd.DataFrame([dict(run_id=str(r), answer_type='BOOLEAN', label=y, prediction=p)
            for r, pairs in enumerate([[(0, 0), (1, 1)], [(0, 1), (1, 0)], [(0, 0), (1, 0)]])
            for y, p in pairs])
        first = audit.run_bootstrap(frame, repeats=200)
        duplicate = audit.run_bootstrap(pd.concat([frame]*10), repeats=200)
        np.testing.assert_allclose(first[['estimate', 'lower', 'upper']],
                                   duplicate[['estimate', 'lower', 'upper']])
        self.assertEqual(set(first.source_runs), {3})
        self.assertTrue((first.upper > first.lower).any())
        one = audit.run_bootstrap(frame.loc[frame.run_id.eq('0')])
        self.assertTrue(one.lower.isna().all())
        self.assertEqual(set(one.status), {'unavailable'})

    def test_unlabelled_originals_get_alarm_rates_not_false_positive_rates(self):
        frame = pd.DataFrame([dict(step='p', answer_type='BOOLEAN', prediction=p) for p in (0, 1, 1)])
        row = audit.original_alarm_rates(frame).iloc[0]
        self.assertEqual(row.alarms, 2)
        self.assertAlmostEqual(row.alarm_rate, 2/3)
        self.assertFalse(row.verified_normal)
        self.assertNotIn('false_positive_rate', row.index)

    def test_pecvd_session_matches_frozen_cache_reference_and_benchmark_all_centers(self):
        r = runtime(); m = r.module
        meta = m.STEP_ENCODERS['phase'][1]
        meta.update(mean=np.zeros(2), std=np.ones(2))
        x, _ = fa.fusion_features(m, np.zeros((8, 2)), np.zeros((8, 2)), 'phase', ['a'])
        m.FUSION_ANSWER_HEAD = fa.ContextReadout(x.shape[-1], 16).requires_grad_(False).eval()
        m.FUSION_ANSWER_TRAINING = dict(trained_steps=['phase'], train_runs=['t1'], validation_runs=['v'])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'windows').mkdir()
            m.ENCODER_ROOT = root
            (root/'encoder_manifest.json').write_text('{}', encoding='utf-8')
            payload = dict(train=np.stack([np.zeros((8, 2)), np.ones((8, 2))*2]),
                train_process_id=np.array(['t1', 't2']), train_cohort_key=np.array(['recipe']*2),
                train_position_bin=np.array([0, 0]), validation_process_id=np.array(['v']),
                test_process_id=np.array(['new']))
            np.savez(root/'windows/phase.npz', **payload)
            runner = online.InferenceSession(r)
            original = m.numeric_state
            seen = []
            def state(encoder, current, reference):
                seen.append(reference.detach().cpu().numpy())
                return original(encoder, current, reference)
            with patch.object(m, 'numeric_state', side_effect=state):
                with self.assertRaisesRegex(ValueError, 'structured'):
                    runner.process_window(current=np.zeros((8, 2)), step='phase', cohort='recipe', run_id='new')
                result = runner.process_window(current=np.zeros((8, 2)), step='phase',
                    cohort=('recipe', 0), run_id='new')
                self.assertEqual(len(result['answers']), 6)
                np.testing.assert_array_equal(seen[-1], np.ones((1, 8, 2)))
                runner.process_window(current=np.zeros((8, 2)), step='phase',
                    cohort=('recipe', 0), run_id='t1')
                np.testing.assert_array_equal(seen[-1], np.full((1, 8, 2), 2))
            self.assertEqual(audit.split_audit(r).partition_overlap_count.sum(), 0)
            with patch('pecvd.experiments.discovery_calibration.cached_comparisons',
                       return_value=[dict(source_rows('test')[0], run_id='new')]):
                cases = timing.pecvd_cases(r)
            progress = []
            report = timing.benchmark(r, cases, repeats=2, warmup=1, domain='PECVD', session=runner,
                                      progress=progress.append)
            self.assertEqual(progress[-1], 'Measured requests: 2/2')
            self.assertIn('Warmup Step groups: 1/1', progress)
            self.assertEqual(set(report.details.answers), {6})
            self.assertEqual(int(report.summary.measured_windows.iloc[0]), 2)
            payload['test_process_id'] = np.array(['t1'])
            np.savez(root/'windows/phase.npz', **payload)
            self.assertGreater(audit.split_audit(r).partition_overlap_count.sum(), 0)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                online.InferenceSession(r)

    def test_missing_truth_or_run_identity_does_not_invent_intervals(self):
        self.assertEqual(audit.run_bootstrap(pd.DataFrame()).status.iloc[0], 'unavailable')

    def test_positive_only_questions_have_no_measurable_false_positive_rate(self):
        from pecvd.experiments.reporting import ExperimentReport
        report = ExperimentReport('Invented positive-only answers', pd.DataFrame(),
            details=pd.DataFrame([dict(answer_type='BOOLEAN', label=1, prediction=1)]),
            metadata=dict(decision_system='fusion_answers'))
        row = report.diagnostics['boolean_metrics'].iloc[0]
        self.assertFalse(row.fpr_available)
        self.assertTrue(np.isnan(row.false_positive_rate))

    def test_multiscale_results_warn_when_initial_weights_were_selected(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        root = Path(__file__).resolve().parents[1]
        doc = json.loads((root/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_bytes())
        code = ''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='01-results'))
        ck = dict(parameter_count=1, encoder_backend='gevis_native_multiscale_blind_v3',
            history=[dict(epoch=0, train_mse=1., validation_mse=1.),
                     dict(epoch=1, train_mse=.8, validation_mse=.9999)],
            training_status=dict(best_epoch=0, improved_over_initial=False),
            test_metrics=dict(blind_mse=1., zero_baseline_mse=2., relative_mse=.5), elapsed_seconds=1.)
        shown = []; output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(plt,'show'):
            exec(code,dict(encoder_checkpoints={'invented':ck},np=np,pd=pd,plt=plt,display=shown.append))
        row = next(t for t in shown if 'encoder_backend' in t).iloc[0]
        self.assertEqual(shown[0].best_epoch.iloc[0], 0)
        self.assertEqual(row.best_epoch, 0)
        self.assertFalse(row.improved_over_initial)
        self.assertIn('initial epoch selected', output.getvalue())
        plt.close('all')

