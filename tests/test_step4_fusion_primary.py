"""Execute the actual STEP4 Run All using invented windows and a real fast model."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import fusion_answers as fa
from tests.test_fusion_answers import runtime, source_rows


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/CVD_evaluate.ipynb'


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def sources():
    return {c['id']: ''.join(c['source']) for c in notebook()['cells'] if c['cell_type'] == 'code'}


class Step4FusionPrimaryTests(unittest.TestCase):
    def test_active_notebook_has_visible_model_sections_and_no_old_comparator_code(self):
        doc = notebook()
        text = '\n'.join(''.join(c['source']) for c in doc['cells'])
        for marker in ('gmm', 'q99', 'reference_quantiles', 'threshold_comparison',
                       'run_question_experiment', 'RUN_DISCOVERY_EVALUATION',
                       'RUN_TYPED_QUESTION_AUDIT', 'require_routing_runtime'):
            self.assertNotIn(marker.lower(), text.lower())
        ids = [c['id'] for c in doc['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(next(c['id'] for c in doc['cells'] if c['cell_type'] == 'code'),
                         'shared-pipeline-settings')
        for guide, code in [('fusion-test-guide', 'fusion-answer-test'),
                            ('fusion-diagnostics-guide', 'fusion-diagnostics'),
                            ('fusion-errors-guide', 'fusion-errors'),
                            ('fusion-original-guide', 'fusion-original-predictions')]:
            self.assertEqual(ids.index(guide) + 1, ids.index(code))
        self.assertIn('Fast model — Test', text)
        self.assertNotIn('run_gevis_predictions', sources()['fusion-answer-test'])
        for source in sources().values():
            ast.parse(source)

    def test_actual_runall_scores_fast_head_and_displays_errors_without_training_or_rules(self):
        r = runtime(); m = r.module
        x, _ = fa.fusion_features(m, np.zeros((8, 2)), np.zeros((8, 2)), 'phase', ['a'])
        m.FUSION_ANSWER_HEAD = fa.ContextReadout(x.shape[-1], 16).eval()
        # Deterministic wrong answers ensure the real error-display branch executes.
        with torch.no_grad():
            for parameter in m.FUSION_ANSWER_HEAD.parameters():
                parameter.zero_()
        before = {k: v.clone() for k, v in m.FUSION_ANSWER_HEAD.state_dict().items()}
        frame = pd.DataFrame(np.zeros((8, 2)), columns=['a', 'b'])
        env = {}; shown = []; env['display'] = shown.append
        splits = []

        def cached(*args, **kwargs):
            splits.append(kwargs['split'])
            return source_rows(kwargs['split'])

        settings = dict(decision_system='fusion_answers', training_profile='center_relation_v14',
                        model_variant='multiscale')
        with patch('importlib.reload', side_effect=lambda module: module), \
             patch('pecvd.experiments.pipeline_settings.configure_pipeline', return_value=settings), \
             patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
             patch('pecvd.experiments.evaluation.load_experiment_runtime', return_value=r), \
             patch('pecvd.experiments.discovery_calibration.cached_comparisons', side_effect=cached), \
             patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',
                   return_value=[(frame, frame.copy())]) as originals, \
             patch.object(fa, 'fit_readout', side_effect=AssertionError('No Test fitting')), \
             patch('pecvd.experiments.threshold_comparison.prepare_runtime_comparison',
                   side_effect=AssertionError('No numerical policy preparation')), \
             patch('pecvd.experiments.window_policy.WindowPolicy.analyze',
                   side_effect=AssertionError('No numerical decisions')), \
             patch('pecvd.experiments.inference_benchmark.pecvd_cases',return_value=[{'step':'phase'}]), \
             patch('pecvd.experiments.inference_benchmark.benchmark',return_value=SimpleNamespace(
                 summary=pd.DataFrame(),details=pd.DataFrame(),metadata={'domain':'PECVD'},
                 diagnostics={'requests':pd.DataFrame([dict(status='OK',request_ms=2.,answers=6,selected_sensors=2)])})), \
             patch('pandas.read_csv', side_effect=AssertionError('No CSV reads')), \
             contextlib.redirect_stdout(io.StringIO()):
            for cell_id,source in sources().items():
                exec(compile(source, 'STEP4:actual-fusion-runall', 'exec'), env)
                if cell_id=='reference-config':
                    env['SHOW_DETAILS']=True
                    env['RUN_ORIGINAL_PREDICTIONS']=True
        self.assertEqual(set(splits), {'test'})
        self.assertEqual(originals.call_args.kwargs['split'], 'test')
        report = env['REPORT']
        self.assertIs(report, env['FUSION_TEST'])
        self.assertEqual(report.metadata['split'], 'test')
        self.assertFalse(report.metadata['test_used_for_fitting'])
        self.assertFalse(report.metadata['rule_fallback'])
        self.assertEqual(set(report.summary.answer_type), set(fa.KINDS))
        self.assertFalse(report.summary.isna().any().any())
        for key in ('per_class', 'boolean_metrics', 'ordinal_metrics', 'confusion'):
            self.assertTrue(any(isinstance(table,pd.DataFrame) and table.equals(report.diagnostics[key].head(50)) for table in shown), key)
        self.assertIn('expected_answer', env['ERROR_PREVIEW'])
        self.assertIn('predicted_answer', env['ERROR_PREVIEW'])
        self.assertNotIn('Reconstruction error', set(env['ERROR_PREVIEW'].expected_answer)
                         | set(env['ERROR_PREVIEW'].predicted_answer))
        self.assertFalse(env['FUSION_ORIGINAL'].summary.accuracy_available.iloc[0])
        self.assertGreater(env['TEST_EVALUATION_SECONDS'], 0.)
        self.assertIn('TIMING_ASSESSMENT',env)
        self.assertEqual(env['TIMING_ASSESSMENT']['request_latency'].iloc[0].mean_ms,2.)
        self.assertEqual(env['TEST_CONFIG'].max_windows_per_step, env['TEST_WINDOWS_PER_STEP'])
        for key, value in before.items():
            torch.testing.assert_close(m.FUSION_ANSWER_HEAD.state_dict()[key], value)

    def test_wrong_decision_system_and_missing_head_fail_instead_of_skipping_test(self):
        code = sources()
        with self.assertRaisesRegex(RuntimeError, 'fusion_answers'):
            exec(code['reference-config'], {'PIPELINE_SETTINGS': {'decision_system': 'hybrid'}})
        for mode, message in [('hybrid', 'not fusion_answers'), ('fusion_answers', 'head missing')]:
            r = runtime(); r.module.DECISION_SYSTEM = mode
            with patch('importlib.reload', side_effect=lambda module: module), \
                 patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
                 patch('pecvd.experiments.evaluation.load_experiment_runtime', return_value=r), \
                 self.assertRaisesRegex(RuntimeError, message):
                exec(code['reference-load'], {'ROOT': ROOT, 'pd': pd, 'display': lambda _: None})

    def test_original_predictions_are_optional_and_cannot_replace_test_metrics(self):
        code = sources()['fusion-original-predictions']
        with patch('pecvd.experiments.gevis_system.run_gevis_predictions',
                   side_effect=AssertionError('Original predictions disabled')):
            exec(code, {'RUN_ORIGINAL_PREDICTIONS': False})

