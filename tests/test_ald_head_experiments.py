"""ALD schema/valve contract and all three heads, on invented data."""
import ast
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from pecvd.gevis import GEVIS
from pecvd.experiments import ald_preprocessing as p, ald_head_experiments as e
from pecvd.experiments import decision_training as d, current_input as current
from tests.test_decision_training import fixture, ROOT


class AldPreprocessingTests(unittest.TestCase):
    def test_all_schema_widths_are_accepted_using_common_named_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, columns in [('a', ['pressure', 'ValveAct_1:1', 'old']),
                                  ('b', ['pressure', 'ValveAct_1:1', 'new', 'extra']),
                                  ('c', ['ValveAct_1:1', 'pressure'])]:
                pd.DataFrame(columns=columns).to_csv(Path(directory) / (name + '.csv'), index=False)
            columns, audit = p.schema_inventory(directory)
            self.assertEqual(columns, ('ValveAct_1:1', 'pressure'))
            self.assertTrue(audit.included.all())
            self.assertEqual(len(audit), 3)

    def test_data_complete_defaults_do_not_exclude_months_or_recipes(self):
        config = p.NativeConfig()
        config.validate()
        self.assertEqual(config.excluded_months, ())
        self.assertEqual(config.recipe_suffix, '')

    def test_schema_requires_matching_names_and_allows_reordered_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, columns in [('a', ['x', 'y']), ('b', ['y', 'x']), ('c', ['x', 'z']), ('d', ['x'])]:
                pd.DataFrame(columns=columns).to_csv(Path(directory) / (name + '.csv'), index=False)
            columns, audit = p.schema_inventory(directory, p.PreprocessingConfig(schema_columns=2))
            self.assertEqual(columns, ('x', 'y'))
            self.assertEqual(audit.loc[audit.included, 'file'].tolist(), ['a.csv', 'b.csv'])

    def test_valves_numeric_strings_and_live_scaling(self):
        frame = pd.DataFrame({'ValveAct_1:1': [' OPEN ', 'close', 1., 0], 'pressure': ['1', '3', 5., 7.]})
        meta = dict(sensor_columns=list(frame), mean=[0., 1.], std=[1., 2.])
        values = p.transform_live(frame, meta)
        np.testing.assert_array_equal(values, [[1, 0], [0, 1], [1, 2], [0, 3]])
        self.assertEqual(frame.iloc[0, 0], ' OPEN ')

    def test_unknown_text_missing_nonfinite_are_not_normal_states(self):
        for data in ({'ValveAct_1:1': ['MOVING']}, {'ValveAct_1:1': [None]},
                     {'ValveAct_1:1': [2.]}, {'pressure': ['bad']}, {'pressure': [np.inf]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                p.encode_frame(pd.DataFrame(data), list(data))
        with self.assertRaisesRegex(ValueError, 'Missing'):
            p.encode_frame(pd.DataFrame({'x': [1]}), ['missing'])

    def test_metadata_and_ids_are_not_sensor_candidates(self):
        candidates = p.sensor_candidates(['Timestamp', 'ActiveCjobID', 'PPExecStepName',
            'pressure', 'ValveAct_1:1', 'ASCII(name)', 'Unnamed: 0'], p.PreprocessingConfig())
        self.assertEqual(candidates, ['pressure', 'ValveAct_1:1'])

    def test_notebooks_are_valid_and_no_project_copy_is_hardcoded(self):
        for name in ('CVD_evaluate', 'ALD_evaluate'):
            notebook = json.loads((ROOT / 'notebooks' / (name + '.ipynb')).read_text(encoding='utf-8'))
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source']); ast.parse(source)
                    self.assertNotIn('C:\\Projects', source)
                    # Local outputs are user results and must be preserved.
                    # Source-only export sanitization has its own regression test.
            if name == 'ALD_evaluate':
                constants = [node.value for c in notebook['cells'] if c['cell_type'] == 'code'
                    for node in ast.walk(ast.parse(''.join(c['source'])))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
                self.assertFalse(any('discover_package' in value for value in constants))
                self.assertFalse(any('???' in value for value in constants))


class AldHeadExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_sigmoid_is_independent_and_does_not_normalize_to_one(self):
        head = e.ExperimentHead(4, 4, e.ChoiceConfig(mode='sigmoid'))
        probabilities = head.choice_probabilities(torch.full((1, 6), 2.))
        self.assertTrue((probabilities > .5).all())
        self.assertGreater(float(probabilities.sum()), 4.)

    def test_grouped_loss_really_accepts_either_member_and_masks_normal(self):
        head = e.ExperimentHead(4, 4, e.ChoiceConfig(mode='grouped'))
        logits = torch.tensor([[-10., 3., -1., 2., -1., -1.]], requires_grad=True)
        weights = {'CHOICE': None}
        first = e._loss(logits, torch.tensor([1]), 'CHOICE', head, weights)
        third = e._loss(logits, torch.tensor([3]), 'CHOICE', head, weights)
        self.assertEqual(float(first.detach()), float(third.detach()))
        ignored = e._loss(logits, torch.tensor([-1]), 'CHOICE', head, weights)
        ignored.backward(); self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_invalid_groups_and_thresholds_are_rejected(self):
        for config in [e.ChoiceConfig(groups=((1, 2), (3, 4))),
                       e.ChoiceConfig(sigmoid_threshold=0.), e.ChoiceConfig(mode='invalid')]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_continuous_injections_do_not_target_binary_valves(self):
        cases = [dict(injected=False, centers=['ValveAct_1:1'], targets=np.array([[0, 0, 0]])),
                 dict(injected=True, centers=['ValveAct_1:1'], targets=np.array([[1, 1, 1]])),
                 dict(injected=True, centers=['pressure'], targets=np.array([[1, 1, 1]]))]
        kept, audit = e._continuous_interventions(cases, [dict(center_sensor='ValveAct_1:1', generated=1)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['centers'], ['pressure'])
        self.assertEqual(audit[0]['status'], 'excluded_binary_valve_intervention')

    def test_context_is_retained_in_input_but_never_in_supervision(self):
        values = np.array([[1., 0.], [2., 1.]], dtype=np.float32)
        case = dict(step='phase', injected=False, current=values,
            sensor_columns=['pressure', 'ValveAct_1:1'], centers=['pressure', 'ValveAct_1:1'],
            targets=np.array([[0, 0, 0], [1, -1, -1]]))
        kept, _ = e._continuous_interventions([case], [], {'phase': ['pressure']})
        self.assertEqual(kept[0]['centers'], ['pressure'])
        np.testing.assert_array_equal(kept[0]['current'], values)
        self.assertEqual(kept[0]['targets'].shape, (1, 3))

    def test_context_role_is_visible_before_training(self):
        prepared = type('Prepared', (), {'metadata': {'phase': dict(
            sensor_columns=['pressure', 'ValveAct_1:1'], context_sensor_columns=['ValveAct_1:1'])}})()
        roles = p.sensor_roles(prepared).set_index('sensor')
        self.assertEqual(roles.loc['pressure', 'role'], 'prediction_target')
        self.assertEqual(roles.loc['ValveAct_1:1', 'role'], 'context_only')

    def test_reused_preparation_cannot_hide_changed_native_policy(self):
        r = fixture(); gevis = GEVIS(ROOT, r, ['phase'])
        r.prepared.identity.update(preprocessing=asdict(p.PreprocessingConfig()),
                                   native_config=asdict(current.Config()))
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'Run/recipe policy'):
            e.run_comparison(directory, gevis, encoder_step='phase', prepared=r.prepared,
                config=r.config, native_config=replace(current.Config(), recipe_suffix='OTHER'), progress=None)

    def test_three_heads_fit_save_restore_test_and_preserve_frozen_source(self):
        r = fixture(); gevis = GEVIS(ROOT, r, ['phase'])
        prepared = copy.deepcopy(r.prepared)
        policy = p.PreprocessingConfig()
        prepared.identity['preprocessing'] = asdict(policy)
        config = replace(r.config, epochs=1)
        before = current._state_digest(gevis.module.GEVIS_MODEL)
        original_windows = d._hash(prepared.windows)
        with tempfile.TemporaryDirectory() as directory:
            result = e.run_comparison(directory, gevis, encoder_step='phase', prepared=prepared,
                preparation_policy=policy, config=config, progress=None)
            self.assertEqual(set(result.summary['mode']), {'softmax', 'grouped', 'sigmoid'})
            self.assertEqual(result.summary.questions.nunique(), 1)
            self.assertTrue(np.isfinite(result.summary.choice_common_group_macro_f1).all())
            self.assertTrue(pd.isna(result.summary.set_index('mode').loc['grouped', 'choice_exact_macro_f1']))
            for mode, path in result.checkpoints.items():
                runtime = e.load_head(directory, path, gevis=gevis)
                self.assertEqual(runtime.module.FUSION_ANSWER_HEAD.choice.mode, mode)
                self.assertGreater(runtime.module.FUSION_ANSWER_TRAINING['optimizer_updates'], 0)
                record = runtime.references['records'][0]
                step, recipe, loop, offset = record['key']
                request = dict(current=(record['lower'] + record['upper']) / 2, step=step,
                    recipe_key=recipe, loop_key=loop, offset_samples=offset,
                    lineage_id='independent-test-request', standardized=True)
                reply = e.Session(runtime).process_window(**request)
                self.assertEqual(reply['status'], 'OK')
                raw = dict(request, current=pd.DataFrame(request['current'], columns=runtime.encoders[step][1]['sensor_columns']), standardized=False)
                self.assertEqual(reply['answers'], e.Session(runtime).process_window(**raw)['answers'])
                # Decoder/gate behavior survives checkpoint roundtrip exactly.
                head = runtime.module.FUSION_ANSWER_HEAD
                with torch.no_grad():
                    head.answers['BOOLEAN'].weight.zero_(); head.answers['BOOLEAN'].bias.copy_(torch.tensor([10., -10.]))
                no = e.Session(runtime).process_window(**request)
                self.assertTrue(all(a['status'] == 'NOT_APPLICABLE' for a in no['answers'] if a['answer_type'] != 'BOOLEAN'))
                with torch.no_grad():
                    head.answers['BOOLEAN'].bias.copy_(torch.tensor([-10., 10.]))
                yes = e.Session(runtime).process_window(**request)
                self.assertTrue(all(a['evaluated'] for a in yes['answers']))
                if mode == 'sigmoid':
                    with torch.no_grad():
                        head.answers['CHOICE'].weight.zero_(); head.answers['CHOICE'].bias.fill_(-10.)
                    no_type = e.Session(runtime).process_window(**request)
                    choice = next(a for a in no_type['answers'] if a['answer_type'] == 'CHOICE')
                    self.assertEqual(choice['detected_types'], [])
                    self.assertIsNone(choice['answer'])
            self.assertFalse((Path(directory) / d.pointer('PEALD')).exists())
            self.assertEqual(before, current._state_digest(gevis.module.GEVIS_MODEL))
            self.assertEqual(original_windows, d._hash(prepared.windows))


if __name__ == '__main__':
    unittest.main()
