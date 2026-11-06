"""ALD preprocessing memory regression; invented files only, no model training."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pecvd.experiments import ald_input_store as store, ald_preprocessing as prep
from pecvd.experiments.peald_current_data import COLUMNS


class AldInputStoreTests(unittest.TestCase):
    def test_public_preparation_does_not_discover_external_packages(self):
        from pecvd import gevis_training as training
        gevis = SimpleNamespace(encoders={'DEP.': (SimpleNamespace(patch_len=4), {})})
        with (
            patch.object(training, '_prepare_ald', return_value='prepared') as prepare,
            patch('pecvd.experiments.ald_delivery_evaluation.discover_package', side_effect=AssertionError('No external Test lookup')),
        ):
            self.assertEqual(training.prepare_ald('invented-root', gevis), 'prepared')
            self.assertNotIn('delivery_package', prepare.call_args.kwargs)

    def test_preparation_using_external_membership_cannot_be_reused_for_training(self):
        from pecvd.experiments import ald_head_experiments as experiment
        gevis = SimpleNamespace(encoders={'DEP.': (None, {})}, module=SimpleNamespace())
        prepared = SimpleNamespace(identity=dict(preprocessing=asdict(prep.PreprocessingConfig()),
                                                delivery_metadata_identity={'forbidden': True}))
        with self.assertRaisesRegex(ValueError, 'external Test membership'):
            experiment.run_comparison('invented-root', gevis, prepared=prepared, progress=None)

    def test_mapping_preserves_numeric_text_missing_and_native_rows(self):
        frame = pd.DataFrame({'x': ['1.25', 'bad', None, '4'], 'ValveAct_1:1': [0., 1., 0., 1.]})
        with tempfile.TemporaryDirectory() as directory:
            disk = store.stage_frame(frame, list(frame), directory, source_sha256='invented')
            self.assertIsInstance(disk.values, np.memmap)
            self.assertFalse(disk.values.flags.writeable)
            self.assertEqual(disk.iloc[1:3].rows.tolist(), [1, 2])
            original = pd.to_numeric(frame['x'], errors='coerce')
            restored = pd.to_numeric(disk['x'], errors='coerce')
            np.testing.assert_allclose(original, restored, equal_nan=True)
            np.testing.assert_array_equal(frame['x'].notna() & original.isna(),
                                          disk['x'].notna() & restored.isna())
            np.testing.assert_array_equal(disk.iloc[2:4][['ValveAct_1:1']].to_numpy(), [[0.], [1.]])
            del disk

    def test_exact_cache_reuse_does_not_rewrite_files(self):
        frame = pd.DataFrame({'x': np.arange(8, dtype=np.float32)})
        with tempfile.TemporaryDirectory() as directory:
            first = store.stage_frame(frame, ['x'], directory, source_sha256='invented')
            before = {p.name: p.stat().st_mtime_ns for p in Path(directory).iterdir()}
            second = store.stage_frame(frame, ['x'], directory, source_sha256='invented')
            self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in Path(directory).iterdir()})
            np.testing.assert_array_equal(first['x'], second['x'])
            del first, second

    def test_changed_source_uses_a_different_namespace(self):
        frame = pd.DataFrame({'x': [1., 2.]})
        with tempfile.TemporaryDirectory() as directory:
            first = store.stage_frame(frame, ['x'], directory, source_sha256='source-one')
            second = store.stage_frame(frame + 1, ['x'], directory, source_sha256='source-two')
            self.assertNotEqual(first.values.filename, second.values.filename)
            np.testing.assert_array_equal(second['x'], [2., 3.])
            del first, second

    def test_preparation_matches_memory_reference_under_small_working_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / 'data/mt_nit/data_complete'; data.mkdir(parents=True)
            for job in range(14):
                n = 16
                frame = pd.DataFrame({name: ['1'] * n for name in set(COLUMNS.values())})
                frame[COLUMNS['run']] = f'job-{job}'
                frame[COLUMNS['order']] = pd.date_range(pd.Timestamp('2025-03-01') + pd.Timedelta(days=job), periods=n, freq='s')
                frame[COLUMNS['cohort']] = 'MainRecipe/RUN/TEST'
                frame[COLUMNS['execution']] = 'MainRecipe/RUN/TEST'
                frame[COLUMNS['parent_step']] = 'DEPO'
                frame[COLUMNS['step']] = 'DEP'
                frame['pressure'] = np.sin(np.arange(n) / 3) + job * .01
                frame['ValveAct_1:1'] = ['OPEN', 'CLOSE'] * (n // 2)
                frame.to_csv(data / f'{job:02}.csv', index=False)
            options = dict(config=prep.NativeConfig(), patch_length=2,
                           policy=prep.PreprocessingConfig(max_window_length=8), max_memory_mb=.05, progress=None)
            # No external package exists. Fail immediately on any external lookup.
            with (
                patch('pecvd.experiments.ald_delivery_evaluation.discover_package', side_effect=AssertionError('No external Test lookup')),
                patch('pecvd.experiments.peald_current_data.delivery_metadata_identity', side_effect=AssertionError('No external Test membership')),
            ):
                actual = prep.prepare(root, **options)
            # Same native policy using ordinary per-file frames as the reference.
            def in_memory(frame, sensors, *args, **kwargs):
                return frame[sorted(set(sensors) & set(frame.columns))]
            with patch.object(prep.input_store, 'stage_frame', side_effect=in_memory):
                expected = prep.prepare(root, **options)
            self.assertEqual(actual.split_runs, expected.split_runs)
            self.assertEqual(actual.identity['split_input_scope'], 'data_complete_only')
            self.assertNotIn('delivery_metadata_identity', actual.identity)
            self.assertNotIn('reserved_delivery_groups', actual.identity)
            self.assertEqual(set(actual.metadata), set(expected.metadata))
            for step in actual.metadata:
                for field in ('mean', 'std'):
                    np.testing.assert_array_equal(actual.metadata[step][field], expected.metadata[step][field])
                self.assertEqual(actual.metadata[step]['sensor_columns'], expected.metadata[step]['sensor_columns'])
                for split in ('train', 'validation', 'test'):
                    got = actual.windows[step][split]; wanted = expected.windows[step][split]
                    self.assertEqual(len(got), len(wanted))
                    for a, b in zip(got, wanted):
                        self.assertEqual(a['pair_id'], b['pair_id'])
                        np.testing.assert_array_equal(a['current'], b['current'])


if __name__ == '__main__':
    unittest.main()
