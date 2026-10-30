"""Invented immutable partitions and a CPU neural training/evaluation round trip."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from pecvd.experiments.tips_dataset import TipsData, sha
from pecvd.experiments.tips_model import SyntheticCases, TipsModel
from pecvd.experiments import tips_training as training
from pecvd.experiments.tips_profiles import resolve_config


def fixture(root):
    source = root / 'source'; split = root / 'split'
    source.mkdir(); split.mkdir()
    columns = root / 'columns.txt'; columns.write_text('pressure\ntemperature\nValveAct_1:1\n')
    rows = []; source_files = {}
    for rid, partition in enumerate(['train', 'train', 'valid', 'valid', 'test', 'test', 'reserved_transfer'], 1):
        start = pd.Timestamp('2020-01-01') + pd.Timedelta(days=rid)
        t = np.arange(32)
        frame = pd.DataFrame(dict(_run_id=rid, _timestamp=start + pd.to_timedelta(t, unit='s'),
            _continuous_segment=np.where(t < 16, 1, 2), _main_occurrence=1, _cycle_id='one',
            ProcessRecipeStepSeqNo='14', ProcessRecipeStepName='phase',
            pressure=np.sin(t / 3) + rid / 10, temperature=np.cos(t / 4),
            **{'ValveAct_1:1': np.where(t % 2, 'OPEN', 'CLOSE')}))
        path = source / f'{rid}.parquet'; pq.write_table(pa.Table.from_pandas(frame), path)
        source_files[path.name] = dict(sha256=sha(path))
        rows.append(dict(run_id=rid, split=partition, rows_after=len(frame), parquet_path=path.relative_to(root).as_posix()))
    (source / 'manifest.json').write_text(json.dumps(dict(files=source_files)))
    pd.DataFrame(rows).to_csv(split / 'assignments.csv', index=False)
    for partition in ['train', 'valid', 'test', 'reserved_transfer']:
        (split / partition).mkdir()
        pd.DataFrame([r for r in rows if r['split'] == partition]).to_csv(split / partition / 'runs.csv', index=False)
    files = {p.relative_to(split).as_posix(): dict(sha256=sha(p)) for p in split.rglob('*.csv')}
    (split / 'manifest.json').write_text(json.dumps(dict(files=files,
        identity=dict(source='source', source_manifest_sha256=sha(source / 'manifest.json')))))
    return dict(mode='tips_native_scratch', tips_root='.', split_dir='split', columns_file='columns.txt',
        main_steps=[14], window_scope='main', window_length=8, patch_length=4, stride=8,
        target_sensors=['pressure', 'temperature'], text_encoder='byte', encoder_layers=1,
        fusion_width=16, head_width=16, encoder_epochs=1, answer_epochs=1, batch_size=8,
        learning_rate=.001, patience=1, train_windows_per_run=1, evaluation_windows_per_run=1,
        cases_per_window=24, seed=42)


class TipsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_membership_segments_binary_and_train_only_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = fixture(root)
            data = training.data_for(cfg, root); stats = data.fit()
            self.assertEqual(stats[14]['count'], 64)
            windows = data.windows('train', 100)
            self.assertEqual(len(windows), 8)
            self.assertEqual({w['run_id'] for w in windows}, {1, 2})
            self.assertTrue(all(w['values'].shape == (8, 3) for w in windows))
            for w in windows:
                self.assertEqual((pd.Timestamp(w['end']) - pd.Timestamp(w['start'])).total_seconds(), 7)
            with self.assertRaisesRegex(ValueError, 'Reserved'):
                list(data.frames('reserved_transfer'))
            with patch.object(data, 'frames', wraps=data.frames) as reader:
                data.fit()
                self.assertEqual([c.args[0] for c in reader.call_args_list], ['train'])
            cases = SyntheticCases(windows, data.requested, cfg['target_sensors'])
            for i in range(len(cases)):
                c = cases[i]
                torch.testing.assert_close(c['current'][:, 2], c['reference'][:, 2])
                changed = bool((c['current'][:, c['center']] != c['reference'][:, c['center']]).any())
                self.assertEqual(int(c['labels'][0]), int(changed))

    def test_hash_tampering_and_unknown_valve_break_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); cfg = fixture(root)
            path = root / 'source/1.parquet'
            with path.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed source Parquet'):
                training.data_for(cfg, root)

    def test_profiles_are_the_only_selection_source_and_exclude_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            step = root / 'steps.json'; sensor = root / 'sensors.json'; dictionary = root / 'dictionary.json'
            step.write_text(json.dumps(dict(schema_version=1, include=[14, 15], exclude=[15])))
            sensor.write_text(json.dumps(dict(schema_version=1, groups=['thermal'], include=['MFC26_F.PWR'], exclude=['MFC26_F.PWR'])))
            dictionary.write_text(json.dumps([dict(name='Temp_Act_U', group='thermal'),
                dict(name='MFC26_F.PWR', group='power'), dict(name='Timestamp', group='meta')]))
            cfg = dict(tips_root='.', step_profile='steps.json', sensor_profile='sensors.json',
                       sensor_dictionary='dictionary.json')
            resolved = resolve_config(cfg, root)
            self.assertEqual(resolved['main_steps'], [14])
            self.assertEqual(resolved['selected_sensors'], ['Temp_Act_U'])
            self.assertEqual(resolved['target_sensors'], ['Temp_Act_U'])
            before = resolved['selection']['sha256']
            sensor.write_text(json.dumps(dict(schema_version=1, groups=[], include=['MFC26_F.PWR'], exclude=[])))
            self.assertNotEqual(before, resolve_config(cfg, root)['selection']['sha256'])
            with self.assertRaisesRegex(ValueError, 'sole selection authority'):
                resolve_config(dict(cfg, target_sensors=['Temp_Act_U']), root)
            sensor.write_text(json.dumps(dict(schema_version=1, groups=[], include=['Timestamp'], exclude=[])))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                resolve_config(cfg, root)

    def test_small_window_budget_still_covers_all_selected_centers(self):
        windows = [dict(values=np.arange(32, dtype=np.float32).reshape(8, 4),
                        run_id=1, step=14, position=0.)]
        cases = SyntheticCases(windows, ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'])
        centers = {int(cases[i]['center']) for i in range(len(cases))}
        self.assertEqual(centers, {0, 1, 2, 3})
        for i in range(0, len(cases), 2):
            a, b = cases[i], cases[i + 1]
            torch.testing.assert_close(a['current'], b['current'])
            torch.testing.assert_close(a['reference'], b['reference'])
            self.assertEqual(int(b['labels'][0]), 0)
        constant = SyntheticCases(windows, ['a', 'b', 'c', 'd'], ['a'],
            varying_by_step={14: [False, True, True, True]})
        self.assertTrue(all(int(constant[i]['labels'][0]) == 0 for i in range(len(constant))))

    def test_model_rejects_steps_and_sensor_grid_outside_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = fixture(Path(folder))
            model = TipsModel(['pressure', 'temperature', 'ValveAct_1:1'], cfg)
            with self.assertRaisesRegex(ValueError, 'outside the profile'):
                model(torch.zeros(1, 8, 3), torch.zeros(1, 8, 3), torch.tensor([0]),
                      torch.tensor([15]), torch.zeros(1))
            with self.assertRaisesRegex(ValueError, 'sensor grid'):
                model(torch.zeros(1, 8, 4), torch.zeros(1, 8, 4), torch.tensor([0]),
                      torch.tensor([14]), torch.zeros(1))

    def test_profile_selected_training_reload_and_changed_profile_rejection(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); cfg = fixture(root)
            for path in (root / 'source').glob('*.parquet'):
                table = pq.read_table(path)
                names = ['AUX1_MS1 ' if n == 'pressure' else 'AUX3_MS2' if n == 'temperature' else n for n in table.column_names]
                pq.write_table(table.rename_columns(names), path)
            source_manifest = root / 'source/manifest.json'
            source_manifest.write_text(json.dumps(dict(files={p.name: dict(sha256=sha(p))
                for p in (root / 'source').glob('*.parquet')})))
            split_manifest = root / 'split/manifest.json'
            document = json.loads(split_manifest.read_text())
            document['identity']['source_manifest_sha256'] = sha(source_manifest)
            split_manifest.write_text(json.dumps(document))
            (root / 'steps.json').write_text(json.dumps(dict(schema_version=1, include=[14], exclude=[])))
            sensors = root / 'sensors.json'
            sensors.write_text(json.dumps(dict(schema_version=1, groups=[], include=['AUX1_MS1'], exclude=[])))
            (root / 'dictionary.json').write_text(json.dumps([dict(name='AUX1_MS1', group='sensors'),
                dict(name='AUX3_MS2', group='sensors')]))
            for key in ['columns_file', 'main_steps', 'target_sensors']:
                cfg.pop(key)
            cfg.update(step_profile='steps.json', sensor_profile='sensors.json', sensor_dictionary='dictionary.json')
            resolved = resolve_config(cfg, root)
            data = training.data_for(resolved, root)
            output = root / 'trained'; output.mkdir()
            identity = training.experiment_identity(data, resolved)
            training.train(data, resolved, output, identity, torch.device('cpu'))
            training.evaluate(data, resolved, output, identity, torch.device('cpu'))
            details = pd.read_csv(output / 'test_predictions.csv')
            self.assertEqual(set(details.center_sensor), {'AUX1_MS1'})
            self.assertEqual(set(details.step), {14})
            sensors.write_text(json.dumps(dict(schema_version=1, groups=[], include=['AUX3_MS2'], exclude=[])))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                training.run(cfg, root, action='evaluate', device='cpu', output=output)

    def test_cpu_train_roundtrip_never_reads_test_during_fit(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); cfg = fixture(root); output = root / 'experiment'; output.mkdir()
            data = training.data_for(cfg, root); identity = training.experiment_identity(data, cfg)
            with patch.object(data, 'frames', wraps=data.frames) as reader:
                training.train(data, cfg, output, identity, torch.device('cpu'))
                self.assertNotIn('test', [c.args[0] for c in reader.call_args_list])
            report = training.evaluate(data, cfg, output, identity, torch.device('cpu'))
            self.assertEqual(set(report['metrics']), {'BOOLEAN', 'CHOICE', 'ORDINAL'})
            self.assertTrue((output / 'test_by_sensor.csv').is_file())
            predictions = pd.read_csv(output / 'test_predictions.csv')
            self.assertEqual(set(predictions.run_id), {5, 6})
            before = sha(output / 'model.pt')
            training.evaluate(data, cfg, output, identity, torch.device('cpu'))
            self.assertEqual(before, sha(output / 'model.pt'))
            with self.assertRaises(FileExistsError):
                training.train(data, cfg, output, identity, torch.device('cpu'))
            changed = dict(identity, schema='wrong')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                training.evaluate(data, cfg, output, changed, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
