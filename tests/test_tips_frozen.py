"""Frozen-source invariants and an invented CPU target-readout round trip."""
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

from pecvd.legacy_patchtst import PatchEncoder
from pecvd.gevis_backbone import GEVISBackbone, BackbonePretrainHeads, ANSWER_TYPES, token_descriptions
from pecvd.experiments.fusion_answers import KINDS, QUESTIONS
from pecvd.experiments.questions import center_descriptions
from pecvd.experiments.tips_dataset import sha
from pecvd.experiments.tips_frozen_source import FrozenSource, SOURCE_STEPS
from pecvd.experiments import tips_frozen_training as frozen
from tests.test_tips_peald import fixture


def frozen_fixture(root):
    config = fixture(root); folder = root / 'frozen'; folder.mkdir()
    entries = {}
    for i, step in enumerate(SOURCE_STEPS):
        torch.manual_seed(i)
        model = PatchEncoder(2, 4, d_model=8, nhead=2, layers=1, dropout=0.)
        path = folder / f'encoder_{i}.pt'
        torch.save(dict(step=step, sensor_columns=['source_a', 'source_b'], patch_length=4,
            model_config=dict(d_model=8, nhead=2, layers=1, dropout=0.), model_state=model.state_dict()), path)
        entries[step] = dict(path=path.name, sha256=sha(path))
    manifest = folder / 'manifest.json'; manifest.write_text(json.dumps(dict(step_checkpoints=entries)))
    backbone = GEVISBackbone(40, 16, 0., 'byte')
    path = folder / 'backbone.pt'
    torch.save(dict(schema='gevis.backbone.v9', question_language='English', d_model=16, numeric_dim=40,
        text_encoder_name='byte', text_encoder_revision='test', backbone_state=backbone.state_dict(),
        pretrain_heads_state=BackbonePretrainHeads(16).state_dict(),
        step1_bridge=dict(manifest_sha256=sha(manifest), encoder_hashes={s:e['sha256'] for s,e in entries.items()})), path)
    for key in ('encoder_epochs', 'encoder_layers', 'fusion_width', 'text_encoder', 'patch_length'):
        config.pop(key)
    config.update(mode='tips_frozen_step12', source_manifest='frozen/manifest.json',
        source_manifest_sha256=sha(manifest), source_backbone='frozen/backbone.pt', source_backbone_sha256=sha(path),
        source_encoder_step='DEP.', input_adapter='legacy_channel_independent', feature_batch_size=8,
        answer_epochs=2, patience=2, batch_size=16, feature_cache_limit_mb=16)
    return config


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_six_encoders_and_complete_step2_frozen_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = frozen_fixture(root)
            source = FrozenSource(config, root, ['pressure', 'temperature', 'ValveAct_1:1'])
            self.assertEqual(len(source.encoders), 6)
            self.assertFalse(any(p.requires_grad for p in source.parameters()))
            source.train()
            self.assertTrue(all(not m.training for m in source.modules()))
            source.verify_frozen()
            with torch.no_grad():
                source.encoders[0].input.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError, 'tensor values changed'):
                source.verify_frozen()
            path = root / 'frozen/encoder_0.pt'
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
                FrozenSource(config, root, ['pressure'])

    def test_exact_bridge_and_fusion_equivalence_to_source_component_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config = frozen_fixture(root)
            names = ['pressure', 'temperature', 'ValveAct_1:1']
            source = FrozenSource(config, root, names)
            encoder, model = source.active_encoder, source.backbone
            a, b = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
            centers = torch.tensor([0, 2]); steps = torch.tensor([14, 14]); positions = torch.tensor([.2, .8])
            with torch.no_grad():
                at, bt = encoder.tokens(a), encoder.tokens(b)
                ap, p = encoder.patchify(a); bp, _ = encoder.patchify(b)
                ar, br = encoder(a), encoder(b)
                evidence = torch.stack([ap.mean(-1), bp.mean(-1), (ap.mean(-1)-bp.mean(-1)).abs(),
                    ap.std(-1, unbiased=False), bp.std(-1, unbiased=False), (ap-bp).abs().mean(-1),
                    (ar-ap).square().mean(-1), (br-bp).square().mean(-1)], -1)
                state = torch.cat([at, bt, at-bt, (at-bt).abs(), evidence], -1)
                torch.testing.assert_close(source.legacy_state(a, b, None), state)
                actual = source.features(a, b, centers, steps, positions)
                descriptions = [center_descriptions(token_descriptions('Main step 14', names, p), names, p, names[c]) for c in centers]
                keys = model.numeric_key_norm(model.numeric(model.numeric_input_norm(state)))
                selected = keys.reshape(2, 3, p, -1)[torch.arange(2), centers]
                for i, kind in enumerate(KINDS):
                    fused, attn = model(state, [QUESTIONS[kind]]*2, [kind]*2, descriptions)
                    query = model.text([QUESTIONS[kind]]*2, torch.device('cpu')) + model.type_emb.weight[ANSWER_TYPES[kind]]
                    weights = attn.reshape(2, 3, p)[torch.arange(2), centers]
                    local = (selected*weights[..., None]).sum(1)/weights.sum(1, keepdim=True).clamp_min(1e-8)
                    expected = torch.cat([fused, query, selected.mean(1), local, torch.stack([positions, 1-positions], -1)], -1)
                    torch.testing.assert_close(actual[:, i], expected, atol=3e-6, rtol=3e-5)
            source.verify_frozen()
            with self.assertRaisesRegex(ValueError, 'outside the selection profile'):
                source.features(a, b, centers, torch.tensor([15, 15]), positions)
            bad = dict(config, input_adapter='native')
            with self.assertRaisesRegex(ValueError, 'explicit legacy'):
                FrozenSource(bad, root, names)

    def test_head_training_never_updates_source_never_reads_test_and_reuses_features(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder); config = frozen_fixture(root); config['_gevis_root'] = str(root)
            source = FrozenSource(config, root, ['pressure', 'temperature', 'ValveAct_1:1'])
            data = frozen.data_for(config, root); output = root / 'out'; output.mkdir()
            identity = frozen.identity_for(data, config, source.identity)
            with patch.object(data, 'frames', wraps=data.frames) as reader, patch.object(source, 'features', wraps=source.features) as feature_call:
                frozen.train(data, source, config, output, identity, torch.device('cpu'))
                self.assertNotIn('test', [c.args[0] for c in reader.call_args_list])
                self.assertEqual(feature_call.call_count, 24)  # 12 Train + 12 Valid, once across TWO epochs
            source.verify_frozen()
            history = json.loads((output / 'history.json').read_text())
            self.assertTrue(all(h['stage']=='readout_only' and not h['frozen_source_updated'] for h in history))
            report = frozen.evaluate(data, source, config, output, identity, torch.device('cpu'))
            self.assertFalse(report['source_weights_updated'])
            self.assertEqual(set(pd.read_csv(output / 'test_predictions.csv').run_id), {5, 6})
            before = sha(output / 'model.pt')
            with patch.object(source, 'features', side_effect=AssertionError('Features recomputed')):
                frozen.evaluate(data, source, config, output, identity, torch.device('cpu'))
            self.assertEqual(before, sha(output / 'model.pt'))
            path = output / 'features/test/y.npy'
            with path.open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'cache hash mismatch'):
                frozen.evaluate(data, source, config, output, identity, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
