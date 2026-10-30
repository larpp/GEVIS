"""Preserve the pre-optimization model computation and synthetic cases."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pecvd.gevis_backbone import ANSWER_TYPES
from pecvd.temporal_bridge import numeric_state
from pecvd.experiments.fusion_answers import KINDS, QUESTIONS
from pecvd.experiments.tips_model import TipsModel, SyntheticCases
from pecvd.experiments.tips_optimization import (
    FrozenTraceCache, PairedBatchSampler, numeric_state_from_encoded, trace_key)
from pecvd.experiments import tips_training as training


def legacy_features(model, current, reference, centers, steps, positions):
    """Reference implementation before trace caching and question batching."""
    state = numeric_state(model.encoder, current, reference)
    batch, count, _ = state.shape; patches = count // len(model.sensors)
    fusion = model.fusion
    keys = fusion.numeric_key_norm(fusion.numeric(fusion.numeric_input_norm(state)))
    descriptors = [[f'Main step {step}; sensor {sensor}; patch {p}; '
        + ('analysis center' if j == center else 'context sensor')
        for j, sensor in enumerate(model.sensors) for p in range(patches)]
        for center, step in zip(centers.tolist(), steps.tolist())]
    unique = list(dict.fromkeys(t for row in descriptors for t in row))
    lookup = {t: i for i, t in enumerate(unique)}
    text = fusion.text(unique, current.device)
    ids = torch.tensor([[lookup[t] for t in row] for row in descriptors], device=current.device)
    described = fusion.descriptor_norm(text[ids])
    selected = keys.reshape(batch, len(model.sensors), patches, -1)[torch.arange(batch), centers]
    features = []
    for kind in KINDS:
        question = fusion.text([QUESTIONS[kind]], current.device).expand(batch, -1)
        query = question + fusion.type_emb.weight[ANSWER_TYPES[kind]]
        gate = fusion.fusion_gate(torch.cat([keys, described], -1))
        fused, attn = fusion.cross(query[:, None], keys + gate * described, keys, need_weights=True)
        fused = fusion.norm(fused[:, 0] + query)
        weights = attn[:, 0].reshape(batch, len(model.sensors), patches)[torch.arange(batch), centers]
        local = (selected * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        features.append(torch.cat([fused, query, selected.mean(1), local,
                                   torch.stack([positions, 1-positions], -1)], -1))
    return features


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model_inputs(self):
        torch.manual_seed(8)
        model = TipsModel(['a', 'b', 'c'], dict(patch_length=4, encoder_layers=1,
            fusion_width=16, head_width=16, text_encoder='byte', main_steps=[14, 15]))
        model.freeze_encoder(); model.eval()
        current = torch.randn(2, 8, 3).repeat_interleave(2, 0)
        reference = torch.randn(1, 8, 3).expand_as(current)
        args = (current, reference, torch.tensor([0, 1, 1, 2]),
                torch.tensor([14, 14, 15, 15]), torch.tensor([0., .2, .5, .9]))
        return model, args

    def test_cached_bridge_and_question_batching_preserve_logits_and_gradients(self):
        model, args = self.model_inputs()
        baseline = [model.readout(x, k) for x, k in zip(legacy_features(model, *args), KINDS)]
        sum(z.square().sum() for z in baseline).backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        optimized = model(*args)
        for a, b in zip(baseline, optimized):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        sum(z.square().sum() for z in optimized).backward()
        for n, p in model.named_parameters():
            if n in gradients:
                torch.testing.assert_close(gradients[n], p.grad, atol=2e-5, rtol=2e-4, msg=n)
        self.assertEqual(model.trace_cache.encoded_traces, 3)  # two observed + one reference
        again = model(*args)
        for a, b in zip(optimized, again):
            torch.testing.assert_close(a, b)
        self.assertEqual(model.trace_cache.encoded_traces, 3)

    def test_bounded_cache_eviction_no_storage_views_and_trainable_encoder_rejected(self):
        model, args = self.model_inputs(); current, reference = args[:2]
        cache = FrozenTraceCache(.004)  # capacity for one trace in this fixture
        expected = numeric_state(model.encoder, current, reference)
        for _ in range(2):
            encoded = cache.encode_pair(model.encoder, current, reference)
            actual = numeric_state_from_encoded(model.encoder, current, reference, encoded)
            torch.testing.assert_close(actual, expected)
            self.assertLessEqual(cache.bytes, cache.limit)
            for tensors in cache.entries.values():
                for t in tensors:
                    self.assertEqual(t.untyped_storage().nbytes(), t.numel() * t.element_size())
        no_cache = FrozenTraceCache(0)
        no_cache.encode_pair(model.encoder, current, reference)
        self.assertEqual(no_cache.bytes, 0)
        self.assertEqual(no_cache.encoded_traces, 3)  # batch dedup still applies
        model.encoder.train()
        with self.assertRaisesRegex(ValueError, 'frozen'):
            cache.encode_pair(model.encoder, current, reference)

    def test_paired_sampling_retains_every_case_and_intervenes_once_per_pair(self):
        values = np.arange(24, dtype=np.float32).reshape(8, 3)
        windows = [dict(values=values, run_id=1, step=14, position=.2)]
        cases = SyntheticCases(windows, ['a', 'b', 'c'], ['a', 'b'])
        for size in (1, 7, 8):
            sampler = PairedBatchSampler(cases, size, shuffle=True)
            batches = list(sampler); flat = [i for b in batches for i in b]
            self.assertEqual(sorted(flat), list(range(len(cases))))
            self.assertEqual(len(sampler), len(batches))
            self.assertTrue(all(flat[i + 1] == flat[i] + 1 for i in range(0, len(flat), 2)))
        with patch.object(cases, '_intervene', wraps=cases._intervene) as inject:
            for i in (2, 3, 4, 5):
                item = cases[i]
                self.assertEqual(item['current_key'], trace_key(item['current'].numpy()))
                self.assertEqual(item['reference_key'], trace_key(item['reference'].numpy()))
            self.assertEqual(inject.call_count, 2)

    def test_validation_metrics_identical_without_rows_and_progress_file_updates(self):
        model, args = self.model_inputs()
        windows = [dict(values=args[0][0].numpy(), run_id=1, step=14, position=.2)]
        data = type('Data', (), dict(requested=model.sensors, stats={14: dict(varying=[True]*3)}))()
        config = dict(seed=42, cases_per_window=24, target_sensors=['a'], batch_size=8,
                      log_every_batches=1000, log_every_seconds=1000)
        loader = training.case_loader(windows, data, config, 'validation')
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as logs:
            output = Path(folder)
            scores, rows, _ = training.score(model, loader, torch.device('cpu'),
                progress=training.progress_for('Validation 1/1', loader, config, output))
            no_rows_scores, no_rows, _ = training.score(model, loader, torch.device('cpu'), collect_rows=False)
            self.assertEqual(scores, no_rows_scores)
            self.assertEqual(len(rows), len(loader.dataset) * 3); self.assertEqual(no_rows, [])
            progress = json.loads((output / 'progress.json').read_text())
            self.assertEqual(progress['batch'], len(loader)); self.assertEqual(progress['percent'], 100.)
            self.assertIn('batch 0/', logs.getvalue()); self.assertIn('ETA=', logs.getvalue())


if __name__ == '__main__':
    unittest.main()
