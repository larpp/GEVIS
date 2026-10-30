"""Invented features only; never opens private inputs or model bundles."""
import copy,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pecvd.experiments.temporal_localization import (LocalizerConfig,TemporalHead,
    latent_grid,fit_localizer,save_localizer,load_localizer,apply_localization,cached_localization_records)


def records(split):
    rng=np.random.default_rng(14)
    for i in range(2):
        tokens=rng.normal(size=(2,4,8)).astype('float32')
        mask=np.zeros((2,12),dtype='float32');mask[0,3:7]=1
        yield dict(split=split,run_id=split+str(i),reference_run_ids=['train-reference'],tokens=tokens,mask=mask)


class TemporalLocalizationTests(unittest.TestCase):
    def test_retired_localizer_fixture_uses_frozen_runtime(self):
        import json
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        n=json.loads((root/'tests/fixtures/step3_legacy_diagnostics.json').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in n['cells'] if c.get('id')=='temporal-localization-build'))
        env=dict(RUN_TEMPORAL_LOCALIZATION_TRAINING=True,ENCODER_ROOT='synthetic-cache',
            backbone_path=Path('synthetic-backbone.pt'),encoders={'phase':object()},DEVICE='cpu',numeric_state=object(),
            importlib=__import__('importlib'))
        with patch('importlib.reload',side_effect=lambda module:module), \
             patch('pecvd.experiments.temporal_localization.prepare_localizer',return_value=('head',.5)) as prepare:
            exec(compile(source,'STEP3:temporal-localization','exec'),env)
        m=prepare.call_args.args[0]
        self.assertIs(m.STEP_ENCODERS,env['encoders']);self.assertIs(m.numeric_state,env['numeric_state'])
        self.assertEqual(env['TEMPORAL_LOCALIZER'],('head',.5))

    def test_extracts_learned_blocks_not_handcrafted_features(self):
        state=torch.arange(1*8*16).reshape(1,8,16).float()
        tokens,grid=latent_grid(state,2,12)
        self.assertEqual(grid,3);self.assertEqual(tokens.shape,(2,4,8))
        torch.testing.assert_close(tokens.flatten(0,1),state[0,:,:8])
        with self.assertRaises(ValueError):latent_grid(state,2,13)

    def test_same_head_supports_different_step_patch_lengths(self):
        head=TemporalHead(8,16,8)
        self.assertEqual(head(torch.zeros(2,4,8),3).shape,(2,12))
        self.assertEqual(head(torch.zeros(2,8,8),7).shape,(2,56))

    def test_multiscale_bridge_excludes_all_56_handcrafted_features(self):
        state=torch.zeros(1,16,440);state[:,:,384:]=10000
        tokens,grid=latent_grid(state,2,40,latent_width=384)
        self.assertEqual(tokens.shape,(2,8,384));self.assertEqual(grid,5)
        self.assertEqual(float(tokens.abs().sum()),0.)

    def test_fit_save_reload_exact_identity_without_overwrite(self):
        cfg=LocalizerConfig(width=4,epochs=1)
        head,cutoff,evidence=fit_localizer(records('train'),records('validation'),cfg)
        self.assertGreater(evidence['training_updates'],0)
        self.assertFalse(evidence['test_used']);self.assertTrue(evidence['detection_unchanged'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'localizer.pt';identity={'synthetic':True}
            save_localizer(head,cutoff,evidence,path,identity,cfg)
            loaded,actual=load_localizer(path,identity)
            self.assertEqual(actual,cutoff)
            x=torch.zeros(2,4,8)
            torch.testing.assert_close(head(x),loaded(x))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            with self.assertRaises(FileExistsError):save_localizer(head,cutoff,evidence,path,identity,cfg)
            with self.assertRaisesRegex(ValueError,'identity'):load_localizer(path,{'changed':True})

    def test_never_accepts_test_or_overlapping_runs_for_fit(self):
        train=list(records('train'));validation=list(records('validation'))
        with self.assertRaisesRegex(ValueError,'Test'):
            fit_localizer(train,[dict(validation[0],split='test')])
        with self.assertRaisesRegex(ValueError,'overlap'):
            fit_localizer(train,[dict(validation[0],run_id='train0')])
        with self.assertRaisesRegex(ValueError,'Test'):
            list(cached_localization_records(SimpleNamespace(),'test'))

    def test_unsampled_train_references_are_valid_only_in_declared_population(self):
        train=list(records('train'));validation=list(records('validation'))
        validation[0]['reference_run_ids']=['unsampled-train']
        with self.assertRaisesRegex(ValueError,'TRAIN only'):
            fit_localizer(train,validation,LocalizerConfig(epochs=1,width=4))
        fit_localizer(train,validation,LocalizerConfig(epochs=1,width=4),
            train_run_ids=['train0','train1','train-reference','unsampled-train'],
            validation_run_ids=['validation0','validation1'])

    def test_intervals_change_but_detection_and_winner_identity_do_not(self):
        head=TemporalHead(8,16,4)
        for parameter in head.parameters():parameter.data.zero_()
        head.output.bias.data.fill_(10.)
        row=dict(sensor='a',scope_rows=[0,12],changed=True,detected_rows=[2,5],
            normalized_score=7.,pointwise={'exceedance_intervals':[[2,5]]})
        result=dict(accepted=True,changed=True,winner=row,events=[row],ranked_evidence=[row])
        module=SimpleNamespace(INTERVAL_MODE='learned',TEMPORAL_LOCALIZER=(head,.5))
        apply_localization(module,result,torch.zeros(1,8,16),['a','b'],12)
        self.assertIs(result['winner'],row);self.assertIs(result['events'][0],row)
        self.assertTrue(result['accepted']);self.assertTrue(result['changed'])
        self.assertEqual(row['detected_rows'],[0,12]);self.assertEqual(row['numeric_detected_rows'],[2,5])
        self.assertEqual(row['normalized_score'],7.)
        self.assertEqual(row['pointwise']['exceedance_intervals'],[[2,5]])

    def test_missing_learned_weights_is_not_numeric_interval_fallback(self):
        row=dict(sensor='a',scope_rows=[0,12],changed=True,detected_rows=[2,5],pointwise={'exceedance_intervals':[[2,5]]})
        result=dict(accepted=True,changed=True,winner=row,events=[row],ranked_evidence=[row])
        apply_localization(SimpleNamespace(INTERVAL_MODE='learned',TEMPORAL_LOCALIZER=None),result,torch.zeros(1,8,16),['a','b'],12)
        self.assertIsNone(row['detected_rows']);self.assertTrue(result['changed'])
        self.assertEqual(row['numeric_detected_rows'],[2,5])
        self.assertIn('unavailable',result['interval_source'])


if __name__=='__main__':unittest.main()
