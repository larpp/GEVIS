"""Synthetic sample shifts, numeric evidence and matched ablation execution."""
import contextlib
import importlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pecvd.experiments import fusion_answers as fa, reconstruction_evidence as recon
from pecvd.answer_display import ANSWER_CRITERIA as LEGACY_CRITERIA
from tests.test_fusion_answers import runtime, source_rows
from tests.test_fast_experiments import cache
from tests.test_gevis_fast import model_class


def evidence_runtime():
    r=runtime();m=r.module
    m.GEVIS_MODEL=model_class()(numeric_dim=392,d_model=16,dropout=0.).requires_grad_(False).eval()
    def numeric(encoder,a,b):
        # Invented exact 392-wide interface, zero-valued reconstructor.
        ap=a.reshape(1,2,4,2).permute(0,3,1,2).reshape(1,4,4)
        bp=b.reshape(1,2,4,2).permute(0,3,1,2).reshape(1,4,4)
        state=torch.zeros(1,4,392)
        state[...,0]=ap.mean(-1);state[...,96]=bp.mean(-1)
        state[...,192]=(ap-bp).mean(-1)
        state[...,-2]=ap.square().mean(-1);state[...,-1]=bp.square().mean(-1)
        return state
    m.numeric_state=numeric
    m.FUSION_ANSWER_HEAD=fa.ContextReadout(66,16).requires_grad_(False).eval()
    return r


class DelayReconstructionTests(unittest.TestCase):
    def test_active_ui_options_follow_new_head_without_changing_legacy(self):
        from pecvd.gevis_ui import GevisRuntime
        r=evidence_runtime();m=r.module
        m.review_changes=lambda *a,**kw:None
        m.GevisQuestion=lambda text,answer_type,criteria:SimpleNamespace(
            text=text,answer_type=answer_type,criteria=criteria)
        questions=GevisRuntime(m).questions()
        self.assertEqual(next(q.criteria[-1] for q in questions if q.answer_type=='CHOICE'),'Delay')

    def test_causal_shift_not_wraparound_and_constant_not_positive(self):
        x=np.zeros((8,2));x[2,0]=3
        y,mask=fa.delay_trace(x,0,2)
        self.assertEqual(y[4,0],3);self.assertEqual(y[2,0],0)
        self.assertFalse(mask[:,1].any());self.assertEqual(x[2,0],3)
        flat=np.ones((8,2));z,mask=fa.delay_trace(flat,0,4)
        np.testing.assert_array_equal(z,flat);self.assertFalse(mask.any())
        with self.assertRaises(ValueError):fa.delay_trace(x,0,8)
        self.assertEqual(fa.ANSWER_CRITERIA['CHOICE'][5],'Delay')
        self.assertEqual(LEGACY_CRITERIA['CHOICE'][5],'Reconstruction error')
        r=evidence_runtime();flat_rows=source_rows('train')
        for row in flat_rows:row['current']=np.ones_like(row['current'])
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons',return_value=flat_rows),contextlib.redirect_stdout(io.StringIO()):
            examples=fa.controlled_examples(r.module,'train')
        delayed=[e for e in examples if e['kind']=='delay']
        self.assertTrue(delayed)
        self.assertTrue(all(e['label']==0 and not e['injected'] for e in delayed))

    def test_delay_targets_units_and_evidence_from_actual_fusion_generator(self):
        r=evidence_runtime()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons',side_effect=cache),contextlib.redirect_stdout(io.StringIO()):
            rows=fa.controlled_examples(r.module,'train')
        delayed=[e for e in rows if e['kind']=='delay' and e['center_is_perturbed']]
        self.assertTrue(delayed)
        self.assertEqual({e['lag_samples'] for e in delayed},{1,2,3,4})
        for e in delayed:
            self.assertEqual(e['strength_unit'],'native_samples')
            expected=1 if e['answer_type']=='BOOLEAN' else 5 if e['answer_type']=='CHOICE' else e['lag_samples']
            self.assertEqual(e['label'],expected)
            self.assertTrue(e['reconstruction_available'])
            self.assertGreaterEqual(e['current_reconstruction_mse'],0)
        self.assertEqual(recon.indices(440),[438,439,398,399])
        with self.assertRaises(ValueError):recon.indices(8)

    def test_explicit_statistics_mask_restores_and_evidence_not_type_label(self):
        r=evidence_runtime();original=r.module.numeric_state
        a=torch.ones(1,8,2);b=torch.zeros_like(a)
        state=original(None,a,b)
        evidence=recon.evidence(state,['a','b'],[('a','BOOLEAN')])[0]
        self.assertEqual(evidence['current_reconstruction_mse'],1.)
        self.assertEqual(evidence['reference_reconstruction_mse'],0.)
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with recon.without_explicit_reconstruction(r.module):
                masked=r.module.numeric_state(None,a,b)
                self.assertEqual(masked[...,-2:].sum(),0)
                torch.testing.assert_close(masked[...,:384],state[...,:384])
                raise RuntimeError('intentional')
        self.assertIs(r.module.numeric_state,original)

    def test_actual_exp_reconstruction_hook_refits_matched_cases_preserves_runtime(self):
        r=evidence_runtime();m=r.module
        before={k:v.clone() for k,v in m.FUSION_ANSWER_HEAD.state_dict().items()}
        original=m.numeric_state
        cfg=replace(fa.FusionConfig(),epochs=2,width=16,max_seconds=10.)
        path=Path(__file__).resolve().parents[1]/'notebooks/research/CVD_feature_ablation.ipynb'
        doc=json.loads(path.read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='reconstruction-controls'))
        env=dict(RUNTIME=r,CONTROL_CONFIG=cfg,SELECTED_STEPS=None,importlib=importlib,display=lambda _:None,
                 RUN_RECONSTRUCTION_CONTROLS=True)
        with patch('importlib.reload',side_effect=lambda m:m), \
             patch('pecvd.experiments.discovery_calibration.cached_comparisons',side_effect=cache), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(source,env)
        suite=env['RECON_CONTROLS']
        a=suite.reports['full_reconstruction'].details
        b=suite.reports['without_explicit_reconstruction'].details
        self.assertEqual(list(zip(a.pair_id,a.label)),list(zip(b.pair_id,b.label)))
        self.assertIs(m.numeric_state,original)
        for k,v in before.items():torch.testing.assert_close(m.FUSION_ANSWER_HEAD.state_dict()[k],v)
