import ast
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from pecvd.patchtst_adaptation import (blind_prediction, checkerboard_mask,
    detail_components, DetailWeights, adaptation_objective)
from test_official_patchtst import tiny

ROOT=Path(__file__).resolve().parents[1]


class PatchTSTAdaptationTests(unittest.TestCase):
    def test_complementary_masks_hide_each_output_and_never_call_clean_forward(self):
        model=tiny().eval();x=torch.randn(2,24,3)
        calls=[];original=model.forward
        def forward(values,mask=None):
            self.assertIsNotNone(mask)
            self.assertTrue(mask.any(1).all())
            self.assertTrue((~mask).any(1).all())
            calls.append(mask.clone())
            return original(values,mask)
        with patch.object(model,'forward',side_effect=forward):
            pred=blind_prediction(model,x)
        self.assertEqual(len(calls),2)
        self.assertTrue((calls[0]^calls[1]).all())
        direct=[original(x,m) for m in calls]
        torch.testing.assert_close(pred,torch.where(calls[0][...,None],direct[0],direct[1]))
        # Change only the target patch. Its own blind prediction MUST be invariant.
        changed=x.clone();changed[:,8:12,0]+=1000
        torch.testing.assert_close(pred[:,2],blind_prediction(model,changed)[:,2])

    def test_all_losses_have_finite_nonzero_prediction_gradients(self):
        torch.manual_seed(8)
        target=torch.randn(2,18,4,requires_grad=True)
        prediction=torch.randn_like(target,requires_grad=True)
        parts=detail_components(prediction,target,3)
        self.assertEqual(set(parts),{'blind_mse','variability','slope','peak','relationship'})
        for key,value in parts.items():
            self.assertTrue(torch.isfinite(value),key)
            gradient=torch.autograd.grad(value,prediction,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all(),key)
            self.assertGreater(float(gradient.abs().sum()),0,key)
        sum(parts.values()).backward()
        self.assertIsNone(target.grad)  # no teacher/target optimization

    def test_perfect_reconstruction_flat_and_single_channel_are_well_defined(self):
        x=torch.randn(2,18,4)
        self.assertTrue(all(float(v)==0 for v in detail_components(x,x,3).values()))
        for sensors in (1,3):
            pred=torch.zeros(2,sensors*6,4,requires_grad=True)
            parts=detail_components(pred,torch.zeros_like(pred),sensors)
            sum(parts.values()).backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertEqual(float(parts['relationship']),0.)
        with self.assertRaises(ValueError):DetailWeights(peak=float('nan'))
        with self.assertRaises(ValueError):DetailWeights(slope=-1)

    def test_full_encoder_and_new_head_update_with_differential_lr(self):
        model=tiny();model.configure_finetuning(None);model.train()
        encoder=model.pretrained
        before={k:v.clone() for k,v in encoder.state_dict().items()}
        encoder_ids={id(p) for p in encoder.parameters()}
        opt=torch.optim.AdamW([
            {'params':[p for p in encoder.parameters() if p.requires_grad],'lr':1e-5},
            {'params':[p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids],'lr':1e-4}])
        loss,parts=adaptation_objective(model,torch.randn(2,24,3))
        loss.backward()
        self.assertGreater(float(encoder.encoder.embedder.input_embedding.weight.grad.abs().sum()),0)
        self.assertTrue(any(p.grad is not None for p in encoder.encoder.layers[0].parameters()))
        self.assertTrue(any(p.grad is not None for p in model.hf.head.parameters()))
        self.assertFalse(encoder.encoder.positional_encoder.position_enc.requires_grad)
        opt.step()
        self.assertTrue(any(not torch.equal(v,encoder.state_dict()[k]) for k,v in before.items() if 'embedder' in k))
        expected=parts['blind_mse']+.25*parts['variability']+.25*parts['slope']+.1*parts['peak']+.1*parts['relationship']
        torch.testing.assert_close(loss,expected)

    def test_runtime_validation_and_final_evaluation_share_blind_predictions(self):
        import numpy as np
        import pecvd.patchtst_adaptation as adaptation
        from types import SimpleNamespace
        from torch.utils.data import DataLoader
        doc=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c['id']=='01-fit'))
        funcs=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('run_epoch','evaluate_masked')]
        cfg=SimpleNamespace(batch_size=2,num_workers=0,use_amp=False,max_grad_norm=.5,
            variability_weight=.25,slope_weight=.25,peak_weight=.1,relationship_weight=.1)
        env={'torch':torch,'nn':torch.nn,'DataLoader':DataLoader,'DEVICE':torch.device('cpu'),
             'AMP_DTYPE':torch.bfloat16,'CONFIG':cfg,'patchtst_adaptation':adaptation,
             'reconstruction_mask':lambda model,b,p,*args:torch.zeros(b,model.n_sensors*p,dtype=torch.bool)}
        exec(compile(ast.Module(funcs,type_ignores=[]),'STEP1:domain-evaluation','exec'),env)
        model=tiny().eval();values=np.random.default_rng(18).normal(size=(3,24,3)).astype('float32')
        loss,_=env['run_epoch'](model,values,seed=15)
        mse,zero=env['evaluate_masked'](model,values,seed=999)
        self.assertAlmostEqual(mse,model.last_epoch_components['blind_mse'],places=6)
        self.assertGreaterEqual(loss,mse)
        # Scheduler and best-state selection must use validation blind MSE, not total or test.
        self.assertIn("va=validation_components.get('blind_mse',validation_loss)",source)
        self.assertIn('scheduler.step(va)',source)
        self.assertIn('if va<best-CONFIG.min_delta',source)

    def test_new_namespace_strips_prior_suffix_and_keeps_source_config(self):
        from dataclasses import replace
        from pecvd.model_variants import active_config
        from pecvd.process_config import load_process_config
        source=load_process_config(ROOT/'config/processes/pecvd.json')
        old=replace(source,artifacts=replace(source.artifacts,encoder=source.artifacts.encoder+'_hf_patchtst_official_v1'))
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'official'}):
            new=active_config(old)
            self.assertEqual(new.artifacts.encoder,source.artifacts.encoder+'_hf_patchtst_official_domain_v2')
            self.assertEqual(active_config(new),new)
            self.assertTrue(old.artifacts.encoder.endswith('_hf_patchtst_official_v1'))


if __name__=='__main__':unittest.main()
