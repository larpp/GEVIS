"""Synthetic CPU-only multiscale, relationship and long-context contracts."""
import ast,contextlib,io,json,sys,tempfile,types,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from pecvd.multiscale_patchtst import (MultiscalePatchEncoder,BACKEND,CONTRACT,
    crop_window,balanced_window_weights,multiscale_objective,lag_relationship_loss)
from pecvd.relational_features import relational_features
from pecvd.temporal_bridge import numeric_state
from pecvd.encoder_loading import load_encoder

ROOT=Path(__file__).resolve().parents[1]


class MultiscaleTests(unittest.TestCase):
    def model(self,width=16):
        return MultiscalePatchEncoder(3,4,width,4,1,0.).eval()

    def test_long_and_short_native_grids_and_blind_mask_exclusion(self):
        net=self.model()
        for length in (16,40,256):
            x=torch.randn(2,length,3,requires_grad=True);p=length//4
            mask=torch.arange(3*p).remainder(2).bool()[None].expand(2,-1)
            y=net(x,mask);self.assertEqual(y.shape,(2,3*p,4));y.sum().backward()
            grad=net.patchify(x.grad)[0]
            self.assertEqual(int(torch.count_nonzero(grad[mask])),0)
            self.assertGreater(float(net.global_block.sensor.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(net.scale_gate.weight.grad.abs().sum()),0)

    def test_crop_never_mixes_runs_and_preserves_native_samples(self):
        x=torch.arange(64).float()[None,:,None].expand(2,-1,3).clone();x[1]+=1000
        for divisor in (2,4):
            for random in (False,True):
                crop=crop_window(x,4,divisor,random=random)
                self.assertEqual(crop.shape[1],64//divisor)
                torch.testing.assert_close(crop.diff(dim=1),torch.ones_like(crop[:,1:]))
                self.assertLess(float(crop[0].max()),1000);self.assertGreaterEqual(float(crop[1].min()),1000)

    def test_run_position_sampling_has_equal_run_and_bin_mass(self):
        meta={'process_id':['a']*6+['b']*2,'position_bin':[0,0,0,0,1,1,0,1]}
        w=np.asarray(balanced_window_weights(meta))
        self.assertAlmostEqual(w[:6].sum(),w[6:].sum())
        self.assertAlmostEqual(w[:4].sum(),w[4:6].sum())

    def test_multiscale_loss_trains_local_global_gate_and_reports_crops_lags(self):
        net=self.model().train();x=torch.randn(2,64,3)
        mask=torch.arange(48).remainder(2).bool()[None].expand(2,-1)
        cfg=SimpleNamespace(slope_weight=.25,relationship_weight=.1,variability_weight=.25,peak_weight=.1)
        loss,parts=multiscale_objective(net,x,mask,cfg);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIn('crop_mse',parts);self.assertIn('lag_relationship',parts)
        self.assertGreater(float(net.blocks[0].sensor.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(net.global_block.temporal.self_attn.in_proj_weight.grad.abs().sum()),0)

    def test_relationship_features_detect_pair_disruption_and_stuck(self):
        t=torch.linspace(0,12,64);ref=torch.stack([t.sin(),t.sin(),t.cos()],-1)[None]
        same=relational_features(ref,ref,4)
        changed=ref.clone();changed[:,:,1]*=-1
        broken=relational_features(changed,ref,4)
        self.assertGreater(float(broken[...,0].mean()),float(same[...,0].mean()))
        stuck=ref.clone();stuck[:,16:48,0]=stuck[:,16:17,0]
        f=relational_features(stuck,ref,4).reshape(1,3,16,32)
        self.assertGreater(float(f[:,0,6:10,27].mean()),.9)
        self.assertTrue(torch.isfinite(f).all())
        self.assertEqual(float(lag_relationship_loss(ref.transpose(1,2),ref.transpose(1,2))),0.)

    def test_delay_features_and_constant_no_false_lag(self):
        rng=torch.Generator().manual_seed(12)
        ref=torch.randn(1,64,3,generator=rng);cur=torch.cat([ref[:,:2],ref[:,:-2]],1)
        f=relational_features(cur,ref,4).reshape(1,3,16,32)
        self.assertGreater(float(f[:,:,2:-2,18].mean()),0.)
        constant=relational_features(torch.ones_like(ref),torch.ones_like(ref),4)
        self.assertEqual(float(constant[...,18].abs().sum()),0.)

    def test_offline_checkpoint_440_bridge_and_reload_class_safety(self):
        net=self.model(96)
        ck=dict(encoder_backend=BACKEND,reconstruction_contract=CONTRACT,sensor_columns=['a','b','c'],
            patch_length=4,window_length=40,model_config=dict(d_model=96,nhead=4,layers=1,dropout=0.,local_patches=8,coarse_factor=4),model_state=net.state_dict())
        first=load_encoder(ck).eval();second=load_encoder(ck).eval();x=torch.randn(1,40,3)
        torch.testing.assert_close(first.tokens(x),second.tokens(x))
        state=numeric_state(first,x,x)
        self.assertEqual(state.shape,(1,30,440))
        torch.testing.assert_close(state[...,-2],(first.blind_prediction(x)-first.patchify(x)[0]).square().mean(-1))
        with self.assertRaises(ValueError):load_encoder(dict(ck,reconstruction_contract='v2'))

    def test_grid_rule_no_step_name_override_and_distinct_namespace(self):
        from pecvd.model_variants import active_config,active_run_tag
        from pecvd.process_config import load_process_config
        base=load_process_config(ROOT/'config/processes/pecvd.json')
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'multiscale'}):
            cfg=active_config(base)
            self.assertTrue(cfg.artifacts.encoder.endswith('_native_multiscale_blind_v3'))
            self.assertIn('multiscale',active_run_tag())
            for before,after in zip(base.steps,cfg.steps):
                self.assertEqual(before.window_length,after.window_length)
                self.assertGreaterEqual(after.window_length//after.patch_length,8)
                self.assertLessEqual(after.window_length//after.patch_length,64)

    def test_injection_effectiveness_is_not_mean_only_and_reports_flat_noops(self):
        from pecvd.experiments.injection import Injection,inject
        x=np.stack([np.sin(np.arange(32)),np.cos(np.arange(32))],-1)
        _,truth=inject(x,Injection('stuck',0,0,32),np.ones(2))
        self.assertTrue(truth['effective'])
        for kind in ('stuck','delay','relationship'):
            _,truth=inject(np.ones((32,2)),Injection(kind,0,0,32),np.ones(2))
            self.assertFalse(truth['effective']);self.assertEqual(truth['ineffective_reason'],'constant_target_segment')
        _,truth=inject(x,Injection('relationship',0,0,32,3),np.ones(2))
        self.assertTrue(truth['effective']);self.assertTrue(truth['relationship_evaluable'])

    def test_actual_notebook_multiscale_train_save_manifest_reuse_and_consumers(self):
        import pandas as pd
        import torch._dynamo  # initialize before sys.modules snapshot
        from dataclasses import replace
        from pecvd.encoder_reuse import sha256
        cells={c['id']:''.join(c['source']) for c in json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code'}
        module=types.ModuleType('synthetic_multiscale_training');env=module.__dict__
        with patch.dict(sys.modules,{module.__name__:module}),patch.dict('os.environ',{}),contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cells['01-config'].replace("NOTEBOOK_MODEL_VARIANT='hybrid'","NOTEBOOK_MODEL_VARIANT='multiscale'"),'multiscale-config','exec'),env)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder);source=path/'fixture.txt';source.write_text('synthetic only',encoding='utf-8')
                arrays={s:np.random.default_rng(i).normal(size=(4,16,3)).astype('float32') for i,s in enumerate(('train','validation','test'))}
                split={s:[s+str(i) for i in range(4)] for s in arrays}
                meta=pd.DataFrame(dict(process_id=['r0','r1','r2','r3'],source_group=['fixture']*4,pm_cycle=['fixture']*4,cohort_key=['fixture']*4,window_start=[0]*4,window_end=[16]*4,position_ratio=[0.]*4,position_bin=[0]*4))
                env.update(CONFIG=replace(env['CONFIG'],epochs=2,min_epochs=0,warmup_epochs=1,batch_size=2,use_amp=False,fused_adamw=False,validation_interval=1),DEVICE=torch.device('cpu'),DATA_PATH=source,ARTIFACT_ROOT=path/'candidate',STEPS={'Gap-up':(16,4,4)},sensor_columns=['a','b','c'],sensor_descriptors={s:s for s in 'abc'},raw_windows={'Gap-up':arrays},window_metadata={'Gap-up':{s:meta for s in arrays}},run_split=split,cohort_split=pd.DataFrame({'train':[4],'validation':[4],'test':[4]}))
                funcs=[n for n in ast.parse(cells['01-fit']).body if isinstance(n,ast.FunctionDef)]
                exec(compile(ast.Module(funcs,type_ignores=[]),'multiscale-fit','exec'),env)
                ck=env['train_one_step']('Gap-up');net=load_encoder(ck).eval()
                self.assertEqual(ck['encoder_backend'],BACKEND)
                self.assertIn('train_crop_mse',ck['history'][1]);self.assertIn('train_lag_relationship',ck['history'][1])
                self.assertLessEqual(ck['training_status']['best_validation_mse'],ck['training_status']['initial_validation_mse'])
                mean=arrays['train'].mean((0,1),keepdims=True);std=arrays['train'].std((0,1),keepdims=True).clip(1e-6)
                score,_=env['evaluate_masked'](net,((arrays['validation']-mean)/std).astype('float32'),999)
                self.assertAlmostEqual(score,ck['training_status']['best_validation_mse'],places=5)
                env.update(encoder_checkpoints={'Gap-up':ck},REUSE_SAVED_ENCODERS=False)
                exec(compile(cells['01-manifest'],'manifest','exec'),env)
                before={p:sha256(p) for p in env['ARTIFACT_ROOT'].rglob('*') if p.is_file()}
                env.update(REUSE_SAVED_ENCODERS=True,RUN_TRAINING=False,PROCESS_CONFIG=SimpleNamespace(artifacts=SimpleNamespace(encoder=ck['artifact_version']),step_specs={'Gap-up':(16,4,4)}))
                with patch('torch.save',side_effect=AssertionError('No rewrite')),patch('torch.optim.AdamW',side_effect=AssertionError('No retrain')),patch('pandas.read_csv',side_effect=AssertionError('No raw data')):
                    for cell in ('01-load','01-windows','01-fit','01-manifest'):exec(compile(cells[cell],cell,'exec'),env)
                self.assertEqual(before,{p:sha256(p) for p in before})
                x=torch.randn(1,16,3)
                for name,cell in [('STEP_3_train_gevis_decision.ipynb','02-functions'),('STEP_3.1_run_gevis_inference.ipynb','03-types')]:
                    doc=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
                    code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')==cell))
                    fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='numeric_state')
                    scope={'torch':torch};exec(compile(ast.Module([fn],type_ignores=[]),'state','exec'),scope)
                    torch.testing.assert_close(scope['numeric_state'](net,x,x),numeric_state(net,x,x))

    def test_step2_and_runtime_numeric_adapters_share_440_state(self):
        from pecvd import gevis_backbone
        from torch import nn
        backbone=gevis_backbone.GEVISBackbone(440,32,0.,text_encoder_name='byte')
        for name,cell in [('STEP_3_train_gevis_decision.ipynb','02-config'),('STEP_3.1_run_gevis_inference.ipynb','03-types')]:
            doc=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
            code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')==cell))
            nodes=[n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name in ('ByteTextEncoder','GEVIS')]
            scope={'torch':torch,'nn':nn,'gevis_backbone':gevis_backbone}
            exec(compile(ast.Module(nodes,type_ignores=[]),'runtime','exec'),scope)
            runtime=scope['GEVIS'](440,32,0.,'byte');runtime.numeric.load_state_dict(backbone.numeric.state_dict(),strict=True)
            x=torch.randn(2,12,440);torch.testing.assert_close(runtime.numeric(x),backbone.numeric(x))
