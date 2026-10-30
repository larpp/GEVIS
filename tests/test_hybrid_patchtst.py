import ast,contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from transformers import PatchTSTConfig,PatchTSTForPretraining,PatchTSTModel
from pecvd.hybrid_patchtst import HybridPatchEncoder,BACKEND
from pecvd.legacy_patchtst import PatchEncoder
from pecvd.pretrained_patchtst import load_encoder
from pecvd.experiments.frozen_transfer import FrozenChannelAdapter
from pecvd.patchtst_training import ValidationSelection,apply_warmup,legacy_rich_objective

ROOT=Path(__file__).resolve().parents[1]


def tiny_hybrid():
    cfg=PatchTSTConfig(context_length=16,patch_length=4,patch_stride=4,num_input_channels=1,
        d_model=16,num_hidden_layers=2,num_attention_heads=2,ffn_dim=32,
        share_embedding=True,channel_attention=False,use_cls_token=True,scaling='std',positional_encoding_type='sincos')
    return HybridPatchEncoder(3,5,20,public_config=cfg.to_dict(),d_model=12,nhead=3,layers=1,dropout=0.)


class HybridPatchTSTTests(unittest.TestCase):
    def test_backup_original_architecture_and_zero_residual_exact_equivalence(self):
        import hashlib
        # AST digest measured against the pre-public backup; no private outputs
        # or ignored backup dependency is required to run this regression.
        source=(ROOT/'src/pecvd/legacy_patchtst.py').read_text(encoding='utf-8')
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PatchEncoder')
        self.assertEqual(hashlib.sha256(ast.dump(cls,include_attributes=False).encode()).hexdigest(),
                         '67ef304f95ab054dca356d0845bca1e199dac52a9ec610bc7eb765464cbefd5d')
        env={'torch':torch,'nn':torch.nn};exec(compile(ast.Module([cls],type_ignores=[]),'backup:original-class','exec'),env)
        model=tiny_hybrid().eval();original=env['PatchEncoder'](3,5,12,3,1,0.).eval()
        original.load_state_dict(model.legacy.state_dict(),strict=True)
        x=torch.randn(2,20,3);mask=torch.zeros(2,12,dtype=torch.bool);mask[:,2]=True
        torch.testing.assert_close(model.tokens(x,mask),original.tokens(x,mask),rtol=0,atol=0)
        torch.testing.assert_close(model(x,mask),original(x,mask),rtol=0,atol=0)
        self.assertEqual(model.patch_len,5);self.assertEqual(model.d_model,12)

    def test_other_sensor_is_used_and_hidden_patch_is_not_leaked(self):
        model=tiny_hybrid().eval();model.public_gain.data.fill_(.1)
        x=torch.randn(2,20,3);mask=torch.zeros(2,12,dtype=torch.bool);mask[:,2]=True
        changed=x.clone();changed[:,10:15,0]+=1000
        torch.testing.assert_close(model(x,mask),model(changed,mask))
        other=x.clone();other[:,:,1]+=4
        self.assertGreater(float((model(x,mask)[:,2]-model(other,mask)[:,2]).abs().max()),1e-6)

    def test_public_adapter_sensor_embedding_and_joint_attention_receive_gradients(self):
        model=tiny_hybrid();model.public_gain.data.fill_(.1);model.train()
        loss=model(torch.randn(2,20,3)).square().mean();loss.backward()
        self.assertGreater(float(model.legacy.sensor_emb.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.legacy.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(model.public_projection.weight.grad.abs().sum()),0)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.pretrained.encoder.layers[0].parameters()))
        self.assertTrue(all(p.grad is None for p in model.public.hf.head.parameters()))

    def test_exact_legacy_warmstart_and_mismatches_fail_closed(self):
        model=tiny_hybrid().eval();original=PatchEncoder(3,5,12,3,1,0.).eval()
        expected=dict(sensor_columns=['a','b','c'],mean=np.zeros(3),std=np.ones(3),run_split={'train':['r0']},source_sha256='invented')
        ck={**expected,'window_length':20,'patch_length':5,'model_config':{'d_model':12,'nhead':3,'layers':1,'dropout':0.},'model_state':original.state_dict()}
        model.load_legacy(ck,**expected)
        x=torch.randn(2,20,3);torch.testing.assert_close(model(x),original(x),rtol=0,atol=0)
        for key,value in [('sensor_columns',['b','a','c']),('run_split',{'train':['test']}),('mean',np.ones(3)),('source_sha256','other')]:
            bad=copy.deepcopy(ck);bad[key]=value
            with self.assertRaises(ValueError):model.load_legacy(bad,**expected)
        bad=copy.deepcopy(ck);bad['model_config']['nhead']=1
        with self.assertRaises(ValueError):model.load_legacy(bad,**expected)

    def test_offline_roundtrip_and_explicit_external_channel_adapter(self):
        model=tiny_hybrid().eval();model.public_gain.data.fill_(.1)
        ck={'encoder_backend':BACKEND,'sensor_columns':['a','b','c'],'patch_length':5,'window_length':20,
            'model_config':{'d_model':12,'nhead':3,'layers':1,'dropout':0.},'pretrained_config':model.hf_config,
            'pretrained_identity':{'fixture':True},'model_state':model.state_dict()}
        with patch('transformers.PatchTSTForPretraining.from_pretrained',side_effect=AssertionError('Offline trained reload only')):
            loaded=load_encoder(ck,None).eval()
        x=torch.randn(2,20,3);torch.testing.assert_close(model(x),loaded(x))
        adapted=FrozenChannelAdapter(loaded,5).eval();external=torch.randn(2,20,5)
        self.assertEqual(adapted.tokens(external).shape,(2,20,12))
        torch.testing.assert_close(adapted(external),loaded.channel_independent_forward(external))
        altered=external.clone();altered[:,:,1]+=10
        torch.testing.assert_close(adapted(external).reshape(2,5,4,5)[:,0],adapted(altered).reshape(2,5,4,5)[:,0])

    def test_validation_selection_warmup_and_patience_are_explicit(self):
        selection=ValidationSelection(min_epochs=10,patience=2,min_delta=.01)
        self.assertEqual(selection.observe(.5,0),(True,False))
        self.assertEqual(selection.observe(.6,5),(False,False))
        self.assertEqual(selection.observe(.6,10),(False,True))
        self.assertEqual(selection.best_epoch,0)
        self.assertEqual(selection.observe(.4,15),(True,False))
        self.assertEqual(selection.stale,0)
        self.assertEqual(selection.observe(.395,16),(True,False))
        self.assertEqual(selection.best_epoch,16)
        self.assertEqual(selection.stale,1)  # retain true minimum without resetting patience
        with self.assertRaises(ValueError):selection.observe(float('nan'),20)
        params=[torch.nn.Parameter(torch.zeros(())),torch.nn.Parameter(torch.zeros(()))]
        opt=torch.optim.SGD([{'params':[params[0]],'lr':1e-4},{'params':[params[1]],'lr':1e-5}])
        apply_warmup(opt,[1e-4,1e-5],1,5)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],2e-5)
        self.assertAlmostEqual(opt.param_groups[1]['lr'],2e-6)
        apply_warmup(opt,[1e-4,1e-5],5,5)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-4)
        opt.param_groups[0]['lr']=5e-5;apply_warmup(opt,[1e-4,1e-5],6,5)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],5e-5)

    def test_default_variant_restores_original_grid_and_392_numeric_width(self):
        from pecvd.model_variants import active_config,active_run_tag,selected_variant
        from pecvd.process_config import load_process_config
        original=load_process_config(ROOT/'config/processes/pecvd.json')
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'hybrid'},clear=True):
            self.assertEqual(selected_variant(),'hybrid')
            active=active_config(original)
            self.assertEqual(active.steps,original.steps)
            self.assertEqual(active_config(active),active)
            self.assertTrue(active.artifacts.encoder.endswith('_hf_patchtst_joint_hybrid_v1'))
            self.assertEqual(active_run_tag(),'real_step1_joint_hybrid_reference_v11')
        import sys,types
        doc=json.loads((ROOT/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c['id']=='01-config'))
        module=types.ModuleType('synthetic_hybrid_step2')
        module.PIPELINE_SETTINGS={'training_profile':'reference_v11'}
        with patch.dict(sys.modules,{module.__name__:module}),patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'hybrid'}),contextlib.redirect_stdout(io.StringIO()),patch.object(Path,'exists',return_value=False):
            exec(compile(source,'STEP2:hybrid-config','exec'),module.__dict__)
        self.assertEqual(module.CONFIG.numeric_dim,392)
        self.assertEqual(module.RUN_TAG,'real_step1_joint_hybrid_reference_v11')
        self.assertFalse(module.CONFIG.reuse_fusion_weights)

    def test_actual_notebook_hybrid_training_saves_reloads_and_restores_validation_best(self):
        self._exercise_notebook_training('hybrid')

    def test_actual_notebook_portable_training_saves_reloads_and_restores_validation_best(self):
        self._exercise_notebook_training('portable')

    def _exercise_notebook_training(self,variant):
        import sys,types,pandas as pd
        from dataclasses import replace
        doc=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        cells={c['id']:''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code'}
        module=types.ModuleType('synthetic_hybrid_training');env=module.__dict__
        with patch.dict(sys.modules,{module.__name__:module}),patch.dict('os.environ',{'GEVIS_PATCHTST_PATH':str(ROOT/'.cache/pretrained_patchtst')}),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            comparison_config=cells['01-config'].replace("NOTEBOOK_MODEL_VARIANT='hybrid'",f"NOTEBOOK_MODEL_VARIANT='{variant}'")
            exec(compile(comparison_config,'STEP1:hybrid-config','exec'),env)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/'fixture.txt';source.write_text('invented fixture only',encoding='utf-8')
                legacy_root=root/'original';(legacy_root/'encoders').mkdir(parents=True)
                cfg=replace(env['CONFIG'],epochs=2,min_epochs=0,warmup_epochs=1,batch_size=2,use_amp=False,fused_adamw=False,validation_interval=1)
                rng=np.random.default_rng(91);arrays={s:rng.normal(size=(4,20,3)).astype('float32') for s in ('train','validation','test')}
                split={s:[s+str(i) for i in range(4)] for s in arrays}
                mean=arrays['train'].mean((0,1));std=arrays['train'].std((0,1)).clip(1e-6)
                legacy=PatchEncoder(3,5,96,8,3,.05)
                ck={'sensor_columns':['a','b','c'],'window_length':20,'patch_length':5,'source_sha256':env['sha256_file'](source),
                    'mean':mean,'std':std,'run_split':split,'model_config':{'d_model':96,'nhead':8,'layers':3,'dropout':.05},'model_state':legacy.state_dict()}
                torch.save(ck,legacy_root/'encoders/Gap-up.pt')
                meta=pd.DataFrame({'process_id':['r0','r1','r2','r3'],'source_group':['fixture']*4,'pm_cycle':['fixture']*4,
                    'cohort_key':['fixture']*4,'window_start':[0]*4,'window_end':[20]*4,'position_ratio':[0.]*4,'position_bin':[0]*4})
                env.update(CONFIG=cfg,DEVICE=torch.device('cpu'),DATA_PATH=source,ARTIFACT_ROOT=root/'candidate',LEGACY_ENCODER_ROOT=legacy_root,
                    STEPS={'Gap-up':(20,5,5)},sensor_columns=['a','b','c'],sensor_descriptors={'a':'a','b':'b','c':'c'},
                    raw_windows={'Gap-up':arrays},window_metadata={'Gap-up':{s:meta for s in arrays}},run_split=split)
                funcs=[n for n in ast.parse(cells['01-fit']).body if isinstance(n,ast.FunctionDef)]
                exec(compile(ast.Module(funcs,type_ignores=[]),'STEP1:hybrid-fit','exec'),env)
                candidate=env['train_one_step']('Gap-up')
                self.assertEqual(candidate['encoder_backend'],'hf_patchtst_portable_joint_v1' if variant=='portable' else BACKEND)
                self.assertEqual(candidate['training_status']['best_epoch'],min(candidate['history'],key=lambda row:row['validation_mse'])['epoch'])
                self.assertLessEqual(candidate['training_status']['best_validation_mse'],candidate['training_status']['initial_validation_mse'])
                loaded=load_encoder(candidate,None).eval();x=torch.randn(1,20,3)
                self.assertEqual(loaded.tokens(x).shape,(1,12,96))
                validation=((arrays['validation']-mean)/std).astype('float32')
                restored_mse,_=env['evaluate_masked'](loaded,validation,999)
                self.assertAlmostEqual(restored_mse,candidate['training_status']['best_validation_mse'],places=5)
                from pecvd.gevis_ui import notebook_runtime_sources
                fn=next(n for n in ast.parse(notebook_runtime_sources(ROOT/'notebooks/STEP_3.1_run_gevis_inference.ipynb')['03-types']).body if isinstance(n,ast.FunctionDef) and n.name=='numeric_state')
                scope={'torch':torch};exec(compile(ast.Module([fn],type_ignores=[]),'STEP31:hybrid-state','exec'),scope)
                self.assertEqual(scope['numeric_state'](loaded,x,x).shape,(1,12,392))
                self.assertTrue((legacy_root/'encoders/Gap-up.pt').is_file())
                # Force validation regression and verify actual initial weights
                # are restored rather than merely reporting the minimum score.
                env['ARTIFACT_ROOT']=root/'forced-regression'
                scores=iter([.1,.2,.3,.4])
                env['evaluate_masked']=lambda *args:(next(scores),1.)
                fallback=env['train_one_step']('Gap-up')
                self.assertEqual(fallback['training_status']['best_epoch'],0)
                self.assertFalse(fallback['training_status']['improved_over_initial'])
                restored=load_encoder(fallback,None).eval()
                for key,value in legacy.state_dict().items():torch.testing.assert_close(restored.legacy.state_dict()[key],value,rtol=0,atol=0)
                self.assertEqual(float(restored.public_gain),0.)
                if variant=='portable':
                    torch.testing.assert_close(restored.shared_sensor_token,legacy.sensor_emb.weight.mean(0))
                    self.assertEqual(restored(torch.randn(1,20,5)).shape,(1,20,5))


if __name__=='__main__':unittest.main()
