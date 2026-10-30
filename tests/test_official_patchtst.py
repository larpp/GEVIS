import ast,contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import PatchTSTConfig,PatchTSTForPretraining,PatchTSTModel
from pecvd.official_patchtst import OfficialPatchEncoder,BACKEND
from pecvd.pretrained_patchtst import load_encoder
from pecvd.experiments.frozen_transfer import FrozenChannelAdapter

ROOT=Path(__file__).resolve().parents[1]


def tiny():
    cfg=PatchTSTConfig(context_length=24,patch_length=4,patch_stride=4,num_input_channels=1,
        d_model=16,num_hidden_layers=2,num_attention_heads=2,ffn_dim=32,
        share_embedding=True,channel_attention=False,use_cls_token=True,scaling='std',mask_type='random',random_mask_ratio=.5)
    model=OfficialPatchEncoder(3,hf_config=cfg.to_dict());model.configure_finetuning(1)
    return model


class OfficialPatchTSTTests(unittest.TestCase):
    def test_step1_warm_bootstrap_and_preflight_before_training(self):
        import os,sys,types
        from dataclasses import replace
        import pecvd.model_variants as variants
        doc=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        cells={c['id']:''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code'}
        module=types.ModuleType('synthetic_bootstrap');env=module.__dict__
        sys.modules[module.__name__]=module
        self.addCleanup(sys.modules.pop,module.__name__,None)
        with patch.dict(os.environ,{'GEVIS_MODEL_VARIANT':'pretrained'}), contextlib.redirect_stdout(io.StringIO()):
            # Simulate a cached helper from before the official default existed.
            original=variants.selected_variant
            variants.selected_variant=lambda:'pretrained'
            try:
                exec(compile(cells['01-config'],'STEP1:config','exec'),env)
                self.assertEqual(env['CONFIGURED_MODEL_VARIANT'],'hybrid')
                self.assertEqual(env['CONFIG'].d_model,96)
                self.assertTrue(str(env['ARTIFACT_ROOT']).endswith('_hf_patchtst_joint_hybrid_v1'))
                env['CONFIG']=replace(env['CONFIG'],reuse_legacy_checkpoint=False)
                fn=next(n for n in ast.parse(cells['01-fit']).body if isinstance(n,ast.FunctionDef) and n.name=='preflight_training')
                exec(compile(ast.Module([fn],type_ignores=[]),'STEP1:preflight','exec'),env)
                with tempfile.TemporaryDirectory() as d:
                    env['ROOT']=Path(d)
                    env['ARTIFACT_ROOT']=Path(d)/'artifacts'/env['ARTIFACT_VERSION']
                    env['preflight_training']()
                    target=env['ARTIFACT_ROOT']/'encoders'/f'{next(iter(env["STEPS"]))}.pt'
                    target.parent.mkdir(parents=True);target.touch()
                    with self.assertRaisesRegex(FileExistsError,'training has NOT started'):
                        env['preflight_training']()
                    self.assertEqual(target.stat().st_size,0)
                    env['CONFIG']=replace(env['CONFIG'],overwrite_artifacts=True)
                    env['preflight_training']()
                    env['ARTIFACT_ROOT']=Path(d)/'old_variant'
                    with self.assertRaisesRegex(RuntimeError,'Artifact root'):
                        env['preflight_training']()
                    os.environ['GEVIS_MODEL_VARIANT']='pretrained'
                    with self.assertRaisesRegex(RuntimeError,'variant changed'):
                        env['preflight_training']()
                tree=ast.parse(cells['01-fit'])
                branch=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='RUN_TRAINING')
                self.assertEqual(ast.unparse(branch.body[0]),'preflight_training()')
            finally:
                variants.selected_variant=original

    def test_official_forward_matches_tokens_and_restored_reconstruction(self):
        model=tiny().eval();x=torch.randn(2,24,3)
        flat=x.transpose(1,2).reshape(6,24,1)
        with patch.object(model.features,'forward',wraps=model.features.forward) as called:
            tokens=model.tokens(x)
        self.assertTrue(called.called)
        direct=model.features(past_values=flat,return_dict=True)
        expected=direct.last_hidden_state[:,:,1:].reshape(2,18,16)
        torch.testing.assert_close(tokens,expected)
        reconstruction=model.hf.head(direct.last_hidden_state)*direct.scale.unsqueeze(-1)+direct.loc.unsqueeze(-1)
        torch.testing.assert_close(model(x),reconstruction.reshape(2,18,4))
        self.assertIsInstance(model.hf,PatchTSTForPretraining)
        self.assertIsInstance(model.features,PatchTSTModel)
        self.assertIs(model.features.encoder,model.hf.model.encoder)
        self.assertFalse(any(hasattr(model,k) for k in ('projection','local_statistics','mask_token','sensor_emb')))

    def test_blind_masks_hide_values_before_official_scaling(self):
        model=tiny().eval();x=torch.randn(2,24,3)
        mask=torch.zeros(2,18,dtype=torch.bool);mask[:,2]=True
        changed=x.clone();changed[:,8:12,0]+=1000
        torch.testing.assert_close(model.tokens(x,mask),model.tokens(changed,mask))
        torch.testing.assert_close(model(x,mask),model(changed,mask))

    def test_official_ssl_loss_and_frozen_buffer_gradient_contract(self):
        model=tiny();model.train();before=copy.deepcopy(model.pretrained.encoder.layers[0].state_dict())
        with patch.object(model.hf,'forward',wraps=model.hf.forward) as called:
            loss=model.pretraining_loss(torch.randn(2,24,3),42)
        self.assertTrue(called.called);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.hf.head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.pretrained.encoder.layers[-1].parameters()))
        for key,value in before.items():torch.testing.assert_close(value,model.pretrained.encoder.layers[0].state_dict()[key])

    def test_offline_checkpoint_and_external_channels_keep_reconstruction_units(self):
        model=tiny().eval();x=torch.randn(2,24,3)
        ck={'encoder_backend':BACKEND,'sensor_columns':['a','b','c'],'patch_length':4,
            'model_config':{'d_model':16},'pretrained_config':model.hf_config,
            'pretrained_identity':{'synthetic':True},'model_state':model.state_dict()}
        with patch('transformers.PatchTSTForPretraining.from_pretrained',side_effect=AssertionError('No network')):
            loaded=load_encoder(ck,None).eval()
        torch.testing.assert_close(model(x),loaded(x))
        external=FrozenChannelAdapter(loaded,5).eval();values=torch.randn(2,24,5)
        torch.testing.assert_close(external(values),loaded.channel_independent_forward(values))
        self.assertEqual(external.tokens(values).shape,(2,30,16))

    def test_default_variant_has_official_grid_without_mutating_source(self):
        from pecvd.model_variants import active_config,active_run_tag,selected_variant
        from pecvd.process_config import load_process_config
        source=load_process_config(ROOT/'config/processes/pecvd.json')
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'official'},clear=True):
            target=active_config(source)
            self.assertEqual(selected_variant(),'official')
            self.assertTrue(target.artifacts.encoder.endswith('_hf_patchtst_official_domain_v2'))
            self.assertEqual(active_run_tag(),'real_step1_official_reference_v11')
            self.assertTrue(all(s.patch_length==12 and s.window_length%12==0 for s in target.steps))
            self.assertTrue(all(a.window_length<=b.window_length for a,b in zip(target.steps,source.steps)))
            self.assertEqual(active_config(target),target)

    def test_actual_step2_config_selects_520_width_for_official_encoder(self):
        import sys,types
        doc=json.loads((ROOT/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c['id']=='01-config'))
        module=types.ModuleType('synthetic_official_step2')
        module.PIPELINE_SETTINGS={'training_profile':'reference_v11'}
        with patch.dict(sys.modules,{module.__name__:module}), \
             patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'official'}), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source,'STEP2:official-config','exec'),module.__dict__)
        self.assertEqual(module.CONFIG.numeric_dim,520)
        self.assertFalse(module.CONFIG.reuse_fusion_weights)
        self.assertEqual(module.RUN_TAG,'real_step1_official_reference_v11')

    def test_actual_step1_epoch_calls_domain_loss_through_official_model(self):
        import numpy as np
        from types import SimpleNamespace
        from torch.utils.data import DataLoader
        doc=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c['id']=='01-fit'))
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run_epoch')
        import pecvd.patchtst_adaptation as adaptation
        cfg=SimpleNamespace(batch_size=2,num_workers=0,use_amp=False,max_grad_norm=.5,
                            variability_weight=.25,slope_weight=.25,peak_weight=.1,relationship_weight=.1)
        env={'torch':torch,'nn':torch.nn,'DataLoader':DataLoader,'DEVICE':torch.device('cpu'),'AMP_DTYPE':torch.bfloat16,'CONFIG':cfg,
             'patchtst_adaptation':adaptation,
             'reconstruction_mask':lambda model,b,p,*args:torch.zeros(b,model.n_sensors*p,dtype=torch.bool)}
        exec(compile(ast.Module([fn],type_ignores=[]),'STEP1:official-epoch','exec'),env)
        model=tiny();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        loss,gradient=env['run_epoch'](model,np.random.default_rng(4).normal(size=(4,24,3)).astype('float32'),optimizer,torch.amp.GradScaler('cuda',enabled=False))
        self.assertTrue(np.isfinite(loss));self.assertTrue(np.isfinite(gradient))

    def test_actual_public_encoder_weights_are_retained_except_fixed_positions(self):
        from transformers import PatchTSTForPrediction
        model=OfficialPatchEncoder.from_public(ROOT/'.cache/pretrained_patchtst',3,36).eval()
        public=PatchTSTForPrediction.from_pretrained(ROOT/'.cache/pretrained_patchtst',local_files_only=True)
        for name,value in public.model.encoder.state_dict().items():
            if name!='positional_encoder.position_enc':torch.testing.assert_close(model.pretrained.encoder.state_dict()[name],value)
        self.assertEqual(model.tokens(torch.randn(1,36,3)).shape,(1,9,128))
        self.assertIn('NOT pretrained',model.pretrained_identity['task_head'])

    def test_incomplete_learned_encoder_load_is_rejected(self):
        with patch('transformers.PatchTSTForPretraining.from_pretrained',return_value=(None,
                {'missing_keys':['model.encoder.layers.0.self_attn.q_proj.weight'],
                 'mismatched_keys':[],'unexpected_keys':[],'error_msgs':[]})):
            with self.assertRaisesRegex(ValueError,'encoder mismatch'):
                OfficialPatchEncoder.from_public(ROOT/'.cache/pretrained_patchtst',3,36)

    def test_actual_step1_public_epoch_saves_official_checkpoint(self):
        import numpy as np,pandas as pd,sys,types
        from dataclasses import replace
        doc=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        cells={c['id']:''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code'}
        module=types.ModuleType('synthetic_official_step1');env=module.__dict__
        with patch.dict(sys.modules,{module.__name__:module}), \
             patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'official','GEVIS_PATCHTST_PATH':str(ROOT/'.cache/pretrained_patchtst')}), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            official_config=cells['01-config'].replace("NOTEBOOK_MODEL_VARIANT='hybrid'", "NOTEBOOK_MODEL_VARIANT='official'")
            exec(compile(official_config,'STEP1:official-config','exec'),env)
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);source=root/'invented.csv';source.write_text('synthetic only\n',encoding='utf-8')
                cfg=replace(env['CONFIG'],epochs=1,batch_size=2,use_amp=False,fused_adamw=False,validation_interval=1)
                meta=pd.DataFrame({'process_id':['r0','r1','r2','r3'],'source_group':['fixture']*4,
                    'pm_cycle':['fixture']*4,'cohort_key':['fixture']*4,'window_start':[0]*4,'window_end':[48]*4,
                    'position_ratio':[0.]*4,'position_bin':[0]*4})
                rng=np.random.default_rng(12)
                env.update(CONFIG=cfg,DEVICE=torch.device('cpu'),DATA_PATH=source,ARTIFACT_ROOT=root/'artifacts',
                    STEPS={'Gap-up':(48,12,12)},sensor_columns=['a','b','c'],sensor_descriptors={'a':'a','b':'b','c':'c'},
                    raw_windows={'Gap-up':{s:rng.normal(size=(4,48,3)).astype('float32') for s in ('train','validation','test')}},
                    window_metadata={'Gap-up':{s:meta.assign(process_id=[s+str(i) for i in range(4)]) for s in ('train','validation','test')}},
                    run_split={s:[s+str(i) for i in range(4)] for s in ('train','validation','test')})
                functions=[n for n in ast.parse(cells['01-fit']).body if isinstance(n,ast.FunctionDef)]
                exec(compile(ast.Module(functions,type_ignores=[]),'STEP1:official-fit','exec'),env)
                ck=env['train_one_step']('Gap-up')
                self.assertEqual(ck['encoder_backend'],BACKEND)
                self.assertIn('blind_variability_preservation',ck['self_supervised_objectives'])
                self.assertEqual(ck['adaptation_contract']['selection'],'validation blind_mse; never test')
                self.assertEqual(ck['adaptation_contract']['encoder_scope'],'all learned encoder parameters')
                self.assertTrue(all('validation_variability' in row for row in ck['history']))
                loaded=load_encoder(ck,None).eval()
                self.assertEqual(loaded.tokens(torch.randn(1,48,3)).shape,(1,12,128))
                from pecvd.gevis_ui import notebook_runtime_sources
                tree=ast.parse(notebook_runtime_sources(ROOT/'notebooks/STEP_3.1_run_gevis_inference.ipynb')['03-types'])
                fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='numeric_state')
                target={'torch':torch};exec(compile(ast.Module([fn],type_ignores=[]),'STEP31:official-residual','exec'),target)
                state=target['numeric_state'](loaded,torch.randn(1,48,3),torch.randn(1,48,3))
                self.assertEqual(state.shape,(1,12,520))
                self.assertTrue(torch.isfinite(state).all())


if __name__=='__main__':unittest.main()
