import copy,unittest
from unittest.mock import patch
import torch
from transformers import PatchTSTConfig
from pecvd.hybrid_patchtst import PortableJointPatchEncoder,PORTABLE_BACKEND
from pecvd.pretrained_patchtst import load_encoder


def portable():
    cfg=PatchTSTConfig(context_length=16,patch_length=4,patch_stride=4,num_input_channels=1,
        d_model=16,num_hidden_layers=1,num_attention_heads=2,ffn_dim=32,
        share_embedding=True,channel_attention=False,use_cls_token=True,scaling='std')
    return PortableJointPatchEncoder(3,5,20,public_config=cfg.to_dict(),d_model=12,nhead=3,layers=1,dropout=0.).eval()


class PortableNativeTests(unittest.TestCase):
    def test_source_external_same_weights_joint_context_and_permutation(self):
        model=portable();before=copy.deepcopy(model.state_dict())
        for sensors in (2,3,5):
            x=torch.randn(2,20,sensors);out=model(x).reshape(2,sensors,4,5)
            perm=torch.randperm(sensors)
            torch.testing.assert_close(model(x[:,:,perm]).reshape_as(out),out[:,perm],rtol=1e-5,atol=1e-6)
            other=x.clone();other[:,:,1]+=7
            self.assertGreater(float((model(other).reshape_as(out)[:,0]-out[:,0]).abs().max()),1e-6)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_hidden_values_isolated_and_shared_embedding_trains(self):
        model=portable();model.public_gain.data.fill_(.1)
        x=torch.randn(2,20,5);mask=torch.zeros(2,20,dtype=torch.bool);mask[:,2]=True
        changed=x.clone();changed[:,10:15,0]+=100
        torch.testing.assert_close(model(x,mask),model(changed,mask))
        model.train();model(x,mask).square().mean().backward()
        self.assertGreater(float(model.shared_sensor_token.grad.abs().sum()),0)
        self.assertIsNone(model.legacy.sensor_emb.weight.grad)
        self.assertGreater(float(model.legacy.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()),0)

    def test_strict_offline_reload_keeps_variable_sensor_forward(self):
        model=portable();ck={'encoder_backend':PORTABLE_BACKEND,'sensor_columns':['a','b','c'],
            'patch_length':5,'window_length':20,'model_config':{'d_model':12,'nhead':3,'layers':1,'dropout':0.},
            'pretrained_config':model.hf_config,'pretrained_identity':{'fixture':True},'model_state':model.state_dict()}
        with patch('transformers.PatchTSTForPretraining.from_pretrained',side_effect=AssertionError('No network')):
            loaded=load_encoder(ck,None).eval()
        x=torch.randn(1,20,5);torch.testing.assert_close(loaded(x),model(x))
        self.assertTrue(loaded.supports_native_sensor_transfer)

    def test_fixed_sensor_old_model_is_rejected_not_silently_adapted(self):
        from types import SimpleNamespace
        from pecvd.experiments.frozen_transfer import bind_external_runtime
        runtime=SimpleNamespace(encoders={'old':(torch.nn.Linear(1,1),{})})
        with self.assertRaisesRegex(ValueError,'fixed sensor IDs'):
            bind_external_runtime('.',runtime,None,None)

    def test_default_namespace_preserves_original_and_has_consistent_numeric_width(self):
        from pecvd.model_variants import selected_variant,active_config,active_run_tag
        from pecvd.process_config import load_process_config
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'portable'},clear=True):
            self.assertEqual(selected_variant(),'portable')
            cfg=load_process_config(root/'config/processes/pecvd.json');candidate=active_config(cfg)
            self.assertEqual(candidate.steps,cfg.steps)
            self.assertEqual(active_config(candidate),candidate)
            self.assertTrue(candidate.artifacts.encoder.endswith('_hf_patchtst_portable_joint_v1'))
            self.assertEqual(active_run_tag(),'real_step1_portable_reference_v11')
        import json,types,sys,contextlib,io
        book=json.loads((root/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in book['cells'] if c.get('id')=='01-config'))
        module=types.ModuleType('synthetic_portable_step2')
        module.PIPELINE_SETTINGS={'training_profile':'reference_v11'}
        with patch.dict(sys.modules,{module.__name__:module}),patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'portable'}),patch.object(Path,'exists',return_value=False),contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source,'STEP2:portable-config','exec'),module.__dict__)
        self.assertEqual(module.CONFIG.numeric_dim,392)
        self.assertEqual(module.RUN_TAG,'real_step1_portable_reference_v11')
