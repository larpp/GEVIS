import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import PatchTSTConfig
from pecvd.pretrained_patchtst import PretrainedPatchEncoder,load_encoder,detail_loss,BACKEND
from pecvd.experiments.frozen_transfer import FrozenChannelAdapter
from pecvd.model_variants import active_config,active_run_tag

def invented_model(sensors=3,patch=4):
    # Random tiny model ONLY for shape/gradient tests; never labelled pretrained.
    config=PatchTSTConfig(context_length=24,patch_length=4,patch_stride=4,d_model=16,
        num_hidden_layers=2,num_attention_heads=2,ffn_dim=32,num_input_channels=7,
        share_embedding=True,channel_attention=False,use_cls_token=True,norm_type='batchnorm')
    return PretrainedPatchEncoder(sensors,patch,8,hf_config=config.to_dict())

class PretrainedPatchTSTTests(unittest.TestCase):
    def test_native_tokens_variable_channels_and_permutation(self):
        model=invented_model().eval(); x=torch.randn(2,24,3)
        self.assertEqual(model.tokens(x).shape,(2,18,8))
        permutation=[2,0,1]
        expected=model.tokens(x).reshape(2,3,6,8)[:,permutation]
        torch.testing.assert_close(model.tokens(x[:,:,permutation]).reshape(2,3,6,8),expected)
        adapter=FrozenChannelAdapter(model,5).eval()
        self.assertEqual(adapter.tokens(torch.randn(2,24,5)).shape,(2,30,8))
        with self.assertRaises(ValueError): model.tokens(torch.randn(2,24,5))

    def test_mask_hides_values_from_embedding_and_statistics(self):
        model=invented_model().eval(); x=torch.randn(2,24,3)
        mask=torch.zeros(2,18,dtype=torch.bool); mask[:,2]=True
        changed=x.clone(); changed[:,8:12,0]+=1000
        torch.testing.assert_close(model.tokens(x,mask),model.tokens(changed,mask))

    def test_frozen_layers_and_statistics_do_not_drift_and_last_layer_trains(self):
        model=invented_model(); model.configure_finetuning(1); model.train()
        before=copy.deepcopy(model.pretrained.layers[0].state_dict())
        model(torch.randn(2,24,3)).square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.pretrained.layers[-1].parameters()))
        for key,value in before.items(): torch.testing.assert_close(model.pretrained.layers[0].state_dict()[key],value)
        self.assertFalse(any(p.grad is not None for p in model.pretrained.layers[0].parameters()))

    def test_offline_checkpoint_roundtrip_never_loads_public_weights(self):
        model=invented_model().eval(); x=torch.randn(1,24,3)
        ck={'encoder_backend':BACKEND,'sensor_columns':['a','b','c'],'patch_length':4,
            'model_config':{'d_model':8},'pretrained_config':model.hf_config,
            'pretrained_identity':{'synthetic':True},'model_state':model.state_dict()}
        with patch('transformers.PatchTSTForPrediction.from_pretrained',side_effect=AssertionError('Network forbidden')):
            loaded=load_encoder(ck,None).eval()
        torch.testing.assert_close(model(x),loaded(x))

    def test_detail_objectives_have_finite_local_gradients(self):
        target=torch.zeros(2,6,4); pred=target.clone(); pred[0,0,1]=5; pred.requires_grad_()
        variance,peak=detail_loss(pred,target)
        self.assertGreater(float(variance),0); self.assertGreater(float(peak),0)
        (variance+peak).backward(); self.assertTrue(torch.isfinite(pred.grad).all())

    def test_actual_step1_epoch_trains_rich_ssl_on_invented_values(self):
        import ast,json,numpy as np
        from types import SimpleNamespace
        from torch.utils.data import DataLoader
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='01-fit'))
        fn=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='run_epoch')
        cfg=SimpleNamespace(batch_size=2,num_workers=0,use_amp=False,clean_weight=.5,slope_weight=.25,
            relationship_weight=.1,variability_weight=.25,peak_weight=.1,max_grad_norm=.5)
        env={'torch':torch,'nn':torch.nn,'DataLoader':DataLoader,'DEVICE':torch.device('cpu'),'AMP_DTYPE':torch.bfloat16,'CONFIG':cfg,
            'reconstruction_mask':lambda model,b,p,*args:torch.arange(model.n_sensors*p).remainder(2).eq(0)[None].expand(b,-1)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'STEP1:synthetic-rich-epoch','exec'),env)
        model=invented_model(); model.configure_finetuning(1)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        loss,gradient=env['run_epoch'](model,np.random.default_rng(4).normal(size=(4,24,3)).astype('float32'),optimizer,scaler)
        self.assertTrue(np.isfinite(loss)); self.assertTrue(np.isfinite(gradient))

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'.cache/pretrained_patchtst/model.safetensors').is_file(),'Public weights not downloaded')
    def test_actual_pinned_public_weight_loading_is_not_random_initialization(self):
        root=Path(__file__).resolve().parents[1]
        model=PretrainedPatchEncoder.from_public(root/'.cache/pretrained_patchtst',3,5).eval()
        self.assertEqual(model.pretrained_identity['revision'],'7fe295d8bc8fbac8041b60ab351882634165517f')
        self.assertEqual(model.tokens(torch.randn(1,40,3)).shape,(1,24,96))
        self.assertFalse(any(p.requires_grad for p in model.pretrained.parameters()))

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'.cache/pretrained_patchtst/model.safetensors').is_file(),'Public weights not downloaded')
    def test_actual_step1_public_adaptation_saves_a_loadable_native_checkpoint(self):
        import ast,json,numpy as np,pandas as pd
        from dataclasses import replace
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        code={c['id']:''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code'}
        import sys,types
        module=types.ModuleType('synthetic_public_training')
        sys.modules[module.__name__]=module
        self.addCleanup(sys.modules.pop,module.__name__,None)
        env=module.__dict__
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'pretrained','GEVIS_PATCHTST_PATH':str(root/'.cache/pretrained_patchtst')}):
            comparison_config=code['01-config'].replace("NOTEBOOK_MODEL_VARIANT='hybrid'", "NOTEBOOK_MODEL_VARIANT='pretrained'")
            exec(compile(comparison_config,'STEP1:public-config','exec'),env)
            with tempfile.TemporaryDirectory() as folder:
                directory=Path(folder); source=directory/'invented.csv'; source.write_text('synthetic fixture only\n',encoding='utf-8')
                config=replace(env['CONFIG'],epochs=1,batch_size=2,use_amp=False,fused_adamw=False,validation_interval=1)
                metadata=pd.DataFrame({'process_id':['r0','r1','r2','r3'],'source_group':['fixture']*4,
                    'pm_cycle':['fixture']*4,'cohort_key':['fixture']*4,'window_start':[0]*4,'window_end':[40]*4,
                    'position_ratio':[0.]*4,'position_bin':[0]*4})
                rng=np.random.default_rng(41)
                env.update(CONFIG=config,DEVICE=torch.device('cpu'),DATA_PATH=source,ARTIFACT_ROOT=directory/'artifacts',
                    STEPS={'Gap-up':(40,5,5)},sensor_columns=['a','b','c'],sensor_descriptors={'a':'a','b':'b','c':'c'},
                    raw_windows={'Gap-up':{s:rng.normal(size=(4,40,3)).astype('float32') for s in ('train','validation','test')}},
                    window_metadata={'Gap-up':{s:metadata.assign(process_id=[s+str(i) for i in range(4)]) for s in ('train','validation','test')}},
                    run_split={s:[s+str(i) for i in range(4)] for s in ('train','validation','test')})
                functions=[n for n in ast.parse(code['01-fit']).body if isinstance(n,ast.FunctionDef)]
                exec(compile(ast.Module(functions,type_ignores=[]),'STEP1:public-fit-definitions','exec'),env)
                ck=env['train_one_step']('Gap-up')
                self.assertEqual(ck['encoder_backend'],BACKEND)
                self.assertIn('patch_variability_preservation',ck['self_supervised_objectives'])
                loaded=load_encoder(ck,env['PatchEncoder']).eval()
                self.assertEqual(loaded.tokens(torch.randn(1,40,3)).shape,(1,24,96))
                with np.load(directory/'artifacts/windows/Gap-up.npz') as cache:
                    self.assertFalse(set(cache['train_process_id'])&set(cache['test_process_id']))
                (directory/'artifacts/encoder_manifest.json').write_text(json.dumps({'step_checkpoints':{
                    'Gap-up':{'path':'encoders/Gap-up.pt'}}}),encoding='utf-8')
                env['display']=lambda *args:None
                exec(compile(code['encoder-sensitivity-audit'],'STEP1:public-audit','exec'),env)
                self.assertEqual(len(env['ENCODER_SENSITIVITY']),1)
                from pecvd.gevis_backbone import GEVISBackbone,token_descriptions
                a=loaded.tokens(torch.randn(1,40,3)); b=loaded.tokens(torch.randn(1,40,3))
                state=torch.cat([a,b,a-b,(a-b).abs(),torch.zeros(1,24,8)],-1)
                backbone=GEVISBackbone(392,32,0.).eval()
                _,weights=backbone(state,['Sensor a, full interval.'],['BOOLEAN'],[token_descriptions('phase',['a','b','c'],8)])
                self.assertEqual(weights.shape,(1,24))

    def test_variant_is_explicit_and_does_not_mutate_source_config(self):
        from pecvd.process_config import load_process_config
        root=Path(__file__).resolve().parents[1]; original=load_process_config(root/'config/processes/pecvd.json')
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'pretrained'}):
            selected=active_config(original)
            self.assertTrue(selected.artifacts.encoder.endswith('_hf_patchtst_v1'))
            self.assertNotEqual(selected.artifacts.encoder,original.artifacts.encoder)
            self.assertEqual(active_config(selected),selected)
            self.assertEqual(active_run_tag(),'real_step1_pretrained_reference_v11')
        with patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'legacy'}): self.assertIs(active_config(original),original)

    def test_joint_binding_and_paraphrase_gradients(self):
        from pecvd import gevis_backbone as gb
        packed=gb.real_step1_cross_batch(torch.randn(1,12,8),'phase',['flow_a','flow_b','pressure'],
            seed=3,binding_refinement=True,runtime_wording=True,rename_sensors=True,permute_sensors=True)
        numeric,questions,kinds,descriptions,feature,targets=packed
        alternate=gb.binding_paraphrases('phase',descriptions,feature,targets,kinds)
        self.assertEqual(len(alternate),len(questions))
        for i,text in enumerate(alternate):
            self.assertNotIn(f'channel_3_{int(targets["sensor"][i]):03d}_signal',text)
        logits=torch.randn(len(questions),12,requires_grad=True)
        a=logits.softmax(-1); b=(logits+.1*torch.randn_like(logits)).softmax(-1)
        loss=gb.joint_binding_loss(a,targets)+gb.attention_consistency_loss(a,b)
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0)
        self.assertAlmostEqual(float(gb.attention_consistency_loss(a,a)),0)

if __name__=='__main__': unittest.main()
