import ast,contextlib,hashlib,io,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import PatchTSTForPretraining,PatchTSTModel
from pecvd.encoder_loading import load_encoder
from pecvd.legacy_patchtst import PatchEncoder
from test_hybrid_patchtst import tiny_hybrid

ROOT=Path(__file__).resolve().parents[1]
RUNTIME_FIXTURE={}


class EnglishBackboneSourceTests(unittest.TestCase):
    def test_step31_entries_refresh_stale_answer_modules_before_binding(self):
        import pecvd.gevis_fast as fast
        import pecvd.answer_display as text
        for cell_id in ('03-types','03-load'):
            source=cells('STEP_3.1_run_gevis_inference.ipynb')[cell_id]
            start=source.index('import importlib\nimport pecvd.answer_display as answer_text')
            end=source.index('gevis_fast=importlib.reload(gevis_fast)',start)+len('gevis_fast=importlib.reload(gevis_fast)')
            saved_fast=fast.DEFAULT_ANSWER_CRITERIA;saved_text=text.ANSWER_CRITERIA
            del fast.DEFAULT_ANSWER_CRITERIA
            del text.ANSWER_CRITERIA
            try:
                scope={}
                exec(compile(source[start:end],cell_id+':refresh','exec'),scope)
                self.assertIs(scope['gevis_fast'].DEFAULT_ANSWER_CRITERIA,text.ANSWER_CRITERIA)
                self.assertEqual(text.ANSWER_CRITERIA['BOOLEAN'],('No','Yes'))
            finally:
                if not hasattr(fast,'DEFAULT_ANSWER_CRITERIA'):fast.DEFAULT_ANSWER_CRITERIA=saved_fast
                if not hasattr(text,'ANSWER_CRITERIA'):text.ANSWER_CRITERIA=saved_text

    def test_only_audited_english_source_successor_is_compatible(self):
        from pecvd.encoder_loading import backbone_source_matches
        source=ROOT/'src/pecvd/gevis_backbone.py'
        checkpoint={'source_module_sha256':'2e68ede51e97aca5a9379b16b029b6ce5d8b900a36ebe6637e358f1270859b81','question_language':'English'}
        before=dict(checkpoint)
        self.assertTrue(backbone_source_matches(checkpoint,source))
        self.assertEqual(checkpoint,before)
        self.assertFalse(backbone_source_matches(dict(checkpoint,question_language='other'),source))
        self.assertFalse(backbone_source_matches(dict(checkpoint,source_module_sha256='unknown'),source))
        with tempfile.TemporaryDirectory() as directory:
            altered=Path(directory)/'source.py'
            altered.write_bytes(source.read_bytes()+b'\n# unreviewed edit\n')
            self.assertFalse(backbone_source_matches(checkpoint,altered))


def cells(name):
    return {c['id']:''.join(c['source']) for c in json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code'}


def stale_factory():
    import pecvd.pretrained_patchtst as factory
    def old_loader(*args):raise ValueError('Unknown STEP 1 encoder backend')
    factory.load_encoder=old_loader
    return factory


class TinyGEVIS(nn.Module):
    def __init__(self,numeric_dim,d_model,*unused):
        super().__init__()
        self.text=nn.Linear(1,1);self.numeric_input_norm=nn.LayerNorm(numeric_dim)
        self.numeric=nn.Linear(numeric_dim,d_model);self.numeric_key_norm=nn.LayerNorm(d_model)
        self.descriptor_norm=nn.LayerNorm(d_model);self.fusion_gate=nn.Linear(2*d_model,d_model)
        self.type_emb=nn.Embedding(3,d_model);self.cross=nn.MultiheadAttention(d_model,4,batch_first=True)
        self.answer_norm=nn.LayerNorm(d_model);self.support_head=nn.Linear(d_model,1)
        self.semantic_heads=nn.ModuleDict({'fixture':nn.Linear(d_model,2)})


def fixture(root):
    from pecvd.process_config import load_process_config
    base=load_process_config(ROOT/'config/processes/pecvd.json')
    cfg=replace(base,steps=(replace(base.steps[0],name='fixture',window_length=20,patch_length=5,stride=5),),
                artifacts=replace(base.artifacts,encoder='fixture_hf_patchtst_joint_hybrid_v1'))
    folder=root/'artifacts'/cfg.artifacts.encoder
    (folder/'encoders').mkdir(parents=True);(folder/'windows').mkdir()
    model=tiny_hybrid().eval()
    ck={'schema':'gevis.encoder.rich_ssl.v1','encoder_backend':'hf_patchtst_legacy_joint_v1',
        'sensor_columns':['a','b','c'],'patch_length':5,'window_length':20,
        'model_config':{'d_model':12,'nhead':3,'layers':1,'dropout':0.},
        'pretrained_config':model.hf_config,'pretrained_identity':{'synthetic':True},'model_state':model.state_dict()}
    path=folder/'encoders/fixture.pt';torch.save(ck,path)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest={'schema':'gevis.encoder_manifest.rich_ssl.v1','step_checkpoints':{'fixture':{
        'path':'encoders/fixture.pt','sha256':sha(path),'step_config':{'window_length':20,'patch_length':5,'stride':5}}}}
    (folder/'encoder_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    payload={}
    for split in ('train','validation','test'):
        payload[split]=np.zeros((4,20,3),dtype='float32')
        for column in ('process_id','source_group','pm_cycle','cohort_key','window_start','window_end','position_ratio','position_bin'):
            payload[f'{split}_{column}']=np.asarray([split+str(i) for i in range(4)]) if column=='process_id' else np.zeros(4,dtype='int64')
    np.savez(folder/'windows/fixture.npz',**payload)
    env={'ROOT':root,'Path':Path,'torch':torch,'nn':nn,'np':np,'pd':pd,'json':json,'hashlib':hashlib,
         'ENCODER_ROOT':folder,'PROCESS_CONFIG_PATH':root/'fixture.json','PROCESS_CONFIG':cfg,
         'STEPS':{'fixture':(20,5,5)},'CONFIG':SimpleNamespace(numeric_dim=56),'DEVICE':torch.device('cpu'),
         'display':lambda *args:None,'sha256_file':sha,'PatchEncoder':PatchEncoder}
    return model,ck,cfg,env


class EncoderLoadingTests(unittest.TestCase):
    def test_all_retained_backends_load_offline_through_the_same_boundary(self):
        from test_official_patchtst import tiny
        from test_pretrained_patchtst import invented_model
        models=[('legacy',PatchEncoder(3,4,12,3,1,0.).eval(),{'d_model':12,'nhead':3,'layers':1,'dropout':0.}),
                ('hf_patchtst_official_ssl_v1',tiny().eval(),{'d_model':16}),
                ('hf_patchtst_rich_ssl_v1',invented_model().eval(),{'d_model':8})]
        for backend,model,cfg in models:
            ck={'encoder_backend':backend,'sensor_columns':['a','b','c'],'patch_length':4,'model_config':cfg,'model_state':model.state_dict()}
            if backend!='legacy':ck.update(pretrained_config=model.hf_config,pretrained_identity={'synthetic':True})
            stale_factory()
            with patch('transformers.PatchTSTForPretraining.from_pretrained',side_effect=AssertionError('No download')),patch('transformers.PatchTSTForPrediction.from_pretrained',side_effect=AssertionError('No download')):
                restored=load_encoder(ck).eval()
            x=torch.zeros(1,24,3)
            torch.testing.assert_close(restored(x),model(x))

    def test_stale_unknown_factory_is_repaired_before_hybrid_load_and_unknown_stays_error(self):
        model=tiny_hybrid().eval()
        ck={'encoder_backend':'hf_patchtst_legacy_joint_v1','sensor_columns':['a','b','c'],
            'patch_length':5,'window_length':20,'model_config':{'d_model':12,'nhead':3,'layers':1,'dropout':0.},
            'pretrained_config':model.hf_config,'pretrained_identity':{},'model_state':model.state_dict()}
        old=stale_factory()
        with self.assertRaisesRegex(ValueError,'Unknown'):old.load_encoder(ck,None)
        with patch('transformers.PatchTSTForPretraining.from_pretrained',side_effect=AssertionError('Offline reload only')):
            loaded=load_encoder(ck).eval()
        torch.testing.assert_close(model(torch.zeros(1,20,3)),loaded(torch.zeros(1,20,3)))
        with self.assertRaisesRegex(ValueError,'unsupported_fixture'):
            load_encoder({**ck,'encoder_backend':'unsupported_fixture'})
        corrupted={**ck,'model_state':dict(ck['model_state'])}
        corrupted['model_state'].pop(next(iter(corrupted['model_state'])))
        with self.assertRaises(RuntimeError):load_encoder(corrupted)

    def test_actual_step2_prefix_and_full_step3_load_cell_repair_stale_factory(self):
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'GEVIS_MODEL_VARIANT':'hybrid'}),contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);model,ck,cfg,env=fixture(root)
            source=cells('STEP_2_train_gevis_backbone.ipynb')['07-cross-training']
            from tests.notebook_branch import training_source
            source=training_source(source)
            tree=ast.parse(source);prefix=[]
            for node in tree.body:
                if isinstance(node,ast.ClassDef) and node.name=='RealStep1Pairs':break
                prefix.append(node)
            stale_factory()
            stage2={**env,'SESSION_SIGNATURE':'STEP2_GEVIS_GENERIC_DISCOVERY_V10'}
            with patch('pecvd.process_config.load_process_config',return_value=cfg):
                exec(compile(ast.Module(prefix,type_ignores=[]),'STEP2:actual-warm-load','exec'),stage2)
            encoder=stage2['step1_encoders']['fixture'][0]
            self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
            self.assertFalse(encoder.training)
            self.assertEqual(encoder.tokens(torch.zeros(1,20,3)).shape,(1,12,12))
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='real_numeric_state')
            exec(compile(ast.Module([fn],type_ignores=[]),'STEP2:state','exec'),stage2)
            self.assertEqual(stage2['real_numeric_state'](encoder,torch.zeros(1,20,3),torch.zeros(1,20,3)).shape,(1,12,56))
            stale_factory();stage3=dict(env)
            exec(compile(cells('STEP_3_train_gevis_decision.ipynb')['02-load'],'STEP3:actual-warm-load','exec'),stage3)
            self.assertEqual(stage3['ENCODER_D_MODEL'],12)
            self.assertEqual(stage3['encoders']['fixture'][0].tokens(torch.zeros(1,20,3)).shape,(1,12,12))

    def test_actual_step31_load_via_step4_runtime_entry_repairs_stale_factory(self):
        import pecvd.gevis_ui as ui
        import pecvd.gevis_backbone as backbone_helper
        with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{'GEVIS_DECISION_POLICY':'legacy'}),contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);model,ck,cfg,env=fixture(root)
            manifest=env['ENCODER_ROOT']/'encoder_manifest.json';backbone=root/'backbone.pt'
            net=TinyGEVIS(56,16)
            state={}
            for name in ('text','numeric_input_norm','numeric','numeric_key_norm','descriptor_norm','fusion_gate','type_emb','cross','answer_norm'):
                prefix='norm' if name=='answer_norm' else name
                state.update({prefix+'.'+k:v for k,v in getattr(net,name).state_dict().items()})
            heads={}
            for name in ('support_head','semantic_heads'):heads.update({name+'.'+k:v for k,v in getattr(net,name).state_dict().items()})
            torch.save({'schema':'gevis.backbone.generic.v10','step1_bridge':{'manifest_sha256':env['sha256_file'](manifest)},
                'numeric_dim':56,'d_model':16,'text_encoder_name':backbone_helper.DEFAULT_TEXT_ENCODER,
                'text_encoder_revision':backbone_helper.DEFAULT_TEXT_ENCODER_REVISION,
                'backbone_state':state,'pretrain_heads_state':heads},backbone)
            RUNTIME_FIXTURE.clear();RUNTIME_FIXTURE.update(env)
            RUNTIME_FIXTURE.update(GEVIS=TinyGEVIS,gevis_backbone=backbone_helper,
                ENCODER_ARTIFACT_VERSION=cfg.artifacts.encoder,active_run_tag=lambda:'fixture',
                candidate_backbone_path=lambda *args:backbone,EXPERIMENT_BACKBONE_PATH=backbone)
            source={'03-types':f'from {__name__} import RUNTIME_FIXTURE\nglobals().update(RUNTIME_FIXTURE)',
                    '03-load':cells('STEP_3.1_run_gevis_inference.ipynb')['03-load'],'03-function':''}
            stale_factory()
            try:
                with patch.object(ui,'notebook_runtime_sources',return_value=source), \
                     patch('pecvd.experiments.pipeline_settings.configure_pipeline',return_value={'decision_system':'hybrid'}):
                    runtime=ui.load_gevis_runtime(root,ROOT/'notebooks/STEP_3.1_run_gevis_inference.ipynb',root/'not-read.csv')
                encoder=runtime.encoders['fixture'][0]
                self.assertEqual(encoder.tokens(torch.zeros(1,20,3)).shape,(1,12,12))
                self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
                self.assertEqual(runtime.checkpoint['schema'],'gevis.backbone.generic.v10')
            finally:RUNTIME_FIXTURE.clear()


if __name__=='__main__':unittest.main()
