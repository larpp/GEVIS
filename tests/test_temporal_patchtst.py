"""CPU-only synthetic checks: no process files or pretrained downloads."""
from types import SimpleNamespace
import torch
import unittest
import ast
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
from pecvd.temporal_patchtst import TemporalPatchEncoder, blind_objective, BACKEND, CONTRACT
from pecvd.temporal_bridge import numeric_state, GatedTemporalAdapter
from pecvd.encoder_loading import load_encoder


def model(width=16):
    torch.manual_seed(17)
    return TemporalPatchEncoder(3,4,width,4,1,0.).eval()


def test_hidden_values_never_enter_masked_features():
    net=model();x=torch.randn(2,16,3,requires_grad=True)
    mask=torch.zeros(2,12,dtype=torch.bool);mask[:,::2]=True
    net(x,mask).sum().backward()
    grad,_=net.patchify(x.grad)
    assert torch.count_nonzero(grad[mask])==0
    assert torch.count_nonzero(grad[~mask])>0


def test_temporal_and_sensor_attention_gradients():
    net=model().train();x=torch.randn(2,16,3)
    mask=torch.arange(12).remainder(2).bool()[None].expand(2,-1)
    cfg=SimpleNamespace(slope_weight=.25,relationship_weight=.1,variability_weight=.25,peak_weight=.1)
    loss,parts=blind_objective(net,x,mask,cfg);loss.backward()
    assert torch.isfinite(loss) and parts['visible_mse']==0
    assert net.blocks[0].sensor.in_proj_weight.grad.abs().sum()>0
    assert net.blocks[0].temporal.self_attn.in_proj_weight.grad.abs().sum()>0


def test_sensor_permutation_equivariance():
    net=model();x=torch.randn(2,16,3);order=[2,0,1]
    a=net.tokens(x).reshape(2,3,4,16)
    b=net.tokens(x[:,:,order]).reshape_as(a)
    torch.testing.assert_close(b,a[:,order],atol=2e-6,rtol=2e-5)


def test_blind_predictions_are_complementary_hidden_outputs():
    net=model();x=torch.randn(2,16,3)
    mask=(torch.arange(3)[:,None]+torch.arange(4)[None,:]).remainder(2)==0
    mask=mask.reshape(1,-1).expand(2,-1)
    torch.testing.assert_close(net.blind_prediction(x),torch.where(mask[...,None],net(x,mask),net(x,~mask)))


def test_bridge_and_gate_keep_blind_residual_and_time_information():
    net=model(96);x=torch.randn(2,16,3);r=x.roll(1,1)
    state=numeric_state(net,x,r)
    cp,_=net.patchify(x)
    assert state.shape==(2,12,408)
    torch.testing.assert_close(state[...,-2],(net.blind_prediction(x)-cp).square().mean(-1))
    assert state[...,384:400].abs().sum()>0
    adapter=GatedTemporalAdapter(408,32);adapter(state).square().mean().backward()
    assert adapter.gate.weight.grad.abs().sum()>0
    assert adapter.temporal[0].weight.grad.abs().sum()>0


def test_offline_strict_checkpoint_roundtrip():
    net=model();ck=dict(encoder_backend=BACKEND,reconstruction_contract=CONTRACT,
        sensor_columns=['a','b','c'],patch_length=4,window_length=16,
        model_config=dict(d_model=16,nhead=4,layers=1,dropout=0.),model_state=net.state_dict())
    restored=load_encoder(ck).eval();x=torch.randn(2,16,3)
    torch.testing.assert_close(restored.blind_prediction(x),net.blind_prediction(x))
    ck['reconstruction_contract']='old'
    with unittest.TestCase().assertRaisesRegex(ValueError,'contract'):load_encoder(ck)


def test_actual_notebook_temporal_training_and_runtime_bridge():
    """Execute notebook functions on invented arrays, never the process CSV."""
    import numpy as np
    import pandas as pd
    # patch.dict(sys.modules) must not remove lazily registered PyTorch ops on
    # exit; initialize its optimizer dependency before taking that snapshot.
    import torch._dynamo
    from dataclasses import replace
    root=Path(__file__).resolve().parents[1]
    cells={c['id']:''.join(c['source']) for c in json.loads((root/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))['cells'] if c['cell_type']=='code'}
    module=types.ModuleType('synthetic_temporal_training');env=module.__dict__
    with patch.dict(sys.modules,{module.__name__:module}),patch.dict('os.environ',{}),contextlib.redirect_stdout(io.StringIO()):
        exec(compile(cells['01-config'].replace("NOTEBOOK_MODEL_VARIANT='hybrid'","NOTEBOOK_MODEL_VARIANT='temporal'"),'temporal-config','exec'),env)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);source=path/'fixture.txt';source.write_text('synthetic fixture',encoding='utf-8')
            arrays={s:np.random.default_rng(i).normal(size=(4,16,3)).astype('float32') for i,s in enumerate(('train','validation','test'))}
            split={s:[s+str(i) for i in range(4)] for s in arrays}
            meta=pd.DataFrame(dict(process_id=['r0','r1','r2','r3'],source_group=['fixture']*4,pm_cycle=['fixture']*4,cohort_key=['fixture']*4,window_start=[0]*4,window_end=[16]*4,position_ratio=[0.]*4,position_bin=[0]*4))
            env.update(CONFIG=replace(env['CONFIG'],epochs=2,min_epochs=0,warmup_epochs=1,batch_size=2,use_amp=False,fused_adamw=False,validation_interval=1),DEVICE=torch.device('cpu'),DATA_PATH=source,ARTIFACT_ROOT=path/'candidate',STEPS={'Gap-up':(16,4,4)},sensor_columns=['a','b','c'],sensor_descriptors={s:s for s in 'abc'},raw_windows={'Gap-up':arrays},window_metadata={'Gap-up':{s:meta for s in arrays}},run_split=split)
            funcs=[n for n in ast.parse(cells['01-fit']).body if isinstance(n,ast.FunctionDef)]
            exec(compile(ast.Module(funcs,type_ignores=[]),'temporal-fit','exec'),env)
            ck=env['train_one_step']('Gap-up');net=load_encoder(ck).eval()
            assert ck['encoder_backend']==BACKEND and not ck['adaptation_contract']['legacy_warmstart']
            assert ck['training_status']['best_validation_mse']<=ck['training_status']['initial_validation_mse']
            mean=arrays['train'].mean((0,1),keepdims=True);std=arrays['train'].std((0,1),keepdims=True).clip(1e-6)
            score,_=env['evaluate_masked'](net,((arrays['validation']-mean)/std).astype('float32'),999)
            unittest.TestCase().assertAlmostEqual(score,ck['training_status']['best_validation_mse'],places=5)
            x=torch.randn(1,16,3)
            for name,cell in [('STEP_3_train_gevis_decision.ipynb','02-functions'),('STEP_3.1_run_gevis_inference.ipynb','03-types')]:
                doc=json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
                code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')==cell))
                fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='numeric_state')
                scope={'torch':torch};exec(compile(ast.Module([fn],type_ignores=[]),'runtime-state','exec'),scope)
                torch.testing.assert_close(scope['numeric_state'](net,x,x),numeric_state(net,x,x))


def test_backbone_and_runtime_construct_identical_temporal_adapters():
    from pecvd import gevis_backbone
    from pecvd.temporal_bridge import GatedTemporalAdapter
    from torch import nn
    root=Path(__file__).resolve().parents[1]
    backbone=gevis_backbone.GEVISBackbone(408,32,0.,text_encoder_name='byte')
    assert isinstance(backbone.numeric,GatedTemporalAdapter)
    state=backbone.numeric.state_dict()
    for name,cell in [('STEP_3_train_gevis_decision.ipynb','02-config'),('STEP_3.1_run_gevis_inference.ipynb','03-types')]:
        doc=json.loads((root/'notebooks'/name).read_text(encoding='utf-8'))
        code=''.join(next(c['source'] for c in doc['cells'] if c.get('id')==cell))
        nodes=[n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name in ('ByteTextEncoder','GEVIS')]
        scope={'torch':torch,'nn':nn,'gevis_backbone':gevis_backbone}
        exec(compile(ast.Module(nodes,type_ignores=[]),'runtime-model','exec'),scope)
        runtime=scope['GEVIS'](408,32,0.,'byte')
        runtime.numeric.load_state_dict(state,strict=True)
        x=torch.randn(2,12,408)
        torch.testing.assert_close(runtime.numeric(x),backbone.numeric(x))


def test_unseen_sensor_count_uses_same_native_joint_weights():
    net=model(96);before={k:v.clone() for k,v in net.state_dict().items()}
    x=torch.randn(2,16,5)
    assert net.tokens(x).shape==(2,20,96)
    assert net.blind_prediction(x).shape==(2,20,4)
    assert numeric_state(net,x,x).shape==(2,20,408)
    for key,value in before.items():torch.testing.assert_close(value,net.state_dict()[key],rtol=0,atol=0)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().items() if name.startswith('test_'))
