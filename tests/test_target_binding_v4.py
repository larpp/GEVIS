import ast,json,math,time
from pathlib import Path
from types import ModuleType
import unittest
import torch
from torch import nn
import pecvd.gevis_backbone as gb
from pecvd.gevis_fast import frozen_text

ROOT=Path(__file__).resolve().parents[1]


class TargetBindingTests(unittest.TestCase):

    def test_fusion_contract_legacy_anchor_and_finite_gradients(self):
        numeric=torch.randn(2,6,8,requires_grad=True); descriptor=torch.randn_like(numeric,requires_grad=True)
        text=torch.randn(2,8); typed=text+10
        gate=nn.Sequential(nn.Linear(16,8),nn.Sigmoid())
        q,keys=gb.attention_inputs(numeric,descriptor,text,typed,gate,'legacy')
        torch.testing.assert_close(q,typed)
        torch.testing.assert_close(keys,numeric+gate(torch.cat([numeric,descriptor],-1))*descriptor)
        q,keys=gb.attention_inputs(numeric,descriptor,text,typed,gate,gb.TARGET_FUSION_MODE)
        torch.testing.assert_close(q,text)
        keys.square().sum().backward()
        self.assertGreater(float(numeric.grad.abs().sum()),0)
        self.assertGreater(float(descriptor.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(numeric.grad).all())
        with self.assertRaises(ValueError): gb.attention_inputs(numeric,None,text,typed,gate,gb.TARGET_FUSION_MODE)
        with self.assertRaises(ValueError): gb.attention_inputs(numeric,descriptor,text,typed,gate,'unknown')

    def test_type_embedding_cannot_redirect_identical_named_query(self):
        model=gb.GEVISBackbone(8,32,0.).eval(); model.fusion_mode=gb.TARGET_FUSION_MODE
        numeric=torch.randn(1,6,8).repeat(3,1,1)
        descriptions=[gb.token_descriptions('phase',['pressure_a','pressure_b'],3)]*3
        _,weights=model(numeric,['Inspect pressure_a over the full interval.']*3,list(gb.ANSWER_TYPES),descriptions)
        torch.testing.assert_close(weights[0],weights[1]); torch.testing.assert_close(weights[0],weights[2])
        _,permuted=model(numeric.reshape(3,2,3,8).flip(1).reshape(3,6,8),
            ['Inspect pressure_a over the full interval.']*3,list(gb.ANSWER_TYPES),[descriptions[0][3:]+descriptions[0][:3]]*3)
        torch.testing.assert_close(permuted.reshape(3,2,3),weights.reshape(3,2,3).flip(1))

    def test_permuted_numeric_targets_move_while_questions_remain_sensor_free(self):
        names=['pressure_a','pressure_b','flow']
        numeric=torch.arange(3.).reshape(1,3,1,1).expand(1,3,3,8).reshape(1,9,8).clone()
        before=numeric.clone()
        values,questions,kinds,descriptions,_,targets=gb.real_step1_cross_batch(numeric,'phase',names,
            seed=21,binding_refinement=True,runtime_wording=True,permute_sensors=True)
        torch.testing.assert_close(numeric,before)
        for i,question in enumerate(questions):
            sensor=int(targets['sensor'][i]); source_id=int(values[i,sensor*3,0]); name=names[source_id]
            self.assertNotIn(name,question); self.assertIn('sensor '+name+';',descriptions[i][sensor*3])
            if kinds[i]!='BOOLEAN':
                self.assertNotIn('increase in',question); self.assertNotIn('reconstruction residual',question)
        renamed=gb.real_step1_cross_batch(numeric,'phase',names,seed=21,binding_refinement=True,
                                        runtime_wording=True,permute_sensors=True,rename_sensors=True)
        for i,question in enumerate(renamed[1]):
            name=f"channel_21_{int(renamed[-1]['sensor'][i]):03d}_signal"
            self.assertNotIn(name,question); self.assertIn(name,renamed[3][i][int(renamed[-1]['sensor'][i])*3])

    def test_actual_runtime_and_training_fusion_are_identical(self):
        from pecvd.gevis_ui import notebook_runtime_sources
        code=notebook_runtime_sources(ROOT/'notebooks/STEP_3.1_run_gevis_inference.ipynb')['03-types']
        nodes=[node for node in ast.parse(code).body if isinstance(node,ast.ClassDef) and node.name in ('ByteTextEncoder','GEVIS')]
        module=ModuleType('v4_parity')
        module.__dict__.update(torch=torch,nn=nn,math=math,gevis_backbone=gb,frozen_text=frozen_text,ANSWER_TYPES=gb.ANSWER_TYPES)
        exec(compile(ast.Module(nodes,type_ignores=[]),'v4:runtime','exec'),module.__dict__)
        source=gb.GEVISBackbone(8,32,0.).requires_grad_(False).eval()
        runtime=module.GEVIS(8,32,0.).requires_grad_(False).eval()
        # Production uses the same E5 implementation in both classes. Share an
        # invented frozen encoder here; historical byte implementations differ.
        runtime.text=source.text
        state={('answer_norm.'+k.removeprefix('norm.') if k.startswith('norm.') else k):v for k,v in source.state_dict().items()}
        result=runtime.load_state_dict(state,strict=False)
        self.assertFalse(result.unexpected_keys)
        for mode in ('legacy',gb.TARGET_FUSION_MODE,gb.REFERENCE_FUSION_MODE):
            source.fusion_mode=mode; runtime.fusion_mode=mode
            numeric=torch.randn(2,6,8); questions=['Sensor pressure_a, early interval.','Sensor pressure_b, late interval.']
            kinds=['BOOLEAN','CHOICE']; descriptions=[gb.token_descriptions('phase',['pressure_a','pressure_b'],3)]*2
            expected,weights=source(numeric,questions,kinds,descriptions)
            actual=runtime(numeric,questions,kinds,[['no','yes']]*2,descriptions,return_fusion=True)
            torch.testing.assert_close(actual[0],expected); torch.testing.assert_close(actual[2],weights)


if __name__=='__main__': unittest.main()
