"""Detailed domain learning with invented Runs only; no private assets or CSVs."""
import ast, json, math, tempfile, unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import decision_training as d, decision_replay as replay
from pecvd.experiments import fusion_answers as fa, current_input as c, peald_training as p
from pecvd.gevis_backbone import ANSWER_TYPES, token_descriptions
from pecvd.gevis_fast import frozen_text

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    # Production library factory; no retired notebook fixture dependency.
    from pecvd.gevis_runtime import GEVIS as Model
    torch.manual_seed(7);model=Model(numeric_dim=8,d_model=16,dropout=0.).requires_grad_(False).eval()
    encoder=torch.nn.Identity();encoder.d_model=8;encoder.patch_len=4;encoder.supports_native_sensor_transfer=True
    def state(e,a,b):
        am=a.reshape(1,2,4,2).mean(2).permute(0,2,1);bm=b.reshape(1,2,4,2).mean(2).permute(0,2,1);delta=am-bm
        return torch.stack([am,bm,delta,delta.abs(),am.square(),bm.square(),delta.square(),torch.ones_like(delta)],-1).reshape(1,4,8)
    meta=dict(sensor_columns=['p','q'],window_length=8,mean=np.zeros(2),std=np.ones(2),cadence_seconds=1.,native_row_origin=0)
    encoders={'phase':(encoder,meta)}
    module=NS(ROOT=ROOT,DECISION_SYSTEM=fa.SYSTEM,DEVICE='cpu',GEVIS_MODEL=model,STEP_ENCODERS=encoders,
        numeric_state=state,numeric_token_descriptions=lambda step,meta,n,batch:[token_descriptions(step,meta['sensor_columns'],n)]*batch,
        CURRENT_SOURCE_RECIPE_SHA256='invented-source-only',ENCODER_ROOT=Path('invented'),BACKBONE_PATH=Path('invented.pt'))
    splits={s:[s+str(i) for i in range(n)] for s,n in [('train',12),('validation',4),('test',4)]}
    windows={'phase':{s:[] for s in splits}}
    for split,runs in splits.items():
        for i,run in enumerate(runs):
            values=np.stack([np.sin(np.arange(8)/2)+i*.005,np.cos(np.arange(8)/3)],-1).astype(np.float32)
            windows['phase'][split].append(dict(current=values,run_id=run,lineage_id='lineage-'+run,pair_id=run,
                step='phase',split=split,sensor_columns=['p','q'],cohort=('recipe','loop-1',0,0)))
    prepared=p.Prepared(windows,{'phase':meta},{},splits,dict(split_runs=splits,invented=True))
    prepared.process=NS(source_columns=dict(run='run',step='step',order='row'))
    r=NS(module=module,encoders=encoders,prepared=prepared,source_steps={'phase':'phase'},domain='PECVD')
    r.config=d.Config(train_windows_per_step=8,examples_per_sensor_family_level=1,epochs=3,patience=3,batch_size=32,width=16,max_seconds=60.)
    return r


class DetailedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_reference_basis_independence_and_no_heldout_influence(self):
        r=fixture();first=d.fit_references(r,r.config)
        self.assertFalse(set(first['basis_lineages'])&set(first['fitting_lineages']))
        for split in ('validation','test'):
            for w in r.prepared.windows['phase'][split]:w['current']+=999999
        again=d.fit_references(r,r.config)
        self.assertEqual(d._hash(first),d._hash(again))
        self.assertTrue(all(set(v['lineages'])<=set(first['basis_lineages']) for v in first['records']))

    def test_each_sensor_family_level_is_effective_and_training_only(self):
        r=fixture();reference=d.fit_references(r,r.config)
        cases,coverage,excluded=d.training_cases(r,reference,r.config)
        self.assertEqual(len(coverage),2*5*4)
        self.assertTrue(all(v['status']=='covered' for v in coverage if v['family']!='delay'))
        self.assertTrue(all(v['generated']<=v['required'] for v in coverage))
        self.assertTrue(all(v['split']=='train' and v['lineage_id'] in reference['fitting_lineages'] for v in cases))
        for case in cases:
            observed=d.case_current(case)
            if case['injected']:
                sid=case['sensor_columns'].index(case['centers'][0])
                self.assertTrue(np.any(observed[:,sid]!=case['current'][:,sid]))
                self.assertEqual(tuple(case['targets'][0]),(1,fa.TYPE_TARGET[case['kind']],case['targets'][0,2]))
                self.assertTrue(np.array_equal(observed[:,1-sid],case['current'][:,1-sid]))

    def test_same_deployment_reference_for_original_and_injected_and_elapsed_coordinates(self):
        r=fixture();ref=d.fit_references(r,r.config);cases,_,_=d.training_cases(r,ref,r.config)
        injected=next(v for v in cases if v['injected']);normal=dict(injected,current=injected['current'],sensor_update=None)
        normal.pop('sensor_update');seen=[];real=fa.fusion_features
        def capture(module,a,b,step,centers,position=0.):
            seen.append(np.array(b,copy=True));self.assertEqual(position,0.)
            return real(module,a,b,step,centers,position)
        with patch.object(fa,'fusion_features',side_effect=capture):
            x=d._features(r,injected,d._bank(ref));y=d._features(r,normal,d._bank(ref))
        np.testing.assert_array_equal(seen[0],seen[2])
        np.testing.assert_array_equal(seen[1],seen[3])
        self.assertEqual(x.shape,y.shape)
        self.assertTrue(torch.allclose(x[...,-2:],torch.tensor([0.,math.log1p(8.)]).expand_as(x[...,-2:])))

    def test_session_feature_parity_two_boundary_numeric_calls_and_missing_context_withholding(self):
        r=fixture();ref=d.fit_references(r,r.config);r.references=ref;r.reference_bank=d._bank(ref)
        cases=d.evaluation_cases(r,'test',r.config);case=cases[0]
        features=d._features(r,case,r.reference_bank)
        head=fa.ContextReadout(features.shape[-1],16).requires_grad_(False).eval()
        r.module.FUSION_ANSWER_HEAD=head;r.module.FUSION_ANSWER_TRAINING=dict(checkpoint_binding=d.SCHEMA,optimizer_updates=1)
        r.module.FUSION_ANSWER_PATH=Path('invented.pt');session=d.Session(r,center_batch_size=1)
        with patch.object(r.module,'numeric_state',wraps=r.module.numeric_state) as encode:
            reply=session.process_window(current=case['current'],step='phase',recipe_key='recipe',loop_key='loop-1',
                centers=case['centers'],lineage_id=case['lineage_id'],standardized=True)
        self.assertEqual(reply['status'],'OK');self.assertEqual(encode.call_count,2)
        expected=[]
        for i in range(len(case['centers'])):
            gate=int(head(features[i,0:1],'BOOLEAN').argmax());expected.append(gate)
            for j,kind in enumerate(fa.KINDS[1:],1):
                logits=head(features[i,j:j+1],kind).clone();logits[:,0]=-torch.inf
                expected.append(int(logits.argmax()) if gate else None)
        self.assertEqual([a['prediction'] for a in reply['answers']],expected)
        missing=session.process_window(current=case['current'][:,:1],step='phase',recipe_key='recipe',sensor_columns=['p'])
        self.assertEqual(missing['reason'],'incomplete_observed_sensor_grid')
        wrong_loop=session.process_window(current=case['current'],step='phase',recipe_key='recipe',loop_key='loop-2',lineage_id=case['lineage_id'])
        self.assertEqual(wrong_loop['reason'],'native_normal_reference_unobserved')
        basis=session.process_window(current=case['current'],step='phase',recipe_key='recipe',loop_key='loop-1',lineage_id=ref['basis_lineages'][0])
        self.assertEqual(basis['reason'],'reference_basis_lineage_not_independent')
        omitted=session.process_window(current=case['current'],step='phase',recipe_key='recipe',loop_key='loop-1')
        self.assertEqual(omitted['reason'],'source_lineage_required')

    def test_complete_mlp_epochs_learn_invented_separable_features_and_never_update_source(self):
        r=fixture();before={k:v.clone() for k,v in r.module.GEVIS_MODEL.state_dict().items()}
        def cases(split):
            result=[]
            for i in range(24):
                changed=i%2;level=1+i%4 if changed else 0
                target=np.array([[changed,1 if changed else 0,level]],np.int64)
                feature=torch.zeros(1,3,12);feature[:,:,0]=changed*8-4;feature[:,:,1]=level*2
                result.append(dict(split=split,lineage_id=split+str(i),targets=target,features=feature))
            return result
        config=replace(r.config,epochs=80,patience=20,learning_rate=.01)
        head,training=d.fit_mlp(cases('train'),cases('validation'),config,progress=None)
        self.assertTrue(all(row['epoch_completed'] for row in training['history']))
        self.assertGreater(max(row['boolean_positive_f1'] for row in training['history']),.95)
        for k,v in before.items():torch.testing.assert_close(v,r.module.GEVIS_MODEL.state_dict()[k])

    def test_package_roundtrip_domain_and_tamper_rejection_and_no_inference_csv(self):
        r=fixture();r.config=replace(r.config,epochs=1)
        r.prepared.metadata['phase']=dict(r.prepared.metadata['phase'],
            mean=torch.zeros(1,1,2,requires_grad=True),std=torch.ones(1,1,2),
            model_state={'invented_weight':torch.ones(3)},optimizer_state={'tensor':torch.zeros(2)})
        with tempfile.TemporaryDirectory() as f:
            root=Path(f)
            with patch.object(c,'source_identity',return_value={'invented':'source'}):
                training=d.train_or_load(root,r,config=r.config,allow_training=True,progress=None)
                self.assertNotIn('model_state',r.prepared.metadata['phase'])
                self.assertNotIn('optimizer_state',r.prepared.metadata['phase'])
                self.assertEqual(r.prepared.metadata['phase']['mean'],[0.,0.])
                source=NS(module=r.module,encoders=r.encoders)
                with patch.object(c,'load_frozen_source',return_value=source),patch.object(pd,'read_csv',side_effect=AssertionError('No inference CSV')):
                    loaded=d.load(root,domain='PECVD')
                self.assertEqual(loaded.runtime.domain,'PECVD')
                self.assertEqual(loaded.runtime.module.FUSION_ANSWER_TRAINING['selected_epoch'],training['selected_epoch'])
                with self.assertRaises(FileNotFoundError):d.load(root,domain='PEALD')
                path=r.module.FUSION_ANSWER_PATH;payload=torch.load(path,map_location='cpu',weights_only=False)
                with self.assertRaisesRegex(ValueError,'domain'):d.validate_payload(payload,path,'PEALD')
                payload['references']['records'][0]['lower'][0,0]+=1
                with self.assertRaisesRegex(ValueError,'content'):d.validate_payload(payload,path,'PECVD')

    def test_joint_selection_does_not_hide_a_weak_answer_family(self):
        def score(b,c,o):
            return dict(boolean_positive_f1=b,macro_f1={'BOOLEAN':b,'CHOICE':c,'ORDINAL':o},normal_false_positive_rate=.01)
        self.assertGreater(d.selection_key(score(.95,.97,.97),.2),d.selection_key(score(.99,.3,.3),.1))
        self.assertEqual(d.typed_f1(score(.96,.97,.98)),{'BOOLEAN':.96,'CHOICE':.97,'ORDINAL':.98})

    def test_pecvd_window_coordinates_do_not_depend_on_future_extent(self):
        r=fixture();process=NS(columns=NS(run='run',step='phase',order='row'))
        frame=pd.DataFrame(dict(run=['R']*16,phase=['phase']*16,row=np.arange(16),row_in_step=np.arange(16),
            step_number=[1]*16,p=np.arange(16,dtype=float),q=np.arange(16,dtype=float),_cohort=['recipe']*16))
        first,_=replay.pecvd_windows(frame.iloc[:8],r,process)
        full,_=replay.pecvd_windows(frame,r,process)
        self.assertEqual(first[0]['offset_samples'],full[0]['offset_samples'])
        self.assertEqual(first[0]['recipe_key'],full[0]['recipe_key'])
        np.testing.assert_array_equal(first[0]['current'],full[0]['current'])

    def test_native_receiver_preserves_loop_and_canonical_source_lineage(self):
        from pecvd.experiments import peald_current_data as native
        step=native.route_key('subcycle','SubRecipe',1,'Parent');seen=[]
        meta=dict(sensor_columns=['p','q'],window_length=8,cadence_seconds=1.,tolerance_seconds=.05,
            native_kind='subcycle',parent_step='Parent',allowed_recipe_keys=[p._hash('recipe')],
            expected_substeps=(('1','Child'),))
        class Capture:
            supports_reference_context=True
            metadata={step:meta}
            def process_window(self,**kwargs):
                seen.append(kwargs)
                return dict(status='OK',answers=[],timings={},reason=None,selected_sensors=2)
        receiver=native.Receiver(Capture())
        for i in range(8):
            row={column:'' for column in native.COLUMNS.values()}
            row.update(ActivePjobID='invented-run',Timestamp=str(pd.Timestamp('2020-01-01')+pd.Timedelta(seconds=i)),
                PMStoredProcessRecipeName='recipe',PPExecname='SubRecipe',ProcessRecipeStepName='Parent',
                ProcessRecipeStepSeqNo=1,PPExecStepName='Child',PPExecStepSeqNo=1,PPExecStepID='C',
                SubRecipeLoopMoniterValue=7,p=float(i),q=float(i+1))
            receiver.push(row)
        receiver.finish()
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0]['loop_key'],'7')
        self.assertEqual(seen[0]['lineage_id'],p._hash('invented-run'))
        self.assertEqual(seen[0]['offset_samples'],0)

    def test_feature_cache_tamper_and_paired_dependency_binding(self):
        r=fixture();reference=d.fit_references(r,r.config)
        cases=d.evaluation_cases(r,'test',r.config)[:1]
        self.assertTrue({'temporal_bridge.py','relational_features.py'}<=set(d._codes()))
        with tempfile.TemporaryDirectory() as f:
            d.encoded_cases(r,cases,reference,cache_root=Path(f),source={'invented':1})
            with patch.object(d,'_features',side_effect=AssertionError('No repeated Fusion call')):
                d.encoded_cases(r,cases,reference,cache_root=Path(f),source={'invented':1})
            path=next(Path(f).glob('*/00000000.npy'))
            path.write_bytes(path.read_bytes()+b'corrupted invented features')
            with self.assertRaisesRegex(ValueError,'content'):
                d.encoded_cases(r,cases,reference,cache_root=Path(f),source={'invented':1})

    def test_ald_native_replay_completes_and_keeps_detailed_metadata(self):
        from pecvd.experiments import peald_current_data as native
        r=fixture();step=native.route_key('subcycle','SubRecipe',1,'Parent')
        meta=dict(r.encoders['phase'][1],native_kind='subcycle',parent_step='Parent',tolerance_seconds=.05,
            allowed_recipe_keys=[p._hash('recipe')],expected_substeps=(('1','Child'),))
        r.encoders={step:(r.encoders['phase'][0],meta)};r.module.STEP_ENCODERS=r.encoders;r.domain='PEALD'
        r.references=dict(records=[dict(key=[step,p._hash('recipe'),'7',0],lower=np.zeros((8,2),np.float32),upper=np.ones((8,2),np.float32),
            reference_runs=2,lineages=['basis1','basis2'])],basis_lineages=['basis1','basis2'],fitting_lineages=['fit1','fit2'])
        r.reference_bank=d._bank(r.references)
        features,_=d._window_features(r,np.zeros((8,2)),r.reference_bank[(step,p._hash('recipe'),'7',0)],step,['p'],0,1)
        r.module.FUSION_ANSWER_HEAD=fa.ContextReadout(features.shape[-1],16).requires_grad_(False).eval()
        r.module.FUSION_ANSWER_TRAINING=dict(checkpoint_binding=d.SCHEMA,optimizer_updates=1)
        r.module.FUSION_ANSWER_PATH=Path('invented.pt')
        records=[]
        for i in range(8):
            row={column:'' for column in native.COLUMNS.values()}
            row.update(ActivePjobID='new-invented-run',Timestamp=str(pd.Timestamp('2020-01-01')+pd.Timedelta(seconds=i)),
                PMStoredProcessRecipeName='recipe',PPExecname='SubRecipe',ProcessRecipeStepName='Parent',ProcessRecipeStepSeqNo=1,
                PPExecStepName='Child',PPExecStepSeqNo=1,PPExecStepID='C',SubRecipeLoopMoniterValue=7,p=float(i),q=float(i+1))
            records.append(row)
        contract=NS(runtime=r)
        report=replay._replay_frame(contract,pd.DataFrame(records),[dict(run_id='CASE-1',start=0,end=8)],progress=None)
        self.assertEqual(len(report.details),6)
        self.assertEqual(report.metadata['input_mode'],d.INPUT_MODE)
        self.assertEqual(report.metadata['normal_reference_recipe'],d.RECIPE)
        self.assertEqual(report.diagnostics['coverage'].answered_windows.iloc[0],1)

    def test_pecvd_preparation_identity_is_equal_for_training_and_test_only_reads(self):
        from pecvd.experiments import pecvd_current_data as data
        r=fixture();partitions={s:[s+str(i) for i in range(n)] for s,n in [('train',6),('validation',2),('test',2)]}
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);directory=root/'artifacts/invented';(directory/'windows').mkdir(parents=True)
            (root/'data').mkdir();csv=root/'data/all_processes_analysis.csv'
            frame=pd.DataFrame([dict(process_key=run,row_in_process=t,row_in_step=t,step_number=1,step_elapsed_seconds=t)
                for split,runs in partitions.items() for run in runs for t in range(8)])
            frame.to_csv(csv,index=False)
            manifest=dict(source_sha256=p.digest(csv),run_split=partitions)
            (directory/'encoder_manifest.json').write_text(json.dumps(manifest),encoding='utf8')
            cache={}
            for split,runs in partitions.items():
                cache[split]=np.ones((len(runs),8,2),np.float32)
                cache[split+'_process_id']=np.array(runs)
                cache[split+'_window_start']=np.zeros(len(runs),np.int64)
                cache[split+'_window_end']=np.full(len(runs),7,np.int64)
                cache[split+'_cohort_key']=np.array(['recipe']*len(runs))
            np.savez_compressed(directory/'windows/phase.npz',**cache)
            source=NS(module=NS(ENCODER_ROOT=directory),encoders=r.encoders)
            train=data.prepare(root,source,splits=('train','validation'))
            test=data.prepare(root,source,splits=('test',))
            self.assertEqual(d._hash(train.identity),d._hash(test.identity))
            self.assertEqual(d._hash(train.metadata),d._hash(test.metadata))
            self.assertFalse(test.windows['phase']['train'])
            self.assertEqual(test.windows['phase']['test'][0]['cohort'][-1],0)


if __name__=='__main__':unittest.main()
