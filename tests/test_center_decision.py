"""Learned detection must bypass old candidate gates and preserve run boundaries."""
import os,unittest,tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from pecvd.experiments.center_decision import (center_features,pair_context,targets_for_center,CenterDecision,
                                              CenterConfig,save_candidate,load_candidate)
from pecvd.experiments.center_training import adoption_rule,calibrate_head
from pecvd.experiments.injection import Injection,inject
from pecvd.experiments.optimization_heads import TYPE_LABELS


class CenterDecisionTests(unittest.TestCase):
    def test_features_use_upstream_attention_and_center_context(self):
        a=np.stack([np.sin(np.arange(32)/3),np.cos(np.arange(32)/3)],1)
        state=np.zeros((1,8,40));weights=np.full((2,4),.25)
        first=center_features(a,a,state,np.array([.8,.2]),weights,0,8)
        second=center_features(a,a,state,np.array([.2,.8]),weights,0,8)
        self.assertEqual(first.shape,(2,8,97))
        self.assertFalse(np.array_equal(first,second))
        np.testing.assert_array_equal(first[0,:,-1],1.)
        np.testing.assert_array_equal(first[1,:,-1],0.)
        self.assertTrue(np.isfinite(first).all())

    def test_flat_unrelated_center_does_not_create_relationship_target(self):
        before=np.stack([np.zeros(32),np.sin(np.arange(32)/3)],1)
        spec=Injection('spike',1,4,20,2.)
        after,truth=inject(before,spec,np.ones(2))
        target,mask=targets_for_center(before,after,truth,spec,0,8)
        self.assertIsNone(target);self.assertFalse(mask.any())
        target,mask=targets_for_center(before,after,truth,spec,1,8)
        self.assertEqual(target,1);self.assertTrue(mask.any())

    def test_changed_pair_has_observed_relation_labels_without_center_wave_change(self):
        wave=np.sin(np.arange(64)/3);before=np.stack([wave,wave],1)
        spec=Injection('relationship',1,0,64,3.)
        after,truth=inject(before,spec,np.ones(2))
        target,mask=targets_for_center(before,after,truth,spec,0,8)
        np.testing.assert_array_equal(before[:,0],after[:,0])
        self.assertEqual(target,1);self.assertTrue(mask[1].any());self.assertFalse(mask[0].any())

    def test_adoption_rejects_alert_growth_or_step_regression(self):
        baseline={'a':{'controls':0,'hits':2},'b':{'controls':1,'hits':2}}
        self.assertFalse(adoption_rule({'a':{'controls':1,'hits':4},'b':{'controls':1,'hits':2}},baseline)['accepted'])
        self.assertFalse(adoption_rule({'a':{'controls':0,'hits':1},'b':{'controls':1,'hits':5}},baseline)['accepted'])
        self.assertTrue(adoption_rule({'a':{'controls':0,'hits':3},'b':{'controls':1,'hits':2}},baseline)['accepted'])

    def test_calibration_uses_independent_run_maximum_not_window_count(self):
        detector=SimpleNamespace(scores=lambda feature,step:np.asarray(feature))
        rows=[{'step':'a','run_id':str(i),'spec':None,'features':np.array([i,0])} for i in range(4)]
        rows.append(dict(rows[0],features=np.array([8.,0])))
        table=calibrate_head(detector,rows)
        self.assertEqual(table['0.99']['a']['threshold'],8.)
        self.assertEqual(table['0.99']['a']['runs'],4)
        self.assertTrue(table['0.99']['a']['tail_unresolved'])
        self.assertEqual(table['0.99']['a']['self_threshold'],8.)
        self.assertEqual(table['0.99']['a']['pair_threshold'],0.)

    def test_selected_learned_path_does_not_call_old_numeric_gate(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery import review_changes
        from pecvd.experiments.window_policy import MODE
        runtime,current,reference,_=fixture();m=runtime.module;m.DISCOVERY_POLICY_MODE=MODE
        marker={'changed':True,'decision_source':'learned'}
        learned=SimpleNamespace(review=lambda *args,**kwargs:marker)
        with patch('pecvd.experiments.center_decision.selected_candidate',return_value=learned), \
             patch('pecvd.experiments.discovery.review_window_changes',side_effect=AssertionError('old gate must not veto learned result')):
            self.assertIs(review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1),center_sensor='a'),marker)

    def test_rejected_candidate_cannot_be_saved(self):
        with self.assertRaisesRegex(ValueError,'Rejected'):save_candidate(None,'unused',{}, {'accepted':False})

    def test_learned_scores_control_boolean_type_and_interval_in_actual_ui_path(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery import review_changes
        from pecvd.experiments.window_policy import MODE
        runtime,current,reference,_=fixture();m=runtime.module;m.DISCOVERY_POLICY_MODE=MODE
        m.DISCOVERY_POLICY=SimpleNamespace(config=SimpleNamespace(nominal_window_alert_rate=.01))
        kind=np.zeros((2,len(TYPE_LABELS)));kind[:,TYPE_LABELS.index('spike')]=1.
        outputs=SimpleNamespace(predict=lambda *args:{'sensor_scores':np.array([-10.,10.]),
            'patch_probabilities':np.tile(np.array([0.,0.,1.,1.,0.,0.,0.,0.]),(2,1)),
            'type_probabilities':kind})
        head=CenterDecision(None,outputs,{'0.99':{'invented_phase':{'threshold':1.,'ordinal_edges':[1.,2.,3.,4.]}}},CenterConfig(patches=8))
        with patch('pecvd.experiments.center_decision.selected_candidate',return_value=head), \
             patch.object(runtime.encoders['invented_phase'][0],'to',wraps=runtime.encoders['invented_phase'][0].to) as move, \
             patch('pecvd.experiments.discovery.review_window_changes',side_effect=AssertionError('old gate veto')):
            result=review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('test',1),center_sensor='b')
        move.assert_called_with(m.DEVICE)
        self.assertTrue(result['self_analysis']['changed'])
        self.assertFalse(result['relationship_analysis']['changed'])
        self.assertEqual(result['answers'][0]['answer'],'Yes')
        self.assertEqual(result['self_analysis']['winner']['statistic'],'local_change')
        self.assertIsNotNone(result['self_analysis']['winner']['detected_rows'])
        self.assertEqual(result['answers'][0]['decision_source'],'gevis.center_decision.v1')

    def test_partner_interval_scores_the_explicit_partner_not_the_ui_center(self):
        from pecvd.experiments.discovery_reporting import localization_scores
        event={'sensor':'center','partner_sensor':'target','detected_rows':[2,6],
               'pointwise':{'exceedance_intervals':[[2,6]]}}
        result=localization_scores({'winner':event,'events':[event]},'target',[2,6])
        self.assertEqual(result['winner_interval_iou'],1.)

    def test_saved_candidate_roundtrip_and_strict_identity(self):
        from pathlib import Path
        from pecvd.experiments.optimization_models import PairedEncoderTrial,PairedPatchDetector
        from pecvd.experiments.optimization_heads import GroundedOutputTrial,GroundedOutputHeads
        detector=PairedEncoderTrial('paired_cross_attention',np.ones(4),width=8)
        detector.mean=np.zeros(3,dtype='float32');detector.scale=np.ones(3,dtype='float32')
        detector.model=PairedPatchDetector(3,np.ones(4),'paired_cross_attention',8).eval()
        outputs=GroundedOutputTrial(detector);outputs.heads=GroundedOutputHeads(8,24).eval()
        thresholds={'0.99':{'a':{'threshold':1.,'ordinal_edges':[1.,2.,3.,4.]}}}
        head=CenterDecision(detector,outputs,thresholds,CenterConfig(width=8))
        feature=np.zeros((2,4,3),dtype='float32');before=head.predict_features(feature,'a',0)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'candidate.pt';save_candidate(head,path,{'fixture':True},{'accepted':True})
            with patch('pecvd.experiments.center_decision.artifact_identity',return_value={'fixture':True}):
                restored=load_candidate(SimpleNamespace(DEVICE='cpu'),path)
            np.testing.assert_array_equal(before['sensor_scores'],restored.predict_features(feature,'a',0)['sensor_scores'])
            with patch('pecvd.experiments.center_decision.artifact_identity',return_value={'fixture':False}):
                with self.assertRaisesRegex(ValueError,'identity mismatch'):load_candidate(SimpleNamespace(DEVICE='cpu'),path)

    def test_internal_selection_runs_cannot_enter_any_train_reference(self):
        from pathlib import Path
        from pecvd.experiments.center_training import collect_center_rows
        from pecvd.experiments.optimization_models import split_runs
        from pecvd.experiments.window_policy import WindowConfig
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'windows').mkdir();arrays={}
            for split,count in (('train',12),('validation',8),('test',8)):
                arrays[split]=np.zeros((count,16,2),dtype='float32')
                arrays[split+'_process_id']=np.array([f'{split}-{i}' for i in range(count)])
                arrays[split+'_cohort_key']=np.array(['fixture']*count)
                arrays[split+'_position_bin']=np.zeros(count,int)
            np.savez(root/'windows'/'a.npz',**arrays)
            runtime=SimpleNamespace(encoders={'a':None},module=SimpleNamespace(ENCODER_ROOT=root,
                DISCOVERY_POLICY=SimpleNamespace(config=WindowConfig())))
            config=CenterConfig(train_runs_per_step=8,validation_runs_per_step=8)
            prepared,audit=collect_center_rows(runtime,config)
            selection,fit=split_runs([{'run_id':str(v)} for v in sorted(arrays['train_process_id'])],.25,config.seed+1)
            selection_ids={e['run_id'] for e in selection}
            sampled_fit={e['run_id'] for e in prepared['a']['train'] if e['head_partition']=='fit'}
            reference_ids=set()
            for row in prepared['a']['train']:
                self.assertFalse(selection_ids&set(row['reference_run_ids']))
                self.assertNotIn(row['run_id'],row['reference_run_ids'])
                reference_ids.update(row['reference_run_ids'])
            self.assertTrue(reference_ids-sampled_fit)
            self.assertFalse(audit['test_values_read'])

    def test_retired_center_fit_fixture_calls_training_entry(self):
        import json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/'tests/fixtures/step3_legacy_diagnostics.json').read_text(encoding='utf-8'))
        source=''.join(next(c['source'] for c in notebook['cells'] if c.get('id')=='learned-center-fit'))
        report=SimpleNamespace(summary=[],diagnostics={'by_kind':[]},metadata={'adoption':{'accepted':False}})
        with tempfile.TemporaryDirectory() as directory, \
             patch('pecvd.experiments.routing_audit.refresh_evaluation_modules'), \
             patch('pecvd.experiments.center_decision.candidate_path',return_value=Path(directory)/'absent.pt'), \
             patch('pecvd.experiments.reporting.ready_runtime',return_value='invented runtime'), \
             patch('pecvd.experiments.center_training.train_center_decision',return_value=(object(),report)) as train:
            exec(source,{'RUN_STRUCTURED_EVALUATION':True,'EVALUATION_MODE':'center','ROOT':root,
                'backbone_path':Path(directory)/'backbone.pt','display':lambda value:None,'os':os})
        self.assertTrue(train.call_args.kwargs['save_accepted'])
        self.assertEqual(train.call_args.kwargs['runtime'],'invented runtime')


if __name__=='__main__':unittest.main()
