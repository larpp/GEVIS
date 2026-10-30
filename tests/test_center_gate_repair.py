"""Invented-input checks: gate scope, independent control audit and coverage."""
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from pecvd.experiments.window_policy import WindowConfig,WindowPolicy,fit_window_policy


def records():
    zero=np.zeros((64,2));reference_ids=['train-other']
    train=[dict(split='train',step='fixture',run_id=f'train-{i}',reference_run_ids=reference_ids,
                sensors=['quiet','busy'],current=zero.copy(),reference=zero.copy(),residual=zero.copy()) for i in range(8)]
    val=[]
    for i in range(16):
        current=zero.copy();current[:,1]=.001*np.sin(np.arange(64)*.4)
        val.append(dict(split='validation',step='fixture',run_id=f'val-{i}',reference_run_ids=['train-0'],
            sensors=['quiet','busy'],current=current,reference=zero.copy(),residual=np.column_stack([np.zeros(64),np.full(64,10.)])))
    return train,val


class CenterGateRepairTests(unittest.TestCase):
    def test_unrelated_sensor_cannot_veto_quiet_center_after_audit_passes(self):
        train,val=records();cfg=WindowConfig(gate_scope='center_views_v3',noise_scale_floor=True)
        policy=WindowPolicy(fit_window_policy(train,val,{},cfg));g=policy.payload['groups']['fixture']
        self.assertTrue(g['center_gate_adopted']);self.assertEqual(g['audit_runs'],8)
        self.assertEqual(g['new_audit_alert_runs'],0);self.assertFalse(g['center_gate_tail_supported'])
        current=val[0]['current'].copy();current[16:48,0]=.0005
        zero=np.zeros_like(current)
        result=policy.analyze(current,zero,zero,'fixture',['quiet','busy'],selected=['quiet'],center_sensor='quiet')
        self.assertTrue(result['changed']);self.assertGreater(result['winner']['ordinal_index'],0)
        self.assertIsNotNone(result['winner']['detected_rows'])
        legacy=WindowPolicy(fit_window_policy(train,val,{}))
        self.assertFalse(legacy.analyze(current,zero,zero,'fixture',['quiet','busy'],selected=['quiet'])['changed'])
        self.assertGreaterEqual(np.min(g['candidate_scales']),1/9)

    def test_new_untouched_audit_alert_rejects_candidate_and_preserves_baseline(self):
        train,val=records();ids=sorted(r['run_id'] for r in val)
        order=np.random.default_rng(42).permutation(len(ids));audit_id=ids[order[len(ids)//2]]
        next(r for r in val if r['run_id']==audit_id)['current'][:,0]=.0001*np.sin(np.arange(64)*.4)
        cfg=WindowConfig(gate_scope='center_views_v3',noise_scale_floor=True)
        policy=WindowPolicy(fit_window_policy(train,val,{},cfg));g=policy.payload['groups']['fixture']
        # New correlation with busy is relevant to BOTH centers, so both reject.
        self.assertEqual(g['center_adoption_mask'],[False,False]);self.assertGreater(g['new_audit_alert_runs'],0)
        legacy=fit_window_policy(train,val,{})['groups']['fixture']
        self.assertEqual(g['scales'],legacy['scales']);self.assertEqual(g['window_threshold'],legacy['window_threshold'])
        current=np.zeros((64,2));current[16:48,0]=.0005
        result=policy.analyze(current,np.zeros_like(current),np.zeros_like(current),'fixture',['quiet','busy'],selected=['quiet'],center_sensor='quiet')
        self.assertFalse(result['changed']);self.assertIn('retained',result['change_decision_source'])

    def test_unrelated_audit_alarm_does_not_reject_clean_selected_center(self):
        train,val=records();ids=sorted(r['run_id'] for r in val)
        order=np.random.default_rng(42).permutation(len(ids));audit_id=ids[order[len(ids)//2]]
        next(r for r in val if r['run_id']==audit_id)['residual'][:,1]=20.
        cfg=WindowConfig(gate_scope='center_views_v3',noise_scale_floor=True)
        policy=WindowPolicy(fit_window_policy(train,val,{},cfg));g=policy.payload['groups']['fixture']
        self.assertEqual(g['center_adoption_mask'],[True,False])
        current=np.zeros((64,2));current[16:48,0]=.0005;zero=np.zeros_like(current)
        self.assertTrue(policy.analyze(current,zero,zero,'fixture',['quiet','busy'],selected=['quiet'],center_sensor='quiet')['changed'])
        legacy=WindowPolicy(fit_window_policy(train,val,{}))
        actual=policy.analyze(current,zero,zero,'fixture',['quiet','busy'],selected=['busy'],center_sensor='busy')
        expected=legacy.analyze(current,zero,zero,'fixture',['quiet','busy'],selected=['busy'])
        self.assertEqual(actual['ranked_evidence'],expected['ranked_evidence'])
        from pecvd.experiments.threshold_comparison import center_audit_table
        audit=center_audit_table({.95:policy})
        self.assertEqual(audit.center_gate_adopted.tolist(),[True,False])

    def test_duplicate_windows_do_not_create_independent_calibration_or_audit_runs(self):
        train,val=records();cfg=WindowConfig(gate_scope='center_views_v3',noise_scale_floor=True)
        first=fit_window_policy(train,val,{},cfg)['groups']['fixture']
        second=fit_window_policy(train,val*3,{},cfg)['groups']['fixture']
        self.assertEqual(first['center_thresholds'],second['center_thresholds'])
        self.assertEqual(first['audit_runs'],second['audit_runs']);self.assertEqual(first['calibration_validation_runs'],8)

    def test_quantile_comparison_never_changes_normalization_between_rows(self):
        from pecvd.experiments.threshold_comparison import fit_comparison
        train,val=records();cfg=WindowConfig(gate_scope='center_views_v3',noise_scale_floor=True)
        def fitting(t,v,identity,config):
            result=fit_window_policy(t,v,identity,config)
            if config.gate_scope=='center_views_v3':
                result['groups']['fixture']['center_gate_adopted']=config.nominal_window_alert_rate>.03
            return result
        with patch('pecvd.experiments.threshold_comparison.fit_window_policy',side_effect=fitting):
            result=fit_comparison(train,val,lambda cfg:{},cfg)
        self.assertFalse(result[.95].payload['groups']['fixture']['center_gate_adopted'])
        self.assertFalse(result[.99].payload['groups']['fixture']['center_gate_adopted'])
        self.assertEqual(result[.95].payload['groups']['fixture']['scales'],result[.99].payload['groups']['fixture']['scales'])

    def test_threshold_comparison_bypasses_and_restores_learned_head(self):
        import os
        from pecvd.experiments.threshold_comparison import run_threshold_comparison,fit_comparison
        from pecvd.experiments.reporting import ExperimentReport
        train,val=records();policies=fit_comparison(train,val,lambda cfg:{})
        runtime=SimpleNamespace(module=SimpleNamespace(DISCOVERY_POLICY=policies[.95],ENCODER_ROOT='fixture',BACKBONE_PATH='fixture'))
        def evaluate(*args,**kwargs):
            self.assertNotIn('GEVIS_CENTER_DECISION_PATH',os.environ)
            details=pd.DataFrame([dict(step='fixture',pair_id='pair',kind='spike',strength=1.,expected_sensor='quiet',
                effective=True,raw_attention_sensor_correct=True,injected=True,alert=True,top_sensor_correct=True)])
            return ExperimentReport('fixture',pd.DataFrame({'recall':[1.]}),pd.DataFrame({'step':['fixture']}),details,
                diagnostics={'by_kind_and_strength':pd.DataFrame({'kind':['spike']})})
        with patch.dict('os.environ',{'GEVIS_CENTER_DECISION_PATH':'fixture-head'}),patch('pecvd.experiments.threshold_comparison.load_comparison',return_value=policies),patch('pecvd.experiments.discovery_reporting.run_discovery_experiment',side_effect=evaluate):
            run_threshold_comparison('fixture',runtime=runtime)
            self.assertEqual(os.environ['GEVIS_CENTER_DECISION_PATH'],'fixture-head')
        self.assertIs(runtime.module.DISCOVERY_POLICY,policies[.95])

    def test_joint_quantiles_preserve_clean_center_when_other_center_differs(self):
        from pecvd.experiments.threshold_comparison import fit_comparison
        train,val=records();cfg=WindowConfig(gate_scope='center_views_v3',noise_scale_floor=True)
        def fitting(t,v,identity,config):
            result=fit_window_policy(t,v,identity,config)
            result['groups']['fixture']['center_adoption_mask']=[True,config.nominal_window_alert_rate>.03]
            return result
        with patch('pecvd.experiments.threshold_comparison.fit_window_policy',side_effect=fitting):
            result=fit_comparison(train,val,lambda cfg:{},cfg)
        for policy in result.values():
            g=policy.payload['groups']['fixture']
            self.assertEqual(g['center_adoption_mask'],[True,False])
            self.assertTrue(g['center_gate_adopted'])

    def test_dynamic_challenge_reports_unavailable_instead_of_making_flat_faults(self):
        from pecvd.experiments.discovery_reporting import dynamic_challenges
        current=pd.DataFrame({'a':np.ones(32),'b':np.ones(32)})
        runtime=SimpleNamespace(encoders={'fixture':(None,{'sensor_columns':['a','b'],'std':[1.,1.]})},module=SimpleNamespace())
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(current,current)]),patch('pecvd.experiments.discovery_reporting.review_changes',side_effect=AssertionError('No varying signal')):
            report=dynamic_challenges(runtime,['fixture'],'validation',1,42)
        self.assertEqual(len(report['dynamic_challenge']),5)
        self.assertTrue(report['dynamic_challenge']['detection_recall'].isna().all())
        self.assertTrue(report['dynamic_challenge']['effective_cases'].eq(0).all())

    def test_dynamic_challenge_uses_real_variation_and_full_interval_separately(self):
        from pecvd.experiments.discovery_reporting import dynamic_challenges
        t=np.arange(64)*.4;frame=pd.DataFrame({'a':np.sin(t),'b':np.sin(t+.1)})
        runtime=SimpleNamespace(encoders={'fixture':(None,{'sensor_columns':['a','b'],'std':[1.,1.]})},
            module=SimpleNamespace(ReferenceInfo=lambda *args:object()))
        def review(module,current,reference,step,info,**kwargs):
            self.assertEqual(set(kwargs),{'center_sensor'})
            changed=bool((current[['a','b']].to_numpy()!=reference[['a','b']].to_numpy()).any())
            return dict(changed=changed,self_analysis={'changed':changed,'events':[{'statistic':'drift'}] if changed else []},
                        relationship_analysis={'changed':changed,'events':[{'statistic':'relationship'}] if changed else []})
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(frame,frame)]),patch('pecvd.experiments.discovery_reporting.review_changes',side_effect=review):
            report=dynamic_challenges(runtime,['fixture'],'validation',1,42)
        self.assertTrue(report['dynamic_challenge']['effective_cases'].gt(0).all())
        self.assertTrue(report['dynamic_challenge']['evaluation_design'].eq('varying_input_full_interval_v1').all())
        self.assertFalse(report['dynamic_challenge_controls']['untouched_alert'].any())
        own=report['dynamic_challenge'].query("kind != 'relationship'")
        self.assertTrue(own.detection_recall.eq(1.).all())
        self.assertTrue(own.mechanism_recall.eq(0.).all())
