import unittest
import torch
from pecvd.experiments.discovery import _learned_generic_attention, _rank_with_attention
from types import SimpleNamespace
import numpy as np


class GenericGroundingTests(unittest.TestCase):
    def test_reference_curriculum_uses_real_reference_and_preserves_inputs(self):
        from pecvd.gevis_backbone import reference_evidence_pairs,numeric_evidence_loss
        current=torch.zeros(8,16,4);reference=torch.ones_like(current)*.1
        changed,source,truth=reference_evidence_pairs(current,reference,4,seed=42)
        self.assertTrue(truth['cross_run'].any());self.assertTrue((~truth['cross_run']).any())
        _,_,single=reference_evidence_pairs(current[:1],reference[:1],4,seed=43)
        self.assertTrue(single['cross_run'].all())
        torch.testing.assert_close(source[truth['cross_run']],reference.repeat_interleave(2,0)[truth['cross_run']])
        torch.testing.assert_close(source[~truth['cross_run']],current.repeat_interleave(2,0)[~truth['cross_run']])
        self.assertEqual(float(current.abs().sum()),0.)
        for i in range(len(changed)):
            observed=changed[i].ne(0)
            self.assertTrue(observed[:,truth['sensor'][i]].any())
            self.assertEqual(int(observed.any(0).sum()),1)
            self.assertTrue(torch.equal(observed.any(-1).reshape(4,4).any(-1),truth['patch_mask'][i]))
        logits=torch.randn(16,16,requires_grad=True)
        loss,_=numeric_evidence_loss(logits.softmax(-1),truth);loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_evaluation_sensor_assignment_changes_with_seed(self):
        import pandas as pd
        from pecvd.experiments.evaluation import intervention_cases
        frame=pd.DataFrame(np.zeros((24,12)),columns=[f's{i}' for i in range(12)])
        def cases(seed):return list(intervention_cases(frame,list(frame),np.ones(12),seed=seed,max_cases=14))
        first=cases(42);second=cases(43)
        self.assertEqual(len(first),15)
        self.assertEqual(len({(s.kind,s.strength) for s,_,_ in first[1:]}),14)
        self.assertNotEqual([s.sensor for s,_,_ in first[1:]],[s.sensor for s,_,_ in second[1:]])
        for (a,x,_),(b,y,_) in zip(first,cases(42)):
            self.assertEqual(a,b);np.testing.assert_array_equal(x,y)

    def test_generic_baseline_comparison_never_scans_named_questions(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.baselines import compare_frozen_runtime
        from unittest.mock import patch
        runtime,current,reference,_=fixture()
        runtime.module.GEVIS_CHECKPOINT['schema']='gevis.backbone.generic.v10'
        def pairs(*args,limit=2,**kwargs):return [(current,reference)]*limit
        with patch('pecvd.experiments.evaluation.load_cached_pairs',side_effect=pairs), \
             patch('pecvd.experiments.evaluation.runtime_scan',side_effect=AssertionError('No named scan')):
            summary,_=compare_frozen_runtime(runtime,'invented_phase',validation_limit=2,test_limit=1,max_cases=2)
        self.assertIn('GEVIS_shared_evidence_discovery',set(summary.method))
        self.assertNotIn('GEVIS_candidate_score',set(summary.method))



    def test_missing_model_never_fabricates_attention(self):
        with self.assertRaises(RuntimeError):
            _learned_generic_attention(SimpleNamespace(),torch.zeros(1,6,8),'phase',{'sensor_columns':['a','b']},3)

    def test_learned_rank_changes_winner_without_inventing_events(self):
        candidates=[dict(sensor='a',normalized_score=9.),dict(sensor='b',normalized_score=2.)]
        events=list(candidates)
        _rank_with_attention(events,candidates,['a','b'],np.array([.1,.9]),np.ones((2,3))/3)
        self.assertEqual(events[0]['sensor'],'b')
        empty=[]
        _rank_with_attention(empty,candidates,['a','b'],np.array([.9,.1]),np.ones((2,3))/3)
        self.assertEqual(empty,[])


if __name__=='__main__':unittest.main()
