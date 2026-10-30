import unittest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from pecvd.experiments.optimization_data import restrict_training_references,collect_protocol_rows


class OptimizationDataTests(unittest.TestCase):
    def test_member_context_keeps_exact_pool_and_rejects_self_or_mismatch(self):
        from pecvd.experiments.optimization_data import attach_reference_context
        a=np.tile([-1.,1.],8)[:,None];values=np.stack([a,-a,np.ones_like(a)*100])
        row=dict(run_id='heldout',reference_run_ids=['a','b'],cohort='c',position=0,
                 reference=np.zeros_like(a),current=a)
        args=(values,np.array(['a','b','excluded']),np.array(['c']*3),np.zeros(3,int))
        result=attach_reference_context([row],*args)[0]
        self.assertIs(result['current'],a);self.assertIs(result['reference'],row['reference'])
        self.assertGreater(result['reference_context'][:,:,1].min(),.8)
        with self.assertRaises(ValueError):attach_reference_context([dict(row,run_id='a')],*args)
        with self.assertRaises(ValueError):attach_reference_context([dict(row,reference=a)],*args)

    def test_references_exclude_selection_and_self_without_mutating_current(self):
        values=np.array([np.full((4,2),v) for v in (1.,3.,100.)])
        ids=np.array(['fit-a','fit-b','selection']);cohorts=np.array(['c']*3);positions=np.zeros(3,int)
        rows=[dict(split='train',run_id=key,cohort='c',position=0,current=values[i].copy(),reference=values[2].copy()) for i,key in enumerate(ids)]
        out=restrict_training_references(rows,values,ids,cohorts,positions,{'fit-a','fit-b'},max_references=8)
        np.testing.assert_array_equal(out[0]['reference'],values[1])
        np.testing.assert_array_equal(out[2]['reference'],np.full((4,2),2.))
        for old,new in zip(rows,out):
            self.assertIs(old['current'],new['current']);self.assertNotIn('selection',new['reference_run_ids'])
            self.assertNotIn(old['run_id'],new['reference_run_ids'])
            np.testing.assert_array_equal(old['reference'],values[2])

    def test_missing_support_and_nontrain_fail_closed(self):
        values=np.ones((2,4,2));ids=np.array(['a','b']);cohorts=np.array(['c','d']);positions=np.zeros(2,int)
        row=dict(split='train',run_id='a',cohort='c',position=0)
        with self.assertRaises(ValueError):restrict_training_references([row],values,ids,cohorts,positions,{'a','b'},max_references=2)
        with self.assertRaises(ValueError):restrict_training_references([dict(row,split='validation')],values,ids,cohorts,positions,{'a','b'},max_references=2)

    def test_collect_preserves_validation_and_isolates_reference_membership(self):
        ids=np.array(['a','b','c','d']);values=np.arange(32,dtype=float).reshape(4,4,2)
        validation=[]
        def comparisons(root,step,*,split,config):
            for i,key in enumerate(ids if split=='train' else ['e','f','g','h']):
                row=dict(split=split,run_id=key,cohort='c',position=0,current=values[i].copy(),
                    reference=values.mean(0),reference_run_ids=[x for x in ids if x!=key])
                if split=='validation':validation.append(row)
                yield row
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'windows').mkdir()
            np.savez(root/'windows'/'phase.npz',train=values,train_process_id=ids,
                train_cohort_key=np.array(['c']*4),train_position_bin=np.zeros(4,int))
            runtime=SimpleNamespace(encoders={'phase':None},module=SimpleNamespace(ENCODER_ROOT=root,
                DISCOVERY_POLICY=SimpleNamespace(config=SimpleNamespace(max_references=8))))
            config=dict(train_runs_per_step=4,validation_runs_per_step=4,train_selection_fraction=.25,seed=1)
            with patch('pecvd.experiments.discovery_calibration.cached_comparisons',side_effect=comparisons):
                rows,audit=collect_protocol_rows(runtime,config)
        self.assertEqual(audit['prior_fit_windows_using_selection_reference'],3)
        self.assertEqual(audit['restricted_train_windows_using_selection_reference'],0)
        for expected,actual in zip(validation,rows['phase']['validation']):
            self.assertIs(expected['current'],actual['current']);self.assertIs(expected['reference'],actual['reference'])


if __name__=='__main__':unittest.main()
