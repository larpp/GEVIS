import json,tempfile,unittest
from pathlib import Path
import numpy as np
from pecvd.experiments.reference_policy_v2 import ReferenceConfig,ReferencePolicy,fit_records,save_policy,load_policy

def data(n=12,zero=False):
    for i in range(n):
        values=np.zeros((3,2,5)) if zero else np.full((3,2,5),i*.01)
        yield {'split':'train','step':'phase','run_id':str(i),'sensors':['a','b'],'features':values,'cohort':'fixture','position':0}

class ReferencePolicyV2Tests(unittest.TestCase):
    def test_small_sample_support_is_coarse_not_falsely_certified(self):
        policy=ReferencePolicy(fit_records(data(),{}))
        result=policy.decide([1]*5,'phase','a',0)
        self.assertTrue(all(result['supported'].values()))
        self.assertTrue(result['ordinal_tied_boundaries'])
        self.assertFalse(result['nominal_tail_resolution_supported'])
        self.assertAlmostEqual(result['tail_resolution'],1/12)
        self.assertTrue(result['boolean_any'])
        self.assertEqual(result['labels']['CHOICE'],1)
        self.assertEqual(result['labels']['ORDINAL'],4)

    def test_zero_boundaries_are_valid_flat_reference_not_missing_evidence(self):
        policy=ReferencePolicy(fit_records(data(zero=True),{}))
        unchanged=policy.decide([0]*5,'phase','a',0)
        changed=policy.decide([.2,0,0,0,0],'phase','a',0)
        self.assertTrue(all(unchanged['supported'].values()))
        self.assertEqual(unchanged['labels'],{'BOOLEAN':0,'CHOICE':0,'ORDINAL':0})
        self.assertEqual(changed['raw_empirical_threshold'],0)
        self.assertGreater(changed['threshold'],0)
        self.assertEqual(changed['labels']['BOOLEAN'],1)
        self.assertEqual(changed['labels']['CHOICE'],1)
        for probabilities in changed['probabilities'].values(): self.assertAlmostEqual(sum(probabilities),1.)

    def test_signed_direction_and_numerical_tolerance(self):
        policy=ReferencePolicy(fit_records(data(zero=True),{}))
        self.assertEqual(policy.decide([-1,0,0,0,0],'phase','a',0,direction='increase')['labels']['BOOLEAN'],0)
        self.assertEqual(policy.decide([-1,0,0,0,0],'phase','a',0,direction='decrease')['labels']['BOOLEAN'],1)
        self.assertEqual(policy.decide([1e-8,0,0,0,0],'phase','a',0)['labels']['BOOLEAN'],0)

    def test_sparse_unknown_sensors_and_held_out_fitting_still_fail_closed(self):
        policy=ReferencePolicy(fit_records(data(7),{}))
        self.assertFalse(any(policy.decide([1]*5,'phase','a',0)['supported'].values()))
        policy=ReferencePolicy(fit_records(data(),{}))
        self.assertFalse(any(policy.decide([1]*5,'phase','unknown',0)['supported'].values()))
        with self.assertRaises(ValueError): fit_records([{**next(data()),'split':'test'}],{})

    def test_roundtrip_and_no_overwrite(self):
        payload=fit_records(data(),{'synthetic':'exact'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'v2.json'; save_policy(payload,path)
            self.assertEqual(load_policy(path,{'synthetic':'exact'}).payload['schema'],'gevis.reference_policy.v2')
            with self.assertRaises(FileExistsError): save_policy(payload,path)
            with self.assertRaises(ValueError): load_policy(path,{'different':True})

if __name__=='__main__': unittest.main()
