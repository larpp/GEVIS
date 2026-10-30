import unittest
import numpy as np
import torch
from pecvd.experiments.sensor_discrimination import discrimination_loss,TrainReferencePairs

class SensorDiscriminationTests(unittest.TestCase):
    def test_relation_pairs_preserve_marginals_and_mark_real_partner_changes(self):
        from pecvd.experiments.sensor_discrimination import center_relation_pairs
        x=torch.randn(6,32,4)
        changed,source,truth=center_relation_pairs(x,x,4,seed=13)
        self.assertTrue(torch.all(truth['center_sensor']!=truth['sensor']))
        torch.testing.assert_close(truth['center_sensor'][::2],truth['center_sensor'][1::2])
        torch.testing.assert_close(source[::2],source[1::2])
        torch.testing.assert_close(changed.sort(dim=1).values,source.sort(dim=1).values)
        delta=changed-source
        torch.testing.assert_close(delta.abs().sum(1).argmax(-1),truth['sensor'])
        for i in range(len(changed)):
            expected=delta[i,:,truth['sensor'][i]].reshape(8,4).ne(0).any(-1)
            torch.testing.assert_close(expected,truth['patch_mask'][i])
        flat=center_relation_pairs(torch.zeros_like(x),torch.zeros_like(x),4,seed=13)
        self.assertTrue(flat[2]['patch_mask'].any(-1).all())

    def test_correct_moved_sensor_beats_stationary_attention_and_backpropagates(self):
        targets={'sensor':torch.tensor([0,1]),'patch':torch.tensor([0,0]),'patch_count':2}
        good=torch.tensor([[.7,.1,.1,.1],[.1,.1,.7,.1]],requires_grad=True)
        wrong=torch.tensor([[.7,.1,.1,.1],[.7,.1,.1,.1]],requires_grad=True)
        low,metrics=discrimination_loss(good,targets);high,bad=discrimination_loss(wrong,targets)
        self.assertLess(float(low.detach()),float(high.detach()));self.assertEqual(metrics['moved_sensor_pair_accuracy'],1.)
        self.assertEqual(bad['moved_sensor_pair_accuracy'],0.)
        high.backward();self.assertTrue(torch.isfinite(wrong.grad).all());self.assertGreater(float(wrong.grad.abs().sum()),0.)

    def test_zero_auxiliary_weights_preserve_original_loss(self):
        from pecvd.gevis_backbone import numeric_evidence_loss
        attention=torch.tensor([[.4,.1,.3,.2],[.2,.1,.3,.4]])
        targets={'sensor':torch.tensor([0,1]),'patch_count':2,'patch_mask':torch.tensor([[True,False],[False,True]])}
        old,_=numeric_evidence_loss(attention,targets)
        new,_=discrimination_loss(attention,targets,hard_weight=0,pair_weight=0)
        torch.testing.assert_close(old,new)
        with self.assertRaises(ValueError):discrimination_loss(attention,targets,margin=-1)

    def test_validation_references_are_train_only_and_self_excluded(self):
        train=np.stack([np.full((4,2),v,dtype=np.float32) for v in (1.,3.,90.)])
        meta={'process_id':np.array(['a','b','c']),'cohort_key':np.array(['x','x','y']),'position_bin':np.zeros(3,int)}
        val=np.full((1,4,2),1000.,np.float32)
        vm={'process_id':np.array(['v']),'cohort_key':np.array(['x']),'position_bin':np.zeros(1,int)}
        ds=TrainReferencePairs(val,vm,train,meta,42,None,split='validation')
        current,reference=ds[0];self.assertTrue(torch.all(current==1000));self.assertTrue(torch.all(reference==2))
        ds=TrainReferencePairs(train,meta,train,meta,42,None,split='train')
        self.assertTrue(torch.all(ds[0][1]==3))
        with self.assertRaises(ValueError):TrainReferencePairs(train,meta,train,meta,42,None,split='validation')
