import unittest
import numpy as np
import torch
from pecvd.experiments.injection import Injection,inject
from pecvd.experiments.optimization_heads import patch_rows,temporal_targets,compatible_type_targets,decode_interval,GroundedOutputHeads,GroundedOutputTrial,output_loss,TYPE_LABELS
from pecvd.experiments.optimization_models import PairedEncoderTrial


class GroundedOutputTests(unittest.TestCase):
    def test_native_rows_and_singletons(self):
        np.testing.assert_array_equal(patch_rows(7,3),[[0,3],[3,5],[5,7]])
        rows=patch_rows(2,5)
        self.assertTrue((rows[:,0]>=0).all());self.assertTrue((rows[:,1]<=2).all())
        mask=np.zeros((7,2),bool);mask[4,1]=True
        np.testing.assert_array_equal(temporal_targets(mask,3),[[0,0,0],[0,1,0]])

    def test_decoder_uses_peak_component_not_truth(self):
        self.assertEqual(decode_interval([.8,.7,.1,.99],8),[6,8])
        self.assertIsNone(decode_interval([.1,.2],8))
        with self.assertRaises(ValueError):decode_interval([float('nan')],8)

    def test_aliases_are_not_forced_into_one_class(self):
        before=np.arange(16,dtype=float).reshape(8,2);scale=np.ones(2)
        spec=Injection('drift',0,2,4,2.,7);after,_=inject(before,spec,scale)
        allowed=compatible_type_targets(before,after,spec,scale)
        self.assertTrue(allowed[TYPE_LABELS.index('drift')]);self.assertTrue(allowed[TYPE_LABELS.index('spike')])
        self.assertFalse(allowed[0])

    def test_heads_are_equivariant_and_trainable(self):
        torch.manual_seed(4);head=GroundedOutputHeads(8,24)
        tokens=torch.randn(2,3,4,8);context=torch.randn(2,3,24)
        result=head(tokens,context);permutation=[2,0,1]
        permuted=head(tokens[:,permutation],context[:,permutation])
        for key in result:torch.testing.assert_close(permuted[key],result[key][:,permutation])
        truth=torch.zeros(2,3,4);truth[:,1,1:3]=1
        kinds=torch.zeros(2,3,len(TYPE_LABELS),dtype=torch.bool);kinds[:,:,0]=True
        kinds[:,1,0]=False;kinds[:,1,1]=True
        loss=output_loss(result,truth,kinds);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(head.temporal.weight.grad)

    def test_fit_preserves_detector_and_rejects_leakage(self):
        rng=np.random.default_rng(4);examples=[]
        for i in range(8):
            changed=i%2==1;mask=np.zeros((8,3),bool)
            if changed:mask[2:4,1]=True
            kinds=np.zeros((3,len(TYPE_LABELS)),bool);kinds[:,0]=True
            if changed:kinds[1,0]=False;kinds[1,1]=True
            examples.append(dict(split='train',run_id=str(i//2),target=1 if changed else None,
                features=rng.normal(size=(3,8,20)).astype(np.float32),
                patch_targets=temporal_targets(mask,8),type_targets=kinds))
        detector=PairedEncoderTrial('paired_cross_attention',np.ones(12),width=16)
        detector.fit(examples[:4],examples[4:],epochs=1,batch_size=2)
        original=detector.scores(examples[0]['features']).copy()
        heads=GroundedOutputTrial(detector)
        with self.assertRaises(ValueError):heads.fit(examples[:4],examples[:4],epochs=1)
        with self.assertRaises(ValueError):heads.fit(examples[:4],[dict(e,split='validation') for e in examples[4:]],epochs=1)
        heads.fit(examples[:4],examples[4:],epochs=2,batch_size=2)
        np.testing.assert_array_equal(original,detector.scores(examples[0]['features']))
        result=heads.predict(examples[0]['features'])
        np.testing.assert_array_equal(original,result['sensor_scores'])
        changed_query=heads.predict(examples[0]['features'],query=-np.ones(12))
        np.testing.assert_array_equal(changed_query['sensor_scores'],detector.scores(examples[0]['features'],query=-np.ones(12)))
        self.assertEqual(result['patch_probabilities'].shape,(3,8))
        self.assertTrue(np.isfinite(result['type_probabilities']).all())


if __name__=='__main__':unittest.main()
