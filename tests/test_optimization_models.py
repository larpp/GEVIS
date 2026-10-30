import unittest
import numpy as np
import torch
from pecvd.experiments.optimization_models import paired_patch_features,continuous_difference_features,split_runs,PairedPatchDetector,PairedEncoderTrial


class PairedEncoderTests(unittest.TestCase):
    def test_member_context_does_not_flatten_opposite_phase_oscillation(self):
        from pecvd.experiments.optimization_models import local_waveform_statistics,local_context_features
        a=np.tile([-1.,1.],16)[:,None];b=-a;median=(a+b)/2
        member=np.median(np.stack([local_waveform_statistics(a),local_waveform_statistics(b)]),axis=0)
        self.assertGreater(member[:,:,1].min(),.8)
        self.assertEqual(local_waveform_statistics(median)[:,:,1].max(),0.)
        base=paired_patch_features(a,median,32)
        out=local_context_features(a,median,base,reference_summary=member)
        np.testing.assert_array_equal(out[:,:,25],0.)
        np.testing.assert_array_equal(out[:,:,:20],base)

    def test_local_context_preserves_raw_features_and_observes_singleton_variance(self):
        from pecvd.experiments.optimization_models import local_context_features
        a=np.tile([0.,2.],16)[:,None];b=np.zeros_like(a)
        base=continuous_difference_features(a,b,paired_patch_features(a,b,32))
        out=local_context_features(a,b,base)
        self.assertEqual(out.shape,(1,32,34))
        np.testing.assert_array_equal(out[:,:,:20],base)
        self.assertTrue(np.all(out[:,:,23]>0))
        self.assertTrue(np.all(base[:,:,3]==0))
        equal=local_context_features(a,a,base)
        np.testing.assert_array_equal(equal[:,:,[22,25,26,29,32,33]],0.)
        with self.assertRaises(ValueError):local_context_features(a,b,base,windows=(2,))

    def test_local_context_is_sensor_equivariant_and_finite_on_short_constant_input(self):
        from pecvd.experiments.optimization_models import local_context_features
        rng=np.random.default_rng(42);a=rng.normal(size=(21,3));b=rng.normal(size=(21,3))
        base=paired_patch_features(a,b,8);out=local_context_features(a,b,base)
        perm=[2,0,1]
        np.testing.assert_allclose(local_context_features(a[:,perm],b[:,perm],base[perm]),out[perm])
        x=np.ones((1,2));f=local_context_features(x,x,paired_patch_features(x,x,8))
        self.assertTrue(np.isfinite(f).all());self.assertEqual(f.shape,(2,8,34))

    def test_actual_query_input_reaches_attention_and_default_is_preserved(self):
        torch.manual_seed(3);q=np.ones(12);model=PairedPatchDetector(20,q,'paired_cross_attention',16).eval()
        x=torch.randn(2,3,8,20)
        torch.testing.assert_close(model(x),model(x,query=q),rtol=0,atol=0)
        self.assertFalse(torch.equal(model(x),model(x,query=-q)))
        with self.assertRaises(ValueError):model(x,query=np.ones(4))
        with self.assertRaises(ValueError):model(x,query=np.full(12,np.nan))
    def test_dilation_preserves_parameter_budget_and_initialization(self):
        models=[]
        for dilation in (1,4,16):
            torch.manual_seed(4)
            models.append(PairedPatchDetector(20,np.ones(12),'paired_cross_attention',16,dilation))
        for model in models[1:]:
            for key,value in models[0].state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        x=torch.randn(2,3,64,20)
        for model in models:self.assertEqual(model(x).shape,(2,3))
        self.assertFalse(torch.equal(models[0](x),models[1](x)))
        with self.assertRaises(ValueError):PairedPatchDetector(20,np.ones(12),'paired_cross_attention',16,0)

    def test_singleton_patch_differences_are_not_silently_zeroed(self):
        a=np.arange(8,dtype=float)[:,None];b=a*.5
        base=paired_patch_features(a,b,8);out=continuous_difference_features(a,b,base)
        np.testing.assert_array_equal(base[:,:,12:17],0.)
        np.testing.assert_allclose(out[0,1:,15],np.log(2))
        np.testing.assert_allclose(out[0,1:,16],np.log(1.5))
        unchanged=list(range(12))+list(range(17,20))
        np.testing.assert_array_equal(out[:,:,unchanged],base[:,:,unchanged])
        self.assertEqual(out[0,0,15],0.)

    def test_continuous_differences_follow_sensor_permutation(self):
        rng=np.random.default_rng(2);a=rng.normal(size=(9,3));b=rng.normal(size=(9,3))
        base=paired_patch_features(a,b,16)
        first=continuous_difference_features(a,b,base)
        permutation=[2,0,1]
        second=continuous_difference_features(a[:,permutation],b[:,permutation],base[permutation])
        np.testing.assert_array_equal(first[permutation],second)
    def test_features_preserve_input_amplitude_and_sensor_permutation(self):
        rng = np.random.default_rng(7); a = rng.normal(size=(32, 3)); b = a+.1
        original = a.copy(); f = paired_patch_features(a, b, 8)
        np.testing.assert_array_equal(a, original)
        np.testing.assert_allclose(paired_patch_features(a[:,[2,0,1]], b[:,[2,0,1]],8),f[[2,0,1]])
        self.assertFalse(np.allclose(f,paired_patch_features(a*2,b*2,8)))
        self.assertEqual(paired_patch_features(a[:2],b[:2],8).shape,(3,8,20))

    def test_all_models_equivariant_and_finite(self):
        torch.manual_seed(1); x=torch.randn(2,3,8,20)
        for method in ('paired_conv','paired_cross_attention'):
            model=PairedPatchDetector(20,np.ones(12),method,16).eval()
            with torch.no_grad():
                y=model(x); perm=model(x[:,[2,0,1]])
            torch.testing.assert_close(perm,y[:,[2,0,1]],rtol=1e-5,atol=1e-6)
            self.assertEqual(y.shape,(2,3)); self.assertTrue(torch.isfinite(y).all())

    def test_split_is_global_run_disjoint(self):
        examples=[dict(run_id=str(i),step=step) for i in range(8) for step in ('a','b')]
        a,b=split_runs(examples,.5,3)
        self.assertFalse({e['run_id'] for e in a}&{e['run_id'] for e in b})
        self.assertEqual(len(a)+len(b),len(examples))

    def test_fit_rejects_leakage_and_validation(self):
        model=PairedEncoderTrial('paired_conv',np.ones(12),width=16)
        e=dict(split='train',run_id='a',target=0,features=np.zeros((3,8,20),np.float32))
        with self.assertRaises(ValueError):model.fit([e],[e],epochs=1)
        with self.assertRaises(ValueError):model.fit([e],[dict(e,split='validation',run_id='b')],epochs=1)

    def test_no_change_and_train_only_fit(self):
        rng=np.random.default_rng(3)
        examples=[dict(split='train',run_id=str(i//2),target=None if i%2==0 else 1,
            features=rng.normal(size=(3,8,20)).astype(np.float32)) for i in range(8)]
        model=PairedEncoderTrial('paired_cross_attention',np.ones(12),width=16)
        model.fit(examples[:4],examples[4:],epochs=2,batch_size=2)
        self.assertTrue(np.isfinite(model.scores(examples[0]['features'])).all())
        self.assertIn(model.best_epoch,(1,2))

    def test_control_centered_fit_does_not_use_selection_controls(self):
        rng=np.random.default_rng(8)
        examples=[dict(split='train',step='a',run_id=str(i//2),target=None if i%2==0 else 1,
            features=rng.normal(size=(3,8,20)).astype(np.float32)) for i in range(12)]
        model=PairedEncoderTrial('paired_conv',np.ones(12),width=16,control_centered=True)
        model.fit(examples[:8],examples[8:],epochs=1,batch_size=4)
        expected=np.median(np.stack([e['features'] for e in examples[:8] if e['target'] is None]),axis=0)
        np.testing.assert_array_equal(model.control_scales['a'][0],expected)
        self.assertTrue((model.control_scales['a'][1]>=.1).all())
        with self.assertRaises(ValueError):model.scores(examples[0]['features'],'missing')

    def test_fit_balanced_selection(self):
        rng=np.random.default_rng(4)
        examples=[dict(split='train',run_id=str(i//2),target=None if i%2==0 else 1,
            features=rng.normal(size=(3,8,20)).astype(np.float32)) for i in range(8)]
        model=PairedEncoderTrial('paired_cross_attention',np.ones(12),width=16)
        model.fit(examples[:4],examples[4:],epochs=1,batch_size=2,balanced_selection=True)
        self.assertTrue(np.isfinite(model.scores(examples[0]['features'])).all())

    def test_same_seed_replays_exactly_on_synthetic_cpu_data(self):
        rng=np.random.default_rng(4)
        examples=[dict(split='train',run_id=str(i//2),target=None if i%2==0 else 1,
            features=rng.normal(size=(3,8,20)).astype(np.float32)) for i in range(8)]
        models=[PairedEncoderTrial('paired_cross_attention',np.ones(12),width=16,seed=12) for _ in range(2)]
        for model in models:model.fit(examples[:4],examples[4:],epochs=2,batch_size=2,balanced_selection=True)
        self.assertEqual(models[0].input_fingerprint,models[1].input_fingerprint)
        np.testing.assert_array_equal(models[0].scores(examples[0]['features']),models[1].scores(examples[0]['features']))


if __name__=='__main__':unittest.main()
