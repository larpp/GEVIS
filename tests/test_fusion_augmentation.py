"""Synthetic tests for signed span coverage and conditional answer provenance."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from pecvd.experiments import fusion_answers as fa
from tests.test_fusion_answers import runtime, source_rows


class FusionAugmentationTests(unittest.TestCase):
    def test_signed_spans_are_deterministic_copy_only_and_truth_is_exact(self):
        base=np.zeros((64,2));before=base.copy()
        a=list(fa.paired_variants(base,0,0,42))
        b=list(fa.paired_variants(base,0,0,42))
        self.assertEqual(len(a),15)
        np.testing.assert_array_equal(base,before)
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x['observed'],y['observed'])
            np.testing.assert_array_equal(x['mask'],x['observed']!=base)
            self.assertFalse(x['mask'][:,1].any())
            self.assertTrue(0<=x['start']<x['end']<=len(base))
            if x['kind'] not in ('equal','delay'):
                self.assertEqual(x['strength'],(.5,1.,2.,4.)[x['level']-1])
                delta=x['observed'][:,0]-base[:,0]
                if x['kind']!='variance':
                    self.assertTrue((delta*x['direction']>=0).all())
                    self.assertAlmostEqual(np.abs(delta).max(),x['strength'])
        for kind in ('level_shift','drift','spike'):
            self.assertEqual({x['direction'] for x in a if x['kind']==kind},{-1,1})
        self.assertEqual({x['direction'] for x in a if x['kind']=='variance'},{1})
        self.assertGreater(len({(x['start'],x['end']) for x in a}),3)
        c=list(fa.paired_variants(base,0,0,100045))
        self.assertTrue(any(not np.array_equal(x['observed'],y['observed'])
                            for x,y in zip(a,c)))

    def test_actual_fusion_targets_preserve_type_and_local_stage_across_signs(self):
        r=runtime()
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons',
                   return_value=source_rows('train')),contextlib.redirect_stdout(io.StringIO()):
            rows=fa.controlled_examples(r.module,'train')
        self.assertEqual(len({e['pair_id'] for e in rows}),4*15)
        for e in rows:
            self.assertEqual(e['question'],fa.QUESTIONS[e['answer_type']])
            source_index=int(e['run_id'].rsplit('-',1)[1])
            changed=e['injected'] and e['center_sensor']==('a','b')[source_index%2]
            self.assertEqual(e['center_is_perturbed'],changed)
            if not changed:self.assertEqual(e['label'],0)
            elif e['answer_type']=='CHOICE':self.assertEqual(e['label'],fa.TYPE_TARGET[e['kind']])
            elif e['answer_type']=='ORDINAL':
                self.assertEqual(e['label'],e['lag_samples'] if e['kind']=='delay' else (.5,1.,2.,4.).index(e['strength'])+1)
            else:self.assertEqual(e['label'],1)
        negative=[e for e in rows if e['direction']==-1 and e['center_is_perturbed']]
        self.assertTrue(negative)
        self.assertEqual({e['kind'] for e in negative},{'level_shift','drift','spike'})
        pairs={}
        for e in rows:
            if e['answer_type']=='BOOLEAN' and e['injected']:
                pairs.setdefault(e['pair_id'],[]).append(e['label'])
        self.assertTrue(all(set(labels)=={0,1} for labels in pairs.values()))

    def test_split_mismatch_and_invalid_windows_fail_before_training(self):
        with self.assertRaises(ValueError):list(fa.paired_variants(np.zeros((3,2)),0,0,42))
        with self.assertRaises(ValueError):list(fa.paired_variants(np.zeros((8,2)),2,0,42))
        with self.assertRaises(ValueError):fa.controlled_examples(runtime().module,'unknown')
        with patch('pecvd.experiments.discovery_calibration.cached_comparisons',
                   return_value=source_rows('test')),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'another split'):
                fa.controlled_examples(runtime().module,'train')
