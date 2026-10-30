"""Questions cannot invent localization truth; injected inputs can."""
import unittest
import torch
from pecvd.experiments.questions import question_bank, legacy_named_question_bank, localization_questions
from pecvd.gevis_backbone import reference_evidence_pairs, evidence_questions


class QuestionTruthTests(unittest.TestCase):

    def test_fixed_training_runtime_prompt_ignores_ui_identifiers(self):
        from unittest.mock import patch
        import os
        with patch.dict(os.environ,{'GEVIS_DISCOVERY_TRAINING':'fixed_questions_v13'}):
            self.assertEqual(evidence_questions('DEP.',2),['Is there a change in this interval?']*2)
            self.assertEqual(evidence_questions('other selected step',2),evidence_questions('DEP.',2))

    def test_temporal_curriculum_targets_actual_changed_coordinates(self):
        # Nonflat and flat inputs both require real nonempty target patches.
        for current in (torch.arange(32.).reshape(1,32,1).repeat(12,1,3),torch.zeros(12,32,3)):
            changed,reference,truth=reference_evidence_pairs(current,current,4,seed=7,background='self',temporal=True)
            delta=changed-reference
            torch.testing.assert_close(delta.abs().sum(1).argmax(-1),truth['sensor'])
            for row in range(len(changed)):
                actual=delta[row,:,truth['sensor'][row]].reshape(8,4).ne(0).any(-1)
                torch.testing.assert_close(actual,truth['patch_mask'][row])
                self.assertTrue(actual.any())

    def test_stage1_fixed_questions_and_sensor_choice(self):
        from pecvd.experiments.questions import DISCOVERY_QUESTIONS, sensor_choice
        self.assertEqual(DISCOVERY_QUESTIONS['BOOLEAN'],'Is there a change in this interval?')
        answer=sensor_choice(['a','b'],[1.,3.],accepted=True,changed=False)
        self.assertEqual(answer['answer'],'b')
        self.assertEqual(answer['probabilities'],{'a':.25,'b':.75})
        self.assertTrue(answer['reference_only'])
        self.assertNotIn('No change',answer['probabilities'])
        withheld=sensor_choice(['a','b'],[1.,3.],accepted=False,changed=None)
        self.assertIsNone(withheld['answer'])
        self.assertEqual(withheld['probabilities'],{})
        for values in ([0.,0.],[float('nan'),1.],[-1.,2.]):
            with self.assertRaises(ValueError):sensor_choice(['a','b'],values,accepted=True,changed=False)

    def test_sensor_free_bank_has_no_arbitrary_targets_or_duplicates(self):
        for split in ('train', 'validation', 'test'):
            first=question_bank('fixture', ['a', 'b'], split=split, limit=10000)
            self.assertEqual(first, question_bank('fixture', ['other'], split=split, limit=10000))
            self.assertTrue(all(c.sensor is None for c in first))
            self.assertEqual(len(first), len({c.text for c in first}))
            self.assertEqual({c.answer_type for c in first}, {'BOOLEAN','CHOICE','ORDINAL'})

    def test_legacy_named_truth_is_explicit_in_text(self):
        for case in legacy_named_question_bank('fixture', ['sensor_a','sensor_b']):
            self.assertIn(f'Inspect only sensor {case.sensor}.', case.text)

    def test_same_query_different_target_requires_different_numeric_input(self):
        current=torch.zeros(8,24,4)
        changed,reference,truth=reference_evidence_pairs(current,current,4,seed=42,background='self')
        delta=changed-reference
        actual=delta.abs().sum(1).argmax(-1)
        torch.testing.assert_close(actual,truth['sensor'])
        self.assertTrue(torch.all(truth['sensor'][::2]!=truth['sensor'][1::2]))
        for row in range(len(changed)):
            affected=delta[row,:,int(truth['sensor'][row])].reshape(6,4).ne(0).any(-1)
            torch.testing.assert_close(affected,truth['patch_mask'][row])
        prompts=evidence_questions('fixture',len(changed))
        self.assertEqual(prompts,localization_questions('fixture',len(changed)))
        self.assertEqual(len(set(prompts)),1)

