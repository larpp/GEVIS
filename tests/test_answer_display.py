import copy,unittest
import ast,json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pecvd.answer_display import english_answer,english_result,english_frame,english_output


class AnswerDisplayTests(unittest.TestCase):
    def test_canonical_options_and_executable_notebook_text_are_english(self):
        from pecvd.answer_display import ANSWER_CRITERIA,QUESTIONS
        from pecvd.experiments.questions import CRITERIA
        from pecvd.gevis_fast import DEFAULT_ANSWER_CRITERIA,TYPED_CRITERIA
        for criteria in (CRITERIA,DEFAULT_ANSWER_CRITERIA,TYPED_CRITERIA):
            self.assertEqual({k:tuple(v) for k,v in criteria.items()},ANSWER_CRITERIA)
        self.assertTrue(all(text.isascii() for text in QUESTIONS.values()))
        root=Path(__file__).resolve().parents[1]
        forbidden=re.compile(r'[\uac00-\ud7a3\ufffd]')
        for path in list((root/'src/pecvd').rglob('*.py')):
            self.assertIsNone(forbidden.search(path.read_text(encoding='utf-8')),str(path))
        for path in list((root/'notebooks').rglob('*.ipynb')):
            for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type']!='code':continue
                source=''.join(cell['source'])
                self.assertIsNone(forbidden.search(source),str(path))
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node,ast.Constant) and isinstance(node.value,str):
                        self.assertIsNone(forbidden.search(node.value),str(path))

    def test_unicode_sensor_identifiers_are_not_relabelled(self):
        sensor=chr(0xac00)+'_physical_id'
        result={'answer_type':'CHOICE_SENSOR','answer':sensor,'probabilities':{sensor:1.}}
        self.assertEqual(english_answer(result),result)

    def test_labels_only_preserve_scores_decision_and_evidence_identity(self):
        evidence={'sensor':'sensor-A','detected_rows':(4,8)}
        original={'answer_type':'BOOLEAN','question':'Is there a change in this interval?',
            'answer':'Yes','candidate_answer':'Yes','accepted':True,'probabilities':{'No':.26894142137,'Yes':.73105857863},
            'confidence':{'answer':.73105857863},'evidence':evidence,'score_meaning':'deterministic display scores; not correctness/fault probabilities'}
        before=copy.deepcopy(original);result=english_answer(original)
        self.assertEqual(original,before);self.assertEqual(result['answer'],'Yes')
        self.assertEqual(result['probabilities'],{'No':.26894142137,'Yes':.73105857863})
        self.assertIs(result['evidence'],evidence);self.assertIs(result['confidence'],original['confidence'])
        self.assertEqual(result['accepted'],original['accepted'])
        audit=english_answer(dict(original,question='Inspect only pressure in the early interval.'))
        self.assertEqual(audit['question'],'Inspect only pressure in the early interval.')

    def test_sensor_identifiers_and_abstention_never_translated(self):
        sensor={'answer_type':'CHOICE_SENSOR','answer':'Yes','probabilities':{'Yes':.8,'No':.2}}
        self.assertEqual(english_answer(sensor),sensor)
        abstain={'answer_type':'BOOLEAN','answer':None,'candidate_answer':None,'accepted':False,'probabilities':{}}
        self.assertEqual(english_answer(abstain),abstain)

    def test_nested_reports_preserve_metrics_and_raw_inputs(self):
        source=pd.DataFrame({'answer_type':['CHOICE','CHOICE_SENSOR'],'answer':['Sensor relationship','Yes'],
            'probabilities':[{'Sensor relationship':.7,'Stuck':.3},{'Yes':1.}],
            'answers':[{'BOOLEAN':'No','CHOICE':'Stuck','ORDINAL':'Very large','CHOICE_SENSOR':'Yes'},{}],
            'detection_recall':[.8,.2]})
        before=copy.deepcopy(source);result=english_frame(source)
        pd.testing.assert_frame_equal(source,before)
        np.testing.assert_array_equal(result['detection_recall'],source['detection_recall'])
        self.assertEqual(result['answer'].tolist(),['Sensor relationship','Yes'])
        self.assertEqual(result['answers'][0],{'BOOLEAN':'No','CHOICE':'Stuck','ORDINAL':'Very large','CHOICE_SENSOR':'Yes'})

    def test_report_boundary_english_without_relabeling_model_criteria(self):
        from pecvd.experiments.questions import CRITERIA,DISCOVERY_QUESTIONS
        from pecvd.experiments.reporting import ExperimentReport
        criteria=copy.deepcopy(CRITERIA);questions=copy.deepcopy(DISCOVERY_QUESTIONS)
        frame=pd.DataFrame({'answers':[{'BOOLEAN':'Yes','CHOICE':'変化 unknown','ORDINAL':'Small'}]})
        report=ExperimentReport('fixture',pd.DataFrame({'recall':[.5]}),details=frame,
            diagnostics={'center_analysis':frame})
        self.assertEqual(report.details['answers'][0]['BOOLEAN'],'Yes')
        self.assertEqual(report.diagnostics['center_analysis']['answers'][0]['ORDINAL'],'Small')
        self.assertEqual(CRITERIA,criteria);self.assertEqual(DISCOVERY_QUESTIONS,questions)

    def test_ui_decorator_calls_inference_once_and_leaves_inputs_unchanged(self):
        calls=[];question=SimpleNamespace(text='original trained query')
        @english_output
        def inference(q):
            calls.append(q.text)
            return [{'answer_type':'ORDINAL','answer':'Moderate','probabilities':{'Moderate':1.}}]
        result=inference(question)
        self.assertEqual(calls,['original trained query']);self.assertEqual(question.text,'original trained query')
        self.assertEqual(result[0]['answer'],'Moderate')

    def test_typed_probability_table_preserves_all_values(self):
        frame=pd.DataFrame({'answer_type':['BOOLEAN','BOOLEAN'],'option':['No','Yes'],
                            'probability':[.27,.73],'selected':[False,True]})
        @english_output
        def aggregate():return frame,2
        result,count=aggregate();self.assertEqual(count,2)
        self.assertEqual(result.option.tolist(),['No','Yes'])
        np.testing.assert_array_equal(result.probability,frame.probability)
        np.testing.assert_array_equal(result.selected,frame.selected)
