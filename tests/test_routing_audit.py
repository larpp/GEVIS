import unittest
import pandas as pd
from pecvd.experiments.routing_audit import (explicit_slots,acceptance_gates,gate_report,confidence_replay,
    ensure_gate_report,refresh_evaluation_modules,require_routing_runtime)


def gates(**overrides):
    args=dict(criterion_ready=True,criteria_match=True,support_confidence=.9,comparable=True,
        answer_confidence=.7,semantic_scores={'time_scope':.8,'statistic':.9,'direction':.9},
        sensor_confidence=.2,required_semantics=('time_scope','statistic','direction'),
        confidence_floor=.55,semantic_floor=.60,grounding_floor=.30,support_floor=.50,score_gate=False)
    return acceptance_gates(**{**args,**overrides})


class RoutingAuditTests(unittest.TestCase):
    def test_refresh_recovers_missing_nested_question_symbol(self):
        import pecvd.experiments.questions as questions
        import pecvd.experiments.pipeline_settings as settings
        import pecvd.experiments.discovery_reporting as reporting
        from pecvd.gevis_backbone import evidence_questions
        saved=questions.localization_questions
        del questions.localization_questions
        try:
            with self.assertRaises(ImportError): evidence_questions('fixture',2)
            refresh_evaluation_modules()
            self.assertEqual(evidence_questions('fixture',2),questions.localization_questions('fixture',2))
            self.assertTrue(callable(settings.activate_prepared_policy))
            import pecvd.experiments.discovery as discovery
            self.assertIs(reporting.review_changes,discovery.review_changes)
        finally:
            if not hasattr(questions,'localization_questions'):questions.localization_questions=saved

    def test_missing_aggregation_recovers_only_from_real_gate_records(self):
        from types import SimpleNamespace
        frame=pd.DataFrame([dict(step='phase',answer_type='BOOLEAN',supported=True,gate_checks=gates())])
        report=SimpleNamespace(details=frame,diagnostics={})
        result=ensure_gate_report(report)
        self.assertIs(result,report.diagnostics['gate_blockers'])
        self.assertIs(ensure_gate_report(report),result)
        for old in (frame.drop(columns=['gate_checks']),frame.assign(gate_checks=[{}])):
            with self.assertRaisesRegex(RuntimeError,'gate'):
                ensure_gate_report(SimpleNamespace(details=old,diagnostics={}))

    def test_stale_runtime_capability_fails_before_evaluation(self):
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError,r'STEP3\.1'):
            require_routing_runtime(SimpleNamespace(module=SimpleNamespace(ask_gevis=lambda:None)))

    def test_reload_rebinds_reporting_to_current_evaluation_function(self):
        from unittest.mock import patch
        import pecvd.experiments.evaluation as evaluation
        import pecvd.experiments.reporting as reporting
        old=evaluation.evaluate_questions
        with patch.object(evaluation,'evaluate_questions',lambda *a,**kw:None):
            refresh_evaluation_modules()
            self.assertIs(reporting.evaluate_questions,evaluation.evaluate_questions)
            self.assertIsNot(old,evaluation.evaluate_questions)

    def test_parser_exact_boundaries_unknown_and_ambiguous_names(self):
        sensors=['PREC1_MFC_Flow','PREC2_MFC_Flow']
        self.assertEqual(explicit_slots('Sensor PREC1_MFC_Flow, late interval',sensors)['sensor'],sensors[0])
        self.assertIsNone(explicit_slots('Sensor PREC1_MFC_Flow_extra',sensors)['sensor'])
        result=explicit_slots('Compare PREC1_MFC_Flow and PREC2_MFC_Flow',sensors)
        self.assertIsNone(result['sensor']);self.assertEqual(result['sensor_status'],'ambiguous')
        self.assertIsNone(explicit_slots('Unknown pressure',sensors)['sensor'])
        self.assertIsNone(explicit_slots('a reference value',['a','b'])['sensor'])
        self.assertEqual(explicit_slots('Sensor a, full interval',['a','b'])['sensor'],'a')

    def test_scope_conflict_and_duplicate_alias_never_choose_first(self):
        self.assertIsNone(explicit_slots('early interval and late interval',['p'])['time_scope'])
        result=explicit_slots('pressure, full interval',['p1','p2'],{'p1':'pressure','p2':'pressure'})
        self.assertIsNone(result['sensor']);self.assertEqual(result['time_scope'],'whole')
        self.assertEqual(explicit_slots('sensor p1, late interval',['p1'])['time_scope'],'late')

    def test_explicit_slots_exempt_only_binding_and_time_gates(self):
        check=gates(explicit_sensor=True,explicit_scope=True,semantic_scores={'time_scope':.1,'statistic':.1,'direction':.9})
        self.assertTrue(check['sensor_binding']['passed']);self.assertTrue(check['semantic_time_scope']['passed'])
        self.assertFalse(check['semantic_statistic']['passed'])
        check=gates(explicit_sensor=True,explicit_scope=True,support_confidence=.1,criterion_ready=False)
        self.assertFalse(check['question_support']['passed']);self.assertFalse(check['reference_criterion']['passed'])

    def test_overlapping_and_sole_blockers_have_explicit_denominators(self):
        frame=pd.DataFrame([
            dict(step='phase',answer_type='BOOLEAN',supported=True,gate_checks=gates()),
            dict(step='phase',answer_type='BOOLEAN',supported=True,gate_checks=gates(support_confidence=.1)),
        ])
        result=gate_report(frame).set_index('reason')
        self.assertEqual(result.loc['sensor_binding','blocked_questions'],2)
        self.assertEqual(result.loc['sensor_binding','sole_blocker_questions'],1)
        self.assertEqual(result.loc['question_support','sole_blocker_questions'],0)

    def test_validation_replay_never_selects_test_or_mutates_gates(self):
        import copy
        frame=pd.DataFrame([dict(supported=True,answer_type='BOOLEAN',answer_correct=True,
            joint_query_correct=False,gate_checks=gates())])
        before=copy.deepcopy(frame.gate_checks.iloc[0])
        settings=[dict(semantic_floor=.6,grounding_floor=.1,support_floor=.5)]
        with self.assertRaises(ValueError): confidence_replay(frame,split='test',settings=settings)
        report=confidence_replay(frame,split='validation',settings=settings)
        self.assertEqual(report.coverage.iloc[0],1.)
        self.assertEqual(report.accepted_wrong_query_share.iloc[0],1.)
        self.assertEqual(report.grounded_accepted_accuracy.iloc[0],0.)
        self.assertFalse(report.parameters_applied.any());self.assertEqual(frame.gate_checks.iloc[0],before)

    def test_default_replay_matches_runtime_gate_acceptance(self):
        checks=[gates(),gates(explicit_sensor=True),gates(explicit_sensor=True,support_confidence=.1),
                gates(explicit_sensor=True,criterion_ready=False)]
        frame=pd.DataFrame([dict(supported=True,answer_type='CHOICE',answer_correct=True,
            joint_query_correct=True,gate_checks=check) for check in checks])
        result=confidence_replay(frame,split='validation',settings=[dict(semantic_floor=.6,grounding_floor=.3,support_floor=.5)])
        self.assertEqual(result.coverage.iloc[0],sum(all(g['passed'] for g in c.values()) for c in checks)/len(checks))


if __name__=='__main__': unittest.main()
