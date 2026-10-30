"""Text-only routing and gate diagnostics; no labels or data fitting."""
import re
import numpy as np
import pandas as pd

ROUTING_MODES=('learned','explicit_text')


def refresh_evaluation_modules():
    """Reload local dependencies BEFORE runtime creation, never fit or load data."""
    import importlib
    importlib.invalidate_caches()
    importlib.reload(importlib.import_module('pecvd.gevis_fast'))
    for name in ('temporal_patchtst','multiscale_patchtst','relational_features','temporal_bridge'):
        importlib.reload(importlib.import_module('pecvd.'+name))
    # Leaf contracts first: reloading only their consumers leaves old cached
    # modules without newly added symbols (e.g. localization_questions).
    for name in ('questions','pipeline_settings','reference_policy','reference_policy_v2',
                 'window_policy','position_rank_policy','discovery_calibration','execution_contract',
                 'evaluation','reporting','discovery','center_decision','center_training',
                 'discovery_reporting','threshold_comparison'):
        importlib.reload(importlib.import_module('pecvd.experiments.'+name))


def require_routing_runtime(runtime):
    import inspect
    if 'routing_mode' not in inspect.signature(runtime.module.ask_gevis).parameters:
        raise RuntimeError('STEP3.1 inference code is stale. Reopen the saved notebook, restart STEP4 and Run All.')


def ensure_gate_report(report):
    """Recover aggregation only; missing per-question evidence requires rerun."""
    details=report.details
    if 'gate_checks' not in details or 'supported' not in details or details.empty:
        raise RuntimeError('REPORT has no gate details. Restart STEP4 and Run All; STEP1/2/3 retraining is not required.')
    checks=details.loc[details.supported,'gate_checks']
    if checks.empty or not checks.map(lambda value:isinstance(value,dict) and bool(value)).all():
        raise RuntimeError('REPORT gate details are empty. Check saved STEP3.1, restart STEP4 and Run All.')
    for check in checks:
        if any(not isinstance(gate,dict) or 'passed' not in gate for gate in check.values()):
            raise RuntimeError('REPORT gate details have an incompatible format. Run STEP4 from the first cell.')
    if 'gate_blockers' not in report.diagnostics:
        report.diagnostics['gate_blockers']=gate_report(details)
    return report.diagnostics['gate_blockers']


def explicit_slots(text,sensors,descriptors=None):
    """Resolve only unique names and supported literal scope phrases.

    Ambiguity never chooses the first match. Single-letter identifiers require
    an explicit sensor/measurement/signal/target introducer (avoid article 'a').
    No question.sensor or evaluation target metadata is accepted here.
    """
    descriptors=descriptors or {}
    text=str(text)
    matches=set()
    for sensor in sensors:
        for alias in {sensor,descriptors.get(sensor,sensor)}:
            if not isinstance(alias,str) or not alias.strip(): continue
            pattern=r'(?<![\w])'+re.escape(alias)+r'(?![\w])'
            if len(alias)==1:
                pattern=r'\b(?:sensor|measurement|signal|target)\s+'+re.escape(alias)+r'(?![\w])'
            if re.search(pattern,text,re.I): matches.add(sensor)
    scope_patterns={
        'whole':r'\b(?:full|whole|entire)\s+(?:interval|window|segment)\b',
        'early':r'\b(?:early|initial)\s+(?:interval|window|segment)\b',
        'late':r'\b(?:late|final)\s+(?:interval|window|segment)\b',
    }
    scopes=[key for key,pattern in scope_patterns.items() if re.search(pattern,text,re.I)]
    return {'sensor':next(iter(matches)) if len(matches)==1 else None,
            'time_scope':scopes[0] if len(scopes)==1 else None,
            'sensor_status':'unique' if len(matches)==1 else 'ambiguous' if matches else 'not_explicit',
            'scope_status':'unique' if len(scopes)==1 else 'ambiguous' if scopes else 'not_explicit'}


def acceptance_gates(*,criterion_ready,criteria_match,support_confidence,comparable,
                     answer_confidence,semantic_scores,sensor_confidence,required_semantics,
                     confidence_floor,semantic_floor,grounding_floor,support_floor,
                     score_gate,explicit_sensor=False,explicit_scope=False):
    """Record every check, including thresholds and explicit binding exemptions."""
    gates={
        'reference_criterion':{'passed':bool(criterion_ready)},
        'criteria_match':{'passed':bool(criteria_match)},
        'question_support':{'passed':support_confidence>=support_floor,'value':float(support_confidence),'threshold':support_floor},
        'comparable_reference':{'passed':bool(comparable)},
        'answer_score':{'passed':not score_gate or answer_confidence>=confidence_floor,
                        'enabled':bool(score_gate),'value':float(answer_confidence),'threshold':confidence_floor},
        'sensor_binding':{'passed':explicit_sensor or sensor_confidence>=grounding_floor,
                          'explicit':bool(explicit_sensor),'value':float(sensor_confidence),'threshold':grounding_floor},
    }
    for slot in required_semantics:
        value=float(semantic_scores[slot])
        exempt=slot=='time_scope' and explicit_scope
        gates['semantic_'+slot]={'passed':exempt or value>=semantic_floor,'explicit':bool(exempt),
                                'value':value,'threshold':semantic_floor}
    return gates


def gate_report(details):
    """Overlapping and sole blockers: denominator is all supported questions."""
    rows=[]
    for keys,group in details[details.supported].groupby(['step','answer_type'],sort=False):
        checks=list(group.gate_checks)
        reasons=sorted({k for check in checks for k in check})
        for reason in reasons:
            failed=[{k for k,v in check.items() if not v['passed']} for check in checks]
            rows.append({'step':keys[0],'answer_type':keys[1],'reason':reason,'questions':len(group),
                         'blocked_questions':sum(reason in f for f in failed),
                         'sole_blocker_questions':sum(f=={reason} for f in failed),
                         'blocked_share':sum(reason in f for f in failed)/len(group)})
    return pd.DataFrame(rows)


def confidence_replay(details,*,split,settings):
    """Validation-only gate replay: never changes routing, quantiles or weights."""
    if split!='validation': raise ValueError('Confidence comparisons require validation, never test selection')
    rows=[]
    supported=details[details.supported]
    if supported.empty or not all(bool(g) for g in supported.gate_checks):
        raise ValueError('Actual runtime gate checks required for confidence replay')
    for setting in settings:
        floors=dict(setting)
        if set(floors)!={'semantic_floor','grounding_floor','support_floor'} or any(not 0<=x<=1 for x in floors.values()):
            raise ValueError('Explicit probability floors in [0,1] required')
        for kind,group in supported.groupby('answer_type',sort=False):
            accepted=[]
            for check in group.gate_checks:
                decisions=[]
                for name,gate in check.items():
                    floor=('semantic_floor' if name.startswith('semantic_') else
                           'grounding_floor' if name=='sensor_binding' else
                           'support_floor' if name=='question_support' else None)
                    decisions.append(gate['passed'] if floor is None or gate.get('explicit') else gate['value']>=floors[floor])
                accepted.append(all(decisions))
            mask=np.asarray(accepted,bool); correct=group.answer_correct.eq(True).to_numpy(); joint=group.joint_query_correct.to_numpy(bool)
            rows.append({'split':'validation','answer_type':kind,**floors,'questions':len(group),
                'coverage':float(mask.mean()),'accepted_proxy_accuracy':float(correct[mask].mean()) if mask.any() else np.nan,
                'accepted_wrong_query_share':float((~joint[mask]).mean()) if mask.any() else np.nan,
                'grounded_accepted_accuracy':float((mask & correct & joint).mean()),
                'parameters_applied':False,'meaning':'validation replay only; no test tuning or fault guarantee'})
    return pd.DataFrame(rows)
