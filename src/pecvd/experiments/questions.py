"""Sensor-free question semantics; questions alone never supply sensor truth."""
from dataclasses import dataclass
from itertools import product
from ..answer_display import ANSWER_CRITERIA, QUESTIONS

STATISTICS = ('level', 'variability', 'slope', 'local_change', 'residual')
DIRECTIONS = ('increase', 'decrease', 'either')
SCOPES = ('whole', 'early', 'late')
CRITERIA = dict(ANSWER_CRITERIA)
WORDS = dict(zip(STATISTICS, ('mean level', 'variability', 'trend slope', 'local deviation', 'reconstruction residual')))
SCOPE_WORDS = dict(zip(SCOPES, ('full interval', 'early interval', 'late interval')))
# Sensor identities are deliberately absent. Localization truth belongs to the
# paired numeric example, not to this question generator.
TEMPLATES = {
    'train': ('For {step}, {body} over the {scope}; inspect all measurements.',
              'Inspect every signal in {step}: {body}, restricted to the {scope}.',
              'Use the {scope} of phase {step}. Compare all measurements and {body}.',
              'Regarding {step}, search all signals and {body} in the {scope}.'),
    'validation': ('During the {scope} of {step}, inspect all signals and {body}.',
                   'Review every measurement in phase {step} within the {scope}; {body}.'),
    'test': ('Process phase {step}, {scope}: inspect all signals and {body}.',
             'With the reference as baseline, {body}; search all measurements in {step}, {scope}.'),
}

# Automatic discovery asks about the selected process range, never a target
# sensor. The runtime binds one winning event to the three evidence answers;
# the separate sensor CHOICE is a relative ranking, not an additional event.
DISCOVERY_QUESTIONS = {kind:QUESTIONS[kind] for kind in CRITERIA}

# A second CHOICE task, not a fourth neural answer family. Keep its task ID
# separate from CHOICE (change type) so consumers cannot overwrite either answer.
SENSOR_QUESTION = QUESTIONS['CHOICE_SENSOR']
PRIMARY_GOAL_VERSION = 'gevis.fixed_questions.stage1.english.v2'
MODEL_LOCALIZATION_QUESTION = 'Is there a change in this interval?'


def sensor_choice(sensor_names, weights, *, accepted, changed):
    """Relative attention ranking, never a fault probability or causal claim.

    Call with UI-selected sensors only. A no-change decision still permits a
    relative ranking; missing calibration withholds the answer entirely.
    """
    import numpy as np
    names = list(sensor_names)
    values = np.asarray(weights, dtype=float)
    if (not names or len(set(names)) != len(names) or values.shape != (len(names),)
            or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0):
        raise ValueError('Unique selected sensors and finite positive attention mass required')
    values = values / values.sum()
    answer = names[int(values.argmax())] if accepted else None
    return {'answer_type': 'CHOICE_SENSOR', 'typed_family': 'CHOICE',
        'question_id': 'CHOICE_SENSOR', 'question': SENSOR_QUESTION,
        'answer': answer, 'candidate_answer': answer, 'accepted': bool(accepted),
        'probabilities': dict(zip(names, map(float, values))) if accepted else {},
        'confidence': {'answer': float(values.max()) if accepted else None},
        'evidence': None, 'reference_only': changed is not True,
        'coverage_warnings': ['relative_ranking_not_anomaly_or_cause'],
        'score_meaning': 'normalized attention selection scores, not calibrated correctness/fault probabilities'}


@dataclass(frozen=True)
class QuestionCase:
    case_id: str
    text: str
    answer_type: str
    sensor: str | None
    statistic: str
    direction: str
    time_scope: str
    family: str
    supported: bool = True


def make_question(step, sensor, kind='BOOLEAN', statistic='level', direction='either',
                  scope='whole', *, split='test', variant=0):
    if kind not in CRITERIA or statistic not in STATISTICS or direction not in DIRECTIONS or scope not in SCOPES:
        raise ValueError('Unsupported typed operation; comparisons/ranking require a new execution contract')
    templates = TEMPLATES[split]
    if kind == 'BOOLEAN':
        action = {'increase': 'evaluate the increase', 'decrease': 'evaluate the decrease', 'either': 'evaluate the change'}[direction]
        body = f'{action} in {WORDS[statistic]}'
    elif kind == 'CHOICE':
        body = 'which numeric feature differs most from the reference?'
    else:
        body = 'what review level matches the observed difference?'
    return templates[variant % len(templates)].format(step=step, body=body, scope=SCOPE_WORDS[scope])


def question_bank(step, sensors=(), *, split='test', limit=120):
    """Unique sensor-free semantic questions, with NO localization labels.

    ``sensors`` is retained for call compatibility but does not affect questions,
    IDs or targets. ``limit`` is an upper bound, not a request for duplicates.
    CHOICE/ORDINAL do not specify statistic/direction; their placeholder slots
    must not be scored as semantic truth. Numeric examples provide their labels.
    """
    if limit < 1:
        raise ValueError('Positive limit required')
    result = []
    combinations = list(product(STATISTICS, DIRECTIONS, SCOPES))
    for statistic, direction, scope in combinations:
        for kind in CRITERIA:
            if kind != 'BOOLEAN' and (statistic != 'level' or direction != 'increase'):
                continue
            family = f'{step}/{kind}/{statistic}/{direction}/{scope}'
            for variant in range(len(TEMPLATES[split])):
                result.append(QuestionCase(f'{family}/{variant}', make_question(step, None, kind, statistic,
                    direction, scope, split=split, variant=variant), kind, None, statistic, direction, scope, family))
                if len(result) >= limit:
                    return result
    return result


def legacy_named_question_bank(step, sensors, *, split='test', limit=120):
    """Explicit named-query audits ONLY; never generic discovery supervision.

    Historical evaluators need an actual sensor instruction to score named
    routing. Do not attach cycling labels to identical sensor-free inputs.
    """
    from dataclasses import replace
    if not sensors or limit < 1:
        raise ValueError('Positive limit and configured sensors required')
    result = []
    for index, case in enumerate(question_bank(step, split=split, limit=limit)):
        sensor = sensors[(index // len(TEMPLATES[split])) % len(sensors)]
        text=case.text
        for wording in ('all measurements','every signal','all signals','every measurement'):
            text=text.replace(wording,'the specified signal')
        result.append(replace(case, sensor=sensor,
            text=f'Inspect only sensor {sensor}. {text}',
            case_id=f'{sensor}/{case.case_id}', family=f'{sensor}/{case.family}'))
    return result


def localization_questions(phase, count):
    """Shared TRAIN/runtime localization query; no target IDs or answer labels."""
    if count < 0:
        raise ValueError('Nonnegative question count required')
    import os
    if os.environ.get('GEVIS_DISCOVERY_TRAINING') in ('fixed_questions_v13','center_relation_v14'):
        # E5 is English: this is the fixed internal counterpart of the Korean
        # BOOLEAN display, not an LLM translation or a named-sensor instruction.
        return [MODEL_LOCALIZATION_QUESTION] * count
    return [f'Process phase {phase}: locate the sensor and time interval with a local deviation from the reference.'
            for _ in range(count)]


def center_descriptions(descriptions, sensor_names, patch_count, center_sensor):
    """UI anchor in numeric-token context, never in the fixed question text."""
    if center_sensor not in sensor_names or len(descriptions)!=len(sensor_names)*patch_count:
        raise ValueError('Known center and complete numeric token descriptions required')
    sid=list(sensor_names).index(center_sensor)
    return [text+('; UI analysis center.' if i//patch_count==sid else '; context sensor.')
            for i,text in enumerate(descriptions)]


def unsupported_bank(step, sensor):
    return [QuestionCase(f'unsupported/{i}', text, 'BOOLEAN', sensor, 'level', 'either', 'whole',
                         f'unsupported/{i}', False) for i, text in enumerate((
        f'Does the signal review for {step} prove the wafer is defective?',
        f'What is the exact remaining lifetime of the equipment during {step}?',
        f'Which maintenance action will certainly fix {step}?',
    ))]


def runtime_questions(step, sensor, question_type, *, bank=False, limit=120):
    """Return sensor-free questions; legacy ``sensor`` argument is ignored."""
    if bank:
        return [question_type(c.text, c.answer_type, CRITERIA[c.answer_type])
                for c in question_bank(step, limit=limit)]
    return discovery_questions(question_type)


def discovery_questions(question_type):
    """Return the sensor-free default questions for automatic GEVIS review."""
    return [question_type(DISCOVERY_QUESTIONS[kind], kind, CRITERIA[kind])
            for kind in ('BOOLEAN', 'CHOICE', 'ORDINAL')]
