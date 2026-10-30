"""Canonical English text and score-preserving output helpers.

Questions and options are English BEFORE encoding. No late translation of
model inputs or legacy answer labels occurs here. Sensor identifiers stay intact.
"""
from functools import wraps

ANSWER_CRITERIA={
    'BOOLEAN': ('No', 'Yes'),
    'CHOICE': ('No clear change', 'Mean level', 'Variability', 'Trend slope', 'Local change', 'Reconstruction error'),
    'ORDINAL': ('No change', 'Small', 'Moderate', 'Large', 'Very large'),
}
# Separate from the preserved STEP2/legacy vocabulary. New Fusion heads bind
# these exact options in their source/config identity; old weights cannot reuse
# class 5 under a different meaning.
FUSION_ANSWER_CRITERIA = dict(ANSWER_CRITERIA,
    CHOICE=ANSWER_CRITERIA['CHOICE'][:5] + ('Delay',))
QUESTIONS={
    'BOOLEAN':'Is there a change in this interval?',
    'CHOICE':'What kind of change is present?',
    'ORDINAL':'What is the degree of change?',
    'CHOICE_SENSOR':'Which sensor shows the most prominent change?',
}
QUESTION_LABELS={text:text for text in QUESTIONS.values()}


def label(value):
    return value


def english_answer(answer):
    if not isinstance(answer,dict) or 'answer_type' not in answer:return answer
    result=dict(answer);kind=answer['answer_type']
    if 'question' in result:result['question']=QUESTION_LABELS.get(result['question'],result['question'])
    if kind!='CHOICE_SENSOR':
        for key in ('answer','candidate_answer'):
            if key in result:result[key]=label(result[key])
        if isinstance(result.get('probabilities'),dict):
            result['probabilities']={label(k):v for k,v in result['probabilities'].items()}
        if 'criteria' in result:
            result['criteria']=type(result['criteria'])(label(v) for v in result['criteria'])
    # Sensor names, evidence objects, numeric scores and abstentions are intact.
    return result


def english_result(value):
    if isinstance(value,list):return [english_result(v) for v in value]
    if isinstance(value,tuple):return tuple(english_result(v) for v in value)
    if isinstance(value,dict):
        if 'answer_type' in value:return english_answer(value)
        if 'answers' in value:
            result=dict(value);result['answers']=english_result(value['answers']);return result
        if any(k in value for k in ('BOOLEAN','CHOICE','ORDINAL','CHOICE_SENSOR')):
            return {k:label(v) if k in ('BOOLEAN','CHOICE','ORDINAL') else v for k,v in value.items()}
    return value


def english_frame(frame):
    """Copy only display columns; metrics, indices and raw trace fields stay put."""
    import pandas as pd
    if not isinstance(frame,pd.DataFrame):return frame
    columns=set(frame.columns)&{'answers','answer','candidate_answer','expected_answer','predicted_answer','question','option','probabilities'}
    if not columns:return frame
    result=frame.copy()
    for column in columns:
        if column=='answers':result[column]=result[column].map(english_result);continue
        if column=='probabilities':
            result[column]=[({label(k):v for k,v in value.items()} if isinstance(value,dict) and kind!='CHOICE_SENSOR' else value)
                for value,kind in zip(result[column],result.get('answer_type',[None]*len(result)))];continue
        if column=='question':
            result[column]=result[column].map(lambda text:QUESTION_LABELS.get(text,text) if isinstance(text,str) else text);continue
        result[column]=[label(value) if kind!='CHOICE_SENSOR' else value
            for value,kind in zip(result[column],result.get('answer_type',[None]*len(result)))]
    return result


def english_output(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        value=function(*args,**kwargs)
        if isinstance(value,tuple):return tuple(english_frame(v) for v in value)
        return english_result(value)
    return wrapped
