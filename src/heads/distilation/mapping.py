"""Fixed V7 mapping of original argmax decisions; never pool probabilities."""
from copy import deepcopy
import numpy as np
import pandas as pd
from pecvd.experiments import fusion_answers as fa, decision_training as d

GROUPS = (('수준·추세', (1, 3)), ('변동성', (2,)), ('국소 변화·지연', (4, 5)))
MAPPING_POLICY = 'original_five_argmax_to_level_trend_variability_local_delay_v7'


def remap(array):
    source = np.asarray(array)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isin(source[:, 1], [-1, 0, 1, 2, 3, 4, 5]).all():
        raise ValueError('Original-five Nx3 decisions required')
    result = np.array(source, copy=True)
    for target, (_, indices) in enumerate(GROUPS, 1):
        result[np.isin(source[:, 1], indices), 1] = target
    return result


def remap_reply(reply):
    reply = deepcopy(reply)
    lookup = {source: (i, name) for i, (name, indices) in enumerate(GROUPS, 1) for source in indices}
    for answer in reply.get('answers', []):
        if answer['answer_type'] != 'CHOICE' or not answer.get('evaluated', False):
            continue
        original = answer['prediction']
        if original not in lookup:
            raise ValueError('Original positive five-class answer required')
        target, name = lookup[original]
        answer.update(original_five_prediction=original, original_five_answer=answer.get('answer'),
            original_five_probabilities=answer.get('probabilities'), prediction=target, answer=name,
            probabilities=None, probability_scope='argmax_regrouping_no_group_probabilities')
        if 'detected_types' in answer:
            answer['original_five_detected_types'] = answer['detected_types']
            answer['detected_types'] = [name]
        if 'top1_type' in answer:
            answer['original_five_top1_type'] = answer['top1_type']
            answer['top1_type'] = name
    reply.update(grouping=MAPPING_POLICY, choice_classes=3)
    return reply


def mapping_table():
    return pd.DataFrame([dict(original_class=source, original_answer=fa.ANSWER_CRITERIA['CHOICE'][source],
        group_class=i, group_answer=name) for i, (name, indices) in enumerate(GROUPS, 1) for source in indices])


def metrics(truth, prediction, *, model, split):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or len(truth) == 0:
        raise ValueError('Nonempty aligned truth and prediction required')
    original = d._scores(truth, prediction)
    grouped = d._scores(remap(truth), remap(prediction))
    values = d.typed_f1(grouped)
    rows = []
    for i, kind in enumerate(fa.KINDS):
        rows.append(dict(model=model, split=split, answer_type=kind, f1=values[kind],
            questions=len(truth), known_targets=int((truth[:, i] >= 0).sum()),
            positive_targets=int((truth[:, i] > 0).sum()),
            original_five_f1=original['macro_f1']['CHOICE'] if kind == 'CHOICE' else np.nan,
            normal_fpr=grouped['normal_false_positive_rate'] if kind == 'BOOLEAN' else np.nan))
    support = dict(boolean_classes=sorted(set(map(int, truth[:, 0]))),
        source_choice_support={i: int((truth[:, 1] == i).sum()) for i in range(1, 6)},
        ordinal_support={i: int((truth[:, 2] == i).sum()) for i in range(1, 5)})
    return pd.DataFrame(rows), support
