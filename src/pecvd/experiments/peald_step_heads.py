"""Step-specific target readouts; shared frozen encoders/Fusion are unchanged."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from . import fusion_answers as fa, peald_readout as readout

RECIPE = 'peald_validation_selected_step_readouts_v1'


@torch.no_grad()
def score(head, examples, device):
    x, y = readout.grouped(examples, 'validation', device)
    head = head.to(device).eval()
    return readout.metrics([head(x[:, i], kind) for i, kind in enumerate(fa.KINDS)], y)


def prefer_step(shared, candidate, goal):
    # A specialist must improve interpretation as well as meet detection policy.
    interpretation = lambda s: (s['choice_macro_f1'] + s['ordinal_macro_f1']) / 2
    return (interpretation(candidate) > interpretation(shared) and
            candidate['boolean_f1'] >= min(goal, shared['boolean_f1']) and
            readout.selection_key(candidate, 0., goal) > readout.selection_key(shared, 0., goal))


def fit_steps(common, train, validation, config, device):
    heads = {}; selections = []
    local_config = replace(config, epochs=config.step_epochs, max_seconds=config.step_max_seconds,
                           patience=config.step_patience, learning_rate=config.step_learning_rate)
    for step in sorted({row['step'] for row in train}):
        local_train = [row for row in train if row['step'] == step]
        local_validation = [row for row in validation if row['step'] == step]
        runs = {row['run_id'] for row in local_train}
        held_runs = {row['run_id'] for row in local_validation}
        row = dict(step=step, selected_head='shared', train_runs=len(runs), validation_runs=len(held_runs))
        if not config.step_heads_enabled:
            selections.append(dict(row, reason='step_heads_disabled')); continue
        if len(runs) < config.min_step_train_runs or len(held_runs) < config.min_step_validation_runs:
            selections.append(dict(row, reason='insufficient_independent_run_support')); continue
        # Do not invent missing class evidence. Shared model remains available.
        coverage = {kind: {r['label'] for r in local_train if r['answer_type'] == kind} for kind in fa.KINDS}
        required = {kind: set(torch.nonzero(getattr(common, 'allowed_' + kind), as_tuple=True)[0].cpu().tolist())
                    for kind in fa.KINDS}
        if any(len(coverage[k]) < 2 or coverage[k] != required[k] for k in fa.KINDS) or any(
                r['label'] not in coverage[r['answer_type']] for r in local_validation):
            selections.append(dict(row, reason='insufficient_step_class_coverage')); continue
        shared = score(common, local_validation, device)
        candidate, training = readout.fit_readout(local_train, local_validation, local_config,
                                                 device, initial_head=common)
        candidate_score = score(candidate, local_validation, device)
        selected = prefer_step(shared, candidate_score, config.boolean_f1_goal)
        if selected:
            heads[step] = candidate
        selections.append(dict(row, selected_head='step' if selected else 'shared',
            reason='validation_interpretation_improved' if selected else 'shared_validation_preferred',
            shared_validation=shared, candidate_validation=candidate_score,
            optimizer_updates=training['optimizer_updates'], selected_epoch=training['selected_validation']['epoch']))
    return heads, selections


def dump(heads):
    return {step: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            for step, head in heads.items()}


def restore(states, selections, dimension, width, selected_steps):
    expected = {row['step'] for row in selections if row['selected_head'] == 'step'}
    if set(states) != expected or not expected <= set(selected_steps):
        raise ValueError('Selected PEALD Step head state missing or unexpected; no fallback')
    if len(selections) != len(selected_steps) or {r['step'] for r in selections} != set(selected_steps):
        raise ValueError('PEALD Step head selection coverage mismatch')
    if any(r['selected_head'] not in ('shared', 'step') for r in selections):
        raise ValueError('Unknown PEALD head selection')
    heads = {}
    for step, state in states.items():
        head = fa.ContextReadout(dimension, width)
        head.load_state_dict(state, strict=True)
        heads[step] = head.requires_grad_(False).eval()
    return heads


def for_step(runtime, step):
    if step not in runtime.encoders:
        raise ValueError('Unknown PEALD Step')
    common = fa.require_head(runtime.module)
    heads = getattr(runtime, 'step_heads', {})
    selection = (getattr(runtime.module, 'FUSION_ANSWER_TRAINING', None) or {}).get('step_head_selection', [])
    if any(row['step'] == step and row['selected_head'] == 'step' for row in selection) and step not in heads:
        raise RuntimeError('Selected PEALD Step head missing at inference; no fallback')
    module = SimpleNamespace(**runtime.module.__dict__)
    module.FUSION_ANSWER_HEAD = heads.get(step, common)
    result = SimpleNamespace(**runtime.__dict__)
    result.module = module
    return result
