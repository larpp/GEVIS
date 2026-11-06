"""Native, same-window PECVD encoder selection for PEALD; no Test selection.

The old peald_training module and its checkpoint identities stay unchanged.
Every candidate fits its own target head. A selected encoder is never paired
with a head trained on a different representation.
"""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json

import pandas as pd
import torch

from . import peald_training as p, peald_step_heads as heads, peald_readout as readout
from . import fusion_answers as fa, fast_experiments as fast
from .reporting import ExperimentReport

RECIPE = 'peald_native_validation_encoder_routes_v1'


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_steps(source, candidates=None):
    fast.require_runtime(source)
    names = list(source.encoders) if candidates is None else list(candidates)
    if not names or len(set(names)) != len(names) or any(s not in source.encoders for s in names):
        raise ValueError('Choose unique existing PECVD source encoders')
    return names


def preparation_patch_length(source, candidates=None):
    names = source_steps(source, candidates)
    sizes = [int(source.encoders[s][0].patch_len) for s in names
             if getattr(source.encoders[s][0], 'supports_native_sensor_transfer', False)]
    if not sizes or min(sizes) < 2:
        raise ValueError('At least one valid native-transfer encoder is required')
    return min(sizes)


def runtime_for(source, prepared, *, candidates=None, baseline='DEP.'):
    names = source_steps(source, candidates)
    if baseline not in names or not prepared.windows:
        raise ValueError('Baseline must be a candidate; supported PEALD windows required')
    return NS(source_runtime=source, prepared=prepared, candidates=names, baseline=baseline,
              step_runtimes={}, encoders={}, selection=[], candidate_results=[])


def candidate_for(runtime, step, source_step):
    encoder = runtime.source_runtime.encoders[source_step][0]
    length = int(runtime.prepared.metadata[step]['window_length'])
    patch = int(encoder.patch_len)
    if not getattr(encoder, 'supports_native_sensor_transfer', False):
        return None, 'native_sensor_transfer_unsupported'
    baseline_encoder = runtime.source_runtime.encoders[runtime.baseline][0]
    if getattr(encoder, 'd_model', None) != getattr(baseline_encoder, 'd_model', None):
        return None, 'frozen_Fusion_latent_width_mismatch'
    if length < 2 * patch or length % patch:
        return None, 'native_patch_incompatible_with_shared_window'
    subset = p.Prepared({step: runtime.prepared.windows[step]},
                        {step: runtime.prepared.metadata[step]}, runtime.prepared.tables,
                        runtime.prepared.split_runs,
                        dict(runtime.prepared.identity, encoder_routing_recipe=RECIPE,
                             source_encoder_state_sha256=state_digest(encoder),
                             routing_source_sha256=p.digest(__file__)))
    if hasattr(runtime.prepared, 'process'):
        subset.process = runtime.prepared.process
    # Target heads train afresh; source answer-head Step coverage is not a
    # requirement for a verified frozen source encoder. Do not borrow its head.
    source = runtime.source_runtime
    module = NS(**source.module.__dict__)
    original_meta = source.encoders[source_step][1]
    module.STEP_ENCODERS = {step:(encoder, dict(original_meta, **subset.metadata[step]))}
    module.FUSION_ANSWER_HEAD = None
    module.FUSION_ANSWER_TRAINING = None
    return NS(module=module, encoders=module.STEP_ENCODERS, prepared=subset,
              source_runtime=source, source_step=source_step), 'compatible'


def candidate_config(config):
    # Each candidate is already a Step head; do not fit a second specialist.
    return replace(config, epochs=config.step_epochs, patience=config.step_patience,
                   learning_rate=config.step_learning_rate, max_seconds=config.step_max_seconds,
                   step_heads_enabled=False)


def identity(runtime, steps, config):
    source = runtime.source_runtime
    return dict(recipe=RECIPE, prepared=runtime.prepared.identity, steps=steps,
        candidates=runtime.candidates, baseline=runtime.baseline, config=asdict(config),
        source_encoder_states={s:state_digest(source.encoders[s][0]) for s in runtime.candidates},
        backbone_state=state_digest(source.module.GEVIS_MODEL),
        source_head_sha256=p.digest(source.module.FUSION_ANSWER_PATH),
        source_backbone_sha256=p.digest(source.module.BACKBONE_PATH),
        source_files={Path(m.__file__).name:p.digest(m.__file__) for m in (p, fa, readout, heads)},
        routing_source_sha256=p.digest(__file__),
        window_lengths={s:int(runtime.prepared.metadata[s]['window_length']) for s in steps})


def choose(scores, baseline, goal):
    if not scores:
        raise ValueError('No supported trained encoder candidate')
    eligible = list(scores)
    if baseline in scores:
        eligible = [baseline] + [s for s in scores if s != baseline and
                                heads.prefer_step(scores[baseline], scores[s], goal)]
    return max(eligible, key=lambda s: readout.selection_key(scores[s], 0., goal))


def _support(runtime, step, config):
    pools = runtime.prepared.windows[step]
    if {w['run_id'] for w in pools['train']} & {w['run_id'] for w in pools['validation']}:
        raise ValueError('TRAIN/Validation source run overlap')
    if (len({w['run_id'] for w in pools['train']}) < config.min_step_train_runs or
            len({w['run_id'] for w in pools['validation']}) < config.min_step_validation_runs):
        return 'insufficient_independent_TRAIN_Validation_runs'
    # Targets only; no forward pass and no Test window access.
    train = p.cases(runtime, 'train', steps=[step], config=config)
    validation = p.cases(runtime, 'validation', steps=[step], config=config)
    for kind in fa.KINDS:
        observed = {v[kind] for c in train for v in c['targets'].values()}
        required = {v[kind] for c in validation for v in c['targets'].values()}
        if len(observed) < 2 or not required <= observed:
            return 'insufficient_TRAIN_class_coverage'
    return None


def train_or_load(root, runtime, *, steps=None, config=p.PealdConfig(), allow_training=True):
    config.validate()
    process = getattr(runtime.prepared, 'process', None)
    def verify_inputs():
        if process is not None and (
                p.digest(Path(root)/process.data_path) != runtime.prepared.identity['csv_sha256'] or
                p.digest(process.source_path) != runtime.prepared.identity['process_sha256']):
            raise ValueError('Prepared PEALD input or mapping changed; no route selection or reload')
    verify_inputs()
    selected = list(runtime.prepared.windows) if steps is None else list(steps)
    if not selected or len(selected) != len(set(selected)) or any(s not in runtime.prepared.windows for s in selected):
        raise ValueError('Choose unique prepared target Steps')
    if not runtime.prepared.identity:
        raise ValueError('Verified preparation identity required')
    expected = identity(runtime, selected, config)
    stamp = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(root)/'artifacts/gevis_peald_encoder_routes'/stamp/'selection.json'
    fit_config = candidate_config(config)
    if path.is_file():
        payload = json.loads(path.read_text(encoding='utf-8'))
        routes = set(payload.get('routes', {})); withheld = set(payload.get('withheld', {}))
        if (payload.get('identity') != expected or routes & withheld or
                routes | withheld != set(selected) or not routes):
            raise ValueError('Encoder route manifest identity or coverage mismatch; no fallback')
        runtimes = {}
        for step, row in payload['routes'].items():
            if row['source_step'] not in runtime.candidates:
                raise ValueError('Unknown selected source encoder')
            candidate, reason = candidate_for(runtime, step, row['source_step'])
            if candidate is None:
                raise ValueError('Selected encoder no longer supports native window: ' + reason)
            p.train_or_load(root, candidate, steps=[step], config=fit_config, allow_training=False)
            if p.digest(candidate.module.FUSION_ANSWER_PATH) != row['head_sha256']:
                raise ValueError('Selected encoder/head binding changed; no fallback')
            runtimes[step] = candidate
    else:
        if not allow_training:
            raise FileNotFoundError('Matching encoder route manifest missing; no selection or fitting')
        runtimes = {}; routes = {}; withheld = {}; results = []; histories = []; updates = 0
        for step in selected:
            scores = {}; fitted = {}
            for name in runtime.candidates:
                candidate, reason = candidate_for(runtime, step, name)
                row = dict(step=step, source_step=name,
                           window_length=int(runtime.prepared.metadata[step]['window_length']))
                if candidate is None:
                    results.append(dict(row, status='NOT_EVALUATED', reason=reason)); continue
                reason = _support(candidate, step, config)
                if reason:
                    results.append(dict(row, status='NOT_EVALUATED', reason=reason)); continue
                print('PEALD encoder candidate:', step, '/', name, flush=True)
                training = p.train_or_load(root, candidate, steps=[step], config=fit_config)
                examples = p.examples(candidate, 'validation', steps=[step], config=fit_config)
                scores[name] = heads.score(fa.require_head(candidate.module), examples, candidate.module.DEVICE)
                fitted[name] = candidate
                updates += training['optimizer_updates']
                histories.extend(dict(h, step=step, source_step=name) for h in training['history'])
                results.append(dict(row, status='EVALUATED', reason='TRAIN_fit_Validation_score',
                                    **scores[name]))
            if not scores:
                withheld[step] = 'no_eligible_encoder_TRAIN_Validation_support'
                continue
            winner = choose(scores, runtime.baseline, config.boolean_f1_goal)
            chosen = fitted[winner]; runtimes[step] = chosen
            routes[step] = dict(source_step=winner, validation=scores[winner],
                baseline_available=runtime.baseline in scores,
                baseline_validation=scores.get(runtime.baseline),
                head_sha256=p.digest(chosen.module.FUSION_ANSWER_PATH))
        if not runtimes:
            raise ValueError('No eligible trained target Steps; inspect native and TRAIN/Validation support')
        if identity(runtime, selected, config) != expected:
            raise ValueError('Source weights or identity changed during encoder selection')
        payload = dict(identity=expected, routes=routes, withheld=withheld, candidates=results, history=histories,
                       optimizer_updates=updates, selection_split='validation', test_used_for_selection=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    verify_inputs()
    if identity(runtime, selected, config) != expected:
        raise ValueError('Encoder route sources changed during operation')
    # Never expose partial routing if loading or selection fails.
    runtime.step_runtimes = runtimes
    runtime.encoders = {s:r.encoders[s] for s, r in runtimes.items()}
    runtime.selection = [dict(step=s, status='READY', **row) for s, row in payload['routes'].items()]
    runtime.selection.extend(dict(step=s, source_step=None, status='NOT_EVALUATED', reason=reason)
                             for s,reason in payload['withheld'].items())
    runtime.candidate_results = payload['candidates']
    runtime.manifest_path = path
    return dict(payload, selection_manifest=str(path), step_head_selection=runtime.selection)


def for_step(runtime, step):
    if step not in runtime.step_runtimes:
        raise RuntimeError('Selected target Step encoder/head missing; no fallback')
    return runtime.step_runtimes[step]


def evaluate(runtime, *, steps=None, config=p.PealdConfig()):
    selected = list(runtime.encoders) if steps is None else list(steps)
    reports = [p.evaluate(for_step(runtime, s), steps=[s], config=config) for s in selected]
    rows = pd.concat([r.details.assign(source_encoder_step=runtime.step_runtimes[s].source_step)
                      for s,r in zip(selected,reports)], ignore_index=True).to_dict('records')
    result = fast._report('PEALD Validation-selected encoder/head routes', rows, all_answers=rows)
    result.diagnostics['encoder_selection'] = pd.DataFrame(runtime.selection)
    result.diagnostics['encoder_candidates'] = pd.DataFrame(runtime.candidate_results)
    result.metadata.update(encoder_routing_recipe=RECIPE, selection_split='validation',
                           test_used_for_selection=False, source_weights_updated=False)
    return result


def original(runtime, *, steps=None, config=p.PealdConfig(), center_batch_size=8):
    selected = list(runtime.encoders) if steps is None else list(steps)
    reports = [p.original(for_step(runtime, s), steps=[s], config=config,
                          center_batch_size=center_batch_size) for s in selected]
    details = pd.concat([r.details.assign(source_encoder_step=runtime.step_runtimes[s].source_step)
                         for s,r in zip(selected,reports)], ignore_index=True)
    unmatched = pd.concat([r.diagnostics['unmatched_references'] for r in reports], ignore_index=True)
    return ExperimentReport('PEALD routed original unlabelled comparisons',
        pd.DataFrame([dict(questions=len(details), accuracy_available=False)]), details=details,
        diagnostics=dict(unmatched_references=unmatched,
                         answer_consistency=fast._consistency(details.to_dict('records'))),
        metadata=dict(encoder_routing_recipe=RECIPE, original_cross_run_fault_accuracy_available=False))


def predict_arrays(runtime, current, reference, step, center_sensor, position=0.):
    return p.predict_arrays(for_step(runtime, step), current, reference, step, center_sensor, position)
