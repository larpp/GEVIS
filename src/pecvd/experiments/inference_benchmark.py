"""Actual frozen inference timing, including PEALD selected Step readouts."""
import time

import numpy as np
import pandas as pd
import torch

from . import fast_experiments as fast, fusion_answers as fa
from . import peald_training as peald


def module_for(runtime, step):
    if hasattr(runtime, 'step_runtimes'):
        return runtime.step_runtimes[step].module
    return runtime.module


def predict(runtime, case, center):
    if hasattr(runtime, 'step_runtimes'):
        from . import peald_encoder_routing as routing
        return routing.predict_arrays(runtime, case['current'], case['reference'],
                                      case['step'], center, case.get('position', 0.))
    if hasattr(runtime, 'prepared'):
        return peald.predict_arrays(runtime, case['current'], case['reference'],
                                    case['step'], center, case.get('position', 0.))
    return fa.predict_arrays(runtime.module, case['current'], case['reference'],
                             case['step'], center, case.get('position', 0.))


def synchronize(runtime, step):
    device = torch.device(module_for(runtime, step).DEVICE)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def predict_centers(runtime, case, centers):
    """Explicit paired diagnostic; shared learned computation with one encoding."""
    from .online_inference import predict_pair
    return predict_pair(runtime, case, centers)[0]


def benchmark(runtime, cases, *, repeats=20, warmup=3, domain='PEALD', session=None, progress=None):
    """The same full-window request used by EXP6/7/8, all selected sensors."""
    from .online_inference import InferenceSession, latency_tables
    cases = list(cases)
    if not cases or repeats < 1 or warmup < 0:
        raise ValueError('Prepared cases, positive repeats and nonnegative warmup required')
    runner = session or InferenceSession(runtime)
    if runner.runtime is not runtime:
        raise ValueError('Session belongs to a different runtime')
    groups = {}
    for case in cases:
        groups.setdefault(case['step'], []).append(case)
    def forward(case):
        return runner.process_window(current=case.get('raw_current', case['current']),
            step=case['step'], cohort=case['cohort'], run_id=case['run_id'],
            centers=case.get('centers'), position=case.get('position', 0.),
            standardized='raw_current' not in case)
    for group_index, (step, group) in enumerate(groups.items(), 1):
        for i in range(warmup):
            forward(group[i % len(group)])
        if progress is not None:
            progress(f'Warmup Step groups: {group_index}/{len(groups)}')
    rows = []
    for i in range(repeats):
        for step, group in groups.items():
            case = group[i % len(group)]; result = forward(case)
            rows.append(dict(repetition=i, step=step, request_id=f'{step}/{i}', pair_id=case['pair_id'],
                status=result['status'], reason=result['reason'],
                answers=len(result['answers']), selected_sensors=result.get('selected_sensors', 0),
                latency_ms=result['timings']['request_ms'], **result['timings']))
            if progress is not None and (len(rows)==1 or len(rows)%10==0 or len(rows)==repeats*len(groups)):
                progress(f'Measured requests: {len(rows)}/{repeats*len(groups)}')
    summary, by_step = latency_tables(rows, domain=domain)
    return fast.FastSuite(summary, details=pd.DataFrame(rows), diagnostics=dict(
        by_step=by_step, requests=pd.DataFrame(rows), device_memory=pd.DataFrame()),
        metadata=dict(domain=domain, inference_recipe=runner.__class__.__module__,
            repeats_per_step=repeats, warmup_per_step=warmup, deployed_weights_updated=False,
            percentile_scope='Few repeats per Step give coarse tail estimates; increase repeats for a fixed acceptance protocol',
            processing_unit='one complete native Step model window; all selected sensors; three answers per sensor',
            cycle_KPI_verified=False,
            timing_scope='complete model-window request -> normalization/reference -> encoder/Fusion/selected head -> answers; excludes disk/preprocessing segmentation, startup and transport',
            limitation='All selected sensor answers are measured together. No physical-cycle/Edge KPI certification. Prepared Test windows are reconstructed in frozen TRAIN units before timing.'))


def peald_cases(runtime, *, steps=None, config=peald.PealdConfig()):
    """Native Test pairs for timing only; no artificial change or fitting."""
    selected = list(runtime.step_runtimes) if hasattr(runtime, 'step_runtimes') else fast._steps(runtime, steps)
    if steps is not None:
        if not steps or any(s not in selected for s in steps):
            raise ValueError('Supported PEALD timing Steps required')
        selected = list(steps)
    result = []
    for step in selected:
        scoped = runtime.step_runtimes[step] if hasattr(runtime, 'step_runtimes') else runtime
        meta = scoped.encoders[step][1]
        for w in peald._select(scoped.prepared.windows[step]['test'], config.test_windows_per_step, config.seed):
            result.append(dict(step=step, current=w['current'], reference=w['current'].copy(),
                centers=list(meta['sensor_columns']), position=w['position'], pair_id=w['pair_id'],
                cohort=w['cohort'], run_id=w['run_id'],
                raw_current=(w['current']*meta['std']+meta['mean']).astype(np.float32)))
    return result


def pecvd_cases(runtime, *, steps=None, config=fa.FusionConfig()):
    """Prepared Test currents; reference is independently selected by Session."""
    result = []
    for case in fast.original_cases(runtime, steps=steps, config=config):
        meta = runtime.encoders[case['step']][1]
        # original_cases stores normalized relative position, so recover the
        # exact cache bin rather than interpreting it as an ALD time offset.
        position = int(round(case['position'] * 9))
        result.append(dict(case, cohort=(case['cohort'], position),
            raw_current=(case['current']*meta['std']+meta['mean']).astype(np.float32)))
    return result
