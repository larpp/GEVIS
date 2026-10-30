"""Explicit reconstruction statistics: evidence, never a change-type oracle."""
from contextlib import contextmanager
import pandas as pd


def indices(width):
    # numeric_state contracts: 4x96 latent blocks, optional temporal/relational
    # blocks, then 8 historical statistics ending in current/reference patch MSE.
    if width not in (392,408,440):
        raise ValueError('Unsupported reconstruction numeric contract')
    return [width-2,width-1] + ([398,399] if width in (408,440) else [])


def evidence(state, names, tasks):
    try: indices(state.shape[-1])
    except ValueError:
        return [dict(reconstruction_available=False) for _ in tasks]
    grid=state[0].reshape(len(names),-1,state.shape[-1]).detach().cpu()
    result=[]
    for center,kind in tasks:
        mse=grid[names.index(center),:,-2:].mean(0)
        result.append(dict(reconstruction_available=True,
            current_reconstruction_mse=float(mse[0]),reference_reconstruction_mse=float(mse[1]),
            reconstruction_mse_delta=float(mse[0]-mse[1])))
    return result


def summary(details):
    if 'reconstruction_available' not in details or not details.reconstruction_available.any():
        return pd.DataFrame([dict(status='UNAVAILABLE',reason='numeric contract has no verified MSE coordinates')])
    rows=details[details.reconstruction_available & details.answer_type.eq('BOOLEAN')]
    return rows.groupby(['kind','center_is_perturbed'],sort=False).agg(
        sensor_comparisons=('current_reconstruction_mse','size'),
        current_mse=('current_reconstruction_mse','mean'),
        reference_mse=('reference_reconstruction_mse','mean'),
        mse_delta=('reconstruction_mse_delta','mean')).reset_index()


@contextmanager
def without_explicit_reconstruction(module):
    """Ablate explicit MSE/reconstruction-difference coordinates before Fusion.

    Reconstruction-trained latent content is retained; this is not an ablation
    of reconstruction pretraining itself. Restores callable on every exit.
    """
    original=module.numeric_state
    def masked(encoder,current,reference):
        state=original(encoder,current,reference).clone()
        state[...,indices(state.shape[-1])]=0
        return state
    module.numeric_state=masked
    try: yield
    finally: module.numeric_state=original


def run_controls(runtime, *, config, steps=None):
    """Same waves/labels, readout size and caps; refit only on TRAIN/Validation."""
    from . import fusion_answers as fa
    from .fast_experiments import _score_features, _suite
    data={s:fa.controlled_examples(runtime.module,s,config,steps) for s in ('train','validation','test')}
    runs={s:{e['run_id'] for e in rows} for s,rows in data.items()}
    if any(runs[a]&runs[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('Reconstruction control split overlap')
    with without_explicit_reconstruction(runtime.module):
        masked={s:fa.controlled_examples(runtime.module,s,config,steps) for s in data}
    for s in data:
        key=lambda e:(e['pair_id'],e['center_sensor'],e['answer_type'],e['label'])
        if list(map(key,data[s]))!=list(map(key,masked[s])):
            raise ValueError('Reconstruction controls must score identical cases/targets')
    reports={};training=[]
    for name,rows in (('full_reconstruction',data),('without_explicit_reconstruction',masked)):
        head,fit=fa.fit_readout(rows['train'],rows['validation'],config,runtime.module.DEVICE)
        reports[name]=_score_features(head,rows['test'],runtime.module.DEVICE)
        training.append(dict(variant=name,validation_loss=fit['validation_loss'],
            optimizer_updates=fit['optimizer_updates'],fit_seconds=fit['fit_seconds']))
    suite=_suite(reports,experiment='explicit_reconstruction_statistics_ablation',
        limitation='Reconstruction-pretrained latents retained; only explicit statistics removed.',
        default_model_replaced=False)
    suite.diagnostics['training']=pd.DataFrame(training)
    return suite
