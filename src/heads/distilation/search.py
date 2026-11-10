"""Optional Optuna search: fixed architecture/Head/mapping, Validation only."""
from dataclasses import replace
import math
import time

import pandas as pd
import torch


def choose_trial(records, target):
    """Fastest p95 among quality-qualified trials; otherwise highest weakest F1.

    A fallback is explicitly not a target achievement. Test never enters this
    decision, and architecture, Head, mapping and sample seed remain fixed.
    """
    frame = pd.DataFrame(records)
    if frame.empty or not frame[['min_f1', 'p95_ms']].map(math.isfinite).all().all():
        raise ValueError('Complete finite Validation quality/timing trials required')
    qualified = frame.loc[frame.coverage_complete & frame.min_f1.ge(target)]
    ranked = (qualified.sort_values(['p95_ms', 'min_f1', 'trial'], ascending=[True, False, True]) if len(qualified)
              else frame.sort_values(['min_f1', 'p95_ms', 'trial'], ascending=[False, True, True]))
    return int(ranked.iloc[0].trial), bool(len(qualified))


def optimize(experiment, *, n_trials=12, timeout=None):
    """Each trial fits Student only; reuse exact Teacher rows and Run partitions."""
    experiment._open_training()
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError('Positive trial budget required')
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Positive finite search timeout required')
    try:
        import optuna
    except ImportError:
        raise ImportError('Optional search needs optuna. Install this project with [fdc,notebook,distilation].') from None
    base = experiment.config
    sampler = optuna.samplers.TPESampler(seed=base.seed, n_startup_trials=4)
    study = optuna.create_study(directions=['maximize', 'minimize'], sampler=sampler)
    study.enqueue_trial(dict(learning_rate=base.learning_rate, temperature=base.temperature,
                             alpha=base.alpha, feature_weight=base.feature_weight))
    folder = experiment.folder / 'trials'
    folder.mkdir(parents=True, exist_ok=False)
    records, saved = [], {}
    def objective(trial):
        config = replace(base, learning_rate=trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
            temperature=trial.suggest_categorical('temperature', [1., 2., 4.]),
            alpha=trial.suggest_float('alpha', .1, .9),
            feature_weight=trial.suggest_float('feature_weight', .1, 3., log=True))
        reuse = (trial.number == 0 and experiment.student is not None and experiment.training is not None
                 and experiment.training['config'] == config.__dict__)
        if not reuse:
            experiment.fit(config)
        report = experiment.report('validation')
        experiment.parity_report()
        timing = experiment.benchmark('validation')['summary']
        f1 = report['metrics'].loc[lambda t: t.model.eq('control48'), 'f1']
        support = report['support']['control48']
        complete = (set(support['boolean_classes']) == {0, 1}
            and all(support['source_choice_support'].values()) and all(support['ordinal_support'].values()))
        value = timing.loc[timing.model.eq('control48')].iloc[0]
        row = dict(trial=trial.number, min_f1=float(f1.min()), mean_ms=float(value.mean_ms),
            p95_ms=float(value.p95_ms), coverage_complete=bool(complete), **trial.params)
        records.append(row)
        path = folder / f'{trial.number:05d}.pt'
        torch.save(dict(state={k: v.detach().cpu().clone() for k, v in experiment.student.state_dict().items()},
                        training=experiment.training), path)
        saved[trial.number] = path
        trial.set_user_attr('validation', row)
        return row['min_f1'], row['p95_ms']
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    selected, qualified = choose_trial(records, base.quality_target)
    snapshot = torch.load(saved[selected], map_location='cpu', weights_only=False)
    experiment.student.load_state_dict(snapshot['state'], strict=True)
    experiment.training = snapshot['training']
    experiment.training['optuna'] = dict(selected_trial=selected, validation_qualified=qualified,
        objectives=['maximize weakest Validation task F1', 'minimize whole-request Validation p95_ms'],
        selection='minimum p95 among target-qualified trials; otherwise maximum weakest F1, explicitly unqualified',
        trials=records)
    # Recheck the selected model without refitting or opening Test.
    validation = experiment.report('validation')
    experiment.parity_report()
    timing = experiment.benchmark('validation')
    experiment.assert_preserved()
    return dict(study=study, trials=pd.DataFrame(records), selected_trial=selected,
                validation_qualified=qualified, validation=validation, timing=timing,
                scope='Validation-only search; source data seed, 48x2 architecture and original Head remain fixed')
