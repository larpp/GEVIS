"""Direct GMM boundary comparison inside the frozen GEVIS runtime.

No supervised targets, downstream classifier or percentile gate. The component
with greater TRAIN reference departure is a change hypothesis, not fault truth.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from .position_rank_policy import context_key
from .window_policy import WindowConfig, OPERATIONS, active_operations, operation_traces

SCHEMA = 'gevis.direct_gmm.v1'


class GMMNotIdentifiable(ValueError):
    def __init__(self, diagnostics):
        super().__init__('No identifiable TRAIN GMM contexts; q99 retained, no GMM artifact saved')
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class GMMConfig:
    seed: int = 42
    n_init: int = 3
    max_iter: int = 200
    reg_covar: float = 1e-6


def _probability(model, raw):
    x = (np.log1p(raw) - np.asarray(model['mean'])) / np.asarray(model['scale'])
    means = np.asarray(model['centers']); variances = np.asarray(model['variances'])
    logp = np.log(model['weights']) - .5 * (
        np.log(2 * np.pi * variances).sum(-1) + ((x[..., None, :] - means)**2 / variances).sum(-1))
    scores = np.exp(logp - logp.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def fit_models(records, window_config, config=GMMConfig()):
    """One maximum per independent TRAIN run/context/sensor; no held-out values."""
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.exceptions import ConvergenceWarning
    except ModuleNotFoundError as error:
        raise ImportError('Install in the notebook kernel: python -m pip install -r config/requirements-gmm.txt') from error
    window_config.validate()
    if window_config.gate_scope != 'position_rank_v4' or window_config.operation_profile != 'core':
        raise ValueError('Direct GMM comparison currently requires position_rank_v4/core')
    if config.n_init < 1 or config.max_iter < 1 or config.reg_covar <= 0:
        raise ValueError('Positive GMM fit parameters required')
    grouped = {}; sensors_by_step = {}; runs = set(); references = set()
    for row in records:
        if row['split'] != 'train':raise ValueError('GMM fits TRAIN only')
        run = str(row['run_id']); refs = set(map(str, row['reference_run_ids']))
        if not refs or run in refs:raise ValueError('Independent TRAIN reference required')
        runs.add(run); references.update(refs)
        step = row['step']; names = list(row['sensors'])
        if sensors_by_step.setdefault(step, names) != names:raise ValueError('Sensor order changed')
        trace, _ = operation_traces(row['current'], row['reference'], row.get('residual'), window_config)
        peak = trace[:, :, :5].max(0)
        key = (step, context_key(row['cohort'], row['position']), run)
        grouped[key] = np.maximum(grouped.get(key, peak), peak)
    models = {}; diagnostics = []
    for step, names in sensors_by_step.items():
        for context in sorted({k[1] for k in grouped if k[0] == step}):
            values = np.asarray([v for (s, c, _), v in grouped.items() if s == step and c == context])
            for sid, name in enumerate(names):
                entry = dict(step=step, context=context, sensor=name, train_runs=len(values))
                x = np.log1p(values[:, sid]); mean = x.mean(0); scale = x.std(0)
                scale = np.where(scale > 0, scale, 1.)
                if len(values) < window_config.min_train_runs:
                    diagnostics.append(dict(entry, status='UNSUPPORTED', reason='insufficient_train_runs'));continue
                if len(np.unique(x, axis=0)) < 2:
                    diagnostics.append(dict(entry, status='UNSUPPORTED', reason='constant_features'));continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    fit = GaussianMixture(2, covariance_type='diag', n_init=config.n_init,
                        max_iter=config.max_iter, reg_covar=config.reg_covar, random_state=config.seed).fit((x-mean)/scale)
                assigned = fit.predict((x-mean)/scale)
                if not fit.converged_ or len(np.unique(assigned)) != 2:
                    diagnostics.append(dict(entry, status='UNSUPPORTED', reason='unconverged_or_empty_component'));continue
                departure = np.array([x[assigned == i].mean() for i in range(2)])
                if np.isclose(*departure, rtol=1e-8, atol=1e-10):
                    diagnostics.append(dict(entry, status='UNSUPPORTED', reason='ambiguous_component_orientation'));continue
                model = dict(mean=mean.tolist(), scale=scale.tolist(), centers=fit.means_.tolist(),
                    variances=fit.covariances_.tolist(), weights=fit.weights_.tolist(),
                    change_component=int(departure.argmax()), train_runs=len(values))
                models.setdefault(step, {}).setdefault(context, {})[name] = model
                diagnostics.append(dict(entry, status='FITTED', reason=None))
    return dict(schema=SCHEMA, config=asdict(config), window_config=asdict(window_config),
        sensors=sensors_by_step, models=models, diagnostics=diagnostics,
        train_runs=sorted(runs), reference_runs=sorted(references), test_used_for_fitting=False,
        decision_rule='two_component_posterior_argmax', fault_probability_calibrated=False)


class GMMPolicy:
    decision_kind = 'direct_gmm'

    def __init__(self, baseline, bundle):
        if bundle.get('schema') != SCHEMA or bundle['window_config'] != asdict(baseline.config):
            raise ValueError('GMM schema or feature configuration mismatch')
        self.config = baseline.config
        self.bundle = bundle
        # Diagnostics may inspect the shared context metadata; never mutate q99.
        self.payload = deepcopy(baseline.payload)
        self.payload['threshold_source'] = 'TRAIN direct two-component GMM argmax; no percentile gate'
        for step, group in self.payload['groups'].items():
            for key, context in group.get('contexts', {}).items():
                fitted = bundle['models'].get(step, {}).get(key, {})
                context.update(supported=bool(fitted), validation_runs=0,
                    thresholds=None, ordinal_edges=None,
                    support_failure=None if fitted else 'missing_or_unidentifiable_train_gmm',
                    control_budget_resolved=False)
            group.update(window_threshold=None, validation_runs=0,
                insufficient_controls_for_nominal_rate=False, control_rate_resolution=None)

    def analyze(self, current, reference, residual, step, sensors, *, selected=None,
                scopes=None, center_sensor=None, cohort=None, position=None):
        sensors = list(sensors); selected = sensors if selected is None else list(selected)
        if sensors != self.bundle['sensors'].get(step):raise ValueError('GMM sensor/Step mismatch')
        if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(sensors):
            raise ValueError('Known unique selected sensors required')
        if center_sensor is not None and selected != [center_sensor]:raise ValueError('Select exactly the UI center')
        if scopes is not None and list(scopes) != ['whole']:
            raise ValueError('GMM is fitted on complete windows; only whole scope is supported')
        key = context_key(cohort, position)
        bank = self.bundle['models'].get(step, {}).get(key, {})
        trace, aux = operation_traces(current, reference, residual, self.config)
        n = len(trace); rows = []; supported = all(sensor in bank for sensor in selected)
        for sensor in selected:
            sid = sensors.index(sensor); raw = trace[:, sid, :5].max(0); model = bank.get(sensor)
            probability = _probability(model, raw) if model else None
            score = float(probability[model['change_component']]) if model else 0.
            change = bool(supported and probability[model['change_component']] > probability[1-model['change_component']])
            # A joint GMM does not learn change type or event boundaries. This
            # explanatory operation is the largest standardized departure only.
            strength = ((np.log1p(raw)-model['mean'])/model['scale']) if model else raw
            dominant = int(strength.argmax())
            for op, name in enumerate(active_operations(self.config)):
                peak = int(trace[:, sid, op].argmax())
                rows.append(dict(sensor=sensor, time_scope='whole', scope_rows=[0,n], statistic=name,
                    statistic_index=op, normalized_score=score, window_threshold=.5,
                    raw_score=float(raw[op]), normalization_scale=None, unrelaxed_threshold=.5,
                    threshold_multiplier=1., active_gate='direct GMM argmax', context_key=key,
                    changed=change and op == dominant, detected_rows=None, related_sensors=[],
                    related_sensor_scores={}, estimated_lag_rows=None, ordinal_index=1 if change else 0,
                    pointwise=dict(peak_row=peak, scope_rows=[0,n], exceedance_intervals=[],
                        threshold_normalized=.5, local_width=aux['local_width'],
                        coordinate_system='prepared window row; half-open; not seconds'),
                    localization_basis='GMM window decision provides no event boundary',
                    score_meaning='mixture component responsibility; not equipment fault probability'))
        rows.sort(key=lambda r:-r['normalized_score']); events = [r for r in rows if r['changed']]
        return dict(ranked_evidence=rows, events=events, winner=events[0] if events else None,
            accepted=supported, changed=bool(events) if supported else None,
            maximum_score=rows[0]['normalized_score'], disabled_operations=list(OPERATIONS[5:]),
            change_decision_source='TRAIN direct GMM posterior argmax; no q99 veto',
            calibration=dict(context_key=key, supported=supported,
                train_runs=min((bank[s]['train_runs'] for s in selected if s in bank), default=0),
                validation_runs=0, insufficient_controls_for_nominal_rate=False,
                unsupported_reason=None if supported else 'missing_or_unidentifiable_train_gmm'))


def _identity(baseline, config):
    return dict(schema=SCHEMA, baseline_identity=baseline.payload['identity'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), config=asdict(config))


def _path(module, identity):
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()[:20]
    return Path(module.BACKBONE_PATH).parent / f'direct_gmm_{signature}.json'


def load_gmm(module, config=GMMConfig()):
    identity = _identity(module.DISCOVERY_POLICY, config); path = _path(module, identity)
    if not path.is_file():raise FileNotFoundError('Run STEP 3 direct GMM preparation first; no inference fitting')
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('identity') != identity:raise ValueError('GMM artifact identity mismatch')
    return GMMPolicy(module.DISCOVERY_POLICY, bundle)


def compare_pair(module, current, reference, step, reference_info, **kwargs):
    """STEP 3.1 comparison using the same actual public inference function."""
    from .discovery import review_changes
    original = module.DISCOVERY_POLICY
    gmm = load_gmm(module)
    try:
        q99 = review_changes(module,current,reference,step,reference_info,**kwargs)
        module.DISCOVERY_POLICY = gmm
        direct = review_changes(module,current,reference,step,reference_info,**kwargs)
    finally:module.DISCOVERY_POLICY = original
    return {'q99':q99, 'gmm_direct':direct}


@torch.no_grad()
def prepare_gmm(runtime, config=GMMConfig()):
    """Explicit offline preparation; reads only TRAIN arrays, never Test values."""
    from .discovery_calibration import cached_comparisons, residual_trace, identity_for
    from importlib.metadata import version
    m = runtime.module; baseline = m.DISCOVERY_POLICY
    identity = _identity(baseline, config); path = _path(m, identity)
    if baseline.payload['identity'] != identity_for(m.ENCODER_ROOT, m.BACKBONE_PATH, baseline.config):
        raise ValueError('Baseline identity changed; prepare exact STEP 3 criteria first')
    if path.is_file():return load_gmm(m, config)
    records = []
    for step, (encoder, meta) in runtime.encoders.items():
        print(f'Direct GMM: frozen encoder TRAIN preparation / {step}', flush=True)
        encoder.to(m.DEVICE).eval()
        try:
            for row in cached_comparisons(m.ENCODER_ROOT, step, split='train', config=baseline.config, calibration=True):
                a = torch.as_tensor(row['current'][None], dtype=torch.float32, device=m.DEVICE)
                b = torch.as_tensor(row['reference'][None], dtype=torch.float32, device=m.DEVICE)
                state = m.numeric_state(encoder,a,b)
                records.append(dict(row, step=step, sensors=list(meta['sensor_columns']),
                    residual=residual_trace(state,len(meta['sensor_columns']),len(row['current']))))
        finally:encoder.cpu()
    bundle = fit_models(records, baseline.config, config)
    bundle.update(identity=identity, fit_packages={'scikit-learn':version('scikit-learn'), 'numpy':np.__version__})
    if not any(row['status'] == 'FITTED' for row in bundle['diagnostics']):
        raise GMMNotIdentifiable(bundle['diagnostics'])
    if identity != _identity(baseline, config) or baseline.payload['identity'] != identity_for(m.ENCODER_ROOT,m.BACKBONE_PATH,baseline.config):
        raise ValueError('GMM source or inputs changed during preparation')
    text = json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(text)
    return GMMPolicy(baseline,bundle)


def run_boundary_comparison(root, *, runtime=None, gmm_policy=None, baseline_report=None, **kwargs):
    """Exactly two decisions; identical held-out cases and frozen GEVIS modules."""
    from .gevis_system import ready_gevis_runtime, run_gevis_experiment
    from .discovery_reporting import summarize_discovery
    from .reporting import ExperimentReport
    runtime = ready_gevis_runtime(root,runtime,prepare=False)
    m = runtime.module; baseline = m.DISCOVERY_POLICY
    if getattr(baseline,'decision_kind',None) == 'direct_gmm':raise ValueError('Original q99 policy required')
    if not np.isclose(baseline.config.nominal_window_alert_rate,.01):
        raise ValueError('This two-boundary comparison requires the unchanged q99 baseline')
    gmm = load_gmm(m) if gmm_policy is None else gmm_policy
    if asdict(gmm.config) != asdict(baseline.config):raise ValueError('Feature settings differ')
    reports = {}
    try:
        reports['q99'] = baseline_report if baseline_report is not None else run_gevis_experiment(root,runtime=runtime,**kwargs)
        m.DISCOVERY_POLICY = gmm
        reports['gmm_direct'] = run_gevis_experiment(root,runtime=runtime,**kwargs)
        reports['gmm_direct'].metadata.update(decision_path='direct_gmm',
            score_meaning='mixture responsibility, not calibrated fault probability',
            type_meaning='largest standardized operation departure, not learned type',
            ordinal_meaning='binary change candidate, not calibrated severity')
    finally:m.DISCOVERY_POLICY = baseline
    a = reports['q99'].details; b = reports['gmm_direct'].details
    keys = [k for k in ('step','pair_id','task','center_sensor','kind','strength','injected','effective','expected_sensor') if k in a]
    if not keys or not a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True)):
        raise ValueError('Boundary comparison case/truth schedules differ')
    common = a.accepted.eq(True).to_numpy() & b.accepted.eq(True).to_numpy()
    summary = []; by_step = []; details = []; common_rows = []; by_kind = []
    for name, report in reports.items():
        summary.append(report.summary.assign(boundary=name, nominal_quantile=(1-baseline.config.nominal_window_alert_rate) if name=='q99' else np.nan))
        by_step.append(report.by_step.assign(boundary=name))
        details.append(report.details.assign(boundary=name))
        if 'by_kind' in report.diagnostics:
            by_kind.append(report.diagnostics['by_kind'].assign(boundary=name))
        own = report.details.loc[common]
        if 'task' in own:own = own[own.task.eq('self')]
        if len(own):common_rows.append(summarize_discovery(own,()).assign(boundary=name))
    summary = pd.concat(summary,ignore_index=True)
    summary = summary[['boundary',*[c for c in summary if c != 'boundary']]]
    return ExperimentReport('GEVIS q99 versus direct GMM',summary,
        pd.concat(by_step,ignore_index=True),pd.concat(details,ignore_index=True),
        diagnostics={'common_support':pd.concat(common_rows,ignore_index=True) if common_rows else pd.DataFrame(),
            'by_kind':pd.concat(by_kind,ignore_index=True) if by_kind else pd.DataFrame(),
            'gmm_fit_support':pd.DataFrame(gmm.bundle['diagnostics'])},
        metadata=dict(execution_contracts={name:r.metadata.get('execution_contract') for name,r in reports.items()},
            paired_cases_verified=True,split=reports['q99'].metadata.get('split'),
            gmm_training='TRAIN only; no Validation cutoff',baseline_preserved=True,
            fault_probability_calibrated=False,test_used_for_fitting=False))
