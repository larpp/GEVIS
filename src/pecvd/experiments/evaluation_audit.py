"""Read-only evidence diagnostics; never fit, select, relabel or export inputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def split_audit(runtime):
    """Check all recorded preprocessing partitions and head-fit identities.

    Disjoint partitions cannot establish that Test results never influenced a
    human development decision. Missing provenance remains unknown.
    """
    if hasattr(runtime, 'step_runtimes'):
        return pd.concat([split_audit(r).assign(route_step=s)
                          for s, r in runtime.step_runtimes.items()], ignore_index=True)
    training = getattr(runtime.module, 'FUSION_ANSWER_TRAINING', None) or {}
    if hasattr(runtime, 'prepared'):
        partitions = [('*', {k: set(map(str, v)) for k, v in runtime.prepared.split_runs.items()})]
    else:
        manifest = Path(runtime.module.ENCODER_ROOT)/'encoder_manifest.json'
        doc = json.loads(manifest.read_text(encoding='utf-8'))
        partitions = []
        for step in runtime.encoders:
            with np.load(manifest.parent/'windows'/f'{step}.npz', allow_pickle=False) as cache:
                partitions.append((step, {s: set(cache[f'{s}_process_id'].astype(str))
                                          for s in ('train', 'validation', 'test')}))
        # Include the complete declared split, not just sampled head windows.
        if 'run_split' in doc:
            partitions.append(('manifest', {s: set(map(str, doc['run_split'][s]))
                                           for s in ('train', 'validation', 'test')}))
    rows = []
    # Runs can span multiple Steps. Check union as well as each Step cache.
    union = {s: set().union(*(p.get(s, set()) for _, p in partitions))
             for s in ('train', 'validation', 'test')}
    for name, splits in partitions + [('all_steps', union)]:
        train, validation, test = (splits.get(s, set()) for s in ('train', 'validation', 'test'))
        head_train = set(map(str, training.get('train_runs', [])))
        head_val = set(map(str, training.get('validation_runs', [])))
        overlaps = len(train & validation) + len(train & test) + len(validation & test)
        rows.append(dict(scope=name, train_runs=len(train), validation_runs=len(validation),
            test_runs=len(test), partition_overlap_count=overlaps,
            head_fit_test_overlap=len(head_train & test), head_selection_test_overlap=len(head_val & test),
            head_provenance_available=bool(head_train and head_val),
            final_untouched_holdout_certified=False,
            interpretation='Recorded split/head audit only; prior Test-guided development is not excluded'))
    return pd.DataFrame(rows)


def run_bootstrap(details, *, repeats=500, seed=42):
    """Percentile intervals resampling complete runs, not question rows.

    Pooled question-weighted metrics; overlapping windows stay with their run.
    These intervals describe the inspected benchmark, not an unseen holdout.
    """
    if repeats < 100:
        raise ValueError('At least 100 cluster resamples required')
    required = {'run_id', 'answer_type', 'label', 'prediction'}
    if not required <= set(details):
        return pd.DataFrame([dict(status='unavailable', reason='run identities and scored labels required')])
    from pecvd.answer_display import FUSION_ANSWER_CRITERIA
    rows = []
    rng = np.random.default_rng(seed)
    for kind, group in details.groupby('answer_type', sort=False):
        n = len(FUSION_ANSWER_CRITERIA[kind]); matrices = []
        for _, cluster in group.groupby('run_id', sort=True):
            matrix = np.zeros((n, n), dtype=np.int64)
            raw = cluster[['label', 'prediction']].to_numpy(dtype=float)
            if (not np.isfinite(raw).all() or (raw != np.floor(raw)).any()
                    or (raw < 0).any() or (raw >= n).any()):
                raise ValueError('Finite supported class indices required')
            np.add.at(matrix, tuple(raw.astype(int).T), 1)
            matrices.append(matrix)
        matrices = np.asarray(matrices); count = len(matrices)
        def metrics(matrix):
            total = matrix.sum(); diag = matrix.diagonal(); support = matrix.sum(1)
            predicted = matrix.sum(0); active = support+predicted > 0
            changed = support[1:].sum()
            values = dict(answer_accuracy=float(diag.sum()/total),
                macro_f1=float(np.mean(2*diag[active]/(support+predicted)[active])),
                changed_accuracy=float(diag[1:].sum()/changed) if changed else np.nan)
            if kind == 'BOOLEAN':
                tn, fp, fn, tp = matrix.ravel()
                values.update(detection_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                              synthetic_false_positive_rate=fp/(fp+tn) if fp+tn else np.nan)
            return values
        point = metrics(matrices.sum(0))
        samples = [metrics(matrices[rng.integers(0, count, count)].sum(0))
                   for _ in range(repeats)] if count > 1 else []
        for metric, estimate in point.items():
            values = np.array([s[metric] for s in samples]); values = values[np.isfinite(values)]
            rows.append(dict(answer_type=kind, metric=metric, estimate=estimate, source_runs=count,
                questions=len(group), confidence_level=.95, resamples=repeats,
                valid_resamples=len(values), lower=float(np.quantile(values, .025)) if len(values) else np.nan,
                upper=float(np.quantile(values, .975)) if len(values) else np.nan,
                status='available' if len(values) else 'unavailable',
                caution='run-cluster benchmark uncertainty; few runs give unstable intervals; not fault performance'))
    return pd.DataFrame(rows)


def original_alarm_rates(details):
    """Unlabelled Yes rates; no unverified normal/FPR assumption."""
    boolean = details.loc[details.answer_type.eq('BOOLEAN')]
    rows = []
    for step, group in boolean.groupby('step', sort=False):
        rows.append(dict(step=step, questions=len(group), alarms=int(group.prediction.ne(0).sum()),
            alarm_rate=float(group.prediction.ne(0).mean()),
            verified_normal=False, fault_false_positive_rate_available=False))
    return pd.DataFrame(rows)
