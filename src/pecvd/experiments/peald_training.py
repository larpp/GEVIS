"""Local PEALD quality boundaries and a separately trained GEVIS readout.

Signal changes are review candidates, never automatic fault truth or deletion
criteria. Only structural input failures prevent model windows. No resampling,
clipping, interpolation, source-weight update or numerical detector fallback.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from . import fast_experiments as fast, fusion_answers as fa
from . import peald_readout, peald_step_heads
from .process_inputs import phase_key, step_description
from .reporting import ExperimentReport

SCHEMA = 'gevis.peald_quality_target_readout.v2'
REFERENCE_CONTRACT = 'peald.nominal_original_vs_deployed_TRAIN_reference.v1'


@dataclass(frozen=True)
class PealdConfig:
    seed: int = 42
    reference_mode: str = 'normal_train'
    original_runs_normal: bool = True
    feature_center_batch_size: int = 8
    train_windows_per_step: int = 8
    validation_windows_per_step: int = 4
    test_windows_per_step: int = 4
    max_window_length: int = 64
    cadence_relative_tolerance: float = .05
    cadence_mad_multiplier: float = 8.
    review_z: float = 12.
    epochs: int = 40
    batch_size: int = 64
    width: int = 96
    learning_rate: float = .001
    max_seconds: float = 300.
    patience: int = 8
    consistency_weight: float = .2
    ordinal_weight: float = .2
    boolean_f1_goal: float = .96
    step_heads_enabled: bool = True
    step_epochs: int = 16
    step_patience: int = 5
    step_max_seconds: float = 30.
    step_learning_rate: float = .0003
    min_step_train_runs: int = 3
    min_step_validation_runs: int = 2

    def validate(self):
        if self.reference_mode != 'normal_train' or self.original_runs_normal is not True:
            raise ValueError('This head requires the explicit nominal-original normal premise and normal_train references')
        if type(self.feature_center_batch_size) is not int or not 1 <= self.feature_center_batch_size <= 32:
            raise ValueError('Feature center batch size must be in 1..32')
        if min(self.train_windows_per_step, self.validation_windows_per_step,
               self.test_windows_per_step, self.max_window_length, self.epochs,
               self.batch_size, self.width, self.patience) < 1:
            raise ValueError('Positive PEALD preparation and learning budgets required')
        if not 0 < self.cadence_relative_tolerance < 1 or self.cadence_mad_multiplier <= 0 or self.review_z <= 0:
            raise ValueError('Positive structural cadence tolerances and review level required')
        if self.learning_rate <= 0 or self.max_seconds <= 0:
            raise ValueError('Positive PEALD learning rate and time budget required')
        if not all(np.isfinite(v) and v >= 0 for v in (self.consistency_weight, self.ordinal_weight)):
            raise ValueError('Finite nonnegative auxiliary loss weights required')
        if not np.isfinite(self.boolean_f1_goal) or not 0 < self.boolean_f1_goal <= 1:
            raise ValueError('Validation BOOLEAN F1 goal must be in (0, 1]')
        if min(self.step_epochs, self.step_patience, self.min_step_train_runs, self.min_step_validation_runs) < 1:
            raise ValueError('Positive Step head budgets and independent run support required')
        if not all(np.isfinite(v) and v > 0 for v in (self.step_max_seconds, self.step_learning_rate)):
            raise ValueError('Finite positive Step head learning budgets required')


@dataclass
class Prepared:
    windows: dict
    metadata: dict
    tables: dict
    split_runs: dict
    identity: dict = field(default_factory=dict)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def load_raw(root, config, *, chunk_size=20000):
    """Bounded parsing; retain file-row coordinates through later quarantine."""
    mapping = config.source_columns
    if not {'run','step','order'} <= set(mapping):
        raise ValueError('Explicit raw PEALD run/Step/time mapping required')
    chunks = []
    for chunk in pd.read_csv(Path(root)/config.data_path,
            usecols=list(dict.fromkeys([*mapping.values(), *config.sensor_columns])),
            dtype={name:'string' for name in mapping.values()},
            chunksize=chunk_size, low_memory=False):
        for name in config.sensor_columns:
            chunk[name] = pd.to_numeric(chunk[name], errors='coerce').astype('float32')
        chunks.append(chunk)
    if not chunks:
        raise ValueError('Empty PEALD input')
    raw = pd.concat(chunks, ignore_index=True)
    raw['_input_row'] = np.arange(len(raw))
    return raw


def _run_splits(ids, seed):
    values = np.array(sorted(ids), dtype=object)
    np.random.default_rng(seed).shuffle(values)
    a = int(.7*len(values)); b = a+int(.15*len(values))
    result = dict(train=set(values[:a]), validation=set(values[a:b]), test=set(values[b:]))
    if min(map(len, result.values())) < 2:
        raise ValueError('Need at least two independent jobs in each PEALD split')
    return result


def _select(pool, limit, seed):
    # Spread bounded selection over jobs before taking second windows per job.
    by_run = {}
    for window in pool:
        by_run.setdefault(window['run_id'], []).append(window)
    rng = np.random.default_rng(seed)
    jobs = list(by_run); rng.shuffle(jobs)
    for job in jobs:
        rng.shuffle(by_run[job])
    result = []
    while len(result) < limit:
        active = False
        for job in jobs:
            if by_run[job]:
                result.append(by_run[job].pop()); active = True
                if len(result) == limit:
                    break
        if not active:
            break
    return result


def prepare_frame(raw, process, *, patch_length, config=PealdConfig()):
    config.validate()
    if patch_length < 2 or config.max_window_length < 2*patch_length:
        raise ValueError('Native source requires at least two complete patches')
    m = process.source_columns; sensors = list(process.sensor_columns)
    required = [*m.values(), *sensors]
    if not sensors or len(sensors) != len(set(sensors)) or any(c not in raw for c in required):
        raise ValueError('Explicit unique sensor and metadata columns required')
    f = raw[required + (['_input_row'] if '_input_row' in raw else [])].copy()
    f['_input_row'] = raw['_input_row'].to_numpy() if '_input_row' in raw else np.arange(len(raw))
    f['_run'] = f[m['run']].astype('string').str.strip()
    raw_step = f[m['step']].astype('string').str.strip()
    stamp = pd.to_datetime(f[m['order']], format='mixed', errors='coerce')
    f['_time'] = stamp
    context = f[m['step_context']].fillna('') if m.get('step_context') else ['']*len(f)
    f['_phase'] = [phase_key(s, c) for s, c in zip(raw_step.fillna(''), context)]
    split_runs = _run_splits(f.loc[f['_run'].notna() & f['_run'].ne(''), '_run'].unique(), config.seed)
    split_map = {rid: split for split, runs in split_runs.items() for rid in runs}
    f['_split'] = f['_run'].map(split_map)
    reject = pd.Series(False, index=f.index); quarantine = []; locations = []
    def mark(mask, reason):
        nonlocal reject
        mask = mask.fillna(False)
        if mask.any():
            quarantine.append(dict(reason=reason, rows=int(mask.sum()),
                runs=int(f.loc[mask, '_run'].nunique())))
            selected = f.loc[mask, ['_input_row','_run']].copy()
            selected['_safe_run'] = selected['_run'].fillna('<missing>').map(_hash)
            boundary = selected['_input_row'].diff().ne(1) | selected['_safe_run'].ne(selected['_safe_run'].shift())
            for _, group in selected.groupby(boundary.cumsum(), sort=False):
                locations.append(dict(reason=reason, run_id=group['_safe_run'].iloc[0],
                    source_row_start=int(group['_input_row'].iloc[0]),
                    source_row_end=int(group['_input_row'].iloc[-1]), rows=len(group)))
        reject |= mask
    missing = f['_run'].isna() | f['_run'].eq('') | raw_step.isna() | raw_step.eq('')
    mark(missing, 'missing_run_or_step')
    mark(stamp.isna(), 'invalid_timestamp')
    if m.get('cohort'):
        cohort = f[m['cohort']].astype('string').str.strip()
        mark(cohort.isna() | cohort.eq(''), 'missing_recipe')
    else:
        cohort = pd.Series('explicit_single_recipe', index=f.index)
    f['_cohort'] = cohort.fillna('').map(_hash)
    # Reappearing jobs and recipe changes make run/reference identity ambiguous.
    boundaries = f['_run'].ne(f['_run'].shift()).fillna(True)
    blocks = f.loc[boundaries & f['_run'].notna(), '_run']
    ambiguous = set(blocks[blocks.duplicated(keep=False)])
    mark(f['_run'].isin(ambiguous), 'disjoint_job_identity')
    recipe_changes = f.groupby('_run')['_cohort'].nunique()
    mark(f['_run'].isin(recipe_changes[recipe_changes > 1].index), 'recipe_changes_inside_job')
    exact = f.duplicated(required, keep='first') & ~missing
    # Remove exact copies only. Their original row coordinates still split gaps.
    mark(exact, 'exact_duplicate_row')
    duplicate_time = f.loc[~exact & ~reject].duplicated(['_run','_time'], keep=False)
    conflict = pd.Series(False, index=f.index); conflict.loc[duplicate_time.index] = duplicate_time
    mark(conflict, 'conflicting_duplicate_timestamp')
    kept = f.loc[~reject]
    delta = kept.groupby('_run')['_time'].diff().dt.total_seconds()
    bad_runs = kept.loc[delta.le(0), '_run'].unique()
    mark(f['_run'].isin(bad_runs), 'nonmonotonic_job_time')
    numeric = f[sensors].apply(pd.to_numeric, errors='coerce').astype('float32')
    mark(pd.Series(~np.isfinite(numeric.to_numpy()).all(1), index=f.index), 'nonfinite_sensor_row')
    f[sensors] = numeric
    # Phase occurrence is based on raw transitions, not quality-induced pieces.
    transition = f['_run'].ne(f['_run'].shift()).fillna(True) | f['_phase'].ne(f['_phase'].shift()).fillna(True)
    f['_occurrence'] = transition.groupby([f['_run'], f['_phase']], dropna=False).cumsum().astype(int)-1
    origins = f.groupby(['_run','_phase','_occurrence'], dropna=False)['_input_row'].min().to_dict()
    spans = f.groupby(['_run','_phase','_occurrence'], dropna=False)['_input_row'].agg(lambda s: int(s.max()-s.min()+1)).to_dict()
    time_origins = f.groupby(['_run','_phase','_occurrence'], dropna=False)['_time'].first().to_dict()
    f = f.loc[~reject].copy()
    if f.empty:
        raise ValueError('No structurally valid PEALD rows')
    train = f.loc[f['_split'].eq('train'), sensors].to_numpy(float)
    if not len(train):
        raise ValueError('No structurally valid TRAIN rows')
    global_scale = np.std(train, axis=0)
    usable = np.isfinite(global_scale) & (global_scale > 1e-6)
    sensor_table = pd.DataFrame(dict(sensor=sensors,
        status=np.where(usable, 'SUPPORTED', 'NO_TRAIN_SCALE'),
        action=np.where(usable, 'retain_without_clipping', 'withhold_no_physical_scale')))
    names = [s for s, ok in zip(sensors, usable) if ok]
    global_scale = global_scale[usable]
    if not names:
        raise ValueError('No target sensor has a defensible TRAIN scale')
    same = f['_run'].eq(f['_run'].shift()) & f['_phase'].eq(f['_phase'].shift())
    adjacent = f['_input_row'].diff().eq(1)
    dt = f['_time'].diff().dt.total_seconds()
    cadence = {}; cadence_rows = []
    for phase in process.steps:
        values = dt[same & adjacent & f['_split'].eq('train') & f['_phase'].eq(phase.name) & dt.gt(0)].to_numpy()
        if len(values) < 4:
            continue
        median = float(np.median(values)); mad = float(np.median(np.abs(values-median)))
        tolerance = max(config.cadence_relative_tolerance*median, config.cadence_mad_multiplier*mad)
        cadence[phase.name] = (median, tolerance)
        cadence_rows.append(dict(step=phase.name, train_interval_seconds=median,
                                 tolerance_seconds=tolerance, fit_split='train'))
    expected = f['_phase'].map({s:v[0] for s,v in cadence.items()})
    tolerance = f['_phase'].map({s:v[1] for s,v in cadence.items()})
    irregular = same & adjacent & (dt-expected).abs().gt(tolerance)
    cut = ~same | ~adjacent | irregular
    f['_piece'] = cut.cumsum()
    boundary_table = pd.DataFrame([dict(reason='cadence_discontinuity', boundaries=int(irregular.sum())),
        dict(reason='removed_row_discontinuity', boundaries=int((same & ~adjacent).sum()))])
    windows = {}; metadata = {}; support = []; candidates = []
    edge_runs = {_hash(str(f['_run'].iloc[0])), _hash(str(f['_run'].iloc[-1]))}
    for phase in process.steps:
        rows = f.loc[f['_phase'].eq(phase.name)]
        pieces = [g for _, g in rows.groupby('_piece', sort=False)]
        lengths = [len(g) for g in pieces if g['_split'].iloc[0] == 'train']
        length = min(config.max_window_length, int(np.quantile(lengths, .25)))//patch_length*patch_length if lengths else 0
        if length < 2*patch_length or phase.name not in cadence:
            support.append(dict(step=phase.name, status='NOT_EVALUATED',
                reason='insufficient_contiguous_TRAIN_rows_or_cadence', window_length=length))
            continue
        pools = {key:[] for key in split_runs}; short = 0; tails = 0
        # Original row offsets remain anchored to the actual phase occurrence.
        for g in pieces:
            if len(g) < length:
                short += len(g); continue
            split = g['_split'].iloc[0]; rid = str(g['_run'].iloc[0]); occurrence = int(g['_occurrence'].iloc[0])
            origin = origins[(rid, phase.name, occurrence)]; tails += len(g)%length
            for offset in range(0, len(g)-length+1, length):
                w = g.iloc[offset:offset+length]
                start = int(w['_input_row'].iloc[0])-int(origin)
                elapsed = (w['_time'].iloc[0]-time_origins[(rid,phase.name,occurrence)]).total_seconds()
                native_tick = int(round(elapsed/cadence[phase.name][0]))
                pools[split].append(dict(current=w[names].to_numpy(dtype=np.float32),
                    run_id=_hash(rid), pair_id=_hash((phase.name,rid,occurrence,start)),
                    cohort=(g['_cohort'].iloc[0], occurrence, start, native_tick),
                    input_row_start=int(w['_input_row'].iloc[0]), input_row_end=int(w['_input_row'].iloc[-1]),
                    maximum_interval_seconds=float(w['_time'].diff().dt.total_seconds().max()),
                    position=float(start/max(1,spans[(rid,phase.name,occurrence)]-1)), split=split, step=phase.name))
        train_pool = pools['train']
        if not train_pool or not pools['validation']:
            support.append(dict(step=phase.name, status='NOT_EVALUATED',
                reason='no_complete_TRAIN_or_validation_windows', window_length=length,
                short_piece_rows=short, unused_tail_rows=tails,
                **{split+'_windows':len(pool) for split,pool in pools.items()}))
            continue
        values = np.concatenate([w['current'] for w in train_pool])
        mean = np.median(values, axis=0)
        q = np.quantile(values, [.25, .75], axis=0)
        robust_scale = (q[1]-q[0])/1.349
        scale = np.where(robust_scale > 1e-6, robust_scale, global_scale)
        for split, pool in pools.items():
            for w in pool:
                w['current'] = ((w['current']-mean)/scale).astype(np.float32)
                review = []
                if np.max(np.abs(w['current'])) > config.review_z:
                    review.append('large_train_relative_signal')
                if np.all(np.std(w['current'], axis=0) < 1e-7):
                    review.append('flat_measurement_window')
                if w['run_id'] in edge_runs:
                    review.append('dataset_edge_job_review')
                for reason in review:
                    candidates.append(dict(step=phase.name, run_id=w['run_id'], pair_id=w['pair_id'],
                        split=split, source_row_start=w['input_row_start'], source_row_end=w['input_row_end'],
                        reason=reason, action='retain_review_not_fault_truth'))
        metadata[phase.name] = dict(sensor_columns=names, window_length=length,
            mean=mean, std=scale, cadence_seconds=cadence[phase.name][0],
            step_descriptor=step_description(process, phase.name), sensor_descriptors={s:s for s in names})
        windows[phase.name] = pools
        support.append(dict(step=phase.name, status='READY', reason='split_safe_contiguous_native_windows',
            window_length=length, short_piece_rows=short, unused_tail_rows=tails,
            **{split+'_windows':len(pool) for split,pool in pools.items()}))
    tables = dict(quarantine=pd.DataFrame(quarantine), quarantine_locations=pd.DataFrame(locations), boundaries=boundary_table,
        sensors=sensor_table, cadence=pd.DataFrame(cadence_rows), support=pd.DataFrame(support),
        signal_candidates=pd.DataFrame(candidates),
        summary=pd.DataFrame([dict(input_rows=len(raw), quarantined_rows=int(reject.sum()),
            structurally_retained_rows=len(f), unconfigured_phase_rows=int((~f['_phase'].isin([p.name for p in process.steps])).sum()),
            supported_steps=len(windows), signal_outliers_removed=False)]))
    return Prepared(windows, metadata, tables,
        {split:sorted(_hash(r) for r in runs) for split,runs in split_runs.items()})


def prepare(root, process, *, patch_length, config=PealdConfig()):
    before = dict(csv_sha256=digest(Path(root)/process.data_path),
                  process_sha256=digest(process.source_path))
    raw = load_raw(root, process)
    prepared = prepare_frame(raw, process, patch_length=patch_length, config=config)
    after = dict(csv_sha256=digest(Path(root)/process.data_path),
                 process_sha256=digest(process.source_path))
    if before != after:
        raise ValueError('PEALD sources changed during preprocessing')
    prepared.identity = dict(before, policy=asdict(config), split_runs=prepared.split_runs,
        patch_length=patch_length, preprocessing_source_sha256=digest(__file__))
    prepared.process = process
    return prepared


def runtime_for(source_runtime, prepared, *, source_step):
    fast.require_runtime(source_runtime); fast._steps(source_runtime, [source_step])
    encoder, original_meta = source_runtime.encoders[source_step]
    if not getattr(encoder, 'supports_native_sensor_transfer', False):
        raise ValueError('Native unseen-sensor encoder required; no ID reassignment')
    if not prepared.windows:
        raise ValueError('No supported PEALD windows. Inspect quality tables before training')
    module = SimpleNamespace(**source_runtime.module.__dict__)
    module.STEP_ENCODERS = {step:(encoder, dict(original_meta, **meta)) for step,meta in prepared.metadata.items()}
    module.FUSION_ANSWER_HEAD = None
    module.FUSION_ANSWER_TRAINING = None
    return SimpleNamespace(module=module, encoders=module.STEP_ENCODERS,
        prepared=prepared, source_runtime=source_runtime, source_step=source_step)


def reference_session(runtime, *, steps=None, config=PealdConfig()):
    """Use the deployed reference builder before fitting, without needing a head."""
    from . import online_inference as online
    config.validate()
    selected = list(runtime.encoders) if steps is None else list(steps)
    if not selected or len(selected) != len(set(selected)) or any(s not in runtime.encoders for s in selected):
        raise ValueError('Choose unique supported PEALD Steps')
    partitions = runtime.prepared.split_runs
    if any(set(partitions[a]) & set(partitions[b]) for a, b in
           (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('PEALD Run partitions overlap')
    for step in selected:
        for split, pool in runtime.prepared.windows[step].items():
            if any(w['run_id'] not in partitions[split] or w.get('split', split) != split for w in pool):
                raise ValueError('Window belongs to another Run partition')
    scoped = SimpleNamespace(**runtime.__dict__)
    scoped.module = SimpleNamespace(**runtime.module.__dict__)
    scoped.module.FUSION_ANSWER_TRAINING = dict(trained_steps=selected)
    return online.InferenceSession(scoped, center_batch_size=config.feature_center_batch_size)


def reference_support(runtime, *, steps=None, config=PealdConfig()):
    """Report support before encoding; TEST never determines trained Steps."""
    selected = list(runtime.encoders) if steps is None else list(steps)
    session = reference_session(runtime, steps=selected, config=config)
    rows = []
    for step in selected:
        row = dict(step=step)
        for split in ('train', 'validation'):
            pool = runtime.prepared.windows[step][split]
            eligible = [w for w in pool if session._reference(step, w['cohort'], w['run_id'])[0] is not None]
            row[split+'_windows'] = len(pool)
            row[split+'_matched_windows'] = len(eligible)
            row[split+'_matched_runs'] = len({w['run_id'] for w in eligible})
        ready = row['train_matched_windows'] > 0 and row['validation_matched_windows'] > 0
        row.update(status='READY' if ready else 'UNSUPPORTED',
                   reason='' if ready else 'no_independent_TRAIN_reference_for_train_or_validation')
        rows.append(row)
    return pd.DataFrame(rows)


def cases(runtime, split, *, steps=None, config=PealdConfig()):
    """Nominal A/B No controls and injected A-prime/SAME-B positives.

    Normality of untouched originals is the user's dataset premise, not a
    measured fault label. All split currents use the deployed TRAIN-only bank.
    No TEST original or label participates in reference selection.
    """
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Known PEALD split required')
    selected = list(runtime.encoders) if steps is None else list(steps)
    session = reference_session(runtime, steps=selected, config=config)
    limits = dict(train=config.train_windows_per_step, validation=config.validation_windows_per_step,
                  test=config.test_windows_per_step)
    seed = config.seed + dict(train=0, validation=100003, test=200003)[split]
    result = []
    for step in selected:
        names = list(runtime.encoders[step][1]['sensor_columns'])
        # Find compatible windows BEFORE bounded selection; no self-reference fallback.
        eligible = [w for w in runtime.prepared.windows[step][split]
                    if session._reference(step, w['cohort'], w['run_id'])[0] is not None]
        for i, w in enumerate(_select(eligible, limits[split], seed)):
            base = w['current']
            reference, count = session._reference(step, w['cohort'], w['run_id'])
            reference = np.array(reference, dtype=np.float32, copy=True)
            sid = (i*17) % len(names)
            for v in fa.paired_variants(base, sid, i, seed):
                target = v.get('sensor', sid)
                # Every supported center learns normal cross-Run variation. The
                # unchanged neighbor on injected inputs also remains a No target.
                centers = names if v['kind'] == 'equal' else [names[target]] + (
                    [names[(target+1)%len(names)]] if len(names)>1 else [])
                truth = {}
                for center in centers:
                    changed = bool(v['mask'][:, names.index(center)].any())
                    truth[center] = dict(BOOLEAN=int(changed),
                        CHOICE=fa.TYPE_TARGET[v['kind']] if changed else 0,
                        ORDINAL=v['level'] if changed else 0)
                result.append(dict(step=step, current=v['observed'], reference=reference,
                    centers=centers, targets=truth, changed_mask=v['mask'], kind=v['kind'],
                    reference_mode=config.reference_mode, reference_count=count,
                    reference_contract=REFERENCE_CONTRACT, cohort=w['cohort'],
                    normality_premise='untouched_original_runs_are_nominal',
                    strength=v['strength'], lag_samples=v.get('lag_samples',0),
                    strength_unit='native_samples' if v['kind']=='delay' else 'target_TRAIN_scale',
                    direction=v['direction'], augmentation=v['augmentation'],
                    challenge='peald_nominal_train_reference', position=w['position'], run_id=w['run_id'],
                    pair_id=w['pair_id']+'/'+v['tag'], split=split))
    return result


def examples(runtime, split, *, steps=None, config=PealdConfig()):
    result = []; previous = None
    for case in cases(runtime, split, steps=steps, config=config):
        if case['step'] != previous:
            print(f'PEALD frozen Fusion features: {split} / {case["step"]}', flush=True)
            previous = case['step']
        # Match inference center batching, encode the numeric pair once per case.
        module = SimpleNamespace(**runtime.module.__dict__)
        original_numeric = module.numeric_state
        state = None
        def numeric_once(encoder, current, reference):
            nonlocal state
            if state is None:
                state = original_numeric(encoder, current, reference)
            return state
        module.numeric_state = numeric_once
        for offset in range(0, len(case['centers']), config.feature_center_batch_size):
            centers = case['centers'][offset:offset+config.feature_center_batch_size]
            features, tasks = fa.fusion_features(module, case['current'], case['reference'],
                case['step'], centers, case['position'])
            for x, (center, kind) in zip(features, tasks):
                result.append(dict(features=x, answer_type=kind, label=case['targets'][center][kind],
                    split=split, run_id=case['run_id'], step=case['step'], center_sensor=center,
                    pair_id=case['pair_id'], reference_mode=config.reference_mode))
    return result


def identity(runtime, *, steps=None, config=PealdConfig()):
    source = runtime.source_runtime.module
    selected = list(runtime.encoders) if steps is None else list(steps)
    from . import online_inference as online, injection
    support = reference_support(runtime, steps=selected, config=config)
    supported = support.loc[support.status.eq('READY'), 'step'].tolist()
    if not supported:
        raise ValueError('No TRAIN/Validation Steps have independent TRAIN references; inspect reference_support')
    return dict(schema=SCHEMA, prepared=runtime.prepared.identity,
        reference_contract=REFERENCE_CONTRACT, reference_builder_sha256=digest(online.__file__),
        reference_recipe=online.RECIPE, injection_source_sha256=digest(injection.__file__),
        reference_support=support.to_dict('records'),
        requested_steps=selected, source_encoder_step=runtime.source_step,
        selected_steps=supported, policy=asdict(config),
        source_backbone_sha256=digest(source.BACKBONE_PATH),
        verified_source_head_sha256=digest(source.FUSION_ANSWER_PATH),
        fusion_source_sha256=digest(fa.__file__), training_source_sha256=digest(__file__),
        readout_source_sha256=digest(peald_readout.__file__), readout_recipe=peald_readout.RECIPE,
        step_readout_source_sha256=digest(peald_step_heads.__file__), step_readout_recipe=peald_step_heads.RECIPE,
        questions={k:fa.QUESTIONS[k] for k in fa.KINDS},
        criteria={k:list(fa.ANSWER_CRITERIA[k]) for k in fa.KINDS})


def train_or_load(root, runtime, *, steps=None, config=PealdConfig(), allow_training=True):
    """Only the new target readout trains. Test is not even encoded here."""
    config.validate()
    if not runtime.prepared.identity:
        raise ValueError('Verified data/config/source identity required before checkpointing')
    process = getattr(runtime.prepared, 'process', None)
    if process is not None and (digest(Path(root)/process.data_path)!=runtime.prepared.identity['csv_sha256'] or
                               digest(process.source_path)!=runtime.prepared.identity['process_sha256']):
        raise ValueError('PEALD source changed since preprocessing; prepare again')
    expected = identity(runtime, steps=steps, config=config)
    stamp = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(root)/'artifacts/gevis_peald_fast'/stamp/'answer_head.pt'
    if path.is_file():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('identity') != expected or payload.get('schema') != SCHEMA or payload.get('training',{}).get('optimizer_updates',0)<1:
            raise ValueError('PEALD checkpoint identity/training mismatch; no fallback')
        head = fa.ContextReadout(payload['dimension'], config.width)
        head.load_state_dict(payload['state'], strict=True)
        training = payload['training']
        step_heads = peald_step_heads.restore(payload.get('step_states', {}),
            training.get('step_head_selection', []), payload['dimension'], config.width, expected['selected_steps'])
    else:
        if not allow_training:
            raise FileNotFoundError('Matching normal_train PEALD head missing. Run ALD_evaluate.ipynb with RUN_PEALD_TRAINING=True once. Existing exact-reference heads are preserved, not substituted.')
        train = examples(runtime, 'train', steps=expected['selected_steps'], config=config)
        validation = examples(runtime, 'validation', steps=expected['selected_steps'], config=config)
        head, training = peald_readout.fit_readout(train, validation, config, runtime.module.DEVICE)
        step_heads, selection = peald_step_heads.fit_steps(head, train, validation, config, runtime.module.DEVICE)
        training.update(step_head_selection=selection, step_readout_recipe=peald_step_heads.RECIPE)
        if identity(runtime, steps=steps, config=config) != expected:
            raise ValueError('PEALD/source identity changed during training; no checkpoint saved')
        # Recheck the actual CSV/config after the feature preparation/training span.
        process = getattr(runtime.prepared, 'process', None)
        if process is not None and (digest(Path(root)/process.data_path)!=runtime.prepared.identity['csv_sha256'] or
                                    digest(process.source_path)!=runtime.prepared.identity['process_sha256']):
            raise ValueError('PEALD source changed during training; no checkpoint saved')
        training.update(target_domain='PEALD', backbone_updated=False,
            reference_mode=config.reference_mode, reference_contract=REFERENCE_CONTRACT,
            original_runs_normal=config.original_runs_normal,
            reference_recipe=expected['reference_recipe'],
            reference_builder_sha256=expected['reference_builder_sha256'],
            reference_support=expected['reference_support'],
            normal_centers='all_supported_centers_on_each_sampled_original',
            target_contract='Nominal original A vs deployed TRAIN B is No; injected A-prime vs SAME B is center-labelled; not real fault truth')
        payload = dict(schema=SCHEMA, identity=expected, dimension=head.feature_mean.numel(),
            state={k:v.detach().cpu() for k,v in head.state_dict().items()}, training=training,
            step_states=peald_step_heads.dump(step_heads),
            preprocessing=runtime.prepared.metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            torch.save(payload, stream)
    if (training.get('reference_contract') != REFERENCE_CONTRACT or
            training.get('reference_mode') != config.reference_mode or training.get('original_runs_normal') is not True):
        raise ValueError('Head was not trained with the deployed normal_train contract; no legacy fallback')
    runtime.module.FUSION_ANSWER_HEAD = head.requires_grad_(False).eval().to(runtime.module.DEVICE)
    runtime.step_heads = {step: model.to(runtime.module.DEVICE) for step, model in step_heads.items()}
    runtime.module.FUSION_ANSWER_TRAINING = training
    runtime.module.FUSION_ANSWER_PATH = path
    return training


def evaluate(runtime, *, steps=None, config=PealdConfig()):
    """Same input-only deployment forward, followed by synthetic target join."""
    from . import online_inference as online
    fa.require_head(runtime.module)
    trained = runtime.module.FUSION_ANSWER_TRAINING
    if trained.get('reference_contract') != REFERENCE_CONTRACT or config.reference_mode != trained.get('reference_mode'):
        raise ValueError('Internal Test requires the same normal_train head contract as external inference')
    selected = fast._steps(runtime, trained['trained_steps'] if steps is None else steps)
    runner = online.InferenceSession(runtime, center_batch_size=config.feature_center_batch_size)
    rows = []; unobserved = []
    for step in selected:
        test_cases = cases(runtime, 'test', steps=[step], config=config)
        if not test_cases:
            unobserved.append(dict(step=step, reason='no_Test_window_with_independent_TRAIN_reference'))
        for case in test_cases:
            reply = runner.process_window(current=case['current'], step=step, cohort=case['cohort'],
                run_id=case['run_id'], centers=case['centers'], position=case['position'], standardized=True)
            if reply['status'] != 'OK':
                unobserved.append(dict(step=step, pair_id=case['pair_id'], reason=reply['reason']))
                continue
            names = list(runtime.encoders[step][1]['sensor_columns'])
            for answer in reply['answers']:
                center = answer['center_sensor']; kind = answer['answer_type']
                changed = bool(case['changed_mask'][:, names.index(center)].any())
                rows.append(dict(answer, step=step, pair_id=case['pair_id'], run_id=case['run_id'],
                    kind=case['kind'], challenge=case['challenge'], strength=case['strength'],
                    lag_samples=case.get('lag_samples', 0), strength_unit=case['strength_unit'],
                    injected=case['kind'] != 'equal', split='test', center_is_perturbed=changed,
                    scored=True, label=case['targets'][center][kind]))
    if not rows:
        return ExperimentReport('PEALD internal Test: NOT_EVALUATED',
            pd.DataFrame([dict(status='NOT_EVALUATED', questions=0,
                reason='no_Test_window_with_independent_TRAIN_reference')]),
            diagnostics=dict(unobserved_test_steps=pd.DataFrame(unobserved),
                             boolean_populations=pd.DataFrame()),
            metadata=dict(decision_system=fa.SYSTEM, split='test', reference_contract=REFERENCE_CONTRACT,
                          nominal_original_normality_assumed=True, real_fault_truth=False,
                          test_used_for_fitting=False, deployed_weights_updated=False))
    report = fast._report('PEALD nominal/injected Test using deployed TRAIN-reference inference', rows, all_answers=rows)
    report.diagnostics['unobserved_test_steps'] = pd.DataFrame(unobserved)
    report.diagnostics['step_head_selection'] = pd.DataFrame(trained.get('step_head_selection', []))
    booleans = report.details.loc[report.details.answer_type.eq('BOOLEAN')].copy()
    booleans['population'] = np.where(~booleans.injected, 'nominal_original_all_centers',
        np.where(booleans.center_is_perturbed, 'injected_centers', 'unchanged_centers_on_injected_input'))
    report.diagnostics['boolean_populations'] = pd.DataFrame([
        dict(population=name, questions=len(g), yes_predictions=int(g.prediction.eq(1).sum()),
             correct=int(g.prediction.eq(g.label).sum()), no_targets=int(g.label.eq(0).sum()),
             false_positives=int((g.prediction.eq(1) & g.label.eq(0)).sum()),
             yes_targets=int(g.label.eq(1).sum()), false_negatives=int((g.prediction.eq(0) & g.label.eq(1)).sum()))
        for name, g in booleans.groupby('population', sort=False)])
    report.metadata.update(target_domain='PEALD', target_readout_trained=True,
        source_weights_updated=False, normalization_fit_split='train', reference_mode=config.reference_mode,
        reference_contract=REFERENCE_CONTRACT, nominal_original_normality_assumed=True, real_fault_truth=False)
    return report


def predict_arrays(runtime, current, reference, step, center_sensor, position=0.):
    """Target inference routes by explicit Step; never mutates the common head."""
    scoped = peald_step_heads.for_step(runtime, step)
    return fa.predict_arrays(scoped.module, current, reference, step, center_sensor, position)


def original(runtime, *, steps=None, config=PealdConfig(), center_batch_size=8):
    """Replay through the deployment session; no separate reference algorithm."""
    from . import online_inference as online, inference_benchmark as timing
    selected = (runtime.module.FUSION_ANSWER_TRAINING or {}).get('trained_steps') if steps is None else steps
    session = online.InferenceSession(runtime, center_batch_size=center_batch_size)
    report = online.run_cases(session, timing.peald_cases(runtime, steps=selected, config=config), domain='PEALD')
    report.metadata.update(reference_contract=REFERENCE_CONTRACT, real_fault_truth=False,
                           nominal_original_normality_assumed=True)
    return report
