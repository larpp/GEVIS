"""Read immutable TIPS partitions; never repartition, interpolate or join gaps."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PARTITIONS = {'train': 'train', 'validation': 'valid', 'test': 'test'}
META = ['_run_id', '_timestamp', '_continuous_segment', '_main_occurrence',
        '_cycle_id', 'ProcessRecipeStepSeqNo', 'ProcessRecipeStepName']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    p = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact paths must stay inside their source root')
    return p


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class TipsData:
    def __init__(self, tips_root, split_dir, columns_file=None, *, main_steps=(14,),
                 window_length=64, stride=64, window_scope='main', seed=42,
                 selected_sensors=None, selection_identity=None):
        self.root = Path(tips_root).resolve()
        self.split = Path(split_dir).resolve()
        self.columns_file = Path(columns_file).resolve() if columns_file is not None else None
        self.selection_identity = selection_identity
        self.window_length, self.stride = int(window_length), int(stride)
        self.steps = None if main_steps == 'all' else {int(s) for s in main_steps}
        if min(self.window_length, self.stride) < 1 or window_scope not in ('main', 'cycle'):
            raise ValueError('Positive window/stride and main or cycle scope required')
        self.scope, self.seed = window_scope, seed
        if selected_sensors is not None and columns_file is not None:
            raise ValueError('Choose profile sensors or a legacy columns file, not both')
        self.requested = list(selected_sensors) if selected_sensors is not None else self.columns_file.read_text(encoding='utf-8-sig').splitlines()
        if not self.requested or len(set(self.requested)) != len(self.requested):
            raise ValueError('Unique source column names required; whitespace is significant')
        manifest = json.loads((self.split / 'manifest.json').read_text(encoding='utf-8'))
        for name, entry in manifest['files'].items():
            if sha(inside(self.split, name)) != entry['sha256']:
                raise ValueError('Changed split artifact: ' + name)
        self.source = inside(self.root, manifest['identity']['source'])
        if sha(self.source / 'manifest.json') != manifest['identity']['source_manifest_sha256']:
            raise ValueError('Changed source manifest')
        source_manifest = json.loads((self.source / 'manifest.json').read_text(encoding='utf-8'))
        self.assignments = csv_rows(self.split / 'assignments.csv')
        ids = [r['run_id'] for r in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError('Run membership overlaps across partitions')
        self.rows = {}
        for part in [*PARTITIONS.values(), 'reserved_transfer']:
            rows = csv_rows(self.split / part / 'runs.csv')
            if rows != [r for r in self.assignments if r['split'] == part]:
                raise ValueError('Partition membership differs from assignments: ' + part)
            self.rows[part] = rows
        # Hash verification does not inspect held-out sensor values.
        for row in self.assignments:
            path = inside(self.root, row['parquet_path'])
            name = path.relative_to(self.source).as_posix()
            if sha(path) != source_manifest['files'][name]['sha256']:
                raise ValueError('Changed source Parquet: ' + name)
            if pq.read_metadata(path).num_rows != int(row['rows_after']):
                raise ValueError('Parquet row count differs from manifest')
        self.identity = dict(split_manifest=sha(self.split / 'manifest.json'),
            source_manifest=sha(self.source / 'manifest.json'),
            columns=sha(columns_file) if columns_file else list(self.requested), selection=selection_identity,
            main_steps='all' if self.steps is None else sorted(self.steps),
            window_length=self.window_length, stride=self.stride, window_scope=self.scope)
        self.stats = None
        self.audit = []

    def frames(self, partition):
        if partition not in PARTITIONS:
            raise ValueError('Reserved transfer data cannot enter training or evaluation')
        for row in self.rows[PARTITIONS[partition]]:
            path = inside(self.root, row['parquet_path'])
            schema = pq.read_schema(path).names
            aliases = {}
            for name in schema:
                aliases.setdefault(name.strip(), []).append(name)
            physical = {}
            for name in self.requested:
                matches = aliases.get(name.strip(), [])
                if len(matches) != 1:
                    raise ValueError('Missing or ambiguous source sensor: ' + repr(name))
                physical[name] = matches[0]
            frame = pq.read_table(path, columns=META + list(physical.values())).to_pandas()
            frame = frame.rename(columns={raw: canonical for canonical, raw in physical.items()})
            if set(frame['_run_id'].astype(str)) != {row['run_id']}:
                raise ValueError('Run ID differs from partition manifest')
            frame['_step'] = pd.to_numeric(frame['ProcessRecipeStepSeqNo'], errors='raise').astype(int)
            if self.steps is not None:
                frame = frame.loc[frame['_step'].isin(self.steps)].copy()
            # Preserve provenance coordinates. Do not reindex after exclusions.
            converted = {}
            for name in self.requested:
                if name.startswith('ValveAct_'):
                    converted[name] = frame[name].map({'OPEN': 1., 'CLOSE': 0.})
                else:
                    converted[name] = pd.to_numeric(frame[name], errors='coerce')
            frame = pd.concat([frame.drop(columns=self.requested), pd.DataFrame(converted, index=frame.index)], axis=1)
            frame.attrs['run_id'] = int(row['run_id'])
            yield frame

    def fit(self):
        """Streaming TRAIN-only first/second moments, separately for each Main Step."""
        accum = {}
        for frame in self.frames('train'):
            for step, g in frame.groupby('_step', sort=False):
                x = g[self.requested].to_numpy(dtype=np.float64)
                x = x[np.isfinite(x).all(1)]
                if not len(x):
                    continue
                n, mean, m2 = accum.get(int(step), (0, np.zeros(x.shape[1]), np.zeros(x.shape[1])))
                count = len(x); local = x.mean(0); delta = local - mean
                m2 = m2 + ((x - local) ** 2).sum(0) + delta ** 2 * n * count / (n + count)
                mean = mean + delta * count / (n + count)
                accum[int(step)] = (n + count, mean, m2)
        if not accum:
            raise ValueError('No finite TRAIN rows for configured Main Steps')
        self.stats = {s: dict(count=n, mean=m.tolist(), scale=np.where(
            np.sqrt(v / n) > 1e-6, np.sqrt(v / n), 1.).tolist(),
            varying=(v / n > 1e-12).tolist()) for s, (n, m, v) in accum.items()}
        return self.stats

    def windows(self, partition, max_per_run=16):
        if self.stats is None:
            raise ValueError('Fit or restore TRAIN statistics first')
        result = []; rng = np.random.default_rng(self.seed + list(PARTITIONS).index(partition))
        for frame in self.frames(partition):
            rid = frame.attrs['run_id']
            if frame.empty:
                self.audit.append(dict(partition=partition, run_id=rid, step=None,
                    candidate_windows=0, retained_windows=0, status='NO_CONFIGURED_STEP_ROWS'))
                continue
            selected = {int(s): [0, []] for s in frame['_step'].unique()}
            keys = ['_continuous_segment', '_main_occurrence', '_step']
            if self.scope == 'cycle':
                keys.append('_cycle_id')
            for _, g in frame.groupby(keys, sort=False, dropna=False):
                step = int(g['_step'].iloc[0])
                if step not in self.stats:
                    continue
                x = g[self.requested].to_numpy(dtype=np.float32)
                t = pd.to_datetime(g['_timestamp']).astype('datetime64[ns]').astype('int64').to_numpy() / 1e9
                # A removed row, nonfinite reading or cadence gap splits the native grid.
                bad = ~np.isfinite(x).all(1)
                dt = np.diff(t)
                cuts = np.r_[True, (dt < .95) | (dt > 1.05) | bad[1:] | bad[:-1]]
                pieces = np.cumsum(cuts)
                for piece in np.unique(pieces):
                    ix = np.flatnonzero((pieces == piece) & ~bad)
                    if len(ix) < self.window_length:
                        continue
                    for offset in range(0, len(ix) - self.window_length + 1, self.stride):
                        positions = ix[offset:offset + self.window_length]
                        st = self.stats[step]
                        values = ((x[positions] - np.array(st['mean'])) / np.array(st['scale'])).astype(np.float32)
                        window = dict(values=values, run_id=rid, step=step,
                            occurrence=int(g['_main_occurrence'].iloc[0]),
                            segment=int(g['_continuous_segment'].iloc[0]),
                            cycle=None if pd.isna(g['_cycle_id'].iloc[0]) else str(g['_cycle_id'].iloc[0]),
                            start=str(g['_timestamp'].iloc[positions[0]]),
                            end=str(g['_timestamp'].iloc[positions[-1]]),
                            position=float(positions[0] / max(1, len(g) - 1)))
                        bucket = selected.setdefault(step, [0, []]); bucket[0] += 1
                        # Reservoir selection scans the full timeline without retaining every window.
                        if len(bucket[1]) < max_per_run:
                            bucket[1].append(window)
                        else:
                            j = int(rng.integers(bucket[0]))
                            if j < max_per_run:
                                bucket[1][j] = window
            for step, (candidates, windows) in selected.items():
                result.extend(windows)
                self.audit.append(dict(partition=partition, run_id=rid, step=step,
                    candidate_windows=candidates, retained_windows=len(windows),
                    status='READY' if windows else 'NO_COMPLETE_FINITE_CONTIGUOUS_WINDOW'))
        if not result:
            raise ValueError('No eligible windows; check scope, Main Steps and window length')
        return result
