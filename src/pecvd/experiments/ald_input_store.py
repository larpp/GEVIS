"""File-backed ALD sensor columns; retain native rows and invalid-value evidence."""
from pathlib import Path
from uuid import uuid4
import hashlib
import json

import numpy as np
import pandas as pd


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class _Rows:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, indices):
        return DiskSensorFrame(self.frame.values, self.frame.invalid, self.frame.columns,
                               self.frame.rows[indices])


class DiskSensorFrame:
    """The bounded pandas operations required by native ALD preparation.

    Raw sensor matrices stay in read-only numpy mappings. A scalar column read
    materializes one unit/column; a list read materializes one requested window.
    Missing data remain missing. Non-numeric text remains distinguishable from
    missing cells so TRAIN rejection follows the original preparation policy.
    """
    def __init__(self, values, invalid, columns, rows=None):
        self.values = values
        self.invalid = invalid
        self.columns = pd.Index(columns)
        self.rows = np.arange(len(values), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __contains__(self, name):
        return name in self.columns

    @property
    def iloc(self):
        return _Rows(self)

    def __getitem__(self, names):
        if isinstance(names, str):
            index = self.columns.get_loc(names)
            values = np.array(self.values[self.rows, index], copy=True)
            invalid = self.invalid[self.rows, index]
            if invalid.any():
                values = values.astype(object)
                values[invalid] = '<non-numeric sensor value>'
            return pd.Series(values, index=self.rows, name=names)
        return pd.DataFrame({name: self[name] for name in names}, index=self.rows)


def stage_frame(frame, sensors, cache_root, *, source_sha256):
    """Atomically cache one file and release its original DataFrame afterwards."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    names = sorted(name for name in sensors if name in frame)
    identity = dict(schema='gevis.ald_sensor_columns.v1', source_sha256=source_sha256,
                    columns=names, rows=len(frame), code_sha256=_digest(__file__))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
    manifest_path = cache_root / (key + '.json')
    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text(encoding='utf-8'))
        if record.get('identity') != identity:
            raise ValueError('ALD input cache identity differs')
        paths = {}
        for kind in ('values', 'invalid'):
            path = (cache_root / record[kind]['file']).resolve()
            path.relative_to(cache_root.resolve())
            if _digest(path) != record[kind]['sha256']:
                raise ValueError('ALD input cache content differs; remove the named cache manifest to rebuild: ' + str(manifest_path))
            paths[kind] = path
    else:
        # Unique data names keep concurrent notebook kernels from replacing an
        # already mapped file. Only the tiny completion manifest is replaced.
        nonce = uuid4().hex
        paths = {kind: cache_root / f'{key}-{nonce}-{kind}.npy' for kind in ('values', 'invalid')}
        temporary = {kind: path.with_suffix('.tmp') for kind, path in paths.items()}
        shape = (len(frame), len(names))
        values = np.lib.format.open_memmap(temporary['values'], mode='w+', dtype=np.float64, shape=shape)
        invalid = np.lib.format.open_memmap(temporary['invalid'], mode='w+', dtype=np.bool_, shape=shape)
        for index, name in enumerate(names):
            raw = frame[name]
            numeric = pd.to_numeric(raw, errors='coerce')
            values[:, index] = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
            invalid[:, index] = (raw.notna() & numeric.isna()).to_numpy(dtype=bool)
        values.flush(); invalid.flush()
        del values, invalid
        for kind in paths:
            temporary[kind].replace(paths[kind])
        record = dict(identity=identity, **{kind: dict(file=path.name, sha256=_digest(path))
                                          for kind, path in paths.items()})
        temporary_manifest = cache_root / f'{key}-{nonce}.json.tmp'
        temporary_manifest.write_text(json.dumps(record, sort_keys=True), encoding='utf-8')
        temporary_manifest.replace(manifest_path)
    values = np.load(paths['values'], mmap_mode='r', allow_pickle=False)
    invalid = np.load(paths['invalid'], mmap_mode='r', allow_pickle=False)
    if (values.shape != (len(frame), len(names)) or invalid.shape != values.shape
            or values.dtype != np.float64 or invalid.dtype != np.bool_):
        raise ValueError('ALD input cache shape/dtype differs')
    return DiskSensorFrame(values, invalid, names)
