"""Source-bound PECVD native windows, without relative/future Step progress."""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from . import peald_training as p


def input_metadata(original):
    """Small native input contract, never a copy of a full encoder checkpoint."""
    fields=('sensor_columns','sensor_descriptors','window_length','mean','std','step_descriptor',
        'cadence_seconds','tolerance_seconds','native_kind','native_row_origin','allowed_recipe_keys',
        'execution','parent_sequence','parent_step','expected_substeps')
    def plain(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [plain(v) for v in value]
        return value
    result={k:plain(original[k]) for k in fields if k in original}
    for key in ('mean','std'):
        result[key]=np.asarray(result[key]).reshape(-1).tolist()
    return result


def prepare(root, source, *, splits=('train', 'validation')):
    root = Path(root).resolve()
    encoder_root = Path(source.module.ENCODER_ROOT)
    manifest_path = encoder_root/'encoder_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    csv_path = root/'data/all_processes_analysis.csv'
    if p.digest(csv_path) != manifest['source_sha256']:
        raise ValueError('PECVD CSV differs from the preserved STEP1 source')
    if not set(splits) <= {'train', 'validation', 'test'}:
        raise ValueError('Unknown PECVD preparation split')
    partitions = {k: list(map(str, v)) for k, v in manifest['run_split'].items()}
    if any(set(partitions[a]) & set(partitions[b]) for a, b in
           [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('PECVD original Run partitions overlap')
    # Native coordinates only. No sensor values or final duration are read here.
    columns = ['process_key', 'row_in_process', 'row_in_step', 'step_number', 'step_elapsed_seconds']
    coords = pd.read_csv(csv_path, usecols=columns, dtype={'process_key': 'string'})
    if coords.duplicated(['process_key', 'row_in_process']).any():
        raise ValueError('PECVD Run/native row identity is ambiguous')
    train_coords = coords.loc[coords.process_key.isin(partitions['train'])]
    origin = int(train_coords.row_in_step.min())
    if origin not in (0, 1):
        raise ValueError('Unknown native row_in_step origin; no guessed elapsed position')
    lookup = coords.set_index(['process_key', 'row_in_process'])
    windows = {}; metadata = {}; cache_files = {}; support = []
    for step, (encoder, original) in source.encoders.items():
        path = encoder_root/'windows'/f'{step}.npz'
        cache_files[path.relative_to(root).as_posix()] = p.digest(path)
        pools = {s: [] for s in partitions}
        meta = input_metadata(original); length = int(meta['window_length'])
        cadences = []; train_recipes = []
        with np.load(path, allow_pickle=False) as cache:
            for split in partitions:
                ids = cache[f'{split}_process_id'].astype(str)
                if not set(ids) <= set(partitions[split]):
                    raise ValueError('PECVD cached window belongs to another Run partition')
                starts = cache[f'{split}_window_start']; ends = cache[f'{split}_window_end']
                cohorts = cache[f'{split}_cohort_key'].astype(str)
                if split=='train':train_recipes=sorted(set(cohorts))
                # TRAIN metadata supplies timing even in evaluation-only preparation.
                if split not in splits and split != 'train':
                    continue
                arrays = cache[split] if split in splits else None
                for i, (run, start, end, cohort) in enumerate(zip(ids, starts, ends, cohorts)):
                    a = lookup.loc[(run, int(start))]; b = lookup.loc[(run, int(end))]
                    if (int(end)-int(start) != length-1 or a.step_number != b.step_number
                            or int(b.row_in_step)-int(a.row_in_step) != length-1):
                        raise ValueError('PECVD cached window crosses an occurrence or missing native row')
                    if split == 'train':
                        cadence = (float(b.step_elapsed_seconds)-float(a.step_elapsed_seconds))/(length-1)
                        if np.isfinite(cadence) and cadence > 0:
                            cadences.append(cadence)
                    if arrays is None:
                        continue
                    values = np.asarray(arrays[i], np.float32)
                    if values.shape != (length, len(meta['sensor_columns'])) or not np.isfinite(values).all():
                        raise ValueError('Invalid PECVD frozen sensor/window grid')
                    offset = int(a.row_in_step)-origin
                    pools[split].append(dict(current=values.copy(), run_id=run,
                        lineage_id=p._hash(run), pair_id=p._hash((step, run, int(start), int(end))),
                        step=step, split=split, sensor_columns=list(meta['sensor_columns']),
                        cohort=(cohort, '', offset, offset), input_row_start=int(start), input_row_end=int(end)))
        if not cadences:
            raise ValueError('No known TRAIN elapsed-time cadence for PECVD')
        if p.digest(path)!=cache_files[path.relative_to(root).as_posix()]:
            raise ValueError('PECVD source window cache changed during preparation')
        meta['cadence_seconds'] = float(np.median(cadences))
        meta['native_kind'] = 'pecvd'; meta['native_row_origin'] = origin
        meta['allowed_recipe_keys'] = train_recipes
        metadata[step] = meta; windows[step] = pools
        support.append(dict(step=step, **{s+'_windows':len(v) for s, v in pools.items()}))
    if p.digest(csv_path) != manifest['source_sha256']:
        raise ValueError('PECVD source changed during native preparation')
    identity = dict(domain='PECVD', source_files={csv_path.relative_to(root).as_posix(): manifest['source_sha256']},
        cache_files=cache_files, manifest_sha256=p.digest(manifest_path), split_runs=partitions,
        native_preparation_sha256=p.digest(__file__), native_row_origin=origin,
        input_policy='preserved_standardized_STEP1_windows_and_native_elapsed_only')
    result = p.Prepared(windows, metadata, dict(support=pd.DataFrame(support),
        cadence=pd.DataFrame([dict(step=s, cadence_seconds=m['cadence_seconds']) for s,m in metadata.items()])), partitions, identity)
    result.process = SimpleNamespace(source_columns=dict(run='process_key', order='row_in_process',
        step='step_label', cohort='_cohort', elapsed='row_in_step'))
    return result
