"""Read-only reuse of a complete, matching STEP 1 bundle. Never repairs files."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_saved_encoders(root,config,*,backend,source_path=None):
    root=Path(root).resolve()
    manifest_path=root/'encoder_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema')!='gevis.encoder_manifest.rich_ssl.v1' or
        manifest.get('artifact_version')!=config.artifacts.encoder or
        set(manifest.get('step_checkpoints',{}))!=set(config.step_specs)):
        raise ValueError('Saved STEP 1 manifest/config mismatch; no files changed')
    if source_path is not None and sha256(source_path)!=manifest['source_sha256']:
        raise ValueError('Saved STEP 1 source identity mismatch; no files changed')
    checkpoints={}
    for step,(length,patch_length,_) in config.step_specs.items():
        item=manifest['step_checkpoints'][step];path=(root/item['path']).resolve()
        if not path.is_relative_to(root) or sha256(path)!=item['sha256']:
            raise ValueError('Saved STEP 1 encoder path/hash mismatch; no files changed')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if (ck.get('schema')!='gevis.encoder.rich_ssl.v1' or ck.get('encoder_backend','legacy')!=backend or
            ck.get('sensor_columns')!=manifest['sensor_columns'] or ck.get('window_length')!=length or
            ck.get('patch_length')!=patch_length or ck.get('source_sha256')!=manifest['source_sha256'] or
            {k:list(map(str,v)) for k,v in ck.get('run_split',{}).items()}!=manifest['run_split']):
            raise ValueError('Saved STEP 1 checkpoint identity mismatch; no files changed')
        with np.load(root/'windows'/f'{step}.npz',allow_pickle=False) as cache:
            for split in ('train','validation','test'):
                array=cache[split]
                if array.ndim!=3 or array.shape[1:]!=(length,len(manifest['sensor_columns'])):
                    raise ValueError('Saved STEP 1 window grid mismatch; no files changed')
        checkpoints[step]=ck
    return manifest,checkpoints
