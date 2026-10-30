"""Portable array contract. Validation does not establish model transfer."""
import numpy as np


def validate_transfer_pack(pack):
    """Each split: windows [N,T,C], run IDs [N]. Sensor order must be explicit."""
    required={'train','validation','test','train_run_ids','validation_run_ids','test_run_ids','sensors','dataset_id'}
    if not required<=pack.keys(): raise ValueError('Incomplete transfer pack')
    sensors=list(pack['sensors'])
    if not sensors or len(sensors)!=len(set(sensors)): raise ValueError('Unique sensors required')
    ids={}
    for split in ('train','validation','test'):
        x=np.asarray(pack[split]); runs=np.asarray(pack[split+'_run_ids'])
        if x.ndim!=3 or x.shape[0]!=len(runs) or not len(runs) or x.shape[2]!=len(sensors) or not np.isfinite(x).all():
            raise ValueError('Invalid split arrays/run identities')
        ids[split]=set(map(str,runs))
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        if ids[a]&ids[b]: raise ValueError('Run leakage across dataset splits')
    train=np.asarray(pack['train'])
    scale=train.std(axis=(0,1))
    if (scale<=1e-8).any(): raise ValueError('Constant channels need an explicit physical scale')
    return {'dataset_id':str(pack['dataset_id']),'sensors':len(sensors),'train_scale':scale,
            'contract_valid':True,'gevis_transfer_verified':False,
            'next':'Bind matching trained STEP 1/2 artifacts before GEVIS inference'}
