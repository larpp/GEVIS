"""Immutable per-case disk features and bounded row gathering; no model fitting."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from . import current_input as current

SCHEMA='gevis.frozen_feature_parts.v1'


def _digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _atomic_json(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(current._json(value),encoding='utf8');temporary.replace(path)


@dataclass(frozen=True)
class DiskFeatures:
    path: Path
    shape: tuple
    sha256: str
    stamp: tuple

    def take(self,indices):
        stat=self.path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=self.stamp:raise ValueError('Frozen feature part changed during fitting')
        array=np.load(self.path,mmap_mode='r',allow_pickle=False)
        try:return torch.from_numpy(np.array(array[np.asarray(indices,dtype=np.int64)],copy=True))
        finally:array._mmap.close()


class FeatureCache:
    def __init__(self,root,identity,max_bytes):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.identity_hash=current._sha(identity);self.max_bytes=max_bytes
        manifest=self.root/'identity.json'
        if manifest.exists():
            record=json.loads(manifest.read_text(encoding='utf8'))
            if record.get('schema')!=SCHEMA or current._sha(record.get('identity'))!=self.identity_hash:
                raise ValueError('Frozen feature corpus identity differs')
        else:_atomic_json(dict(schema=SCHEMA,identity=identity),manifest)

    def get(self,index,centers):
        path=self.root/f'{index:08d}.npy';marker=self.root/f'{index:08d}.json'
        if not marker.exists():return None  # Uncommitted data/temp files are recomputed.
        record=json.loads(marker.read_text(encoding='utf8'))
        if (record.get('corpus')!=self.identity_hash or record.get('index')!=index
                or not path.is_file() or record.get('sha256')!=_digest(path)):
            raise ValueError('Frozen feature part identity/content differs')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:
            if (array.dtype!=np.float32 or array.ndim!=3 or array.shape[:2]!=(centers,3)
                    or list(array.shape)!=record.get('shape') or record.get('dtype')!='float32'
                    or array.nbytes>self.max_bytes):
                raise ValueError('Frozen feature part shape/dtype/working budget differs')
            if not np.isfinite(array).all():raise ValueError('Nonfinite frozen feature part')
            shape=tuple(array.shape)
        finally:array._mmap.close()
        stat=path.stat()
        return DiskFeatures(path,shape,record['sha256'],(stat.st_size,stat.st_mtime_ns))

    def put(self,index,value):
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.ndim!=3 or value.shape[1]!=3:
            raise ValueError('Finite float32 case grid required')
        if value.numel()*value.element_size()>self.max_bytes:
            raise MemoryError('One feature case exceeds the working budget; reduce center generation chunk size')
        if not torch.isfinite(value).all():raise ValueError('Nonfinite frozen feature case')
        path=self.root/f'{index:08d}.npy';temporary=path.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:np.save(stream,value.detach().cpu().numpy(),allow_pickle=False)
        temporary.replace(path)
        _atomic_json(dict(corpus=self.identity_hash,index=index,shape=list(value.shape),dtype='float32',sha256=_digest(path)),
            self.root/f'{index:08d}.json')
        return self.get(index,value.shape[0])

    def finish(self,parts):
        record=dict(corpus=self.identity_hash,cases=len(parts),parts_sha256=current._sha([p.sha256 for p in parts]))
        path=self.root/'complete.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))!=record:raise ValueError('Frozen feature completion manifest differs')
        else:_atomic_json(record,path)


class FeatureRows:
    """Preserve corpus row order and repeated sampler IDs without full torch.cat."""
    def __init__(self,cases,max_bytes):
        if not cases:raise ValueError('Nonempty feature cases required')
        self.parts=[c['features'] for c in cases];self.max_bytes=max_bytes
        shapes=[tuple(p.shape) for p in self.parts]
        if any(len(s)!=3 or s[1]!=3 or s[2]!=shapes[0][2] for s in shapes):
            raise ValueError('Fusion feature dimensions differ')
        self.ends=np.cumsum([s[0] for s in shapes]);self.starts=np.r_[0,self.ends[:-1]]
        self.shape=(int(self.ends[-1]),3,shapes[0][2])

    def take(self,indices):
        ids=np.asarray(indices,dtype=np.int64)
        if ids.ndim!=1 or ((ids<0)|(ids>=self.shape[0])).any():raise ValueError('Valid feature row indices required')
        if len(ids)*self.shape[1]*self.shape[2]*4>self.max_bytes:
            raise MemoryError('One MLP minibatch exceeds the working feature budget; lower batch_size')
        result=torch.empty((len(ids),*self.shape[1:]),dtype=torch.float32)
        part_ids=np.searchsorted(self.ends,ids,side='right')
        for part_id in np.unique(part_ids):
            positions=np.flatnonzero(part_ids==part_id);local=ids[positions]-self.starts[part_id];part=self.parts[part_id]
            if isinstance(part,DiskFeatures):values=part.take(local)
            elif isinstance(part,torch.Tensor):values=part[torch.as_tensor(local)].detach().cpu()
            else:raise ValueError('Unsupported feature storage')
            if values.dtype!=torch.float32 or not torch.isfinite(values).all():raise ValueError('Invalid frozen feature rows')
            result[torch.as_tensor(positions)]=values
        return result

    def moments(self,batch_size):
        count=0;mean=torch.zeros(self.shape[-1],dtype=torch.float64);m2=torch.zeros_like(mean)
        for start in range(0,self.shape[0],batch_size):
            block=self.take(np.arange(start,min(start+batch_size,self.shape[0]))).reshape(-1,self.shape[-1]).double()
            n=len(block);variance,average=torch.var_mean(block,dim=0,unbiased=False)
            delta=average-mean;total=count+n
            m2+=variance*n+delta.square()*(count*n/total);mean+=delta*(n/total);count=total
        return mean.float(),(m2/count).clamp_min(0).sqrt().float()
