"""Discover a compatible local worker without changing GEVIS dependencies."""
from functools import lru_cache
from io import BytesIO
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
WORKER=Path(__file__).with_name('merlion_worker.py')


def environment_python(root=ROOT):
    return Path(root)/'.cache/venvs/merlion'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def worker_environment():
    """Notebook inline backends must not leak into a non-notebook process."""
    environment=os.environ.copy()
    environment['MPLBACKEND']='Agg'
    return environment


class _ProbeFailure(Exception):
    pass


@lru_cache(maxsize=8)
def _probe_ready(executable,worker_mtime):
    # Exceptions are not cached by lru_cache. Cache only successful imports;
    # include worker identity so replacing its code invalidates a positive probe.
    try:
        result=subprocess.run([executable,'-I',str(WORKER),'--probe'],capture_output=True,
                              stdin=subprocess.DEVNULL,env=worker_environment(),timeout=60)
        if result.returncode: raise _ProbeFailure('IMPORT_FAILED')
        payload=json.loads(result.stdout.decode('utf-8'))
        if payload.get('ready') is not True: raise _ProbeFailure('INVALID_PROBE')
        return payload
    except subprocess.TimeoutExpired:
        raise _ProbeFailure('PROBE_TIMEOUT') from None
    except (OSError,ValueError,subprocess.SubprocessError):
        raise _ProbeFailure('PROBE_FAILED') from None


def _probe(executable):
    try:
        return _probe_ready(executable,WORKER.stat().st_mtime_ns)
    except _ProbeFailure as error:
        return {'ready':False,'reason':str(error)}
    except OSError:
        return {'ready':False,'reason':'WORKER_MISSING'}


_probe.cache_clear=_probe_ready.cache_clear
_probe.cache_info=_probe_ready.cache_info


def merlion_backend(*,refresh=False):
    if refresh: _probe.cache_clear()
    configured=os.environ.get('GEVIS_MERLION_PYTHON')
    candidates=[Path(configured)] if configured else [environment_python()]
    if not configured and importlib.util.find_spec('merlion') is not None:
        candidates.append(Path(sys.executable))
    reason='NOT_INSTALLED'
    for path in candidates:
        if not path.is_file(): continue
        info=_probe(str(path.resolve()))
        if info['ready']:
            return {**info,'python':str(path.resolve()),'backend':'local isolated worker'}
        reason=info['reason']
    return {'ready':False,'reason':reason,'backend':'unavailable'}


def require_merlion():
    info=merlion_backend()
    if not info['ready']:
        raise ImportError(f"Merlion preflight failed [{info['reason']}]. From the project root run: "
                          'python scripts/setup_merlion.py --check ; repair with '
                          'python scripts/setup_merlion.py if needed, then rerun the first notebook cell.')
    return info


def worker_scores(train,test,*,seed=42):
    train=np.asarray(train,dtype=np.float64); test=np.asarray(test,dtype=np.float64)
    if train.ndim!=2 or test.ndim!=2 or train.shape[1]!=test.shape[1] or train.shape[1]<1 or min(len(train),len(test))<1:
        raise ValueError('Expected nonempty aligned time/sensor matrices')
    if not np.isfinite(train).all() or not np.isfinite(test).all(): raise ValueError('Finite arrays required')
    info=require_merlion()
    data=BytesIO(); np.savez(data,train=train,test=test,seed=np.asarray(seed,dtype=np.int64))
    try:
        result=subprocess.run([info['python'],'-I',str(WORKER)],input=data.getvalue(),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=worker_environment(),timeout=180)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Merlion local worker exceeded its time budget') from None
    if result.returncode:
        raise RuntimeError('Merlion local worker failed; run python scripts/setup_merlion.py --check')
    try:
        with np.load(BytesIO(result.stdout),allow_pickle=False) as returned: scores=returned['scores'].copy()
    except (OSError,ValueError,KeyError):
        raise RuntimeError('Merlion returned an invalid score response') from None
    if scores.shape!=(len(test),) or not np.isfinite(scores).all(): raise RuntimeError('Merlion score alignment failed')
    return scores
