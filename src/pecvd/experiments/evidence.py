"""Observed-value evidence. Not calibrated fault labels or causal explanations."""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
import torch

_WINDOW_CACHE = ContextVar('gevis_window_cache', default=None)


@contextmanager
def reuse_window_inputs():
    """One synchronous question scan only; reset on success AND exception."""
    token = _WINDOW_CACHE.set({})
    try:
        yield
    finally:
        _WINDOW_CACHE.reset(token)


def prepare_inputs(current_rows, reference_rows, step, encoder, meta, device,
                   prepare_window, numeric_state, feature_function):
    cache = _WINDOW_CACHE.get()
    key = (id(current_rows), id(reference_rows), step, id(encoder), id(meta),
           str(device), id(prepare_window), id(numeric_state), id(feature_function))
    if cache is not None and key in cache:
        return cache[key]
    with torch.no_grad():
        a, ac = prepare_window(current_rows, meta)
        b, bc = prepare_window(reference_rows, meta)
        a, b = a.to(device), b.to(device)
        state = numeric_state(encoder, a, b)
        patches = state.shape[1] // len(meta['sensor_columns'])
        features = feature_function(a, b, state, patches)
    result = a, b, ac, bc, state, patches, features
    if cache is not None:
        cache[key] = result
    return result


def segments(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def pointwise_evidence(current, reference, sensor, scope=0, *, threshold=1., original_scale=1.):
    """Coordinates refer to prepared window rows, half-open intervals.

    Inputs are standardized using TRAIN constants. Keep the legacy five-feature
    answer operation unchanged; peak/duration evidence is a separate output.
    """
    cache=_WINDOW_CACHE.get()
    key=('pointwise',id(current),id(reference),sensor,scope,threshold,original_scale)
    if cache is not None and key in cache: return cache[key]
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    a, b = array(current), array(reference)
    if a.ndim == 3 and len(a) == 1: a, b = a[0], b[0]
    if a.ndim != 2 or a.shape != b.shape or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matching finite prepared time/sensor arrays required')
    if not 0 <= sensor < a.shape[1] or scope not in (0, 1, 2) or threshold <= 0 or original_scale <= 0:
        raise ValueError('Valid sensor/scope and positive evidence scales required')
    lo, hi = ((0, len(a)) if scope == 0 else (0, max(2, len(a)//3)) if scope == 1
              else (max(0, 2*len(a)//3), len(a)))
    delta = a[lo:hi, sensor] - b[lo:hi, sensor]
    peak = int(np.abs(delta).argmax())
    intervals = [(x+lo, y+lo) for x, y in segments(np.abs(delta) > threshold)]
    result={'peak_row': lo+peak, 'peak_absolute_train_std': float(abs(delta[peak])),
            'peak_signed_original_units': float(delta[peak]*original_scale),
            'mean_absolute_train_std': float(np.abs(delta).mean()),
            'exceedance_intervals': intervals, 'exceedance_rows': sum(y-x for x,y in intervals),
            'longest_exceedance_rows': max((y-x for x,y in intervals), default=0),
            'scope_rows': [lo,hi], 'threshold_train_std': float(threshold),
            'coordinate_system': 'prepared window row; not seconds',
            'meaning': 'observed current/reference difference; not fault probability'}
    if cache is not None: cache[key]=result
    return result


def evidence_survey(current, reference, *, threshold=1.):
    """All-sensor observed peak ranking, distinct from learned grounding."""
    count = current.shape[-1]
    rows = [dict(sensor_index=i, **pointwise_evidence(current, reference, i, threshold=threshold)) for i in range(count)]
    return sorted(rows, key=lambda r:r['peak_absolute_train_std'], reverse=True)
