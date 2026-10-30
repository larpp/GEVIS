"""Deterministic copy-only interventions with independent positional truth."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Injection:
    kind: str
    sensor: int
    start: int
    end: int
    strength: float = 2.0
    seed: int = 42


KINDS = ('level_shift', 'drift', 'spike', 'variance', 'stuck', 'delay', 'relationship')


def inject(values, spec, scale):
    """Scale must come from TRAIN only, not the held-out window.

    Truth identifies an intervention; it does not assign defect/severity labels.
    A stuck intervention can be ineffective on an already constant segment.
    """
    x = np.array(values, dtype=float, copy=True)
    s = np.asarray(scale, dtype=float)
    if x.ndim != 2 or s.shape != (x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError('Finite time/sensor array and positive TRAIN scales required')
    if spec.kind not in KINDS or not 0 <= spec.sensor < x.shape[1] or not 0 <= spec.start < spec.end <= len(x):
        raise ValueError('Invalid intervention coordinates or kind')
    if not np.isfinite(spec.strength) or spec.strength <= 0:
        raise ValueError('Positive finite intervention strength required')
    a, b, c = spec.start, spec.end, spec.sensor
    before = x.copy()
    amplitude = spec.strength * s[c]
    rng = np.random.default_rng(spec.seed)
    if spec.kind == 'level_shift': x[a:b, c] += amplitude
    elif spec.kind == 'drift': x[a:b, c] += np.linspace(0, amplitude, b-a)
    elif spec.kind == 'spike': x[(a+b)//2, c] += amplitude
    elif spec.kind == 'variance': x[a:b, c] += rng.normal(0, amplitude, b-a)
    elif spec.kind == 'stuck': x[a:b, c] = x[a, c]
    elif spec.kind == 'delay':
        lag = min(b-a-1, max(1, round(spec.strength)))
        x[a:b, c] = x[np.maximum(a, np.arange(a, b)-lag), c]
    else:
        if x.shape[1] < 2: raise ValueError('Relationship intervention requires multiple sensors')
        original=x[a:b,c].copy()
        shuffled=rng.permutation(original)
        mixing=min(1.,spec.strength/3.)
        x[a:b,c]=original+mixing*(shuffled-original)
    mask = np.abs(x-before) > 0
    truth = {**asdict(spec), 'meaning': 'controlled intervention, not verified equipment fault',
             'strength_unit':('samples' if spec.kind=='delay' else 'duration-controlled' if spec.kind=='stuck'
                              else 'mixing_fraction=min(1,strength/3)' if spec.kind=='relationship' else 'train_std'),
             'effective': bool(mask.any()), 'changed_mask': mask}
    truth['effectiveness_rule']='any actual numeric sample changed; not a mean-shift test'
    truth['ineffective_reason']=(None if truth['effective'] else
        'constant_target_segment' if np.ptp(before[a:b,c])==0 else 'transformation_left_samples_unchanged')
    truth['relationship_evaluable']=None
    if spec.kind=='relationship':
        def correlations(v):
            z=v-v.mean(0,keepdims=True)
            z=z/np.maximum(np.sqrt((z*z).sum(0,keepdims=True)),1e-8)
            return z.T@z
        delta=np.abs(correlations(x[a:b])-correlations(before[a:b]))[c]
        delta[c]=0
        truth['relationship_evaluable']=bool((delta>1e-6).any())
    return x, truth


def synthetic_fixture(seed=42, length=96, sensors=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4*np.pi, length)
    reference = np.stack([np.sin(t+i*.15) for i in range(sensors)], 1)
    current = reference + rng.normal(0, .025, reference.shape)
    return current, reference, np.full(sensors, .25)
