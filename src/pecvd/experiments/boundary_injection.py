"""Copy-only shape-preserving interventions relative to fixed TRAIN extrema."""
import numpy as np
from .injection import Injection, inject

RECIPE = 'signed_shape_first_boundary_crossing_v1'
EXCESS_LEVELS = (.05, .15, .5, 1.)
STRENGTH_UNIT = 'excess / max(TRAIN envelope width, float32 resolution floor)'
CONTROLLED_RECIPE = 'known_family_measured_boundary_excess_v2'
CONTROLLED_LEVELS = (.5, 1., 2., 4.)
CONTROLLED_UNIT = 'maximum pointwise excess / max(TRAIN envelope width, one TRAIN-normalized unit)'


def variant(base,lower,upper,*,sensor,kind,level,seed,start,end,direction=1,strength_floor=0.,excess_levels=EXCESS_LEVELS):
    """Scale an entire signed shape, preserving spike support and drift slope.

    The level describes boundary excess, not legacy amplitude-stage truth.
    Delay retains its native lag and is never amplitude-amplified.
    """
    base=np.asarray(base,np.float32);lower=np.asarray(lower,np.float32);upper=np.asarray(upper,np.float32)
    if (base.ndim!=2 or lower.shape!=base.shape or upper.shape!=base.shape
            or not np.isfinite(base).all() or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or (lower>upper).any() or not 0<=sensor<base.shape[1] or not 0<=start<end<=len(base)
            or level not in (1,2,3,4) or direction not in (-1,1)):
        raise ValueError('Finite aligned bounds and explicit intervention coordinates required')
    audit=dict(recipe=RECIPE,kind=kind,requested_excess_stage=level,direction=direction,start=start,end=end,
        strength_unit='native_lag_samples' if kind=='delay' else STRENGTH_UNIT)
    if np.any((base[:,sensor]<lower[:,sensor])|(base[:,sensor]>upper[:,sensor])):
        return None,dict(audit,status='base_outside_envelope')
    if kind=='delay':
        from .fusion_answers import delay_trace
        if level>=len(base) or not np.ptp(base[:,sensor]):return None,dict(audit,status='ineffective_delay')
        observed,_=delay_trace(base,sensor,level)
        outside=np.any((observed[:,sensor]<lower[:,sensor])|(observed[:,sensor]>upper[:,sensor]))
        return observed.astype(np.float32),dict(audit,status='outside' if outside else 'inside',lag_samples=level)
    if kind not in ('level_shift','drift','spike','variance'):raise ValueError('Unknown boundary intervention family')
    if kind in ('drift','variance') and end-start<4:return None,dict(audit,status='insufficient_temporal_support')
    changed,_=inject(base,Injection(kind,sensor,start,end,1.,seed),np.ones(base.shape[1]))
    shape=(changed[:,sensor]-base[:,sensor])*direction;active=np.flatnonzero(shape!=0)
    if not len(active):return None,dict(audit,status='ineffective_shape')
    resolution=np.abs(np.spacing(np.maximum(np.abs(lower[:,sensor]),np.abs(upper[:,sensor])))).astype(float)
    width=np.maximum(upper[:,sensor].astype(float)-lower[:,sensor],np.maximum(strength_floor,np.maximum(32*resolution,np.finfo(np.float32).eps)))
    gap=np.where(shape[active]>0,upper[active,sensor]-base[active,sensor],base[active,sensor]-lower[active,sensor]).astype(float)
    excess=excess_levels[level-1]
    multiplier=float(np.min((gap+excess*width[active])/np.abs(shape[active])))
    observed=base.copy();observed[:,sensor]=(base[:,sensor].astype(float)+multiplier*shape).astype(np.float32)
    if not np.isfinite(observed).all():return None,dict(audit,status='nonfinite_scaled_shape')
    delta=np.maximum(lower[:,sensor]-observed[:,sensor],observed[:,sensor]-upper[:,sensor])
    if not (delta>0).any():return None,dict(audit,status='outside_not_representable_float32')
    return observed,dict(audit,status='outside',requested_excess=excess,multiplier=multiplier,
        maximum_normalized_excess=float((np.maximum(delta,0.).astype(float)/width).max()))


def augment(cases,bank,*,seed):
    """Add boundary examples; preserve all legacy cases and their stage labels."""
    from . import fusion_answers as fa
    result=[];audit=[]
    for index,case in enumerate(cases):
        if not case['injected']:continue
        truth=np.asarray(case['targets']);positive=np.flatnonzero(truth[:,0]>0)
        if not len(positive):continue
        center=int(positive[0]);sensor=case['sensor_columns'].index(case['centers'][center])
        base=np.asarray(case.get('original_current',case['current']),np.float32)
        record=bank.get(tuple(case['reference_key']))
        if record is None:continue
        level=int(truth[center,2]);rng=np.random.default_rng(seed+index*7919)
        width=max(2,min(len(base),round(len(base)*rng.uniform(.25,.875))))
        if case['kind'] in ('drift','variance'):width=min(len(base),max(4,width))
        start=int(rng.integers(0,len(base)-width+1))
        observed,info=variant(base,record['lower'],record['upper'],sensor=sensor,kind=case['kind'],
            level=level,seed=seed+index,start=start,end=start+width,direction=1 if index%2==0 else -1)
        audit.append(dict(step=case['step'],center_sensor=case['centers'][center],**info))
        if observed is None:continue
        targets=np.zeros_like(truth);targets[center]=(1,fa.TYPE_TARGET[case['kind']],level if case['kind']=='delay' else -1)
        new=dict(case,current=base,targets=targets,sensor_update=(sensor,observed[:,sensor].copy()),
            injection=info,pair_id=case['pair_id']+'/boundary')
        result.append(new)
    return result,audit


def measured_excess(observed,lower,upper,sensor):
    """One common strength unit, including unamplified native Delay."""
    observed=np.asarray(observed,np.float32);lower=np.asarray(lower,np.float32);upper=np.asarray(upper,np.float32)
    width=np.maximum(upper[:,sensor].astype(float)-lower[:,sensor],1.)
    excess=np.maximum(np.maximum(lower[:,sensor]-observed[:,sensor],observed[:,sensor]-upper[:,sensor]),0.)
    return float(np.max(excess.astype(float)/width))


def excess_stage(value):
    if not np.isfinite(value) or value<=0:raise ValueError('A positive finite realized boundary excess is required')
    edges=np.sqrt(np.asarray(CONTROLLED_LEVELS[:-1])*np.asarray(CONTROLLED_LEVELS[1:]))
    return int(np.searchsorted(edges,value,side='right')+1)


def controlled_variant(base,lower,upper,*,sensor,kind,level,seed,start,end,direction=1):
    """Known intervention with an ordinal measured from its realized waveform."""
    if kind!='delay':
        observed,info=variant(base,lower,upper,sensor=sensor,kind=kind,level=level,seed=seed,
            start=start,end=end,direction=direction,strength_floor=1.,excess_levels=CONTROLLED_LEVELS)
    else:
        # A lag is not an amplitude change. Search bounded native lags for the
        # requested excess band; unsupported bands stay explicitly unsupported.
        from .fusion_answers import delay_trace
        base=np.asarray(base,np.float32)
        info=dict(kind=kind,requested_excess_stage=level,direction=direction,start=0,end=len(base))
        if np.any((base[:,sensor]<lower[:,sensor])|(base[:,sensor]>upper[:,sensor])):
            return None,dict(info,status='base_outside_envelope')
        observed=None;best=None
        lags=np.unique(np.rint(np.geomspace(1,max(1,len(base)-1),min(32,max(1,len(base)-1)))).astype(int))
        for lag in lags:
            if lag>=len(base):continue
            candidate,_=delay_trace(base,sensor,int(lag))
            amount=measured_excess(candidate,lower,upper,sensor)
            if amount<=0 or excess_stage(amount)!=level:continue
            key=(abs(amount-CONTROLLED_LEVELS[level-1]),int(lag))
            if best is None or key<best:observed=candidate.astype(np.float32);best=key;info['lag_samples']=int(lag)
        if observed is None:return None,dict(info,status='native_delay_stage_unreachable')
        info['status']='outside'
    if observed is None:return None,info
    amount=measured_excess(observed,lower,upper,sensor)
    actual=excess_stage(amount)
    if actual!=level:return None,dict(info,status='realized_stage_differs',realized_stage=actual)
    return observed,dict(info,recipe=CONTROLLED_RECIPE,strength_unit=CONTROLLED_UNIT,
        maximum_normalized_excess=amount,realized_stage=actual,requested_excess=CONTROLLED_LEVELS[level-1])


def generate(originals,bank,*,seed,examples_per_combination,sensor_inventory=None,split=None):
    """Cover sensor/family/stage combinations on unedited inside-center bases.

    Each combination uses independent source lineages. TRAIN extrema are fixed;
    no clipping, central trace, fabricated family, or historical ordinal labels.
    """
    import hashlib
    from . import fusion_answers as fa
    def rank(*items):return hashlib.sha256(repr(items).encode()).hexdigest()
    result=[];coverage=[]
    inventory=sensor_inventory or {c['step']:c['sensor_columns'] for c in originals}
    for step,names in sorted(inventory.items()):
        raw=[c for c in originals if c['step']==step]
        pool=[c for c in raw if tuple(c['reference_key']) in bank and list(c['sensor_columns'])==list(names)
            and np.shape(c['current'])==np.shape(bank[tuple(c['reference_key'])]['lower'])]
        for sensor,name in enumerate(names):
            eligible=[]
            for c in pool:
                record=bank[tuple(c['reference_key'])];base=np.asarray(c['current'],np.float32)
                if np.all((base[:,sensor]>=record['lower'][:,sensor])&(base[:,sensor]<=record['upper'][:,sensor])):eligible.append(c)
            for kind in fa.TYPE_TARGET:
                for level in range(1,5):
                    ordered=sorted(eligible,key=lambda c:rank(seed,step,name,kind,level,c['pair_id']))
                    seen=set();failures={};count=0;directions={-1:0,1:0}
                    for c in ordered:
                        if c['lineage_id'] in seen:continue
                        record=bank[tuple(c['reference_key'])];base=np.asarray(c['current'],np.float32)
                        salt=int(rank(seed,c['pair_id'],name,kind,level)[:8],16);rng=np.random.default_rng(salt)
                        width=min(len(base),max(4,round(len(base)*rng.uniform(.5,1.))))
                        start=int(rng.integers(0,len(base)-width+1))
                        direction=1 if (int(rank(seed,step,name,kind,level)[:8],16)+count)%2 else -1
                        observed,info=controlled_variant(base,record['lower'],record['upper'],sensor=sensor,
                            kind=kind,level=level,seed=salt,start=start,end=start+width,direction=direction)
                        if observed is None:
                            failures[info['status']]=failures.get(info['status'],0)+1;continue
                        inside=np.all((base>=record['lower'])&(base<=record['upper']),axis=0)
                        other=next((j for j in np.roll(np.arange(len(names)),-sensor-1) if j!=sensor and inside[j]),None)
                        ids=[sensor]+([] if other is None else [int(other)])
                        targets=np.zeros((len(ids),3),np.int64);targets[0]=(1,fa.TYPE_TARGET[kind],info['realized_stage'])
                        result.append(dict(c,current=base,original_current=base,centers=[names[j] for j in ids],
                            targets=targets,kind=kind,injected=True,sensor_update=(sensor,observed[:,sensor].copy()),
                            injection=info,pair_id=c['pair_id']+f'/known-{sensor}-{kind}-{level}'))
                        seen.add(c['lineage_id']);count+=1
                        if kind!='delay':directions[direction]+=1
                        if count>=examples_per_combination:break
                    status=('covered' if count>=examples_per_combination else
                        'missing_native_reference' if raw and not any(tuple(c['reference_key']) in bank for c in raw) else
                        'no_supported_base' if not pool else 'no_inside_base' if not eligible else 'insufficient_supported_examples')
                    coverage.append(dict(step=step,center_sensor=name,family=kind,level=level,
                        split=split or (raw[0]['split'] if raw else 'unknown'),required=examples_per_combination,requested=examples_per_combination,
                        generated=count,independent_fitting_runs=count,eligible_lineages=len({c['lineage_id'] for c in eligible}),
                        positive_direction=directions[1],negative_direction=directions[-1],
                        strength_unit=CONTROLLED_UNIT,status=status,reasons=failures))
    return result,coverage
