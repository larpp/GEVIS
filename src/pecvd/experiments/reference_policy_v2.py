"""Finite-sample empirical criteria with explicit coarse ordinal resolution.

Quantiles are observed sample order statistics, NOT certified population tails.
Repeated/zero boundaries are legitimate, not missing calibration. Numerical
tolerances prevent zero division without inventing a physical anomaly scale.
"""
from dataclasses import dataclass
import numpy as np
from . import reference_policy as legacy
from .questions import SCOPES,STATISTICS

SCHEMA='gevis.reference_policy.v2'

@dataclass(frozen=True)
class ReferenceConfig(legacy.ReferenceConfig):
    min_runs:int=8


def fit_records(records,identity,config=ReferenceConfig()):
    payload=legacy.fit_records(records,identity,config)
    payload.update(schema=SCHEMA,ordinal_resolution='empirical observed bands; tied bands may skip intermediate grades',
        population_tail_guarantee=False,zero_scale_policy='float32 operation tolerance, not an invented physical scale')
    return payload


class ReferencePolicy(legacy.ReferencePolicy):
    def __init__(self,payload):
        if payload.get('schema')!=SCHEMA: raise ValueError('Expected reference policy v2')
        super().__init__({**payload,'schema':legacy.SCHEMA})
        self.payload=payload

    def lookup(self,step,sensor,scope,direction,context=None):
        context=context or {}
        for key in legacy._keys(step,sensor,scope,context.get('cohort'),context.get('position')):
            group=self.groups.get((*key,direction))
            if group and group['n_runs']>=self.payload['config']['min_runs']:
                return group,('cohort_position' if key[-1]!='*' else 'cohort' if key[-2]!='*' else 'step_sensor_scope')
        return None,'insufficient_independent_reference_runs'

    def decide(self,features,step,sensor,scope,statistic='level',direction='either',context=None,temperature=.75):
        # Base query validation; its conservative v1 result is not used as v2 policy.
        super().decide(features,step,sensor,scope,statistic,direction,context,temperature)
        values=np.asarray(features,float)
        scope=SCOPES[int(scope)] if isinstance(scope,(int,np.integer)) else scope
        statistic=STATISTICS[int(statistic)] if isinstance(statistic,(int,np.integer)) else statistic
        slot=STATISTICS.index(statistic)
        group,fallback=self.lookup(step,sensor,scope,direction,context)
        absolute,abs_fallback=self.lookup(step,sensor,scope,'either',context)
        supported={kind:False for kind in ('BOOLEAN','CHOICE','ORDINAL')}
        labels={kind:None for kind in supported}; probabilities={kind:None for kind in supported}
        directed=float(values[slot] if direction=='increase' else -values[slot] if direction=='decrease' else abs(values[slot]))
        def softmax(scores):
            x=np.asarray(scores,float)/temperature; x=np.exp(x-x.max()); return (x/x.sum()).tolist()
        def bounds(item):
            raw=np.asarray(item['thresholds'],float)
            tolerance=32*np.finfo(np.float32).eps*np.maximum(raw[:,-1],1.)
            return np.maximum(raw,tolerance[:,None]),tolerance
        threshold=None; raw_threshold=None; edges=None; ratios=None; any_alert=None; tied=None
        if group is not None:
            effective,tolerance=bounds(group); threshold=float(effective[slot,0]); raw_threshold=float(group['thresholds'][slot][0])
            ratio=directed/threshold; supported['BOOLEAN']=True; labels['BOOLEAN']=int(directed>threshold)
            probabilities['BOOLEAN']=softmax([1-ratio,ratio-1])
        if absolute is not None:
            edges,_=bounds(absolute); ratios=np.abs(values)/edges[:,0]; maximum=float(ratios.max())
            supported['CHOICE']=supported['ORDINAL']=True
            labels['CHOICE']=int(ratios.argmax())+1 if maximum>1 else 0
            probabilities['CHOICE']=softmax(np.r_[1-maximum,ratios-1])
            grade=int((np.abs(values)[:,None]>edges).sum(1).max())
            labels['ORDINAL']=grade; probabilities['ORDINAL']=softmax(-(np.arange(5)-grade)**2)
            tied=bool(np.any(np.diff(edges,axis=1)==0)); any_alert=maximum>1
        return {'labels':labels,'probabilities':probabilities,'supported':supported,
            'requested_value':directed,'threshold':threshold,'raw_empirical_threshold':raw_threshold,
            'threshold_quantile':self.payload['config']['quantiles'][0],
            'reference_runs':group['n_runs'] if group else 0,'fallback':fallback,
            'absolute_reference_runs':absolute['n_runs'] if absolute else 0,'absolute_fallback':abs_fallback,
            'thresholds':edges.tolist() if edges is not None else None,
            'normalized_feature_scores':ratios.tolist() if ratios is not None else None,
            'boolean_any':any_alert,'ordinal_meaning':'observed-reference rarity, not physical severity or certified tail probability',
            'ordinal_tied_boundaries':tied,'tail_resolution':1/absolute['n_runs'] if absolute else None,
            'nominal_tail_resolution_supported':all(absolute['quantile_supported']) if absolute else False,
            'score_meaning':'operation display scores, not correctness or fault probability',
            'reason':None if all(supported.values()) else 'insufficient_independent_reference_runs'}

    def summary(self):
        result=super().summary()
        if not result.empty:
            result['base_quantile_supported']=result.reference_runs.ge(self.payload['config']['min_runs'])
            result['ordinal_quantiles_supported']=result.base_quantile_supported
            result['meaning']='sample support for observed/coarse bands, not population tail assurance'
        return result


def identity_for(root,encoder_root,backbone_path,config):
    identity=legacy.identity_for(root,encoder_root,backbone_path,config)
    identity['policy_v2_source_sha256']=legacy.digest_file(__file__)
    return identity

def load_policy(path,identity):
    import json
    from pathlib import Path
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('identity')!=identity: raise ValueError('Reference v2 identity mismatch; prepare a new policy path')
    return ReferencePolicy(payload)

save_policy=legacy.save_policy
cached_records=legacy.cached_records
validation_alert_rates=legacy.validation_alert_rates
