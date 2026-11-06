"""Frozen runtime evaluation; oracle metadata never enters model routing."""
import hashlib
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .questions import CRITERIA, STATISTICS, DIRECTIONS, SCOPES, legacy_named_question_bank
from .injection import Injection, inject
from .evidence import reuse_window_inputs, pointwise_evidence


def binary_metrics(target, prediction):
    y, p = np.asarray(target, bool), np.asarray(prediction, bool)
    if y.shape != p.shape or not y.size: raise ValueError('Matching nonempty labels required')
    tp, fp, fn, tn = (int(v.sum()) for v in (y&p, ~y&p, y&~p, ~y&~p))
    ratio = lambda a,b: float(a/b) if b else None
    return {'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn, 'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn), 'false_positive_rate':ratio(fp,fp+tn), 'f1':ratio(2*tp,2*tp+fp+fn)}


def interval_iou(a, b):
    lo, hi = max(a[0],b[0]), min(a[1],b[1])
    intersection = max(0,hi-lo)
    union = max(a[1],b[1])-min(a[0],b[0])
    return intersection/union if union else 0.


def class_metrics_from_confusion(confusion):
    """Pool counts before class metrics; never average per-Step F1 values."""
    matrix=np.asarray(confusion,dtype=np.int64)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or (matrix<0).any() or matrix.sum()<1:
        raise ValueError('Nonnegative square confusion with observed cases required')
    support=matrix.sum(1); predicted=matrix.sum(0); tp=matrix.diagonal(); present=support>0
    return {'questions':int(matrix.sum()),'answer_accuracy':float(tp.sum()/matrix.sum()),
            'balanced_accuracy_present_classes':float((tp[present]/support[present]).mean()),
            'macro_f1_present_classes':float((2*tp[present]/(support[present]+predicted[present])).mean())}


def source_identity(root):
    """Code identity only. Confidential paths/outputs are never auto-exported."""
    root = Path(root)
    paths = list((root/'src/pecvd/experiments').glob('*.py')) + [root/'notebooks/STEP_3.1_run_gevis_inference.ipynb']
    import json
    result={}
    for path in paths:
        if not path.is_file():continue
        if path.suffix=='.ipynb':
            doc=json.loads(path.read_text(encoding='utf-8'))
            payload=json.dumps([(c.get('id'),c['cell_type'],c.get('source',[])) for c in doc['cells']],ensure_ascii=False,sort_keys=True).encode('utf-8')
        else:payload=path.read_bytes()
        result[path.relative_to(root).as_posix()]=hashlib.sha256(payload).hexdigest()
    return result


def ask_batched(runtime, current, reference, step, questions, *, batch_size=32, progress=None,routing_mode=None):
    """Same public runtime path, bounded batches, and no hidden fallback."""
    if batch_size < 1 or not questions:
        raise ValueError('Positive batch size and nonempty questions required')
    module=runtime.module
    answers=[]
    with reuse_window_inputs():
        for start in range(0,len(questions),batch_size):
            chunk=questions[start:start+batch_size]
            options={} if routing_mode is None else {'routing_mode':routing_mode}
            batch=module.ask_gevis(current,reference,step,chunk,module.ReferenceInfo('held-out aligned reference',1),**options)
            if len(batch)!=len(chunk): raise RuntimeError('Runtime returned an incomplete answer batch')
            answers.extend(batch)
            if progress is not None: progress(min(start+batch_size,len(questions)),len(questions))
    return answers


def evaluate_questions(runtime, current, reference, step, *, limit=120, include_unsupported=True,
                       batch_size=32, progress=None,routing_mode=None):
    """Input rows stay private. Returns local detailed rows; does not write files."""
    from .questions import unsupported_bank
    module = runtime.module
    if getattr(module,'GEVIS_CHECKPOINT',{}).get('schema')=='gevis.backbone.generic.v10':
        raise ValueError('Named-target accuracy is undefined for generic v10; use run_discovery_experiment')
    meta = runtime.encoders[step][1]
    bank = legacy_named_question_bank(step, meta['sensor_columns'], limit=limit)
    if include_unsupported:
        bank += unsupported_bank(step, meta['sensor_columns'][0])
    a,_ = module.prepare_window(current,meta); b,_ = module.prepare_window(reference,meta)
    with torch.no_grad():
        encoder = runtime.encoders[step][0].to(module.DEVICE).eval()
        state = module.numeric_state(encoder,a.to(module.DEVICE),b.to(module.DEVICE))
        features = module.semantic_comparison_features(a.to(module.DEVICE),b.to(module.DEVICE),state,state.shape[1]//a.shape[2])
    records = []
    started=time.perf_counter()
    answers=ask_batched(runtime,current,reference,step,
        [module.GevisQuestion(case.text,case.answer_type,CRITERIA[case.answer_type]) for case in bank],
        batch_size=batch_size,progress=progress,routing_mode=routing_mode)
    amortized_ms=(time.perf_counter()-started)*1000/len(bank)
    for case,answer in zip(bank,answers):
        predicted = answer['predicted_query']
        # Independent question contract selects the oracle sensor and slots only for scoring.
        with torch.no_grad():
            selected = features[:,SCOPES.index(case.time_scope),meta['sensor_columns'].index(case.sensor)]
            probabilities,_ = module.gevis_fast.structured_answer_probabilities(selected,
                torch.tensor([STATISTICS.index(case.statistic)],device=module.DEVICE),
                torch.tensor([DIRECTIONS.index(case.direction)],device=module.DEVICE),temperature=.75)
            oracle = CRITERIA[case.answer_type][int(probabilities[case.answer_type][0].argmax())]
        from .reference_policy import runtime_decision
        criterion=runtime_decision(module,selected[0],step,meta['sensor_columns'].index(case.sensor),
            SCOPES.index(case.time_scope),STATISTICS.index(case.statistic),DIRECTIONS.index(case.direction),current)
        reference_supported=True if criterion is None else criterion['supported'][case.answer_type]
        if criterion is not None:
            label=criterion['labels'][case.answer_type]
            oracle=CRITERIA[case.answer_type][label] if label is not None else None
        reference_paths={}
        if criterion is not None and case.supported:
            # All paths use the SAME empirical policy. Oracle metadata is for
            # scoring/ablation only; it never enters ask_batched/model routing.
            reference_paths={'oracle':oracle,'full_gevis':answer['candidate_answer']}
            for path,sensor,scope,stat,direction in (
                ('semantic_only',case.sensor,case.time_scope,predicted['statistic'],predicted['direction']),
                ('grounding_only',predicted['sensor'],predicted['time_scope'],case.statistic,case.direction)):
                si=meta['sensor_columns'].index(sensor); ti=SCOPES.index(scope)
                choice=runtime_decision(module,features[0,ti,si],step,si,ti,
                    STATISTICS.index(stat),DIRECTIONS.index(direction),current)
                value=choice['labels'][case.answer_type]
                reference_paths[path]=CRITERIA[case.answer_type][value] if value is not None else None
        records.append({'case_id':case.case_id, 'family':case.family, 'answer_type':case.answer_type,
            'reference_paths':reference_paths,
            'supported':case.supported, 'accepted':bool(answer['accepted']),
            'sensor_correct':predicted['sensor']==case.sensor, 'scope_correct':predicted['time_scope']==case.time_scope,
            'statistic_correct':predicted['statistic']==case.statistic if case.answer_type=='BOOLEAN' else None,
            'direction_correct':predicted['direction']==case.direction if case.answer_type=='BOOLEAN' else None,
            'reference_supported':reference_supported,
            'predicted_criterion_supported':answer.get('reference_criterion',{}).get('supported',{}).get(case.answer_type,True) if answer.get('reference_criterion') else True,
            'reference_reason':criterion.get('reason') if criterion is not None else None,
            'reference_runs':criterion.get('reference_runs') if criterion is not None else None,
            'ordinal_tied_boundaries':criterion.get('ordinal_tied_boundaries') if criterion is not None else None,
            'answer_correct':answer['candidate_answer']==oracle if case.supported and reference_supported else None,
            'candidate':answer['candidate_answer'], 'oracle_proxy':oracle if case.supported else None,
            'answer_scores':answer['probabilities'], 'semantic_scores':answer.get('semantic_scores',{}),
            'sensor_weights':answer.get('sensor_weights',{}), 'latency_ms':amortized_ms,
            'question':case.text,'expected_sensor':case.sensor,'predicted_sensor':predicted['sensor'],
            'expected_scope':case.time_scope,'predicted_scope':predicted['time_scope'],
            'warnings':answer.get('coverage_warnings',[]),
            'gate_checks':answer.get('gate_checks',{}),
            'routing':answer.get('routing',{'mode':'learned','audit_available':False}),
            'joint_query_correct':predicted['sensor']==case.sensor and predicted['time_scope']==case.time_scope
                and (case.answer_type!='BOOLEAN' or (predicted['statistic']==case.statistic and predicted['direction']==case.direction)),
            'end_to_end_correct':bool(answer['accepted']) and answer['candidate_answer']==oracle if case.supported and reference_supported else None})
    return records


def summarize_questions(records):
    frame = pd.DataFrame(records)
    if frame.empty: raise ValueError('No model answers to summarize')
    rows=[]
    for kind, group in frame[frame.supported].groupby('answer_type'):
        accepted=group[group.accepted]
        pair_keys=[key for key in ('step','pair_id','family') if key in group]
        paired=group.groupby(pair_keys).filter(lambda g:len(g)>1)
        consistency = paired.groupby(pair_keys).candidate.nunique().eq(1).mean() if len(paired) else np.nan
        rows.append({'answer_type':kind,'questions':len(group),'coverage':group.accepted.mean(),
            'reference_coverage':float(group.reference_supported.mean()) if 'reference_supported' in group else 1.,
            'scored_reference_questions':int(group.answer_correct.notna().sum()),
            'candidate_proxy_accuracy':group.answer_correct.astype(float).mean(),
            'accepted_proxy_accuracy':accepted.answer_correct.astype(float).mean() if len(accepted) else np.nan,
            'sensor_accuracy':group.sensor_correct.mean(),'scope_accuracy':group.scope_correct.mean(),
            'statistic_accuracy':pd.to_numeric(group.statistic_correct).mean(),
            'direction_accuracy':pd.to_numeric(group.direction_correct).mean(),'paraphrase_consistency':consistency,
            'joint_query_accuracy':group.joint_query_correct.mean() if 'joint_query_correct' in group else np.nan,
            'grounded_accepted_accuracy':float((group.accepted & group.answer_correct.eq(True) & group.joint_query_correct).mean()) if 'joint_query_correct' in group else np.nan,
            'accepted_wrong_query_share':float((~accepted.joint_query_correct).mean()) if len(accepted) and 'joint_query_correct' in group else np.nan,
            'end_to_end_accuracy':group.end_to_end_correct.astype(float).mean() if 'end_to_end_correct' in group else np.nan,
            'mean_ms_per_question':group.latency_ms.mean() if 'latency_ms' in group else np.nan})
    unsupported=frame[~frame.supported]
    return pd.DataFrame(rows), {'unsupported_questions':len(unsupported),
        'unsupported_rejection_rate':float((~unsupported.accepted).mean()) if len(unsupported) else None,
        'meaning':'proxy answer reproduction and question interpretation, not fault accuracy'}


def summarize_reference_paths(records):
    """Pool counts, keep unsupported targets explicit, count abstentions as misses.

    Candidate operation agreement is separate from final runtime acceptance.
    These are reference-rule targets, not independently labelled faults.
    """
    frame=pd.DataFrame(records); rows=[]
    if frame.empty or 'reference_paths' not in frame: return pd.DataFrame()
    frame=frame[frame.supported & frame.reference_paths.map(bool)]
    for kind,group in frame.groupby('answer_type',sort=False):
        eligible=group[group.reference_supported]
        for path in ('oracle','semantic_only','grounding_only','full_gevis'):
            labels=list(CRITERIA[kind]); matrix=np.zeros((len(labels),len(labels)+1),dtype=int)
            for r in eligible.itertuples():
                prediction=r.reference_paths[path]
                matrix[labels.index(r.oracle_proxy),labels.index(prediction) if prediction is not None else -1]+=1
            support=matrix.sum(1); predicted=matrix[:,:-1].sum(0); hits=np.diag(matrix[:,:-1]); present=support>0
            n=int(support.sum()); abstained=int(matrix[:,-1].sum())
            rows.append({'execution_path':path,'answer_type':kind,'questions':len(group),
                'reference_supported_questions':n,'reference_coverage':n/len(group),
                'predicted_criterion_abstentions':abstained,
                'answer_accuracy':float(hits.sum()/n) if n else np.nan,
                'balanced_accuracy_present_classes':float((hits[present]/support[present]).mean()) if n else np.nan,
                'macro_f1_present_classes':float((2*hits[present]/(support[present]+predicted[present])).mean()) if n else np.nan,
                'confusion_with_abstention':matrix.tolist(),
                'meaning':'reference-rule candidate agreement; abstentions count as misses; not fault accuracy'})
    return pd.DataFrame(rows)


def runtime_scan(runtime, current, reference, step, *, max_questions=180,batch_size=32,progress=None):
    """Ask each sensor/operation/scope rather than a single representative sensor.

    This is a bounded window scan. Evidence intervals are requested scopes, not
    precise detected event boundaries. Attention weights are not probabilities.
    """
    module=runtime.module; meta=runtime.encoders[step][1]
    required=len(meta['sensor_columns'])*len(STATISTICS)*len(SCOPES)
    if required>max_questions: raise ValueError(f'Window scan needs {required} calls; increase max_questions explicitly')
    from .questions import make_question
    requests=[]
    for sensor in meta['sensor_columns']:
        for scope in SCOPES:
            for stat in STATISTICS:
                text=make_question(meta.get('step_descriptor',step),meta.get('sensor_descriptors',{}).get(sensor,sensor),statistic=stat,scope=scope)
                requests.append((sensor,scope,stat,module.GevisQuestion(text,'BOOLEAN',CRITERIA['BOOLEAN'])))
    answers=ask_batched(runtime,current,reference,step,[r[3] for r in requests],batch_size=batch_size,progress=progress)
    result=[]
    for (sensor,scope,stat,_),answer in zip(requests,answers):
        result.append({'requested_sensor':sensor,'requested_statistic':stat,'requested_scope':scope,
            'accepted':answer['accepted'],'alert':answer['accepted'] and answer['answer']==CRITERIA['BOOLEAN'][1],
            'candidate_alert':answer['candidate_answer']==CRITERIA['BOOLEAN'][1],
            'score':answer['probabilities'][CRITERIA['BOOLEAN'][1]],
            'sensor':answer['evidence']['sensor'],'interval':answer['evidence']['scope_rows'],
            'warnings':answer.get('coverage_warnings',[]),
            'pointwise_evidence':answer['evidence'].get('pointwise',{}),
            'predicted_scope':answer['predicted_query']['time_scope'],
            'predicted_statistic':answer['predicted_query']['statistic']})
    return result


def intervention_cases(current,sensors,scale,*,kinds=('level_shift','drift','spike','variance','stuck','delay','relationship'),
                       strengths=(.5,2.),seed=42,max_cases=7,target_sensor=None):
    """Seeded kinds/strengths and intervals; optionally fix a target before generation.

    A fixed target never moves to a different active channel. Ineffective flat
    interventions remain declared ineffective, independently of model outputs.
    """
    if not kinds or not strengths or max_cases<1: raise ValueError('Nonempty intervention settings required')
    if target_sensor is not None and target_sensor not in sensors:
        raise ValueError('Known preselected intervention sensor required')
    original=current[sensors].to_numpy(float)
    yield None,current.copy(),None
    length=len(current)
    rng=np.random.default_rng(seed)
    kind_order=rng.permutation(len(kinds))
    strength_orders=[rng.permutation(len(strengths)) for _ in kinds]
    sensor_order=rng.permutation(len(sensors))
    for i in range(min(max_cases,len(kinds)*len(strengths))):
        kind_id=int(kind_order[i%len(kinds)]); kind=kinds[kind_id]
        strength=strengths[int(strength_orders[kind_id][i//len(kinds)])]
        duration=int(rng.integers(min(2,length),length+1))
        start=int(rng.integers(length-duration+1));end=start+duration
        if i and i%len(sensors)==0:sensor_order=rng.permutation(len(sensors))
        sensor=int(sensor_order[i%len(sensors)])
        if target_sensor is not None:
            sensor=sensors.index(target_sensor)
        elif kind in ('stuck','delay','relationship'):
            # Fixed generation rule, independent of predictions: do not select
            # an already constant sensor if an actually varying one is available.
            activity=np.ptp(original[start:end],axis=0)/np.asarray(scale)
            eligible=np.flatnonzero(activity>1e-6)
            if len(eligible): sensor=int(np.random.default_rng(seed+i).choice(eligible))
        spec=Injection(kind,sensor,start,end,strength,seed+i)
        changed,truth=inject(original,spec,scale)
        active=np.flatnonzero(truth['changed_mask'].any(axis=1))
        truth['event_interval']=(int(active[0]),int(active[-1])+1) if len(active) else None
        # Replace columns in float64 before assignment. Pandas dtype warnings
        # otherwise interpolate the private sensor values into stderr.
        frame=current.astype({sensor:'float64' for sensor in sensors})
        frame.loc[:,sensors]=changed
        yield spec,frame,truth


def evaluate_injections(runtime,current,reference,step,scale,*,kinds=('level_shift','drift','spike','variance'),
                        strengths=(.5,2.),seed=42,max_cases=8,max_questions=180,batch_size=32,progress=None):
    meta=runtime.encoders[step][1]; sensors=meta['sensor_columns']; length=meta['window_length']
    base=current.sort_values(runtime.module.ORDER_COLUMN).tail(length).copy()
    scenarios=list(intervention_cases(base,sensors,scale,kinds=kinds,strengths=strengths,seed=seed,max_cases=max_cases))
    rows=[]; control_scan=None
    for index,(spec,frame,truth) in enumerate(scenarios):
        started=time.perf_counter()
        scan=runtime_scan(runtime,frame,reference,step,max_questions=max_questions,batch_size=batch_size)
        alerts=[r for r in scan if r['alert']]
        top=max(scan,key=lambda r:r['score'])
        if spec is None: control_scan=scan
        diagnostics={}
        if spec is not None and truth['effective']:
            m=runtime.module
            a,_=m.prepare_window(frame,meta); b,_=m.prepare_window(reference,meta)
            with torch.no_grad():
                state=m.numeric_state(runtime.encoders[step][0].to(m.DEVICE).eval(),a.to(m.DEVICE),b.to(m.DEVICE))
                features=m.semantic_comparison_features(a.to(m.DEVICE),b.to(m.DEVICE),state,state.shape[1]//a.shape[2])
            scope=0 if spec.start==0 and spec.end==len(base) else 1 if spec.start==0 else 2
            stat={'level_shift':0,'drift':2,'spike':3,'variance':1}.get(spec.kind)
            target_values=features[0,scope,spec.sensor]
            oracle_value=float(target_values[stat].abs()) if stat is not None else float(target_values.abs().max())
            oracle=oracle_value>1.
            from .reference_policy import runtime_decision
            decision=runtime_decision(m,target_values,step,spec.sensor,scope,stat if stat is not None else 0,2,frame)
            reference_supported=True
            if decision is not None:
                reference_supported=decision['supported']['BOOLEAN'] if stat is not None else decision['supported']['CHOICE']
                oracle=(bool(decision['labels']['BOOLEAN']) if stat is not None else decision['boolean_any']) if reference_supported else None
            target_sensor=sensors[spec.sensor]
            target_rows=[r for r in scan if r['sensor']==target_sensor and interval_iou(r['interval'],truth['event_interval'])>0]
            before_rows=[r for r in control_scan if r['sensor']==target_sensor and interval_iou(r['interval'],truth['event_interval'])>0]
            target_candidate=any(r['candidate_alert'] for r in target_rows)
            target_accepted=any(r['alert'] for r in target_rows)
            blocked=[r for r in target_rows if r['candidate_alert'] and not r['accepted']]
            warning_counts={name:sum(name in r.get('warnings',[]) for r in blocked)
                for name in ('low_answer_confidence','low_semantic_confidence','low_sensor_grounding_confidence','unsupported_question_or_criteria')}
            raw=pointwise_evidence(a,b,spec.sensor,scope)
            if decision is not None:
                from .reference_policy import policy_pointwise_evidence
                raw=policy_pointwise_evidence(a,b,spec.sensor,scope,decision)
            learned={}
            if hasattr(m,'GEVIS_MODEL') and hasattr(m,'numeric_token_descriptions'):
                from pecvd.gevis_backbone import evidence_questions
                patches=state.shape[1]//len(sensors)
                with torch.no_grad():
                    fusion=m.GEVIS_MODEL.to(m.DEVICE).eval()(state,evidence_questions(meta.get('step_descriptor',step),1),
                        ['BOOLEAN'],[CRITERIA['BOOLEAN']],m.numeric_token_descriptions(step,meta,patches,1),return_fusion=True)
                weights=fusion[2].reshape(len(sensors),patches)
                learned_sensor=int(weights.sum(-1).argmax())
                learned_patch=int(weights[learned_sensor].argmax())
                learned={'learned_evidence_sensor_correct':learned_sensor==spec.sensor,
                         'learned_evidence_patch_hits_event':learned_sensor==spec.sensor and
                            interval_iou((learned_patch*len(base)//patches,(learned_patch+1)*len(base)//patches),truth['event_interval'])>0}
            # Rank change of the SAME requested query score, not saturated
            # post-injection score alone. This is still a routing diagnostic.
            deltas={sensor:max((r['score']-before['score'] for r,before in zip(scan,control_scan)
                               if r['requested_sensor']==sensor),default=0.) for sensor in sensors}
            delta_sensor=max(deltas,key=deltas.get)
            stage=('reference_unsupported' if not reference_supported else 'no_declared_operation_response' if not oracle else 'routing_or_operation_mismatch' if not target_candidate
                   else 'withheld_by_runtime' if not target_accepted else 'target_alert')
            diagnostics={'oracle_operation_value':oracle_value,'oracle_declared_alert':oracle,'reference_supported':reference_supported,
                'target_candidate_alert':target_candidate,'target_accepted_alert':target_accepted,
                'control_target_alert':any(r['alert'] for r in before_rows),
                'new_target_alert':target_accepted and not any(r['alert'] for r in before_rows),
                'diagnostic_stage':stage,'gate_warning_counts':warning_counts,
                'target_peak_train_std':raw['peak_absolute_train_std'],
                'pointwise_target_iou':max((interval_iou(interval,truth['event_interval']) for interval in raw['exceedance_intervals']),default=0.) if raw.get('criterion_supported',True) else None,
                'top_delta_sensor_correct':delta_sensor==target_sensor,**learned}
        rows.append({'kind':spec.kind if spec else 'untouched','strength':spec.strength if spec else 0.,
            'injected':spec is not None,'effective':truth['effective'] if truth else None,
            'alert':bool(alerts),'alert_count':len(alerts),'max_operation_score':top['score'],
            'candidate_alert':any(row['candidate_alert'] for row in scan),
            'accepted_question_share':float(np.mean([row['accepted'] for row in scan])),
            'questions':len(scan),'latency_ms':(time.perf_counter()-started)*1000,
            'top_sensor_correct':top['sensor']==sensors[spec.sensor] if spec else None,
            'best_alert_interval_iou':max((interval_iou(r['interval'],truth['event_interval']) for r in alerts
                                          if r['sensor']==sensors[spec.sensor]),default=0.) if spec and truth['effective'] else None,
            **diagnostics,
            'meaning':'intervention detection; untouched alert rate is not verified-normal FPR'})
        if progress is not None: progress(index+1,len(scenarios))
    return pd.DataFrame(rows)


def load_cached_pairs(runtime, step, *, split='test', limit=2,seed=42):
    """Explicit artifact-only loader, run separation + cohort/position matching.

    Existing cached windows are already standardized; convert to original units
    for the same prepare_window runtime path. Never opens a source CSV.
    """
    if split not in ('validation','test') or not 1<=limit<=32: raise ValueError('Bounded held-out split required')
    if hasattr(runtime,'external_pairs'):
        pairs=runtime.external_pairs[step][split][:limit]
        if not pairs: raise ValueError('No independent external evaluation pairs')
        return pairs
    module=runtime.module; meta=runtime.encoders[step][1]
    path=module.ENCODER_ROOT/'windows'/f'{step}.npz'
    with np.load(path,allow_pickle=False) as z:
        ids={s:set(map(str,z[f'{s}_process_id'])) for s in ('train','validation','test')}
        if any(ids[a]&ids[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
            raise ValueError('Process overlap across splits')
        x=z[split]; process=z[f'{split}_process_id']; pos=z[f'{split}_position_bin']; cohort=z[f'{split}_cohort_key']
        result=[]; seen=set()
        order=np.random.default_rng(seed).permutation(len(x))
        # Prefer distinct Process IDs before adding extra windows from a run.
        unique=[]; extra=[]
        for i in order:
            if str(process[i]) not in seen: unique.append(i); seen.add(str(process[i]))
            else: extra.append(i)
        for i in unique+extra:
            match=np.flatnonzero((process!=process[i])&(pos==pos[i])&(cohort==cohort[i]))[:16]
            if not len(match): continue
            pair=[]
            for values in (x[i],np.median(x[match],axis=0)):
                frame=pd.DataFrame(values*np.asarray(meta['std'])+np.asarray(meta['mean']),columns=meta['sensor_columns'])
                frame[module.ORDER_COLUMN]=np.arange(len(frame))
                frame.attrs['evaluation_pair_id']=hashlib.sha256(f'{split}/{step}/{i}'.encode()).hexdigest()[:16]
                frame.attrs['evaluation_run_id']=hashlib.sha256(str(process[i]).encode()).hexdigest()[:16]
                frame.attrs['position_bin']=int(pos[i])
                frame.attrs['cohort_key']=str(cohort[i])
                pair.append(frame)
            result.append(tuple(pair))
            if len(result)==limit: break
    if not result: raise ValueError('No aligned cross-process held-out pairs')
    return result


def load_experiment_runtime(root, *, process_config=None, backbone_path=None):
    """No data read: runtime loads only configured local trained artifacts."""
    from pecvd.gevis_ui import load_gevis_runtime, notebook_runtime_sources, GevisRuntime
    from pecvd.process_config import load_process_config
    root=Path(root)
    if process_config is None and backbone_path is not None:
        raise ValueError('An explicit backbone also requires an explicit process configuration')
    from .process_inputs import configured_process_path
    config=load_process_config(process_config or configured_process_path(root))
    from pecvd.model_variants import active_config
    config=active_config(config)
    if process_config is None:
        runtime=load_gevis_runtime(root,root/'notebooks/STEP_3.1_run_gevis_inference.ipynb',root/config.data_path)
    else:
        if backbone_path is None or not Path(backbone_path).is_file():
            raise ValueError('External process inference requires matching trained backbone_path')
        from types import ModuleType
        from contextlib import redirect_stdout
        from io import StringIO
        import sys
        sources=notebook_runtime_sources(root/'notebooks/STEP_3.1_run_gevis_inference.ipynb')
        module=ModuleType('_gevis_external_experiment_runtime')
        module.__file__=str(root/'notebooks/STEP_3.1_run_gevis_inference.ipynb')
        sys.modules[module.__name__]=module
        with redirect_stdout(StringIO()):
            exec(compile(sources['03-types'],'external:types','exec'),module.__dict__)
            module.ROOT=root; module.PROCESS_CONFIG=config; module.STEPS=config.step_specs
            module.ENCODER_ARTIFACT_VERSION=config.artifacts.encoder
            module.ENCODER_ROOT=root/'artifacts'/config.artifacts.encoder
            module.RUN_COLUMN=config.columns.run; module.STEP_COLUMN=config.columns.step; module.ORDER_COLUMN=config.columns.order
            module.DATA_PATH=root/config.data_path
            module.EXPERIMENT_BACKBONE_PATH=Path(backbone_path).resolve()
            exec(compile(sources['03-load'],'external:load','exec'),module.__dict__)
            exec(compile(sources['03-function'],'external:function','exec'),module.__dict__)
        runtime=GevisRuntime(module)
    runtime.module.GEVIS_MODEL.requires_grad_(False).eval()
    return runtime
