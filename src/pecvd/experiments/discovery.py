"""Automatic change discovery inside a selected range, not named-query routing.

No injected event metadata or expected sensor/time is an input. Existing marginal
reference quantiles are retained; full-window false-alarm control is NOT implied.
"""
import time
import numpy as np
import torch
from .questions import CRITERIA,DISCOVERY_QUESTIONS,SCOPES,STATISTICS,sensor_choice
from .reference_policy import context_from_rows,policy_features,policy_pointwise_evidence
from .evidence import prepare_inputs

CONTRACT='gevis.learned_generic_discovery.v2'


def contract():
    return {'schema':CONTRACT,'analysis_input':'UI-selected sensors/scopes; defaults to all',
        'decision':'generic-question Cross-Attention ranks sensors; calibrated numerical evidence decides change/no-change',
        'question_target_metadata_used':False,'new_training':True,
        'localization':'pointwise intervals when supported; scope is otherwise only coarse',
        'step2_cross_attention_executed':True,
        'threshold_family':'existing marginal reference quantiles; not window FPR controlled',
        'dedicated_unavailable_operations':['delay','relationship','stuck'],
        'meaning':'unusual reference-relative difference, not confirmed equipment fault'}


def _learned_generic_attention(module,state,step,meta,patch_count,center_sensor=None):
    """Return sensor/patch probabilities without placing a sensor in the question."""
    from pecvd.gevis_backbone import evidence_questions
    if not hasattr(module,'GEVIS_MODEL') or not hasattr(module,'numeric_token_descriptions'):
        raise RuntimeError('Generic discovery requires the retrained STEP 2 Cross-Attention artifact')
    sensors=list(meta['sensor_columns'])
    descriptions=module.numeric_token_descriptions(step,meta,patch_count,1)
    if center_sensor is not None:
        from .questions import center_descriptions
        descriptions=[center_descriptions(row,sensors,patch_count,center_sensor) for row in descriptions]
    prompt=evidence_questions(meta.get('step_descriptor',step),1)
    output=module.GEVIS_MODEL.to(module.DEVICE).eval()(state,prompt,
        ['BOOLEAN'],[CRITERIA['BOOLEAN']],descriptions,return_fusion=True)
    attention=output[2] if isinstance(output,(tuple,list)) and len(output)>2 else output[1]
    weights=attention.reshape(len(sensors),patch_count).clamp_min(0)
    if not torch.isfinite(weights).all() or weights.sum()<=0:
        raise ValueError('Invalid generic discovery attention')
    sensor_mass=weights.sum(-1);sensor_prob=sensor_mass/sensor_mass.sum().clamp_min(1e-12)
    patch_prob=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    return sensor_prob.detach().cpu().numpy(),patch_prob.detach().cpu().numpy()


def _rank_with_attention(events,candidates,sensors,sensor_prob,patch_prob):
    for row in candidates:
        sid=sensors.index(row['sensor'])
        row['learned_sensor_probability']=float(sensor_prob[sid])
        row['learned_patch_index']=int(patch_prob[sid].argmax())
        row['localization_basis']='sensor-free generic-question Cross-Attention, constrained by calibrated numerical evidence'
    events.sort(key=lambda row:(-row['learned_sensor_probability'],-row['normalized_score']))
    candidates.sort(key=lambda row:(-row['learned_sensor_probability'],-row['normalized_score']))


def _sensor_answer(all_sensors, selected, sensor_prob, accepted, changed):
    # UI selection bounds the answer vocabulary, never the expected target.
    return sensor_choice(selected, [sensor_prob[list(all_sensors).index(s)] for s in selected],
                         accepted=accepted, changed=changed)


@torch.no_grad()
def review_changes(module,current_rows,reference_rows,step,reference_info,*,sensors=None,scopes=None,center_sensor=None):
    if getattr(module,'DECISION_SYSTEM','hybrid')=='fusion_answers':
        from .fusion_answers import review
        if scopes is not None and list(scopes)!=['whole']:raise ValueError('Fusion answers address the selected input window')
        if sensors is not None and list(sensors)!=[center_sensor]:raise ValueError('Select one analysis center; retain all context inputs')
        return review(module,current_rows,reference_rows,step,reference_info,center_sensor=center_sensor)
    from .window_policy import MODE
    if center_sensor is not None:
        if sensors is not None:
            raise ValueError('Center analysis uses all context sensors; do not also restrict sensors')
        return review_center_changes(module,current_rows,reference_rows,step,reference_info,
                                     center_sensor=center_sensor,scopes=scopes)
    if getattr(module,'DISCOVERY_POLICY_MODE',None)==MODE:
        return review_window_changes(module,current_rows,reference_rows,step,reference_info,sensors=sensors,scopes=scopes)
    if getattr(module,'DECISION_POLICY_MODE',None)!='reference_quantiles_v2':
        raise ValueError('Automatic change discovery requires prepared reference_quantiles_v2')
    if reference_info.process_count<1: raise ValueError('At least one reference process required')
    encoder,meta=module.STEP_ENCODERS[step]
    selected=list(meta['sensor_columns']) if sensors is None else list(sensors)
    selected_scopes=list(SCOPES) if scopes is None else list(scopes)
    if not selected or len(set(selected))!=len(selected) or any(s not in meta['sensor_columns'] for s in selected):
        raise ValueError('Select known unique physical sensors')
    if not selected_scopes or len(set(selected_scopes))!=len(selected_scopes) or any(s not in SCOPES for s in selected_scopes):
        raise ValueError('Select known unique scopes')
    started=time.perf_counter()
    encoder=encoder.to(module.DEVICE).eval()
    current,reference,ac,bc,state,patches,features=prepare_inputs(current_rows,reference_rows,step,
        encoder,meta,module.DEVICE,module.prepare_window,module.numeric_state,policy_features)
    sensor_prob,patch_prob=_learned_generic_attention(module,state,step,meta,patches)
    comparable=abs(ac.get('position_bin',0)-bc.get('position_bin',0))<=1
    if not comparable: raise ValueError('Incomparable reference position')
    candidates=[];missing=[];supported_groups=0
    for sensor in selected:
        sid=meta['sensor_columns'].index(sensor)
        for scope in selected_scopes:
            tid=SCOPES.index(scope);values=features[0,tid,sid].detach().cpu().numpy()
            decision=module.REFERENCE_POLICY.decide(values,step,sensor,tid,'level','either',context_from_rows(current_rows))
            if not all(decision['supported'].values()):
                missing.append({'sensor':sensor,'scope':scope,'reason':decision['reason']});continue
            ratios=np.asarray(decision['normalized_feature_scores']);supported_groups+=1
            # Same winning sensor/scope and operation define all three answers.
            evidence=policy_pointwise_evidence(current,reference,sid,tid,decision,original_scale=float(meta['std'][sid]))
            lo,hi=evidence.get('scope_rows', [0,current.shape[1]])
            if not evidence.get('scope_rows'):
                lo,hi=((0,current.shape[1]) if tid==0 else (0,max(2,current.shape[1]//3)) if tid==1 else (2*current.shape[1]//3,current.shape[1]))
            intervals=evidence.get('exceedance_intervals',[])
            peak=evidence.get('peak_row')
            detected=next((list(iv) for iv in intervals if iv[0]<=peak<iv[1]),None) if peak is not None else None
            for op in range(len(STATISTICS)):
                boundaries=np.asarray(decision['thresholds'])[op]
                grade=int((abs(values[op])>boundaries).sum())
                score=float(ratios[op]);changed=score>1.
                candidates.append({'sensor':sensor,'time_scope':scope,'scope_rows':[int(lo),int(hi)],
                    'detected_rows':detected,
                    'statistic':STATISTICS[op],'statistic_index':op,'signed_value':float(values[op]),
                    'threshold':float(boundaries[0]),'normalized_score':score,'changed':changed,
                    'ordinal_index':grade,'pointwise':evidence,
                    'localization_basis':'observed pointwise difference against reference peak criterion; not learned attention',
                    'reference_runs':decision['absolute_reference_runs'],
                    'ordinal_tied_boundaries':decision['ordinal_tied_boundaries']})
    events=[c for c in candidates if c['changed']]
    _rank_with_attention(events,candidates,list(meta['sensor_columns']),sensor_prob,patch_prob)
    complete=not missing
    # A supported exceedance establishes yes, but incomplete coverage cannot establish no.
    accepted=bool(events) or complete
    changed=bool(events) if accepted else None
    winner=events[0] if events else None
    labels={'BOOLEAN':int(changed) if accepted else None,
            'CHOICE':winner['statistic_index']+1 if winner else 0 if accepted else None,
            'ORDINAL':winner['ordinal_index'] if winner else 0 if accepted else None}
    answers=[]
    strongest=max(candidates,key=lambda row:row['normalized_score']) if candidates else None
    maximum=strongest['normalized_score'] if strongest else 0.
    def softmax(values):
        values=np.asarray(values,float)/.75;values=np.exp(values-values.max());return values/values.sum()
    for kind in CRITERIA:
        idx=labels[kind]
        scores=(softmax([1-maximum,maximum-1]) if kind=='BOOLEAN' else
                softmax(-abs(np.arange(len(CRITERIA[kind]))-(idx or 0))))
        answers.append({'answer_type':kind,'answer':CRITERIA[kind][idx] if idx is not None else None,
            'candidate_answer':CRITERIA[kind][idx] if idx is not None else None,'accepted':accepted,
            'question':DISCOVERY_QUESTIONS[kind],
            'probabilities':dict(zip(CRITERIA[kind],map(float,scores))),
            'confidence':{'answer':float(scores.max())},
            'coverage_warnings':['deterministic_evidence_display_scores_not_fault_probabilities']+(['incomplete_reference_coverage'] if missing else []),
            'evidence':winner,'decision_source':CONTRACT,
            'score_meaning':'deterministic shared reference evidence; no correctness/fault probability'})
    answers.append(_sensor_answer(meta['sensor_columns'], selected, sensor_prob, accepted, changed))
    if isinstance(module.DEVICE,torch.device) and module.DEVICE.type=='cuda':torch.cuda.synchronize(module.DEVICE)
    return {'contract':CONTRACT,'step':step,'accepted':accepted,'changed':changed,
        'attention_top_sensor':meta['sensor_columns'][int(sensor_prob.argmax())],
        'attention_top_patch':int(patch_prob[int(sensor_prob.argmax())].argmax()),
        'complete_coverage':complete,'criterion_coverage':supported_groups/(len(selected)*len(selected_scopes)),
        'selected_sensors':selected,'selected_scopes':selected_scopes,'winner':winner,'events':events,
        'ranked_evidence':candidates,'missing_criteria':missing,'answers':answers,
        'warnings':(['incomplete_reference_coverage'] if missing else [])+['marginal_quantiles_not_window_false_alarm_control'],
        'latency_ms':(time.perf_counter()-started)*1000,
        'meaning':contract()['meaning']}


@torch.no_grad()
def review_window_changes(module,current_rows,reference_rows,step,reference_info,*,sensors=None,scopes=None,center_sensor=None):
    """One calibrated event controls all typed answers and displayed evidence."""
    from .window_policy import SCHEMA,CHOICES
    from .discovery_calibration import residual_trace
    started=time.perf_counter();policy=getattr(module,'DISCOVERY_POLICY',None)
    if reference_info.process_count<1:raise ValueError('At least one reference process required')
    if policy is None:raise RuntimeError('Window discovery criteria missing. Run the new STEP 3 window-evidence preparation cell; no STEP 1/2 retraining is needed.')
    encoder,meta=module.STEP_ENCODERS[step];encoder.to(module.DEVICE).eval()
    a,b,ac,bc,state,patches,_=prepare_inputs(current_rows,reference_rows,step,encoder,meta,module.DEVICE,
        module.prepare_window,module.numeric_state,policy_features)
    sensor_prob,patch_prob=_learned_generic_attention(module,state,step,meta,patches,center_sensor=center_sensor)
    if abs(ac.get('position_bin',0)-bc.get('position_bin',0))>1:raise ValueError('Incomparable reference position')
    def array(value):return value.detach().cpu().numpy()[0]
    gate_kwargs={'center_sensor':center_sensor} if getattr(getattr(policy,'config',None),'gate_scope','full_window')=='center_views_v3' else {}
    if getattr(getattr(policy,'config',None),'gate_scope',None)=='position_rank_v4':
        cohort=current_rows.attrs.get('cohort_key');position=current_rows.attrs.get('position_bin')
        if cohort!=reference_rows.attrs.get('cohort_key') or position!=reference_rows.attrs.get('position_bin'):
            raise ValueError('Position-rank inference requires exactly matched current/reference contexts')
        gate_kwargs=dict(center_sensor=center_sensor,cohort=cohort,position=position)
    result=policy.analyze(array(a),array(b),residual_trace(state,len(meta['sensor_columns']),a.shape[1]),
        step,meta['sensor_columns'],selected=sensors,scopes=scopes,**gate_kwargs)
    _rank_with_attention(result['events'],result['ranked_evidence'],list(meta['sensor_columns']),sensor_prob,patch_prob)
    result['winner']=result['events'][0] if result['accepted'] and result['events'] else None
    from .temporal_localization import apply_localization
    apply_localization(module,result,state,list(meta['sensor_columns']),a.shape[1],
        latent_width=4*encoder.d_model if hasattr(encoder,'d_model') else None)
    winner=result['winner'];accepted=result['accepted'];changed=result['changed'];calibration=result['calibration']
    labels={'BOOLEAN':int(changed) if accepted else None,
            'CHOICE':winner['statistic_index']+1 if winner else 0 if accepted else None,
            'ORDINAL':winner['ordinal_index'] if winner else 0 if accepted else None}
    warnings=['empirical_control_rarity_not_fault_probability','localization_conditional_not_pointwise_fpr_control']
    if reference_rows.attrs.get('reference_split')!='train':warnings.append('runtime_reference_pool_not_verified_as_train_protocol')
    if not accepted:warnings.append('insufficient_independent_calibration_runs')
    if calibration['insufficient_controls_for_nominal_rate']:warnings.append('insufficient_controls_for_nominal_window_rate')
    answers=[]
    choices=CRITERIA['CHOICE'] if getattr(policy.config,'operation_profile','all')=='core' else CHOICES
    for kind,options in {**CRITERIA,'CHOICE':choices}.items():
        index=labels[kind];values=-abs(np.arange(len(options))-(index or 0)).astype(float)
        values=np.exp(values-values.max());scores=values/values.sum()
        answers.append({'answer_type':kind,'answer':options[index] if index is not None else None,
            'candidate_answer':options[index] if index is not None else None,'accepted':accepted,
            'question':DISCOVERY_QUESTIONS[kind],
            'probabilities':dict(zip(options,map(float,scores))),'confidence':{'answer':float(scores.max())},
            'coverage_warnings':list(warnings),'evidence':winner,'decision_source':SCHEMA,
            'score_meaning':'deterministic display scores; not correctness/fault probabilities'})
    selected=list(meta['sensor_columns']) if sensors is None else list(sensors)
    answers.append(_sensor_answer(meta['sensor_columns'], selected, sensor_prob, accepted, changed))
    return {**result,'contract':SCHEMA,'step':step,'answers':answers,
        'attention_sensor_scores':dict(zip(meta['sensor_columns'],map(float,sensor_prob))),
        'attention_top_sensor':meta['sensor_columns'][int(sensor_prob.argmax())],
        'attention_top_patch':int(patch_prob[int(sensor_prob.argmax())].argmax()),
        'selected_sensors':list(meta['sensor_columns']) if sensors is None else list(sensors),
        'selected_scopes':list(SCOPES) if scopes is None else list(scopes),
        'complete_coverage':accepted,'criterion_coverage':float(accepted),
        'missing_criteria':[] if accepted else [{'reason':'insufficient_independent_calibration_runs'}],
        'warnings':warnings,'latency_ms':(time.perf_counter()-started)*1000,
        'threshold_source':policy.payload['threshold_source'],
        'meaning':'Window-calibrated reference-relative difference, not equipment fault; intervals use operation-specific traces'}


@torch.no_grad()
def review_center_changes(module,current_rows,reference_rows,step,reference_info,*,center_sensor,scopes=None):
    """One UI anchor, two simultaneous views; preserve full numeric context.

    Center is an input selection, never an injected label. Full-scan thresholds
    remain conservative for this subset; neither relation nor lag proves cause.
    """
    from .window_policy import MODE
    if getattr(module,'DISCOVERY_POLICY_MODE',None)!=MODE:
        raise ValueError('Center and relationship analysis requires the calibrated window policy')
    names=list(module.STEP_ENCODERS[step][1]['sensor_columns'])
    if center_sensor not in names:raise ValueError('Select a known center sensor')
    from .center_decision import selected_candidate
    learned=(None if getattr(getattr(getattr(module,'DISCOVERY_POLICY',None),'config',None),'gate_scope',None)=='position_rank_v4'
             else selected_candidate(module))
    if learned is not None:
        return learned.review(module,current_rows,reference_rows,step,reference_info,
                              center_sensor=center_sensor,scopes=scopes)
    result=review_window_changes(module,current_rows,reference_rows,step,reference_info,
                                 sensors=[center_sensor],scopes=scopes,center_sensor=center_sensor)
    def view(relation):
        candidates=[r for r in result['ranked_evidence'] if (r['statistic']=='relationship')==relation]
        supported=result['accepted'] and (not relation or (len(names)>1 and 'relationship' not in result.get('disabled_operations',[])))
        events=[r for r in candidates if r['changed']] if supported else []
        winner=max(events,key=lambda r:r['normalized_score']) if events else None
        return {'accepted':bool(supported),'changed':bool(events) if supported else None,
                'unsupported_reason':('operation_disabled_by_profile' if relation and 'relationship' in result.get('disabled_operations',[]) else None),
                'winner':winner,'events':events,
                'related_sensors':sorted({s for r in events for s,v in r.get('related_sensor_scores',{}).items() if v>r['window_threshold']}),
                'meaning':'symmetric relationship change, not cause' if relation else 'selected sensor versus aligned reference cycles'}
    own=view(False);relation=view(True)
    result.update(center_sensor=center_sensor,self_analysis=own,relationship_analysis=relation,
                  context_sensors=names,analysis_mode='center_and_relationship')
    # A one-sensor vocabulary would trivially score 100%. Report partners from
    # actual relation evidence instead, not the center itself or fault odds.
    scores={}
    for row in result['ranked_evidence']:
        if row['statistic']=='relationship':
            for partner,value in row.get('related_sensor_scores',{}).items():
                if partner!=center_sensor and value>0:
                    scores[partner]=max(scores.get(partner,0.),value)
    scores={s:v*result['attention_sensor_scores'][s] for s,v in scores.items()}
    total=sum(scores.values())
    choice=result['answers'][-1]
    probabilities={s:v/total for s,v in scores.items()} if total>0 and relation['accepted'] else {}
    chosen=max(probabilities,key=probabilities.get) if probabilities else None
    choice.update(answer=chosen,candidate_answer=chosen,probabilities=probabilities,
                  accepted=bool(probabilities),confidence={'answer':max(probabilities.values()) if probabilities else None},
                  reference_only=relation['changed'] is not True,
                  score_meaning='attention weighted by center-pair evidence; not calibrated probability or cause')
    summary={'center_sensor':center_sensor,'self_changed':own['changed'],
             'relationship_changed':relation['changed'],'related_sensors':relation['related_sensors'],
             'self_type':own['winner']['statistic'] if own['winner'] else None,
             'self_interval':own['winner']['detected_rows'] if own['winner'] else None,
             'relationship_interval':relation['winner']['detected_rows'] if relation['winner'] else None}
    for answer in result['answers']:answer['center_analysis']=summary
    return result
