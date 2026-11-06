"""TRAIN fitting and disjoint Validation comparison for learned center decisions.

No Test values, external APIs, data exports or existing artifact writes. Save
only a candidate that passes the declared control/recall rule, with exact identity.
"""
from dataclasses import asdict
from pathlib import Path
import copy, hashlib, os, time
import numpy as np
import pandas as pd
import torch
from .center_decision import (CenterConfig,CenterDecision,encode_features,targets_for_center,
                              candidate_path,artifact_identity,save_candidate,load_candidate)
from .optimization_models import PairedEncoderTrial,split_runs
from .optimization_heads import GroundedOutputTrial,TYPE_LABELS,compatible_type_targets
from .optimization_data import restrict_training_references
from .evaluation import intervention_cases,binary_metrics,interval_iou


def adoption_rule(candidate,baseline):
    """Counts on identical cases, including combined self/pair untouched alerts."""
    if set(candidate)!=set(baseline):raise ValueError('Identical Steps required for adoption')
    no_more_alerts=all(candidate[s]['controls']<=baseline[s]['controls'] for s in baseline)
    no_recall_regression=all(candidate[s]['hits']>=baseline[s]['hits'] for s in baseline)
    no_partner_regression=all(candidate[s].get('partner_hits',0)>=baseline[s].get('partner_hits',0) for s in baseline)
    no_interval_regression=sum(v.get('iou_sum',0.) for v in candidate.values())>=sum(v.get('iou_sum',0.) for v in baseline.values())
    no_type_regression=sum(v.get('type_hits',0) for v in candidate.values())>=sum(v.get('type_hits',0) for v in baseline.values())
    more_hits=sum(v['hits'] for v in candidate.values())>sum(v['hits'] for v in baseline.values())
    return {'accepted':bool(no_more_alerts and no_recall_regression and more_hits and
                           no_partner_regression and no_interval_regression and no_type_regression),
        'no_control_alert_increase_per_step':bool(no_more_alerts),
        'no_self_recall_regression_per_step':bool(no_recall_regression),
        'strict_self_detection_gain':bool(more_hits),'selection_split':'exploratory Validation evaluation subset',
        'no_partner_recovery_regression':bool(no_partner_regression),
        'no_interval_iou_regression':bool(no_interval_regression),'no_mechanism_accuracy_regression':bool(no_type_regression),
        'verified_normal_controls':False,'test_used':False}


def collect_center_rows(runtime,config):
    """Fold all TRAIN runs BEFORE bounding sampled head inputs.

    References may use any fitting-fold TRAIN run, not just the few current
    windows sampled for SGD. Selection runs never enter fitting references.
    Training-only eligibility is declared; held-out evaluation is never pruned.
    """
    from .discovery_calibration import cached_comparisons
    module=runtime.module;all_ids=set()
    for step in runtime.encoders:
        with np.load(module.ENCODER_ROOT/'windows'/f'{step}.npz',allow_pickle=False) as cache:
            all_ids.update(map(str,cache['train_process_id']))
    selection,fit=split_runs([{'run_id':r} for r in sorted(all_ids)],.25,config.seed+1)
    selection_ids={e['run_id'] for e in selection};fit_ids={e['run_id'] for e in fit}
    prepared={};excluded=0
    for step in runtime.encoders:
        prepared[step]={}
        with np.load(module.ENCODER_ROOT/'windows'/f'{step}.npz',allow_pickle=False) as cache:
            for split,limit in (('train',config.train_runs_per_step),('validation',config.validation_runs_per_step)):
                rows=[];seen=set()
                for row in cached_comparisons(module.ENCODER_ROOT,step,split=split,config=module.DISCOVERY_POLICY.config):
                    if row['run_id'] in seen:continue
                    if split=='train':
                        try:
                            row=restrict_training_references([row],cache['train'],cache['train_process_id'],
                                cache['train_cohort_key'],cache['train_position_bin'],fit_ids,
                                max_references=module.DISCOVERY_POLICY.config.max_references)[0]
                        except ValueError as error:
                            if 'No fitting-run reference support' not in str(error):raise
                            excluded+=1;continue
                        row['head_partition']='selection' if row['run_id'] in selection_ids else 'fit'
                    rows.append(dict(row,step=step));seen.add(row['run_id'])
                    if len(seen)==limit:break
                if len(seen)!=limit:raise ValueError('Insufficient independent matched runs for declared center protocol')
                prepared[step][split]=rows
    train_runs={r['run_id'] for group in prepared.values() for r in group['train']}
    validation_runs={r['run_id'] for group in prepared.values() for r in group['validation']}
    if train_runs&validation_runs:raise ValueError('TRAIN/Validation run leakage')
    return prepared,{'reference_pool':'all fitting-fold TRAIN runs, selection excluded',
        'train_windows_without_fitting_reference':excluded,'test_values_read':False,
        'original_train_preprocessing_frozen':True}


def _examples(module,prepared,config):
    train=[];validation=[]
    for step,groups in prepared.items():
        sensors=list(module.STEP_ENCODERS[step][1]['sensor_columns'])
        for split,rows in groups.items():
            for index,row in enumerate(rows):
                # Stable center chosen before any intervention or prediction.
                seed=int(hashlib.sha256((str(config.seed)+row['run_id']+step).encode()).hexdigest()[:8],16)
                center=int(np.random.default_rng(seed).integers(len(sensors)))
                designs=[('self',center)]
                if len(sensors)>1:designs.append(('context',(center+1)%len(sensors)))
                current=pd.DataFrame(row['current'],columns=sensors)
                for task,target in designs:
                    for spec,frame,truth in intervention_cases(current,sensors,np.ones(len(sensors)),
                            target_sensor=sensors[target],seed=seed,max_cases=config.max_cases):
                        if task=='context' and spec is None:continue
                        observed=frame[sensors].to_numpy(dtype=float)
                        a=observed.astype('float32');started=time.perf_counter()
                        feature=encode_features(module,a,row['reference'],step,center,config)
                        feature_ms=(time.perf_counter()-started)*1000
                        label,patch=targets_for_center(row['current'],a,truth,spec,center,config.patches)
                        types=np.zeros((len(sensors),len(TYPE_LABELS)),dtype=bool);types[:,0]=True
                        if label is not None:
                            types[label]=compatible_type_targets(row['current'],observed,spec,np.ones(len(sensors)))
                        example={**row,'features':feature,'target':label,'patch_targets':patch,'type_targets':types,
                            'center':center,'task':task,'spec':spec,'truth':truth,'observed':a,'feature_ms':feature_ms}
                        (train if split=='train' else validation).append(example)
                if (index+1)%8==0:print(f'  center inputs: {split}, {index+1}/{len(rows)} windows',flush=True)
    return train,validation


def calibrate_head(detector,examples):
    """Run maxima of learned scores. Old numerical threshold is never a gate."""
    maxima={};self_maxima={};pair_maxima={}
    for e in examples:
        if e['spec'] is not None:continue
        scores=detector.scores(e['features'],e['step']);center=e.get('center',0)
        own=float(scores[center]);partners=np.delete(scores,center)
        pair=float(partners.max()) if len(partners) else own
        value=max(own,pair)
        key=e['step'],e['run_id'];maxima[key]=max(maxima.get(key,value),value)
        self_maxima[key]=max(self_maxima.get(key,own),own)
        pair_maxima[key]=max(pair_maxima.get(key,pair),pair)
    result={}
    for q in (.95,.99):
        groups={}
        for step in {e['step'] for e in examples}:
            values=np.sort([v for (s,_),v in maxima.items() if s==step]);n=len(values)
            if n<4:raise ValueError('At least four independent calibration controls per Step required')
            rank=math_ceil((n+1)*q);threshold=float(values[min(rank,n)-1])
            bands=np.maximum.accumulate([threshold,*[float(np.quantile(values,v,method='higher')) for v in (.975,.99,1.)]])
            # Two distinct tasks have distinct score distributions. Reserve half
            # the nominal tail budget for each, then audit their combined alerts.
            # Sparse controls still resolve neither advertised population rate.
            view_rank=math_ceil((n+1)*(1-(1-q)/2))
            own=np.sort([v for (s,_),v in self_maxima.items() if s==step])
            pair=np.sort([v for (s,_),v in pair_maxima.items() if s==step])
            own_threshold=float(own[min(view_rank,n)-1]);pair_threshold=float(pair[min(view_rank,n)-1])
            own_bands=np.maximum.accumulate([own_threshold,*[float(np.quantile(own,v,method='higher')) for v in (.975,.99,1.)]])
            pair_bands=np.maximum.accumulate([pair_threshold,*[float(np.quantile(pair,v,method='higher')) for v in (.975,.99,1.)]])
            groups[step]={'threshold':threshold,'ordinal_edges':bands.tolist(),'runs':n,
                          'self_threshold':own_threshold,'pair_threshold':pair_threshold,
                          'self_ordinal_edges':own_bands.tolist(),'pair_ordinal_edges':pair_bands.tolist(),
                          'tail_unresolved':view_rank>n,'control_resolution':1/n,
                          'calibration':'separate self/pair run maxima; half tail budget each; observed combined alert audit'}
        result[str(q)]=groups
    return result


def math_ceil(value):
    # Kept explicit for observed order statistics, not a Gaussian Z-score.
    return int(np.ceil(value))


def _frames(module,example):
    meta=module.STEP_ENCODERS[example['step']][1];frames=[]
    for values in (example['observed'],example['reference']):
        frame=pd.DataFrame(values*np.asarray(meta['std'])+np.asarray(meta['mean']),columns=meta['sensor_columns'])
        frame[module.ORDER_COLUMN]=np.arange(len(values))
        frame.attrs.update(cohort_key=example['cohort'],position_bin=example['position'],reference_split='train')
        frames.append(frame)
    return frames


def compare_candidate(module,head,evaluation):
    """Same prepared examples, same center, no truth passed to either runtime."""
    from .discovery import review_changes
    rows=[];counts={'candidate':{},'baseline':{}};q=.99
    prior=os.environ.pop('GEVIS_CENTER_DECISION_PATH',None)
    try:
        for e in evaluation:
            names=module.STEP_ENCODERS[e['step']][1]['sensor_columns'];center=e['center']
            current,reference=_frames(module,e)
            baseline=review_changes(module,current,reference,e['step'],module.ReferenceInfo('TRAIN reference',1),center_sensor=names[center])
            candidate=head.review(module,current,reference,e['step'],module.ReferenceInfo('TRAIN reference',1),center_sensor=names[center])
            self_alert=candidate['self_analysis']['changed'] is True
            relation_alert=candidate['relationship_analysis']['changed'] is True
            spec=e['spec'];effective=bool(e['truth'] and e['truth']['effective'])
            own_winner=candidate['self_analysis']['winner']
            interval=own_winner['detected_rows'] if own_winner else None
            truth_interval=e['truth']['event_interval'] if effective else None
            old_winner=baseline['self_analysis']['winner']
            old_interval=old_winner['detected_rows'] if old_winner else None
            baseline_iou=interval_iou(old_interval,truth_interval) if old_interval and truth_interval else 0.
            partner=candidate['answers'][-1]['answer']
            row={'step':e['step'],'task':e['task'],'kind':spec.kind if spec else 'untouched',
                 'strength':spec.strength if spec else None,'injected':spec is not None,'effective':effective,
                 'candidate_self_alert':self_alert,'baseline_self_alert':baseline['self_analysis']['changed'] is True,
                 'candidate_combined_alert':self_alert or relation_alert,'baseline_combined_alert':baseline['changed'] is True,
                 'candidate_relation_alert':relation_alert,'baseline_relation_alert':baseline['relationship_analysis']['changed'] is True,
                 'candidate_partner_top1':partner==names[spec.sensor] if spec and e['task']=='context' else None,
                 'baseline_partner_top1':baseline['answers'][-1]['answer']==names[spec.sensor] if spec and e['task']=='context' else None,
                 'candidate_interval_iou':interval_iou(interval,truth_interval) if interval and truth_interval else 0.,
                 'baseline_interval_iou':baseline_iou,
                 'candidate_mechanism':own_winner['predicted_mechanism'] if own_winner else TYPE_LABELS[0],
                 'mechanism_ambiguous':bool(spec and e['type_targets'][e['target']].sum()>1) if e['target'] is not None else False,
                 'candidate_ms':candidate['latency_ms'],
                 'baseline_ms':baseline['latency_ms']}
            row['observed_pair_change']=e['target'] is not None if e['task']=='context' else None
            row['candidate_partner_joint']=bool(relation_alert and partner==names[spec.sensor]) if spec and e['task']=='context' else None
            row['baseline_partner_joint']=bool(row['baseline_relation_alert'] and row['baseline_partner_top1']) if spec and e['task']=='context' else None
            type_mapping={'level':'level_shift','variability':'variance','slope':'drift','local_change':'spike',
                          'stuck':'stuck','delay':'delay','relationship':'relationship'}
            old_kind=type_mapping.get(old_winner['statistic']) if old_winner else TYPE_LABELS[0]
            if spec and e['task']=='self' and effective:
                allowed=e['type_targets'][center]
                row['candidate_type_correct']=bool(allowed[TYPE_LABELS.index(row['candidate_mechanism'])])
                row['baseline_type_correct']=bool(old_kind in TYPE_LABELS and allowed[TYPE_LABELS.index(old_kind)])
            else:row['candidate_type_correct']=row['baseline_type_correct']=None
            rows.append(row)
            for method in counts:
                group=counts[method].setdefault(e['step'],{'controls':0,'hits':0,'cases':0,'control_cases':0,
                    'partner_hits':0,'partner_cases':0,'iou_sum':0.,'type_hits':0})
                if spec and effective and e['task']=='context' and e['target'] is not None:
                    group['partner_cases']+=1;group['partner_hits']+=int(row[method+'_partner_joint'])
            if e['task']=='self':
                for method in counts:
                    group=counts[method][e['step']]
                    if spec is None:
                        group['control_cases']+=1;group['controls']+=int(row[method+'_combined_alert'])
                    elif effective:
                        group['cases']+=1;group['hits']+=int(row[method+'_self_alert'])
                        group['iou_sum']+=row[method+'_interval_iou'];group['type_hits']+=int(row[method+'_type_correct'])
    finally:
        if prior is not None:os.environ['GEVIS_CENTER_DECISION_PATH']=prior
    details=pd.DataFrame(rows);summary=[]
    for method in counts:
        for step,values in counts[method].items():
            summary.append({'method':method,'step':step,**values,
                'self_detection_recall':values['hits']/values['cases'] if values['cases'] else np.nan,
                'untouched_combined_alert_rate':values['controls']/values['control_cases'],
                'mean_interval_iou':values['iou_sum']/values['cases'] if values['cases'] else np.nan,
                'compatible_mechanism_accuracy':values['type_hits']/values['cases'] if values['cases'] else np.nan,
                'partner_joint_recall':values['partner_hits']/values['partner_cases'] if values['partner_cases'] else np.nan})
    return pd.DataFrame(summary),details,adoption_rule(counts['candidate'],counts['baseline'])


def train_center_decision(root,*,runtime=None,config=CenterConfig(),save_accepted=False):
    """Bounded actual training, shared-runtime validation; does not read Test values."""
    from .reporting import ready_runtime,ExperimentReport
    from .questions import localization_questions
    config.validate();started=time.perf_counter();runtime=runtime or ready_runtime(root);module=runtime.module
    if not hasattr(module,'DISCOVERY_POLICY') or module.DISCOVERY_POLICY is None:
        raise RuntimeError('Prepare STEP3 window criteria before learned-head training')
    identity=artifact_identity(module,config)
    prepared,audit=collect_center_rows(runtime,config)
    print('Frozen STEP1/STEP2 inputs + center/pair/lag features; TRAIN and Validation only.',flush=True)
    train,validation=_examples(module,prepared,config)
    fit=[e for e in train if e['head_partition']=='fit'];selection=[e for e in train if e['head_partition']=='selection']
    calibration,evaluation=split_runs(validation,.5,config.seed+2)
    if ({e['run_id'] for e in fit}&{e['run_id'] for e in selection} or
            {e['run_id'] for e in calibration}&{e['run_id'] for e in evaluation}):raise RuntimeError('Run leakage')
    with torch.no_grad():query=module.GEVIS_MODEL.text(localization_questions('process',1),module.DEVICE)[0].detach().cpu().numpy()
    detector=PairedEncoderTrial('paired_cross_attention',query,seed=config.seed,width=config.width,device=module.DEVICE)
    print('Fitting learned change/sensor head, no old numerical candidate gate...',flush=True)
    detector.fit(fit,selection,epochs=config.epochs,batch_size=config.batch_size,
                 learning_rate=config.learning_rate,balanced_selection=True)
    print('Fitting conditional time/type heads...',flush=True)
    outputs=GroundedOutputTrial(detector,seed=config.seed).fit(fit,selection,epochs=config.output_epochs,
        batch_size=config.batch_size,learning_rate=config.learning_rate)
    head=CenterDecision(detector,outputs,calibrate_head(detector,calibration),config)
    summary,details,adoption=compare_candidate(module,head,evaluation)
    if artifact_identity(module,config)!=identity:raise RuntimeError('Source/artifact changed during fitting; discard candidate')
    path=None
    if save_accepted and adoption['accepted']:
        path=candidate_path(module.BACKBONE_PATH,config)
        if path.exists():load_candidate(module,path) # Identity required, never overwrite.
        else:save_candidate(head,path,identity,adoption)
        os.environ['GEVIS_CENTER_DECISION_PATH']=str(path)
    parameters=sum(p.numel() for p in detector.model.parameters())+sum(p.numel() for p in outputs.heads.parameters())
    report=ExperimentReport('Learned center decision versus preserved numeric gate',summary,
        details=details,metadata={'status':'COMPLETED','model_evaluated':True,'adoption':adoption,
        'saved':path is not None,'weights_updated':True,'test_used':False,'config':asdict(config),
        'trainable_parameters':parameters,'elapsed_seconds':time.perf_counter()-started,
        'training_controls':'untouched and unverified; synthetic intervention discrimination',
        'selection':'TRAIN run-disjoint; Validation calibration/evaluation disjoint, exploratory adoption',
        'reference_audit':audit,'new_gate':'learned scores calibrated on Validation; not old operation thresholds'},
        diagnostics={'by_kind':details[details.injected & details.effective].groupby(['step','task','kind'],sort=False).agg(
            cases=('effective','size'),candidate_self_recall=('candidate_self_alert','mean'),
            baseline_self_recall=('baseline_self_alert','mean'),candidate_relation_alert=('candidate_relation_alert','mean'),
            candidate_partner_top1=('candidate_partner_top1','mean'),baseline_partner_top1=('baseline_partner_top1','mean'),
            candidate_interval_iou=('candidate_interval_iou','mean')).reset_index()})
    return head,report
