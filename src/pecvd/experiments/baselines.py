"""Optional libraries loaded only on explicit calls; no automatic installation."""
import numpy as np
from .evaluation import binary_metrics


def statistical_scores(values, reference, scale):
    x,b,s=np.asarray(values),np.asarray(reference),np.asarray(scale)
    if x.shape!=b.shape or x.ndim!=2 or s.shape!=(x.shape[1],) or (s<=0).any():
        raise ValueError('Aligned time/sensor arrays and TRAIN scale required')
    return np.max(np.abs(x-b)/s,axis=1)


def merlion_scores(train, test, *, seed=42):
    """Actual Merlion IsolationForest, trained only on TRAIN in a local worker."""
    from .merlion_backend import worker_scores
    return worker_scores(train,test,seed=seed)


def merlion_train_array(runtime,step,*,max_windows=16):
    """Read only the existing TRAIN cache; never fit Merlion on validation/test."""
    meta=runtime.encoders[step][1]; sensors=meta['sensor_columns']
    with np.load(runtime.module.ENCODER_ROOT/'windows'/f'{step}.npz',allow_pickle=False) as z:
        normalized=z['train'][:max_windows]
    if normalized.ndim!=3 or normalized.shape[-1]!=len(sensors) or not len(normalized):
        raise ValueError('Invalid Merlion TRAIN window cache')
    return normalized.reshape(-1,len(sensors))*np.asarray(meta['std'])+np.asarray(meta['mean'])


def cached_train_windows(runtime,step,limit=32):
    with np.load(runtime.module.ENCODER_ROOT/'windows'/f'{step}.npz',allow_pickle=False) as z:
        return z['train'][:limit].copy()


def tsaug_drift(values, *, seed=42):
    from tsaug import Drift
    # Augmentation only; this does not create trusted fault labels.
    return Drift(max_drift=.2,seed=seed).augment(np.asarray(values)[None])[0]


def gutentag_dataset(seed=42):
    from gutenTAG import GutenTAG
    generator=GutenTAG(seed=seed)
    generator.load_config_dict({'timeseries':[{'name':'synthetic_fixture','length':256,
        'base-oscillations':[{'kind':'sine'}],
        'anomalies':[{'length':24,'exact-position':160,'channel':0,
                      'kinds':[{'kind':'mean','offset':2.}]}]}]})
    return generator.generate(return_timeseries=True)[0].timeseries


def timeeval_metrics(labels, scores):
    from timeeval.metrics import RocAUC, PrAUC
    y,s=np.asarray(labels,dtype=int),np.asarray(scores,dtype=float)
    if len(np.unique(y))!=2: raise ValueError('Both classes required for AUC evaluation')
    return {'roc_auc':float(RocAUC()(y,s)), 'pr_auc':float(PrAUC()(y,s))}


def fit_threshold(validation_scores, *, quantile=.99):
    scores=np.asarray(validation_scores,float)
    if scores.size<2 or not np.isfinite(scores).all() or not 0<quantile<1: raise ValueError('Finite validation scores required')
    return float(np.quantile(scores,quantile))


def compare_baselines(train, validation, test, reference, scale, labels, *, use_merlion=False, use_timeeval=False):
    """Same timestamps/truth; select thresholds on untouched validation only."""
    stats=statistical_scores(test,reference,scale)
    # Reference for validation is an independent TRAIN typical level.
    val_ref=np.broadcast_to(np.median(train,axis=0),validation.shape)
    scores={'statistics':(statistical_scores(validation,val_ref,scale),stats)}
    if use_merlion:
        combined=merlion_scores(train,np.concatenate([validation,test],axis=0))
        scores['Merlion_IsolationForest']=(combined[:len(validation)],combined[len(validation):])
    rows=[]
    for name,(val,score) in scores.items():
        threshold=fit_threshold(val)
        row={'method':name,'threshold':threshold,**binary_metrics(labels,score>threshold)}
        if use_timeeval: row.update(timeeval_metrics(labels,score))
        rows.append(row)
    return rows


def compare_frozen_runtime(runtime, step, *, validation_limit=3, test_limit=2,
                           strength=2., max_questions=400, use_merlion=False, seed=42,
                           max_cases=4,batch_size=32,progress=None,use_temporal=False,temporal_epochs=10,
                           use_learned_answers=False):
    """Paired window-level comparison on identical untouched/injected inputs.

    Unchanged windows are controls, not certified normal. Threshold selection
    uses validation controls only; the original 1.0 runtime policy remains intact.
    """
    import pandas as pd
    import torch
    from .evaluation import load_cached_pairs, runtime_scan,intervention_cases
    m=runtime.module; encoder,meta=runtime.encoders[step]
    generic=getattr(m,'GEVIS_CHECKPOINT',{}).get('schema')=='gevis.backbone.generic.v10'
    if getattr(m,'DISCOVERY_POLICY_MODE',None)=='window_calibrated_v2' and not use_learned_answers:
        if getattr(m,'DISCOVERY_POLICY',None) is None:raise RuntimeError('Run STEP 3 window-evidence preparation before EXP 3')
        from .discovery_calibration import load_discovery_pairs
        load_cached_pairs=load_discovery_pairs
    val=load_cached_pairs(runtime,step,split='validation',limit=validation_limit,seed=seed)
    test=load_cached_pairs(runtime,step,split='test',limit=test_limit,seed=seed)
    sensors=meta['sensor_columns']; scale=np.asarray(meta['std'])
    cases=[]
    if len(val)<2: raise ValueError('At least two validation controls needed to select a threshold')
    for split,pairs in (('validation',val),('test',test)):
        for index,(current,reference) in enumerate(pairs):
            if split=='validation': cases.append((split,False,current,reference,'untouched',True))
            else:
                for spec,frame,truth in intervention_cases(current,sensors,scale,
                    kinds=('level_shift','drift','spike','variance'),strengths=(strength,),max_cases=max_cases,seed=seed+index):
                    cases.append((split,spec is not None,frame,reference,spec.kind if spec else 'untouched',
                                  truth['effective'] if truth else True))
    details=[]; timing=[]
    for case_index,(split,injected,current,reference,kind,effective) in enumerate(cases):
        import time
        started=time.perf_counter()
        a,_=m.prepare_window(current,meta); b,_=m.prepare_window(reference,meta)
        with torch.no_grad():
            state=m.numeric_state(encoder.to(m.DEVICE).eval(),a.to(m.DEVICE),b.to(m.DEVICE))
            features=m.semantic_comparison_features(a.to(m.DEVICE),b.to(m.DEVICE),state,state.shape[1]//a.shape[2])
        feature_ms=(time.perf_counter()-started)*1000
        started=time.perf_counter()
        scan=[] if generic else runtime_scan(runtime,current,reference,step,max_questions=max_questions,batch_size=batch_size)
        if str(m.DEVICE).startswith('cuda'): torch.cuda.synchronize()
        scan_ms=(time.perf_counter()-started)*1000
        details.append({'split':split,'injected':injected,
            'kind':kind,'effective':effective,
            'pair_id':current.attrs.get('evaluation_pair_id',str(case_index)),
            'statistics':float(features[...,:4].abs().max()),
            'PatchTST_residual_difference':float(features[...,-1].abs().max()),
            'GEVIS_candidate_score':max((row['score'] for row in scan),default=float('nan')),
            'pointwise_max_difference':float((a-b).abs().max()),
            'shared_feature_ms':feature_ms,'gevis_scan_ms':scan_ms,
            'GEVIS_accepted_alert':any(row['alert'] for row in scan)})
        if use_learned_answers:
            from .reporting import predict_window_answers
            # Same complete sensor search and exact windows as other baselines;
            # intervention labels never select the center or question.
            started=time.perf_counter()
            answers=predict_window_answers(runtime,current,reference,step,centers=sensors,
                question_variants=[('BOOLEAN','BOOLEAN','Is there a change in this interval?')])
            if str(m.DEVICE).startswith('cuda'):torch.cuda.synchronize()
            details[-1]['GEVIS_learned_answer_ms']=(time.perf_counter()-started)*1000
            details[-1]['GEVIS_learned_answer']=max(row['change_probability'] for row in answers)
        numeric_ready=(getattr(m,'DISCOVERY_POLICY_MODE',None)!='window_calibrated_v2'
                       or getattr(m,'DISCOVERY_POLICY',None) is not None)
        if getattr(m,'DECISION_POLICY_MODE',None)=='reference_quantiles_v2' and numeric_ready:
            from .discovery import review_changes
            discovery=review_changes(m,current,reference,step,m.ReferenceInfo('same paired comparison',1))
            details[-1].update(GEVIS_discovery_alert=discovery['changed'] is True,
                GEVIS_discovery_accepted=discovery['accepted'],GEVIS_discovery_ms=discovery['latency_ms'],
                GEVIS_discovery_threshold_source=discovery.get('threshold_source','TRAIN marginal quantiles; NOT matched window FPR'))
        if progress is not None: progress(case_index+1,len(cases))
    methods=['statistics','PatchTST_residual_difference']+([] if generic else ['GEVIS_candidate_score'])
    if use_learned_answers:methods.append('GEVIS_learned_answer')
    if use_temporal:
        from .temporal_baseline import temporal_scores
        train_windows=cached_train_windows(runtime,step)
        test_windows=np.stack([(current[sensors].to_numpy()-np.asarray(meta['mean']))/scale for _,_,current,_,_,_ in cases])
        scores,fit=temporal_scores(train_windows,test_windows,seed=seed,epochs=temporal_epochs)
        timing.append({'method':'LSTM_AE','step':step,**fit})
        for row,score in zip(details,scores): row['LSTM_AE']=float(score)
        methods.extend(['LSTM_AE','pointwise_max_difference'])
    if use_merlion:
        # Fixed TRAIN cap prevents large implicit fits; existing model preprocessing is reused.
        train=merlion_train_array(runtime,step)
        combined=np.concatenate([current[sensors].to_numpy() for _,_,current,_,_,_ in cases])
        started=time.perf_counter()
        score=merlion_scores(train,combined,seed=seed)
        timing.append({'method':'Merlion_IsolationForest','step':step,
            'pipeline_total_ms':(time.perf_counter()-started)*1000,
            'timing_mode':'worker startup + TRAIN fit + batched scoring; not warm inference',
            'scored_windows':len(cases),'fit_split':'train'})
        cursor=0
        for row,(_,_,current,_,_,_) in zip(details,cases):
            row['Merlion_IsolationForest']=float(score[cursor:cursor+len(current)].max()); cursor+=len(current)
        methods.append('Merlion_IsolationForest')
    frame=pd.DataFrame(details); valid=frame[frame.split=='validation']; held=frame[(frame.split=='test') & frame.effective]
    rows=[]
    for name in methods:
        threshold=fit_threshold(valid[name].to_numpy())
        prediction=held[name]>threshold
        frame[f'prediction::{name}']=frame[name]>threshold
        rows.append({'method':name,'threshold_source':'validation controls; window-level',
            'validation_windows':len(valid),'test_windows':len(held),
            'threshold':threshold,**binary_metrics(held.injected,prediction)})
        if name=='GEVIS_learned_answer':
            rows[-1].update(score_source='trained_answer_decoder_softmax; max over all sensors',
                            metric_target='synthetic_added_change_vs_unlabelled_untouched')
    if not generic:
        rows.append({'method':'GEVIS_accepted_policy','threshold_source':'existing runtime gates',
            'threshold':None,'validation_windows':len(valid),'test_windows':len(held),
            **binary_metrics(held.injected,held.GEVIS_accepted_alert)})
    if 'GEVIS_discovery_alert' in held:
        rows.append({'method':'GEVIS_shared_evidence_discovery','threshold_source':held.GEVIS_discovery_threshold_source.iloc[0],
            'analysis_mode':'blind','comparison_task':'full sensor search; distinct from UI-centered evaluation',
            'threshold':None,'validation_windows':len(valid),'test_windows':len(held),
            'prediction_abstentions':int((~held.GEVIS_discovery_accepted).sum()),
            'answer_coverage':float(held.GEVIS_discovery_accepted.mean()),
            **binary_metrics(held.injected,held.GEVIS_discovery_alert)})
    result=pd.DataFrame(rows)
    operating=[]
    for nominal_rate in (.10,.05,.01):
        for name in methods:
            threshold=fit_threshold(valid[name].to_numpy(),quantile=1-nominal_rate)
            operating.append({'method':name,'nominal_control_alert_rate':nominal_rate,
                'threshold':threshold,'validation_windows':len(valid),'test_windows':len(held),
                'validation_alert_rate':float((valid[name]>threshold).mean()),
                'control_rate_resolution':1/len(valid),
                'insufficient_controls_for_rate':len(valid)<1/nominal_rate,
                **binary_metrics(held.injected,held[name]>threshold)})
    result.attrs['operating_points']=pd.DataFrame(operating)
    result.attrs['temporal_fit']=pd.DataFrame(timing)
    return result,frame
