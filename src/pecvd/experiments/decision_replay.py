"""Incoming CSV replay for detailed domain heads; no learning or evaluator input."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from . import current_input as current, decision_training as decision
from .reporting import ExperimentReport


def availability(requests,exclusions,coverage):
    """Inference availability, distinct from answer accuracy and confidence."""
    input_table=pd.DataFrame(coverage)
    answered=sum(r['status']=='OK' and r.get('retained_after_Run_quality_check',True) for r in requests)
    invalidated=sum(r['status']=='OK' and not r.get('retained_after_Run_quality_check',True) for r in requests)
    summary=pd.DataFrame([dict(attempted_requests=len(requests),answered_requests=answered,
        uncomputed_requests=sum(r['status']!='OK' for r in requests),invalidated_requests=invalidated,
        answered_fraction=answered/len(requests) if requests else np.nan,
        all_attempted_requests_answered=bool(requests) and answered==len(requests),
        all_input_rows_observed=bool(len(input_table)) and bool(input_table.input_row_coverage.eq(1.).all()),
        confidence_abstention_enabled=False)])
    reasons=[]
    for source,items in [('model_request',[r for r in requests if r['status']!='OK']),('native_input_audit',exclusions)]:
        frame=pd.DataFrame(items)
        if not frame.empty and 'reason' in frame:
            reasons.extend(dict(source=source,reason=reason,records=int(count)) for reason,count in frame.reason.value_counts(dropna=False).items())
    return summary,pd.DataFrame(reasons,columns=['source','reason','records'])

def _replay_frame(contract,frame,boundaries,*,center_batch_size=8,progress=None,on_result=None):
    """Same native Receiver as live input; predictions precede evaluator labels."""
    from . import external_test as external,ald_delivery_evaluation as delivery,online_inference as online
    from . import peald_current_data as native
    runtime=contract.runtime;model_before=external._fingerprint(runtime)
    session=decision.Session(runtime,center_batch_size=center_batch_size)
    started=time.perf_counter()
    stamps=pd.to_datetime(frame[native.COLUMNS['order']],format='mixed',errors='coerce')
    seconds=np.where(stamps.notna(),stamps.astype('int64').to_numpy()/1e9,np.nan)
    preparation_ms=1000*(time.perf_counter()-started)
    rows=[];requests=[];exclusions=[];coverage=[];provisional=[]
    for order,boundary in enumerate(boundaries):
        rid=boundary['run_id'];start=boundary['start'];end=boundary['end'];observed=np.zeros(end-start,bool)
        receiver=native.Receiver(session,on_result=on_result)
        local=[];local_requests=[];answered=0
        def consume(replies):
            nonlocal answered
            for result in replies:
                a=result['row_start'];b=result['row_end'];step=result['step'];request_id=f'{rid}/{step}/{a}:{b}'
                local_requests.append(dict(case_id=rid,request_id=request_id,step=step,row_start=a,row_end=b,
                    status=result['status'],reason=result['reason'],answers=len(result['answers']),
                        evaluated_answers=result.get('evaluated_answers',len(result['answers'])),conditional_answering=result.get('conditional_answering',False),
                        selected_sensors=result.get('selected_sensors',0),reference_count=result.get('reference_count',0),**result['timings']))
                if result['status']!='OK':
                    exclusions.append(dict(case_id=rid,step=step,source_row_start=a,source_row_end=b,
                        rows=b-a,reason=result['reason'],unsupported_normal_sensors=len(result.get('normal_support_issues',[]))))
                if result['status']=='OK':
                    answered+=1;observed[a-start:b-start]=True
                    local.extend(dict(answer,case_id=rid,run_id=rid,source_group=result['run_id'],step=step,
                        row_start=a,row_end=b,pair_id=request_id) for answer in result['answers'])
                    if progress and (answered==1 or answered%50==0):
                        progress(f'Detailed Run {order+1}/{len(boundaries)}: {answered} windows answered',flush=True)
        part=frame.iloc[start:end]
        source_ids=part[native.COLUMNS['run']].astype('string').str.strip()
        identities=source_ids.dropna().unique()
        if source_ids.isna().any() or source_ids.eq('').any() or len(identities)!=1 or not identities[0]:
            exclusions.append(dict(case_id=rid,step=None,source_row_start=start,source_row_end=end,rows=end-start,reason='ambiguous_delivery_Run_identity'))
            coverage.append(dict(case_id=rid,run_order=order,input_start=start,input_end=end,input_rows=end-start,available_windows=0,answered_windows=0,sensor_windows=0,yes_sensor_windows=0,observed_prediction=np.nan,run_prediction=np.nan,scope_status='UNSCORED',whole_run_negative_certified=False,input_row_coverage=0.,overlap_status='not_certified_independent',preparation_ms=0.))
            continue
        for offset,values in enumerate(part.itertuples(index=False,name=None)):
            record=dict(zip(part.columns,values));record['_seconds']=float(seconds[start+offset])
            consume(receiver.push(record,row_index=start+offset))
            exclusions.extend(dict(item,case_id=rid) for item in receiver.drain_audit())
        consume(receiver.finish());exclusions.extend(dict(item,case_id=rid) for item in receiver.drain_audit())
        if receiver.invalid_run:
            provisional.extend(local);local=[];observed[:]=False;answered=0
        for request in local_requests:request['retained_after_Run_quality_check']=not receiver.invalid_run
        requests.extend(local_requests);rows.extend(local)
        boolean=[v for v in local if v['answer_type']=='BOOLEAN'];yes=sum(v['prediction']==1 for v in boolean)
        coverage.append(dict(case_id=rid,run_order=order,input_start=start,input_end=end,input_rows=end-start,
            available_windows=len(local_requests),answered_windows=answered,sensor_windows=len(boolean),yes_sensor_windows=yes,
            observed_prediction=int(yes>0) if boolean else np.nan,run_prediction=int(yes>0) if boolean else np.nan,
            scope_status='INVALID_MAIN_TIMESTAMP' if receiver.invalid_run else ('COMPLETE_SUPPORTED_ROWS' if observed.all() else ('PARTIAL_SUPPORTED_ROWS' if boolean else 'UNSCORED')),
            whole_run_negative_certified=False,input_row_coverage=float(observed.mean()),
            overlap_status='not_certified_independent_development_delivery',preparation_ms=preparation_ms if order==0 else 0.))
    if external._fingerprint(runtime)!=model_before:raise RuntimeError('Detailed weights changed during inference')
    details=pd.DataFrame(rows);latency,by_step=online.latency_tables(requests)
    available,unavailable=availability(requests,exclusions,coverage)
    return ExperimentReport('ALD detailed normal-reference sequential input',pd.DataFrame([dict(runs=len(boundaries),questions=len(rows),accuracy_available=False)]),
        details=details,diagnostics=dict(coverage=pd.DataFrame(coverage),exclusions=pd.DataFrame(exclusions),
            requests=pd.DataFrame(requests),provisional_invalidated_answers=pd.DataFrame(provisional),
            decisions=delivery.decisions(details),request_latency=latency,request_latency_by_step=by_step,
            inference_availability=available,unavailable_reasons=unavailable),
        metadata=dict(version=delivery.VERSION,input_mode=decision.INPUT_MODE,model_schema=decision.SCHEMA,normal_reference_recipe=decision.RECIPE,training_recipe=decision.RECIPE,decision_system=decision.fa.SYSTEM,
            checkpoint=str(runtime.module.FUSION_ANSWER_PATH),model_sha256=model_before,
            head_training_reference_mode='packaged_independent_TRAIN',reference_contract=decision.SCHEMA,original_used_for_prediction=False,
            run_rule=delivery.RUN_RULE,training=False,selection=False,automatic_export=False,
            independent_test_certified=False,real_fault_truth=False,confidence_abstention_enabled=False,conditional_answering=True,
            timing_scope='detailed complete-window request; excludes acquisition, native-unit validation and file reading',
            limitation='Main results are provisional until Run quality is known. Complete SubCycles validate before replies. '
                       'Boundary tail windows emit only when that boundary arrives. '
                       'Any-Yes is a diagnostic, not a trained Run probability. Offline replay retains the full input/report in memory.'))

def replay_delivery(contract,package,*,expected_runs=20,center_batch_size=8,progress=print,on_result=None):
    from . import external_test as external,ald_delivery_evaluation as delivery
    if contract.domain!='PEALD':raise ValueError('ALD delivery requires the PEALD detailed checkpoint')
    package=Path(package).resolve();identity=delivery._inputs(package);started=time.perf_counter()
    frame=current.read_frame(package/'inputs/test.csv',contract)
    boundaries=delivery.delivery_boundaries(pd.read_csv(package/'inputs/run_boundaries.csv',dtype={'run_id':str}),len(frame))
    if expected_runs is not None and len(boundaries)!=expected_runs:raise ValueError('Unexpected delivery Run count')
    read_ms=1000*(time.perf_counter()-started)
    report=_replay_frame(contract,frame,boundaries,center_batch_size=center_batch_size,progress=progress,on_result=on_result)
    if delivery._inputs(package)!=identity:raise ValueError('Delivery inputs changed during replay')
    from . import peald_current_data as native
    reservation_match=contract.runtime.prepared.identity.get('delivery_metadata_identity')==native.delivery_metadata_identity(package)[1]
    report.metadata.update(package=str(package),input_identity=identity,input_read_ms=read_ms,
        delivery_reservation_metadata_matches_training=reservation_match)
    report.metadata['prediction_sha256']=delivery._seal(report)
    return report


def pecvd_windows(frame,runtime,process,*,steps=None,centers=None,stride=None):
    """Known native counters only; a boundary tail emits at the observed boundary."""
    selected=set(runtime.encoders if steps is None else steps)
    if not selected<=set(runtime.encoders):raise ValueError('Untrained PECVD Step requested')
    run=process.columns.run;step=process.columns.step;order=process.columns.order
    if 'row_in_step' not in frame:raise ValueError('PECVD incoming data needs its native row_in_step counter')
    counter=pd.to_numeric(frame.row_in_step,errors='coerce')
    if not np.isfinite(counter).all() or counter.ne(np.floor(counter)).any():raise ValueError('Invalid native Step counter')
    cuts=frame[run].ne(frame[run].shift())|frame[step].ne(frame[step].shift())|frame.groupby(run,sort=False)[order].diff().ne(1)
    cuts|=counter.diff().ne(1)
    if 'step_number' in frame:cuts|=frame.step_number.ne(frame.step_number.shift())
    cases=[];excluded=[]
    for _,group in frame.groupby(cuts.cumsum(),sort=False):
        phase=group[step].iloc[0]
        if phase not in selected:continue
        meta=runtime.encoders[phase][1];names=list(meta['sensor_columns']);length=int(meta['window_length'])
        chosen=names if centers is None else list(centers)
        if not chosen or any(c not in names for c in chosen):raise ValueError('Unsupported PECVD center')
        advance=length if stride is None else stride
        if type(advance) is not int or not 1<=advance<=length:raise ValueError('Native stride must be 1..window length')
        starts=list(range(0,len(group)-length+1,advance))
        if starts and starts[-1]!=len(group)-length:starts.append(len(group)-length)
        if not starts:excluded.append(dict(step=phase,reason='short_native_occurrence',rows=len(group)))
        for start in starts:
            piece=group.iloc[start:start+length];values=piece[names].to_numpy(np.float32)
            if not np.isfinite(values).all():excluded.append(dict(step=phase,reason='nonfinite_native_window',rows=length));continue
            offset=int(piece.row_in_step.iloc[0])-int(meta['native_row_origin'])
            if offset<0:raise ValueError('Native Step counter precedes trained origin')
            cases.append(dict(current=values,step=phase,centers=chosen,recipe_key=str(piece['_cohort'].iloc[0]),
                loop_key='',offset_samples=offset,run_id=str(piece[run].iloc[0]),lineage_id=decision.p._hash(str(piece[run].iloc[0])),
                row_start=int(piece.index[0]),row_end=int(piece.index[-1])+1))
    return cases,excluded


def replay_pecvd(contract,process,paths,*,steps=None,centers=None,stride=None,
                  max_windows_per_file=None,center_batch_size=8,progress=print):
    from . import pecvd_external,external_test as external,online_inference as online
    if contract.domain!='PECVD' or process.process_family!='PECVD':raise ValueError('PECVD detailed checkpoint/config required')
    if max_windows_per_file is not None and (type(max_windows_per_file) is not int or max_windows_per_file<1):
        raise ValueError('Positive window limit or None required')
    paths=[Path(p) for p in paths]
    if not paths or len({p.stem for p in paths})!=len(paths):raise ValueError('Unique nonempty PECVD file inventory required')
    runtime=contract.runtime;before=external._fingerprint(runtime);session=decision.Session(runtime,center_batch_size=center_batch_size)
    rows=[];requests=[];coverage=[];excluded=[]
    for path in paths:
        if progress:progress('Detailed PECVD: prepare one supplied file',flush=True)
        checksum=decision.p.digest(path);frame=pecvd_external.read_frame(path,runtime,process)
        header=pd.read_csv(path,nrows=0).columns
        coords=[c for c in ('row_in_step','step_number') if c in header]
        if 'row_in_step' not in coords:raise ValueError('PECVD CSV must provide native row_in_step; no future-progress inference')
        counters=pd.read_csv(path,usecols=coords)
        for name in coords:frame[name]=counters[name]
        cases,rejected=pecvd_windows(frame,runtime,process,steps=steps,centers=centers,stride=stride)
        excluded.extend(dict(c,case_id=path.stem) for c in rejected);available=len(cases)
        if max_windows_per_file is not None:cases=cases[:max_windows_per_file]
        answered=0;yes=0
        for i,c in enumerate(cases):
            reply=session.process_window(**{k:v for k,v in c.items() if k not in ('row_start','row_end')})
            requests.append(dict(case_id=path.stem,request_id=f'{path.stem}/{i}',step=c['step'],status=reply['status'],reason=reply['reason'],
                answers=len(reply['answers']),evaluated_answers=reply.get('evaluated_answers',0),conditional_answering=True,
                selected_sensors=len(c['centers']),**reply['timings']))
            if reply['status']!='OK':excluded.append(dict(case_id=path.stem,step=c['step'],reason=reply['reason'],rows=c['row_end']-c['row_start']));continue
            answered+=1;yes+=sum(a['answer_type']=='BOOLEAN' and a['prediction']==1 for a in reply['answers'])
            rows.extend(dict(a,case_id=path.stem,run_id=c['run_id'],step=c['step'],row_start=c['row_start'],row_end=c['row_end']) for a in reply['answers'])
        coverage.append(dict(case_id=path.stem,available_windows=available,windows=len(cases),answered_windows=answered,
            windows_not_requested_by_limit=available-len(cases),yes_sensor_windows=yes,
            observed_prediction=int(yes>0) if answered else np.nan,whole_run_negative_certified=False))
        if decision.p.digest(path)!=checksum:raise ValueError('PECVD supplied input changed during replay')
    if external._fingerprint(runtime)!=before:raise ValueError('Frozen detailed weights changed during replay')
    latency,by_step=online.latency_tables(requests,domain='PECVD')
    return ExperimentReport('Detailed PECVD supplied input',pd.DataFrame([dict(domain='PECVD',accuracy_available=False,questions=len(rows))]),
        details=pd.DataFrame(rows),diagnostics=dict(coverage=pd.DataFrame(coverage),requests=pd.DataFrame(requests),exclusions=pd.DataFrame(excluded),
            request_latency=latency,request_latency_by_step=by_step),metadata=dict(domain='PECVD',model_schema=decision.SCHEMA,
            input_mode=decision.INPUT_MODE,checkpoint=str(runtime.module.FUSION_ANSWER_PATH),training=False,selection=False,conditional_answering=True,
            original_used_for_prediction=False,real_fault_truth=False,automatic_export=False))


def replay_csv(contract,path,*,center_batch_size=8,progress=print,on_result=None):
    if contract.domain!='PEALD':raise ValueError('Use replay_pecvd with a PECVD process config')
    from . import ald_delivery_evaluation as delivery
    path=Path(path).resolve();before=decision.p.digest(path);frame=current.read_frame(path,contract)
    runs=frame[contract.process.source_columns['run']].astype('string').fillna('');bounds=[]
    for i,(_,g) in enumerate(frame.groupby(runs.ne(runs.shift()).cumsum(),sort=False)):
        bounds.append(dict(run_id=f'INPUT-{i+1:04}',start=int(g.index[0]),end=int(g.index[-1])+1))
    report=_replay_frame(contract,frame,bounds,center_batch_size=center_batch_size,progress=progress,on_result=on_result)
    if decision.p.digest(path)!=before:raise ValueError('Input CSV changed during replay')
    report.metadata.update(input_identity={'csv_sha256':before},paired_truth_available=False)
    report.metadata['prediction_sha256']=delivery._seal(report)
    return report
