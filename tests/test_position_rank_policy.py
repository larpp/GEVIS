"""Invented signals only: conditional calibration and shared runtime contracts."""
from tests.legacy_notebooks import legacy_exp_book
import copy, unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from pecvd.experiments.window_policy import WindowConfig, WindowPolicy, fit_window_policy
from pecvd.experiments.position_rank_policy import context_key, rank_score


def records(split,n=8):
    rows=[]
    for position,amplitude in ((0,.01),(9,100.)):
        for i in range(n):
            a=np.zeros((32,2));a[:,0]=amplitude*(1+i*.01);a[:,1]=np.sin(np.arange(32)*.3)*.01
            rows.append(dict(split=split,step='phase',run_id=f'{split}-{i}',reference_run_ids=['train-reference'],
                sensors=['center','partner'],cohort='recipe',position=position,current=a,
                reference=np.zeros_like(a),residual=a*a))
    return rows


class PositionRankTests(unittest.TestCase):
    def test_core_profile_excludes_types_from_calibration_and_actual_inference(self):
        core=replace(self.config,operation_profile='core')
        train=copy.deepcopy(self.train);val=copy.deepcopy(self.val)
        policy=WindowPolicy(fit_window_policy(train,val,{},core));row=self.val[0]
        result=policy.analyze(row['current'],row['reference'],row['residual'],'phase',row['sensors'],
            selected=['center'],center_sensor='center',cohort='recipe',position=0)
        self.assertEqual(set(result['disabled_operations']),{'stuck','delay','relationship'})
        self.assertEqual({r['statistic'] for r in result['ranked_evidence']},{'level','variability','slope','local_change','residual'})
        self.assertEqual(policy.payload['supported_kinds'],['level_shift','drift','spike','variance'])
        self.assertTrue(all(c['requested_order_statistic']==9 for c in policy.payload['groups']['phase']['contexts'].values()))

    def test_core_shared_runtime_removes_choice_labels_and_marks_relation_unavailable(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery import review_changes
        from pecvd.experiments.window_policy import MODE
        from pecvd.experiments.questions import CRITERIA
        runtime,current,reference,_=fixture();m=runtime.module
        train=copy.deepcopy(self.train);val=copy.deepcopy(self.val)
        for r in train+val:r.update(step='invented_phase',sensors=['a','b'])
        m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,val,{},replace(self.config,operation_profile='core')))
        m.DISCOVERY_POLICY_MODE=MODE
        for f in (current,reference):f.attrs.update(cohort_key='recipe',position_bin=0,reference_split='train')
        result=review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('fixture',1),center_sensor='a')
        choice=next(a for a in result['answers'] if a['answer_type']=='CHOICE')
        self.assertEqual(tuple(choice['probabilities']),CRITERIA['CHOICE'])
        self.assertFalse(result['relationship_analysis']['accepted'])
        self.assertIsNone(result['relationship_analysis']['changed'])
        self.assertEqual(result['relationship_analysis']['unsupported_reason'],'operation_disabled_by_profile')

    def test_core_primary_report_does_not_generate_or_hide_removed_cases(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.window_policy import MODE
        from pecvd.experiments.discovery_reporting import run_discovery_experiment
        runtime,current,reference,_=fixture();m=runtime.module
        train=copy.deepcopy(self.train);val=copy.deepcopy(self.val)
        for r in train+val:r.update(step='invented_phase',sensors=['a','b'])
        m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,val,{},replace(self.config,operation_profile='core')))
        m.DISCOVERY_POLICY_MODE=MODE
        for f in (current,reference):f.attrs.update(cohort_key='recipe',position_bin=0,reference_split='train')
        with patch('pecvd.experiments.discovery_calibration.load_discovery_pairs',return_value=[(current,reference)]),\
             patch('pecvd.experiments.discovery_reporting.dynamic_challenges',side_effect=AssertionError('Disabled challenge must not run')):
            # Execute the actual EXP2 config cell: stale seven-kind defaults
            # must never be silently reintroduced by notebook-only changes.
            import ast,json
            from pathlib import Path
            doc=legacy_exp_book('02_anomaly_injection.ipynb')
            code=next(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code'
                      and any('KINDS=' in line for line in c['source']))
            nodes=[n for n in ast.parse(code).body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='KINDS' for t in n.targets)]
            scope={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'exp2-kinds','exec'),scope)
            self.assertIsNone(scope['KINDS'])
            report=run_discovery_experiment('.',runtime=runtime,analysis_mode='center',max_cases=14,kinds=scope['KINDS'])
        self.assertEqual(set(report.details.kind),{'untouched','level_shift','drift','spike','variance'})
        self.assertEqual(set(report.details.task),{'self'})
        self.assertEqual(report.metadata['operation_profile'],'core')
        with self.assertRaisesRegex(ValueError,'disabled'):
            run_discovery_experiment('.',runtime=runtime,analysis_mode='center',kinds=('delay',))

    def test_coverage_sampler_covers_independent_contexts_before_repeat_windows(self):
        from pecvd.experiments.discovery_calibration import _coverage_indices
        runs=[];cohorts=[];positions=[]
        for run in range(10):
            for cohort in ('one','two'):
                for pos in range(10):
                    for repeat in range(3):
                        runs.append(f'{cohort}-{run}');cohorts.append(cohort);positions.append(pos)
        indices=_coverage_indices(np.array(runs),np.array(cohorts),np.array(positions),42,256)
        keys=[(cohorts[i],positions[i],runs[i]) for i in indices]
        self.assertEqual(len(set(keys[:200])),200)
        self.assertEqual(len(indices),256);self.assertEqual(len(set(indices)),256)
        self.assertEqual(indices,_coverage_indices(np.array(runs),np.array(cohorts),np.array(positions),42,256))

    def test_calibration_budget_recovers_support_without_changing_eval_schedule(self):
        from pecvd.experiments.discovery_calibration import cached_comparisons
        data={}
        for split,n,repeats in (('train',10,3),('validation',4,3),('test',4,1)):
            metadata=[(f'{split}-{cohort}-{run}',cohort,pos) for run in range(n)
                for cohort in ('one','two') for pos in range(10) for _ in range(repeats)]
            data[split]=np.zeros((len(metadata),8,2),dtype=np.float32)
            data[split+'_process_id']=np.array([v[0] for v in metadata])
            data[split+'_cohort_key']=np.array([v[1] for v in metadata])
            data[split+'_position_bin']=np.array([v[2] for v in metadata])
        class Guard:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def __getitem__(self,key):
                if key=='test':raise AssertionError('Test values cannot fit policy')
                return data[key]
        cfg=replace(self.config,max_windows_per_step=32,max_calibration_windows_per_step=256)
        with patch('pecvd.experiments.discovery_calibration.np.load',return_value=Guard()):
            first=list(cached_comparisons('.', 'phase',split='validation',config=cfg))
            train=list(cached_comparisons('.', 'phase',split='train',config=cfg,calibration=True))
            val=list(cached_comparisons('.', 'phase',split='validation',config=cfg,calibration=True))
            second=list(cached_comparisons('.', 'phase',split='validation',config=cfg))
        self.assertEqual([r['cache_index'] for r in first],[r['cache_index'] for r in second])
        self.assertEqual(len(first),32);self.assertEqual(len(train),256)
        for r in train+val:r.update(step='phase',sensors=['center','partner'],residual=np.zeros_like(r['current']))
        g=fit_window_policy(train,val,{},cfg)['groups']['phase']
        self.assertEqual(len(g['contexts']),20)
        self.assertTrue(all(c['supported'] for c in g['contexts'].values()))
        self.assertTrue(all(c['train_runs']==10 and c['validation_runs']==4 for c in g['contexts'].values()))

    def setUp(self):
        self.config=WindowConfig(gate_scope='position_rank_v4')
        self.train=records('train');self.val=records('validation')
        self.policy=WindowPolicy(fit_window_policy(self.train,self.val,{},self.config))

    def analyze(self,row,**kwargs):
        options=dict(selected=['center'],center_sensor='center',cohort='recipe',position=row['position'])
        options.update(kwargs)
        return self.policy.analyze(row['current'],row['reference'],row['residual'],'phase',row['sensors'],**options)

    def test_position_bank_does_not_pool_quiet_and_high_variation_regions(self):
        row=copy.deepcopy(self.val[0]);row['current'][:,0]=1.;row['residual'][:,0]=1.
        early=self.analyze(row)
        self.assertTrue(early['accepted']);self.assertTrue(early['changed'])
        late=self.analyze(row,position=9)
        self.assertFalse(late['changed'])
        self.assertIn('no legacy fallback',early['change_decision_source'])

    def test_unknown_cohort_or_position_withholds_without_fallback(self):
        for options in (dict(cohort='unknown'),dict(position=4),dict(cohort=None),dict(position=None)):
            result=self.analyze(self.val[0],**options)
            self.assertFalse(result['accepted']);self.assertIsNone(result['changed'])
            self.assertEqual(result['calibration']['unsupported_reason'],'missing_or_insufficient_same_context_runs')

    def test_duplicate_windows_do_not_create_independent_support(self):
        first=self.policy.payload['groups']['phase']['contexts']
        second=fit_window_policy(self.train*2,self.val*3,{},self.config)['groups']['phase']['contexts']
        self.assertEqual(first,second)
        self.assertEqual(first[context_key('recipe',0)]['validation_runs'],8)

    def test_sparse_context_is_not_borrowed_from_another_position(self):
        val=[r for r in self.val if r['position']==9 or r['run_id']=='validation-0']
        policy=WindowPolicy(fit_window_policy(self.train,val,{},self.config))
        row=self.val[0]
        result=policy.analyze(row['current'],row['reference'],row['residual'],'phase',row['sensors'],selected=['center'],center_sensor='center',cohort='recipe',position=0)
        self.assertFalse(result['accepted'])

    def test_unavailable_context_is_not_reported_as_numeric_gate_rejection(self):
        from pecvd.experiments.discovery_reporting import _gate_diagnostics
        policy=WindowPolicy(fit_window_policy(self.train,[],{},self.config));row=self.val[0]
        result=policy.analyze(row['current'],row['reference'],row['residual'],'phase',row['sensors'],selected=['center'],center_sensor='center',cohort='recipe',position=0)
        result.update(center_sensor='center',attention_top_sensor='partner')
        diagnostic=_gate_diagnostics(result,'partner')
        self.assertFalse(diagnostic['attention_correct_but_numeric_gate_rejected'])
        self.assertEqual(diagnostic['target_gate_status'],'UNSUPPORTED_CONTEXT_OR_CRITERIA')

    def test_test_fit_overlap_and_multiplier_are_rejected(self):
        with self.assertRaises(ValueError):fit_window_policy(self.train,records('test'),{},self.config)
        bad=copy.deepcopy(self.val);bad[0]['run_id']='train-0'
        with self.assertRaises(ValueError):fit_window_policy(self.train,bad,{},self.config)
        with self.assertRaises(ValueError):fit_window_policy(self.train,self.val,{},replace(self.config,threshold_multiplier=.9))

    def test_rank_ties_and_above_support_magnitudes_remain_ordered(self):
        score=rank_score([0.,1.,2.],[0.,0.,0.],1.)
        self.assertEqual(score[0],0.);self.assertGreater(score[2],score[1]);self.assertGreater(score[1],1.)

    def test_comparison_uses_same_banks_and_exposes_context_support(self):
        from pecvd.experiments.threshold_comparison import fit_comparison,position_calibration_table
        policies=fit_comparison(self.train,self.val,lambda cfg:{},self.config)
        banks=[p.payload['groups']['phase']['contexts'][context_key('recipe',0)]['bank'] for p in policies.values()]
        for bank in banks[1:]:self.assertEqual(bank,banks[0])
        table=position_calibration_table(policies)
        self.assertEqual(len(table),12);self.assertFalse(table.legacy_fallback.any());self.assertTrue(table.supported.all())

    def test_three_operation_traces_are_retained(self):
        result=self.analyze(self.val[0])
        self.assertTrue({'stuck','delay','relationship'}<={r['statistic'] for r in result['ranked_evidence']})

    def test_json_roundtrip_and_inference_are_read_only(self):
        import json
        payload=json.loads(json.dumps(self.policy.payload,allow_nan=False));before=copy.deepcopy(payload)
        policy=WindowPolicy(payload);row=self.val[0]
        policy.analyze(row['current'],row['reference'],row['residual'],'phase',row['sensors'],selected=['center'],center_sensor='center',cohort='recipe',position=0)
        self.assertEqual(payload,before)

    def test_shared_center_path_preserves_context_and_one_neural_call(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery import review_changes
        from pecvd.experiments.window_policy import MODE
        runtime,current,reference,_=fixture();m=runtime.module
        train=copy.deepcopy(self.train);val=copy.deepcopy(self.val)
        for r in train+val:r.update(step='invented_phase',sensors=['a','b'])
        m.DISCOVERY_POLICY=WindowPolicy(fit_window_policy(train,val,{},self.config));m.DISCOVERY_POLICY_MODE=MODE
        for frame in (current,reference):frame.attrs.update(cohort_key='recipe',position_bin=0,reference_split='train')
        current.loc[4,'a']=2.
        with patch.object(m,'numeric_state',wraps=m.numeric_state) as state,patch.object(m.GEVIS_MODEL,'forward',wraps=m.GEVIS_MODEL.forward) as attention:
            result=review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('fixture',1),center_sensor='a')
        self.assertTrue(result['accepted']);self.assertTrue(result['self_analysis']['changed'])
        self.assertEqual(state.call_count,1);self.assertEqual(attention.call_count,1)
        self.assertEqual(result['calibration']['context_key'],context_key('recipe',0))
        reference.attrs['position_bin']=9
        with self.assertRaisesRegex(ValueError,'exactly matched'):
            review_changes(m,current,reference,'invented_phase',m.ReferenceInfo('fixture',1),center_sensor='a')
