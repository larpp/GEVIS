"""Structural tests only: invented frames, no model, CSV or weight execution."""
from tests.legacy_notebooks import legacy_exp_book
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from pecvd.experiments.process_inputs import align_step_occurrences,load_process_frame,screen_external_quality
from pecvd.experiments.frozen_transfer import external_pairs


def fixture():
    rows=[]
    for r in range(30):
        for occurrence in range(2):
            for t in range(16):
                rows.append(dict(run=str(r),step='phase',order=occurrence*16+t,
                    _gevis_segment_id=occurrence,_gevis_position_ratio=(occurrence*16+t)/31,
                    source_group_recipe=1,sensor=float(r+100*occurrence+t)))
    config=SimpleNamespace(columns=SimpleNamespace(run='run',step='step',order='order'),
        sensor_columns=('sensor',),steps=[SimpleNamespace(name='phase')])
    return pd.DataFrame(rows),config


class ExternalPreprocessingTests(unittest.TestCase):
    def test_retired_exp4_fixture_keeps_previous_preprocessing_options(self):
        import ast,json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        doc=legacy_exp_book('04_dataset_transfer.ipynb')
        setup=ast.parse(''.join(next(c['source'] for c in doc['cells'] if c.get('id')=='setup')))
        value=next(n.value for n in setup.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='QUALITY_OPTIONS' for t in n.targets))
        options={k.arg:ast.literal_eval(k.value) for k in value.keywords}
        self.assertEqual(options['excluded_months'],(4,7))
        self.assertNotIn('structure_columns',options);self.assertNotIn('gap_factor',options)
        source=(root/'src/pecvd/experiments/process_inputs.py').read_text(encoding='utf-8')
        self.assertNotIn('def prepare_external_alignment(',source)

    def test_calendar_hints_do_not_drop_or_relabel_sensor_values(self):
        frame,cfg=fixture();frame['_gevis_time']=pd.Timestamp('2020-04-01')+pd.to_timedelta(frame.order,unit='s')
        before=frame.copy(deep=True)
        result,tables=screen_external_quality(frame,cfg,suspect_months=(4,7))
        self.assertEqual(len(result),len(frame));self.assertTrue(result['_gevis_suspect_month'].all())
        pd.testing.assert_series_equal(result.sensor,frame.sensor)
        self.assertEqual(tables['external_quality_summary'].iloc[0].excluded_runs,0)
        pd.testing.assert_frame_equal(frame,before)

    def test_excluding_a_month_removes_entire_cross_month_run(self):
        frame,cfg=fixture();frame['_gevis_time']=pd.Timestamp('2020-05-01')+pd.to_timedelta(frame.order,unit='s')
        frame.loc[(frame.run=='0')&(frame.order==0),'_gevis_time']=pd.Timestamp('2020-04-30')
        result,tables=screen_external_quality(frame,cfg,excluded_months=(4,7))
        self.assertFalse(result.run.eq('0').any());self.assertEqual(len(result),len(frame)-32)
        self.assertEqual(tables['external_quality_summary'].iloc[0].excluded_runs,1)

    def test_edge_candidates_not_automatically_removed_and_explicit_minimum(self):
        frame,cfg=fixture();frame['_gevis_time']=pd.Timestamp('2020-05-01')+pd.to_timedelta(np.arange(len(frame)),unit='s')
        result,tables=screen_external_quality(frame,cfg)
        self.assertEqual(len(result),len(frame))
        self.assertEqual(tables['external_step_quality'].dataset_edge_candidates.sum(),2)
        kept,tables=screen_external_quality(frame,cfg,minimum_step_rows={'phase':17})
        self.assertTrue(kept.empty)
        self.assertEqual(tables['external_quality_summary'].iloc[0].excluded_runs,30)
        kept,_=screen_external_quality(frame,cfg,exclude_file_edge_runs=True)
        self.assertFalse(kept.run.isin(['0','29']).any())

    def test_quality_screen_requires_time_for_month_filters_and_valid_options(self):
        frame,cfg=fixture()
        with self.assertRaisesRegex(ValueError,'timestamps'):
            screen_external_quality(frame,cfg,suspect_months=(4,7))
        with self.assertRaises(ValueError):screen_external_quality(frame,cfg,excluded_months=(13,))
        with self.assertRaises(ValueError):screen_external_quality(frame,cfg,minimum_step_rows={'phase':0})

    def test_position_is_local_to_occurrence_and_values_unchanged(self):
        frame,cfg=fixture();before=frame.copy(deep=True)
        result=align_step_occurrences(frame,run='run',step='step',order='order')
        for _,g in result.groupby(['run','_gevis_segment_id']):
            self.assertEqual(g['_gevis_step_row'].tolist(),list(range(16)))
            self.assertEqual(g['_gevis_step_position_ratio'].iloc[0],0.)
            self.assertEqual(g['_gevis_step_position_ratio'].iloc[-1],1.)
        pd.testing.assert_frame_equal(frame,before)
        pd.testing.assert_series_equal(result.sensor.sort_index(),frame.sensor)

    def test_matches_occurrence_offset_unique_runs_and_distinct_pair_ids(self):
        frame,cfg=fixture();packs,support=external_pairs(frame,cfg,8,limit=2)
        paired,_,_=packs['phase']
        self.assertEqual(len(paired['calibration_validation']),4*2*2)
        allids=[]
        for current,ref in paired['calibration_validation']:
            self.assertEqual(current['_gevis_step_occurrence'].iloc[0],ref['_gevis_step_occurrence'].iloc[0])
            self.assertEqual(current['_gevis_step_row'].iloc[0],ref['_gevis_step_row'].iloc[0])
            refs=current.attrs['reference_run_ids']
            self.assertEqual(len(refs),len(set(refs)))
            self.assertNotIn(current.attrs['evaluation_run_id'],refs)
            allids.append(current.attrs['evaluation_pair_id'])
            context=current.attrs['cohort_key'];pos=current.attrs['position_bin']
            pool=[w for w,c,p in paired['reference'] if str(c)==context and p==pos][:16]
            np.testing.assert_allclose(ref[['sensor']],np.median(np.stack([w[['sensor']] for w in pool]),axis=0))
        self.assertEqual(len(allids),len(set(allids)))
        self.assertEqual(support.iloc[0].reference_runs,21)

    def test_heldout_values_cannot_change_train_scaling_or_reference(self):
        frame,cfg=fixture();first,_=external_pairs(frame,cfg,8)
        ids=np.asarray(sorted(frame.run.unique()),object);np.random.default_rng(42).shuffle(ids)
        changed=frame.copy();changed.loc[changed.run.isin(ids[21:]),'sensor']+=1e6
        second,_=external_pairs(changed,cfg,8)
        np.testing.assert_array_equal(first['phase'][1],second['phase'][1])
        np.testing.assert_array_equal(first['phase'][2],second['phase'][2])
        for a,b in zip(first['phase'][0]['calibration_validation'],second['phase'][0]['calibration_validation']):
            np.testing.assert_array_equal(a[1].sensor,b[1].sensor)

    def test_short_nonfinite_tail_audits_and_no_imputation(self):
        frame,cfg=fixture();frame.loc[(frame.run=='0')&(frame.order==0),'sensor']=np.nan
        _,support=external_pairs(frame,cfg,6)
        self.assertEqual(support.iloc[0].nonfinite_windows,1)
        self.assertGreater(support.iloc[0].unused_tail_rows,0)
        self.assertTrue(pd.isna(frame.loc[0,'sensor']))
        packs,support=external_pairs(frame,cfg,20)
        self.assertFalse(packs);self.assertEqual(support.iloc[0].short_segments,60)

    def test_duplicate_coordinates_and_disjoint_segment_reuse_rejected(self):
        frame,_=fixture()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            align_step_occurrences(pd.concat([frame,frame.iloc[:1]]),run='run',step='step',order='order')
        frame.loc[(frame.run=='0')&(frame.order==31),'_gevis_segment_id']=0
        with self.assertRaisesRegex(ValueError,'reappears'):
            align_step_occurrences(frame,run='run',step='step',order='order')

    def test_loader_duplicate_timestamps_rejected_without_reading_csv(self):
        cfg=SimpleNamespace(source_columns={'run':'rawrun','step':'rawstep','order':'time'},
            sensor_columns=('sensor',),data_path='invented.csv')
        frame=pd.DataFrame(dict(rawrun=['a','a'],rawstep=['s','s'],
            time=['2020-01-01','2020-01-01'],sensor=[1.,2.]))
        with patch('pandas.read_csv',return_value=frame):
            with self.assertRaisesRegex(ValueError,'Duplicate external timestamps'):
                load_process_frame('.',cfg)


if __name__=='__main__':unittest.main()
