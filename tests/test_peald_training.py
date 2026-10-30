"""Invented raw jobs, structural corruption, real frozen Fusion/head training."""
import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pecvd.experiments import peald_training as p, fusion_answers as fa
from pecvd.experiments.process_inputs import phase_key
from tests.test_peald_fast import fixture as source_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    source, _, _ = source_fixture()
    phase = phase_key('cycle', 'main')
    process = NS(sensor_columns=['p','q','r'],
        source_columns=dict(run='job', step='phase', order='stamp', cohort='recipe', step_context='context'),
        steps=[NS(name=phase, description='Invented target phase')],
        source_path=Path('synthetic_mapping.json'), data_path='synthetic.csv')
    raw = pd.DataFrame([dict(job=f'job{i:02}', phase='cycle', recipe='one', context='main',
        stamp=str(pd.Timestamp('2020-01-01')+pd.Timedelta(seconds=i*1000+t)),
        p=np.sin(t/3)+i/10, q=np.cos(t/2), r=t/7)
        for i in range(24) for t in range(24)])
    config = replace(p.PealdConfig(), max_window_length=8, train_windows_per_step=4,
        validation_windows_per_step=2, test_windows_per_step=2, epochs=1, width=16,
        max_seconds=30, batch_size=128, step_epochs=1)
    return source, process, raw, config, phase


def prepared_runtime(source, process, raw, config):
    prepared = p.prepare_frame(raw, process, patch_length=4, config=config)
    prepared.identity = dict(synthetic_fixture='no private data', policy=p.asdict(config))
    return p.runtime_for(source, prepared, source_step='DEP.')


class PealdTrainingTests(unittest.TestCase):
    def test_quarantine_and_native_time_boundaries_do_not_stitch_rows(self):
        _, process, raw, config, phase = fixture()
        # A metadata hole, a nonfinite value, a timestamp gap, exact duplicate,
        # conflicting duplicate and an invalid timestamp must remain distinguishable.
        raw.loc[4,'phase'] = None
        raw.loc[24+4,'p'] = np.inf
        raw.loc[48+12:48+23,'stamp'] = pd.to_datetime(raw.loc[48+12:48+23,'stamp']).add(pd.Timedelta(seconds=10)).astype(str)
        duplicate = raw.iloc[[80]].copy()
        conflict = raw.iloc[[104]].copy(); conflict['p'] += 5
        raw.loc[128,'stamp'] = 'invalid'
        raw = pd.concat([raw.iloc[:81],duplicate,raw.iloc[81:105],conflict,raw.iloc[105:]],ignore_index=True)
        prepared = p.prepare_frame(raw,process,patch_length=4,config=config)
        reasons = set(prepared.tables['quarantine'].reason)
        self.assertTrue({'missing_run_or_step','nonfinite_sensor_row','invalid_timestamp',
            'exact_duplicate_row','conflicting_duplicate_timestamp'} <= reasons)
        self.assertGreater(prepared.tables['boundaries'].boundaries.sum(),0)
        for pools in prepared.windows.values():
            for windows in pools.values():
                for window in windows:
                    self.assertEqual(window['input_row_end']-window['input_row_start'],7)
                    self.assertLessEqual(window['maximum_interval_seconds'],1.05)
        self.assertEqual(len(raw),prepared.tables['summary'].input_rows.iloc[0])

    def test_cadence_scaling_and_windows_are_train_only_outliers_are_retained(self):
        _, process, raw, config, phase = fixture()
        baseline = p.prepare_frame(raw,process,patch_length=4,config=config)
        split = p._run_splits(raw.job.unique(),config.seed)
        modified = raw.copy(); job = sorted(split['test'])[0]
        modified.loc[modified.job.eq(job),'p'] = 1e8
        result = p.prepare_frame(modified,process,patch_length=4,config=config)
        for name in ('mean','std','window_length','cadence_seconds'):
            np.testing.assert_array_equal(result.metadata[phase][name],baseline.metadata[phase][name])
        self.assertEqual(len(result.windows[phase]['test']),len(baseline.windows[phase]['test']))
        self.assertTrue(result.tables['signal_candidates'].split.eq('test').any())
        self.assertFalse(result.tables['summary'].signal_outliers_removed.iloc[0])
        self.assertFalse(set(result.split_runs['train']) & set(result.split_runs['test']))

    def test_ambiguous_job_identity_recipe_and_backwards_time_are_quarantined(self):
        _, process, raw, config, _ = fixture()
        raw.loc[24+2,'recipe']='other'
        raw.loc[48+2,'stamp']=str(pd.Timestamp(raw.loc[48+1,'stamp'])-pd.Timedelta(milliseconds=500))
        raw=pd.concat([raw,raw.iloc[:2]],ignore_index=True)
        result=p.prepare_frame(raw,process,patch_length=4,config=config)
        self.assertTrue({'disjoint_job_identity','recipe_changes_inside_job','nonmonotonic_job_time'}<=set(result.tables['quarantine'].reason))

    def test_unknown_train_scale_is_withheld_and_prefix_holes_keep_native_origin(self):
        _,process,raw,config,phase=fixture()
        process.sensor_columns=['p','q','r','unknown_scale']
        raw['unknown_scale']=1.
        job=sorted(p._run_splits(raw.job.unique(),config.seed)['test'])[0]
        raw.loc[raw.job.eq(job),'unknown_scale']=500.
        first=raw.index[raw.job.eq(job)][0];raw.loc[first,'p']=np.inf
        result=p.prepare_frame(raw,process,patch_length=4,config=config)
        self.assertNotIn('unknown_scale',result.metadata[phase]['sensor_columns'])
        self.assertEqual(result.tables['sensors'].set_index('sensor').loc['unknown_scale','status'],'NO_TRAIN_SCALE')
        windows=[w for w in result.windows[phase]['test'] if w['run_id']==p._hash(job)]
        self.assertTrue(windows)
        self.assertEqual(windows[0]['cohort'][2],1)
        self.assertGreater(result.tables['quarantine_locations'].rows.sum(),0)

    def test_trains_only_target_head_roundtrips_and_test_never_enters_fit(self):
        source, process, raw, config, phase = fixture()
        target = prepared_runtime(source,process,raw,config)
        modules=[source.module.GEVIS_MODEL,source.encoders['DEP.'][0],source.module.FUSION_ANSWER_HEAD]
        before=[{k:v.clone() for k,v in model.state_dict().items()} for model in modules]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source.module.BACKBONE_PATH=root/'source_backbone';source.module.BACKBONE_PATH.write_bytes(b'invented source')
            source.module.FUSION_ANSWER_PATH=root/'source_head';source.module.FUSION_ANSWER_PATH.write_bytes(b'invented source binding')
            seen=[];real=p.examples
            def examples(runtime,split,**kwargs):
                seen.append(split)
                return real(runtime,split,**kwargs)
            with patch.object(p,'examples',side_effect=examples),contextlib.redirect_stdout(io.StringIO()):
                training=p.train_or_load(root,target,config=config)
            self.assertEqual(seen,['train','validation'])
            self.assertGreater(training['optimizer_updates'],0)
            self.assertEqual(training['trained_steps'],[phase])
            self.assertFalse(set(training['train_runs'])&set(training['validation_runs']))
            self.assertIsNot(target.module.FUSION_ANSWER_HEAD,source.module.FUSION_ANSWER_HEAD)
            window=target.prepared.windows[phase]['test'][0]
            self.assertEqual(len(fa.predict_arrays(target.module,window['current'],window['current'],phase,'p')),3)
            scored=p.evaluate(target,config=config)
            self.assertEqual(set(scored.summary.answer_type),set(fa.KINDS))
            self.assertIn('f1',scored.diagnostics['boolean_metrics'])
            original=p.original(target,config=config)
            self.assertFalse(original.details.empty)
            self.assertNotIn('label',original.details)
            path=target.module.FUSION_ANSWER_PATH;content=path.read_bytes()
            fresh=prepared_runtime(source,process,raw,config)
            with patch.object(p,'examples',side_effect=AssertionError('Reload must not fit')):
                p.train_or_load(root,fresh,config=config,allow_training=False)
            self.assertEqual(path.read_bytes(),content)
            for k,v in target.module.FUSION_ANSWER_HEAD.state_dict().items():
                torch.testing.assert_close(v,fresh.module.FUSION_ANSWER_HEAD.state_dict()[k],rtol=0,atol=0)
            # A tampered identity is refused, never silently reused.
            payload=torch.load(path,weights_only=False);payload['identity']['source_encoder_step']='wrong';torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'identity/training'):
                p.train_or_load(root,fresh,config=config)
        for model,saved in zip(modules,before):
            for k,v in saved.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        self.assertEqual(list(source.encoders),['DEP.'])

    def test_untrained_target_withholds_answers_and_load_only_does_not_fit(self):
        source,process,raw,config,_=fixture()
        target=prepared_runtime(source,process,raw,config)
        with self.assertRaisesRegex(RuntimeError,'head missing'):
            p.evaluate(target,config=config)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source.module.BACKBONE_PATH=root/'backbone';source.module.BACKBONE_PATH.write_bytes(b'fixture')
            source.module.FUSION_ANSWER_PATH=root/'head';source.module.FUSION_ANSWER_PATH.write_bytes(b'fixture')
            with patch.object(p,'examples',side_effect=AssertionError('No fitting')):
                with self.assertRaises(FileNotFoundError):p.train_or_load(root,target,config=config,allow_training=False)

    def test_changed_input_is_rejected_before_fitting(self):
        source,process,raw,config,_=fixture()
        target=prepared_runtime(source,process,raw,config)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);csv=root/'input.csv';csv.write_bytes(b'invented input');mapping=root/'mapping.json';mapping.write_bytes(b'{}')
            target.prepared.process=NS(data_path='input.csv',source_path=mapping)
            target.prepared.identity.update(csv_sha256=p.digest(csv),process_sha256=p.digest(mapping))
            csv.write_bytes(b'changed invented input')
            with patch.object(p,'examples',side_effect=AssertionError('No stale training')):
                with self.assertRaisesRegex(ValueError,'changed since preprocessing'):
                    p.train_or_load(root,target,config=config)

    def test_actual_exp6_run_all_trains_and_scores_real_synthetic_model(self):
        source,process,raw,config,_=fixture()
        notebook=json.loads((ROOT/'notebooks/ALD_evaluate.ipynb').read_text(encoding='utf-8'))
        settings=dict(decision_system=fa.SYSTEM,training_profile='center_relation_v14')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source.module.BACKBONE_PATH=root/'backbone';source.module.BACKBONE_PATH.write_bytes(b'fixture')
            source.module.FUSION_ANSWER_PATH=root/'head';source.module.FUSION_ANSWER_PATH.write_bytes(b'fixture')
            target=prepared_runtime(source,process,raw,config)
            env={'display':lambda x:None};exists=Path.is_file
            with patch('importlib.reload',side_effect=lambda m:m), \
                 patch.object(Path,'is_file',lambda q:True if q.as_posix().endswith('.cache/external/peald/process.json') else exists(q)), \
                 patch('pecvd.experiments.pipeline_settings.configure_pipeline',return_value=settings), \
                 patch('pecvd.experiments.fast_experiments.load_runtime',return_value=source), \
                 patch('pecvd.process_config.load_process_config',return_value=process), \
                 patch.object(p,'prepare',return_value=target.prepared), \
                 patch('pecvd.experiments.peald_encoder_routing.train_or_load',side_effect=AssertionError('Default must use the original DEP path')), \
                 patch('pandas.read_csv',side_effect=AssertionError('No private CSV reads')), \
                 contextlib.redirect_stdout(io.StringIO()):
                for cell in notebook['cells']:
                    if cell['cell_type']=='code':
                        exec(compile(''.join(cell['source']),'EXP6:'+cell['id'],'exec'),env)
                        if cell['id']=='setup':
                            self.assertEqual(env['ENCODER_SELECTION_MODE'],'single')
                            self.assertEqual(env['SOURCE_ENCODER_STEP'],'DEP.')
                            self.assertIs(env['PEALD_API'],p)
                            self.assertEqual(env['PREPARATION_PATCH_LENGTH'],source.encoders['DEP.'][0].patch_len)
                            self.assertFalse(env['RUN_PEALD_TRAINING'])
                            self.assertFalse(env['RUN_ORIGINAL_REPLAY'])
                            env['RUN_PEALD_TRAINING']=True
                            env['RUN_ORIGINAL_REPLAY']=True
                            env['CONFIG']=config;env['ROOT']=root
            self.assertFalse(env['SUMMARY'].empty)
            self.assertFalse(env['ORIGINAL'].details.empty)
            self.assertGreater(env['TRAINING']['optimizer_updates'],0)
