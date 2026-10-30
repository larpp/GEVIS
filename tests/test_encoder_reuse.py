import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from pecvd.encoder_reuse import load_saved_encoders,sha256

ROOT=Path(__file__).resolve().parents[1]


class EncoderReuseTests(unittest.TestCase):
    def bundle(self,root,backend='hf_patchtst_legacy_joint_v1'):
        (root/'encoders').mkdir();(root/'windows').mkdir()
        ck={'schema':'gevis.encoder.rich_ssl.v1','encoder_backend':backend,
            'sensor_columns':['a','b'],'window_length':8,'patch_length':2,'source_sha256':'synthetic',
            'run_split':{'train':['r0'],'validation':['r1'],'test':['r2']},'model_state':{'fixture':torch.ones(1)}}
        torch.save(ck,root/'encoders/phase.pt')
        np.savez(root/'windows/phase.npz',**{s:np.zeros((1,8,2),dtype='float32') for s in ck['run_split']})
        manifest={'schema':'gevis.encoder_manifest.rich_ssl.v1','artifact_version':'fixture',
            'sensor_columns':ck['sensor_columns'],'source_sha256':'synthetic','run_split':ck['run_split'],
            'sensor_descriptors':{'a':'a','b':'b'},'cohort_split_counts':{},
            'step_checkpoints':{'phase':{'path':'encoders/phase.pt','sha256':sha256(root/'encoders/phase.pt')}}}
        (root/'encoder_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        config=SimpleNamespace(artifacts=SimpleNamespace(encoder='fixture'),step_specs={'phase':(8,2,2)})
        return config,manifest

    def test_saved_bundle_reused_without_writes_and_rejects_hash_or_grid_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config,manifest=self.bundle(root)
            before={p:sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch('pandas.read_csv',side_effect=AssertionError('No CSV parsing')):
                result,ck=load_saved_encoders(root,config,backend='hf_patchtst_legacy_joint_v1')
            self.assertEqual(result,manifest);self.assertEqual(set(ck),{'phase'})
            self.assertEqual(before,{p:sha256(p) for p in before})
            with self.assertRaises(ValueError):load_saved_encoders(root,config,backend='other')
            config.step_specs={'phase':(10,2,2)}
            with self.assertRaises(ValueError):load_saved_encoders(root,config,backend='hf_patchtst_legacy_joint_v1')
            config.step_specs={'phase':(8,2,2)}
            (root/'encoders/phase.pt').write_bytes(b'corrupted fixture')
            with self.assertRaises(ValueError):load_saved_encoders(root,config,backend='hf_patchtst_legacy_joint_v1')

    def test_actual_step1_reuse_cells_skip_training_cache_and_manifest_writes(self):
        book=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        cells={c.get('id'):''.join(c.get('source',[])) for c in book['cells']}
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);cfg,_=self.bundle(root);before={p:sha256(p) for p in root.rglob('*') if p.is_file()}
            saved=load_saved_encoders(root,cfg,backend='hf_patchtst_legacy_joint_v1')[1]
            env={'torch':torch,'REUSE_SAVED_ENCODERS':True,'RUN_TRAINING':False,'SAVED_ENCODERS':saved,
                 'SESSION_SIGNATURE':'STEP1_GEVIS_HYBRID_REUSE_V6'}
            for cid in ('01-windows','01-fit','01-manifest'):
                exec(compile(cells[cid],cid,'exec'),env)
            self.assertEqual(env['encoder_checkpoints'],saved)
            self.assertEqual(before,{p:sha256(p) for p in before})

    def test_actual_temporal_load_and_reuse_never_train_or_write(self):
        book=json.loads((ROOT/'notebooks/STEP_1_train_patchtst_encoder.ipynb').read_text(encoding='utf-8'))
        cells={c.get('id'):''.join(c.get('source',[])) for c in book['cells']}
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);cfg,_=self.bundle(root,backend='gevis_native_temporal_blind_v2')
            before={p:sha256(p) for p in root.rglob('*') if p.is_file()}
            env={'torch':torch,'REUSE_SAVED_ENCODERS':True,'RUN_TRAINING':False,
                 'CONFIGURED_MODEL_VARIANT':'temporal','ARTIFACT_ROOT':root,
                 'PROCESS_CONFIG':cfg,'DATA_PATH':None,'pd':__import__('pandas'),
                 'SESSION_SIGNATURE':'STEP1_GEVIS_HYBRID_REUSE_V6'}
            with patch('torch.save',side_effect=AssertionError('No checkpoint writes')),patch('pandas.read_csv',side_effect=AssertionError('No source parsing')),patch('torch.optim.AdamW',side_effect=AssertionError('No retraining')):
                for cid in ('01-load','01-windows','01-fit','01-manifest'):
                    exec(compile(cells[cid],cid,'exec'),env)
            self.assertEqual(env['_backend'],'gevis_native_temporal_blind_v2')
            self.assertEqual(set(env['encoder_checkpoints']),{'phase'})
            self.assertEqual(before,{p:sha256(p) for p in root.rglob('*') if p.is_file()})

    def test_default_is_retained_hybrid_not_untrained_portable(self):
        from pecvd.model_variants import selected_variant,active_run_tag
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(selected_variant(),'hybrid')
            self.assertEqual(active_run_tag(),'real_step1_joint_hybrid_reference_v11')

    def test_actual_step2_reuse_branches_never_construct_or_save_model(self):
        book=json.loads((ROOT/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        saved={'text_metrics':[],'semantic_metrics':[],'cross_history':[]}
        env={'REUSE_SAVED_BACKBONE':True,'SAVED_BACKBONE':saved,'pd':__import__('pandas'),'display':lambda *a:None}
        with patch('torch.save',side_effect=AssertionError('No overwrite')),contextlib.redirect_stdout(io.StringIO()):
            for c in book['cells']:
                if c.get('id') in ('04-text-training','05-text-curve','07-cross-training','08-save'):
                    exec(compile(''.join(c['source']),c['id'],'exec'),env)
        self.assertIs(env['checkpoint'],saved)

    def test_exp4_fixed_sensor_limitation_is_visible_not_an_exception_or_fake_evaluation(self):
        from pecvd.experiments.reporting import run_transfer_experiment
        runtime=SimpleNamespace(module=SimpleNamespace(),encoders={'fixture':(torch.nn.Linear(1,1),{})})
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            config=Path(folder)/'external.json'
            config.write_text((ROOT/'config/processes/pecvd.json').read_text(encoding='utf-8'),encoding='utf-8')
            report=run_transfer_experiment(ROOT,process_config=config,source_runtime=runtime,prepared_frame=object())
        self.assertFalse(report.metadata['model_evaluated'])
        self.assertTrue(report.summary.status.eq('NOT_EVALUATED').all())
        contract=report.diagnostics['source_artifact_contract'].iloc[0]
        self.assertFalse(contract.step1_encoder_executed)
        self.assertFalse(contract.step2_cross_attention_executed)
