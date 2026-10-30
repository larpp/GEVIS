import contextlib,copy,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pecvd.experiments.discovery_reporting import localization_scores,interval_union_iou
from pecvd.experiments.execution_contract import execution_contract
from pecvd.experiments.pipeline_audit import NOTEBOOKS,audit_pipeline,notebook_variant,selected_profile

ROOT=Path(__file__).resolve().parents[1]


class PipelineContractTests(unittest.TestCase):
    def test_canonical_pipeline_has_no_retired_label_classifier_dependency(self):
        # A removed experiment must not remain necessary for Run All or packaging.
        retired=('autolabel', 'snorkel', 'dawid_skene')
        for path in (ROOT/'notebooks').glob('STEP*.ipynb'):
            document=json.loads(path.read_text(encoding='utf-8'))
            for cell in document['cells']:
                source=''.join(cell.get('source',[])).lower()
                for name in retired:self.assertNotIn(name,source,str(path))
        from scripts.build_gitlab_release import TREE_PATTERNS
        for pattern in TREE_PATTERNS:
            self.assertNotIn(retired[0],pattern)
        self.assertFalse((ROOT/'config/requirements-autolabel.txt').exists())
        self.assertFalse((ROOT/'config/experiments/autolabel.json').exists())
        for name in ('autolabel.py','autolabel_pipeline.py'):
            self.assertFalse((ROOT/'src/pecvd/experiments'/name).exists())

    def test_localization_diagnostic_does_not_masquerade_as_displayed_winner(self):
        target={'sensor':'target','detected_rows':[15,16],'pointwise':{'exceedance_intervals':[(10,20),(15,16)]}}
        wrong={'sensor':'other','detected_rows':[10,20],'pointwise':{'exceedance_intervals':[(10,20)]}}
        result={'events':[wrong,target],'winner':wrong}
        score=localization_scores(result,'target',(10,20))
        self.assertEqual(score['target_interval_iou'],1.)
        self.assertEqual(score['target_union_iou'],1.)
        self.assertEqual(score['winner_interval_iou'],0.)
        result['winner']=target
        self.assertEqual(localization_scores(result,'target',(10,20))['winner_interval_iou'],.1)
        target['detected_rows']=None
        score=localization_scores(result,'target',(10,20))
        self.assertEqual(score['winner_interval_iou'],0.)
        self.assertFalse(score['winner_interval_available'])

    def test_union_iou_penalizes_extra_segments_and_merges_scope_duplicates(self):
        self.assertEqual(interval_union_iou([(10,20),(10,20),(15,20)],(10,20)),1.)
        self.assertEqual(interval_union_iou([(10,15),(30,35)],(10,20)),1/3)
        self.assertEqual(interval_union_iou([],(10,20)),0.)

    def test_discovery_executes_generic_attention_and_reports_its_real_iou_source(self):
        from tests.test_shared_discovery import fixture
        from pecvd.experiments.discovery_reporting import run_discovery_experiment
        runtime,current,reference,_=fixture()
        with patch('pecvd.experiments.discovery_reporting.load_cached_pairs',return_value=[(current,reference)]),patch.object(runtime.module,'ask_gevis',side_effect=AssertionError('Not part of discovery')),contextlib.redirect_stdout(io.StringIO()):
            report=run_discovery_experiment(ROOT,runtime=runtime,max_cases=4)
        self.assertTrue(report.metadata['execution_contract']['step2_cross_attention_executed'])
        self.assertIn('mean_winner_interval_iou',report.summary)
        self.assertIn('mean_target_union_iou',report.summary)
        self.assertEqual(len(report.diagnostics['localization_metrics']),4)

    def test_source_identity_excludes_saved_outputs_and_changes_with_source(self):
        from pecvd.experiments.evaluation import source_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'notebooks').mkdir()
            p=root/'notebooks/STEP_3.1_run_gevis_inference.ipynb'
            doc={'cells':[{'id':'fixture','cell_type':'code','source':['x=1'],'outputs':[]}]}
            p.write_text(json.dumps(doc),encoding='utf-8');before=source_identity(root)
            doc['cells'][0]['outputs']=[{'text':'invented output only'}]
            p.write_text(json.dumps(doc),encoding='utf-8');self.assertEqual(before,source_identity(root))
            doc['cells'][0]['source']=['x=2'];p.write_text(json.dumps(doc),encoding='utf-8')
            self.assertNotEqual(before,source_identity(root))

    def test_canonical_entries_are_unique_and_archives_not_active(self):
        self.assertEqual(notebook_variant(ROOT),'multiscale')
        self.assertEqual(len(list((ROOT/'notebooks').glob('STEP_1*.ipynb'))),1)
        for name in NOTEBOOKS:self.assertTrue((ROOT/'notebooks'/name).is_file())
        with selected_profile('hybrid'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                # Synthetic copy of code/config ONLY, no data/outputs/artifacts.
                import shutil
                (root/'notebooks/research').mkdir(parents=True);(root/'config/processes').mkdir(parents=True)
                shutil.copyfile(ROOT/'config/processes/pecvd.json',root/'config/processes/pecvd.json')
                for name in NOTEBOOKS:
                    doc=json.loads((ROOT/'notebooks'/name).read_text(encoding='utf-8'))
                    for cell in doc['cells']:
                        cell.pop('outputs',None);cell.pop('execution_count',None)
                    (root/'notebooks'/name).write_text(json.dumps(doc),encoding='utf-8')
                rows=audit_pipeline(root)
                self.assertEqual({r['variant'] for r in rows},{'hybrid'})
                self.assertEqual([r['status'] for r in rows if r['stage']!='notebook'],['MISSING']*4)
                self.assertFalse((root/'artifacts').exists())

    def test_readiness_rejects_encoder_and_backbone_identity_mismatches(self):
        from pecvd.experiments.pipeline_audit import file_sha256
        from pecvd.process_config import load_process_config
        from pecvd.model_variants import active_config,active_run_tag
        from pecvd.experiments.process_inputs import candidate_backbone_path
        from pecvd.experiments.reference_policy_v2 import ReferenceConfig,identity_for,fit_records,save_policy
        import torch
        with tempfile.TemporaryDirectory() as folder,selected_profile('hybrid'),patch.dict('os.environ',{'GEVIS_PROCESS_CONFIG':''}):
            root=Path(folder);(root/'config/processes').mkdir(parents=True);(root/'notebooks/research').mkdir(parents=True)
            for name in NOTEBOOKS:
                (root/'notebooks'/name).write_text(json.dumps({'cells':[]}),encoding='utf-8')
            raw=json.loads((ROOT/'config/processes/pecvd.json').read_text(encoding='utf-8'))
            raw['steps']=[{'name':'fixture','window_length':20,'patch_length':5,'stride':5}]
            cfgpath=root/'config/processes/pecvd.json';cfgpath.write_text(json.dumps(raw),encoding='utf-8')
            cfg=active_config(load_process_config(cfgpath));encoder=root/'artifacts'/cfg.artifacts.encoder
            (encoder/'encoders').mkdir(parents=True);(encoder/'windows').mkdir()
            ep=encoder/'encoders/fixture.pt';ep.write_bytes(b'invented fixture identity')
            (encoder/'windows/fixture.npz').write_bytes(b'invented cache identity only')
            manifest={'schema':'gevis.encoder_manifest.rich_ssl.v1','artifact_version':cfg.artifacts.encoder,
                      'step_checkpoints':{'fixture':{'path':'encoders/fixture.pt','sha256':file_sha256(ep)}}}
            mp=encoder/'encoder_manifest.json';mp.write_text(json.dumps(manifest),encoding='utf-8')
            bp=candidate_backbone_path(root,cfg,active_run_tag());bp.parent.mkdir(parents=True)
            ck={'schema':'gevis.backbone.generic.v10','step1_bridge':{'manifest_sha256':file_sha256(mp)}};torch.save(ck,bp)
            policy=fit_records([],identity_for(root,encoder,bp,ReferenceConfig()))
            save_policy(policy,bp.parent/'reference_policy_v2.json')
            self.assertTrue(all(r['status']=='READY' for r in audit_pipeline(root,variant='hybrid',include_discovery=False)))
            settings=root/'config/experiments/gevis_pipeline.json'
            settings.parent.mkdir(parents=True)
            settings.write_text(json.dumps(dict(decision_system='fusion_answers')),encoding='utf-8')
            def trained_head(module):
                module.FUSION_ANSWER_PATH=bp.parent/'synthetic_readout.pt'
                module.FUSION_ANSWER_HEAD=torch.nn.Linear(1,1)
            with patch('pecvd.experiments.fusion_answers.attach',side_effect=trained_head), \
                 patch('pecvd.experiments.reference_policy_v2.load_policy',
                       side_effect=AssertionError('Active Fusion must not require numerical criteria')):
                rows=audit_pipeline(root,variant='hybrid')
                self.assertEqual(next(r['status'] for r in rows if r['stage']=='STEP3'),'READY')
                self.assertNotIn('STEP3-window',[r['stage'] for r in rows])
            with patch('pecvd.experiments.fusion_answers.attach',side_effect=ValueError('Bad trained state')):
                self.assertEqual(next(r['status'] for r in audit_pipeline(root,variant='hybrid')
                                      if r['stage']=='STEP3'),'FAIL')
            settings.write_text(json.dumps(dict(decision_system='hybrid')),encoding='utf-8')
            ck['step1_bridge']['manifest_sha256']='wrong';torch.save(ck,bp)
            rows=audit_pipeline(root,variant='hybrid')
            self.assertEqual(next(r['status'] for r in rows if r['stage']=='STEP2'),'FAIL')
            self.assertEqual(next(r['status'] for r in rows if r['stage']=='STEP3'),'BLOCKED')
            ep.write_bytes(b'corrupted identity')
            self.assertEqual(next(r['status'] for r in audit_pipeline(root,variant='hybrid') if r['stage']=='STEP1'),'FAIL')
