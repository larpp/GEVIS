"""Invented paths and checkpoints only; no private artifacts or data are read."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pecvd.experiments.process_inputs import candidate_backbone_path

ROOT=Path(__file__).resolve().parents[1]


class BackboneRevisionTests(unittest.TestCase):
    def test_explicit_revision_preserves_historical_path_and_encoder_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=SimpleNamespace(source_path=root/'config/processes/pecvd.json',
                                   artifacts=SimpleNamespace(encoder='synthetic_encoder'))
            old=candidate_backbone_path(root,config,'fixture',revision='')
            new=candidate_backbone_path(root,config,'fixture',revision='source_v2')
            self.assertNotEqual(old,new)
            self.assertEqual(new.name,old.name)
            self.assertTrue(new.parent.name.endswith('_source_v2'))
            self.assertEqual(config.artifacts.encoder,'synthetic_encoder')
            self.assertFalse(old.exists());self.assertFalse(new.exists())

    def test_shared_config_selects_revision_without_stale_environment_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=root/'config/experiments/gevis_pipeline.json'
            settings.parent.mkdir(parents=True)
            config=SimpleNamespace(source_path=root/'config/processes/pecvd.json',
                                   artifacts=SimpleNamespace(encoder='synthetic_encoder'))
            settings.write_text(json.dumps({'backbone_revision':'source_v2'}),encoding='utf-8')
            with patch.dict(os.environ,{'GEVIS_BACKBONE_REVISION':'unrelated'}):
                self.assertEqual(candidate_backbone_path(root,config,'fixture'),
                                 candidate_backbone_path(root,config,'fixture',revision='source_v2'))
            settings.write_text('{}',encoding='utf-8')
            self.assertEqual(candidate_backbone_path(root,config,'fixture'),
                             candidate_backbone_path(root,config,'fixture',revision=''))

    def test_revision_rejects_path_traversal_and_invalid_types(self):
        config=SimpleNamespace(source_path=ROOT/'config/processes/pecvd.json',
                               artifacts=SimpleNamespace(encoder='synthetic_encoder'))
        for revision in ('../other','a/b','a\\b','A','a b','x'*65,None):
            if revision is None:
                continue  # None means read the explicit shared configuration.
            with self.assertRaises(ValueError):
                candidate_backbone_path(ROOT,config,'fixture',revision=revision)
        for revision in (True,17,[]):
            with self.assertRaises(ValueError):
                candidate_backbone_path(ROOT,config,'fixture',revision=revision)

    def test_step2_retains_guard_and_only_validates_predecessor_for_text_initialization(self):
        notebook=json.loads((ROOT/'notebooks/STEP_2_train_gevis_backbone.ipynb').read_text(encoding='utf-8'))
        sources={c.get('id'):''.join(c.get('source',[])) for c in notebook['cells']}
        setup=sources['01-config'];text=sources['04-text-training']
        self.assertIn("candidate_backbone_path(ROOT,PROCESS_CONFIG,RUN_TAG,revision='')",setup)
        self.assertIn("SAVED_BACKBONE.get('sensor_discrimination_source_sha256')!=sensor_discrimination.source_identity()",setup)
        self.assertIn('Candidate training source changed',setup)
        self.assertIn('if any(prior.get(key)!=value for key,value in expected.items())',text)
        self.assertIn("reuse_fusion_weights:bool=False",setup)
        self.assertIn('backbone.text.load_state_dict',text)


if __name__=='__main__':unittest.main()
