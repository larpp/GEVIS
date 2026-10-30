"""Real synthetic Fusion fits, six frozen source candidates and exact routing."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pecvd.temporal_patchtst import TemporalPatchEncoder
from pecvd.experiments import peald_training as p, peald_encoder_routing as route
from tests.test_peald_training import fixture


def target(root):
    source, process, raw, config, phase = fixture()
    source.module.BACKBONE_PATH = root/'backbone.pt'
    source.module.BACKBONE_PATH.write_bytes(b'synthetic frozen backbone identity')
    source.module.FUSION_ANSWER_PATH = root/'source_head.pt'
    source.module.FUSION_ANSWER_PATH.write_bytes(b'synthetic verified source identity')
    for name, size in [('Gap-up',2),('PRE-HEAT',3),('STAB1',2),('PURGE',5),('PUMPING',8)]:
        encoder = TemporalPatchEncoder(2,size,d_model=8,nhead=2,layers=1,dropout=0.).requires_grad_(False).eval()
        source.encoders[name] = (encoder, dict(sensor_columns=['a','b'], window_length=8))
    prepared = p.prepare_frame(raw, process, patch_length=route.preparation_patch_length(source), config=config)
    prepared.identity = dict(synthetic='fixture only', policy=p.asdict(config))
    return source, route.runtime_for(source,prepared), config, phase


class EncoderRoutingTests(unittest.TestCase):
    def test_six_candidates_share_windows_and_patch_incompatibility_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            source, runtime, config, step = target(Path(folder))
            self.assertEqual(len(runtime.candidates),6)
            self.assertEqual(route.preparation_patch_length(source),2)
            dep, _ = route.candidate_for(runtime,step,'DEP.')
            short, _ = route.candidate_for(runtime,step,'Gap-up')
            self.assertIs(dep.prepared.windows[step],short.prepared.windows[step])
            for name in ['PRE-HEAT','PURGE','PUMPING']:
                candidate, reason = route.candidate_for(runtime,step,name)
                self.assertIsNone(candidate)
                self.assertIn('native_patch',reason)
            self.assertNotEqual(dep.prepared.identity,short.prepared.identity)
            with self.assertRaises(ValueError): route.runtime_for(source,runtime.prepared,candidates=['missing'])

    def test_validation_policy_preserves_baseline_unless_interpretation_and_detection_improve(self):
        baseline = dict(boolean_f1=.98,macro_f1=.8,choice_macro_f1=.7,ordinal_macro_f1=.7)
        better = dict(boolean_f1=.97,macro_f1=.85,choice_macro_f1=.8,ordinal_macro_f1=.8)
        worse_detection = dict(better,boolean_f1=.8)
        self.assertEqual(route.choose({'DEP.':baseline,'short':better},'DEP.',.96),'short')
        self.assertEqual(route.choose({'DEP.':baseline,'short':worse_detection},'DEP.',.96),'DEP.')
        self.assertEqual(route.choose({'short':better},'DEP.',.96),'short')

    def test_real_fit_never_reads_test_preserves_weights_and_exact_route_reload(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder); source, runtime, config, step=target(root)
            withheld_step=step+'_insufficient_support'
            runtime.prepared.metadata[withheld_step]=dict(runtime.prepared.metadata[step])
            runtime.prepared.windows[withheld_step]=dict(train=[],validation=[],test=[])
            frozen={s:route.state_digest(e) for s,(e,_) in source.encoders.items()}
            backbone=route.state_digest(source.module.GEVIS_MODEL)
            original_cases=p.cases
            def training_cases(r,split,**kwargs):
                if split=='test': raise AssertionError('No Test during fitting or selection')
                return original_cases(r,split,**kwargs)
            with patch.object(p,'cases',side_effect=training_cases):
                training=route.train_or_load(root,runtime,config=config)
            self.assertEqual(len(training['candidates']),12)
            self.assertIn(withheld_step,training['withheld'])
            self.assertNotIn(withheld_step,runtime.encoders)
            self.assertFalse(training['test_used_for_selection'])
            self.assertEqual(training['selection_split'],'validation')
            self.assertEqual(frozen,{s:route.state_digest(e) for s,(e,_) in source.encoders.items()})
            self.assertEqual(backbone,route.state_digest(source.module.GEVIS_MODEL))
            loaded=route.runtime_for(source,runtime.prepared)
            with patch.object(p,'examples',side_effect=AssertionError('No selection or training on reload')):
                route.train_or_load(root,loaded,config=config,allow_training=False)
            self.assertEqual(runtime.selection,loaded.selection)
            selected=route.for_step(loaded,step)
            window=runtime.prepared.windows[step]['test'][0]['current']
            expected=p.predict_arrays(selected,window,window,step,'p')
            actual=route.predict_arrays(loaded,window,window,step,'p')
            self.assertEqual(actual,expected)
            report=route.evaluate(loaded,config=config)
            self.assertFalse(report.summary.empty)
            self.assertFalse(route.original(loaded,config=config).details.empty)
            manifest=json.loads(loaded.manifest_path.read_text())
            manifest['routes'][step]['head_sha256']='corrupted binding'
            loaded.manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'binding changed'):
                route.train_or_load(root,route.runtime_for(source,runtime.prepared),config=config,allow_training=False)

    def test_load_only_never_fits_when_manifest_absent_and_missing_route_never_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); _,runtime,config,step=target(root)
            with patch.object(p,'examples',side_effect=AssertionError('No fitting')):
                with self.assertRaises(FileNotFoundError):
                    route.train_or_load(root,runtime,config=config,allow_training=False)
            with self.assertRaisesRegex(RuntimeError,'missing'):
                route.predict_arrays(runtime,np.zeros((8,3)),np.zeros((8,3)),step,'p')


if __name__=='__main__': unittest.main()
