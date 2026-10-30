from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pecvd.process_config import load_process_config


ROOT = Path(__file__).resolve().parents[1]


class ProcessConfigTests(unittest.TestCase):
    def test_pecvd_process_config_preserves_current_step_contract(self) -> None:
        config = load_process_config(ROOT / "config" / "processes" / "pecvd.json")

        self.assertEqual(config.process_family, "PECVD")
        self.assertEqual(config.columns.run, "process_key")
        self.assertEqual(config.artifacts.encoder, "gevis_v4_rich_ssl_final_v1")
        self.assertEqual(config.artifacts.decision, "gevis_fdc_e5_v1")
        self.assertEqual(config.step_specs["Gap-up"], (40, 5, 5))
        self.assertEqual(config.step_specs["PRE-HEAT"], (40, 5, 5))
        self.assertEqual(config.step_specs["STAB1"], (32, 4, 4))
        self.assertEqual(config.step_specs["DEP."], (256, 16, 64))
        self.assertEqual(config.step_specs["PURGE"], (56, 7, 7))
        self.assertEqual(config.step_specs["PUMPING"], (72, 9, 9))
        self.assertEqual(len(config.steps), 6)
        self.assertEqual(len(config.sha256), 64)

    def test_process_config_rejects_duplicate_steps(self) -> None:
        payload = json.loads(
            (ROOT / "config" / "processes" / "pecvd.json").read_text(encoding="utf-8")
        )
        payload["steps"].append(dict(payload["steps"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_process_config(path)

    def test_e5_artifact_contract_is_connected_through_validation(self) -> None:
        backbone = (ROOT / "notebooks" / "STEP_2_train_gevis_backbone.ipynb").read_text(
            encoding="utf-8"
        )
        training = (ROOT / "notebooks" / "STEP_3_train_gevis_decision.ipynb").read_text(
            encoding="utf-8"
        )
        inference = (ROOT / "notebooks" / "STEP_3.1_run_gevis_inference.ipynb").read_text(
            encoding="utf-8"
        )
        validation = (ROOT / "notebooks" / "CVD_evaluate.ipynb").read_text(
            encoding="utf-8"
        )

        self.assertIn("numeric_dim:int=440 if selected_variant()=='multiscale'", backbone)
        self.assertIn("real_step1_binding_v2", backbone)  # Explicit warm-start source.
        self.assertIn("gevis_backbone_e5_v1", training)
        self.assertIn("fdc_semantic_step1_bridge_all_types_v1", training)
        self.assertIn('BACKBONE_RUN_TAG=active_run_tag()', training)
        self.assertIn('active_run_tag()', inference)
        self.assertIn('active_run_tag()', validation)
        self.assertIn("NUMERIC_DIM=4*ENCODER_D_MODEL+(56 if selected_variant()=='multiscale'", training)
        self.assertIn("gevis.backbone.generic.v10", training)
        self.assertIn("gevis.encoder_manifest.rich_ssl.v1", training)
        self.assertNotIn("gevis.question_model.v16", training)  # No legacy decision-head writes.
        self.assertIn("gevis.requires_grad_(False).eval()", training)
        self.assertIn("RUN_FUSION_ANSWER_TRAINING=DECISION_SYSTEM=='fusion_answers'", training)
        self.assertIn("RUN_STRUCTURED_EVALUATION=not RUN_FUSION_ANSWER_TRAINING", training)
        self.assertIn("RUN_TRAINING=False", training)
        self.assertIn("RUN_INTEGRATED_EXPERIMENT=False", training)
        self.assertIn("fusion_answers.evaluate(FUSION_RUNTIME", training)
        self.assertNotIn("evaluate_generic_grounding", training)
        self.assertIn("STEP4_GEVIS_FUSION_TEST_V1", validation)
        self.assertIn('run_gevis_experiment',validation)
        self.assertNotIn('run_question_experiment',validation)
        self.assertNotIn('ljr_contract',validation)
        self.assertIn("gevis.backbone.generic.v10", inference)
        self.assertIn("text_encoder_revision", inference)
        self.assertIn("Missing STEP 2 v9 backbone", inference)
        self.assertIn("predicted semantic slots + predicted sensor", inference)
        # STEP 4 delegates loading/identity validation to the same STEP 3.1 runtime.
        self.assertIn('load_experiment_runtime(ROOT)',validation)
        self.assertIn('load_runtime()',inference)
        self.assertIn('load_policy(REFERENCE_POLICY_PATH,identity_for',inference)
        self.assertIn("fast_assessment.compact(FUSION_TEST,display", validation)
        self.assertIn("inference_timing.benchmark", validation)
        self.assertIn("FUSION_HEAD=require_head", validation)
        self.assertIn("ERROR_PREVIEW", validation)
        self.assertNotIn("elif EXPERIMENT_MODE=='integrated':", validation)
        self.assertIn("REPORT.diagnostics['confusion']",validation)


if __name__ == "__main__":
    unittest.main()
