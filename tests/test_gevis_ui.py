import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
from types import SimpleNamespace

from pecvd.gevis_ui import (
    aligned_sensor_trace,
    choose_typical_reference_id,
    compatible_reference_ids,
    configured_csv_path,
    derive_step_label,
    fdc_deviation_grid,
    matching_position_window,
    load_gevis_runtime,
    notebook_runtime_sources,
    rank_sensors_by_activity,
    reference_envelope,
    select_step_rows,
    step_summary,
    validate_fdc_frame,
    window_at_progress,
    window_position_ratio,
    GevisRuntime,
)


class GevisUiTests(unittest.TestCase):
    def test_discovery_runtime_uses_sensor_free_question_contract(self):
        from pecvd.experiments.questions import DISCOVERY_QUESTIONS, SENSOR_QUESTION
        question_type=lambda text,kind,criteria:SimpleNamespace(text=text,answer_type=kind,criteria=criteria)
        module=SimpleNamespace(
            DECISION_POLICY_MODE='reference_quantiles_v2',review_changes=lambda *args:None,
            GEVIS_CHECKPOINT={},STEP_ENCODERS={'Gap-up':(None,{'sensor_columns':['secret_sensor']})},
            GevisQuestion=question_type,CRITERIA={kind:('a','b') for kind in DISCOVERY_QUESTIONS},
        )
        questions=GevisRuntime(module).questions()
        self.assertEqual([q.text for q in questions],list(DISCOVERY_QUESTIONS.values())+[SENSOR_QUESTION])
        self.assertTrue(all('secret_sensor' not in q.text for q in questions))

    def test_current_runtime_questions_come_from_grounded_question_factory(self):
        expected = [SimpleNamespace(answer_type=kind) for kind in ("BOOLEAN", "CHOICE", "ORDINAL")]
        module = SimpleNamespace(
            GEVIS_CHECKPOINT={},
            STEP_ENCODERS={"Gap-up": (None, {"sensor_columns": ["sensor"]})},
            make_grounded_questions=lambda step, sensor: expected,
        )
        self.assertEqual(GevisRuntime(module).questions(), expected)

    def test_runtime_loader_does_not_require_retired_model_artifact_name(self):
        sources = {
            "03-types": "ENCODER_ARTIFACT_VERSION = 'encoder-v1'\n",
            "03-load": (
                "GEVIS_MODEL = object()\n"
                "STEP_ENCODERS = {}\n"
                "GEVIS_CHECKPOINT = {}\n"
            ),
            "03-function": "pass\n",
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pecvd.gevis_ui.notebook_runtime_sources", return_value=sources
        ):
            runtime = load_gevis_runtime(
                temporary,
                Path(temporary) / "runtime.ipynb",
                Path(temporary) / "approved.csv",
            )
        self.assertEqual(runtime.module.ENCODER_ROOT, Path(temporary) / "artifacts" / "encoder-v1")
        self.assertEqual(runtime.module.DATA_PATH, (Path(temporary) / "approved.csv").resolve())
        self.assertFalse(hasattr(runtime.module, "MODEL_ARTIFACT_VERSION"))

    def test_csv_path_requires_explicit_local_configuration(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PECVD_FDC_CSV"):
                configured_csv_path(Path("project"))

    def test_relative_csv_path_is_resolved_under_project_root(self):
        with patch.dict("os.environ", {"PECVD_FDC_CSV": "private/input.csv"}):
            self.assertEqual(
                configured_csv_path(Path("project")),
                Path("project/private/input.csv"),
            )

    def test_selected_question_uses_one_aggregated_pool_call(self):
        question = SimpleNamespace(answer_type="BOOLEAN")
        window = pd.DataFrame({"row_in_process": range(4)})
        module = SimpleNamespace(
            STEP_ENCODERS={"Gap-up": (None, {"window_length": 4})},
            GEVIS_CHECKPOINT={},
            aggregate_reference_windows=lambda references, meta: window.copy(),
            ReferenceInfo=lambda description, count: SimpleNamespace(description=description, process_count=count),
            ask_gevis=lambda current, reference, step, selected, info: [{
                "answer_type": selected[0].answer_type,
                "reference_count": info.process_count,
            }],
        )
        runtime = GevisRuntime(module)
        with patch.object(GevisRuntime, "questions", return_value=[question]):
            result = runtime.ask_selected_question(window, (window, window), "Gap-up", "BOOLEAN")
        self.assertEqual(result, {"answer_type": "BOOLEAN", "reference_count": 2})

    def test_all_questions_share_one_aggregated_reference(self):
        questions = [SimpleNamespace(answer_type=name) for name in ("BOOLEAN", "CHOICE", "ORDINAL")]
        calls = []
        window = pd.DataFrame({"row_in_process": range(4)})
        module = SimpleNamespace(
            STEP_ENCODERS={"Gap-up": (None, {"window_length": 4})},
            GEVIS_CHECKPOINT={},
            aggregate_reference_windows=lambda references, meta: calls.append((len(references), meta)) or window.copy(),
            ReferenceInfo=lambda description, count: SimpleNamespace(description=description, process_count=count),
            ask_gevis=lambda current, reference, step, selected, info: [item.answer_type for item in selected],
        )
        runtime = GevisRuntime(module)
        with patch.object(GevisRuntime, "questions", return_value=questions):
            result = runtime.ask_all_questions(window, (window, window), "Gap-up")
        self.assertEqual(result, ["BOOLEAN", "CHOICE", "ORDINAL"])
        self.assertEqual(calls, [(2, {"window_length": 4})])

    def frame(self):
        rows = []
        for process, source_a, pm_before in (
            ("p1", 1, 1), ("p2", 1, 1), ("p3", 0, 1), ("p4", 1, 0)
        ):
            for order in range(4):
                rows.append({
                    "process_key": process,
                    "step_label": "Gap-up",
                    "row_in_process": order,
                    "source_group_A": source_a,
                    "source_group_B": 1 - source_a,
                    "pm_cycle_before": pm_before,
                    "pm_cycle_after": 1 - pm_before,
                    "sensor_1": float(order + (process == "p2")),
                })
        return pd.DataFrame(rows)

    def test_reference_selection_uses_all_other_processes(self):
        references = compatible_reference_ids(self.frame(), "p1", "Gap-up", minimum_rows=4)
        self.assertEqual(references, ("p2", "p3", "p4"))
        selected, count = choose_typical_reference_id(
            self.frame(), "p1", "Gap-up", 4, ("sensor_1",)
        )
        self.assertEqual(count, 3)
        self.assertIn(selected, {"p2", "p3", "p4"})

    def test_select_and_align_original_traces(self):
        frame = self.frame()
        current = select_step_rows(frame, "p1", "Gap-up")
        reference = select_step_rows(frame, "p2", "Gap-up")
        trace = aligned_sensor_trace(current, reference, "sensor_1", points=5)
        self.assertEqual(trace.shape, (5, 2))
        self.assertAlmostEqual(trace.iloc[0, 0], 0.0)
        self.assertAlmostEqual(trace.iloc[-1, 1], 4.0)

    def test_derives_step_label_from_one_hot_columns(self):
        frame = pd.DataFrame({
            "process_key": ["p1", "p1"], "row_in_process": [0, 1],
            "step_name_A": [1, 0], "step_name_B": [0, 1],
        })
        result = derive_step_label(frame)
        self.assertEqual(result["step_label"].tolist(), ["A", "B"])
        validate_fdc_frame(result)

    def test_window_selection_preserves_original_rows(self):
        rows = select_step_rows(self.frame(), "p1", "Gap-up")
        self.assertEqual(window_at_progress(rows, 2, 0.0)["row_in_process"].tolist(), [0, 1])
        self.assertEqual(window_at_progress(rows, 2, 1.0)["row_in_process"].tolist(), [2, 3])

    def test_reference_window_uses_original_step_position(self):
        rows = pd.DataFrame({
            "row_in_step": range(10),
            "step_row_count": [10] * 10,
            "sensor": range(10),
        })
        selected = matching_position_window(rows, 4, 1.0)
        self.assertEqual(selected["row_in_step"].tolist(), [6, 7, 8, 9])
        self.assertEqual(window_position_ratio(selected, 4), 1.0)

    def test_aggregated_reference_uses_current_window_coordinates(self):
        current = pd.DataFrame({
            "row_in_process": range(4), "row_in_step": range(4),
            "step_row_count": [4] * 4, "sensor": range(4),
        })
        reference = pd.DataFrame({
            "row_in_process": range(4), "row_in_step": range(6, 10),
            "step_row_count": [10] * 4, "sensor": range(4),
        })
        aligned = GevisRuntime._align_reference_coordinates(current, reference, 4)
        self.assertEqual(aligned["row_in_step"].tolist(), current["row_in_step"].tolist())
        self.assertEqual(aligned["step_row_count"].tolist(), [4] * 4)

    def test_active_sensors_are_ranked_before_flat_sensors(self):
        current = pd.DataFrame({"flat": [0.0, 0.0], "active": [0.0, 2.0]})
        reference = pd.DataFrame({"flat": [0.0, 0.0], "active": [1.0, 3.0]})
        ranked, activity = rank_sensors_by_activity(current, reference, ("flat", "active"))
        self.assertEqual(ranked, ("active", "flat"))
        self.assertEqual(activity["flat"], 0.0)

    def test_fdc_summaries_use_original_rows(self):
        frame = self.frame()
        rows = select_step_rows(frame, "p1", "Gap-up")
        summary = step_summary(rows, ("Gap-up",))
        self.assertEqual(summary.iloc[0]["samples"], 4)
        envelope = reference_envelope(frame, ("p2",), "Gap-up", "sensor_1", points=5)
        self.assertEqual(envelope.shape, (5, 4))
        grid = fdc_deviation_grid(frame, "sensor_1", ("Gap-up",))
        self.assertEqual(len(grid), 4)
        self.assertIn("absolute_deviation", grid)

    def test_runtime_source_contract_fails_closed(self):
        document = {
            "cells": [
                {"cell_type": "code", "id": cell_id, "source": [f"# {cell_id}\n"]}
                for cell_id in ("03-types", "03-load")
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.ipynb"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "03-function"):
                notebook_runtime_sources(path)


if __name__ == "__main__":
    unittest.main()
