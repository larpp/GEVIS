"""Synthetic tests for the analysis-table builder; never opens project data."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pecvd.analysis_table import build_analysis_table, write_analysis_table


class AnalysisTableTests(unittest.TestCase):
    def synthetic_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "process_id": [2, 1, 1, 1, 1, 2, 2],
                "timestamp": [
                    "2026-01-01 00:00:03",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:02",
                    "2026-01-01 00:00:03",
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:02",
                ],
                "declared_start": [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:00:00",
                ],
                "step_name": [
                    "PURGE",
                    "Gap-up",
                    "Gap-up",
                    "PRE-HEAT",
                    "Gap-up",
                    "Gap-up",
                    "PURGE",
                ],
                "pressure": [23, 10, 11, 12, 13, 20, 22],
            }
        )

    def test_builds_process_and_contiguous_step_coordinates(self):
        result = build_analysis_table(
            self.synthetic_frame(), declared_start_column="declared_start"
        )

        process_one = result[result["process_id"] == 1].reset_index(drop=True)
        self.assertEqual(process_one["pressure"].tolist(), [10, 11, 12, 13])
        self.assertEqual(process_one["step_number"].tolist(), [1, 1, 2, 3])
        self.assertTrue(pd.isna(process_one.loc[0, "previous_step"]))
        self.assertEqual(process_one.loc[2, "previous_step"], "Gap-up")
        self.assertEqual(process_one.loc[3, "previous_step"], "PRE-HEAT")
        self.assertEqual(process_one["step_progress"].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(
            process_one["elapsed_from_first_sample_seconds"].tolist(),
            [0.0, 0.0, 1.0, 2.0],
        )
        self.assertEqual(
            process_one["elapsed_from_declared_start_seconds"].tolist(),
            [1.0, 1.0, 2.0, 3.0],
        )

    def test_declared_start_is_not_guessed(self):
        frame = self.synthetic_frame().drop(columns="declared_start")
        with self.assertRaisesRegex(ValueError, "Declared start cannot be inferred"):
            build_analysis_table(frame)

    def test_invalid_timestamp_fails_closed(self):
        frame = self.synthetic_frame()
        frame.loc[0, "timestamp"] = "not-a-time"
        with self.assertRaisesRegex(ValueError, "invalid timestamps"):
            build_analysis_table(frame, declared_start_column="declared_start")

    def test_write_refuses_existing_output_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "analysis.csv"
            destination.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_analysis_table(pd.DataFrame({"value": [1]}), destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "existing\n")


if __name__ == "__main__":
    unittest.main()
