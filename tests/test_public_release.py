import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_gitlab_release import build_release, sanitized_gitignore, sanitized_notebook, validate_release, release_paths


class PublicReleaseTests(unittest.TestCase):
    def test_complete_handoff_preserves_originals_and_excludes_misleading_report(self):
        root = Path(__file__).resolve().parents[1]
        notebooks = list((root / 'notebooks').rglob('*.ipynb'))
        before = {path: path.read_bytes() for path in notebooks}
        self.assertFalse((root / 'notebooks/reports/gevis_result_summary.ipynb').exists())
        self.assertFalse((root / 'src/pecvd/experiments/result_report.py').exists())
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'handoff'
            result = build_release(root, target)
            self.assertEqual(result['notebooks'], 12)
            self.assertNotIn('.agents/', (target / 'FAILURE_LOG.md').read_text(encoding='utf-8'))
            self.assertIn('gevis-failure-lessons.md', (target / 'FAILURE_LOG.md').read_text(encoding='utf-8'))
            self.assertTrue((target / 'tests/fixtures/step3_legacy_diagnostics.json').is_file())
            self.assertTrue((target / '.agents/eidos/direction.md').is_file())
            self.assertFalse((target / '.agents/eidos/work').exists())
            self.assertFalse((target / 'src/pecvd/experiments/result_report.py').exists())
            config = json.loads((target / 'config/processes/pecvd.json').read_text(encoding='utf-8'))
            self.assertEqual(config['data_path'], 'data/LOCAL_APPROVED_ANALYSIS.csv')
        self.assertEqual(before, {path: path.read_bytes() for path in notebooks})

    def test_shared_runtime_and_experiment_notebooks_are_in_source_allowlist(self):
        root=Path(__file__).resolve().parents[1]
        paths={p.relative_to(root).as_posix() for p in release_paths(root)}
        self.assertIn('src/pecvd/experiments/operations.py',paths)
        self.assertIn('src/pecvd/experiments/questions.py',paths)
        self.assertEqual(sum(p.startswith('notebooks/research/') and p.endswith('.ipynb') for p in paths),3)
        self.assertIn('notebooks/CVD_predict_external_csv.ipynb',paths)
        self.assertIn('notebooks/ALD_predict_external_csv.ipynb',paths)
        self.assertIn('notebooks/ALD_evaluate.ipynb',paths)
        self.assertIn('config/experiments/peald_fast.json',paths)
        self.assertFalse(any(p.startswith('notebooks/test/') for p in paths))
        self.assertIn('config/experiments/fixed_pattern_example.json',paths)
        self.assertIn('config/experiments/tips_peald.json',paths)
        self.assertIn('_docs/tips-peald-training.md',paths)
        self.assertIn('notebooks/TIPS_PEALD_train_evaluate.ipynb',paths)
        self.assertNotIn('notebooks/reports/gevis_result_summary.ipynb',paths)
        self.assertIn('_docs/gevis-failure-lessons.md',paths)
        self.assertIn('tests/fixtures/exp_legacy_cells.json',paths)
        self.assertFalse(any('_backups' in p or p.startswith(('data/','artifacts/')) for p in paths))

    def test_public_gitignore_tracks_active_notebooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / ".gitignore"
            source.write_text(
                "/notebooks/*.ipynb\n/notebooks/**/*.ipynb\n/notebooks/_backups/\n/data/\n",
                encoding="utf-8",
            )
            cleaned = sanitized_gitignore(source).decode("utf-8")
            self.assertNotIn("/notebooks/*.ipynb\n", cleaned)
            self.assertNotIn("/notebooks/**/*.ipynb\n", cleaned)
            self.assertIn("/notebooks/RL_test*.ipynb", cleaned)
            self.assertIn("/notebooks/_backups/", cleaned)
            self.assertIn("/data/", cleaned)

    def test_notebook_sanitizer_removes_results_and_widget_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.ipynb"
            source.write_text(json.dumps({
                "cells": [{
                    "cell_type": "code", "execution_count": 3,
                    "outputs": [{"output_type": "stream", "text": ["private"]}],
                    "attachments": {"plot": {}}, "source": ["print('code')"],
                    "metadata": {},
                }],
                "metadata": {"widgets": {"state": {}}},
                "nbformat": 4, "nbformat_minor": 5,
            }), encoding="utf-8")
            cleaned = json.loads(sanitized_notebook(source))
            self.assertIsNone(cleaned["cells"][0]["execution_count"])
            self.assertEqual(cleaned["cells"][0]["outputs"], [])
            self.assertNotIn("attachments", cleaned["cells"][0])
            self.assertNotIn("widgets", cleaned["metadata"])

    def test_release_validator_rejects_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notebooks").mkdir()
            for name in (
                "STEP_1_train_patchtst_encoder.ipynb",
                "STEP_2_train_gevis_backbone.ipynb",
                "STEP_3_train_gevis_decision.ipynb",
                "STEP_3.1_run_gevis_inference.ipynb",
                "CVD_evaluate.ipynb",
            ):
                (root / "notebooks" / name).write_text(
                    json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
                    encoding="utf-8",
                )
            (root / "leak.csv").write_text("secret", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Prohibited release path"):
                validate_release(root)


if __name__ == "__main__":
    unittest.main()
