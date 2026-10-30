"""Model/EXP documentation must track the executable routing contract."""
from pathlib import Path
import tempfile
import unittest

from scripts.check_gevis_docs import check_docs
from scripts.build_gitlab_release import build_release

ROOT = Path(__file__).resolve().parents[1]


class DocumentContractTests(unittest.TestCase):
    def test_current_tree_and_clean_handoff_match_actual_contract(self):
        self.assertEqual(check_docs(ROOT), [])
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'source'
            build_release(ROOT,target)
            self.assertEqual(check_docs(target), [])
            path=target/'_blueprint/experiment-plan.md'
            path.write_text(path.read_text(encoding='utf-8').replace('peald_native_validation_encoder_routes_v1','outdated_recipe'),encoding='utf-8')
            self.assertTrue(any('stale' in error for error in check_docs(target)))

    def test_changing_notebook_default_without_docs_review_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'source'
            build_release(ROOT,target)
            path=target/'notebooks/ALD_evaluate.ipynb'
            original=path.read_text(encoding='utf-8')
            path.write_text(original.replace("ENCODER_SELECTION_MODE='single'","ENCODER_SELECTION_MODE='validation'"),encoding='utf-8')
            self.assertTrue(any('PEALD evaluation default' in error for error in check_docs(target)))
            path.write_text(original.replace("SOURCE_ENCODER_STEP='DEP.'","SOURCE_ENCODER_STEP='STAB1'"),encoding='utf-8')
            self.assertTrue(any('PEALD evaluation default' in error for error in check_docs(target)))

    def test_source_change_with_unchanged_recipe_still_requires_docs_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'source'
            build_release(ROOT,target)
            path=target/'src/pecvd/experiments/peald_encoder_routing.py'
            path.write_bytes(path.read_bytes()+b'\n# Synthetic source change with the same recipe.\n')
            self.assertTrue(any('without document synchronization' in error for error in check_docs(target)))
