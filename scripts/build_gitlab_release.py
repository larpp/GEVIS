"""Build and validate a clean, source-only GitLab release tree.

The builder uses an explicit allowlist. It never traverses ``data/``,
``artifacts/``, ``preprocessing/``, caches, or the existing Git history.
Active STEP notebooks are copied after outputs, execution counts, attachments,
and widget metadata are removed. The destination must not already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT_FILES = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "FAILURE_LOG.md",
)
TREE_PATTERNS = (
    # The regression suite reads the active model contract; omit Work/Failure history.
    ".agents/eidos/direction.md",
    "src/pecvd/*.py",
    "src/pecvd/experiments/*.py",
    "tests/*.py",
    "tests/fixtures/*.json",
    "scripts/*.py",
    "config/requirements*.txt",
    "config/processes/*.json",
    "config/experiments/gevis_pipeline.json",
    "config/experiments/peald_fast.json",
    "config/experiments/fixed_pattern_example.json",
    "config/experiments/tips_peald.json",
    "_blueprint/experiment-plan.md",
    "_blueprint/gevis-experiment-guide.md",
    "_docs/README.md",
    "_docs/verification.md",
    "_docs/gevis-failure-lessons.md",
    "_docs/tips-peald-training.md",
    "notebooks/README.md",
    "notebooks/research/README.md",
    "notebooks/research/CVD_question_sensitivity.ipynb",
    "notebooks/research/CVD_synthetic_change_generalization.ipynb",
    "notebooks/research/CVD_feature_ablation.ipynb",
    "notebooks/ALD_evaluate.ipynb",
    "notebooks/CVD_predict_external_csv.ipynb",
    "notebooks/ALD_predict_external_csv.ipynb",
    "notebooks/CVD_evaluate.ipynb",
    "notebooks/TIPS_PEALD_train_evaluate.ipynb",
)
ACTIVE_NOTEBOOKS = (
    "notebooks/STEP_1_train_patchtst_encoder.ipynb",
    "notebooks/STEP_2_train_gevis_backbone.ipynb",
    "notebooks/STEP_3_train_gevis_decision.ipynb",
    "notebooks/STEP_3.1_run_gevis_inference.ipynb",
)
PROHIBITED_PARTS = {
    ".git", ".cache", "data", "artifacts", "preprocessing", "node_modules",
    "backup", "_backups", ".ipynb_checkpoints", "__pycache__",
}
PROHIBITED_SUFFIXES = {".csv", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl"}


def release_paths(root: Path) -> list[Path]:
    root = root.resolve()
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for pattern in TREE_PATTERNS:
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    paths.extend(root / name for name in ACTIVE_NOTEBOOKS)
    missing = [path.relative_to(root).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required release files are missing: {missing}")
    return sorted(set(paths))


def sanitized_notebook(source: Path) -> bytes:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cell.pop("attachments", None)
    metadata = notebook.setdefault("metadata", {})
    metadata.pop("widgets", None)
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def sanitized_process_config(source: Path) -> bytes:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["data_path"] = "data/LOCAL_APPROVED_ANALYSIS.csv"
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sanitized_gitignore(source: Path) -> bytes:
    """Keep active sanitized notebooks trackable in the clean repository."""
    lines = source.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip() not in {
        "/notebooks/*.ipynb", "/notebooks/**/*.ipynb",
    }]
    lines.extend((
        "",
        "# Historical/local notebooks are not part of the public pipeline.",
        "/notebooks/RL_test*.ipynb",
    ))
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def validate_release(destination: Path) -> dict[str, int]:
    destination = destination.resolve()
    files = [path for path in destination.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(destination)
        if set(relative.parts) & PROHIBITED_PARTS or path.suffix.lower() in PROHIBITED_SUFFIXES:
            raise ValueError(f"Prohibited release path: {relative.as_posix()}")
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if any(cell.get("outputs") or cell.get("execution_count") is not None
                   or cell.get("attachments") for cell in notebook.get("cells", [])):
                raise ValueError(f"Notebook retains execution material: {relative.as_posix()}")
            if notebook.get("metadata", {}).get("widgets"):
                raise ValueError(f"Notebook retains widget state: {relative.as_posix()}")
    expected = {Path(name).name for name in ACTIVE_NOTEBOOKS}
    actual = {path.name for path in (destination / "notebooks").glob("STEP*.ipynb")}
    if actual != expected:
        raise ValueError(f"Active notebook set mismatch: expected={sorted(expected)}, actual={sorted(actual)}")
    expected_all = set(ACTIVE_NOTEBOOKS) | {
        pattern for pattern in TREE_PATTERNS if pattern.endswith('.ipynb')
    }
    actual_all = {path.relative_to(destination).as_posix() for path in files
                  if path.suffix == '.ipynb'}
    if actual_all != expected_all:
        raise ValueError("Release notebook inventory differs from the explicit allowlist")
    return {"files": len(files), "notebooks": len(actual_all)}


def build_release(root: Path, destination: Path) -> dict[str, int]:
    root, destination = root.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError("Release destination already exists; choose a new empty path")
    if destination.is_relative_to(root):
        allowed_parent = (root / ".cache" / "publish").resolve()
        if not destination.is_relative_to(allowed_parent):
            raise ValueError("In-workspace releases must stay under .cache/publish")
    destination.mkdir(parents=True)
    for source in release_paths(root):
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".ipynb":
            target.write_bytes(sanitized_notebook(source))
        elif relative.as_posix() == ".gitignore":
            target.write_bytes(sanitized_gitignore(source))
        elif relative.as_posix() == "config/processes/pecvd.json":
            target.write_bytes(sanitized_process_config(source))
        elif relative.as_posix() == "FAILURE_LOG.md":
            target.write_text(
                "# GEVIS failure prevention\n\n"
                "See [reviewed reusable lessons](_docs/gevis-failure-lessons.md).\n\n"
                "Detailed local ledgers and historical measurements are not included "
                "in this source-only handoff.\n", encoding="utf-8",
            )
        else:
            shutil.copy2(source, target)
    return validate_release(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = build_release(args.root, args.destination)
    print(json.dumps({"destination": str(args.destination.resolve()), **result}))


if __name__ == "__main__":
    main()
