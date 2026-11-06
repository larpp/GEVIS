"""Read-only dependency/file inventory; optional backup of active assets. No deletion."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil


def native_path(path):
    path = Path(path).resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        text = str(path)
        return Path('\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text)
    return path


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dependency_closure(root):
    """Conservative module-level closure, including literal dynamic imports.

    Import presence is not proof that all functions are active. Preserve rather
    than delete ambiguous modules; saved code-hash references also count.
    """
    base = root / 'src'
    modules = {'.'.join(p.relative_to(base).with_suffix('').parts): p
               for p in (base / 'pecvd').rglob('*.py')}
    pending = ['pecvd.gevis', 'pecvd.gevis_training', 'pecvd.encoder_loading',
               'pecvd.__init__', 'pecvd.experiments.__init__']
    seen = set()
    # Encoder loader refreshes factories using dynamically composed names.
    pending += ['pecvd.' + n for n in ('pretrained_patchtst', 'legacy_patchtst',
        'official_patchtst', 'hybrid_patchtst', 'temporal_patchtst', 'multiscale_patchtst')]
    # Legacy training programs have imports not present in the inference facade.
    for notebook, program in (('STEP_1_train_patchtst_encoder.ipynb', 'encoder.py'),
                              ('STEP_2_train_gevis_backbone.ipynb', 'fusion.py')):
        path = root / 'notebooks' / notebook
        sources = []
        if path.exists():
            nb = json.loads(path.read_text(encoding='utf-8'))
            sources = [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']
        else:
            path = root / 'src/heads/_legacy_training' / program
            if path.exists():
                sources = [path.read_text(encoding='utf-8')]
        for source in sources:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('pecvd'):
                    pending.append(node.module)
                    pending += [node.module + '.' + a.name for a in node.names]
                if isinstance(node, ast.Import):
                    pending += [a.name for a in node.names if a.name.startswith('pecvd')]
    while pending:
        name = pending.pop()
        if name in seen or name not in modules:
            continue
        seen.add(name)
        package = name.rpartition('.')[0]
        for node in ast.walk(ast.parse(modules[name].read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                pending += [a.name for a in node.names if a.name.startswith('pecvd')]
            elif isinstance(node, ast.ImportFrom):
                prefix = (importlib.util.resolve_name('.' * node.level + (node.module or ''), package)
                          if node.level else node.module or '')
                pending.append(prefix)
                pending += [prefix + '.' + a.name for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in modules:
                    pending.append(node.value)
                if node.value.endswith('.py'):
                    for candidate, path in modules.items():
                        if path.name == node.value:
                            pending.append(candidate)
    return sorted(modules[name].relative_to(root).as_posix() for name in seen)


def active_assets(root):
    trees = [root / 'artifacts/gevis_v4_rich_ssl_final_v1_native_multiscale_blind_v3',
             root / ('artifacts/gevis_backbone_e5_v1_fdc_semantic_step1_bridge_all_types_v1_'
                     'real_step1_native_multiscale_blind_v3_center_relation_v14_training_source_v2'),
             root / 'artifacts/gevis_heads']
    return sorted(p for tree in trees if tree.exists() for p in tree.rglob('*') if p.is_file())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, default=Path('docs/audit.local'))
    p.add_argument('--backup', action='store_true')
    p.add_argument('--resume-inventory', action='store_true', help='Reuse a completed CSV when resuming asset backup')
    args = p.parse_args(); root = args.root.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    closure = set(dependency_closure(root))
    assets = active_assets(root); protected = {p.relative_to(root).as_posix() for p in assets}
    (out / 'fdc_dependencies.json').write_text(json.dumps(sorted(closure), indent=2), encoding='utf-8')
    # Virtual environments are third-party installations, not cleanup candidates.
    # Prune them before walking; never classify their individual bytecode for deletion.
    def project_files(directory):
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path.is_relative_to(out) or entry.name == '.git':
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in ('venvs', 'foundation_env', '.venv', 'venv', 'node_modules'):
                        continue
                    yield from project_files(path)
                elif entry.is_file(follow_symlinks=False):
                    yield path, entry.stat().st_size
    counts = Counter(); sizes = Counter()
    if args.resume_inventory:
        with (out/'file_inventory.csv').open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                counts[row['classification']] += 1
                sizes[row['classification']] += int(row['bytes'])
    else:
      with (out / 'file_inventory.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['path', 'classification', 'reason', 'bytes'])
        for path, size in project_files(root):
            rel = path.relative_to(root).as_posix(); parts = path.relative_to(root).parts
            if '__pycache__' in parts or path.suffix == '.pyc':
                label, reason = 'REMOVE', 'regenerable Python bytecode; not a source dependency'
            elif rel in protected:
                label, reason = 'KEEP', 'active Encoder/Fusion/head asset or its reproducibility metadata'
            elif rel.startswith(('data/', '.cache/ald-native-inputs/', '.cache/ald-experiment-features/')):
                label, reason = 'KEEP', 'private source data; never part of source release'
            elif rel.startswith('notebooks/training/') or (rel.startswith('notebooks/') and
                    (Path(rel).name.startswith('STEP_') or Path(rel).name == 'GEVIS_load.ipynb')):
                label, reason = 'KEEP', 'restored development/training notebooks and recovery provenance; retain'
            elif rel in closure:
                label, reason = 'REFACTOR', 'preserved FDC/training dependency; hash-bound compatibility code'
            elif rel.startswith(('src/GEVIS_Model/', 'src/heads/', 'examples/', 'config/', 'docs/', 'tests/portable/')) or rel in (
                'README.md', 'pyproject.toml', '.gitignore', '.gitlab-ci.yml',
                'notebooks/ALD_evaluate.ipynb', 'notebooks/CVD_evaluate.ipynb',
                'scripts/audit_repository.py', 'scripts/build_gitlab_source.py',
                'scripts/verify_model_parity.py', 'scripts/verify_saved_heads.py',
                'scripts/plot_training_history.py', 'scripts/plot_injection_examples.py', 'scripts/cleanup_unused.ps1'):
                label, reason = 'KEEP', 'portable source or current documentation'
            else:
                label, reason = 'ARCHIVE', 'not in conservative active dependency closure; retain pending review'
            counts[label] += 1; sizes[label] += size
            writer.writerow([rel, label, reason, size])
    checksums = []
    for source in assets:
        rel = source.relative_to(root)
        item = dict(path=rel.as_posix(), bytes=source.stat().st_size, sha256=checksum(source))
        if args.backup:
            target = native_path(out / 'backup' / rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(native_path(source), target)
            if checksum(target) != item['sha256']:
                raise ValueError(f'Backup checksum mismatch: {rel}')
        checksums.append(item)
    if args.backup:
        for folder in ('src', 'notebooks', 'config'):
            target = native_path(out / 'backup' / folder)
            if not target.exists():
                shutil.copytree(native_path(root / folder), target, ignore=shutil.ignore_patterns('__pycache__'))
    (out / 'protected_assets.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
    summary = dict(counts=dict(counts), bytes=dict(sizes), fdc_modules=len(closure),
                   protected_files=len(assets), backup_verified=args.backup, deleted_files=0)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
