"""Build a new, allowlisted GitLab source tree; never copy data, weights or history."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from audit_repository import dependency_closure


def paths(root, with_fdc=False):
    names = {'README.md', 'FAILURE_LOG.md', 'pyproject.toml', '.gitignore', '.gitattributes', '.gitlab-ci.yml',
             'checkpoints/README.md', 'scripts/build_gitlab_source.py', 'scripts/audit_repository.py',
             'scripts/plot_injection_examples.py', 'scripts/plot_training_history.py',
             'scripts/draw_architecture.py'}
    for folder in ('GEVIS_Model', 'heads', 'examples', 'configs', 'docs', 'tests/portable'):
        for path in (root / folder).rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                    and path.suffix not in ('.pyc', '.pt', '.npz')
                    and not path.name.endswith('.local.json')
                    and not any(part.endswith('.local') for part in path.parts)):
                if not with_fdc and (path.is_relative_to(root/'heads') or
                    path.name in ('run_pecvd.py', 'run_peald.py', 'train_historical.py')):
                    continue
                names.add(path.relative_to(root).as_posix())
    if with_fdc:
        for required in ('src/pecvd/gevis.py', 'src/pecvd/gevis_training.py'):
            if not (root/required).is_file():
                raise FileNotFoundError(f'Optional FDC source is unavailable: {required}')
        names.update(['scripts/verify_model_parity.py', 'scripts/verify_saved_heads.py'])
        names.add('scripts/check_notebooks.py')
        names.update(['scripts/check_processing.py', 'processing/README.md', 'processing/workflow.py'])
        names.update('processing/' + filename for filename in (
            'STEP_1_train_patchtst_encoder.ipynb', 'STEP_2_train_gevis_backbone.ipynb',
            'STEP_3_train_gevis_decision.ipynb', 'STEP_3.1_run_gevis_inference.ipynb', 'GEVIS_load.ipynb'))
        names.update(['tests/test_ald_input_store.py', 'tests/test_ald_run_lineage.py',
                      'tests/test_ald_head_experiments.py', 'tests/test_decision_training.py'])
        names.update(dependency_closure(root))
        names.update(['src/pecvd/__init__.py', 'config/experiments/gevis_pipeline.json',
                      'config/processes/pecvd.json', 'notebooks/CVD_evaluate.ipynb',
                      'notebooks/ALD_evaluate.ipynb', 'notebooks/README.md'])
        # Empty __init__ files are needed on older Python/package tooling.
        for path in (root/'src/pecvd').rglob('__init__.py'):
            names.add(path.relative_to(root).as_posix())
    return sorted(name for name in names if (root/name).is_file())


def check_source(name, data):
    if name.endswith(('.py','.json','.md','.toml','.yml','.ipynb')):
        text = data.decode('utf-8')
        if re.search(r'(?:glpat|ghp)-[A-Za-z0-9_-]{16,}', text):
            raise ValueError(f'Possible access token in {name}')
        if re.search(r'[A-Za-z]:[\\/](?:Users|Projects)[\\/]', text):
            raise ValueError(f'Personal absolute path in {name}')
        if re.search(r'(?<![\d.])(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])', text):
            raise ValueError(f'Possible internal address in {name}')


def build(root, destination, with_fdc=False, refresh=False):
    root = root.resolve(); destination = destination.resolve()
    previous = None
    if destination.exists() and not refresh:
        raise FileExistsError(f'Destination must be new: {destination}')
    if refresh:
        if destination not in (root/'.gitlab-release/GEVIS-GitLab', root/'.gitlab-release/GEVIS-Encoder'):
            raise ValueError('Refresh is restricted to the two generated GitLab exports')
        previous = json.loads((destination/'release_manifest.json').read_text(encoding='utf-8'))
        if previous.get('source_only') is not True:
            raise ValueError('Not a generated source-only export')
        for item in previous['files']:
            file = (destination/item['path']).resolve()
            file.relative_to(destination)
            if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Export was edited; refusing to overwrite: {item["path"]}')
    entries = []
    for name in paths(root, with_fdc):
        data = (root/name).read_bytes()
        if name.endswith('.ipynb'):
            notebook = json.loads(data)
            for cell in notebook['cells']:
                cell.pop('attachments', None)
                if cell['cell_type'] == 'code':
                    cell['outputs'] = []; cell['execution_count'] = None
            notebook['metadata'].pop('widgets', None)
            data = (json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode('utf-8')
        check_source(name, data)
        entries.append((name,data))
    destination.mkdir(parents=True, exist_ok=refresh)
    if previous is not None:
        current_names = {name for name, _ in entries}
        # Only unchanged files created by this builder may be retired. This
        # cannot operate on project configs, caches, data, or unlisted files.
        for item in previous['files']:
            if item['path'] not in current_names:
                (destination/item['path']).unlink()
    manifest = []
    for name,data in entries:
        target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    (destination/'release_manifest.json').write_text(json.dumps(dict(
        source_only=True, fdc_compatibility=with_fdc, files=manifest), indent=2), encoding='utf-8')
    print(f'Built {len(entries)} source files: {destination}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--with-fdc',action='store_true',help='Include hash-bound legacy FDC implementation and sanitized evaluation notebooks')
    p.add_argument('--refresh',action='store_true',help='Refresh an unchanged manifest-bound export under .gitlab-release')
    args=p.parse_args(); build(args.root,args.output,args.with_fdc,args.refresh)
