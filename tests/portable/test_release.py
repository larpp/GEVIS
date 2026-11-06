"""Source export excludes private assets and strips notebook execution state."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def builder():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('source_builder', SCRIPTS/'build_gitlab_source.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_export_excludes_private_files(tmp_path):
    root=tmp_path/'source';root.mkdir()
    (root/'README.md').write_text('GEVIS',encoding='utf-8')
    (root/'data').mkdir();(root/'data/private.csv').write_text('private',encoding='utf-8')
    (root/'artifacts').mkdir();(root/'artifacts/model.pt').write_bytes(b'private')
    (root/'docs/audit.local').mkdir(parents=True)
    (root/'docs/audit.local/private.md').write_text('private audit',encoding='utf-8')
    output=tmp_path/'release'
    builder().build(root,output)
    assert not (output/'data').exists()
    assert not (output/'artifacts').exists()
    assert not (output/'docs/audit.local').exists()
    assert (output/'README.md').is_file()


def test_existing_destination_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        builder().build(tmp_path,tmp_path)


def test_notebook_outputs_removed(tmp_path):
    root=tmp_path/'source';(root/'notebooks').mkdir(parents=True)
    (root/'src/pecvd').mkdir(parents=True)
    for name in ('gevis.py','gevis_training.py'):
        (root/'src/pecvd'/name).write_text('# fixture',encoding='utf-8')
    nb=dict(cells=[dict(cell_type='code',source=['print(1)'],outputs=[{'secret':'private'}],execution_count=8)],metadata={'widgets':{'private':1}})
    (root/'notebooks/CVD_evaluate.ipynb').write_text(json.dumps(nb),encoding='utf-8')
    output=tmp_path/'release';builder().build(root,output,with_fdc=True)
    result=json.loads((output/'notebooks/CVD_evaluate.ipynb').read_text(encoding='utf-8'))
    assert result['cells'][0]['outputs']==[]
    assert result['cells'][0]['execution_count'] is None
    assert 'widgets' not in result['metadata']
