"""Data-path relocation failure messages (only when optional heads are present)."""
from pathlib import Path
import importlib.util
import pytest


def workspace_module():
    file=Path(__file__).resolve().parents[2]/'src/heads/workspace.py'
    if not file.is_file():
        pytest.skip('Optional FDC source tree not included')
    spec=importlib.util.spec_from_file_location('workspace_under_test',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_missing_external_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        workspace_module().relocate_workspace(tmp_path,tmp_path/'new',domain='PEALD',data_path=tmp_path/'absent')


def test_existing_workspace_not_overwritten(tmp_path):
    data=tmp_path/'data';data.mkdir();(data/'one.csv').write_text('a\n1\n',encoding='utf-8')
    with pytest.raises(FileExistsError):
        workspace_module().relocate_workspace(tmp_path,tmp_path,domain='PEALD',data_path=data)


def test_relocated_files_preserve_bytes(tmp_path):
    tmp_path=workspace_module().native_path(tmp_path)
    root=tmp_path/'root';(root/'config').mkdir(parents=True)
    (root/'config/settings.json').write_text('{}',encoding='utf-8')
    trees=['gevis_v4_rich_ssl_final_v1_native_multiscale_blind_v3',
           'gevis_backbone_e5_v1_fdc_semantic_step1_bridge_all_types_v1_real_step1_native_multiscale_blind_v3_center_relation_v14_training_source_v2',
           'gevis_heads']
    for tree in trees:
        folder=root/'artifacts'/tree;folder.mkdir(parents=True)
        (folder/'asset.txt').write_bytes(b'preserved model bytes')
    data=tmp_path/'external';data.mkdir();(data/'one.csv').write_bytes(b'sensor\n1\n')
    result=workspace_module().relocate_workspace(root,tmp_path/'destination',domain='PEALD',data_path=data,copy_assets=True)
    assert (result/'data/mt_nit/data_complete/one.csv').read_bytes()==b'sensor\n1\n'
    for tree in trees:
        assert (result/'artifacts'/tree/'asset.txt').read_bytes()==b'preserved model bytes'
