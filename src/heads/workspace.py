"""Relocate explicit datasets without changing historical preprocessing code."""
import json
import os
from pathlib import Path
import shutil


def native_path(value):
    path = Path(value).expanduser().resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        text = str(path)
        path = Path('\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text)
    return path


def relocate_workspace(root, destination, *, domain, data_path, copy_assets=False):
    """New workspace; hard-link read-only inputs/assets, optionally copy across disks.

    Outputs are created in destination; no source artifact is overwritten.
    Linked source files must stay immutable, as in the original experiments.
    """
    root = native_path(root); destination = native_path(destination)
    data_path = native_path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if destination.exists():
        raise FileExistsError(f'Use a new workspace path: {destination}')
    if domain == 'PEALD' and (not data_path.is_dir() or not list(data_path.glob('*.csv'))):
        raise ValueError('PEALD --data-path must be a data_complete directory with CSV files')
    if domain == 'PECVD' and not data_path.is_file():
        raise ValueError('PECVD --data-path must be the original source CSV')
    destination.mkdir(parents=True)

    def link_file(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError as error:
            if not copy_assets:
                raise OSError('Hard-link relocation failed (possibly different drives). '
                              'Use --copy-assets to copy inputs and model assets instead.') from error
            shutil.copy2(source, target)

    # Copy configuration so subsequent user edits cannot modify the original.
    shutil.copytree(root / 'config', destination / 'config')
    encoder = root / 'artifacts/gevis_v4_rich_ssl_final_v1_native_multiscale_blind_v3'
    backbone = root / ('artifacts/gevis_backbone_e5_v1_fdc_semantic_step1_bridge_all_types_v1_'
                       'real_step1_native_multiscale_blind_v3_center_relation_v14_training_source_v2')
    for tree in (encoder, backbone, root / 'artifacts/gevis_heads'):
        if not tree.is_dir():
            raise FileNotFoundError(f'Required preserved assets missing: {tree}')
        for source in tree.rglob('*'):
            if source.is_file():
                link_file(source, destination / source.relative_to(root))
    if domain == 'PEALD':
        for source in sorted(data_path.glob('*.csv')):
            link_file(source, destination / 'data/mt_nit/data_complete' / source.name)
    else:
        link_file(data_path, destination / 'data/all_processes_analysis.csv')
    (destination / 'workspace.json').write_text(json.dumps(dict(domain=domain,
        source_root=str(root), input_path=str(data_path), immutable_linked_inputs=True), indent=2), encoding='utf-8')
    return destination
