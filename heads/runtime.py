"""Explicit bridge to preserved FDC experiments (optional legacy source tree).

The compatibility implementation stays byte-for-byte at src/pecvd because old
checkpoint identities include source hashes. It is not part of the portable core.
"""
from pathlib import Path
import sys


def legacy_training(root):
    source = Path(root).resolve() / 'src'
    if not (source / 'pecvd/gevis_training.py').is_file():
        raise FileNotFoundError('FDC examples require a release built with --with-fdc')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from pecvd.gevis import load_gevis
    from pecvd import gevis_training
    return load_gevis, gevis_training
