"""Portable config paths and streaming checksums."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    for key in ('manifest', 'backbone', 'data_path', 'checkpoint'):
        if value.get(key):
            item = Path(value[key]).expanduser()
            value[key] = str(item if item.is_absolute() else (path.parent / item).resolve())
    return value
