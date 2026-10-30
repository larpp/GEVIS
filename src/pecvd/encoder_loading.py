"""Warm-session-safe offline checkpoint reconstruction for every consumer.

Do not keep a `from pretrained_patchtst import load_encoder` alias across notebook
upgrades. Refresh repository-owned model factories BEFORE obtaining that alias.
No Transformers/PyTorch module reload, downloads, training or checkpoint writes.
"""
import importlib
from threading import RLock

_LOCK=RLock()


def backbone_source_matches(checkpoint, source_path):
    """Exact source identity or ONE audited English-input-only successor.

    This migration adds an English-input guard and translates the inactive
    264-D synthetic grounding fixture. Valid English production computation and
    tensor layouts are unchanged. No checkpoint is edited. Any OTHER source
    digest, or non-English predecessor training, must be rejected.
    """
    import hashlib
    from pathlib import Path
    current=hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    recorded=checkpoint.get('source_module_sha256')
    if recorded==current:return True
    predecessor='2e68ede51e97aca5a9379b16b029b6ce5d8b900a36ebe6637e358f1270859b81'
    successor='1dc5479fd2524a540cf69f04da952400dd033e11265bb3768bc12c618084e489'
    return (recorded==predecessor and current==successor and
            checkpoint.get('question_language')=='English')


def refresh_encoder_modules():
    """Return the current local factory, repairing stale notebook imports."""
    importlib.invalidate_caches()
    modules={}
    for name in ('pretrained_patchtst','legacy_patchtst','official_patchtst','hybrid_patchtst','temporal_patchtst','multiscale_patchtst'):
        modules[name]=importlib.reload(importlib.import_module('pecvd.'+name))
    return modules['pretrained_patchtst'],modules['legacy_patchtst']


def load_encoder(checkpoint,legacy_class=None):
    # Serialise our own refresh+construction boundary. Never mutate a checkpoint
    # to conceal a backend/dimension/strict-state mismatch.
    with _LOCK:
        factory,legacy=refresh_encoder_modules()
        backend=checkpoint.get('encoder_backend','legacy')
        if backend not in factory.SUPPORTED_ENCODER_BACKENDS:
            raise ValueError(f'Unknown STEP 1 encoder backend {backend!r}; supported: {sorted(factory.SUPPORTED_ENCODER_BACKENDS)}')
        return factory.load_encoder(checkpoint,legacy_class or legacy.PatchEncoder)
