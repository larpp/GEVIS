"""Retired notebook source for synthetic legacy-core regressions, never runtime."""
import json
from pathlib import Path


def legacy_exp_book(name):
    path = Path(__file__).parent / 'fixtures/exp_legacy_cells.json'
    return json.loads(path.read_text(encoding='utf-8'))['notebooks'][name]
