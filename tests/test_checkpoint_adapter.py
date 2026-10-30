"""Offline checkpoint/runtime compatibility; no pretrained downloads or data."""
import ast
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from pecvd.temporal_bridge import GatedTemporalAdapter, install_checkpoint_adapter


class CheckpointAdapterTests(unittest.TestCase):
    def test_saved_adapters_replace_stale_constructors_strictly(self):
        for width in (264,408,440):
            with self.subTest(width=width):
                saved=nn.Module()
                saved.numeric_input_norm=nn.LayerNorm(width) if width==264 else nn.Identity()
                saved.numeric=nn.Linear(width,16) if width==264 else GatedTemporalAdapter(width,16)
                runtime=nn.Module()
                runtime.numeric_input_norm=nn.LayerNorm(width)
                runtime.numeric=nn.Linear(width,16)
                install_checkpoint_adapter(runtime,saved.state_dict(),width,16)
                runtime.load_state_dict(saved.state_dict(),strict=True)
                x=torch.randn(2,3,width)
                torch.testing.assert_close(runtime.numeric(runtime.numeric_input_norm(x)),
                                           saved.numeric(saved.numeric_input_norm(x)))
                broken=dict(saved.state_dict());broken.pop(next(iter(broken)))
                with self.assertRaises(RuntimeError):runtime.load_state_dict(broken,strict=True)

    def test_unknown_adapter_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):install_checkpoint_adapter(nn.Module(),{},440,16)

    def test_shared_runtime_repairs_adapter_before_component_loading(self):
        root=Path(__file__).resolve().parents[1]
        doc=json.loads((root/'notebooks/STEP_3.1_run_gevis_inference.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code')
        tree=ast.parse(code)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_runtime')
        source=ast.unparse(fn)
        self.assertLess(source.index('install_checkpoint_adapter'),source.index('target.load_state_dict'))
        self.assertIn('importlib.reload(checkpoint_bridge)',source)
