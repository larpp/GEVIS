"""Portable contracts and model parity; all inputs are synthetic."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from GEVIS_Model import create_encoder, load_encoder_bundle
from GEVIS_Model.preprocessing.normalization import SensorNormalizer


def test_head_independent_synthetic_features():
    torch.manual_seed(7)
    model = create_encoder(n_sensors=2, layers=1).eval()
    x = torch.randn(1, 16, 2)
    with torch.no_grad():
        out = model(x, x, sensor_names=['A', 'B'], questions=['Is there a change?'],
                    answer_types=['BOOLEAN'], centers=['A'])
    assert out.features.shape == (1, 768)
    assert torch.isfinite(out.features).all()
    assert not any('pecvd' == name or name.startswith('pecvd.') for name in sys.modules)


def test_missing_values_and_unknown_sensor_fail():
    model = create_encoder(n_sensors=2, layers=1).eval()
    x = torch.randn(1, 16, 2); x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Finite'):
        model(x, x, sensor_names=['A', 'B'], questions=['Change?'], answer_types=['BOOLEAN'], centers=['A'])


def test_train_normalizer_and_sensor_order():
    x = np.arange(24).reshape(12, 2)
    norm = SensorNormalizer.fit(x, ['A', 'B'])
    np.testing.assert_allclose(norm.center, np.median(x, axis=0))
    with pytest.raises(ValueError, match='order'):
        norm.transform(x, ['B', 'A'])
    with pytest.raises(ValueError, match='Constant'):
        SensorNormalizer.fit(np.ones((4, 2)), ['A', 'B'])


def test_missing_manifest_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_encoder_bundle(tmp_path/'missing.json', tmp_path/'missing.pt')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_forward():
    model = create_encoder(n_sensors=2, layers=1).cuda().eval()
    x = torch.randn(1, 16, 2, device='cuda')
    with torch.no_grad():
        out = model(x, x, sensor_names=['A', 'B'], questions=['Change?'], answer_types=['BOOLEAN'], centers=['B'])
    assert out.features.is_cuda
