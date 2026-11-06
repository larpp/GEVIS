"""Explicit TRAIN-only normalization for user-supplied finite numeric arrays."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SensorNormalizer:
    sensor_names: list
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train, sensor_names):
        values = np.asarray(train, dtype=float)
        names = list(sensor_names)
        if values.ndim < 2 or values.shape[-1] != len(names) or len(set(names)) != len(names):
            raise ValueError('Unique names matching the final sensor axis required')
        if not np.isfinite(values).all() or values.size == 0:
            raise ValueError('Finite nonempty TRAIN data required')
        flat = values.reshape(-1, len(names))
        center = np.median(flat, axis=0)
        q = np.quantile(flat, [.25, .75], axis=0)
        scale = (q[1] - q[0]) / 1.349
        scale = np.where(scale <= 1e-6, flat.std(0), scale)
        if (scale <= 0).any():
            raise ValueError('Constant TRAIN sensors must be removed explicitly')
        return cls(names, center, scale)

    def transform(self, values, sensor_names):
        if list(sensor_names) != self.sensor_names:
            raise ValueError('Sensor names/order differ from fitted TRAIN normalization')
        values = np.asarray(values, dtype=float)
        if values.shape[-1] != len(self.sensor_names) or not np.isfinite(values).all():
            raise ValueError('Finite complete sensor grid required')
        return ((values - self.center) / self.scale).astype(np.float32)
