"""Existing tensor operations shared by training diagnostics and runtime.

The last two state channels are current/reference patch reconstruction MSEs.
Their signed difference is not |latent - reconstruction| or equipment fault probability.
"""
import math
import torch


def semantic_comparison_features(current, reference, state, patch_count):
    if current.shape != reference.shape or current.ndim != 3 or patch_count < 1:
        raise ValueError('Expected matching batch/time/sensor windows and positive patch count')
    if current.shape[1] < 2 or state.shape[1] != current.shape[2] * patch_count:
        raise ValueError('Window/state patch contract mismatch')
    outputs = []
    residual = (state[..., -2] - state[..., -1]).reshape(len(current), current.shape[2], patch_count)
    for scope in range(3):
        start, end = ((0, current.shape[1]) if scope == 0 else
                      (0, max(2, current.shape[1] // 3)) if scope == 1 else
                      (max(0, 2 * current.shape[1] // 3), current.shape[1]))
        a, b = current[:, start:end], reference[:, start:end]
        level = a.mean(1) - b.mean(1)
        variability = a.std(1, unbiased=False) - b.std(1, unbiased=False)
        slope = (a[:, -1] - a[:, 0]) - (b[:, -1] - b[:, 0])
        point = (a - b).abs().mean(1)
        p0 = int(start / current.shape[1] * patch_count)
        p1 = max(p0 + 1, int(math.ceil(end / current.shape[1] * patch_count)))
        outputs.append(torch.stack([level, variability, slope, point, residual[:, :, p0:p1].mean(2)], -1))
    return torch.stack(outputs, 1)
