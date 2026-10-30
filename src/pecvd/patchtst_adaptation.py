"""Domain objectives on official PatchTST outputs, without new neural layers.

Targets are used only AFTER blinded forward passes. No fault labels or anomaly
thresholds are produced here. Losses are in TRAIN-standardized sensor units.
"""
from dataclasses import dataclass, asdict
import math
import torch

CONTRACT = 'gevis.patchtst.domain_blind.v2'


@dataclass(frozen=True)
class DetailWeights:
    variability: float = .25
    slope: float = .25
    peak: float = .10
    relationship: float = .10

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Detail weights must be finite and nonnegative')


def checkerboard_mask(model, batch, patches, parity, device):
    sensor = torch.arange(model.n_sensors, device=device)[:, None]
    patch = torch.arange(patches, device=device)[None]
    return ((sensor + patch) % 2 == parity).reshape(1, -1).expand(batch, -1)


def blind_prediction(model, x):
    """Every output patch comes from a forward in which THAT patch is hidden.

The official adapter removes hidden samples before observed-window scaling.
Two complementary passes give the same input protocol in train/validation/test.
No clean prediction or visible target is spliced into the reconstruction.
"""
    patches = x.shape[1] // model.patch_len
    if patches < 2 or x.shape[1] % model.patch_len:
        raise ValueError('Blind reconstruction requires complete patches and visible context')
    prediction = None
    for parity in (0, 1):
        mask = checkerboard_mask(model, len(x), patches, parity, x.device)
        output = model(x, mask)
        prediction = torch.where(mask[..., None], output,
                                 torch.zeros_like(output) if prediction is None else prediction)
    return prediction


def detail_components(prediction, target, sensors):
    """Detail fidelity of a wholly blind reconstruction, not an anomaly score."""
    if prediction.shape != target.shape or prediction.ndim != 3 or sensors < 1:
        raise ValueError('Matching [batch, sensor*patch, width] tensors required')
    batch, tokens, width = target.shape
    if tokens % sensors or width < 2:
        raise ValueError('Complete sensor/patch layout required')
    pred = prediction.float().reshape(batch, sensors, tokens // sensors, width)
    truth = target.detach().float().reshape_as(pred)
    # Flatten ONLY patches belonging to the same sensor, preserving time order.
    series = pred.flatten(-2)
    expected = truth.flatten(-2)
    variability = .5 * ((pred.std(-1, unbiased=False) - truth.std(-1, unbiased=False)).square().mean()
                        + (series.std(-1, unbiased=False) - expected.std(-1, unbiased=False)).square().mean())
    slope = (series.diff(dim=-1) - expected.diff(dim=-1)).square().mean()
    # Worst sample error per patch prevents very short deviations being averaged away.
    peak = (pred - truth).square().amax(-1).mean()

    def unit_shape(z):
        centered = z - z.mean(-1, keepdim=True)
        return centered / centered.square().sum(-1, keepdim=True).clamp_min(1e-6).sqrt()

    # Same-time sensor correlations supervise the shared encoder; they do NOT
    # add cross-channel attention to the channel-independent official network.
    p, t = unit_shape(pred).permute(0, 2, 1, 3), unit_shape(truth).permute(0, 2, 1, 3)
    error = (p @ p.transpose(-1, -2) - t @ t.transpose(-1, -2)).square()
    valid = truth.var(-1, unbiased=False).permute(0, 2, 1) > 1e-6
    pairs = valid[..., :, None] & valid[..., None, :]
    pairs = pairs & ~torch.eye(sensors, device=pred.device, dtype=torch.bool)
    relationship = (error * pairs).sum() / pairs.sum().clamp_min(1)
    return {'blind_mse': (pred - truth).square().mean(), 'variability': variability,
            'slope': slope, 'peak': peak, 'relationship': relationship}


def adaptation_objective(model, x, weights=DetailWeights()):
    prediction = blind_prediction(model, x)
    target, _ = model.patchify(x)
    components = detail_components(prediction, target, model.n_sensors)
    loss = components['blind_mse']
    for key, weight in asdict(weights).items():
        loss = loss + weight * components[key]
    return loss, components
