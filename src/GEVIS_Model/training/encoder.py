"""Numeric SSL on explicit TRAIN/Validation windows using the preserved objective.

This is a new-data training entry point, not a claim to reproduce the historical
data sampling/scheduler. The original historical pipeline is separately exported.
"""
from dataclasses import dataclass
import copy
import numpy as np
import torch
from ..encoder.multiscale_patchtst import multiscale_objective


@dataclass
class EncoderTrainingConfig:
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = .0001
    weight_decay: float = .0001
    slope_weight: float = .25
    relationship_weight: float = .1
    variability_weight: float = .25
    peak_weight: float = .1
    max_grad_norm: float = .5
    patience: int = 12
    seed: int = 42


def fit_encoder(encoder, train, validation, *, config=None, device='cpu'):
    config = config or EncoderTrainingConfig()
    if min(config.epochs, config.batch_size, config.patience) < 1 or config.learning_rate <= 0:
        raise ValueError('Positive training budgets and learning rate required')
    arrays = []
    for name, data in [('TRAIN', train), ('Validation', validation)]:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3 or not len(data) or not np.isfinite(data).all():
            raise ValueError(f'{name}: finite [windows,time,sensors] arrays required')
        if data.shape[-1] != encoder.n_sensors or data.shape[1] % encoder.patch_len:
            raise ValueError(f'{name}: sensor count or patch length mismatch')
        arrays.append(torch.from_numpy(data))
    torch.manual_seed(config.seed)
    encoder.to(device).requires_grad_(True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history = []; best = float('inf'); best_state = None; stale = 0
    rng = np.random.default_rng(config.seed)
    for epoch in range(config.epochs):
        encoder.train(); loss_sum = 0.
        for ids in np.array_split(rng.permutation(len(arrays[0])),
                                 max(1, int(np.ceil(len(arrays[0]) / config.batch_size)))):
            x = arrays[0][ids].to(device)
            _, patches = encoder.patchify(x)
            mask = torch.rand((len(x), x.shape[-1] * patches), device=device) < .5
            optimizer.zero_grad(set_to_none=True)
            loss, _ = multiscale_objective(encoder, x, mask, config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite SSL loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.max_grad_norm)
            optimizer.step(); loss_sum += loss.item() * len(x)
        encoder.eval(); val_sum = 0.
        with torch.no_grad():
            for x in arrays[1].split(config.batch_size):
                x = x.to(device)
                _, patches = encoder.patchify(x)
                mask = ((torch.arange(x.shape[-1], device=device)[:, None]
                         + torch.arange(patches, device=device)[None]) % 2 == 0)
                mask = mask.reshape(1, -1).expand(len(x), -1)
                _, parts = multiscale_objective(encoder, x, mask, config)
                val_sum += parts['masked_mse'].item() * len(x)
        score = val_sum / len(arrays[1])
        history.append(dict(epoch=epoch + 1, train_loss=loss_sum / len(arrays[0]), validation_blind_mse=score))
        if score < best:
            best = score; best_state = copy.deepcopy(encoder.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    encoder.load_state_dict(best_state, strict=True)
    return encoder.eval(), history
