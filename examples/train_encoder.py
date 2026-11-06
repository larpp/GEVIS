"""Train the native numeric encoder on user-prepared, independently split NPZ windows."""
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from GEVIS_Model import create_encoder
from GEVIS_Model.training.encoder import fit_encoder, EncoderTrainingConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True, help='NPZ containing train and validation arrays')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    with np.load(args.data_path, allow_pickle=False) as data:
        train, val = data['train'], data['validation']
    torch.manual_seed(42)
    model = create_encoder(n_sensors=train.shape[-1])
    config = EncoderTrainingConfig(epochs=args.epochs)
    encoder, history = fit_encoder(model.encoder, train, val, config=config, device=args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema='gevis.numeric_ssl.portable.v1', model_state=encoder.cpu().state_dict(),
        n_sensors=train.shape[-1], patch_length=encoder.patch_len,
        model_config=dict(d_model=96, nhead=8, layers=3, dropout=0.),
        training_config=asdict(config), history=history), output)
    print('Saved new numeric encoder (Fusion is not trained):', output)


if __name__ == '__main__':
    main()
