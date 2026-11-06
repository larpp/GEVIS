"""Extract head-independent features from an explicit standardized window pair."""
import argparse
import numpy as np
import torch
from GEVIS_Model import load_encoder_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--backbone', required=True)
    p.add_argument('--data-path', required=True, help='NPZ: current, reference, sensor_names')
    p.add_argument('--step', default='DEP.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--text-model-path')
    p.add_argument('--center', required=True)
    p.add_argument('--question', default='Is there a change in this interval?')
    args = p.parse_args()
    model, _ = load_encoder_bundle(args.manifest, args.backbone, step=args.step,
        device=args.device, text_model_path=args.text_model_path)
    with np.load(args.data_path, allow_pickle=False) as data:
        names = data['sensor_names'].astype(str).tolist()
        current = torch.as_tensor(data['current'], dtype=torch.float32, device=args.device)
        reference = torch.as_tensor(data['reference'], dtype=torch.float32, device=args.device)
    if current.ndim == 2:
        current = current[None]; reference = reference[None]
    with torch.inference_mode():
        output = model(current, reference, sensor_names=names, questions=[args.question],
                       answer_types=['BOOLEAN'], centers=[args.center], phase=args.step)
    print('Feature shape:', tuple(output.features.shape))


if __name__ == '__main__':
    main()
