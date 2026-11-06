"""Offline smoke example: untrained encoder, synthetic arrays, no private assets."""
import argparse
import torch
from GEVIS_Model import create_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.manual_seed(42)
    model = create_encoder(n_sensors=3, layers=1).to(args.device).eval()
    current = torch.randn(1, 16, 3, device=args.device)
    with torch.inference_mode():
        output = model(current, current * .9,
            sensor_names=['sensor_A', 'sensor_B', 'sensor_C'],
            questions=['Is there a change in this interval?'],
            answer_types=['BOOLEAN'], centers=['sensor_A'])
    print('UNTRAINED synthetic example; feature shape:', tuple(output.features.shape))


if __name__ == '__main__':
    main()
