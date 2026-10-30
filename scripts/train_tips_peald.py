"""Run TIPS frozen STEP 1/2 readout training and held-out synthetic evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/experiments/tips_peald.json')
    parser.add_argument('--action', choices=['prepare', 'train', 'evaluate', 'train-evaluate'], default='train-evaluate')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--text-encoder', help='Explicit backend override; byte is for CPU smoke tests')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config.get('mode') == 'tips_frozen_step12':
        from pecvd.experiments.tips_frozen_training import run
        if args.text_encoder:
            parser.error('Frozen mode obtains text weights/architecture only from the source checkpoint')
    else:
        from pecvd.experiments.tips_training import run
    if args.text_encoder:
        config['text_encoder'] = args.text_encoder
    run(config, ROOT, action=args.action, device=args.device, output=args.output)


if __name__ == '__main__':
    main()
