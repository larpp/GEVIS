"""Preserved CVD/ALD head training and single-window inference, without notebooks."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import legacy_training
from .workspace import relocate_workspace


def main(domain):
    p = argparse.ArgumentParser(description=f'{domain} FDC downstream example')
    p.add_argument('--root', type=Path, default=Path.cwd(), help='Preserved source/assets workspace')
    p.add_argument('--data-path', type=Path, help='Explicit source CSV or data_complete directory')
    p.add_argument('--workspace', type=Path, help='New workspace for a relocated dataset')
    p.add_argument('--copy-assets', action='store_true', help='Allow copies if hard links fail')
    p.add_argument('--config', type=Path, help='JSON domain config (domain/encoder validated)')
    p.add_argument('--action', choices=['train', 'predict'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--window', type=Path, help='Inference NPZ: current,sensor_columns')
    p.add_argument('--metadata', type=Path, help='Inference JSON: step,recipe_key,loop_key,offset_samples,lineage_id')
    p.add_argument('--standardized', action='store_true', help='Window already uses packaged TRAIN normalization')
    p.add_argument('--allow-legacy-cvd', action='store_true')
    p.add_argument('--allow-legacy-ald', action='store_true')
    args = p.parse_args()
    cfg = {}
    if args.config:
        cfg = json.loads(args.config.read_text(encoding='utf-8'))
        if cfg.get('domain') != domain:
            p.error('Config domain differs from runner')
        if domain == 'PEALD' and cfg.get('encoder', 'DEP.') != 'DEP.':
            p.error('This preserved ALD example uses DEP. only')
        if args.data_path is None and cfg.get('data_path'):
            args.data_path = (args.config.resolve().parent / cfg['data_path']).resolve()
    load_gevis, training = legacy_training(args.root)
    root = args.root.resolve()
    if args.data_path:
        if not args.workspace:
            p.error('--data-path requires --workspace (a new directory)')
        if args.action == 'predict':
            p.error('Predict reads --window/--metadata, not raw --data-path')
        root = relocate_workspace(root, args.workspace, domain=domain,
                                  data_path=args.data_path, copy_assets=args.copy_assets)
    if args.action == 'train':
        gevis = load_gevis(root, encoder_steps=['DEP.'] if domain == 'PEALD' else None)
        if domain == 'PEALD':
            policy = training.PreprocessingConfig(schema_columns=cfg.get('schema_columns'),
                max_window_length=cfg.get('max_window_length', 64))
            native = training.NativeConfig(excluded_months=tuple(cfg.get('excluded_months', [])),
                                           recipe_suffix=cfg.get('recipe_suffix', ''))
            choices = [training.ChoiceConfig(mode=m) for m in cfg.get('choice_modes', ['softmax','grouped','sigmoid'])]
            prepared = training.prepare_ald(root, gevis, preprocessing=policy, native=native)
            results = training.train_ald(root, gevis, prepared, preprocessing=policy, native=native,
                choices=choices, training=training.TrainingConfig(epochs=40,patience=8,batch_size=256,max_seconds=600.))
        else:
            prepared = training.prepare_cvd(root, gevis)
            results = training.train_cvd(root, gevis, prepared,
                training=training.TrainingConfig(epochs=80,patience=12,batch_size=256,max_seconds=600.))
        print(results.summary.to_string(index=False)); print(results.checkpoints)
    else:
        if not all((args.checkpoint, args.window, args.metadata)):
            p.error('Predict requires --checkpoint, --window and --metadata')
        runtime = training.load_head(root, args.checkpoint, allow_legacy_cvd=args.allow_legacy_cvd,
                                     allow_legacy_ald=args.allow_legacy_ald)
        if runtime.domain != domain:
            p.error('Checkpoint belongs to another domain')
        meta = json.loads(args.metadata.read_text(encoding='utf-8'))
        with np.load(args.window, allow_pickle=False) as data:
            reply = training.Session(runtime).process_window(current=data['current'],
                sensor_columns=data['sensor_columns'].astype(str).tolist(),
                standardized=args.standardized, **meta)
        print(json.dumps(reply, ensure_ascii=False, indent=2))
