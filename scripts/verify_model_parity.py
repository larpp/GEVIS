"""Compare original and portable computation using synthetic data and fixed weights.

No training, process CSV read, or checkpoint modification. Optional actual saved
weights test; JSON includes measured errors rather than inferred success.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saved-weights', action='store_true')
    p.add_argument('--output', type=Path, default=Path('docs/feature-parity.local.json'))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'src'))
    from GEVIS_Model import create_encoder, load_encoder_bundle
    from pecvd.gevis_runtime import GEVIS, numeric_state, numeric_token_descriptions
    from pecvd.experiments.fusion_answers import fusion_features
    from pecvd.answer_display import QUESTIONS
    torch.manual_seed(42)
    new = create_encoder(n_sensors=3, layers=1).eval()
    old_fusion = GEVIS(440, 192, 0., 'byte').eval()
    old_fusion.load_state_dict(new.fusion.state_dict(), strict=True)
    from pecvd.multiscale_patchtst import MultiscalePatchEncoder
    old_encoder = MultiscalePatchEncoder(3, 4, layers=1, dropout=0.).eval()
    old_encoder.load_state_dict(new.encoder.state_dict(), strict=True)
    names = ['sensor_A', 'sensor_B', 'sensor_C']
    x = np.random.default_rng(42).normal(size=(16, 3)).astype(np.float32)
    ref = x * .9

    def compare(model, encoder, fusion, sensor_names, current, reference, device):
        model.to(device).eval(); encoder.to(device).eval(); fusion.to(device).eval()
        module = SimpleNamespace(GEVIS_MODEL=fusion, DEVICE=torch.device(device),
            STEP_ENCODERS={'demo':(encoder,dict(sensor_columns=sensor_names))},
            numeric_state=numeric_state, numeric_token_descriptions=numeric_token_descriptions)
        with torch.no_grad():
            old, tasks = fusion_features(module,current,reference,'demo',[sensor_names[0]])
            out = model(torch.tensor(current[None], device=device), torch.tensor(reference[None],device=device),
                sensor_names=sensor_names, questions=[QUESTIONS[k] for k in ('BOOLEAN','CHOICE','ORDINAL')],
                answer_types=['BOOLEAN','CHOICE','ORDINAL'],centers=[sensor_names[0]]*3,phase='demo')
        delta = (old[:,:-2] - out.features.cpu()).abs().max().item()
        torch.testing.assert_close(old[:,:-2],out.features.cpu(),rtol=1e-5,atol=1e-5)
        return dict(max_absolute_error=delta, tasks=len(tasks), feature_dimension=out.features.shape[-1])

    report = {'synthetic_cpu': compare(new,old_encoder,old_fusion,names,x,ref,'cpu')}
    if torch.cuda.is_available():
        report['synthetic_cuda'] = compare(new,old_encoder,old_fusion,names,x,ref,'cuda')
    else:
        report['synthetic_cuda'] = {'status':'not_run','reason':'CUDA unavailable'}
    if args.saved_weights:
        from pecvd.gevis_runtime import load_source
        old = load_source(root,encoder_steps=['DEP.'])
        new, record = load_encoder_bundle(old.module.ENCODER_ROOT/'encoder_manifest.json',
                                         old.module.BACKBONE_PATH,step='DEP.')
        names = record['sensor_columns']
        x = np.random.default_rng(123).normal(size=(16,len(names))).astype(np.float32)
        report['saved_DEP_cpu'] = compare(new,old.encoders['DEP.'][0],old.module.GEVIS_MODEL,names,x,x*.9,'cpu')
        if torch.cuda.is_available():
            report['saved_DEP_cuda'] = compare(new,old.encoders['DEP.'][0],old.module.GEVIS_MODEL,names,x,x*.9,'cuda')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
