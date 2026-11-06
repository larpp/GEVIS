"""Smoke-test preserved CVD/ALD head packages on reference-derived inputs, no training."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pecvd.gevis import load_gevis
from pecvd import gevis_training as training


def main():
    gevis = load_gevis(ROOT)
    report = []
    for checkpoint in sorted((ROOT/'artifacts/gevis_heads').glob('*/*/answer_head.pt')):
        print('Checking', checkpoint.parent.name, flush=True)
        runtime = training.load_head(ROOT,checkpoint,gevis=gevis,
                                     allow_legacy_cvd=True,allow_legacy_ald=True)
        session = training.Session(runtime)
        for source in runtime.references['records']:
            step,recipe,loop,offset = source['key']
            record = runtime.reference_bank.get(source['key'])
            if record is not None:
                break
        else:
            raise ValueError('No supported reference record')
        baseline = ((record['lower']+record['upper'])/2).astype(np.float32)
        centers = runtime.target_sensors[step][:2]
        observations = [baseline, baseline.copy()]
        sid = runtime.encoders[step][1]['sensor_columns'].index(centers[0])
        observations[1][:,sid] = record['upper'][:,sid]+5.
        outcomes=[]
        for observed in observations:
            reply=session.process_window(current=observed,step=step,recipe_key=recipe,
                loop_key=loop,offset_samples=int(offset),lineage_id='synthetic-independent-smoke',
                centers=centers,standardized=True)
            if reply['status']!='OK':
                raise ValueError(reply)
            outcomes.append(dict(status=reply['status'],answers=len(reply['answers']),
                positive_sensors=sum(a['prediction']==1 for a in reply['answers']
                                     if a['answer_type']=='BOOLEAN')))
        report.append(dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(),domain=runtime.domain,
            mode=runtime.module.FUSION_ANSWER_HEAD.choice.mode,step=step,centers=centers,
            input='reference midpoint and one synthetic out-of-envelope sensor',outcomes=outcomes,
            historical_prediction_parity='not_verified'))
    output=ROOT/'docs/head-smoke.local.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Checked saved heads:',len(report),flush=True)


if __name__=='__main__':
    main()
