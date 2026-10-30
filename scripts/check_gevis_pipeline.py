"""Read-only readiness audit; never trains or exports private experiment results."""
from pathlib import Path
import argparse,sys,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pecvd.experiments.pipeline_audit import audit_pipeline


def main():
    sys.path.insert(0,str(ROOT))
    from scripts.check_gevis_docs import check_docs
    errors=check_docs(ROOT)
    if errors:
        for error in errors:print('Documentation needs revision:',error)
        return 1
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['legacy','pretrained','official','hybrid','portable','temporal','multiscale'])
    parser.add_argument('--legacy-profile',action='store_true',help='Inspect original baseline paths without notebook comparison settings')
    args=parser.parse_args()
    if args.variant:os.environ['GEVIS_MODEL_VARIANT']=args.variant
    if args.legacy_profile:
        os.environ['GEVIS_DISCOVERY_TRAINING']='reference_v11'
        os.environ.pop('GEVIS_DISCOVERY_POLICY_PATH',None)
    if not args.legacy_profile:
        from pecvd.experiments.pipeline_settings import configure_pipeline
        configure_pipeline(ROOT)
    rows=audit_pipeline(ROOT,variant=args.variant)
    for row in rows:print(row['stage'],row['status'],row['path'],row['detail'],sep=' | ')
    return int(any(row['status']!='READY' for row in rows))

if __name__=='__main__':raise SystemExit(main())
