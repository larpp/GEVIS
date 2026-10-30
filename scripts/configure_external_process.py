"""Create local-only external process mapping from explicit user choices.

Reads only the CSV header. Does not infer run identities, sensors, normal labels,
or window lengths. Writes only a new private config, never the source CSV.
"""
import argparse
import csv
import json
from pathlib import Path
import sys


def configure(root,source,*,run,step,order,sensors,steps,output=None):
    root=Path(root).resolve(); source=Path(source)
    if source.is_absolute() or '..' in source.parts: raise ValueError('Source must be a project-relative CSV')
    with (root/source).open(encoding='utf-8-sig',newline='') as stream: header=next(csv.reader(stream))
    required={run,step,order,*sensors}
    if not required<=set(header): raise ValueError('Explicit mapping references absent CSV columns; inspect header locally')
    if not sensors: raise ValueError('Explicit physical sensor columns required')
    payload={'schema':'gevis.process_config.v1','process_family':'PEALD','data_path':source.as_posix(),
        'artifacts':{'encoder':'gevis_peald_rich_ssl_v1','decision':'gevis_peald_evidence_v3','legacy_decision':'gevis_peald_legacy'},
        'columns':{'run':run,'step':step,'order':order},'sensor_columns':list(sensors),'steps':steps}
    destination=Path(output) if output else root/'.cache/external/peald/process.json'
    # Metadata/config derived from private inputs is private too; never put it
    # into the source release allowlist or replace a reviewed existing mapping.
    if not destination.resolve().is_relative_to(root/'.cache'):
        raise ValueError('Generated private mapping must remain under project .cache')
    if destination.exists(): raise FileExistsError('External mapping already exists; review it explicitly')
    from pecvd.process_config import ProcessConfig,ColumnConfig,ArtifactConfig,StepConfig
    cfg=ProcessConfig(payload['schema'],payload['process_family'],payload['data_path'],
        ArtifactConfig(**payload['artifacts']),ColumnConfig(**payload['columns']),
        tuple(StepConfig(**x) for x in steps),destination.resolve(),'pending',tuple(sensors))
    cfg.validate()
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='data/merge_peald.csv')
    parser.add_argument('--auto-peald',action='store_true',help='Use the validated equipment metadata preset and TRAIN-only sensor profiling')
    for key in ('run','step','order'): parser.add_argument('--'+key+'-column')
    parser.add_argument('--sensor-columns',nargs='+')
    parser.add_argument('--step-specs',help='JSON list of name/window_length/patch_length/stride objects')
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    if args.auto_peald:
        from pecvd.experiments.onboarding import prepare_peald
        prepare_peald(root,args.source)
        print('Private PEALD mapping prepared from equipment metadata and TRAIN-only sensor profiles. Models are not trained/evaluated yet.')
        return
    if not all((args.run_column,args.step_column,args.order_column,args.sensor_columns,args.step_specs)):
        parser.error('Use --auto-peald or supply explicit run/step/order/sensor-columns/step-specs')
    configure(root,args.source,run=args.run_column,step=args.step_column,order=args.order_column,
              sensors=args.sensor_columns,steps=json.loads(args.step_specs))
    print('Private mapping created. Set GEVIS_PROCESS_CONFIG to .cache/external/peald/process.json, then prepare STEP 1/2.')


if __name__=='__main__': main()
