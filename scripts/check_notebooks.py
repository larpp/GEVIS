"""평가 노트북의 경로·초기화·모델 로딩·선택적 전처리를 확인한다. 학습은 금지한다."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--workspace',type=Path)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--notebook',choices=['ALD_evaluate.ipynb','CVD_evaluate.ipynb'])
    args=p.parse_args();source=args.source_root.resolve();workspace=(args.workspace or source).resolve()
    if args.notebook is None:
        for name in ['CVD_evaluate.ipynb','ALD_evaluate.ipynb']:
            command=[sys.executable,str(Path(__file__).resolve()),'--source-root',str(source),
                     '--workspace',str(workspace),'--notebook',name]
            if args.prepare:command.append('--prepare')
            subprocess.run(command,check=True)
        return
    notebook=json.loads((source/'notebooks'/args.notebook).read_text(encoding='utf-8'))
    code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    for cell in code:ast.parse(cell)
    os.chdir(source/'notebooks')
    context={'__name__':'notebook_path_check','display':lambda value:None}
    setup=code[0].replace('WORKSPACE_PATH = None','WORKSPACE_PATH = '+repr(str(workspace)),1)
    exec(compile(setup,args.notebook+' setup','exec'),context)
    assert context['ROOT']==workspace
    assert context['SOURCE']==source/'src'
    def forbidden(*a,**kw):raise RuntimeError('이 검사에서는 학습을 실행하지 않습니다.')
    training=context['training'];training.train_ald=forbidden;training.train_cvd=forbidden
    settings=next(c for c in code if c.startswith('ENCODER=') or c.startswith('TRAINING='))
    exec(compile(settings,args.notebook+' settings','exec'),context)
    if 'GEVIS' not in context:
        load=next(c for c in code if c.startswith('GEVIS=load_gevis'))
        exec(compile(load,args.notebook+' load','exec'),context)
    report=dict(notebook=args.notebook,syntax=True,source_path=True,workspace_path=True,
                encoder_count=len(context['GEVIS'].encoders),training_executed=False)
    if args.prepare:
        prepare=next(c for c in code if c.startswith('PREPARED=training.prepare_'))
        exec(compile(prepare,args.notebook+' preparation','exec'),context)
        prepared=context['PREPARED']
        report['prepared_routes']=len(prepared.windows)
        report['run_counts']={s:len(v) for s,v in prepared.split_runs.items()}
        report['windows']={s:sum(len(pool.get(s,[])) for pool in prepared.windows.values())
                           for s in ['train','validation','test']}
        if args.notebook.startswith('ALD'):
            roles=training.sensor_roles(prepared)
            assert (roles.role=='prediction_target').any()
            report['sensor_roles_valid']=True
    output=source/'docs'/('notebook-'+args.notebook.split('_')[0].lower()+'.local.json')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
