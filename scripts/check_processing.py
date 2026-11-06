"""STEP 저장본 로딩과 추론 연결만 검사한다. 학습 셀은 실행하지 않는다."""
import argparse
import ast
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=['step1', 'step2', 'PECVD', 'PEALD'])
    parser.add_argument('--notebook', choices=['inference', 'step3', 'load'], default='inference')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = {'step1': 'STEP_1_train_patchtst_encoder.ipynb',
             'step2': 'STEP_2_train_gevis_backbone.ipynb'}
    filename = names.get(args.target, {'inference': 'STEP_3.1_run_gevis_inference.ipynb',
        'step3': 'STEP_3_train_gevis_decision.ipynb', 'load': 'GEVIS_load.ipynb'}[args.notebook])
    notebook = json.loads((root / 'notebooks/training' / filename).read_text(encoding='utf-8'))
    namespace = globals().copy()
    namespace['display'] = lambda value: None
    import torch
    def forbidden(*args, **kwargs):
        raise RuntimeError('이 검사에서는 학습·가중치 저장을 실행할 수 없습니다.')
    torch.optim.Adam.step = forbidden
    torch.optim.AdamW.step = forbidden
    torch.optim.SGD.step = forbidden
    torch.save = forbidden
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    if args.target in names:
        # 기존 저장본 존재를 먼저 확인한 뒤 모든 셀을 재사용 분기로 실행한다.
        indexes = [i for i, cell in enumerate(notebook['cells']) if cell['cell_type'] == 'code']
        for index in indexes:
            exec(compile(''.join(notebook['cells'][index]['source']), filename, 'exec'), namespace)
            if index == 9 and args.target == 'step1':
                assert namespace['REUSE_SAVED_ENCODERS'], '저장된 인코더를 찾을 수 없습니다.'
            if index == 7 and args.target == 'step2':
                assert namespace['REUSE_SAVED_BACKBONE'], '저장된 Fusion을 찾을 수 없습니다.'
        key = 'REUSE_SAVED_ENCODERS' if args.target == 'step1' else 'REUSE_SAVED_BACKBONE'
        assert namespace[key], '저장본 재사용 실패'
    else:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source']).replace("DOMAIN = 'PECVD'", f'DOMAIN = {args.target!r}')
                exec(compile(source, filename, 'exec'), namespace)
    report = dict(target=args.target, notebook=filename, saved_loading=True, training_executed=False)
    (root / 'notebooks/training' / f'check-{args.target.lower()}-{args.notebook}.local.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('CHECK_OK', args.target, flush=True)


if __name__ == '__main__':
    main()
