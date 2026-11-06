"""처리 노트북의 공통 경로와 현재 판별 헤드 연결. 학습은 명시적으로만 실행한다."""
import json
import os
from pathlib import Path
import sys


def configure(project_path=None):
    source_root = Path(__file__).resolve().parents[2]
    settings = source_root / 'notebooks/training/workspace.local.json'
    if project_path is None:
        project_path = json.loads(settings.read_text(encoding='utf-8'))['project_path'] if settings.is_file() else source_root
    root = Path(project_path).expanduser().resolve()
    for required in ('src/pecvd/gevis.py', 'config/experiments/gevis_pipeline.json'):
        if not (root / required).is_file():
            raise FileNotFoundError(f'프로젝트 경로를 확인하세요: {root / required}')
    # 다른 프로젝트의 모듈과 가중치를 섞지 않는다.
    for name, module in tuple(sys.modules.items()):
        if name == 'pecvd' or name.startswith('pecvd.'):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(root / 'src'):
                raise RuntimeError('다른 프로젝트의 pecvd가 로드되어 있습니다. 커널을 재시작하세요.')
    if project_path != source_root or settings.is_file():
        settings.write_text(json.dumps({'project_path': str(root)}, ensure_ascii=False, indent=2), encoding='utf-8')
    os.chdir(root)
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    print('프로젝트:', root)
    return root


def load_heads(root, gevis, domain, checkpoints=None):
    """현재 evaluate 저장본을 검증해 로딩한다. 없거나 여러 개면 임의 선택하지 않는다."""
    import pandas as pd
    import torch
    from pecvd import gevis_training as training
    root = Path(root)
    domain = domain.upper()
    if domain not in ('PECVD', 'PEALD'):
        raise ValueError('DOMAIN은 PECVD 또는 PEALD로 지정하세요.')
    if checkpoints is None:
        checkpoints = {}
        for path in sorted((root / 'artifacts/gevis_heads' / domain.lower()).glob('*/answer_head.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            mode = payload['identity']['choice']['mode']
            if mode in checkpoints:
                raise ValueError(f'{mode} 저장본이 여러 개입니다. CHECKPOINTS에 경로를 지정하세요.')
            checkpoints[mode] = path
    if not checkpoints:
        raise FileNotFoundError(f'{domain} 헤드가 없습니다. STEP 3에서 ACTION="train"으로 학습하세요.')
    runtimes = {}
    rows = []
    for mode, path in checkpoints.items():
        runtime = training.load_head(root, path, gevis=gevis,
            allow_legacy_cvd=domain == 'PECVD', allow_legacy_ald=domain == 'PEALD')
        if runtime.domain != domain or runtime.module.FUSION_ANSWER_HEAD.choice.mode != mode:
            raise ValueError('선택한 헤드의 도메인 또는 CHOICE 방식이 다릅니다.')
        runtimes[mode] = runtime
        rows.append(dict(domain=domain, mode=mode, checkpoint=str(runtime.module.FUSION_ANSWER_PATH)))
    return runtimes, pd.DataFrame(rows)


def example_request(runtime):
    """로딩·추론 연결 검사용 참조 중앙 파형. 실제 Test 성능 측정용이 아니다."""
    import numpy as np
    for record in runtime.references['records']:
        step, recipe, loop, offset = record['key']
        targets = runtime.target_sensors.get(step, [])
        if targets:
            return dict(current=(np.asarray(record['lower']) + np.asarray(record['upper'])) / 2,
                step=step, recipe_key=recipe, loop_key=str(loop), offset_samples=int(offset),
                lineage_id='processing-artificial-connection-check', run_id='artificial-example',
                centers=targets[:1], sensor_columns=runtime.encoders[step][1]['sensor_columns'],
                standardized=True)
    raise ValueError('추론을 확인할 센서 참조가 없습니다.')


def plot_encoder_history(checkpoints):
    """저장된 학습 이력을 SVG로 표시한다. matplotlib DLL을 필요로 하지 않는다."""
    from html import escape
    import math
    from IPython.display import SVG, display
    charts = []
    for index, (step, checkpoint) in enumerate(checkpoints.items()):
        history = sorted(checkpoint['history'], key=lambda row: row['epoch'])
        values = [float(row[key]) for row in history for key in ('train_mse', 'validation_mse')]
        values = [v for v in values if math.isfinite(v)]
        if not values:
            continue
        lo, hi = min(values), max(values)
        span = max(hi - lo, 1e-12)
        first, last = history[0]['epoch'], history[-1]['epoch']
        lines = []
        for key, color in [('train_mse', '#2563eb'), ('validation_mse', '#ea580c')]:
            points = ' '.join(f"{45+290*(row['epoch']-first)/max(last-first,1):.2f},{175-140*(float(row[key])-lo)/span:.2f}"
                for row in history if math.isfinite(float(row[key])))
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        charts.append(f'<g transform="translate({index%3*370},{index//3*230})">'
            f'<text x="45" y="20">{escape(step)} — 복원 오차(MSE)</text>'
            '<path d="M45 35V175H335" fill="none" stroke="#9ca3af"/>'
            f'<text x="3" y="43" font-size="10">{hi:.3g}</text>'
            f'<text x="3" y="175" font-size="10">{lo:.3g}</text>'
            f'<text x="45" y="192" font-size="10">{first}</text>'
            f'<text x="300" y="192" font-size="10">{last} epoch</text>'
            + ''.join(lines) + '<text x="55" y="215" fill="#2563eb" font-size="11">TRAIN</text>'
            '<text x="135" y="215" fill="#ea580c" font-size="11">Validation</text></g>')
    height = max(1, (len(checkpoints) + 2) // 3) * 230
    display(SVG(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1110 {height}" '
        'style="background:white;color:#111827;font-family:sans-serif">' + ''.join(charts) + '</svg>'))
