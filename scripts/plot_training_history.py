"""저장된 PECVD 인코더 학습 기록을 그림과 한국어 설명으로 변환한다. 재학습하지 않는다."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'artifacts/gevis_v4_rich_ssl_final_v1_native_multiscale_blind_v3'
    destination = root / 'docs/training'; destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((base/'encoder_manifest.json').read_text(encoding='utf-8'))
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ['Malgun Gothic','Noto Sans CJK KR','NanumGothic'] if n in available), None)
    if font is None:
        raise RuntimeError('한국어 그림을 위해 맑은 고딕 또는 Noto Sans CJK KR 글꼴을 설치하세요.')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b6c6dc',
        'axes.labelcolor':'#53657e','text.color':'#182e50','xtick.color':'#53657e',
        'ytick.color':'#53657e','grid.color':'#e1e8f3','grid.alpha':.65,'figure.facecolor':'white'})
    records = []
    for step,item in manifest['step_checkpoints'].items():
        file=base/item['path']; digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'인코더 체크섬 불일치: {step}')
        checkpoint=torch.load(file,map_location='cpu',weights_only=False)
        history=checkpoint['history']; selected=int(checkpoint['training_status']['best_epoch'])
        row=next(h for h in history if h['epoch']==selected)
        score=float(row.get('validation_blind_mse',row['validation_mse']))
        if not math.isclose(score,checkpoint['training_status']['best_validation_mse'],rel_tol=1e-6,abs_tol=1e-8):
            raise ValueError(f'선택 시점과 저장된 검증 점수가 다릅니다: {step}')
        records.append(dict(step=step,checkpoint=file.relative_to(root).as_posix(),sha256=digest,
            window_length=checkpoint['window_length'],patch_length=checkpoint['patch_length'],
            sensors=len(checkpoint['sensor_columns']),selected_epoch=selected,
            initial_validation_mse=checkpoint['training_status']['initial_validation_mse'],
            selected_train_mse=row['train_mse'],selected_validation_mse=score,
            test=checkpoint['test_metrics'],history=history))

    def draw(ax,record,train_key,val_key,ylabel):
        for key,label,color in [(train_key,'학습','#3975ff'),(val_key,'검증','#ff6b35')]:
            rows=[h for h in record['history'] if h.get(key) is not None and math.isfinite(float(h[key]))]
            ax.plot([h['epoch'] for h in rows],[h[key] for h in rows],label=label,color=color,lw=1.5)
        ax.axvline(record['selected_epoch'],color='#16a085',ls='--',lw=1.2,label=f"선택: {record['selected_epoch']}회")
        ax.set(title=record['step'],xlabel='학습 반복 횟수',ylabel=ylabel)
        ax.grid(True);ax.legend(fontsize=8)

    for kind,keys,title,ylabel in [
        ('reconstruction',('train_mse','validation_blind_mse'),'PECVD 공정별 가림 복원 오차 — 저장된 학습 기록','복원 오차 (MSE)'),
        ('objective',('train_loss','validation_loss'),'PECVD 공정별 학습 목적함수 — 저장된 학습 기록','목적함수 값')]:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,record in zip(axes.flat,records):draw(ax,record,*keys,ylabel)
        fig.suptitle(title,fontsize=17,fontweight='bold')
        fig.text(.5,.015,'체크포인트의 기록을 그대로 표시했습니다. 재학습·평활화·새 성능 측정은 하지 않았습니다.',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.95))
        for suffix in ('svg','png'):fig.savefig(destination/f'pecvd_{kind}.{suffix}',dpi=170)
        plt.close(fig)
    dep=next(r for r in records if r['step']=='DEP.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    draw(axes[0],dep,'train_loss','validation_loss','목적함수 값')
    draw(axes[1],dep,'train_mse','validation_blind_mse','복원 오차 (MSE)')
    axes[0].set_title('학습 목적함수');axes[1].set_title('가림 복원 오차')
    fig.suptitle('DEP 인코더 — 저장된 학습 이력',fontsize=16,fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.92))
    for suffix in ('svg','png'):fig.savefig(destination/f'dep_history.{suffix}',dpi=170)
    plt.close(fig)

    def sanitize(value):
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:sanitize(v) for k,v in value.items()}
        if isinstance(value,list):return [sanitize(v) for v in value]
        return value
    (destination/'history.json').write_text(json.dumps(sanitize(records),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# PECVD 시계열 인코더의 실제 학습 기록','',
        '기존 인코더 학습 단계(STEP1)에서 저장한 체크포인트의 학습 이력과 평가 집계입니다. 원본 CSV를',
        '읽거나 모델을 재학습하지 않았습니다. 각 파일의 SHA256을 인코더 목록과 대조한 뒤 생성했습니다.','',
        '## 공정별 가림 복원 오차','',
        '![공정별 가림 복원 오차](pecvd_reconstruction.png)','',
        '파란색은 저장된 학습 MSE, 주황색은 검증 가림 복원 MSE입니다. 초록 점선은 마지막 반복이',
        '아닌 실제 저장 가중치 선택 시점을 나타냅니다. 초기 검증값은 0회에 표시하며 기록에 없는 학습값은',
        '만들어 채우지 않습니다. 서로 다른 공정의 센서 분포와 구간 길이가 달라 MSE만으로 난이도를 단정하지 않습니다.','',
        '## 선택 시점과 저장된 Test 결과','',
        '| 공정 | 원본 시점 수 | 센서 수 | 선택 반복 | 선택 검증 MSE | Test MSE | 영값 기준선 대비 비율 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['step']} | {r['window_length']} | {r['sensors']} | {r['selected_epoch']} | {r['selected_validation_mse']:.6f} | {r['test']['blind_mse']:.6f} | {r['test']['relative_mse']:.6f} |")
    lines += ['', 'MSE는 TRAIN 기준 정규화 공간의 제곱 오차입니다. 비율은 저장된 Test 가림 복원 오차를',
        '영값 복원 기준선 오차로 나눈 값이며 작을수록 좋습니다. Test 집계는 저장된 값을 옮긴 것이고',
        '이번 작업에서 다시 평가한 결과가 아닙니다. 집계 표의 원본 표본 수는 이 지표 필드에 포함되어 있지 않습니다.',
        '복원 오차는 ALD 이상 탐지 F1이나 설비 고장 정확도가 아닙니다. DEP의 오차가 높다는 사실만으로',
        '다른 인코더보다 ALD 전이에 적합하다고 결론 내릴 수 없습니다.','',
        '## 목적함수와 복원 오차의 구분','',
        '![공정별 학습 목적함수](pecvd_objective.png)','',
        '목적함수에는 복원 MSE 외에 기울기·센서 관계·변동성·국소 극값·다중 스케일 항이 들어갑니다.',
        '따라서 목적함수와 MSE 곡선은 서로 다른 값입니다. 모델 선택은 저장된 검증 가림 복원 MSE를 따릅니다.','',
        '## ALD에 재사용하는 DEP 인코더','',
        '![DEP 인코더 학습 이력](dep_history.png)','',
        f"저장 가중치 선택 시점은 **{dep['selected_epoch']}회**, 검증 가림 복원 MSE는 **{dep['selected_validation_mse']:.6f}**입니다.",
        'ALD에서는 이 인코더와 융합 가중치를 고정하고 별도 판별 헤드를 학습합니다.','',
        '## 재생성','',
        '```bash','python -m pip install -e ".[docs]"','python scripts/plot_training_history.py --root .','```','',
        '현재 제조 체크포인트와 한국어 글꼴이 필요합니다. 배포본에 가중치는 포함되지 않습니다.',
        '그림의 원본 수치와 체크섬은 [학습 기록 JSON](history.json)에 저장했습니다.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('공정별 학습 기록 생성 완료:',len(records),'개 공정; 재학습 없음')


if __name__=='__main__':main()
