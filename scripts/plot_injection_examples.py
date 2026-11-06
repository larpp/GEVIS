"""현재 이상 주입 함수로 설명용 파형과 한국어 구조 그림을 생성한다. 실제 센서 데이터는 읽지 않는다."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0,str(ROOT/'src'))
    from pecvd.experiments.boundary_injection import controlled_variant
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((name for name in ['Malgun Gothic','Noto Sans CJK KR','NanumGothic'] if name in available),None)
    if font is None:raise RuntimeError('한국어 글꼴이 필요합니다.')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'text.color':'#182e50',
        'axes.labelcolor':'#53657e','axes.edgecolor':'#b6c6dc','xtick.color':'#53657e','ytick.color':'#53657e'})
    destination=ROOT/'docs/injection_types';destination.mkdir(parents=True,exist_ok=True)
    x=np.arange(64);base=(.5*np.sin(x/6)+.12*np.sin(x/2)).astype(np.float32)[:,None]
    lower=base-.2;upper=base+.2
    def variant(kind,level):
        values,info=controlled_variant(base,lower,upper,sensor=0,kind=kind,level=level,
                                      seed=42,start=20,end=48,direction=1)
        if values is None:raise ValueError((kind,info))
        return values[:,0],info
    def draw(ax,values,title):
        ax.fill_between(x,lower[:,0],upper[:,0],color='#dce7ff',label='설명용 TRAIN MIN/MAX')
        ax.plot(x,base[:,0],'--',color='#8491a5',lw=1,label='주입 전 합성 파형')
        ax.plot(x,values,color='#ff6b35',lw=1.7,label='변형 파형')
        outside=(values<lower[:,0])|(values>upper[:,0])
        ax.scatter(x[outside],values[outside],s=10,color='#e83c40',zorder=3,label='범위 초과 시점')
        ax.set(title=title,xlabel='원본 시점 인덱스');ax.grid(alpha=.15)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(kind,label) in zip(axes.flat,[('level_shift','수준 변화'),('drift','기울기 변화'),
        ('spike','국소 변화'),('variance','변동성 변화'),('delay','응답 지연')]):
        y,info=variant(kind,1)
        draw(ax,y,label+(f" · 지연 {info['lag_samples']}시점" if kind=='delay' else ''))
    draw(axes.flat[-1],base[:,0]+.05*np.sin(x/4),'범위 안 변동 · 정상 라벨')
    fig.suptitle('MIN/MAX 정상 범위와 변화 유형 — 설명용 합성 파형',fontsize=17,fontweight='bold')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    for suffix in ['png','svg']:fig.savefig(destination/f'types.{suffix}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,4.5),sharey=True)
    for level,ax in enumerate(axes,1):
        y,info=variant('drift',level)
        draw(ax,y,f"{level}단계 · 실제 경계 초과량 {info['maximum_normalized_excess']:.2f}")
    fig.suptitle('같은 기울기 변화 모양 · 서로 다른 경계 초과 강도',fontsize=16,fontweight='bold')
    fig.text(.5,.025,'강도 = 최대 경계 초과량 / max(시점별 TRAIN 정상 폭, TRAIN 정규화 1단위). 실제 고장 심각도가 아닙니다.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.91))
    for suffix in ['png','svg']:fig.savefig(destination/f'drift_strength.{suffix}',dpi=170)
    plt.close(fig)
    sys.path.insert(0,str(ROOT/'scripts'))
    from draw_architecture import draw_architecture
    draw_architecture(ROOT/'docs/architecture/gevis.svg')
    (destination/'README.md').write_text('''# 합성 변화 도식의 의미

![변화 유형](types.png)

원본 센서 데이터나 모델 예측이 아닌 설명용 합성 파형입니다. 실제 학습에서 사용하는
`controlled_variant`로 수준·변동성·기울기·국소 변화·지연을 생성했습니다.
주입 전 파형과 정상 범위를 고정하고 선택한 연속형 센서만 변경합니다.
지연은 원본 샘플 이동으로 만들며 진폭을 키워 지연인 것처럼 표현하지 않습니다.

![경계 초과 강도](drift_strength.png)

강도는 최대 경계 초과량을 시점별 정상 범위 폭과 정규화 1단위 중 큰 값으로 나눈 값입니다.
동일한 모양이라도 0.5·1·2·4단계로 다르게 만들 수 있습니다. 물리적 고장 심각도나
일반적인 모든 공정 드리프트를 의미하지 않습니다. 구간 끝에서는 원본으로 복귀할 수 있습니다.

재생성에는 FDC 호환 소스와 문서 그림 의존성이 필요합니다.

```bash
python scripts/plot_injection_examples.py
```
''',encoding='utf-8')
    print('한국어 구조·변화 유형·경계 강도 도식 생성 완료')


if __name__=='__main__':main()
