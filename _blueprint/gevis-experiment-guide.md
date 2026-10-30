# GEVIS 실험 상세 기준

기준: 2026-10-23 · Direction D0011 / `decision_system=fusion_answers`. 빠른 실행 목록은 [실행 안내](../notebooks/README.md), 향후 설계는 [blueprint](experiment-plan.md)를 봅니다.

## 1. 모델이 판단하는 것

현재·참조 시계열을 인코더로 수치 토큰화하고 고정 질문·선택 센서 문맥을 Fusion/Cross-attention으로 결합합니다. 학습된 답변 헤드가 logits/softmax/argmax로 BOOLEAN·CHOICE·ORDINAL을 반환합니다. Attention은 입력마다 계산되는 결합 비중이며 주입 정답·고장 원인이 아닙니다.

`GEVIS` 클래스만으로 CSV 읽기·원본 정렬·채점까지 수행하지 않습니다. 모델은 인코더+Fusion+헤드, 실행 경계는 `online_inference.InferenceSession.process_window`입니다. 원본·주입 기록은 모델 입력이 아닙니다. 맞는 모델·문맥이 없으면 미평가로 남깁니다.

| 답변 | 클래스 |
| --- | --- |
| BOOLEAN | No / Yes |
| CHOICE | No clear change / Mean level / Variability / Trend slope / Local change / Delay |
| ORDINAL | No change / Small / Moderate / Large / Very large |

모두 Fast의 출력입니다. 문서 F1≥0.96, 전체 세 답변 요청 평균≤50ms가 프로젝트 목표이며 개별 수락 정의·물리cycle/Edge 조건은 별도 확정합니다. Precision은 미래 연속값·단위·horizon을 가진 다른 과제입니다.

## 2. 학습·기존 합성 Test의 변화 생성

생성기는 모델 외부 `injection.py`, `fusion_answers.paired_variants` 등에 있습니다. 원본 복사본을 변형하고 실제 변경 마스크로 선택 센서별 정답을 만듭니다. 다른 센서만 변하면 선택 센서의 정답은 변화 없음입니다. 외부 입력 CVD_predict_external_csv / ALD_predict_external_csv는 이 주입기를 실행하지 않습니다.

| 변화 | 연산의 의미 | 주의 |
| --- | --- | --- |
| level shift | 구간에 strength×센서척도 가산 | 기준값 대비 수준 변화 |
| drift | 구간 안에서 0→strength×척도 선형 변화 | 아래 drift 절 참고 |
| variance | 표준편차 strength×척도의 잡음 가산 | 잡음 부호 반전은 분산 감소가 아님 |
| spike | 구간 중간 한 점에 가산 | 실제 바뀐 샘플이 정답 |
| delay | native 샘플을 뒤로 지연, 첫 값 유지 | 원형 이동·재보간 아님 |
| stuck / relationship | 공용 primitive 존재 | 현재 기본 CHOICE 학습 범위와 동일시하지 않음 |

기본 amplitude 단계는 .5/1/2/4, delay는 native1~4샘플입니다. 표준·가변 길이·양/음 방향과 음성 대조군을 구분합니다. paired 기본은 equal1 + amplitude4종×3 + delay2의 명목15변형이며 두 센서에 세 질문을 만듭니다. 명목 비율은 질문 구성의 비율이지 실제 공정 이상 비율이 아닙니다. no-op·상수·미지원은 실제 마스크/분모를 확인합니다.

ORDINAL은 요청한 합성 진폭 또는 lag 단계로 물리 고장 심각도가 아닙니다. 외부 약/중/강과 기존4단계를 자동 연결하지 않습니다. 복합 변화에 단일 유형·강도 정답이 없으면 해당 합성 비교에서 미채점으로 표시합니다.

## 3. drift 기준

구간 [a,b)에서 `x'[a:b]=x[a:b]+linspace(0,A,b-a)`, `A=strength×scale` 형태입니다. 음의 방향은 원본 대비 변형을 반전합니다. 구간 밖은 원본으로 돌아가므로 끝 경계에 복귀 점프가 생길 수 있습니다.

이 A는 종점 편차이며 단위시간당 물리 drift 속도가 아닙니다. 구간이 짧으면 level shift와 구분이 어려울 수 있습니다. 현재 최종 Trend slope 답은 학습 헤드의 CHOICE 선택이며 수동 기울기 임계값 판정이 아닙니다. 실제 허용 방향·변화율·지속시간·센서 관계는 전문가 정답 규약으로 별도 정의해야 합니다.

## 4. 전처리·참조·학습 경계

회차·recipe·Step 발생 구간을 혼합하지 않습니다. 센서 순서·단위·창 길이·patch와 TRAIN 정규화를 유지합니다. 새 입력으로 평균/표준편차를 재학습하지 않습니다. 같은 run 참조는 제외하고 맞는 TRAIN 문맥을 선택합니다. 없으면 미평가입니다.

PECVD는 저장된 Step별 인코더/창과 source/PM·위치 문맥을 사용합니다. PEALD는 회차 분리 뒤 TRAIN 연속 길이·cadence·정규화를 결정하고 기본 PECVD DEP 인코더/Fusion을 동결합니다. 공통 PEALD 헤드와 Validation에서 채택한 Step 헤드를 사용합니다. Step 헤드 선택은 source 인코더 교체가 아닙니다.

PEALD 현재 센서 후보는 Temp_/MFC/AUX/VG와 일부 명시 열, _Set_ 제외 등의 이름 규칙과 TRAIN 유한 비율·분산/표준편차 조건을 사용합니다. 100개 상한은 없습니다. 짧은 Step은 cadence·최소2patch 조건도 확인합니다. 설정값·상수·다른 이름의 측정값을 더 활용하는 것은 전문가 역할 검토와 새 입력 버전 비교가 필요한 후속 과제입니다.

학습은 TRAIN, 상태/후보 선택은 Validation, Test는 평가입니다. 과거 Test를 참고해 개발했으므로 기록상 교집합이 없어도 최종 미열람 holdout으로 선언하지 않습니다.

## 5. STEP별 과정

| 단계 | 입력·계산 | 핵심 출력 |
| --- | --- | --- |
| STEP1 | 회차 분할·TRAIN 정규화·Step별 복원 학습 | 인코더·전처리 규약 |
| STEP2 | 시계열 토큰·질문·문맥 Fusion 학습 | backbone |
| STEP3 | 인코더/Fusion 동결·TRAIN 답변 학습·Validation 선택 | 세 답변 헤드 |
| STEP3.1 | 정확히 맞는 기존 모델 로드·예측 | 선택 센서의 답변 |
| STEP4 | 합성 정답 Test + 원본 Test의 TRAIN 참조 속도 | 성능표와 별도 지연표 |

STEP4 성능은 원본 그대로의 실제 고장 정확도가 아닙니다. 기본으로 bootstrap·전체 혼동표·중복 원본 예측을 실행/표시하지 않습니다. 자세히 볼 때 SHOW_DETAILS, RUN_BOOTSTRAP, RUN_ORIGINAL_PREDICTIONS 등을 선택합니다. 속도는 워밍업 뒤 반복 측정하며 요청 단위를 고정합니다.

## 6. 실행 목적과 과거 번호 대응

파일 통합과 기존 번호형 파일 제거는 완료됐습니다. 아래 과거 이름은 원표를 찾기 위한 대응이며 실행할 파일 목록이 아닙니다.

| 현재 파일 | 과거 이름 | 목적 |
| --- | --- | --- |
| CVD_evaluate | STEP4 + EXP4/5의 기본 확인 | PECVD 합성 성능·원본 답변 예시·전체 요청 지연 |
| ALD_evaluate | EXP6 | PEALD 기존 모델·합성 성능·원본 요청 지연 |
| CVD_predict_external_csv / ALD_predict_external_csv | EXP7/8/9 | 공정별 전용 CSV 예측·지연·처리 범위. 채점 없음 |
| research/CVD_question_sensitivity | EXP1 | 질문 가림·교체의 영향 |
| research/CVD_synthetic_change_generalization | EXP2 | 진폭 .25/.75/1.5/3/6, 위치/기간·음의 방향·복합·delay 등 |
| research/CVD_feature_ablation | EXP3 | 특징 기여. 추가 복원 특징 재학습은 선택 |

### 재학습과 정답을 혼동하지 않기

Feature는 drift·spike 같은 변화 유형이 아니라 파형을 읽은 뒤 만든 판단 정보입니다. **Synthetic change는 입력 파형을 바꾸고, Feature ablation은 답변 헤드에 제공할 정보를 가립니다.** [개념도와 조건별 설명](../notebooks/research/README.md#feature가-무엇인가)을 기준으로 읽습니다.

[실험 구분표](../notebooks/research/README.md#실험-구분표)에 데이터 출처·참조·재학습·채점·목적을 정리했습니다. Feature ablation은 같은 합성 예제에 특징 조건만 달리해 임시 TRAIN 헤드를 학습합니다. 합성변화 일반화는 Test 변화 조건을 달리하고 기존 모델을 고정해 채점합니다. 외부 CSV는 별도 파일과 TRAIN 참조를 사용하며 모델을 고정하지만 현재 정답을 연결하지 않아 채점하지 않습니다. 세 결과를 같은 평가로 합치지 않습니다. 외부 입력의 원본 회차 독립성은 별도로 확인합니다.

기존 EXP4의 파형·복원·Attention 그림과 원본 경보율 검사는 fast_experiments.run_original_review/plot_review에 보존한 별도 분석 도구입니다. STEP4에 모든 그림이 기본으로 나온다는 뜻은 아닙니다. helper의 원본 경보율은 검증된 정상 오탐률이 아닙니다.

change_generalization은 생성기 변형과 대응 원본의 paired 시험이며 외부 TRAIN 참조 시험과 다릅니다. stuck·분산 감소·lead·새 lag/지속시간 등 모든 미학습 변화를 검증한 것은 아닙니다. 질문 교란과 특징 마스킹도 E5 제거·pooling 대체 구조 재학습을 대신하지 않습니다.

ALD_evaluate는 기존 체크포인트만 로드하도록 RUN_PEALD_TRAINING=False가 기본입니다. 학습하려면 명시적으로 True로 바꿉니다. 중복 원본 재생과 bootstrap은 기본 꺼져 있고 별도 반복 속도 측정은 유지합니다.

## 7. 외부 입력의 빠른 확인

MAX_FILES=1, MAX_WINDOWS_PER_FILE=8이 기본이며 WINDOW_STRIDE=None으로 겹치지 않는 native 창을 처리합니다. 센서 수·세 답변 수는 줄이지 않습니다. 상한은 추론 요청 수를 제한하며 선택한 CSV 읽기·구간 준비 시간까지 없애는 것은 아닙니다. 전체 실행은 두 상한을 None으로 명시합니다.

coverage는 준비 가능한 창·실제로 처리한 창·상한으로 요청하지 않은 창을 표시합니다. 일부 파일·창의 결과를 전체 파일 성능으로 해석하지 않습니다. 기본 화면은 앞18개 답변이며 전체 예측은 REPORT.details 또는 PREDICTIONS에 있습니다.

CVD_predict_external_csv / ALD_predict_external_csv에는 원본 정렬·생성 정답·사후 채점 셀이 없습니다. 별도 helper는 기존 재현/회귀 용도로 남아 있지만 주 실행에서 호출하지 않습니다. 정답 없는 예측에 성능 지표를 붙이지 않으며 계산을 생략했다고 정확도가 개선된 것으로 보지 않습니다.

## 8. 성능·속도를 해석하는 기준

성능은 답변별 F1·전체/변화 사례 정확도·클래스 지지·ordinal MAE와 한 단계 이내 정확도를 구분합니다. 질문 행은 독립 회차가 아닙니다. 추가 불확실성은 run 단위 bootstrap으로 계산하지만 기본 실행은 아닙니다. 순전히 미변경인 대조군의 완벽 점수를 전체 유형/강도 성능으로 주장하지 않습니다.

request_ms는 정규화·참조 선택·인코더·Fusion·헤드·답변 구성을 포함합니다. CSV 읽기·모델 로드·window 구성·전송·대기열은 제외됩니다. 첫 요청의 문맥 준비 비용과 워밍업 뒤 반복 측정을 구분합니다. 센서/답변 수로 나눈 시간을 한 요청 실측으로 사용하지 않습니다.

새 입력은 정답 없이 예측할 수 있으나 그 정확성은 독립 정답이 있어야 확인할 수 있습니다. 높은 합성 paired 성능과 외부 입력 경보, 전체 실험 실행시간과 요청 지연은 서로 다른 증거입니다. Orchestrator 큐·timeout/포화·identity·복원·Shadow와 대상 Edge 전체cycle 인수는 별도 검증입니다.

## 9. 기록과 앞으로 할 것

기존 원표는 로컬 `_evidence/local/`에 유지합니다. 이전 출력 복구는 `.cache/notebook-purpose-reset-20261023/before/`, 이번 이름/통합 복구는 `.cache/notebook-consolidation-20261023/before/`에 있으며, 현재 실행 화면에는 오래된 출력이 남지 않게 비웠습니다. 이번 정리가 새 학습·정확도·지연 측정은 아닙니다.

향후 핵심은 **auto labeling 정답 기반 GEVIS 이상 탐지 학습·평가, 깊은 추론이 필요한 사례의 추가 경로, seed·입력·환경·반복 조건을 고정한 재현 실험**입니다. 상세 순서는 [계획](experiment-plan.md#4-향후-방향--우선-실험할-세-가지)에 있습니다. 기존 가중치를 바꾸거나 이 기능을 구현 완료한 것으로 기록하지 않습니다.

## 10. 개발 흐름과 참고 자료

기존 날짜별 Work와 프로젝트 로컬 대화 기록에서 확인한 흐름입니다. **자료를 검토한 것, 구현한 것, 성능이 검증된 것은 서로 구분**합니다.

| 시기 | 검토·변경 흐름 | 현재 해석 |
| --- | --- | --- |
| 9월29~30일 | TCN으로 센서 다음 값 예측, residual 보완과 마지막 값 기준선 비교 | 시계열 모델의 초기 실험. 충분한 예측 성능을 얻지 못해 PatchTST 표현 학습 방향으로 전환 |
| 10월1일 | PatchTST 패치 기반 Transformer와 복원 학습 방향으로 전환 | 현재 인코더 설계의 출발점. 논문 원형과 프로젝트 변경 구현은 구분 |
| 10월2~6일 | 수치 토큰 max pooling + 텍스트 mean pooling 후 결합하는 초기 구조 검토, 질문 기반 Cross-attention으로 변경 | pooling 전에 센서·시간 토큰을 유지하고 질문에 따라 결합하는 방향. 텍스트도 max pooling했다는 설명은 부정확 |
| 10월8~21일 | DUST·Laya·공개 PatchTST 가중치·NCAD·CAPMix·약한 라벨 생성 자료 등 검토 | 각각 학습 방식·질문 기반 결정·사전학습·이상 주입·라벨 준비의 참고 후보. 모두 현행에 채택한 것은 아님 |
| 10월21~22일 | 통계 threshold·분위수 조정과 별도 판정 경로 검토 | 룰베이스와 차이가 부족하다는 판단으로 최종 판정 방식에서 폐기. 10월22일 결정은 사용자 확인 기준 |
| 현재 | 질문 Fusion과 학습 헤드로 합성 변화 판독, PECVD·PEALD 평가·새 CSV 예측 분리 | 실제 이상 탐지 검증은 auto labeling 정답 기반 후속 실험으로 진행할 계획 |

### 공유받아 검토한 주요 자료

날짜는 당시 공유 기록 기준입니다. 링크는 참고 출처이며 특정 논문 전체를 구현했다거나 해당 모델을 이겼다는 의미가 아닙니다.

| 공유 시기 | 자료 | 프로젝트에서 검토한 주제 |
| --- | --- | --- |
| 10월1~2일 | [PatchTST — A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730) | 패치 기반 시계열 표현·자기지도 학습 방향 |
| 10월2일 | [TypeSafe Jev](https://docs.typesafe.ai/introduction/quickstart) · [primitives](https://docs.typesafe.ai/primitives) · [confidence](https://docs.typesafe.ai/confidence) · [patterns](https://docs.typesafe.ai/patterns) | 질문별 정형 답변·신뢰도·사용 방식 참고. 시계열 구현을 제공하는 논문으로 취급하지 않음 |
| 10월8일 | [DUST](https://github.com/qlabs-eng/dust) | 학습 방식의 대안 검토 |
| 10월12·21일 | [Laya](https://github.com/NandhaKishorM/laya) | 경량 질문 기반 결정 모델의 후보 검토 |
| 10월14일 | [IBM Granite PatchTST FM-r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) | 공개 시계열 사전학습 가중치·호환 검토. 현행 모든 인코더와 동일하다고 가정하지 않음 |
| 10월21일 | [NCAD — Neural Contextual Anomaly Detection](https://arxiv.org/abs/2107.07702) · [CAPMix v1](https://arxiv.org/html/2509.06419v1) | 이상 탐지·합성 변화 관련 접근 검토 |
| 10월21일 | [Snorkel 튜토리얼](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb) | auto labeling·약한 감독 학습의 준비 방식 검토 |
| 10월21일 | [Lev](https://huggingface.co/interfaze-ai/lev) | 정형 결정 모델의 후보 검토 |

현재 적용·향후 계획은 [blueprint](experiment-plan.md), 검증과 전달은 [검증 안내](../_docs/verification.md), 변경 이력은 [개발 기록](../FAILURE_LOG.md)을 기준으로 봅니다.
