# CVD 연구와 외부 CSV 시험의 차이

**Feature ablation은 판단 정보의 기여도 분석, synthetic change generalization은 고정 모델의 변화 대응 시험, external CSV는 별도로 준비한 파일의 추론 시험입니다.** CSV 형식인지보다 데이터 출처·재학습·정답 사용 여부가 핵심입니다.

## Feature가 무엇인가

여기서 **Feature(특징)는 모델이 파형을 읽고 만든 판단용 숫자 묶음**입니다. drift·spike는 파형에 넣는 변화의 종류이고, Feature는 그 파형을 인코더·Fusion이 처리한 결과입니다. Feature ablation은 이 숫자 묶음 중 일부를 답변 헤드에 전달하지 않고 비교하는 실험입니다.

```text
현재·참조 파형 → 인코더·질문 Fusion → 판단용 특징 → 답변 헤드 → 유무·유형·강도
     ↑                                ↑
Synthetic change가 바꾸는 곳          Feature ablation이 가리는 곳
```

| 실험 | 무엇을 바꾸나 | 알고 싶은 것 |
| --- | --- | --- |
| Synthetic change | 입력 파형의 drift·spike·지연 등의 조건 | 이런 파형 변화도 기존 모델이 판단하나? |
| Feature ablation | 같은 예제에서 헤드에 전달하는 특징 묶음 | 답변하려면 어떤 정보 조합이 유용한가? |

예를 들어 같은 drift 예제라도, Feature ablation에서는 전체 정보를 주는 조건과 선택 센서 표현만 주는 조건을 비교합니다. 조건마다 별도 임시 헤드를 TRAIN으로 학습하고 Validation으로 선택한 뒤 같은 Test 예제로 채점합니다. Synthetic change에서는 기존 모델을 그대로 두고 drift의 크기·위치·길이 같은 입력 조건을 바꿉니다.

## 실험 구분표

| 파일 | 데이터와 참조 | 이 실험에서 학습하나 | 정답·채점 | 확인 목적 |
| --- | --- | --- | --- | --- |
| [CVD_feature_ablation](CVD_feature_ablation.ipynb) | TRAIN·Validation·Test 각각의 원본 복사본에 동일 규칙으로 합성 변화 생성, 주입 전 파형 참조 | **특징 조건마다 임시 답변 헤드를 TRAIN으로 학습**, Validation으로 선택 | 알려진 합성 정답으로 Test 채점 | 어떤 특징이 답변에 기여하는가 |
| [CVD_synthetic_change_generalization](CVD_synthetic_change_generalization.ipynb) | Test 원본에 주입 크기·위치·방향·복합 조건을 바꾸어 생성, 주입 전 파형 참조 | **학습 없음**, 기존 모델 고정 | 알려진 합성 정답으로 채점 | 변화 조건이 달라도 기존 모델이 판단하는가 |
| [CVD_question_sensitivity](CVD_question_sensitivity.ipynb) | 같은 PECVD 통제 Test 파형에서 질문만 가리거나 교체 | **학습 없음**, 기존 모델 고정 | 같은 합성 정답으로 조건별 채점 | 질문이 답변에 영향을 주는가 |
| [CVD_predict_external_csv](../CVD_predict_external_csv.ipynb) | 별도로 준비한 PECVD CSV, 고정 TRAIN 참조 | **학습 없음**, 기존 모델 고정 | 현재는 정답 연결·채점 없음 | 실제 제공 파일의 예측·요청 지연 |
| [ALD_predict_external_csv](../ALD_predict_external_csv.ipynb) | 별도로 준비한 PEALD CSV, 고정 TRAIN 참조 | **학습 없음**, 기존 모델 고정 | 현재는 정답 연결·채점 없음 | 실제 제공 파일의 예측·요청 지연 |

## Feature ablation에서 실제로 하는 일

동일한 합성 예제를 공통으로 사용하고 전체 특징·Fusion 표현만·선택 센서의 학습된 표현만·위치 정보만 등의 조건으로 임시 헤드를 비교합니다. 각 특징 비교 조건에는 위치 정보가 공통으로 남습니다. 따라서 TRAIN뿐 아니라 Validation/Test에도 각 분할의 합성 예제가 있습니다. Test로 학습하지 않습니다.

조건 이름은 다음처럼 읽습니다. 모든 조건에 Step 안의 창 위치 두 값은 공통으로 남습니다.

| 조건 | 헤드가 받는 정보의 뜻 |
| --- | --- |
| full_retrained | Fusion·질문 표현, 선택 센서 평균 표현, Attention 가중 표현을 모두 사용 |
| fusion_query_only | 파형과 질문을 결합한 Fusion 표현 및 질문 표현 사용. 텍스트만 쓰는 조건이 아님 |
| selected_numeric_only | 선택 센서의 학습된 토큰을 단순 평균한 표현 사용. 원시 센서값만 넣는 조건이 아님 |
| attended_selected_only | 선택 센서 토큰에 Attention 비중을 적용해 모은 표현 사용 |
| position_only | Step 내부에서 창이 어디에 있는지 두 값만 사용. Step 번호가 아님 |

전체 특징을 계산한 뒤 제외할 부분을 0으로 가립니다. 따라서 이 결과는 **헤드에 어떤 정보를 제공하는 것이 유용한지**를 보여주며, 인코더·Attention을 실제 제거했을 때의 속도를 보여주지는 않습니다.

이 실험은 STEP 학습을 다시 실행하거나 실사용 인코더·Fusion·답변 헤드를 교체하지 않습니다. 비교용 임시 헤드만 학습하며 추가 복원 특징 비교도 선택 옵션입니다. 이상을 더 다양하게 넣어 기존 모델을 시험하려는 목적이면 synthetic_change_generalization을 선택합니다.

## 외부 CSV 시험 상태와 한계

외부 CSV 시험은 **새 데이터 준비 후 재개**합니다. 기존 ALD CSV는 사용자가 제거했고 이전 결과만 내부 결과 MD에 보존했습니다. 재개 시 공정별 전용 파일에서 실제 경로를 지정합니다.

“외부 CSV”라는 이름만으로 학습하지 않은 독립 입력이라고 보장되지 않습니다. 원본 회차가 학습·선택에 사용됐는지 확인해야 합니다. 실제 탐지율·오탐률·유형/강도 정확도를 확인하려면 주입 기록 또는 검토 정답을 별도로 연결해야 하며, 현재 예측 노트북이 자동 채점한다고 해석하지 않습니다.

기본 CVD_evaluate·ALD_evaluate의 성능표도 내부 합성 Test이며, 원본 Test 창의 속도표는 고정 TRAIN 참조를 사용하므로 입력·참조 조건이 다릅니다. 합성 변화 정답은 실제 공정 고장 라벨이 아닙니다. CSV 재생은 실제 장비 스트리밍 시험이 아닙니다.

[공정별 기본 실행](../README.md) · [상세 구조](../../_blueprint/gevis-experiment-guide.md)
