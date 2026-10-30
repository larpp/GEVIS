<!-- gevis-contract-sha256: abfc5e15f906a608bc94387c62a5771211ee86adb74ca1761de1c08e94351d11 -->
<!-- peald-encoder-routing: peald_native_validation_encoder_routes_v1 -->

# 공정별 실행 안내

**CVD_는 PECVD, ALD_는 PEALD입니다.** 평가와 외부 주입 CSV를 파일부터 구분합니다.

| 공정 | 무엇을 확인하나 | 실행 파일 |
| --- | --- | --- |
| CVD | 내부 합성 Test 성능 / 원본 Test 요청 속도 | [CVD_evaluate](CVD_evaluate.ipynb) |
| CVD | 제공받은 외부 주입 CSV의 예측·속도 | [CVD_predict_external_csv](CVD_predict_external_csv.ipynb) |
| ALD | 내부 합성 Test 성능 / 원본 Test 요청 속도 | [ALD_evaluate](ALD_evaluate.ipynb) |
| ALD | 제공받은 외부 주입 CSV의 예측·속도 | [ALD_predict_external_csv](ALD_predict_external_csv.ipynb) |

내부 합성 Test는 평가 코드가 원본 Test 파형을 변형하고 알려진 정답으로 채점합니다. 원본 실제 고장 정확도가 아닙니다. 외부 주입 CSV는 다른 곳에서 변형해 준 입력이며 여기서 다시 만들지 않습니다. 외부 파일은 정답 채점 없이 예측·속도만 확인합니다. 원본 Test 속도도 정확도 표와 다른 입력/참조 조건입니다.

## 외부 CSV 경로

- CVD: 기본 `data/injected_pecvd`. 승인된 PECVD 전처리 형식 CSV가 필요합니다. 폴더가 없으면 INPUT_DIRECTORY에 실제 경로를 지정합니다.
- ALD: 기본 `data/injected_data`. PEALD 원시 CSV를 사용합니다.

다른 공정 파일로 대신 실행하지 않습니다. 공정 선택을 바꿔 쓰는 공통 predict_csv 파일은 제거했습니다. 노트북과 출력 첫 줄에 대상 공정을 표시합니다. 기본은 1파일·최대8창이며 모든 지원 센서의 세 답변을 유지합니다. 전체 재생은 MAX_FILES와 MAX_WINDOWS_PER_FILE을 None으로 명시합니다.

## 팀 실험: 새 데이터를 실시간 입력처럼 재생하기

현재 외부 주입 CSV 시험은 새 입력을 준비할 때까지 보류합니다. 기존 ALD 실행 결과는 내부 결과 MD에 별도로 남겨뒀습니다.

1. 학습·Validation/모델 선택에 쓰지 않은 회차의 복사본으로 CVD/ALD 시험 CSV를 준비합니다. run·recipe·Step·시간 순서, 센서 이름·단위·샘플 간격을 유지하고 공정 경계를 이어 붙이지 않습니다.
2. 미주입 대조본과 변형본을 함께 만듭니다. 주입 센서·시작/끝 위치·유형·강도·단위·원본 연결 정보를 별도 정답 파일로 남깁니다. 생성 정답은 합성 변화의 정답이며 실제 고장 라벨과 구분합니다.
3. 공정에 맞는 CVD_predict_external_csv 또는 ALD_predict_external_csv에서 INPUT_DIRECTORY를 지정합니다. 모델·TRAIN 정규화·참조는 고정하며 새 CSV로 재학습하지 않습니다.
4. 기본 1파일·8창으로 먼저 확인한 뒤 필요하면 두 MAX 상한을 None으로 바꿔 전체를 재생합니다. 새 구간이 도착하는 순서처럼 창별로 예측·처리 범위·실패·요청 지연을 확인합니다.
5. 현재 노트북은 예측·속도만 반환합니다. 정확도·탐지율·오탐률·유형/강도 성능까지 비교하려면 별도 생성 정답을 연결하는 채점 절차를 추가합니다. 정답을 모델 입력으로 보내지 않습니다.

이는 **실시간 입력을 흉내 낸 CSV 순차 재생 실험**입니다. 실제 도착 간격에 맞춘 송신·대기열·장비 스트리밍·Edge 전체 지연 검증은 별도이며, 지금 노트북이 이미 수행한다고 해석하지 않습니다. 원본·시험 CSV·정답·결과는 로컬에서만 다룹니다.

## 연구 3개는 모두 CVD 내부 실험

- 질문 민감도: 같은 파형에서 질문만 바꿔 보는 **고정 모델** 시험.
- 합성변화 일반화: Test 파형에 코드로 변화를 만들어 보는 **고정 모델** 시험.
- 특징 기여(Feature ablation): 같은 합성 예제에서 입력 특징을 바꾸고 **임시 TRAIN 헤드**를 다시 학습하는 비교.

**이 세 연구는 제공받은 외부 CSV를 읽는 시험과 다릅니다.** 외부 CSV 전용 노트북은 고정 TRAIN 참조로 예측·시간만 반환하고 현재 채점하지 않습니다. [데이터·재학습·정답·목적 구분표](research/README.md#실험-구분표)에서 파일별 차이를 확인하세요. 기본 평가나 외부 예측을 위해 연구 세 개를 모두 돌릴 필요는 없습니다.

## 모델을 처음 준비할 때

1. [CVD STEP1](STEP_1_train_patchtst_encoder.ipynb): 회차 분리·TRAIN 정규화·인코더 학습.
2. [CVD STEP2](STEP_2_train_gevis_backbone.ipynb): 질문/수치 Fusion 학습.
3. [CVD STEP3](STEP_3_train_gevis_decision.ipynb): TRAIN 답변 헤드 학습·Validation 선택.
4. [CVD STEP3.1](STEP_3.1_run_gevis_inference.ipynb): 정확한 기존 모델 로드·추론.

기존 모델이 준비돼 있으면 이름 정리 때문에 재학습할 필요는 없습니다. ALD 추가 학습은 RUN_PEALD_TRAINING=True를 명시할 때만 합니다.

## 시간과 출력

한 요청은 완성된 모델 창에서 전체 선택 센서의 BOOLEAN·CHOICE·ORDINAL을 반환합니다. 파일 읽기·창 분리·장비 수집·큐·전송은 제외합니다. 물리cycle 전체/Edge 인수 시간이나 실제 장비 스트리밍 시험과 다릅니다. SHOW_DETAILS로 상세 표를 선택합니다.

[현재 계획](../_blueprint/experiment-plan.md) · [구조·과거 이름 대응](../_blueprint/gevis-experiment-guide.md) · [연구 안내](research/README.md)

## 현재 안내와 과거 기록

실행은 위 현재 파일명을 사용합니다. 옛 EXP 번호는 [대응표](../_blueprint/gevis-experiment-guide.md#6-실행-목적과-과거-번호-대응)에서 찾습니다. 현재 결과 요약·과거 수치는 로컬 `_evidence/local/`에, 주요 실패는 [FAILURE_LOG](../FAILURE_LOG.md)에 보존합니다. 이후 검증 방향은 [blueprint](../_blueprint/experiment-plan.md)를 봅니다.
