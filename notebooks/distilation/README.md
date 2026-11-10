# control48 GEVIS 증류·병합

실행은 [01_control48.ipynb](01_control48.ipynb), 병합본 추론만 할 때는 [02_merged_inference.ipynb](02_merged_inference.ipynb)를 사용합니다. 01 하나에서 Student 학습·병합·Validation 배치 선택·고정 배치 Test까지 순서대로 실행합니다. 다른 노트북의 커널 객체는 필요 없습니다. 커널은 `Python (PECVD)`입니다. 프로젝트 위치는 자동 탐색하고 데이터·가중치가 다른 위치에 있으면 `WORKSPACE_PATH`에 지정합니다.

원본 STEP 3·ALD_evaluate·CVD_evaluate와 `src/pecvd`는 직접 수정하지 않았습니다. 이 파일들은 원본 Teacher/Head의 생성·호환 로딩 경로입니다. Student를 학습하고 병합하는 단계는 별도 응용 API [heads.distilation](../../src/heads/distilation/__init__.py)로 연결합니다. 원본 Head의 차원·파라미터·정규화 버퍼·답변별 allowed 마스크를 그대로 유지합니다. Student만 optimizer에 넣으며 원본 파일을 덮어쓰지 않습니다.

## 고정 control48

| 항목 | 설정 |
|---|---|
| Student | 내부48차원, Transformer2층, 4-head Cross-Attention |
| Head 연결 | 내부 출력을 Teacher 특징 차원 `8D+10`으로 투영해 원본 Head에 전달 |
| 질문 | 원래 Teacher의 질문+유형 벡터를 고정 버퍼로 복사 |
| 학습 | seed42, lr0.001, epochs40, patience8, batch_cases8, 최대1200초 |
| legacy 손실 | 0.5×정답 CE + 0.5×T² KL(T=2) + 1.0×특징 Smooth L1 |
| 특징 손실 | 원래 Head의 feature_scale로 정규화. 마지막 경계·시간 통계10개 제외 |
| 매핑 | 원본5유형 argmax `(1,3)/(2)/(4,5)` → 수준·추세 / 변동성 / 국소 변화·지연 |

Student와 경계 통계 함수는 사용자 첨부 `component_distillation.py`의 구조를 그대로 옮겼습니다. [참고 원문](references/component_distillation.py.txt)과 [최종 매핑 참고](references/component_distillation_v7.py.txt)는 실행하지 않는 자료입니다. 참고 파일이 의존하던 외부 실험 버전 전체를 가져오지 않고 현재 입력·Head API로 통합했습니다. 기존 검증 수치가 재현됐다는 뜻은 아닙니다.

5채널 입력은 사용자 확인 원문대로 `values / lower / upper / (values-lower)/scale / (values-upper)/scale`, `scale=max(upper-lower,1)`입니다. 부호를 유지하고 5채널 전체에 `asinh` 후 `[sensors,5,time]`으로 변환합니다. 이 함수 안에서 별도 표준화를 하지 않습니다. 입력 센서·native 시간 순서를 유지합니다.

CHOICE 매핑은 **5유형 최종 argmax 이후**에만 적용합니다. 3출력 Head를 새로 학습하거나 그룹 확률을 합산하지 않습니다. 반환값의 `original_five_prediction/answer/probabilities`에 원래 답변을 남기고 3그룹 확률은 제공하지 않습니다. BOOLEAN·ORDINAL은 원래 Head의 gate·class-zero 제외 규칙을 유지합니다. Student가 근사한 특징을 받으므로 실제 예측이 Teacher와 완전히 같다는 보장은 별도 검증 대상입니다.

## 실행 순서와 저장

1. 원본 5유형 Softmax Teacher를 공식 로더로 불러옵니다. 같은 도메인의 후보가 여러 개면 Softmax를 구분하고 같은 방식이 여러 개일 때만 번호를 선택합니다. 기존 제한적 호환 옵션을 사용하며 소스 해시·참조·상태 로딩 검사는 유지합니다.
2. KD용 원본 Window를 준비합니다. 원래 `PREPARED`가 메모리에 있으면 `PREPARED_IN_MEMORY`로 전달할 수 있습니다. 새로 준비해도 저장 Teacher의 센서 순서·TRAIN 정규화·native 길이/간격·Run 분할·source identity와 다르면 중단합니다. TRAIN 정상 참조는 다시 fitting하지 않습니다.
3. Student만 학습하거나 검증한 live Student·legacy 기록·원래 Head를 함께 전달해 추가 학습 없이 병합합니다. live Student의 원래 Head와 질문 벡터가 선택한 Teacher와 다르면 거부합니다.
4. Validation의 세 과제 F1·원본 클래스 support·생성/제외 현황과 실제 요청 지연을 확인합니다. epoch는 원래 control48처럼 **원본5유형 기준 Validation**으로 선택하고, 선택 이후 최종 답변·보고에만 3유형 매핑을 적용합니다.
5. 선택을 고정하고 새 `artifacts/gevis_distilation/<고유ID>/control48.pt`에 병합본을 저장합니다. 원본 Head state를 그대로 포함하고 Student·저장 입력 메타데이터·참조·매핑·코드/내용 무결성도 묶습니다. 기존 Teacher·Head·완료 포인터는 덮어쓰지 않습니다.
6. 병합본을 Teacher/E5 없이 재로딩하고 선택 모델과 답변 일치를 확인합니다. 이어 Test를 한 번 평가하고 Validation/Test F1·평균/p95 표를 별도 CSV로 저장합니다. 결과는 비공개입니다.

현재 데이터 예산은 공정별 사례 수를 제한한 controlled generation 평가입니다. 가능한 변화 유형·단계 조합을 우선 보존하며, 실제 설비 고장 성능이나 전체 Run 재생 성능으로 부르지 않습니다. unknown 유형 마스킹과 BOOLEAN gate 오류를 유지하고 클래스 지원 여부도 표시합니다. 과거에 이미 본 Test를 새 독립 평가라고 부르지 않습니다.

시간은 **단일 Window 전체 요청**의 실제 wall time(ms)입니다. raw 입력 정규화·TRAIN 참조 조회·GEVIS·원본 Head·조건부 답변·최종 매핑과 GPU 동기화를 포함합니다. 준비/모델 로딩/워밍업은 제외합니다. 각 요청은 Student 인코딩을 다시 실행하며 요청 사이 특징 캐시는 사용하지 않습니다. Teacher/Student 순서를 교차하고 같은 입력을 사용합니다. 전체 센서 시간을 센서 수로 나누지 않습니다. 합성 검사에서 측정한 시간은 제조 모델 지연 결과가 아닙니다.

## 선택 실행: Optuna

01에서 `RUN_OPTUNA=True`로 선택합니다. 기본 False는 사용자가 검증한 고정 legacy 설정을 유지합니다. 실제 Optuna 탐색은 새 Validation 실험입니다. 아키텍처·원본 Head·답변 그룹·표본 seed는 고정하고 학습률 `[1e-4,3e-3]`, T `{1,2,4}`, alpha `[0.1,0.9]`, feature_weight `[0.1,3]`만 탐색합니다.

두 목적은 **가장 약한 Validation 과제 F1 최대화**와 **실제 전체 요청 p95 최소화**입니다. 첫 trial은 검증한 control48 설정이며 이미 동일 설정으로 학습된 Student가 있으면 재학습을 생략합니다. 세 과제 목표와 원본 클래스 지원을 충족한 trial 중 p95가 작은 것을 선택합니다. 충족 trial이 없으면 가장 약한 과제 F1이 높은 것을 택하되 목표 미달로 명시합니다. Test를 열기 전까지만 탐색할 수 있습니다. [Optuna 공식 다목적 API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html)를 사용합니다.

선택 설치는 `python -m pip install -e ".[fdc,notebook,distilation]"`입니다. trial당 기본 최대 학습1200초 외에 특징 준비·Validation·실제 지연 측정 시간이 추가되며 전체 탐색은 오래 걸릴 수 있습니다. 학습 사례와 Teacher 특징은 새 `.cache/distilation/` 폴더에만 저장하고 중간 결과를 자동 삭제하지 않습니다. 코드/정규화/참조가 다른 캐시를 임의 재사용하지 않습니다.

## 병합 API

```python
from heads.distilation import load_merged, Session

runtime = load_merged("artifacts/gevis_distilation/<ID>/control48.pt", device="cuda")
reply = Session(runtime).process_window(
    current=raw_window, sensor_columns=sensor_columns,
    step=step, recipe_key=recipe_key, loop_key=loop_key,
    offset_samples=offset_samples, lineage_id=lineage_id,
    standardized=False,
)
```

이 경로는 원본 Teacher/E5나 CSV를 읽지 않습니다. 저장 모델의 native 문맥·정규화·정상 참조를 사용하고 미지원 입력은 `NOT_EVALUATED`로 분리합니다. 새 입력에 이상을 자동 주입하지 않습니다.

## 01에 통합한 배치 런타임

현재 권장 추론 어댑터는 `from heads.control48_fast import FastSession`입니다. 기존 `heads/distilation` 모듈8개는 변경하지 않아 이미 저장한 병합 패키지의 엄격한 코드/내용 검사를 유지합니다. 새 어댑터의 소스 hash와 실행 배치는 별도 런타임 보고서에 기록합니다. 원본 Head·Student·dtype·입력·최종 매핑은 바꾸지 않습니다.

기존 경로는 센서·유형마다 Head와 `.cpu()`를 반복하고 경계 통계와 전체 key grid를 센서 배치마다 다시 계산했습니다. 새 경로는 한 Window 안에서만 grid·평균·정확한 경계 통계를 재사용하고, 원래 Head를 과제별 전체 행에 한 번씩 적용한 뒤 과제별 CPU 전송을 합니다. 순수 숫자 ndarray는 원래 float32 변환·정규화와 밸브 이진값 검사를 보존하는 벡터 경로를 사용합니다. 문자열/DataFrame/특수 밸브 값은 기존 인코딩 경로로 처리합니다. 요청 사이 숫자 특징 캐시는 없습니다.

01의 병합 저장 절에서 방금 학습·저장한 EXPERIMENT/MERGED와 Validation 사례를 사용합니다. 먼저 원본5유형·BOOLEAN·ORDINAL 결정과 확률 오차를 확인하고, 같은 전체 센서 raw Window에서 순차 경로와 배치8/16/32를 측정합니다. Validation에서 평균 목표를 만족한 후보 중 p95가 작은 배치를 고르며, 만족 후보가 없으면 목표 미달로 표시합니다. 이후 Test 절에서 F1을 평가하고 고정 배치 하나로만 Test 시간을 측정합니다. seed·epoch·모델 파라미터를 Test로 고르지 않습니다.

전체 scored 사례의 두 런타임 F1 확인은 기존의 정규화 입력/판별 센서를 사용하고, 시간 측정은 전체 raw Window를 사용합니다. 이는 기존 프로토콜과 같은 구분입니다. 원래 캐시 특징 기반 보고서와 런타임끼리 재확인한 표는 구분합니다. 판정 차이가 생기면 시간/동일성 주장을 중단합니다.

02는 같은 가속 어댑터를 사용합니다. 같은 커널에 01의 EXECUTION_SELECTION이 있으면 그 배치를 재사용하고, 없으면 32를 시작 설정으로 둡니다. 해당 설정의 제조 지연/판정 일치는 01에서 검증합니다. 순차 기준 시간과 배치 최종 시간은 각각 *_serial_latency_ms.csv와 *_latency_ms.csv에 저장합니다. 선택 배치·어댑터 소스 hash는 새 병합본 옆 execution.json에 남깁니다. 03 별도 실행본은 제거했습니다.

```python
from heads.control48_fast import FastSession

SESSION = FastSession(runtime, center_batch_size=32)  # Validation에서 확인한 배치 사용
reply = SESSION.process_window(**request)
```

동일 조건의 합성 CUDA 비교에서 배치 효과를 검증했지만 그 시간을 실제 제조 모델 결과로 사용하지 않습니다. 전체 센서 수·Window 길이·양성 답변 수·GPU 동기화·반환 범위가 다르면 이전 지연 수치와 직접 비교할 수 없습니다. seed 변경은 같은 구조의 연산/전송 병목을 해결하는 방법이 아닙니다.

## Teacher 장치 오류 후 이어서 실행

Study는 공식 로더가 CPU에 복원한 Teacher 전체(E5 embedding/projection·type embedding 포함), Head와 센서 인코더를 먼저 `DEVICE`로 정렬합니다. 이후 질문 token을 만들어 고정 벡터를 추출합니다. 01의 준비 셀은 갱신된 Study 클래스를 다시 불러오고 기존 `PREPARED`가 있으면 계약 검사 후 재사용합니다. 따라서 초기화에서 CPU/CUDA 오류가 난 경우 커널을 재시작하거나 전체 전처리를 반복하지 않고 수정한 준비 셀만 재실행할 수 있습니다. 원본 가중치·소스 검증·정규화는 변경하지 않습니다.

## 검증 범위와 이전 EXP

에이전트는 합성 배열·합성 모델로만 구조·학습 gradient·Head 보존·분리·병합 저장/재로딩·런타임·매핑을 검증합니다. 실제 제조 Student 학습·Validation/Test F1·실제 지연 개선은 사용자가 위 노트북으로 확인해야 합니다.

사용자는 기존 `notebooks/EXP` 전체 삭제를 지시했습니다. 처음 재귀 삭제가 자동 승인 검토에서 차단되어 로컬 보존 위치로 옮겼고, 이후 휴지통 이동을 명시하여 이전 EXP 전체를 Windows 휴지통으로 보냈습니다. 기존 실행 폴더와 `20261110-retired-exp.local/` 보존 경로는 현재 없습니다. 다른 보존 사본·Eidos 원문·데이터·가중치·캐시는 유지하며 휴지통을 비우지는 않았습니다.
