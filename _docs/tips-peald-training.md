# TIPS PEALD: STEP 1·2 고정 및 답변 헤드 학습

기본 실행은 `notebooks/TIPS_PEALD_train_evaluate.ipynb`이다. `notebooks/STEP_1_train_patchtst_encoder.ipynb`, `STEP_2_train_gevis_backbone.ipynb`의 **학습된 체크포인트를 사용하며 두 노트북을 다시 실행하지 않는다.** STEP 1 인코더 6개, STEP 2 전체 가중치, E5 본체·사영·정규화·Fusion·Cross-attention을 모두 고정한다. TIPS용 `ContextReadout`만 학습한다.

```text
TIPS 고정 Train / Validation / Test 분할
 → TRAIN 통계 정규화
 → 고정 STEP 1 인코더 + 고정 STEP 2 Cross-attention
 → 최종 특징 파일을 한 번 저장
 → 작은 답변 헤드만 여러 epoch 학습
```

## 실행

```powershell
cd C:\project\GEVIS
.\.venv\Scripts\python.exe -m pip install -r config\requirements-tips.txt
.\.venv\Scripts\python.exe -B scripts\train_tips_peald.py --action prepare --device cpu
.\.venv\Scripts\python.exe -B scripts\train_tips_peald.py --device cuda
# 동일 조건의 학습된 헤드 평가만 실행
.\.venv\Scripts\python.exe -B scripts\train_tips_peald.py --action evaluate --device cuda
```

노트북은 **현재 셀 중단 → 커널 재시작 → 처음부터 실행**한다. 이전 scratch 모델을 이어 학습하지 않는다. 이전 결과는 유지하며 새 결과는 `artifacts/tips_peald_frozen/<identity>/`에 저장한다. 기존 scratch 모듈은 기본 실행에서 호출하지 않는다.

## 고정 가중치와 입력 연결

`config/experiments/tips_peald.json`에 경로와 SHA-256을 고정했다.

- STEP 1: `artifacts/gevis_v4_rich_ssl_final_v1/encoder_manifest.json`
- 인코더: **Gap-up, PRE-HEAT, STAB1, DEP., PURGE, PUMPING** 6개 모두 로드·고정
- STEP 2: `artifacts/gevis_backbone_e5_v1_fdc_semantic_step1_bridge_all_types_v1_real_step1_all_types_v1/gevis_backbone.pt`

현재 PC의 세트는 **legacy STEP 1 + v9 STEP 2**이다. 최신 노트북의 native multiscale/v10 가중치로 표시하지 않는다. STEP 2에 기록된 manifest와 6개 인코더 해시가 실제 파일과 모두 일치해야 한다. `strict=True`로 전체 state를 복원하며, 일부 가중치나 무작위 가중치로 대체하지 않는다. E5의 학습된 사영도 체크포인트에서 복원한다. 토크나이저와 모델 구성은 로컬 캐시만 사용한다.

소스는 항상 `requires_grad=False`, `eval()`이다. 옵티마이저에 답변 헤드만 전달한다. 전체 소스 텐서 해시를 특징 추출 전후와 학습 후 비교하고 gradient·값 변경이 있으면 저장을 거부한다. 소스 파일·프로파일·데이터 해시도 재검증한다.

6개를 모두 고정해 로드하지만 **TIPS의 매 Step마다 6개 후보를 반복 학습·비교하지 않는다.** 기본 `source_encoder_step: "DEP."` 하나를 모든 선택 Main Step에 공통 사용한다. 다른 5개는 CPU에 고정한다. 소스 인코더는 이 설정으로 명시적으로 바꿀 수 있으며, 변경 시 새 실험 ID가 생성된다.

legacy 인코더는 PECVD 센서 ID 22개에 묶여 있다. TIPS 센서를 기존 ID에 임의 배정하지 않고 기존 `FrozenChannelAdapter`를 명시적으로 사용한다(`input_adapter: "legacy_channel_independent"`). 기존 sensor embedding 평균을 공통 문맥으로 사용하고 각 TIPS 센서의 시간축에 고정 Transformer를 독립 적용한다. **가중치는 동일하지만 원래 PECVD joint-sensor forward와 같은 입력 전이는 아니다.** 이 한계를 결과에 기록한다. 원래 joint-sensor 구조의 전이가 필요하면 그 구조로 학습된 transferable 체크포인트가 필요하다.

patch 폭은 소스에서 가져온다. 기본 DEP.는 16행이므로 TIPS 64행 창은 센서당 4개 patch이다. 시간 보간·재표본화는 하지 않는다. 다른 소스를 선택해 창 길이가 patch 폭으로 나누어떨어지지 않으면 오류로 알린다. `encoder_epochs`, `encoder_layers`, `fusion_width`, `text_encoder`, `patch_length`로 소스 구조를 우회하면 중단한다.

## 데이터와 과제

- 데이터: `../TIPS/etc/data/hd02_splits/d1c591fbc818b3b2`가 참조하는 Parquet.
- 기존 시간순 Train 58 / Validation 18 / Test 13 Run을 유지한다. HD02-1 `reserved_transfer` 10 Run은 제외한다.
- Step은 `../TIPS/etc/experiment_common/profiles/step_profile.json`, 센서는 `sensor_profile.json`만 따른다. exclude가 우선이며 센서 사전 순서를 유지한다. 선택하지 않은 센서는 문맥 입력에도 넣지 않는다.
- Main Step별 TRAIN 평균·표준편차만 사용한다. Step의 TRAIN에서 상수인 센서는 합성 변화 대상에서 제외하며, 모든 대상이 상수이면 중단한다.
- Run·Main occurrence·continuous segment·Main Step·관측 간격 0.95~1.05초 경계를 지킨다. 결측을 보간하거나 구간을 이어 붙이지 않는다. `window_scope=cycle`이면 cycle 경계도 지킨다.
- 기본 창 길이·stride는 64행이다. Train은 Run/Step당 최대 4창, Validation/Test는 2창을 전체 구간에서 reservoir 방식으로 선택한다. 짧아 창이 생기지 않는 Step은 audit에 기록한다.
- 수치 센서는 분석 중심·합성 대상으로 사용한다. ValveAct를 선택하면 OPEN=1/CLOSE=0 문맥 입력으로만 사용한다. 선택한 설정값도 수치 변화 과제에 포함될 수 있다.

사례 구성은 유지했다. 지원 Step마다 모든 선택 수치 센서를 분석 중심에 포함하도록 창–센서 조합을 만들고, 조합당 기본 24변형과 두 분석 중심을 만든다. **동결만으로 사례 수를 줄이지 않았다.** 최초 특징 추출에는 시간이 필요하지만 모든 epoch가 그 결과를 재사용한다. Test는 정규화·학습·모델 선택에 사용하지 않으며 평가 시점에만 읽는다.

출력은 변화 유무(2종), 형태(6종), 합성 강도(5종)이다. 미래 센서 회귀 예측은 아니다. 정답은 주입 전 원본 창을 참조하는 **paired synthetic 변화 과제**이며 실제 장비 고장 정확도나 물리적 심각도 라벨이 아니다.

## 저장, 학습 및 진행률

- `feature_batch_size: 32`: 초기 고정 특징 추출 배치.
- `batch_size: 256`: 저장된 특징으로 작은 헤드를 학습하는 배치.
- `answer_epochs: 20`, `patience: 5`, AdamW `learning_rate: 0.001`.
- `head_max_seconds: 300`: 헤드 학습 시간 예산. 완료된 epoch의 Validation 뒤 확인하며 **특징 추출 시간은 제외**한다.
- 목적함수는 세 cross-entropy 평균 + 0.2×일관성 + 0.2×ordinal 손실이다. Validation 평균 macro-F1으로 선택한다. 헤드 입력 정규화는 Train 특징만 사용한다.
- `trace_cache_mb: 256`: 특징 추출 중 토큰 재사용 CPU 캐시.
- `feature_cache_limit_mb: 4096`: 파티션별 최종 특징 파일 한도. 기본 특징 차원은 `[사례, 3개 질문, 770]`, float32 메모리맵이다.

특징 추출은 중심 쌍을 인접 배치하고, 헤드 학습에서 사례를 섞는다. 완성된 특징 캐시는 데이터·선택 프로파일·소스 가중치·코드·사례 identity와 파일 해시가 모두 같을 때만 재사용한다. 미완성 `.tmp`는 재사용하지 않는다. 이미 존재하는 헤드 checkpoint는 덮어쓰지 않는다.

`Frozen features train/validation/test`, `Readout`, `Validation`, `Test readout` 단계별 배치 수·진행률·경과 시간·ETA를 표시한다. 학습은 평균 loss도 표시한다. 기본 간격은 50배치 또는 10초이며 `log_every_batches`, `log_every_seconds`로 조절한다. `progress.json`은 현재 상태, `history.json`은 완료된 epoch이다.

| 파일 | 내용 |
| --- | --- |
| `frozen_source_summary.json` | 소스 6개·STEP 2 해시, 전체 텐서 해시, 어댑터, 소스 학습 파라미터 0개 |
| `features/<partition>/x.npy`, `y.npy`, `meta.npy` | 최종 특징, 정답, Run/Step/분석 센서/창 인덱스 |
| `features/<partition>/manifest.json` | 완성 캐시 identity와 파일 해시 |
| `model.pt` | **답변 헤드만의** 가중치, Train 통계, 고정 소스 identity |
| `history.json`, `training_summary.json` | 헤드 학습 이력, 사용 Run/Step, 소스 고정 여부 |
| `window_audit.csv`, `test_window_audit.csv` | 사용 창과 제외 상태 |
| `test_predictions.csv`, `test_by_step.csv`, `test_by_sensor.csv` | Test 질문별 정답·예측과 Step/센서별 성능 |
| `test_metrics.json` | 세 답변 성능, 소스 가중치와 입력 어댑터 정보 |

`cached_head_batch_ms_mean`은 저장된 특징에 대한 헤드 시간이다. 인코더·E5·Fusion·Cross-attention·원시 입력 준비를 제외하므로 **전체 추론 지연 시간이나 Edge KPI로 해석하지 않는다.**
