<!-- gevis-contract-sha256: abfc5e15f906a608bc94387c62a5771211ee86adb74ca1761de1c08e94351d11 -->
<!-- peald-encoder-routing: peald_native_validation_encoder_routes_v1 -->

# 현재 코드와 폴더 구조

GEVIS는 **인코더 + 질문 Fusion + 학습 답변 헤드**입니다. 외부 실행 진입점은 `InferenceSession.process_window`이며 전처리·TRAIN 참조 선택·결과 구성을 맡습니다. `GEVIS` 클래스 하나가 CSV 읽기부터 채점까지 전부 처리하지는 않습니다.

| 역할 | 주요 소스 |
| --- | --- |
| Step별 인코더·체크포인트 로드 | `encoder_loading.py`, `encoder_reuse.py`, 각 encoder backend |
| 질문·수치 Fusion | `gevis_backbone.py`, STEP2/3/3.1의 GEVIS 정의 |
| 최종 학습 답변 | `experiments/fusion_answers.py`, PEALD `peald_readout.py`, `peald_step_heads.py` |
| 공통 새 입력 요청 | `experiments/online_inference.py` |
| PEALD 전처리·전이 | `experiments/peald_training.py`, `onboarding.py` |
| 외부 CSV 예측 | `external_test.py`, `pecvd_external.py` — 기본 채점 없이 제한 범위 추론 |
| 보존한 별도 채점 도구 | `generated_test_scoring.py` — 기본 노트북은 호출하지 않음 |
| Fast 세 답변 프로젝트 평가 표시 | `experiments/fast_assessment.py` — 기존 결과를 읽어 세 지표와 전체 요청 시간을 분리 보고 |
| 성능·불확실성·지연 | `fusion_metrics.py`, `evaluation_audit.py`, `inference_benchmark.py` |
| 공통 로더·기존 추론 도우미 | `gevis_ui.py` — 이름은 남았지만 웹 UI 파일은 아님 |

PECVD는 Step별 인코더·공통 Fusion을 사용합니다. PEALD ALD_evaluate 기본은 PECVD DEP 인코더·Fusion 동결 후 별도 공통/Step 헤드입니다. 학습·정규화는 TRAIN, 선택은 Validation, Test는 평가입니다.

## 폴더별로 무엇이 있나

| 폴더 | 용도 |
| --- | --- |
| `notebooks/` · `notebooks/research/` | 학습 STEP 4개 · 평가/외부 CSV 4개 · CVD 연구 3개 |
| `src/pecvd/` | 모델·입력·평가·호환 코드 |
| `config/` | 모델·공정 설정과 의존성 |
| `scripts/` · `tests/` | 환경 준비·검사·전처리·배포 / 회귀 검증 |
| `_blueprint/` · `_docs/` | 설계·상세 방법 / 짧은 안내·검증 |
| `_evidence/local/` | Git 제외 내부 결과 원표·핵심 요약 |
| `data/` · `artifacts/` | 기밀 입력·학습 가중치. 정리 대상으로 지우지 않음 |
| `.cache/` | 로컬 임시·복구·공개 모델 자산. 활성 코드 아님 |
| `.agents/` · `.claude/` · `_meta/` | 작업 절차·이력·저장소 메타데이터 |

## Python 파일을 남긴 기준

현재 STEP·평가·예측·연구 노트북의 import뿐 아니라 문자열 reload, subprocess worker, 체크포인트 backend, 공용 테스트 fixture까지 확인했습니다. `legacy_patchtst`, `temporal_bridge`, `center_training`, `optimization_data`, `merlion_worker`, `peald_fast` 등은 이 경계 때문에 유지합니다. 일부는 현재 추론의 직접 경로가 아니라 호환·선택 비교·회귀 검증용입니다.

2026-10-23 확인에서 소비자가 전용 테스트뿐인 `preprocessing.py`와 `reconstruction.py`는 전용 테스트와 함께 활성 트리에서 정리했습니다. 현재 복원 학습·복원 특징은 다른 활성 구현에 있으므로 그대로 유지합니다. 웹 앱·frontend는 이미 제거했습니다.

과거 통계 비교 코드의 존재가 통계 규칙으로 현재 답을 결정한다는 뜻은 아닙니다. 현행은 `decision_system=fusion_answers`이며 q99/GMM·무작위 모델로 답을 대체하지 않습니다. 따라서 **모든 남은 Python 파일이 매 요청에서 직접 실행되는 것은 아닙니다.** 비교 코드를 더 제거하려면 로더와 검증 의존성을 별도로 분리해야 합니다.

[실행 순서](../notebooks/README.md) · [현재 설계](../_blueprint/experiment-plan.md) · [상세 실험 방법](../_blueprint/gevis-experiment-guide.md) · [검증·배포](verification.md) · [주요 실패](../FAILURE_LOG.md)

Fast는 BOOLEAN·CHOICE·ORDINAL을 한 요청에서 모두 평가합니다. 세 답변 중 하나의 점수만으로 수락하지 않으며, CHOICE·ORDINAL을 미래 연속값 예측 Precision으로 부르지 않습니다. 상세 목표와 미확정 수락 조건은 blueprint에 있습니다.

현재 화면: STEP4는 PECVD 기본 성능+속도, CVD_predict_external_csv / ALD_predict_external_csv는 한 파일·최대8창의 예측+시간입니다. 상세 표는 선택 표시하며 외부 정답 채점·원본 정렬·중복 평가를 기본 실행에서 제외합니다.
