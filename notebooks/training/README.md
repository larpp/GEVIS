# GEVIS 처리·학습 노트북

STEP 학습 흐름과 GEVIS 로딩을 직접 확인하고 수정하는 폴더입니다. 커널은 `Python (PECVD)`를 사용합니다.

## 공통 경로 설정과 실행 순서

모든 실행용 노트북의 첫 코드 셀에 `PROJECT_PATH`가 있습니다. 기본값 `None`은 현재 프로젝트를 자동으로 찾습니다.
다른 위치를 지정하려면 첫 노트북에서만 `PROJECT_PATH = r"C:\경로\GEVIS"`로 바꿔 실행하세요.
`notebooks/training/workspace.local.json`에 경로를 저장하므로 다음 노트북은 `None` 그대로 같은 설정을 사용합니다.
지정한 프로젝트에는 코드(`src`), 설정(`config`), 데이터와 가중치가 있어야 합니다.
이미 다른 프로젝트 모듈을 불러온 커널이면 재시작해야 합니다. 로컬 경로 설정은 Git 공유에서 제외됩니다.

1. STEP 1: 저장된 인코더가 있으면 재사용합니다. 없으면 원래 학습 흐름을 실행합니다.
2. STEP 2: 저장된 Fusion과 STEP 1 연결을 검증하고 재사용합니다. 없으면 원래 학습 흐름을 실행합니다.
3. STEP 3: `DOMAIN='PECVD'` 또는 `'PEALD'`를 선택합니다. 기본 `ACTION='load'`는 현재 evaluate 헤드를 재사용합니다.
   `ACTION='train'`일 때만 evaluate와 같은 전처리·학습·평가 API를 실행합니다. 인코더·Fusion은 고정합니다.
4. STEP 3.1: 같은 도메인을 선택하면 저장 헤드를 로딩하고 인공 파형으로 추론 연결을 확인합니다. 학습하지 않습니다.

`GEVIS_load`는 1·2의 저장 가중치를 직접 불러오는 단축 경로입니다. 다른 STEP을 먼저 실행할 필요는 없습니다.
여러 저장본이 같은 CHOICE 방식으로 존재하면 임의 선택하지 않습니다. 출력 안내에 따라 `CHECKPOINTS`에 명시하세요.
기존 evaluate 노트북은 변경하지 않았습니다. 과거 헤드의 코드 차이를 허용하는 호환 로딩은 기존의 제한된 검증 규칙을 그대로 사용하며, 경고가 나올 수 있습니다.

| 노트북 | 역할 |
|---|---|
| [STEP_1_train_patchtst_encoder.ipynb](STEP_1_train_patchtst_encoder.ipynb) | 공정별 시계열 인코더 학습 |
| [STEP_2_train_gevis_backbone.ipynb](STEP_2_train_gevis_backbone.ipynb) | 질문·시계열 Fusion 학습 |
| [STEP_3_train_gevis_decision.ipynb](STEP_3_train_gevis_decision.ipynb) | 현재 evaluate API를 이용한 헤드 재사용·학습 |
| [STEP_3.1_run_gevis_inference.ipynb](STEP_3.1_run_gevis_inference.ipynb) | 현재 저장된 헤드의 추론 연결 확인 |
| [GEVIS_load.ipynb](GEVIS_load.ipynb) | 고정 인코더·Fusion 로딩 및 헤드 연결 예시 |

현재 ALD/CVD 실험은 기존대로 `notebooks/ALD_evaluate.ipynb`, `notebooks/CVD_evaluate.ipynb`를 사용합니다. 과거 STEP의 ALD 처리 방식과 현재 평가 방식은 같다고 가정하면 안 됩니다.

## 복구 범위

- STEP 1·3·3.1은 Git 객체에 남아 있던 노트북 원본을 찾았습니다.
- STEP 2는 인접 프로젝트에 남아 있던 원본 사본을 사용했습니다. 코드 셀은 보존된 학습 라이브러리 소스와 일치합니다.
- GEVIS_load는 같은 작업 폴더의 세션 기록으로 모든 셀 본문을 복구했습니다. 노트북 메타데이터는 재구성했으므로 삭제 전 파일과 바이트 단위로 동일하지는 않습니다.
- `originals/`는 변경하지 않은 STEP 원본과 GEVIS 복구 자료입니다. `git_candidates/`와 `restoration_manifest.local.json`에는 복구 근거가 있습니다. 학습 자료이므로 삭제 대상이 아닙니다.
- 이 폴더의 실행 사본에는 공통 프로젝트 경로 설정을 추가했습니다. STEP 3·3.1·GEVIS_load는 현재 evaluate 저장본을 사용하도록 연결했습니다. 과거 구현은 `originals/`에 보존했습니다.
- 과거 런타임이 `notebooks/STEP_...`를 참조하므로 그 위치에도 원본을 복원했습니다. 직접 열어 수정·실행할 때는 이 `notebooks/training/` 폴더의 사본을 사용합니다.

복구 과정에서 학습이나 GitLab 업로드는 실행하지 않았습니다. 기존 ALD/CVD 노트북과 현재 가중치는 수정하지 않았습니다. 저장된 과거 출력은 과거 실행 기록입니다. 전체 재학습과 모든 과거 선택 분기의 동작까지 검증한 것은 아니며, 삭제된 과거 전용 가중치가 필요한 분기는 해당 자산을 따로 준비해야 합니다.

## 이번 실행 확인

현재 환경에서 STEP 1·2의 모든 셀을 저장본 재사용 모드로 실행했습니다. STEP 3은 PECVD/PEALD 저장본 로딩을,
STEP 3.1은 CVD softmax와 ALD softmax·grouped·sigmoid의 실제 추론 연결을 확인했습니다.
학습과 `torch.save`를 금지한 검사에서 통과했으며, 몇 시간짜리 재학습은 수행하지 않았습니다.
STEP 1 학습 이력 그래프는 이 환경의 matplotlib 렌더러 DLL 오류를 피하도록 SVG로 표시합니다.
재현 명령은 `python scripts/check_processing.py step1`, `step2`, `PECVD`, `PEALD`입니다.
STEP 3 로딩을 별도로 확인하려면 도메인 명령 뒤에 `--notebook step3`를 붙입니다.
