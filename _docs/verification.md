# 검증과 Git 전달

## 무엇을 확인하는 검사인가

- 문서 계약: `python -B scripts/check_gevis_docs.py`. 실제 설정·주요 소스·STEP/EXP 실행 셀과 문서 선언을 비교합니다. 내용 검토 없이 해시만 갱신하지 않습니다.
- 모델 준비: `python scripts/check_gevis_pipeline.py`. 필요한 가중치·입력 규약을 확인하며 학습하지 않습니다.
- 코드 회귀: 합성 fixture로 모델 로드·forward·정답 연결·경계·지표·배포 동작을 확인합니다. 실제 공정 정확도나 Edge 성능을 증명하지 않습니다.

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

공개 encoder 통합 시험은 로컬 공개 가중치가 필요합니다. 새 환경은 `python scripts/setup_patchtst.py`로 준비합니다. 원본 공정 데이터나 학습 가중치를 외부로 보내지 않습니다. 시스템 Python과 Jupyter 커널의 환경이 같은지 확인하세요.

| 수정 범위 | 집중 시험 |
| --- | --- |
| Fusion·답변 학습 | `tests.test_fusion_answers tests.test_fusion_augmentation` |
| 지표·분할·STEP4 | `tests.test_fusion_metrics tests.test_evaluation_audit tests.test_step4_fusion_primary` |
| PEALD 준비·헤드 | `tests.test_peald_training tests.test_peald_readout tests.test_peald_step_heads` |
| Fast 세 답변 평가 표시 | `tests.test_fast_assessment` |
| 공통 요청·외부 CSV | `tests.test_online_inference tests.test_external_test tests.test_pecvd_external tests.test_generated_test_scoring` |
| 공통 로더·문서·전달본 | `tests.test_gevis_ui tests.test_document_contract tests.test_public_release` |

`gevis_ui.py`는 실험이 쓰는 공통 로더라 유지합니다. 웹 UI와 API 전용 테스트는 제거했습니다. 복원 학습은 활성 encoder 구현에 남아 있고, 정리한 옛 reconstruction bundle 유틸리티와 다릅니다.

## 기존 결과를 보호하면서 검증

이번 정리 전 STEP4·EXP1~9의 원본 노트북과 저장 출력은 `.cache/notebook-purpose-reset-20261023/before/`에 보관했습니다. 현재 노트북에서는 과거 출력을 지워 다음 실행 결과와 혼동하지 않게 했습니다. 기존 수치는 로컬 evidence에 그대로 남습니다. 모델 가중치와 원본 CSV는 바꾸지 않습니다.

STEP4는 PECVD의 합성 정답 기반 성능과 원본 Test 요청 속도를 각각 측정합니다. ALD_evaluate는 PEALD에서 같은 구분을 사용합니다. CVD_predict_external_csv / ALD_predict_external_csv 기본 경로는 외부 CSV 예측·속도 확인이며 원본 매칭과 채점은 실행하지 않습니다. 기본 범위는 1개 파일·8개 native window입니다. 이 제한을 풀어야 전체 입력 재생이며, 미실행 구간 수를 coverage로 표시합니다. 제한은 모델 요청 횟수에 적용되고 선택한 CSV의 읽기·전처리는 여전히 필요합니다.

`SHOW_DETAILS`는 상세 표·오답·그림, `RUN_BOOTSTRAP`은 추가 신뢰구간 계산을 선택합니다. 독립 정답 없는 예측에 정확도를 만들지 않습니다. 기존 scorer는 별도 도구의 회귀 검증 대상으로 남습니다. 이 코드 검증은 실제 이후 회차의 정확도와 대상 Edge 전체 지연을 증명하지 않습니다.

## 출력 없는 전달본

현재 작업 저장소의 Git 이력에는 기밀 자료가 있습니다. 기존 이력을 그대로 push하지 않습니다.

```powershell
python scripts/build_gitlab_release.py --destination .cache/publish/gevis-source
```

빌더는 새 폴더만 만들고 소스·설정·테스트·학습 STEP 4개·평가/외부 CSV 4개·연구 3개를 명시한 허용 목록으로 복사합니다. 복사본에서 출력·실행 번호·첨부·widget을 제거합니다. 공정 설정의 로컬 경로를 치환하고 데이터·가중치·로컬 결과·복구 캐시·기존 Git 이력은 제외합니다. 원본 노트북은 바꾸지 않습니다.

전달본의 FAILURE_LOG는 [공유용 교훈](gevis-failure-lessons.md)을 가리키며 내부 상세 ledger를 포함하지 않습니다. 전달본 안에서 다시 테스트하고 검토한 뒤에만 별도 승인된 remote로 게시합니다. 빌더는 commit·push·BookStack 업로드를 하지 않습니다.

## 절차 상태

코드 검사·독립 검토·Eidos 등록은 서로 다른 결과입니다. 기존 Eidos 이력 검증 오류와 미도입 requirement registry는 별도로 명시합니다. 현재는 AGENTS와 관련 domain/privacy 규칙을 수동 검토하며, 등록 실패를 Harness 전체 통과로 표시하지 않습니다.

이번 경로 통합 전 바이트와 해시는 `.cache/notebook-consolidation-20261023/`에 보존합니다. 정규 inventory 11개를 readiness·배포·노트북 테스트가 동일하게 확인합니다. CVD_predict_external_csv / ALD_predict_external_csv는 PECVD와 PEALD 두 분기의 실제 Run All을 합성 fixture로 검증합니다.
