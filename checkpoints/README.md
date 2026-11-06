# 학습 가중치 준비

GitLab 소스 배포본에는 실제 공정 데이터와 학습 가중치를 넣지 않습니다.
제조 가중치의 공개·재배포 권한은 별도로 확인해야 합니다.

기존 모델 로딩에는 다음 파일이 필요합니다.

1. 인코더 목록 `encoder_manifest.json`과 목록에 상대 경로로 지정된 인코더 파일
2. 위 목록과 연결된 융합 체크포인트 `gevis_backbone.pt`
3. 고정된 버전의 E5-small-v2 모델과 토크나이저 파일
4. CVD·ALD 판별을 실행한다면 해당 판별 헤드 패키지와 입력 메타데이터

기존 평가 노트북은 E5 버전 `ffb93f3bd4047442299a41ebb6fa998a38507c52`를 사용합니다.
E5 캐시는 작업 폴더와 별도로 준비하며 구체적인 준비 명령은 `notebooks/README.md`에 있습니다.

독립 인코더 로더에는 저장소 밖의 경로를 지정할 수 있습니다. 파일 경로를 바꾸는 것으로
가중치를 수정하거나 모델을 재학습하지 않습니다. 인코더 파일 체크섬, 목록과 융합 가중치의 연결,
모델 차원을 검사합니다. 센서 구성과 전처리는 별도로 일치해야 합니다.

체크섬이나 차원 오류를 숨기기 위해 목록을 편집하거나 일부 가중치만 로드하지 마세요.
기존 파일에는 Python·NumPy 메타데이터가 들어 있어 `weights_only=False`로 읽습니다.
출처를 신뢰할 수 있는 체크포인트만 사용해야 합니다.

## 평가 노트북의 작업 폴더

독립 인코더 API는 임의 경로의 인코더 목록·융합 파일을 받을 수 있지만, 기존 평가 노트북은
가중치 호환성을 위해 아래 작업 폴더 구성을 사용합니다. 코드가 있는 저장소와 다른 위치여도
첫 설정 셀의 `WORKSPACE_PATH`로 지정할 수 있습니다.

```text
작업 폴더/
├── config/
│   ├── experiments/gevis_pipeline.json
│   └── processes/pecvd.json
├── data/
│   ├── all_processes_analysis.csv       # CVD 평가 시 필요
│   └── mt_nit/data_complete/*.csv       # ALD 평가 시 필요
└── artifacts/
    ├── gevis_v4_rich_ssl_final_v1_native_multiscale_blind_v3/
    │   ├── encoder_manifest.json
    │   ├── encoders/*.pt
    │   └── windows/*.npz                # 기존 CVD 입력 구간
    ├── gevis_backbone_e5_v1_fdc_semantic_step1_bridge_all_types_v1_real_step1_native_multiscale_blind_v3_center_relation_v14_training_source_v2/
    │   └── gevis_backbone.pt
    └── gevis_heads/                     # 저장 헤드 재사용 시 필요
```

위 두 설정 파일은 FDC 소스 배포본에도 있습니다. 기존 자산의 폴더 이름과 목록을 임의로
바꾸면 호환성 검사가 실패할 수 있습니다. CVD는 원본 CSV의 체크섬과 기존 입력 구간·Run 분할을
확인하며, ALD CSV는 기존 공정 메타데이터 열을 갖춰야 합니다. 경로만 바꿔 모든 형식의 CSV를
평가할 수 있다는 의미는 아닙니다. 새로운 형식의 데이터는 독립 인코더 예제에서 전처리해야 합니다.
