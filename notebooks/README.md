# 평가 노트북

기존 STEP 학습·추론 및 GEVIS 로딩 노트북도 복원했습니다. 직접 수정·실행할 사본은
[processing/](../processing/README.md)에 있습니다. 이 폴더의 STEP 원본은 과거 런타임 경로 호환을 위해 보존합니다.

현재 실행하는 노트북은 두 개입니다. 다른 노트북을 먼저 실행할 필요가 없습니다.

프로젝트 루트 또는 `notebooks` 폴더에서 실행합니다. 첫 설정 셀의 `WORKSPACE_PATH=None`은
현재 프로젝트의 데이터·가중치를 사용한다는 뜻입니다. GitLab에서 받은 코드와 실제 데이터·가중치가
다른 폴더에 있으면 `WORKSPACE_PATH`에 기존 작업 폴더를 지정하세요. 코드는 현재 저장소의
`src`에서 가져오고 데이터·설정·가중치는 지정한 작업 폴더에서 읽습니다.

```bash
python -m pip install -e ".[fdc,notebook]"
python -m ipykernel install --user --name pecvd --display-name "Python (PECVD)"
```

GitLab 배포본에는 공정 데이터와 가중치가 없으므로 위 설치만으로 제조 모델을 실행할 수는 없습니다.
필요한 자산은 [가중치 안내](../checkpoints/README.md)에 있습니다.

노트북은 네트워크 없이 기존 가중치를 읽도록 설정되어 있습니다. E5 모델이 없는 새 환경에서는
인터넷이 가능한 별도 준비 단계에서 고정 버전의 파일을 먼저 받거나, 준비된 Hugging Face 캐시를
옮겨야 합니다. 다음 명령은 공개 E5 파일만 받으며 제조 가중치를 내려받지 않습니다.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='intfloat/e5-small-v2', revision='ffb93f3bd4047442299a41ebb6fa998a38507c52', allow_patterns=['config.json','model.safetensors','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt'])"
```

기본 Hugging Face 캐시가 아닌 곳을 사용하면 노트북 실행 전에 `HF_HOME` 또는 `HF_HUB_CACHE`를
설정하세요. E5 캐시는 `WORKSPACE_PATH`와 별도입니다.

| 파일 | 용도 |
|---|---|
| [CVD_evaluate.ipynb](CVD_evaluate.ipynb) | 기존 CVD 전처리·공정별 고정 GEVIS 가중치를 이용한 헤드 학습·평가 |
| [ALD_evaluate.ipynb](ALD_evaluate.ipynb) | data_complete 전처리·고정 DEP 인코더를 이용한 ALD 헤드 3방식 비교 |

Python 환경에 프로젝트 의존성과 ipykernel을 설치하고 실행합니다. 로컬에서는 기존
`Python (PECVD)` 커널을 사용할 수 있습니다. 학습 셀은 실제 헤드를 학습하므로, 저장 모델의
추론 시간만 측정할 때는 학습 셀을 다시 실행하지 않습니다.

두 평가 노트북은 기존 가중치의 코드 해시 호환성을 위해 `pecvd.gevis`와
`pecvd.gevis_training` 호환 계층을 사용합니다. 독립 모델 API는 `GEVIS_Model`에 있으며,
노트북 없는 사용법은 [프로젝트 README](../README.md)의 Python 예제를 참고하세요.

ALD의 `DEP.`는 특징 추출에 사용하는 CVD 인코더 이름입니다. ALD 입력을 DEPOSITION 구간으로
한정한다는 뜻은 아닙니다. 전처리 정책은 [전처리 문서](../docs/preprocessing.md)에 정리했습니다.

이전 STEP·GEVIS_load·research 노트북은 정리했습니다. 기존 인코더·융합 학습 코드는
`heads/_legacy_training`에 있으며 `examples/train_historical.py`로 실행할 수 있습니다.
저장된 노트북 결과는 로컬에서 유지하고, GitLab 배포본에서만 출력을 제거합니다.
