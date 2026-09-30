# 모델 구조 실험: 여기부터 시작하세요

이 프로젝트는 **YOLO 구조를 바꿔 KITTI 탐지 정확도와 실행 비용을 비교**하는 연구 도구입니다.
COCO 사전학습 없이 Car / Pedestrian / Cyclist를 처음부터 학습합니다.
일반 구조 실험에서는 Python 코드를 수정할 필요가 없습니다. 기존 학습을 이어 실행하기 전 `pip install -r requirements.txt`로 ONNX 의존성도 설치하세요.

**네 가지 기본 모델부터 실행하려면 [Baseline 실행 가이드](docs/BASELINES.md)를 보세요.**
기본 모델은 small입니다. 구조 변경은 아래 1 → 2 → 3 → 4를 따라 하세요. 다른 PC는 [새 PC 준비](docs/HANDOFF.md)를 먼저 읽으세요.

**정규 실행은 이제 양자화 전후 비교까지 자동 수행합니다.** 자세한 명령과 지표 정의는 [양자화 가이드](docs/QUANTIZATION.md)를 보세요. 이미 학습한 모델은 재학습 없이 양자화할 수 있습니다.

## 1. 내가 수정할 파일은 하나

### 폴더에 있는 모델을 모두 순차 실행

`run_models.py` 상단의 `REASON`, `HYPOTHESIS`, `AUTHOR`, `RUN_SUFFIX`를 수정한 뒤 실행하세요.
`{model}`은 YAML 파일명(확장자 제외)으로 바뀝니다. `configs/models` 바로 아래의
모든 `.yaml`/`.yml`을 파일명 순서로 실행하며, 완료한 모델은 실행 전에 폴더 밖으로 옮기거나 삭제하세요.
필요 없는 모델은 먼저 제외하세요.

```powershell
python run_models.py --dry-run  # 명령만 확인 (학습 안 함)
python run_models.py            # 학습 → 양자화 → 평가 → 엑셀 저장을 모델별로 순차 실행
```

공통 학습 조건은 `configs/baseline_train.yaml`입니다. 실험 이름은
`<YAML 파일명>_<RUN_SUFFIX>`(예: `yolov8n_baseline_quant01`)입니다.
기존 개별 명령의 `nano_quant01` 같은 이름과는 다르므로, 이미 완료한 모델은 직접 제외하세요.
기존 결과와 이름이 충돌하면 시작 전에 중지하고, 실행 중 오류가 나면 다음 모델은 실행하지 않습니다.
별도 실행 로그 파일이나 상태 파일은 만들지 않습니다. 기존 `experiment.py` 명령도 그대로 사용할 수 있습니다.

학습 중에는 dev 평가 결과와 `best.pt` 저장 알림이 표시됩니다.
최종 KITTI Moderate AP40 및 양자화 비교는 학습 후 별도로 수행됩니다.

| 파일 | 역할과 사용법 |
|---|---|
| `configs/models/yolov8s_baseline.yaml` | 현재 기본 모델(YOLOv8s). 비교 기준으로 유지 |
| `configs/models/내_모델.yaml` | 기본 모델을 복사해서 **이 파일만 수정** |
| `configs/train.yaml` | 팀 공통 학습 조건. 구조 비교 중 고정 |
| `configs/evaluation.yaml` | 팀 공통 측정 조건. 구조 비교 중 고정 |

현재 학습은 GPU 0, batch=8, workers=4, 640px, 최대 100 epoch, seed=42입니다.
patience=30의 조기 종료가 적용됩니다. 최종 측정은 CPU, batch=1입니다.
프로젝트 폴더에서 PowerShell을 열고 GPU 사용 가능 여부를 확인하세요.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.is_available())"
```

## 2. 복사하고 한 가지 변경

```powershell
Copy-Item configs/models/yolov8s_baseline.yaml configs/models/yolov8s_width025.yaml
```

복사본의 `scales.s`를 `[0.33, 0.50, 1024]`에서 **`[0.33, 0.25, 1024]`**로 바꾸세요.
세 값은 깊이 배율, 채널 폭 배율, 채널 상한입니다. 이것이 채널 폭을 줄이는 첫 실험입니다.

이후 `backbone`과 `head`도 수정할 수 있지만 다음 규칙은 유지하세요.

- `nc: 3`과 클래스 순서 Car / Pedestrian / Cyclist 유지.
- `yolov8n_...yaml`은 `scales.n`, `yolov8s_...yaml`은 `scales.s` 선택. 이름만 s로 바꾸면 안 됩니다.
- scale을 추정할 수 없는 파일명은 `scales` 항목이 하나일 때만 허용합니다.
- 레이어를 추가·삭제하면 `from` 인덱스와 마지막 `Detect` 입력 연결도 확인합니다.
- 한 번에 한 가지 변경부터 시작하세요. 원인을 해석하기 쉽습니다.

## 3. 동작 확인 → 기준 모델 → 변경 모델

먼저 수정한 모델로 짧게 실행하세요.

```powershell
.\.venv\Scripts\python.exe experiment.py --name width025_check01 --config configs/models/yolov8s_width025.yaml --reason "채널 폭 축소 동작 확인" --hypothesis "구조 연결이 정상이다" --smoke-test
```

이것은 학습 8장·dev 4장, 128px, batch=2, 1 epoch와 평가 10장만 사용합니다.
**성능 비교용이 아니며 엑셀에 기록되지 않습니다. 정규 batch=8의 메모리 적합성도 보장하지 않습니다.**

통과하면 기준 모델과 변경 모델을 같은 조건으로 정규 학습합니다.
이름이 이미 있으면 새 번호를 사용하세요. 기존 Large 실험은 보존돼 있습니다. 완료된 Baseline 이름은 재사용하지 마세요.

```powershell
# 같은 조건의 기준 모델이 이미 완료됐다면 이 단계는 생략
.\.venv\Scripts\python.exe experiment.py --name baseline01 --config configs/models/yolov8s_baseline.yaml --reason "기본 구조" --hypothesis "비교 기준 확보" --author "본인 이름"

# baseline01이 완료된 뒤 실행 (--baseline은 구조 비교 의도 기록용; 양자화 행은 자기 FP32 행에 연결)
.\.venv\Scripts\python.exe experiment.py --name width025_01 --config configs/models/yolov8s_width025.yaml --reason "width 0.50에서 0.25로 축소" --hypothesis "속도 개선 대비 AP40 손실 확인" --baseline baseline01 --author "본인 이름"
```

`--config`는 구조 파일, `--name`은 결과 폴더 이름입니다.
`--baseline`은 비교 대상 ID를 기록할 뿐 가중치를 불러오지 않습니다. 모두 랜덤 초기화부터 학습합니다.
실행 중 원본 YAML을 바꿔도 현재 실행에는 반영되지 않습니다. 다음 실험부터 적용됩니다.

## 4. 결과 확인

```text
구조 YAML + 공통 학습 설정
    ↓
데이터 분리 확인 → GPU 학습 → 내부 dev에서 best.pt 선택
    ↓
train 전용 보정 → ONNX FP32/INT8 변환
    ↓
각각 고정 eval 1,000장 평가 + 동일 CPU 속도/메모리 측정
    ↓
전후 비교 JSON → Excel 두 행 + 양자화비교 시트
```

| 확인할 내용 | 위치 |
|---|---|
| 이유·설정·상태 | `experiments/<이름>/experiment.json` |
| 실행 당시 구조 | `experiments/<이름>/architecture/<원본 파일명>` |
| 학습 중 손실·내부 검증 지표 | `experiments/<이름>/training/results.csv` |
| 학습된 모델 | `experiments/<이름>/training/weights/best.pt` |
| 최종 AP40·속도·메모리 | `evaluations/<이름>_q_fp32/metrics.json`, `<이름>_q_int8/metrics.json` |
| 여러 실험 비교 | `reports/트랙2_실험기록_torch211_cu128.xlsx` |

**학습 로그의 mAP와 최종 KITTI Moderate AP40은 다릅니다.** 구조 비교에는 최종 평가 결과를 사용하세요.
best.pt는 내부 dev의 Ultralytics fitness 기준입니다. 데이터는 train 5,833장 / dev 648장 / eval 1,000장입니다.

클래스별 AP40, latency p50/p95, 파라미터 수, FLOPs, 모델 크기를 함께 비교하세요.
엑셀 메모리는 **전체 프로세스 CPU RSS**이며 GPU VRAM이 아닙니다.
엑셀은 비교용 표이고 JSON과 학습 폴더가 원본 기록입니다. 결과 해석과 다음 실험은 직접 작성합니다.
AP40 구현은 주최 평가기와 최종 대조 전입니다.

## 막혔을 때

| 증상 | 조치 |
|---|---|
| 이미 있는 실험 이름 | 새 이름 사용. 실패·Ctrl+C 중단은 같은 명령 끝에 `--restart` 추가 가능 |
| `--restart` 의미 | 이전 기록을 보관하고 **처음부터** 학습. 이어 학습 아님 |
| 강제 종료 후 training 상태 | 새 이름 사용. 실행 중인 폴더는 삭제하지 않기 |
| CUDA 메모리 부족 | 작은 배치를 사용하되 기준 모델도 같은 학습 조건으로 비교 |
| scale 오류 | 파일명과 YAML의 `scales`·`scale` 일치 여부 확인 |
| 엑셀 기록 실패 | JSON은 보존됨. Excel을 닫고 아래 명령으로 기록만 재시도 |
| 측정조건 불일치 | 다른 PC/버전 결과를 기존 표에 섞지 않기. 인수인계 문서 참조 |

```powershell
.\.venv\Scripts\python.exe compare_quantization.py --fp32 evaluations/width025_01_q_fp32/metrics.json --int8 evaluations/width025_01_q_int8/metrics.json --output quantizations/width025_01_q/comparison.json
```

## 더 알고 싶을 때만 읽기

- [시스템 동작 원리](docs/WORKFLOW.md): 각 단계가 필요한 이유.
- [다른 PC 준비·폴더 전달](docs/HANDOFF.md): 환경, 데이터 경로, 공통 측정 PC.
- [구현 상세](docs/REFERENCE.md): 라벨 정책과 측정 정의.

데이터 원본, `eval_val.txt`, `datasets/`, 학습·평가 Python 코드, `tools/update_workbook.mjs`, 테스트는 시스템 구성품입니다.
구조 실험에서는 수정할 필요가 없습니다. `archive/restarted_experiments/`는 중단된 연구 기록이므로 임의로 지우지 마세요.
