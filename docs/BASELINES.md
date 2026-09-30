# Nano / Small / Medium / Large Baseline

기본 진입 모델은 **small**입니다. 네 구조 모두 KITTI 3클래스이며 COCO 가중치 없이 랜덤 초기화로 학습합니다.

| 모델 | 구조 파일 | depth / width / max_channels |
|---|---|---|
| nano | configs/models/yolov8n_baseline.yaml | 0.33 / 0.25 / 1024 |
| small | configs/models/yolov8s_baseline.yaml | 0.33 / 0.50 / 1024 |
| medium | configs/models/yolov8m_baseline.yaml | 0.67 / 0.75 / 768 |
| large | configs/models/yolov8l_baseline.yaml | 1.00 / 1.00 / 512 |

공통 학습 설정은 `configs/baseline_train.yaml`입니다.
**batch=8, workers=4, GPU 0, FP32, 640px, 최대 100 epochs, patience=30, seed=42**를 사용합니다.
Large에서 사용 가능한 배치 8에 맞춰 네 크기의 조건을 통일했습니다. 작은 모델의 처리량을 최대화하는 설정은 아닙니다.
기존 실험 설정에 영향을 주지 않도록 이 비교가 끝날 때까지 공통 Baseline 설정을 유지하세요.
최종 평가 조건은 기존 CPU·batch=1 그대로입니다.

## 실행 명령

프로젝트 루트의 PowerShell에서 **하나씩 완료된 후 다음 명령**을 실행하세요. 동시 실행은 GPU 메모리를 나눠 씁니다.
각 명령은 학습 → best.pt 선택 → FP32/INT8 양자화 비교 → 엑셀 기록까지 수행합니다.

```powershell
# Nano
.\.venv\Scripts\python.exe experiment.py --name baseline_n01 --config configs/models/yolov8n_baseline.yaml --settings configs/baseline_train.yaml --reason "YOLOv8n 기본 구조" --hypothesis "Nano 기준 성능 측정" --author "양승환"

# Small
.\.venv\Scripts\python.exe experiment.py --name baseline_s01 --config configs/models/yolov8s_baseline.yaml --settings configs/baseline_train.yaml --reason "YOLOv8s 기본 구조" --hypothesis "Small 기준 성능 측정" --author "양승환"

# Medium
.\.venv\Scripts\python.exe experiment.py --name baseline_m01 --config configs/models/yolov8m_baseline.yaml --settings configs/baseline_train.yaml --reason "YOLOv8m 기본 구조" --hypothesis "Medium 기준 성능 측정" --author "양승환"

# Large
.\.venv\Scripts\python.exe experiment.py --name baseline_l01 --config configs/models/yolov8l_baseline.yaml --settings configs/baseline_train.yaml --reason "YOLOv8l 기본 구조" --hypothesis "Large 기준 성능 측정" --author "양승환"
```

각 모델이 독립적인 기준 결과이므로 위 명령에는 --baseline을 넣지 않았습니다.
나중에 small 구조를 수정하면 `--baseline baseline_s01`로 해당 기준 결과와 연결하세요.
이 옵션은 비교 ID만 기록하며 가중치를 복사하지 않습니다.

결과는 `experiments/baseline_<크기>01/`, `evaluations/baseline_<크기>01/`와 `reports/`의 공통 엑셀에 저장됩니다.
기존 이름이 있으면 02 등 새 이름을 사용하세요. 실패/중단 실험만 --restart로 이전 기록을 보관하고 처음부터 다시 시작할 수 있습니다.

현재 파일 준비와 모델 생성 검증만 수행했습니다. 이 문서의 네 정규 학습은 자동 실행하지 않았습니다.
다른 작업과 함께 실행하면 batch=8도 메모리 부족이 발생할 수 있습니다. 로그의 자동 배치 축소 여부도 확인하세요.
기존 Large 결과가 다른 설정으로 생성됐다면 네 크기의 동일 조건 비교 결과와 구분하세요.

양자화 결과는 `evaluations/<실험명>_q_fp32/`, `<실험명>_q_int8/`와 `quantizations/<실험명>_q/`에 저장합니다. [양자화 가이드](QUANTIZATION.md)를 참조하세요.
