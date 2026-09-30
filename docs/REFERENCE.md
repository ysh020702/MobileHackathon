# 구현 상세 참고

처음 사용하는 사람은 먼저 [시작 가이드](../README.md)를 읽으세요. 아래 경로와 명령은 프로젝트 루트 기준입니다.

COCO 사전학습 가중치를 사용하지 않습니다. `yolov8n_custom.yaml`로 YOLO를 새로 만들고,
KITTI의 **Car / Pedestrian / Cyclist**를 처음부터 학습합니다.

현재 준비된 데이터: **train 5,833장 / 내부 dev 648장 / 대회 고정 eval 1,000장**.
`eval_val.txt`의 이미지는 train과 dev 어디에도 포함되지 않습니다.

## 1. 각 파일의 역할

| 파일 | 역할 | 주로 수정할 곳 |
|---|---|---|
| `project_config.py` | 프로젝트 경로와 클래스 ID 공통 정의 | 경로를 바꿀 때 |
| `prepare_kitti.py` | 원본 KITTI 라벨을 YOLO 라벨로 변환, 분할·누출 검사 | 라벨 정책을 실험할 때 |
| `yolov8n_custom.yaml` | 신경망 설계도. nc=3, backbone/head/scales | 구조 실험의 핵심 |
| `configs/train.yaml` | epochs, batch, 학습률, seed, 장치 | 학습 조건 |
| `configs/evaluation.yaml` | 모든 모델에 동일하게 적용할 추론/측정 조건 | 팀 기준을 정할 때만 |
| `train.py` | 랜덤 초기화 모델 학습, 설정·소스·가중치 보존 | 일반적으로 수정 불필요 |
| `evaluate.py` | 학습된 KITTI 모델의 고정 평가·속도·메모리 측정 | 평가 흐름 |
| `kitti_metrics.py` | Moderate 제외 규칙, TP/FP/FN, PR, AP40 계산 | 주최 평가기 대조 시 |
| `experiment.py` | 데이터 확인 → 학습 → 평가 → 엑셀의 일괄 실행 | 실험할 때 실행 |
| `record_experiment.py` | 저장된 평가 JSON을 엑셀에 기록/재시도 | 엑셀만 다시 기록할 때 |
| `tools/update_workbook.mjs` | 엑셀 열 매핑, 수식 보존, 중복 방지, 백업 | 엑셀 양식을 바꿀 때 |
| `run.py` | 학습한 PT로 이미지 한 장 탐지 | 데모 |
| `test_pipeline.py`, `test_kitti_metrics.py`, `tests/test_workbook_export.py` | 데이터·평가·엑셀 자동 기록 회귀 테스트 | 수정 후 검증 |

주요 함수와 설정에는 한국어 주석을 달았습니다. YAML은 **구조**, PT는 **학습 결과**이며,
둘을 혼동하지 마세요. 현재 실행에 필요한 파일만 루트에 두고 결과와 참고 자료는 아래처럼 관리합니다.

```text
configs/                  # 학습·평가 설정
data_object_image_2/      # KITTI 원본 이미지
data_object_label_2/      # KITTI 원본 라벨
datasets/kitti3/          # 변환 라벨·고정 분할·무결성 manifest
experiments/              # KITTI 학습 기록과 체크포인트
evaluations/              # KITTI 평가 결과
reports/                  # 현재 환경의 실험 엑셀과 측정조건 JSON
outputs/                  # 데모 이미지 등 재생성 가능한 출력
archive/restarted_experiments/ # 재시작 전 중단 실험 기록
```

`eval_val.txt`는 고정 평가 목록이므로 루트에 유지합니다. `.venv`, `Ultralytics/settings.json`,
`tools/node_modules`는 실행 환경에서 사용합니다. `__pycache__`, `.matplotlib`는 실행 중 생성되는 캐시입니다.

## 2. 라벨 변환과 데이터 분리

최초 1회 실행합니다. 이미 준비됐다면 파일 해시와 분할을 다시 검사합니다.

```powershell
.\.venv\Scripts\python.exe prepare_kitti.py
```

KITTI 원본 한 줄에는 클래스, 잘림, 가림, 2D 좌표, 3D 정보 등이 들어 있습니다.
YOLO 학습에는 다음 5개 값만 사용합니다.

```text
class_id center_x center_y width height
```

- 클래스 ID: `0=Car`, `1=Pedestrian`, `2=Cyclist`. 자전거만 있는 COCO bicycle로 바꾸지 않습니다.
- 좌표는 이미지 너비/높이로 나눈 0~1 값입니다. 경계 밖 박스는 이미지 안으로 자릅니다.
- 예: 100×100 이미지의 Car 박스 `(20,10,60,50)` → `0 0.4 0.3 0.4 0.4`.
- 학습에는 세 클래스의 **모든 난이도**를 사용합니다. Moderate 필터는 최종 평가에만 적용합니다.
- `Van`, `Truck`, `Person_sitting`, `Tram`, `Misc`, `DontCare`는 학습 대상 클래스에 넣지 않습니다.
- **한계:** 표준 YOLO 라벨에는 ignore-region 필드가 없습니다. 제외한 영역도 학습의 배경 손실에
  영향을 줄 수 있습니다. 이 구현은 DontCare 손실 마스킹을 구현한 것은 아닙니다.
  평가에서는 원본 KITTI 라벨을 읽어 DontCare/유사 클래스 제외 규칙을 적용합니다.

생성된 `datasets/kitti3/dataset.yaml`의 `val`은 내부 dev 648장을 가리킵니다.
원본 이미지와 라벨은 보존하며, 생성 이미지는 가능한 경우 하드링크로 연결하므로 내용을 직접 수정하지 마세요.
실제 학습은 `train.txt`의 이미지 경로만 읽습니다. `images/` 전체 폴더를 학습 데이터로 지정하면 안 됩니다.

`manifest.json`에는 목록과 변환 라벨의 해시, 분할 seed, 제외 라벨 개수가 저장됩니다.
학습 전에 목록이 겹치거나 변환 파일이 바뀌면 중단합니다. 분할을 바꿔야 한다면
`prepare_kitti.py --output datasets/kitti3_new`로 별도 버전을 준비하세요.

## 3. 실험 한 번 실행

`configs/train.yaml`은 현재 PC의 NVIDIA RTX 5070 Ti Laptop GPU를 사용하도록 `device: 0`으로 설정했습니다.
학습 배치는 GPU 메모리 여유를 활용하도록 `batch: 16`으로 설정했습니다(기존 8).
최종 평가·속도 측정은 CPU, batch=1을 유지합니다. 설정 변경은 다음 학습 실행부터 적용되며,
이미 실행 중인 학습에는 적용되지 않습니다. 배치 8과 16의 학습 결과는 학습 조건이 다르므로
구조 비교 시에는 배치를 통일하세요. `--smoke-test`는 빠른 확인을 위해 batch=2를 사용합니다.
기본값은 100 epochs이며, 정규 학습은 오래 걸릴 수 있습니다.

```powershell
.\.venv\Scripts\python.exe experiment.py --name kitti001 --reason "KITTI 기본 구조" --hypothesis "3클래스 기준 성능 측정" --author "담당자"
```

이 명령은 다음 순서로 처리합니다.

1. 학습/내부 검증과 고정 평가셋의 비중복 확인.
2. YAML에서 랜덤 초기화 모델 생성. `pretrained=False`, COCO 가중치 로드 없음.
3. 내부 dev로 학습 상태를 검증하고 `best.pt` 저장.
4. 고정 1,000장에서 KITTI 2D Moderate AP40, PR 곡선, 속도/메모리 측정.
5. `reports/트랙2_실험기록_torch211_cu128.xlsx`의 다음 빈 행에 결과 자동 입력.

`best.pt` 선택은 내부 dev의 **Ultralytics fitness** 기준입니다. 학습 로그의 mAP는
대회 AP40이 아닙니다. 대회 점수는 마지막 `evaluate.py` 결과를 사용하세요.

빠른 설치/동작 확인은 별도 실험 이름으로 실행합니다.

```powershell
.\.venv\Scripts\python.exe experiment.py --name smoke002 --reason "동작 확인" --hypothesis "전체 파이프라인 연결" --smoke-test
```

동작 확인은 학습 8장·dev 4장·128px·1 epoch와 부분 평가를 사용합니다.
**성능 실험이 아니며 엑셀에는 자동 등록하지 않습니다.** 검증이 끝난 임시 smoke 결과는 정리 대상입니다.

## 4. 구조 변경 실험

원본 구조를 복사한 다음 한 가지씩 바꿉니다. 출력 클래스 `nc: 3`과 이름 순서는 유지하세요.

```powershell
Copy-Item yolov8n_custom.yaml configs/yolov8n_width0125.yaml
```

복사본의 `scales.n`을 `[0.33, 0.25, 1024]`에서 `[0.33, 0.125, 1024]`로 바꾸는 예:

```powershell
.\.venv\Scripts\python.exe experiment.py --name kitti002 --config configs/yolov8n_width0125.yaml --reason "width 0.25에서 0.125로 축소" --hypothesis "연산량 감소, AP40 손실 비교" --baseline kitti001 --author "담당자"
```

구조를 바꾸면 새로 학습합니다. 두 실험은 동일한 분할, seed, epoch/optimizer, 평가 조건을 사용하세요.
기본 구조는 `scales.n`을 쓰는 nano 모델입니다. 다른 scale도 사용할 수 있으며,
실험 사본은 `architecture/<원본 파일명>`으로 저장해 파일명에 따른 scale 선택을 보존합니다.
예를 들어 `yolov8s_custom.yaml`은 `scales.s`를 선택합니다. 파일명에서 scale을 추정할 수 없으면
`scales` 항목이 하나일 때만 그 항목을 사용합니다. 여러 항목 중 첫 번째를 묵시적으로 선택하지 않습니다.
YAML에 `scale`을 적었다면 파일명/단일 항목의 선택과 일치해야 합니다. 충돌은 학습 전에 차단합니다.
선택된 scale과 사본 경로는 `experiment.json`에 기록합니다. 기존 실험의 고정 파일명 사본도 평가할 수 있습니다.
임의의 연결/모듈 변경은 텐서 크기 호환도 확인해야 합니다.
기존 실험은 기본적으로 덮어쓰지 않습니다. Ctrl+C로 중단했거나 실패한 실험은 `--restart`로
같은 이름을 재사용할 수 있습니다. 이전 폴더는 `archive/restarted_experiments/<이름>_<UTC시각>/`에
보관하고 현재 설정으로 **처음부터** 학습합니다. 체크포인트 이어 학습(resume)은 아닙니다.

```powershell
.\.venv\Scripts\python.exe experiment.py --name kitti001 --reason "KITTI 기본 구조" --hypothesis "3클래스 기준 성능 측정" --author "담당자" --restart
```

완료된 실험, 같은 이름의 평가 기록이 있는 실험, 강제 종료로 상태가 `training`에 남은 실험은
재시작 대상에서 제외합니다. 이 경우에는 새 실험 이름을 사용하세요.

## 5. 결과가 쌓이는 곳

```text
experiments/<실험명>/
  experiment.json             # 이유/가설/담당자/기준 실험/학습상태/버전
  architecture/<원본 파일명>  # 실행 당시 구조 사본; 파일명 기반 scale 보존
  train_settings.yaml         # 실행 당시 학습 설정
  dataset_manifest.json       # 데이터 버전 및 누출 검사 근거
  train_ids.txt, dev_ids.txt, eval_ids.txt
  source/                     # 실행 당시 주요 코드
  training/weights/best.pt, last.pt
  training/results.csv        # epoch별 손실과 내부 dev 지표

evaluations/<평가명>/
  metrics.json                # AP40, PR 좌표, 모델 크기, FLOPs, 측정 조건
  pr_curve.png                # 보간 전 PR / AP40 계산용 보간 곡선
  predictions.json           # 이미지별 예측 박스·신뢰도·클래스
  evaluation_settings.yaml
  architecture.yaml
```

실험 디렉터리가 원본 기록이며 엑셀은 비교용 표입니다. 엑셀 저장에 실패해도 결과를 잃지 않습니다.
학습이 끝난 모델을 다시 평가할 때는 새로운 평가 이름을 사용합니다.

```powershell
.\.venv\Scripts\python.exe evaluate.py --name kitti001_recheck --experiment kitti001
.\.venv\Scripts\python.exe record_experiment.py evaluations/kitti001_recheck/metrics.json
```

`--checkpoint <PT>`로 별도 KITTI 모델도 평가할 수 있지만, 학습 정보가 없으면 epoch/담당자는 비워집니다.
부분 평가 `--limit 10`은 엑셀에 등록하지 않습니다. 엑셀 기록을 생략하려면 `--no-excel`을 사용합니다.

학습한 모델로 이미지 한 장을 확인하려면 다음 명령을 사용합니다. 기본 출력은 `outputs/demo/output.jpg`입니다.

```powershell
.\.venv\Scripts\python.exe run.py --checkpoint experiments/kitti001/training/weights/best.pt
```

## 6. 엑셀 자동 입력 범위

기존 예시 행은 그대로 보존하며 첫 실제 기록은 다음 빈 행에 들어갑니다.

| 자동 입력 | 사람이 보완 |
|---|---|
| 실험ID·날짜·담당자·변경 이유·가설·기준ID | 결과 해석, 다음 실험 |
| 구조 파일명·입력 크기·정밀도·완료 epoch | 정성적인 탐지 실패 사례 |
| Params·FLOPs·PT 파일 크기 | 별도 양자화 실험의 설계 |
| Car/Pedestrian/Cyclist AP40 | 주최 채점기와 최종 대조 결과 |
| p50/p95 latency·평균 E2E·CPU RSS·PC | |

평균 AP40, Drop Rate, FPS는 기존 취지대로 **엑셀 수식**으로 계산합니다.
기준 실험 ID가 있어야 Drop Rate가 계산되며, 구조 변경끼리의 하락률도 같은 식입니다.
이 값이 실제 양자화 전후 하락률이 되려면 동일 모델의 양자화 전후 실험을 기준/비교로 지정해야 합니다.
양자화 실행은 기본 파이프라인에 추가됐습니다. 최신 동작은 [양자화 가이드](QUANTIZATION.md)를 참조하세요.

- 동일 평가 ID는 두 번 기록하지 않고 기존 행을 보존합니다.
- 기록 전 엑셀과 같은 폴더의 `workbook_backups/`에 원본을 백업합니다. 현재 기본 경로는 `reports/workbook_backups/`입니다.
- Excel이 파일을 열고 있으면 닫은 뒤 `record_experiment.py`로 재시도하세요.
- 측정 PC·PyTorch 버전·추론 설정이 바뀌면 기존 워크북에 기록하지 않습니다. 엑셀과 함께 있는 `.conditions.json`에 기준을 저장합니다.
- 현재 양식의 60개 행 범위가 가득 차면 중단하고 JSON을 보존합니다. 예시 행은 필요할 때 직접 비우세요.
- 기존 COCO 결과는 새 KITTI 실험표에 자동 이관하지 않습니다.
- 엑셀 작성은 Codex 번들의 Node + `@oai/artifact-tool`을 사용합니다. 다른 PC에서는
  `ARTIFACT_NODE`(node 실행 파일), `ARTIFACT_NODE_MODULES`(해당 패키지 경로)를 지정해야 합니다.
  이 런타임이 없어도 학습/평가/JSON 저장은 완료되고 엑셀만 재시도 상태가 됩니다.

## 7. 측정 정의

`configs/evaluation.yaml` 기본값: 640×640 letterbox, CPU 1 thread, FP32, batch=1,
conf=0.001, NMS IoU=0.7, max_det=300, warm-up 50회, benchmark 500회.
이전 COCO 평가의 max_det=3000/warm-up=20과 다르므로 이전 속도 결과를 직접 비교하지 마세요.

- AP40: Car IoU >0.7, Pedestrian/Cyclist >0.5. 세 클래스 동일 비중, 0~100 단위.
- Moderate: GT 높이 >25px, 가림 <=1, 잘림 <=0.3. 탐지 높이 <25px 제외.
- `DontCare`와 관련 클래스 제외 규칙 및 KITTI 41슬롯 중 1..40 평균을 사용합니다.
- Latency: 전처리 + 추론 + NMS + 결과 CPU 복사. 디스크/디코딩 제외.
- E2E: 위 작업에 디스크 읽기와 이미지 디코딩을 포함한 평균. OS 파일 캐시의 영향을 받습니다.
- Memory: 추론 중 전체 프로세스 RSS 피크(decimal MB). 모델 단독 메모리나 GPU VRAM이 아닙니다.
- FLOPs: 학습 모델 구조의 THOP 추정 MACs×2 (G). 지원되지 않는 모듈이면 미측정으로 남깁니다.
- PT 크기: 디스크에 저장된 체크포인트 MB. 추론 FP32와 저장 tensor dtype은 별개입니다.
- max_det 상한에 닿은 이미지는 `max_det_reached_images`에 기록합니다.
- 주최 측 전용 평가 코드와 최종 일치는 아직 대조 전이며 **2D 박스 평가**만 구현했습니다.

## 8. 설치와 테스트

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest test_pipeline test_kitti_metrics
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

현재 PC의 `.venv`는 PyTorch 2.11.0 + CUDA 12.8 / torchvision 0.26.0을 사용합니다.
재설치할 때는 일반 의존성을 설치한 뒤 다음 명령을 실행하세요.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

학습과 학습 중 내부 dev 검증은 GPU 0을 사용하며, `--smoke-test`도 학습 설정의 장치를 사용합니다.
최종 추론 측정은 `configs/evaluation.yaml`의 CPU 조건을 유지합니다.
PyTorch 버전도 측정조건에 포함되므로 현재 환경은 `reports/트랙2_실험기록_torch211_cu128.xlsx`에 기록합니다.
이 엑셀과 `.conditions.json`은 한 쌍으로 보관하며 기본 경로는 `project_config.py`의 `WORKBOOK`에서 지정합니다.
추론 측정 장치까지 바꾸면 엑셀 측정조건도 별도 실험군으로 관리해야 합니다.

공식 자료:
- YOLO 라벨 형식: https://docs.ultralytics.com/datasets/detect/
- KITTI 평가 기준: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
