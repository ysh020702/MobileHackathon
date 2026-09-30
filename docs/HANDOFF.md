# 다른 PC로 전달할 때

## 전달할 파일

루트 코드·requirements.txt·eval_val.txt·기본 모델 YAML·README, configs/, docs/, tests/,
tools/update_workbook.mjs와 원본 데이터 두 폴더를 전달하세요.
비교할 연구 기록은 experiments/, evaluations/, reports/를 함께 전달합니다.

다음은 전달에서 제외하세요: .venv/, tools/node_modules/ 연결, __pycache__/, .matplotlib/,
Ultralytics/, outputs/, archive/. 이들은 새 PC의 환경 또는 과거 기록입니다.
datasets/kitti3/도 절대 경로를 포함하므로 전달에서 제외하고 원본으로 재생성하세요.
현재 PC에서 실행 중인 환경과 데이터를 삭제하라는 뜻은 아닙니다.

## 새 PC에서 한 번 준비

현재 검증 환경은 Windows, Python 3.14, RTX 5070 Ti Laptop GPU입니다.
복사된 가상환경을 사용하지 말고 새로 만드세요. 아래 CUDA 패키지는 현재 PC의 검증 조합이며,
다른 GPU/드라이버에서는 해당 환경에 맞는 빌드를 선택해야 합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe prepare_kitti.py
```

원본 경로는 data_object_image_2/training/image_2/와 data_object_label_2/training/label_2/입니다.
이미 datasets/kitti3/까지 복사했다면 새 PC에서 해당 폴더를 다른 이름으로 보관한 후 prepare_kitti.py를 실행하세요.
고정 eval_val.txt와 seed=42, dev_fraction=0.1은 유지합니다.

## 권장 협업 방식

각 PC에서 학습하고 최종 속도·메모리는 한 PC에서 측정하세요. 학습만 실행하려면:

```powershell
.\.venv\Scripts\python.exe train.py --name junior01 --config configs/models/yolov8n_width0125.yaml --reason "width 축소" --hypothesis "AP40 대비 속도 개선"
```

수정 구조 파일은 먼저 루트 README의 복사·수정 단계로 만드세요.
완료 후 experiments/junior01/ 전체를 측정 담당자에게 전달합니다. 측정 PC에서는:

```powershell
.\.venv\Scripts\python.exe evaluate.py --name junior01_measure --checkpoint experiments/junior01/training/weights/best.pt --reason "후배 모델 공통 PC 측정" --no-excel
```

--checkpoint 평가는 학습 메타데이터를 자동 연결하지 않으므로 전달받은 experiment.json도 함께 비교하세요.
다른 PC의 실험 JSON에는 절대 경로가 있으므로 --experiment로 바로 읽지 마세요.

## 엑셀 기능

기존 엑셀은 현재 측정 PC·PyTorch 버전 조건에 연결돼 있습니다. 다른 PC 결과를 섞지 않습니다.
자동 기록에는 Codex 번들의 Node와 @oai/artifact-tool이 필요하며 pip 설치만으로 준비되지 않습니다.
해당 런타임이 있으면 ARTIFACT_NODE와 ARTIFACT_NODE_MODULES 환경변수로 경로를 지정합니다.
없어도 학습·평가 JSON은 저장됩니다. 새 PC의 구조 연구를 엑셀 설치에 의존시키지 마세요.

## 코드 수정 후 검증

```powershell
.\.venv\Scripts\python.exe -m unittest test_pipeline test_kitti_metrics
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

엑셀 테스트에는 위 Node 런타임이 필요합니다. 일반 구조 변경은 루트 README의 smoke-test로 확인하세요.
