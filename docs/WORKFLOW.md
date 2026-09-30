# 시스템을 이해하는 순서

## 연구 질문

“구조를 바꾸면 정확도와 실행 비용이 어떻게 달라지는가?”를 검증합니다.
데이터·학습 조건·측정 조건을 고정하고 구조를 바꿉니다. 배치까지 바꾸면 구조만의 효과라고 해석할 수 없습니다.

## 전체 실행

`experiment.py`의 `prepare()` → `run_training()` → `evaluate.py` 호출부터 보세요.

| 단계 | 담당 코드 | 목적 |
|---|---|---|
| 데이터 준비·검사 | prepare_kitti.py | 라벨 변환, train/dev/eval 누출 방지 |
| 학습 | train.py | YAML로 랜덤 초기화하고 GPU 학습 |
| 모델 선택 | 학습 중 내부 dev 검증 | dev fitness 기준 best.pt 선택 |
| 최종 평가 | evaluate.py, kitti_metrics.py | 고정 eval의 AP40과 실행 비용 측정 |
| 기록 | record_experiment.py, tools/update_workbook.mjs | JSON을 비교용 엑셀에 기록 |

## 설계도와 결과물

YAML은 설계도, PT는 학습된 가중치입니다. 구조를 바꾸면 다시 학습합니다.
--baseline은 비교 기록용이며 사전학습 옵션이 아닙니다.
실험마다 구조·설정·데이터 목록·코드 사본을 저장해 당시 조건을 확인할 수 있습니다.

## 데이터가 세 개인 이유

train은 가중치를 업데이트하고, dev는 모델을 선택하고, eval은 최종 측정합니다.
고정 eval 점수를 반복해서 보며 설계를 고르면 평가셋에 맞춰질 수 있으므로 참고한 이력도 기록하세요.
학습은 세 클래스의 모든 난이도를 사용하고 최종 평가는 Moderate 규칙을 적용합니다.

## 결과를 읽는 순서

experiment.json으로 조건과 상태 → results.csv로 학습 과정 → metrics.json으로 최종 결과 → 엑셀로 모델 간 비교.
엑셀 실패 때문에 학습을 다시 시작할 필요는 없습니다. JSON에서 기록만 재시도합니다.
속도는 같은 측정 PC와 소프트웨어·추론 조건끼리 비교합니다.

## 구현의 범위

현재는 FP32/INT8 2D 탐지 모델 비교입니다. quantization_pipeline.py가 train 전용 보정과 동일 CPU ONNX 백엔드 전후 평가를 연결합니다.
학습에는 DontCare 손실 마스크가 없으며 KITTI AP40은 주최 평가기와 최종 대조가 필요합니다.
