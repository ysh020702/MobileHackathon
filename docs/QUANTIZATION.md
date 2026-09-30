# 학습부터 양자화 전후 비교까지

## 새 실험 한 번 실행

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe experiment.py --name small_int8_01 --config configs/models/yolov8s_baseline.yaml --settings configs/baseline_train.yaml --reason "Small FP32/INT8 비교" --hypothesis "양자화 후 Moderate AP40 유지와 실행 비용 변화 확인" --author "담당자"
```

정규 실행의 기본 흐름은 다음과 같습니다.

1. 고정 eval 1,000장이 train/dev와 겹치지 않는지 검사.
2. YAML에서 랜덤 초기화 후 KITTI 학습, 내부 dev 기준 best.pt 선택.
3. 동일 best.pt를 FP32 ONNX로 변환하고 학습 이미지에서 PyTorch 출력과 수치 비교.
4. train에서 seed=42로 128장을 골라 정적 INT8 보정. eval은 사용하지 않음.
5. **CPU ONNX Runtime FP32**를 고정 1,000장으로 평가.
6. **동일 CPU ONNX Runtime INT8**을 같은 1,000장으로 평가.
7. 동일 모델·변환 해시·데이터·평가 코드·측정조건을 검증한 뒤 비교 JSON과 Excel 저장.

FP32/INT8 평가는 서로 다른 프로세스에서 순차 실행합니다. 전처리, confidence, NMS, 입력 크기,
threads, batch=1 조건은 같습니다. 측정할 때 다른 학습/추론 작업을 함께 실행하지 마세요.
현재 진행 중인 오래된 실행 명령은 이미 로드한 코드 흐름을 따르므로 완료 후 아래 별도 명령을 사용하세요.

## 이미 학습된 모델: 다시 학습하지 않기

```powershell
.\.venv\Scripts\python.exe quantization_pipeline.py --experiment baseline_s01 --name baseline_s01_int8
```

experiment.json 상태가 trained이고 체크포인트 해시가 일치해야 합니다. 중단된 last.pt를 완료 결과로 처리하지 않습니다.
이미 있는 변환 이름은 덮어쓰지 않으므로 재시도는 새 --name을 사용하세요.
`--calibration-count 256`으로 보정 수를 바꿀 수 있으며, 이 역시 양자화 실험 조건으로 기록됩니다.
보정 파일은 모델/방법별로 보관합니다. 여러 결과 중 eval 점수가 높은 것을 반복 선택하는 행위는
고정 평가셋에 대한 튜닝이 될 수 있으므로 보정 정책은 먼저 고정하세요.

## 현재 Excel에 어떻게 기록되는가

기존 `실험로그`에서 O/P/Q는 Car/Pedestrian/Cyclist AP40, R은 평균, S는 기준 행 대비 Drop Rate입니다.
기존 기능만으로는 --baseline이 단순한 다른 구조 모델일 수도 있어 양자화 효과라고 단정할 수 없었습니다.

새 기능은 검증된 쌍을 원본 엑셀에 한 번에 저장합니다.

- FP32 행: 평가 ID `<변환명>_fp32`, 정밀도 FP32.
- INT8 행: 평가 ID `<변환명>_int8`, 정밀도 INT8_QDQ. 기준 ID는 바로 그 FP32 행.
- `양자화비교` 시트: 평균 AP40 전후, 유지율, Drop Rate, 클래스별 AP40 변화(pp), p50/p95,
  RSS, 파라미터·FLOPs, ONNX 파일 크기와 변화량.
- `보고서요약`: 마지막으로 기록한 FP32/INT8 쌍을 연결.

기존 행은 보존하고 자동 백업합니다. 중복 쌍은 다시 쓰지 않습니다. 실제 실험로그에 빈 행 2개가 없으면 중단합니다.
Excel이 열려 저장에 실패해도 comparison.json과 평가 JSON은 남으며 아래 명령으로 기록을 재시도합니다.

```powershell
.\.venv\Scripts\python.exe compare_quantization.py --fp32 evaluations/baseline_s01_int8_fp32/metrics.json --int8 evaluations/baseline_s01_int8_int8/metrics.json --output quantizations/baseline_s01_int8/comparison.json
```

## 계산 정의

- Moderate mAP40 = (Car AP40 + Pedestrian AP40 + Cyclist AP40) / 3. 값은 0~100입니다.
- 유지율 (%) = INT8 mAP40 / FP32 mAP40 × 100.
- Drop Rate (%) = (FP32 mAP40 − INT8 mAP40) / FP32 mAP40 × 100.
- ΔAP40 (percentage points) = INT8 AP40 − FP32 AP40.
- 지연시간/RSS/파일 크기 감소율 = (FP32 − INT8) / FP32 × 100.
- FP32 AP40이 0이면 유지율·Drop Rate는 정의되지 않으므로 빈칸/null로 둡니다. 개선되면 Drop Rate가 음수일 수 있습니다.

주최 안내의 “유지율(Drop Rate)”은 두 용어가 혼용돼 있어 두 값을 모두 제공합니다.
주최 측 종합 점수 가중치나 상위 50% 판정 공식은 제공되지 않았으므로 임의 순위·합격 여부를 계산하지 않습니다.

## 양자화와 측정의 범위

ONNX Runtime static QDQ, signed INT8 activation/weight, per-channel 가중치를 사용합니다.
Conv/MatMul을 양자화하고 지원되지 않는 연산은 FP32로 남으므로 전체 연산이 INT8이라는 뜻은 아닙니다.
bundle.json에 실제 INT8 가중치 텐서 및 QDQ 노드 수를 기록합니다. CPU에서 반드시 빨라지는 것은 아닙니다.

파라미터 수와 FLOPs는 같은 **원본 FP32 구조 기준**입니다. 정밀도만 줄였다고 1/4로 계산하지 않습니다.
정수 비트 연산량은 별도로 추정하지 않습니다. 파일 크기는 전후 모두 ONNX 파일이며 PT와 비교하지 않습니다.
메모리는 각 평가 프로세스의 peak RSS(decimal MB)입니다. GPU VRAM이나 모델 가중치만의 메모리가 아닙니다.
latency는 전처리+forward+NMS+결과 변환을 포함하고 디스크/디코딩을 제외합니다.

`quantizations/<변환명>/bundle.json`에 보정 목록·이미지 해시, 원본 체크포인트 해시, 모델 해시,
소프트웨어 버전, 설정을 저장합니다. 기존 PyTorch 결과와는 백엔드가 다르므로 쌍 안에서 비교하세요.

## 확정 결과라는 표현의 범위

전체 1,000장 평가가 끝나고 검증을 통과해야 `evaluation_complete`로 기록합니다.
smoke-test/부분 평가/학습 미완료 모델은 최종 쌍으로 기록하지 않습니다.
이는 **이 프로그램의 평가 완료 결과**입니다. 기존 KITTI AP40 구현은 주최 공식 평가기와 대조 전이며
organizer_verified=false로 명시합니다. “주최 측 공식 확정 점수”라는 의미가 아닙니다.

## 기존 동작 유지 옵션

- `experiment.py ... --skip-quantization`: 기존 PyTorch 평가·Excel 기록만 수행.
- `--smoke-test`: 작은 학습/부분 평가만 수행. 양자화와 Excel 최종 기록 생략.
- `quantization_pipeline.py ... --no-excel`: 전후 평가와 비교 JSON까지만 저장.
- `--settings`: 양자화 파이프라인에서는 공통 평가 YAML 경로, experiment.py에서는 학습 YAML 경로입니다.

아직 실제 완료 모델의 공식 1,000장 양자화 결과는 생성하지 않았습니다. 구현 검증은 별도 CPU 소규모
학습·보정·부분 평가와 임시 Excel 수치 테스트로 수행했으며 테스트 숫자를 실험 엑셀에 넣지 않았습니다.
