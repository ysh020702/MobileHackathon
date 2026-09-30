"""한 번의 명령: KITTI 학습 → FP32/INT8 변환 → 고정 평가 → 전후 비교 Excel."""
import argparse
import subprocess
import sys
from pathlib import Path
from project_config import ROOT, DATA, MODEL
from prepare_kitti import prepare
from train import run_training


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--config', type=Path, default=MODEL)
    p.add_argument('--settings', type=Path, default=ROOT / 'configs/train.yaml')
    p.add_argument('--reason', required=True)
    p.add_argument('--hypothesis', required=True)
    p.add_argument('--author', default='')
    p.add_argument('--baseline', default='')
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--restart', action='store_true', help='실패/중단 실험을 보관하고 현재 설정으로 처음부터 학습')
    p.add_argument('--skip-quantization', action='store_true', help='기존 PyTorch 평가만 수행')
    p.add_argument('--calibration-count', type=int, default=128)
    a = p.parse_args()
    if not a.skip_quantization and not a.smoke_test:
        import onnx, onnxruntime  # 장시간 학습 전에 의존성 누락을 확인
        if a.calibration_count < 1:
            p.error('--calibration-count는 1 이상이어야 합니다.')
        for target in [ROOT/'quantizations'/(a.name+'_q'), ROOT/'evaluations'/(a.name+'_q_fp32'), ROOT/'evaluations'/(a.name+'_q_int8')]:
            if target.exists():
                p.error(f'기존 양자화 결과가 있습니다. 새 실험 이름을 사용하세요: {target}')
    prepare()  # 이미 준비됐다면 누출/무결성만 검증합니다.
    if not a.skip_quantization and not a.smoke_test:
        from project_config import read_ids
        if a.calibration_count > len(read_ids(DATA/'train_ids.txt')):
            p.error('--calibration-count가 train 이미지 수를 넘습니다.')
    run_training(a.name, a.config, a.settings, DATA, a.reason, a.hypothesis,
                 a.author, a.baseline, a.smoke_test, a.restart)
    if not a.skip_quantization and not a.smoke_test:
        subprocess.run([sys.executable,str(ROOT/'quantization_pipeline.py'), '--experiment',a.name,
                        '--name',a.name+'_q', '--calibration-count',str(a.calibration_count)],cwd=ROOT,check=True)
        raise SystemExit(0)
    command = [sys.executable, str(ROOT / 'evaluate.py'), '--name', a.name, '--experiment', a.name]
    if a.smoke_test:
        command += ['--limit', '10', '--no-excel']  # 실제 실험표에 동작 테스트를 넣지 않습니다.
    subprocess.run(command, cwd=ROOT, check=True)
