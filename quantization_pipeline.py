"""완료된 학습 결과를 양자화하고 고정 평가와 전후 비교를 순차 실행합니다."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
from project_config import ROOT, WORKBOOK


def run_pair(experiment, name, settings=ROOT/'configs/evaluation.yaml', calibration_count=128,
             workbook=WORKBOOK, no_excel=False):
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in [experiment,name]):
        raise ValueError('실험/변환 이름은 영문·숫자·_·-만 허용합니다.')
    metadata=json.loads((ROOT/'experiments'/experiment/'experiment.json').read_text(encoding='utf-8'))
    if metadata.get('status') != 'trained' or metadata.get('smoke_test'):
        raise ValueError('완료된 정규 학습 모델만 최종 양자화 비교를 실행합니다.')
    for path in [ROOT/'quantizations'/name, ROOT/'evaluations'/f'{name}_fp32', ROOT/'evaluations'/f'{name}_int8']:
        if path.exists():
            raise FileExistsError(f'기존 결과가 있습니다. 새 --name을 사용하세요: {path}')
    def run(script, *args):
        subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)
    # 별도 프로세스로 실행해 export/보정/FP32/INT8 사이에 모델 메모리를 공유하지 않습니다.
    run('quantize.py','--experiment',experiment,'--name',name,'--settings',settings,'--calibration-count',calibration_count)
    bundle=ROOT/'quantizations'/name/'bundle.json'
    for variant in ['fp32','int8']:
        run('evaluate.py','--name',f'{name}_{variant}','--bundle',bundle,'--variant',variant,'--settings',settings,'--no-excel')
    extra=['--no-excel'] if no_excel else []
    run('compare_quantization.py','--fp32',ROOT/'evaluations'/f'{name}_fp32'/'metrics.json',
        '--int8',ROOT/'evaluations'/f'{name}_int8'/'metrics.json',
        '--output',ROOT/'quantizations'/name/'comparison.json','--workbook',workbook,*extra)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--settings',type=Path,default=ROOT/'configs/evaluation.yaml')
    p.add_argument('--calibration-count',type=int,default=128)
    p.add_argument('--workbook',type=Path,default=WORKBOOK)
    p.add_argument('--no-excel',action='store_true')
    a=p.parse_args()
    run_pair(a.experiment,a.name,a.settings,a.calibration_count,a.workbook,a.no_excel)
