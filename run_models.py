"""configs/models에 남아 있는 모든 모델 YAML을 순서대로 실행합니다."""
import argparse
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

# 여기만 수정하세요. {model}은 모델 YAML의 파일명(확장자 제외)으로 바뀝니다.
MODEL_DIR = ROOT / 'configs/models'
SETTINGS = ROOT / 'configs/baseline_train.yaml'
RUN_SUFFIX = 'quant01'
REASON = '{model} 양자화 Baseline'
HYPOTHESIS = '양자화 전후 AP40 및 실행 비용 비교'
AUTHOR = '양승환'


def build_commands():
    models = sorted(
        (p for p in MODEL_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.yaml', '.yml'}),
        key=lambda p: p.name.lower(),
    )
    if not models:
        raise ValueError(f'실행할 모델 YAML이 없습니다: {MODEL_DIR}')
    if not SETTINGS.is_file():
        raise ValueError(f'학습 설정 파일이 없습니다: {SETTINGS}')
    commands, names = [], set()
    for model in models:
        name = f'{model.stem}_{RUN_SUFFIX}'
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'모델 파일명과 RUN_SUFFIX에는 영문, 숫자, _, -만 사용하세요: {name}')
        if name.lower() in names:
            raise ValueError(f'실험 이름이 중복됩니다: {name}')
        names.add(name.lower())
        for output in (ROOT / 'experiments' / name,
                       ROOT / 'quantizations' / f'{name}_q',
                       ROOT / 'evaluations' / f'{name}_q_fp32',
                       ROOT / 'evaluations' / f'{name}_q_int8'):
            if output.exists():
                raise ValueError(f'기존 결과가 있습니다: {output}\n'
                                 '완료 모델 YAML을 실행 폴더 밖으로 옮기거나 RUN_SUFFIX를 바꾸세요.')
        command = [sys.executable, str(ROOT / 'experiment.py'),
                   '--name', name, '--config', str(model), '--settings', str(SETTINGS),
                   '--reason', REASON.replace('{model}', model.stem),
                   '--hypothesis', HYPOTHESIS.replace('{model}', model.stem), '--author', AUTHOR]
        commands.append((name, command))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='실제 학습 없이 실행 명령만 확인')
    args = parser.parse_args()
    try:
        commands = build_commands()
        for index, (name, command) in enumerate(commands, 1):
            print(f'[{index}/{len(commands)}] {name}', flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
                continue
            subprocess.run(command, cwd=ROOT, check=True)
            print(f'[{index}/{len(commands)}] 완료 — {name}', flush=True)
        print('실행 계획 확인 완료' if args.dry_run else '전체 완료', flush=True)
    except (ValueError, OSError) as error:
        print(f'실행 중지: {error}', file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f'실행 실패(종료 코드 {error.returncode}). 다음 모델은 실행하지 않습니다.', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\n사용자가 중단했습니다. 다음 모델은 실행하지 않습니다.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
