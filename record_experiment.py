"""평가 JSON을 엑셀 입력으로 전달합니다. 엑셀이 잠겨도 평가 JSON은 보존됩니다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from project_config import ROOT, WORKBOOK


def node_runtime():
    # 이 PC의 Codex 번들 런타임을 기본으로 사용합니다. 다른 PC는 환경변수로 지정합니다.
    base = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node = Path(os.environ.get('ARTIFACT_NODE', str(base / 'bin/node.exe')))
    modules = Path(os.environ.get('ARTIFACT_NODE_MODULES', str(base / 'node_modules')))
    if not node.is_file() or not (modules / '@oai/artifact-tool').exists():
        raise RuntimeError('엑셀 런타임이 없습니다. ARTIFACT_NODE 및 ARTIFACT_NODE_MODULES를 설정하세요.')
    link = ROOT / 'tools/node_modules'
    if not link.exists():
        if os.name == 'nt':
            # PowerShell의 LiteralPath 대신 ArgumentList로 고정된 두 경로만 전달합니다.
            subprocess.run(['powershell', '-NoProfile', '-Command',
                            'New-Item -ItemType Junction -Path $args[0] -Target $args[1] | Out-Null',
                            str(link), str(modules)], check=True)
        else:
            link.symlink_to(modules, target_is_directory=True)
    return node


def record(metrics, workbook=WORKBOOK):
    metrics, workbook = Path(metrics).resolve(), Path(workbook).resolve()
    report = json.loads(metrics.read_text(encoding='utf-8'))
    if report.get('quantization'):
        raise ValueError('양자화 결과는 compare_quantization.py로 FP32/INT8 쌍을 검증한 뒤 기록하세요.')
    if not report['full_fixed_set'] or report.get('experiment', {}).get('smoke_test'):
        raise ValueError('고정 1,000장 전체의 실제 성능 실험만 엑셀에 기록합니다.')
    try:
        subprocess.run([str(node_runtime()), str(ROOT / 'tools/update_workbook.mjs'),
                        'record', str(workbook), str(metrics)], cwd=ROOT, check=True)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        # 평가 완료 결과를 잃지 않고 나중에 같은 명령으로 재시도할 수 있습니다.
        (metrics.parent / 'excel_pending.txt').write_text(str(error), encoding='utf-8')
        print(f'엑셀 기록 미완료. 파일을 닫은 뒤 재시도하세요: python record_experiment.py "{metrics}"')
        return False
    pending = metrics.parent / 'excel_pending.txt'
    if pending.exists():
        pending.unlink()
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics', type=Path)
    p.add_argument('--workbook', type=Path, default=WORKBOOK)
    a = p.parse_args()
    raise SystemExit(0 if record(a.metrics, a.workbook) else 1)
