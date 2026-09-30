"""동일 체크포인트의 FP32/INT8 평가를 검증하고 변화량과 엑셀을 저장합니다."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from project_config import ROOT, WORKBOOK, CLASSES, read_ids
from record_experiment import node_runtime


def compare(before, after):
    for report, variant in [(before, 'fp32'), (after, 'int8')]:
        if not report.get('full_fixed_set') or report['conditions']['count'] != 1000 or report['experiment'].get('smoke_test'):
            raise ValueError('전체 고정 1,000장 평가만 비교 확정할 수 있습니다. smoke/부분 평가는 제외합니다.')
        q = report['quantization']
        if q['variant'] != variant or report['conditions']['backend'] != 'onnxruntime_cpu':
            raise ValueError('ONNX FP32 → INT8 쌍이 아닙니다.')
        if set(q['calibration']['ids']) & set(read_ids(ROOT/'eval_val.txt', 1000)):
            raise ValueError('DATA LEAKAGE: 보정 목록에 고정 eval이 포함됩니다.')
        scores = [report['per_class'][c]['ap40'] for c in CLASSES]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100 for v in scores):
            raise ValueError('클래스 AP40 결과가 불완전합니다.')
        if not math.isclose(sum(scores)/3, report['mean_ap40'], abs_tol=1e-8):
            raise ValueError('평균 AP40이 클래스 평균과 다릅니다.')
    for key in ['bundle_sha256', 'source_checkpoint_sha256', 'bundle_name']:
        if before['quantization'][key] != after['quantization'][key]:
            raise ValueError('다른 모델/양자화 실행의 결과는 연결할 수 없습니다.')
    ca = {k:v for k,v in before['conditions'].items() if k != 'precision'}
    cb = {k:v for k,v in after['conditions'].items() if k != 'precision'}
    if ca != cb:
        raise ValueError('FP32/INT8 측정 조건, 데이터 또는 평가 코드가 다릅니다.')
    if before['conditions']['precision'] != 'FP32' or after['conditions']['precision'] != 'INT8_QDQ':
        raise ValueError('정밀도 정보가 다릅니다.')
    def delta(a, b):
        if a is None or b is None:
            return dict(before=a, after=b, difference=None, reduction_percent=None)
        if not all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in [a,b]):
            raise ValueError('성능 수치가 유효하지 않습니다.')
        return dict(before=a, after=b, difference=b-a, reduction_percent=(a-b)/a*100 if a else None)
    metrics = dict(mean_ap40=delta(before['mean_ap40'],after['mean_ap40']),
                   parameters=delta(before['model']['parameters'],after['model']['parameters']),
                   flops_g=delta(before['model']['flops_g'],after['model']['flops_g']),
                   file_size_mb=delta(before['model']['file_size_mb'],after['model']['file_size_mb']),
                   peak_rss_mb=delta(before['peak_rss_mb'],after['peak_rss_mb']),
                   latency_p50_ms=delta(before['latency_ms']['median'],after['latency_ms']['median']),
                   latency_p95_ms=delta(before['latency_ms']['p95'],after['latency_ms']['p95']))
    metrics['mean_ap40']['retention_percent'] = after['mean_ap40']/before['mean_ap40']*100 if before['mean_ap40'] else None
    return dict(status='evaluation_complete', organizer_verified=False,
                name=before['quantization']['bundle_name'], created_at=datetime.now(timezone.utc).isoformat(),
                fp32=before, int8=after, metrics=metrics,
                per_class={c:delta(before['per_class'][c]['ap40'],after['per_class'][c]['ap40']) for c in CLASSES},
                note='Drop Rate=(FP32-INT8)/FP32*100; retention=INT8/FP32*100. FP32=0 gives undefined ratios. No organizer ranking formula supplied.')


def save_comparison(before_path, after_path, output, workbook=WORKBOOK, no_excel=False):
    reports = []
    expected = set(read_ids(ROOT/'eval_val.txt', 1000))
    for path in [Path(before_path), Path(after_path)]:
        report = json.loads(path.read_text(encoding='utf-8'))
        predictions = json.loads((path.parent/'predictions.json').read_text(encoding='utf-8'))
        if set(predictions) != expected:
            raise ValueError('저장된 예측 결과가 고정 eval 1,000장과 다릅니다.')
        reports.append(report)
    payload = compare(*reports)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if not no_excel:
        try:
            subprocess.run([str(node_runtime()),str(ROOT/'tools/update_quantization.mjs'),str(Path(workbook).resolve()),str(output)],
                           cwd=ROOT,check=True)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            (output.parent/'excel_pending.txt').write_text(str(error),encoding='utf-8')
            if isinstance(error, subprocess.CalledProcessError) and error.returncode == 75:
                print(f'평가 완료, 엑셀 기록 보류(파일 잠금/권한). 결과는 보존됐으며 다음 작업을 계속합니다: {output}', flush=True)
                return payload
            raise RuntimeError(f'비교 JSON은 저장됐으나 엑셀 저장에 실패했습니다: {output}') from error
        (output.parent/'excel_pending.txt').unlink(missing_ok=True)
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp32', type=Path, required=True)
    p.add_argument('--int8', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workbook', type=Path, default=WORKBOOK)
    p.add_argument('--no-excel', action='store_true')
    a=p.parse_args()
    save_comparison(a.fp32,a.int8,a.output,a.workbook,a.no_excel)
