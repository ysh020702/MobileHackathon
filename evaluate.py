"""KITTI 2D Moderate AP40: 고정 1,000장, Car/Pedestrian/Cyclist 및 PR 곡선.
주최 측 채점기와의 최종 대조는 별도로 필요합니다. 학습은 수행하지 않습니다.
"""
import argparse
import hashlib
import json
import platform
import shutil
import threading
import time
from pathlib import Path

from project_config import ROOT, CLASSES, read_ids
from datetime import datetime, timezone
from copy import deepcopy
import yaml
import cv2
import matplotlib
matplotlib.use('Agg')  # 창을 띄우지 않고 PNG로 저장
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import ultralytics
from ultralytics import YOLO

from kitti_metrics import CLASSES, MIN_OVERLAP, score_class
MODEL_NAMES = {'Car': ('car',), 'Pedestrian': ('person', 'pedestrian'), 'Cyclist': ('cyclist',)}
PROTOCOL = dict(name='KITTI_2D_Moderate_AP40', classes=CLASSES,
                iou=MIN_OVERLAP, gt_height='>25', detection_height='>=25',
                max_occlusion=1, max_truncation=0.3,
                ap='KITTI 41 slots; interpolated precision slots 1..40 averaged; percent',
                reference='https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d')


def read_truth(ids):
    truth, digest = {}, hashlib.sha256()
    for image_id in ids:
        image_path = ROOT / f'data_object_image_2/training/image_2/{image_id}.png'
        label_path = ROOT / f'data_object_label_2/training/label_2/{image_id}.txt'
        digest.update(image_id.encode())
        digest.update(image_path.read_bytes())
        raw = label_path.read_bytes()
        digest.update(raw)
        rows = []
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = line.split()
            if len(r) != 15:
                raise ValueError(f'Invalid KITTI label: {label_path}')
            box = list(map(float, r[4:8]))
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f'Invalid KITTI box: {label_path}')
            rows.append((r[0], float(r[1]), int(r[2]), box))
        truth[image_id] = rows
    return truth, digest.hexdigest()


def plot_curves(scores, out, count):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, score in scores.items():
        label = f"{name}: AP40={score['ap40']:.2f}" if score['ap40'] is not None else f'{name}: no valid GT'
        axes[0].plot(score['raw_recall'], score['raw_precision'], '.-', label=name)
        axes[1].step(score['recall_grid'], score['precision_envelope'], where='post', label=label)
    for ax in axes:
        ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_title('Before interpolation (KITTI score thresholds)')
    axes[1].set_title('KITTI R40 sampled precision envelope')
    fig.suptitle(f'KITTI 2D Moderate — {count} images' + (' (DEBUG subset)' if count != 1000 else ''))
    fig.tight_layout()
    fig.savefig(out / 'pr_curve.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True, help='새 실험 이름 (기존 결과 덮어쓰기 금지)')
    parser.add_argument('--checkpoint', type=Path, help='KITTI 3클래스로 학습한 PT')
    parser.add_argument('--experiment', help='experiments/<이름>/experiment.json에서 가중치/학습정보 읽기')
    parser.add_argument('--bundle', type=Path, help='양자화 bundle.json (동일 CPU ONNX 백엔드 평가)')
    parser.add_argument('--variant', choices=['fp32', 'int8'], default='fp32')
    parser.add_argument('--settings', type=Path, default=ROOT / 'configs/evaluation.yaml')
    parser.add_argument('--no-excel', action='store_true', help='엑셀 자동 기록 생략')
    parser.add_argument('--limit', type=int, default=0, help='빠른 확인용 앞 N장, 0이면 전체')
    parser.add_argument('--reason', default='', help='수정 이유; 생략하면 학습 실험에서 가져옴')
    parser.add_argument('--hypothesis', default='기준값 측정', help='예상 효과와 허용할 손해')
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
        parser.error('name에는 영문, 숫자, _, -만 사용하세요.')
    if args.limit < 0:
        parser.error('limit은 0 이상이어야 합니다.')
    out = ROOT / 'evaluations' / args.name
    if out.exists():
        parser.error('이미 있는 실험입니다. 새로운 name을 사용하세요.')
    ids = read_ids(ROOT / 'eval_val.txt', 1000)
    ids = ids[:args.limit] if args.limit else ids
    metadata = {}
    bundle = None
    if args.bundle:
        if args.experiment or args.checkpoint:
            parser.error('--bundle은 --experiment/--checkpoint와 함께 사용할 수 없습니다.')
        bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
        if bundle['status'] != 'ready' or bundle['eval_sha256'] != hashlib.sha256((ROOT/'eval_val.txt').read_bytes()).hexdigest():
            raise ValueError('미완료 변환 또는 고정 eval 목록 불일치')
        metadata = dict(bundle['experiment'])
        args.checkpoint = args.bundle.parent / bundle['artifacts'][args.variant]['filename']
        if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != bundle['artifacts'][args.variant]['sha256']:
            raise ValueError('ONNX 파일 해시 불일치')
    if args.experiment:
        metadata = json.loads((ROOT / 'experiments' / args.experiment / 'experiment.json').read_text(encoding='utf-8'))
        if metadata['status'] != 'trained':
            raise ValueError('학습이 완료된 실험만 평가할 수 있습니다.')
        if metadata['eval_sha256'] != hashlib.sha256((ROOT / 'eval_val.txt').read_bytes()).hexdigest():
            raise ValueError('학습 시점의 고정 평가 목록과 다릅니다.')
        if args.checkpoint and args.checkpoint.resolve() != Path(metadata['checkpoint']).resolve():
            raise ValueError('실험 정보와 checkpoint가 다릅니다.')
        args.checkpoint = Path(metadata['checkpoint'])
    if not args.checkpoint or not args.checkpoint.is_file():
        parser.error('--experiment 또는 실제 KITTI --checkpoint가 필요합니다. COCO는 사용하지 않습니다.')
    settings = yaml.safe_load(args.settings.read_text(encoding='utf-8'))
    if bundle and (settings != bundle['settings'] or settings['device'] != 'cpu'):
        raise ValueError('변환 시점과 평가 조건이 다릅니다.')
    size = settings['imgsz']
    if not isinstance(size, int) or size <= 0 or size % 32:
        raise ValueError('imgsz는 양의 32 배수여야 합니다.')
    truth, data_hash = read_truth(ids)
    torch.manual_seed(0)
    torch.set_num_threads(settings['threads'])
    weights = args.checkpoint
    weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if metadata and not bundle and weights_hash != metadata['checkpoint_sha256']:
        raise ValueError('학습 후 checkpoint 내용이 변경됐습니다.')
    if bundle:
        from deployment import OnnxDetector
        model = OnnxDetector(weights, settings)
        flops = bundle['model']['flops_g']
        parameters = bundle['model']['parameters']
        names = {i: name.lower() for i, name in enumerate(CLASSES)}
    else:
        model = YOLO(str(weights), task='detect')
        from ultralytics.utils.torch_utils import get_flops
        flops = get_flops(deepcopy(model.model), imgsz=size)
        parameters = sum(p.numel() for p in model.model.parameters())
        names = {int(k): v.lower() for k, v in model.names.items()}
    mapping = {k: cls for k, name in names.items() for cls in CLASSES if name in MODEL_NAMES[cls]}
    missing_classes = sorted(set(CLASSES) - set(mapping.values()))
    if missing_classes or len(names) != 3:
        raise ValueError(f'KITTI 3클래스 모델이 필요합니다. 현재 클래스: {names}')
    out.mkdir(parents=True)
    shutil.copy2(args.settings, out / 'evaluation_settings.yaml')
    if metadata and not bundle:
        # 새 기록은 원본 파일명을 보존하며 이전 고정 파일명 기록도 읽습니다.
        snapshot = metadata.get('architecture_snapshot', 'yolov8n_architecture.yaml')
        shutil.copy2(ROOT / 'experiments' / args.experiment / snapshot, out / 'architecture.yaml')
    predictions = {name: {image_id: [] for image_id in ids} for name in CLASSES}
    latency, saved_predictions, capped_images = [], {}, []
    # 낮은 confidence부터 예측을 모아 PR 곡선을 계산합니다. 0.25로 자르면 AP가 왜곡됩니다.
    def predict(image):
        if bundle:
            return model.predict_rows(image)
        return model.predict(image, imgsz=size, device=settings['device'], conf=settings['conf'], iou=settings['iou'],
                             max_det=settings['max_det'], rect=False, verbose=False)[0].boxes.data.cpu().tolist()
    # RSS는 모델 단독 크기가 아닌 전체 프로세스 메모리입니다. 10ms마다 샘플링합니다.
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample_memory():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)
    watcher = threading.Thread(target=sample_memory, daemon=True)
    watcher.start()
    bench_latency, end_to_end = [], []
    def sync():
        if str(settings['device']) != 'cpu' and torch.cuda.is_available():
            torch.cuda.synchronize()
    try:
        first = cv2.imread(str(ROOT / f'data_object_image_2/training/image_2/{ids[0]}.png'))
        if first is None:
            raise ValueError('첫 이미지 디코딩 실패')
        for _ in range(settings['warmup']):
            predict(first)
        for index, image_id in enumerate(ids):
            image = cv2.imread(str(ROOT / f'data_object_image_2/training/image_2/{image_id}.png'))
            if image is None:
                raise ValueError(f'이미지 디코딩 실패: {image_id}')
            sync()
            start = time.perf_counter()
            rows = predict(image)
            sync()
            latency.append((time.perf_counter() - start) * 1000)
            if len(rows) >= settings['max_det']:
                # max_det은 고정 추론 조건입니다. 상한에 닿은 이미지도 기록해 해석에 사용합니다.
                capped_images.append(image_id)
            saved_predictions[image_id] = rows
            for x1, y1, x2, y2, conf, cls in rows:
                if int(cls) in mapping:
                    predictions[mapping[int(cls)]][image_id].append([x1, y1, x2, y2, conf])
            if (index + 1) % 100 == 0:
                print(f'{index+1}/{len(ids)}장 완료', flush=True)
        # 정확도 평가와 별도로 고정 반복 횟수로 속도를 측정합니다.
        # E2E에는 디스크 읽기/디코딩이 포함되고, latency에는 포함되지 않습니다.
        for index in range(settings['benchmark_repetitions']):
            image_id = ids[index % len(ids)]
            sync()
            outer = time.perf_counter()
            image = cv2.imread(str(ROOT / f'data_object_image_2/training/image_2/{image_id}.png'))
            start = time.perf_counter()
            predict(image)
            sync()
            stop_time = time.perf_counter()
            bench_latency.append((stop_time-start)*1000)
            end_to_end.append((stop_time-outer)*1000)
    finally:
        stop.set()
        watcher.join()
    scores = {name: score_class(truth, predictions[name], name) for name in CLASSES}
    valid = all(scores[name]['gt'] > 0 for name in CLASSES)
    mean_ap40 = float(np.mean([scores[c]['ap40'] for c in CLASSES])) if valid else None
    conditions = dict(protocol=PROTOCOL, data_sha256=data_hash, count=len(ids), **settings,
                      rect=False, precision='INT8_QDQ' if bundle and args.variant == 'int8' else 'FP32', batch=1, seed=0,
                      host=platform.node(), platform=platform.platform(), processor=platform.processor(),
                      ram_mb=psutil.virtual_memory().total / 1e6,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
                      cuda=torch.version.cuda, torch=torch.__version__, ultralytics=ultralytics.__version__,
                      numpy=np.__version__, opencv=cv2.__version__,
                      evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes() + (ROOT / 'kitti_metrics.py').read_bytes()).hexdigest())
    conditions['backend'] = 'onnxruntime_cpu' if bundle else 'pytorch'
    if bundle:
        import onnxruntime
        conditions.update(onnxruntime=onnxruntime.__version__,
                          deployment_sha256=hashlib.sha256((ROOT/'deployment.py').read_bytes()).hexdigest())
    report = dict(name=args.name, created_at=datetime.now(timezone.utc).isoformat(),
                  reason=args.reason or metadata.get('reason', ''),
                  hypothesis=metadata.get('hypothesis', args.hypothesis), experiment=metadata,
                  conditions=conditions,
                  model=dict(mode='KITTI_checkpoint', weights_sha256=weights_hash, parameters=parameters,
                             flops_g=flops if flops > 0 else None, file_size_mb=weights.stat().st_size / 1e6),
                  mean_ap40=mean_ap40, score_unit='percent', full_fixed_set=len(ids) == 1000,
                  missing_model_classes=missing_classes, max_det_reached_images=capped_images,
                  latency_ms=dict(mean=float(np.mean(bench_latency)), median=float(np.median(bench_latency)),
                                  p95=float(np.percentile(bench_latency, 95)), samples=bench_latency,
                                  scope='preprocess + forward + NMS + CPU boxes; disk/decode excluded'),
                  e2e_ms=dict(mean=float(np.mean(end_to_end)), median=float(np.median(end_to_end))),
                  evaluation_latency_ms=latency,
                  peak_rss_mb=peak[0]/1e6, memory_scope='whole process RSS; not GPU VRAM', per_class=scores,
                  note='KITTI 2D Moderate R40 implementation; organizer parity not yet verified. --limit is debug only.')
    if bundle:
        report['quantization'] = dict(bundle_name=bundle['name'], variant=args.variant,
                                     source_checkpoint_sha256=bundle['source_checkpoint_sha256'],
                                     bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                                     method=bundle['method'], calibration=bundle['calibration'])
        report['model']['mode'] = 'ONNX_' + args.variant
        report['model']['complexity_scope'] = bundle['model']['complexity_scope']
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'predictions.json').write_text(json.dumps(saved_predictions), encoding='utf-8')
    plot_curves(scores, out, len(ids))
    for name, score in scores.items():
        print(f"{name}: Moderate GT={score['gt']}, AP40={score['ap40']}")
    print(f'Mean Moderate AP40={mean_ap40}%, median={np.median(latency):.2f} ms')
    if len(ids) != 1000:
        print('DEBUG SUBSET: 공식 고정 1,000장 점수가 아닙니다.')
    print(f'저장: {out}')
    if not bundle and not args.no_excel and report['full_fixed_set'] and not metadata.get('smoke_test', False):
        from record_experiment import record
        record(out / 'metrics.json')
    else:
        print('엑셀 자동 기록 생략: 부분 평가 / smoke test / --no-excel')



if __name__ == '__main__':
    main()
