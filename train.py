"""KITTI 3클래스 YOLO를 처음부터 학습하고 실험별 설정/가중치를 보존합니다."""
import argparse
import csv
from datetime import datetime, timezone
import json
import random
import re
import shutil
from pathlib import Path

from project_config import ROOT, DATA, MODEL, CLASSES
from prepare_kitti import validate_dataset, sha256
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import LOGGER


def announce_best_model(trainer):
    """체크포인트 저장 후 best.pt가 갱신된 경우에만 알립니다(동점 포함)."""
    if trainer.fitness is None or trainer.fitness != trainer.best_fitness:
        return
    checkpoint = Path(trainer.best)
    if not checkpoint.is_file():
        return
    scores = ' | '.join(
        f'{label}={float(trainer.metrics[key]):.4f}'
        for key, label in [('metrics/mAP50(B)', 'dev mAP50'),
                           ('metrics/mAP50-95(B)', 'dev mAP50-95')]
        if key in trainer.metrics
    )
    LOGGER.info(
        f'\n[BEST MODEL SAVED] Epoch {trainer.epoch + 1}/{trainer.epochs}'
        f' | dev fitness={float(trainer.fitness):.6f}'
        f'{" | " + scores if scores else ""}\n'
        f'  저장: {checkpoint}\n'
        '  선택 기준: 내부 dev fitness (최종 KITTI Moderate AP40은 별도 평가)'
    )


def resolve_architecture_scale(architecture, spec):
    """Ultralytics의 파일명 추정과 실제 scales 선택이 일치하는지 확인합니다."""
    scales = spec.get('scales')
    inferred = guess_model_scale(Path(architecture))
    explicit = spec.get('scale')
    if not scales:
        if explicit:
            raise ValueError('scale을 지정하려면 scales도 정의하세요.')
        return None
    selected = inferred or (next(iter(scales)) if len(scales) == 1 else None)
    if selected is None:
        raise ValueError('scales가 여러 개이면 yolov8s_custom.yaml처럼 파일명에 scale을 명시하세요.')
    if selected not in scales:
        raise ValueError(f'파일명에서 추정한 scale={selected!r}가 scales에 없습니다: {list(scales)}')
    if explicit and explicit != selected:
        raise ValueError(f'YAML scale={explicit!r}와 파일명/단일 scales 선택={selected!r}가 충돌합니다. 파일명과 scale을 맞추세요.')
    return selected


def check_restart(out, restart=False):
    if not out.exists():
        return
    if not restart:
        raise FileExistsError(f'기존 실험은 덮어쓰지 않습니다: {out}. 중단/실패 실험을 처음부터 다시 시작하려면 --restart를 추가하세요.')
    metadata = json.loads((out / 'experiment.json').read_text(encoding='utf-8'))
    if metadata.get('status') not in {'failed', 'interrupted'}:
        raise ValueError('실패/중단 상태의 실험만 --restart할 수 있습니다. 완료됐거나 실행 상태가 불명확하면 새 실험 이름을 사용하세요.')
    if (ROOT / 'evaluations' / out.name).exists():
        raise ValueError('같은 이름의 평가 기록이 있습니다. 기록을 보존하려면 새 실험 이름을 사용하세요.')


def archive_for_restart(out):
    # 심볼릭 링크나 프로젝트 밖의 경로는 이동하지 않습니다.
    if out.is_symlink() or out.resolve().parent != (ROOT / 'experiments').resolve():
        raise ValueError('실험 보관 경로가 프로젝트 experiments 폴더 밖입니다.')
    destination = ROOT / 'archive' / 'restarted_experiments' / f'{out.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('보관 대상 경로가 프로젝트 밖입니다.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.rename(destination)
    print(f'이전 실험 보관: {destination}')


def run_training(name, architecture=MODEL, settings=ROOT / 'configs/train.yaml',
                 data=DATA, reason='', hypothesis='', author='', baseline='', smoke=False, restart=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('실험 이름에는 영문, 숫자, _, -만 사용하세요.')
    out = ROOT / 'experiments' / name
    check_restart(out, restart)
    manifest = validate_dataset(data)  # 학습 시작 전 가장 먼저 데이터 누출 검사
    args = yaml.safe_load(Path(settings).read_text(encoding='utf-8'))
    allowed = {'epochs', 'batch', 'imgsz', 'device', 'workers', 'seed', 'optimizer', 'lr0',
               'momentum', 'weight_decay', 'patience', 'deterministic', 'amp', 'cache',
               'plots', 'close_mosaic', 'mosaic', 'fliplr', 'translate', 'scale', 'hsv_h', 'hsv_s', 'hsv_v'}
    if set(args) - allowed:
        raise ValueError(f'허용하지 않은 학습 설정: {set(args)-allowed}; data/pretrained/resume 변경은 금지합니다.')
    if args.get('amp') is not False:
        raise ValueError('이 파이프라인은 외부 사전학습 모델 검사 없이 FP32로 학습합니다: amp=false')
    spec = yaml.safe_load(Path(architecture).read_text(encoding='utf-8'))
    if spec.get('nc') != 3:
        raise ValueError('모델 YAML nc는 3이어야 합니다.')
    architecture_scale = resolve_architecture_scale(architecture, spec)
    if out.exists():
        check_restart(out, restart)
        archive_for_restart(out)
    out.mkdir(parents=True)
    # 구현도 보존해야 나중에 코드가 바뀌어도 당시 실험을 해석할 수 있습니다.
    source_dir = out / 'source'
    source_dir.mkdir()
    for filename in ('train.py', 'prepare_kitti.py', 'evaluate.py', 'kitti_metrics.py',
                     'project_config.py', 'requirements.txt', 'experiment.py', 'quantize.py',
                     'deployment.py', 'quantization_pipeline.py', 'compare_quantization.py'):
        shutil.copy2(ROOT / filename, source_dir / filename)
    # Ultralytics가 파일명에서 scale을 추정하므로 원본 이름을 보존합니다.
    # 별도 폴더로 분리해 train_settings.yaml 등 실험 기록과의 이름 충돌도 방지합니다.
    snapshot = out / 'architecture' / Path(architecture).name
    snapshot.parent.mkdir()
    shutil.copy2(architecture, snapshot)
    shutil.copy2(settings, out / 'train_settings.yaml')
    shutil.copy2(Path(data) / 'manifest.json', out / 'dataset_manifest.json')
    for filename in ('train_ids.txt', 'dev_ids.txt', 'eval_ids.txt'):
        shutil.copy2(Path(data) / filename, out / filename)
    data_yaml = Path(data) / 'dataset.yaml'
    if smoke:
        # 빠른 통합 테스트도 고정 평가셋을 쓰지 않습니다. 성능 실험과 구분해 기록합니다.
        cfg = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
        for split, count in [('train', 8), ('dev', 4)]:
            lines = (Path(data) / f'{split}.txt').read_text(encoding='utf-8').splitlines()[:count]
            (out / f'smoke_{split}.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
            cfg['train' if split == 'train' else 'val'] = (out / f'smoke_{split}.txt').as_posix()
        data_yaml = out / 'smoke_dataset.yaml'
        data_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
        args.update(epochs=1, batch=2, imgsz=128, workers=0, plots=False, close_mosaic=0)
    metadata = dict(name=name, created_at=datetime.now(timezone.utc).isoformat(), status='training',
                    reason=reason, hypothesis=hypothesis, author=author, baseline=baseline,
                    initialization='random_from_yaml_no_COCO', architecture=str(Path(architecture).resolve()),
                    architecture_snapshot=snapshot.relative_to(out).as_posix(), architecture_scale=architecture_scale,
                    architecture_sha256=sha256(snapshot), settings=args, classes=list(CLASSES),
                    dataset_counts=manifest['counts'], eval_sha256=manifest['eval_sha256'],
                    torch_version=torch.__version__, numpy_version=np.__version__, smoke_test=smoke)
    def save():
        (out / 'experiment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        # YAML만으로 생성하며 .load(...pt)를 호출하지 않습니다.
        random.seed(args['seed']); np.random.seed(args['seed']); torch.manual_seed(args['seed'])
        torch.set_num_threads(1)
        model = YOLO(str(snapshot), task='detect')
        model.add_callback('on_model_save', announce_best_model)
        model.train(data=str(data_yaml), pretrained=False, project=str(out), name='training',
                    exist_ok=False, **args)
        checkpoint = Path(model.trainer.best)
        if not checkpoint.is_file():
            raise RuntimeError('학습 결과 best.pt가 없습니다.')
        # best.pt는 내부 dev의 Ultralytics fitness로 선택됩니다. 공식 AP40 선택이 아닙니다.
        results = list(csv.DictReader((Path(model.trainer.save_dir) / 'results.csv').open(encoding='utf-8')))
        metadata.update(status='trained', checkpoint=str(checkpoint), completed_epochs=len(results),
                        checkpoint_sha256=sha256(checkpoint),
                        selection='internal dev Ultralytics fitness; final score is separate KITTI AP40')
        save()
        return metadata
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=str(error) or type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--config', type=Path, default=MODEL)
    p.add_argument('--settings', type=Path, default=ROOT / 'configs/train.yaml')
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--reason', required=True)
    p.add_argument('--hypothesis', default='')
    p.add_argument('--author', default='')
    p.add_argument('--baseline', default='')
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--restart', action='store_true', help='실패/중단 실험을 보관하고 현재 설정으로 처음부터 학습')
    a = p.parse_args()
    run_training(a.name, a.config, a.settings, a.data, a.reason, a.hypothesis, a.author, a.baseline, a.smoke_test, a.restart)
