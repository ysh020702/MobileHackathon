"""KITTI 원본 라벨 → YOLO 3클래스 라벨. 평가 1,000장의 학습 유입을 차단합니다."""
import argparse
from collections import Counter
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image
import yaml
from project_config import ROOT, DATA, CLASSES, read_ids


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_label(text, width, height):
    """KITTI의 픽셀 xyxy를 YOLO의 정규화된 중심점/크기 xywh로 바꿉니다.

    원본은 보존합니다. 학습에는 난이도에 관계없이 3개 클래스의 모든 박스를 씁니다.
    DontCare 등은 YOLO 학습 라벨에서 제외합니다. 표준 YOLO에는 ignore-region 손실
    마스크가 없으므로 제외 영역이 배경으로 취급된다는 한계가 있습니다.
    평가에서는 원본 KITTI 라벨을 읽어 DontCare를 정확히 처리합니다.
    """
    labels, omitted = [], Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 15:
            raise ValueError(f'KITTI 라벨 열 수 오류: {line}')
        name = fields[0]
        if name not in CLASSES:
            omitted[name] += 1
            continue
        x1, y1, x2, y2 = map(float, fields[4:8])
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError(f'NaN/무한대 박스: {line}')
        # 경계 밖 박스는 이미지 범위로 자릅니다. 정답의 클래스 의미는 바꾸지 않습니다.
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'유효하지 않은 박스: {line}')
        labels.append(f'{CLASSES.index(name)} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} '
                      f'{(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    return '\n'.join(labels) + ('\n' if labels else ''), omitted


def validate_dataset(data=DATA):
    """학습 직전에 목록·라벨 무결성과 평가셋 비중복을 다시 검사합니다."""
    data = Path(data).resolve()
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    evaluation = set(read_ids(ROOT / 'eval_val.txt', 1000))
    if manifest['eval_sha256'] != sha256(ROOT / 'eval_val.txt'):
        raise ValueError('고정 평가 목록이 준비 시점과 다릅니다. 데이터셋을 새 경로에 다시 준비하세요.')
    sets = {s: set(read_ids(data / f'{s}_ids.txt')) for s in ('train', 'dev')}
    if sets['train'] & sets['dev'] or (sets['train'] | sets['dev']) & evaluation:
        raise ValueError('DATA LEAKAGE: train/dev/eval 목록이 겹칩니다. 학습을 중단합니다.')
    for relative, digest in manifest['file_hashes'].items():
        if sha256(data / relative) != digest:
            raise ValueError(f'준비된 파일이 변경됐습니다: {relative}')
    for split in ('train', 'dev'):
        paths = (data / f'{split}.txt').read_text(encoding='utf-8').splitlines()
        if {Path(p).stem for p in paths} != sets[split] or len(paths) != len(sets[split]):
            raise ValueError(f'{split} 실제 이미지 목록이 ID 목록과 다릅니다.')
        for path in paths:
            p = Path(path)
            if p.parent.resolve() != (data / 'images').resolve() or not p.is_file():
                raise ValueError(f'허용되지 않은 이미지 경로: {p}')
    return manifest


def prepare(output=DATA, seed=42, dev_fraction=0.1):
    output = Path(output).resolve()
    if output.exists():
        result = validate_dataset(output)
        if result['seed'] != seed or result['dev_fraction'] != dev_fraction:
            raise ValueError('기존 분할 설정과 다릅니다. --output으로 새 폴더를 지정하세요.')
        print(f'기존 데이터 검증 완료: {result["counts"]}')
        return result
    if not 0 < dev_fraction < 1:
        raise ValueError('dev_fraction은 0과 1 사이여야 합니다.')
    source = ROOT / 'data_object_image_2/training/image_2'
    labels = ROOT / 'data_object_label_2/training/label_2'
    evaluation = read_ids(ROOT / 'eval_val.txt', 1000)
    all_ids = sorted(p.stem for p in source.glob('*.png'))
    if not set(evaluation) <= set(all_ids):
        raise ValueError('평가 이미지 일부가 없습니다.')
    if any(not (labels / f'{i}.txt').is_file() for i in all_ids):
        raise ValueError('정답 라벨 일부가 없습니다.')
    candidates = sorted(set(all_ids) - set(evaluation))
    random.Random(seed).shuffle(candidates)
    n_dev = max(1, round(len(candidates) * dev_fraction))
    splits = {'dev': sorted(candidates[:n_dev]), 'train': sorted(candidates[n_dev:]), 'eval': evaluation}
    if not splits['train']:
        raise ValueError('학습 이미지가 없습니다.')
    (output / 'images').mkdir(parents=True)
    (output / 'labels').mkdir()
    omitted, hashes = Counter(), {}
    # 하드링크는 원본 이미지 파일 내용을 공유하므로 원본/생성 이미지를 수정하지 마세요.
    # 다른 디스크 등 하드링크가 불가능한 경우에만 복사합니다.
    for image_id in all_ids:
        image = source / f'{image_id}.png'
        with Image.open(image) as im:
            width, height = im.size
        converted, counts = convert_label((labels / f'{image_id}.txt').read_text(), width, height)
        target = output / 'images' / image.name
        try:
            target.hardlink_to(image)
        except OSError:
            shutil.copy2(image, target)
        target_label = output / 'labels' / f'{image_id}.txt'
        target_label.write_text(converted, encoding='utf-8')
        hashes[target_label.relative_to(output).as_posix()] = sha256(target_label)
        omitted.update(counts)
    for split, ids in splits.items():
        for filename, text in [(f'{split}_ids.txt', '\n'.join(ids)),
                               (f'{split}.txt', '\n'.join((output / 'images' / f'{i}.png').as_posix() for i in ids))]:
            (output / filename).write_text(text + '\n', encoding='utf-8')
            hashes[filename] = sha256(output / filename)
    config = dict(path=output.as_posix(), train='train.txt', val='dev.txt', names=dict(enumerate(CLASSES)))
    # val은 대회 평가 1,000장이 아닌 내부 dev입니다. test도 지정하지 않습니다.
    (output / 'dataset.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    hashes['dataset.yaml'] = sha256(output / 'dataset.yaml')
    manifest = dict(seed=seed, dev_fraction=dev_fraction, counts={k: len(v) for k, v in splits.items()},
                    names=list(CLASSES), eval_sha256=sha256(ROOT / 'eval_val.txt'),
                    file_hashes=hashes, omitted_annotations=dict(omitted),
                    label_policy='3 classes, all difficulties; other labels omitted; standard YOLO background loss (no ignore mask)')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    validate_dataset(output)
    print(f'KITTI 변환 완료: {manifest["counts"]}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev-fraction', type=float, default=.1)
    args = parser.parse_args()
    prepare(args.output, args.seed, args.dev_fraction)
