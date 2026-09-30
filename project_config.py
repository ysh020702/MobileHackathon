"""모든 실행 파일이 공유하는 경로. 현재 터미널 위치와 무관하게 동작합니다."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
CLASSES = ('Car', 'Pedestrian', 'Cyclist')  # 이 순서가 YOLO 클래스 ID 0, 1, 2입니다.
DATA = ROOT / 'datasets/kitti3'
MODEL = ROOT / 'configs/models/yolov8s_baseline.yaml'
WORKBOOK = ROOT / 'reports/트랙2_실험기록_torch211_cu128.xlsx'


def read_ids(path, expected=None):
    ids = Path(path).read_text(encoding='utf-8-sig').split()
    if not ids or len(set(ids)) != len(ids) or any(len(i) != 6 or not i.isdigit() for i in ids):
        raise ValueError(f'중복 없는 6자리 이미지 ID 목록이 필요합니다: {path}')
    if expected is not None and len(ids) != expected:
        raise ValueError(f'{path}: {expected}개여야 하지만 {len(ids)}개입니다.')
    return ids
