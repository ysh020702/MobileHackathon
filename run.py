"""KITTI 학습 모델로 한 장 추론. 학습/평가는 별도 파일에서 수행합니다."""
import argparse
from pathlib import Path
from project_config import ROOT
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # COCO 가중치를 자동 다운로드하지 않습니다. 학습 완료 PT를 지정하세요.
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, default=ROOT / 'data_object_image_2/training/image_2/000000.png')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/demo/output.jpg')
    args = parser.parse_args()
    model = YOLO(str(args.checkpoint), task='detect')
    if set(model.names.values()) != {'Car', 'Pedestrian', 'Cyclist'}:
        raise ValueError('KITTI 3클래스 모델을 지정하세요.')
    result = model.predict(str(args.image), imgsz=640, conf=.25, device='cpu')[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(args.output))
    print(f'결과: {args.output.resolve()}')


if __name__ == '__main__':
    main()
