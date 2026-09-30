"""CPU KITTI 2D Moderate R40 evaluation (not BEV/3D).

Protocol: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
Algorithm cross-check: OpenPCDet kitti_object_eval_python/eval.py.
GT order, strict overlap thresholds and the 41-slot sampling are intentional.
"""
import numpy as np

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
MIN_OVERLAP = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
NEIGHBOR = {'Car': 'Van', 'Pedestrian': 'Person_sitting'}


def overlap_matrix(boxes, others, detection_area=False):
    a = np.asarray(boxes, dtype=float).reshape(-1, 4)
    b = np.asarray(others, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    area = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    denominator = area if detection_area else area + (b[:, 2:] - b[:, :2]).prod(1)[None, :] - intersection
    return intersection / np.maximum(denominator, 1e-12)


def prepare_frame(rows, predictions, name):
    """한 이미지에서 현재 클래스의 유효 GT/제외 GT와 DontCare 겹침을 준비합니다."""
    # Prediction rows: x1,y1,x2,y2,confidence; already filtered by class.
    detections = np.asarray(predictions, dtype=float).reshape(-1, 5)
    boxes, flags, dc = [], [], []
    for row in rows:
        cls, truncated, occluded, box = row
        if cls == 'DontCare':
            dc.append(box)
        if cls == name:
            # Reference devkit uses <=25 for GT, <25 for detections.
            flag = int(box[3] - box[1] <= 25 or occluded > 1 or truncated > 0.3)
        elif cls == NEIGHBOR.get(name):
            flag = 1
        else:
            continue
        boxes.append(box)
        flags.append(flag)
    return dict(gt=np.asarray(flags), det=detections,
                small=(detections[:, 3] - detections[:, 1] < 25),
                overlap=overlap_matrix(detections[:, :4], boxes),
                dc=overlap_matrix(detections[:, :4], dc, True))


def statistics(frame, minimum, threshold=None):
    """신뢰도 임계값 하나에서 TP(정답)/FP(오탐)/FN(놓침)을 계산합니다.

    threshold=None이면 AP 샘플링에 쓸 정답 탐지의 신뢰도를 모읍니다.
    실제 점수 계산에서는 KITTI 방식으로 GT 순서대로 최대 IoU 탐지를 선택합니다.
    """
    det, small = frame['det'], frame['small']
    used = np.zeros(len(det), dtype=bool)
    eligible = np.ones(len(det), dtype=bool) if threshold is None else det[:, 4] >= threshold
    tp = fn = 0
    scores = []
    for index, ignored in enumerate(frame['gt']):
        candidates = np.flatnonzero(~used & eligible & (frame['overlap'][:, index] > minimum))
        if not len(candidates):
            fn += int(ignored == 0)
            continue
        if threshold is None:
            selected = candidates[np.argmax(det[candidates, 4])]
        else:
            regular = candidates[~small[candidates]]
            selected = (regular[np.argmax(frame['overlap'][regular, index])]
                        if len(regular) else candidates[0])
        used[selected] = True
        if ignored == 0 and not small[selected]:
            tp += 1
            scores.append(float(det[selected, 4]))
    fp = 0
    if threshold is not None:
        remaining = ~used & eligible & ~small
        if frame['dc'].shape[1]:
            remaining &= ~np.any(frame['dc'] > minimum, axis=1)
        fp = int(remaining.sum())
    return tp, fp, fn, scores


def select_thresholds(scores, count):
    """정답 탐지 신뢰도에서 재현율 간격 1/40에 대응하는 임계값을 선택합니다."""
    thresholds, recall = [], 0.0
    ordered = sorted(scores, reverse=True)
    for i, score in enumerate(ordered):
        left = (i + 1) / count
        right = (i + 2) / count if i + 1 < len(ordered) else left
        if i + 1 < len(ordered) and right - recall < recall - left:
            continue
        thresholds.append(score)
        recall += 1 / 40
    return thresholds


def score_class(truth, predictions, name):
    """원시 PR → 오른쪽 최대값으로 precision 보간 → 0번을 뺀 40칸 평균.

    반환 AP는 엑셀 양식과 같은 0~100 단위입니다. 유효 정답 자체가 없으면 None입니다.
    """
    frames = [prepare_frame(rows, predictions.get(image_id, []), name)
              for image_id, rows in truth.items()]
    count = sum(int((frame['gt'] == 0).sum()) for frame in frames)
    minimum = MIN_OVERLAP[name]
    scores = [score for frame in frames for score in statistics(frame, minimum)[3]]
    thresholds = select_thresholds(scores, count) if count else []
    raw_precision, raw_recall, counts = [], [], []
    for threshold in thresholds:
        totals = np.sum([statistics(f, minimum, threshold)[:3] for f in frames], axis=0)
        tp, fp, fn = map(int, totals)
        counts.append(dict(tp=tp, fp=fp, fn=fn))
        raw_precision.append(tp / (tp + fp) if tp + fp else 0.0)
        raw_recall.append(tp / (tp + fn) if tp + fn else 0.0)
    precision = np.zeros(41)
    precision[:len(raw_precision)] = raw_precision
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return dict(gt=count, iou=minimum, thresholds=thresholds, counts=counts,
                raw_precision=raw_precision, raw_recall=raw_recall,
                recall_grid=np.linspace(0, 1, 41).tolist(),
                precision_envelope=envelope.tolist(),
                ap40=float(envelope[1:].mean() * 100) if count else None)
