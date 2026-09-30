"""FP32와 INT8에 동일한 전처리/NMS 및 CPU ONNX Runtime 조건을 적용합니다."""
import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes


def preprocess(image, size):
    padded = LetterBox((size, size), auto=False, stride=32)(image=image)
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


class OnnxDetector:
    def __init__(self, path, settings):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings['threads']
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.settings = settings

    def predict_rows(self, image):
        s = self.settings
        tensor = preprocess(image, s['imgsz'])
        output = self.session.run(None, {self.input: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 7:
            raise ValueError(f'YOLOv8 3-class raw output [1,7,N] required: {output.shape}')
        if not np.isfinite(output).all():
            raise ValueError('추론 출력에 NaN/Inf가 있습니다.')
        boxes = non_max_suppression(torch.from_numpy(output), conf_thres=s['conf'],
                                    iou_thres=s['iou'], nc=3, max_det=s['max_det'])[0]
        if len(boxes):
            boxes[:, :4] = scale_boxes((s['imgsz'], s['imgsz']), boxes[:, :4], image.shape)
        return boxes.tolist()
