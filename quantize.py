"""완료된 KITTI 실험을 CPU용 ONNX FP32 / INT8 QDQ로 변환합니다."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil

from project_config import ROOT, DATA, CLASSES, read_ids
from prepare_kitti import validate_dataset, sha256
import cv2
import numpy as np
import torch
import yaml


def calibration_ids(train_ids, eval_ids, count, seed):
    if len(train_ids) != len(set(train_ids)) or set(train_ids) & set(eval_ids):
        raise ValueError('DATA LEAKAGE / duplicate: 학습·보정 후보에 고정 eval이 포함됩니다.')
    if count < 1 or count > len(train_ids):
        raise ValueError('보정 개수는 1 이상이며 train 개수를 넘을 수 없습니다.')
    return random.Random(seed).sample(sorted(train_ids), count)


def export_pair(experiment, name, settings_path, count=128, seed=42):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    from deployment import preprocess

    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not re.fullmatch(r'[A-Za-z0-9_-]+', experiment):
        raise ValueError('실험과 변환 이름은 영문/숫자/_/-만 허용합니다.')
    out = ROOT / 'quantizations' / name
    if out.exists():
        raise FileExistsError(f'변환 이름을 새로 지정하세요: {out}')
    source = ROOT / 'experiments' / experiment
    metadata = json.loads((source / 'experiment.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'trained':
        raise ValueError('학습이 완료된 실험만 양자화합니다.')
    manifest = validate_dataset(DATA)
    if metadata['eval_sha256'] != sha256(ROOT / 'eval_val.txt'):
        raise ValueError('고정 평가 목록이 학습 당시와 다릅니다.')
    trained_ids = read_ids(source / 'train_ids.txt')
    if set(trained_ids) != set(read_ids(DATA / 'train_ids.txt')):
        raise ValueError('현재 데이터와 실험의 train 분할이 다릅니다.')
    if sha256(source / 'dataset_manifest.json') != sha256(DATA / 'manifest.json'):
        raise ValueError('학습 시점과 데이터 manifest가 다릅니다.')
    selected = calibration_ids(trained_ids, read_ids(ROOT / 'eval_val.txt', 1000), count, seed)
    settings = yaml.safe_load(Path(settings_path).read_text(encoding='utf-8'))
    if settings['device'] != 'cpu' or settings['threads'] < 1:
        raise ValueError('ONNX INT8 비교는 CPU와 양의 threads 설정을 사용합니다.')
    checkpoint = Path(metadata['checkpoint'])
    if sha256(checkpoint) != metadata['checkpoint_sha256']:
        raise ValueError('학습 체크포인트 해시가 다릅니다.')
    out.mkdir(parents=True)
    bundle = dict(status='exporting', name=name, created_at=datetime.now(timezone.utc).isoformat(),
                  source_experiment=experiment, source_checkpoint_sha256=sha256(checkpoint),
                  experiment=metadata, settings=settings, eval_sha256=metadata['eval_sha256'],
                  calibration=dict(ids=selected, count=len(selected), seed=seed, split='train',
                                   dataset_manifest_sha256=sha256(DATA / 'manifest.json')),
                  onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                  method='static QDQ S8S8 per-channel Conv/MatMul; remaining operators FP32')
    def save():
        (out / 'bundle.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    try:
        torch.set_num_threads(settings['threads'])
        copy = out / 'source.pt'
        shutil.copy2(checkpoint, copy)
        yolo = YOLO(str(copy), task='detect')
        if tuple(yolo.names[i] for i in range(3)) != CLASSES:
            raise ValueError('클래스 순서는 Car/Pedestrian/Cyclist여야 합니다.')
        bundle['model'] = dict(parameters=sum(p.numel() for p in yolo.model.parameters()),
                              flops_g=get_flops(deepcopy(yolo.model), imgsz=settings['imgsz']),
                              complexity_scope='source FP32 architecture MACs*2; INT8 bit-operations not estimated')
        exported = yolo.export(format='onnx', imgsz=settings['imgsz'], batch=1, dynamic=False,
                               simplify=False, opset=17, device='cpu', half=False, nms=False)
        fp32 = out / 'fp32.onnx'
        quant_pre_process(str(exported), str(fp32), skip_optimization=True, skip_symbolic_shape=True)
        input_name = onnx.load(str(fp32)).graph.input[0].name
        # 양자화 오차와 ONNX 변환 오류를 구분합니다. eval이 아닌 보정 train 이미지로 비교합니다.
        probe = cv2.imread(str(DATA/'images'/f'{selected[0]}.png'))
        if probe is None:
            raise ValueError('ONNX 변환 검증 이미지 읽기 실패')
        tensor = preprocess(probe, settings['imgsz'])
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings['threads']
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(fp32),sess_options=options,providers=['CPUExecutionProvider'])
        with torch.no_grad():
            reference = yolo.model.cpu().float().eval()(torch.from_numpy(tensor))[0].numpy()
        actual = session.run(None,{input_name:tensor})[0]
        if reference.shape != actual.shape or not np.allclose(reference,actual,rtol=1e-3,atol=1e-3):
            raise ValueError('PyTorch→ONNX FP32 출력 검증 실패: 양자화를 중단합니다.')
        bundle['export_check'] = dict(image_id=selected[0], max_absolute_error=float(np.max(np.abs(reference-actual))),
                                      rtol=1e-3, atol=1e-3)
        del session
        image_hashes = {}
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.ids = iter(selected)
            def get_next(self):
                image_id = next(self.ids, None)
                if image_id is None:
                    return None
                path = DATA / 'images' / f'{image_id}.png'
                image_hashes[image_id] = sha256(path)
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'보정 이미지 읽기 실패: {path}')
                return {input_name: preprocess(image, settings['imgsz'])}
        int8 = out / 'int8.onnx'
        quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        per_channel=True, op_types_to_quantize=['Conv', 'MatMul'],
                        calibration_providers=['CPUExecutionProvider'],
                        extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True})
        graph = onnx.load(str(int8))
        onnx.checker.check_model(graph)
        # scalar/vector zero-point도 INT8이므로 Conv/MatMul 가중치(2차원 이상)만 셉니다.
        quantized_weights = sum(x.data_type == onnx.TensorProto.INT8 and len(x.dims) >= 2 for x in graph.graph.initializer)
        qdq_nodes = sum(x.op_type in {'QuantizeLinear', 'DequantizeLinear'} for x in graph.graph.node)
        if quantized_weights == 0 or qdq_nodes == 0:
            raise RuntimeError('실제 INT8 가중치/QDQ가 생성되지 않았습니다.')
        bundle['calibration']['image_sha256'] = image_hashes
        bundle.update(status='ready', quantized_weight_tensors=quantized_weights, qdq_nodes=qdq_nodes,
                      artifacts={kind:dict(filename=path.name, sha256=sha256(path), file_size_mb=path.stat().st_size/1e6)
                                 for kind, path in [('fp32', fp32), ('int8', int8)]})
        save()
        return out / 'bundle.json'
    except BaseException as error:
        bundle.update(status='failed', error=str(error) or type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--settings', type=Path, default=ROOT/'configs/evaluation.yaml')
    parser.add_argument('--calibration-count', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    print(export_pair(a.experiment, a.name, a.settings, a.calibration_count, a.seed))
