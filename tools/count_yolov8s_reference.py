"""Recompute Conv2d operation counts for the Small teaching reference (no training)."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'yolov8s_reference'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics_reference'))
import torch
import ultralytics
from ultralytics import YOLO
import yaml


def shape(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (tuple, list)):
        return [shape(v) for v in x]
    if isinstance(x, dict):
        return {k: shape(v) for k, v in x.items()}


def main():
    if (OUT / 'operations.json').exists():
        raise SystemExit('Reference already exists; preserve it before recomputing.')
    path = ROOT / 'configs/models/yolov8s_baseline.yaml'
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = YOLO(str(path), task='detect').model.cpu().float().eval()
    spec = yaml.safe_load(path.read_text(encoding='utf-8'))
    layers, convs, handles = [], [], []
    for i, block in enumerate(model.model):
        row = dict(index=i, module=type(block).__name__, source=block.f,
                   repeats=len(block.m) if type(block).__name__ == 'C2f' else 1,
                   conv_flops=0)
        layers.append(row)
        def outer(m, ins, out, row=row):
            row.update(input_shapes=shape(ins), output_shapes=shape(out))
        handles.append(block.register_forward_hook(outer))
        for name, m in block.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                def inner(m, ins, out, i=i, name=name, row=row):
                    flops = 2 * out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
                    row['conv_flops'] += flops
                    convs.append(dict(block=i, name=name, in_channels=m.in_channels,
                                      out_channels=m.out_channels, kernel=list(m.kernel_size),
                                      groups=m.groups, input_shapes=shape(ins),
                                      output_shapes=shape(out), conv_flops=flops))
                handles.append(m.register_forward_hook(inner))
    with torch.inference_mode():
        result = model(torch.zeros(1, 3, 640, 640))
    for h in handles:
        h.remove()
    total = sum(r['conv_flops'] for r in layers)
    assert total == sum(r['conv_flops'] for r in convs)
    assert len(layers) == 23 and len(convs) == 64
    assert list(result[0].shape) == [1, 7, 8400]
    assert [layers[i]['repeats'] for i in [2,4,6,8,12,15,18,21]] == [1,2,2,1,1,1,1,1]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'baseline.yaml').write_bytes(path.read_bytes())
    data = dict(model=str(path.relative_to(ROOT)), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                torch=torch.__version__, ultralytics=ultralytics.__version__, input_shape=[1,3,640,640],
                scale=spec['scales']['s'], method='Conv2d multiply/add counts: MAC x 2; structural calculation on actual tensor shapes',
                excludes=['bias','BatchNorm','activation','pooling','resize','concat','softmax','box coordinate arithmetic','NMS'],
                note='Untrained YAML model, no latency or accuracy benchmark. Zero input is sufficient for this fixed-shape Conv count.',
                total_conv_flops=total, layers=layers, convolutions=convs)
    (OUT/'operations.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'layers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(layers[0]))
        w.writeheader()
        w.writerows(layers)
    print(json.dumps(dict(blocks=len(layers),convs=len(convs),conv_tflop=total/1e12,conv_gflop=total/1e9,output=list(result[0].shape))))


if __name__ == '__main__':
    main()
