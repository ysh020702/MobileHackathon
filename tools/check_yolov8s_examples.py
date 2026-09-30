"""Validate independent teaching examples in guide section 3.4, without training."""
import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.ultralytics_reference'))
import torch
import yaml
from ultralytics.nn.tasks import DetectionModel


def measure(spec):
    model = DetectionModel(cfg=copy.deepcopy(spec), ch=3, nc=3, verbose=False).eval()
    costs = [0] * len(model.model)
    handles = []
    shapes = {}
    for i, block in enumerate(model.model):
        def outer(m, ins, out, i=i):
            if isinstance(out, torch.Tensor):
                shapes[i] = list(out.shape)
        handles.append(block.register_forward_hook(outer))
        for m in block.modules():
            if isinstance(m, torch.nn.Conv2d):
                def hook(m, ins, out, i=i):
                    costs[i] += 2*out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                handles.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        out = model(torch.zeros(1,3,640,640))[0]
    for h in handles:
        h.remove()
    assert list(out.shape) == [1,7,8400]
    head = model.model[-1]
    return dict(total_conv_flops=sum(costs), block_conv_flops=costs, block_output_shapes=shapes,
                output_shape=list(out.shape), c2f_repeats={i:len(m.m) for i,m in enumerate(model.model) if type(m).__name__=='C2f'},
                detect_classification_width=head.cv3[0][0].conv.out_channels,
                detect_box_width=head.cv2[0][0].conv.out_channels)


def main():
    path=ROOT/'docs/yolov8s_reference/modification_examples.json'
    if path.exists():
        raise SystemExit('Refusing to overwrite existing example results.')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    spec=yaml.safe_load((ROOT/'docs/yolov8s_reference/baseline.yaml').read_text(encoding='utf-8'))
    spec['scale']='s'
    baseline=measure(spec)
    original=json.loads((ROOT/'docs/yolov8s_reference/operations.json').read_text())
    assert baseline['total_conv_flops']==original['total_conv_flops']
    results=[]
    for name,part,index,new in [
        ('A_repeat','backbone',4,[-1,3,'C2f',[256,True]]),
        ('B_backbone_width','backbone',4,[-1,6,'C2f',[192,True]]),
        ('C_neck_width','head',5,[-1,3,'C2f',[192]]),
        ('D_sppf_bypass','backbone',9,[-1,1,'nn.Identity',[]]),
    ]:
        cfg=copy.deepcopy(spec)
        before=copy.deepcopy(cfg[part][index])
        cfg[part][index]=new
        result=measure(cfg)
        result.update(name=name,yaml_section=part,index_in_section=index,before=before,after=new,
                      reduction_percent=100*(1-result['total_conv_flops']/baseline['total_conv_flops']),
                      changed_cost_blocks=[i for i,(a,b) in enumerate(zip(baseline['block_conv_flops'],result['block_conv_flops'])) if a!=b])
        results.append(result)
        print(name,round(result['total_conv_flops']/1e9,6),round(result['reduction_percent'],2),result['changed_cost_blocks'])
    path.write_text(json.dumps(dict(note='Independent changes to untrained Small; Conv2d MAC x 2 only; no accuracy or latency results.',
                                   baseline=baseline,examples=results),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
