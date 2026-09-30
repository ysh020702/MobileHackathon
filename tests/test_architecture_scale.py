"""원본/실험 사본의 실제 모델 크기가 일치하는지 검증합니다."""
from pathlib import Path
import shutil
import tempfile
import unittest

import yaml
from train import resolve_architecture_scale
from ultralytics import YOLO


class ArchitectureScaleTests(unittest.TestCase):
    def test_original_and_snapshot_models_keep_small_scale(self):
        spec = dict(nc=3, scales={'n':[1,0.25,128], 's':[1,0.5,128]},
                    backbone=[[-1,1,'Conv',[32,3,2]]], head=[[[0],1,'Detect',['nc']]])
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'yolov8s_custom.yaml'
            source.write_text(yaml.safe_dump(spec))
            snapshot = Path(temp)/'experiment'/'architecture'/source.name
            snapshot.parent.mkdir(parents=True)
            shutil.copy2(source, snapshot)
            original = YOLO(str(source), task='detect').model
            saved = YOLO(str(snapshot), task='detect').model
            self.assertEqual(resolve_architecture_scale(source,spec),'s')
            self.assertEqual(saved.yaml['scale'],'s')
            self.assertEqual(saved.model[0].conv.out_channels,16)
            self.assertEqual(sum(p.numel() for p in original.parameters()),sum(p.numel() for p in saved.parameters()))

    def test_ambiguous_conflicting_and_single_scale(self):
        scales={'n':[1,.25,128], 's':[1,.5,128]}
        with self.assertRaises(ValueError):
            resolve_architecture_scale('custom.yaml', {'scales':scales})
        with self.assertRaises(ValueError):
            resolve_architecture_scale('yolov8n_custom.yaml', {'scales':scales,'scale':'s'})
        with self.assertRaises(ValueError):
            resolve_architecture_scale('yolov8m_custom.yaml', {'scales':scales})
        self.assertEqual(resolve_architecture_scale('custom.yaml',{'scales':{'s':scales['s']}}),'s')
        self.assertIsNone(resolve_architecture_scale('custom.yaml',{'depth_multiple':1,'width_multiple':.5}))
