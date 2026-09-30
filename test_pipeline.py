"""변환 정확성과 대회 평가 이미지의 학습 유입 방지를 검증합니다."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from prepare_kitti import convert_label, validate_dataset, sha256


class PipelineTests(unittest.TestCase):
    def test_label_conversion(self):
        text = ('Car 0 0 0 20 10 60 50 0 0 0 0 0 0 0\n'
                'Cyclist 0 0 0 -10 0 110 100 0 0 0 0 0 0 0\n'
                'DontCare -1 -1 -10 0 0 100 100 -1 -1 -1 -1 -1 -1 -10')
        result, omitted = convert_label(text, 100, 100)
        self.assertEqual(result.splitlines()[0], '0 0.40000000 0.30000000 0.40000000 0.40000000')
        self.assertEqual(result.splitlines()[1], '2 0.50000000 0.50000000 1.00000000 1.00000000')
        self.assertEqual(omitted['DontCare'], 1)

    def test_split_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'eval_val.txt').write_text('\n'.join(f'{i:06d}' for i in range(1000)))
            data = root / 'data'
            (data / 'images').mkdir(parents=True)
            for split, image_id in [('train','001000'),('dev','001001')]:
                (data / f'{split}_ids.txt').write_text(image_id)
                image = data / 'images' / f'{image_id}.png'
                image.write_bytes(b'test')
                (data / f'{split}.txt').write_text(str(image))
            manifest = dict(eval_sha256=sha256(root/'eval_val.txt'), file_hashes={})
            (data/'manifest.json').write_text(json.dumps(manifest))
            with patch('prepare_kitti.ROOT', root):
                validate_dataset(data)
                (data/'train_ids.txt').write_text('000000')
                with self.assertRaisesRegex(ValueError, 'DATA LEAKAGE'):
                    validate_dataset(data)

    def test_manifest_detects_label_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'eval_val.txt').write_text('\n'.join(f'{i:06d}' for i in range(1000)))
            (root/'train_ids.txt').write_text('001000')
            (root/'dev_ids.txt').write_text('001001')
            (root/'label.txt').write_text('original')
            manifest = dict(eval_sha256=sha256(root/'eval_val.txt'), file_hashes={'label.txt':sha256(root/'label.txt')})
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'label.txt').write_text('tampered')
            with patch('prepare_kitti.ROOT', root):
                with self.assertRaisesRegex(ValueError, '변경'):
                    validate_dataset(root)


if __name__ == '__main__':
    unittest.main()
