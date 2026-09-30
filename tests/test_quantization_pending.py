"""Excel replacement failures must not discard completed evaluation results."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compare_quantization import save_comparison


class PendingTests(unittest.TestCase):
    def test_pending_retry_and_unexpected_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['fp32', 'int8']:
                folder = root / name
                folder.mkdir()
                (folder / 'metrics.json').write_text('{}')
                (folder / 'predictions.json').write_text('{"image": []}')
            output = root / 'comparison.json'
            payload = {'status': 'evaluation_complete'}
            with patch('compare_quantization.read_ids', return_value=['image']), \
                 patch('compare_quantization.compare', return_value=payload), \
                 patch('compare_quantization.node_runtime', return_value=Path('node')), \
                 patch('compare_quantization.subprocess.run') as run:
                args = (root/'fp32/metrics.json', root/'int8/metrics.json', output)
                run.side_effect = subprocess.CalledProcessError(75, ['node'])
                self.assertEqual(save_comparison(*args), payload)
                self.assertEqual(json.loads(output.read_text()), payload)
                self.assertTrue((root/'excel_pending.txt').exists())
                run.side_effect = None
                save_comparison(*args)
                self.assertFalse((root/'excel_pending.txt').exists())
                run.side_effect = subprocess.CalledProcessError(1, ['node'])
                with self.assertRaises(RuntimeError):
                    save_comparison(*args)
                self.assertTrue((root/'excel_pending.txt').exists())
