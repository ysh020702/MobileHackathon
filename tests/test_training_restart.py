"""재시작 시 기존 체크포인트 보존과 완료/실행 중 실험 보호를 검증합니다."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train import check_restart, archive_for_restart


class RestartTests(unittest.TestCase):
    def test_failed_restart_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, patch('train.ROOT', Path(temp)):
            out = Path(temp) / 'experiments' / 'test'
            out.mkdir(parents=True)
            (out / 'experiment.json').write_text(json.dumps({'status':'failed'}))
            (out / 'last.pt').write_bytes(b'checkpoint')
            with self.assertRaises(FileExistsError):
                check_restart(out)
            check_restart(out, True)
            archive_for_restart(out)
            self.assertFalse(out.exists())
            archived = list((Path(temp) / 'archive/restarted_experiments').iterdir())
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0] / 'last.pt').read_bytes(), b'checkpoint')

    def test_completed_running_and_evaluated_experiments_are_protected(self):
        with tempfile.TemporaryDirectory() as temp, patch('train.ROOT', Path(temp)):
            out = Path(temp) / 'experiments' / 'test'
            out.mkdir(parents=True)
            for status in ['trained', 'training']:
                (out / 'experiment.json').write_text(json.dumps({'status':status}))
                with self.assertRaises(ValueError):
                    check_restart(out, True)
                self.assertTrue(out.exists())
            (out / 'experiment.json').write_text(json.dumps({'status':'interrupted'}))
            check_restart(out, True)
            (Path(temp) / 'evaluations' / 'test').mkdir(parents=True)
            with self.assertRaises(ValueError):
                check_restart(out, True)
