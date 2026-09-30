"""원본 대신 임시 복사본으로 엑셀 기록·수식·중복 방지를 검사합니다."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from project_config import ROOT, WORKBOOK
from record_experiment import node_runtime


class WorkbookTests(unittest.TestCase):
    def test_append_formula_duplicate_and_condition_guard(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temporary:
            directory = Path(temporary)
            book = directory / 'test.xlsx'
            shutil.copy2(WORKBOOK, book)
            signature = Path(str(WORKBOOK) + '.conditions.json')
            shutil.copy2(signature, Path(str(book) + '.conditions.json'))
            conditions = json.loads(signature.read_text(encoding='utf-8'))
            conditions.update(count=1000, precision='FP32')
            report = dict(name='test_fixture_only', created_at='2026-09-29T00:00:00+00:00',
                full_fixed_set=True, conditions=conditions, experiment={}, reason='자동입력 테스트', hypothesis='수식 검증',
                per_class={name:dict(ap40=score) for name, score in [('Car',60),('Pedestrian',40),('Cyclist',20)]},
                model=dict(parameters=3000000,flops_g=8,file_size_mb=6),
                latency_ms=dict(median=20,p95=25),e2e_ms=dict(mean=30),peak_rss_mb=400)
            payload = directory / 'fixture.json'
            def run():
                payload.write_text(json.dumps(report), encoding='utf-8')
                return subprocess.run([str(node_runtime()), str(ROOT/'tools/update_workbook.mjs'),
                                       'record',str(book),str(payload)], cwd=ROOT, capture_output=True)
            result = run()
            self.assertEqual(result.returncode,0,result.stderr.decode('utf-8',errors='replace'))
            original_bytes = book.read_bytes()
            self.assertEqual(run().returncode,0)
            self.assertEqual(original_bytes,book.read_bytes())
            # 기존 실험 행이 늘어났더라도 테스트 ID의 행을 찾아 검사합니다.
            ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            with zipfile.ZipFile(book) as archive:
                xml=ET.fromstring(archive.read('xl/worksheets/sheet3.xml'))
                shared=[]
                if 'xl/sharedStrings.xml' in archive.namelist():
                    shared=[''.join(x.itertext()) for x in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
                row=None
                for cell in xml.findall('.//s:c',ns):
                    value=cell.find('s:v',ns)
                    text=value.text if value is not None else ''
                    if cell.get('t') == 's':
                        text=shared[int(text)]
                    elif cell.get('t') == 'inlineStr':
                        text=''.join(cell.find('s:is',ns).itertext())
                    if text == 'test_fixture_only':
                        row=cell.get('r')[1:]
                self.assertIsNotNone(row)
                self.assertEqual(float(xml.find(f'.//s:c[@r="R{row}"]/s:v',ns).text),40)
                self.assertEqual(float(xml.find(f'.//s:c[@r="W{row}"]/s:v',ns).text),50)
            report['name']='wrong_pc_fixture'
            report['conditions']['host']='another_pc'
            self.assertNotEqual(run().returncode,0)
            self.assertEqual(original_bytes,book.read_bytes())


if __name__ == '__main__':
    unittest.main()
