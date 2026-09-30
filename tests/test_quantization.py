from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import shutil
import xml.etree.ElementTree as ET
import zipfile

from project_config import ROOT, WORKBOOK
from quantize import calibration_ids
from compare_quantization import compare
from record_experiment import node_runtime


def fixture():
    c=json.loads(Path(str(WORKBOOK)+'.conditions.json').read_text(encoding='utf-8'))
    c.update(precision='FP32',count=1000,backend='onnxruntime_cpu',onnxruntime='test',data_sha256='test_data')
    a=dict(name='test_fp32',created_at='2026-09-29T00:00:00+00:00',reason='TEST ONLY',hypothesis='TEST ONLY',
           full_fixed_set=True,conditions=c,experiment=dict(smoke_test=False,architecture='test.yaml',completed_epochs=1),
           mean_ap40=40,per_class={c:{'ap40':v} for c,v in zip(['Car','Pedestrian','Cyclist'],[60,40,20])},
           model=dict(parameters=3000000,flops_g=8,file_size_mb=12),peak_rss_mb=400,
           latency_ms=dict(median=20,p95=25),e2e_ms=dict(mean=30),
           quantization=dict(variant='fp32',bundle_sha256='test_bundle',source_checkpoint_sha256='test_checkpoint',
                             bundle_name='TEST_PAIR',calibration=dict(ids=['007480'])))
    # Pick a genuine train-only id for the leakage guard.
    a['quantization']['calibration']['ids']=(ROOT/'datasets/kitti3/train_ids.txt').read_text().split()[:1]
    b=deepcopy(a);b['name']='test_int8';b['conditions']['precision']='INT8_QDQ';b['quantization']['variant']='int8'
    b['mean_ap40']=38
    b['per_class']={c:{'ap40':v} for c,v in zip(['Car','Pedestrian','Cyclist'],[57,38,19])}
    b['latency_ms']['median']=10;b['peak_rss_mb']=300;b['model']['file_size_mb']=3
    return a,b


class QuantizationTests(unittest.TestCase):
    def test_calibration_leakage_and_determinism(self):
        with self.assertRaises(ValueError): calibration_ids(['a','b'],['b'],1,42)
        with self.assertRaises(ValueError): calibration_ids(['a','a'],[],1,42)
        with self.assertRaises(ValueError): calibration_ids(['a'],[],2,42)
        self.assertEqual(calibration_ids(['a','b','c'],['z'],2,42),calibration_ids(['c','a','b'],['z'],2,42))

    def test_comparison_math_and_guards(self):
        a,b=fixture();r=compare(a,b)
        self.assertEqual(r['metrics']['mean_ap40']['retention_percent'],95)
        self.assertEqual(r['metrics']['mean_ap40']['reduction_percent'],5)
        self.assertEqual(r['metrics']['latency_p50_ms']['reduction_percent'],50)
        for mutation in ['subset','checkpoint','conditions','smoke','leakage']:
            broken=deepcopy(b)
            if mutation=='subset': broken['full_fixed_set']=False
            if mutation=='checkpoint': broken['quantization']['source_checkpoint_sha256']='other'
            if mutation=='conditions': broken['conditions']['threads']=999
            if mutation=='smoke': broken['experiment']['smoke_test']=True
            if mutation=='leakage': broken['quantization']['calibration']['ids']=(ROOT/'eval_val.txt').read_text().split()[:1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): compare(a,broken)
        a['mean_ap40']=0
        for v in a['per_class'].values(): v['ap40']=0
        self.assertIsNone(compare(a,b)['metrics']['mean_ap40']['retention_percent'])

    def test_workbook_pair_formulas_idempotency_and_preservation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            root=Path(temp);book=root/'test.xlsx';payload=root/'comparison.json'
            shutil.copy2(WORKBOOK,book)
            shutil.copy2(str(WORKBOOK)+'.conditions.json',str(book)+'.conditions.json')
            a,b=fixture();payload.write_text(json.dumps(compare(a,b)),encoding='utf-8')
            cmd=[str(node_runtime()),str(ROOT/'tools/update_quantization.mjs'),str(book),str(payload)]
            result=subprocess.run(cmd,cwd=ROOT,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
            saved=book.read_bytes()
            self.assertEqual(subprocess.run(cmd,cwd=ROOT,capture_output=True).returncode,0)
            self.assertEqual(saved,book.read_bytes())
            ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            with zipfile.ZipFile(book) as z:
                worksheets=[ET.fromstring(z.read(n)) for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')]
                matches=[s for s in worksheets if (c:=s.find('.//s:c[@r="F5"]/s:f',ns)) is not None and 'E5/D5' in c.text]
                self.assertEqual(len(matches),1)
                sheet=matches[0]
                shared=[]
                if 'xl/sharedStrings.xml' in z.namelist():
                    shared=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
                def cell_text(cell):
                    value=cell.find('s:v',ns)
                    text=value.text if value is not None else ''
                    if cell.get('t')=='s': return shared[int(text)]
                    if cell.get('t')=='inlineStr': return ''.join(cell.find('s:is',ns).itertext())
                    return text
                row=next(c.get('r')[1:] for c in sheet.findall('.//s:c',ns)
                         if c.get('r').startswith('A') and cell_text(c)=='TEST_PAIR')
                for col,expected in [('F',95),('G',5),('M',50)]:
                    self.assertEqual(float(sheet.find(f'.//s:c[@r="{col}{row}"]/s:v',ns).text),expected)
                summary=next(s for s in worksheets
                             if (c:=s.find('.//s:c[@r="B4"]',ns)) is not None and cell_text(c)=='test_fp32')
                for cell,expected in [('B8',40),('C8',38),('B15',20),('C15',10)]:
                    c=summary.find(f'.//s:c[@r="{cell}"]',ns)
                    self.assertIn("'실험로그'!",c.find('s:f',ns).text)
                    self.assertAlmostEqual(float(c.find('s:v',ns).text),expected)
                self.assertAlmostEqual(float(summary.find('.//s:c[@r="D15"]/s:v',ns).text),-50)
            broken=compare(a,b);broken['int8']['conditions']['host']='wrong'
            payload.write_text(json.dumps(broken),encoding='utf-8')
            self.assertNotEqual(subprocess.run(cmd,cwd=ROOT,capture_output=True).returncode,0)
            self.assertEqual(saved,book.read_bytes())
