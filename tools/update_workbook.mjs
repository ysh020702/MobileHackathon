// 기존 엑셀의 입력 칸만 채웁니다. 평균 AP40/Drop Rate/FPS 수식과 예시 행은 보존합니다.
// 실행: node tools/update_workbook.mjs setup|record <xlsx> <입력 JSON>
import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const [mode, filename, payloadFile] = process.argv.slice(2);
if (!['setup', 'record'].includes(mode) || !payloadFile) throw Error('setup|record workbook.xlsx payload.json');
const payload = JSON.parse(await fs.readFile(payloadFile, 'utf8'));
const report = mode === 'record' ? payload : null;
const condition = report ? report.conditions : payload.conditions;
// 동시에 같은 파일을 수정하지 않도록 잠급니다. 실패해도 finally에서 해제합니다.
const lockPath = filename + '.lock';
const lock = await fs.open(lockPath, 'wx');
try {
  const original = await fs.readFile(filename);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(filename));
  const log = workbook.worksheets.getItem('실험로그');
  const conditions = workbook.worksheets.getItem('측정조건');
  const summary = workbook.worksheets.getItem('보고서요약');
  // 원본 LibreOffice 수식의 한글 시트 참조를 Excel 표준 따옴표 형식으로 정규화합니다.
  // 자동 기록이 영향을 주는 계산 칸만 대상으로 하며 수식의 계산 의미는 유지합니다.
  for (const [sheet, range] of [[log,'R5:S64'],[log,'W5:W64'],[summary,'B8:D18']]) {
    const formulas = sheet.getRange(range).formulas;
    sheet.getRange(range).formulas = formulas.map(line => line.map(f => f ?
      '=' + f.replace(/^=/,'').replace(/(?<!')실험로그!/g,"'실험로그'!") : null));
  }
  // 측정 PC/추론 조건이 달라지면 기존 결과와 같은 표에 몰래 섞지 않습니다.
  const signatureKeys = ['host','platform','processor','torch','ultralytics','imgsz','device','threads',
                         'conf','iou','max_det','warmup','benchmark_repetitions'];
  const signature = Object.fromEntries(signatureKeys.map(k => [k, condition[k]]));
  const signaturePath = filename + '.conditions.json';
  let savedSignature = null;
  try { savedSignature = JSON.parse(await fs.readFile(signaturePath, 'utf8')); }
  catch (e) { if (e.code !== 'ENOENT') throw e; }
  if (savedSignature && JSON.stringify(savedSignature) !== JSON.stringify(signature)) {
    throw Error('측정조건이 기존 엑셀과 다릅니다. 조건을 복구하거나 별도 워크북을 사용하세요.');
  }
  const set = (sheet, cell, value) => {
    const literal = typeof value === 'string' && value.startsWith('=') ? "'" + value : value;
    sheet.getRange(cell).values = [[literal ?? null]];
  };
  let row = null;
  if (report) {
    if (!report.full_fixed_set || report.conditions.count !== 1000 || report.experiment?.smoke_test) {
      throw Error('부분 평가/smoke test는 실제 실험로그에 넣을 수 없습니다.');
    }
    for (const name of ['Car','Pedestrian','Cyclist']) {
      const score = report.per_class[name]?.ap40;
      if (!Number.isFinite(score) || score < 0 || score > 100) throw Error('AP40 데이터가 불완전합니다.');
    }
    const ids = log.getRange('A5:A64').values.flat();
    if (ids.includes(report.name)) {
      console.log(`이미 기록된 실험: ${report.name}. 기존 행을 보존합니다.`);
      process.exitCode = 0;
    } else {
      row = ids.findIndex(v => v === null || v === '') + 5;
      if (row < 5) throw Error('실험로그 60행이 찼습니다. 결과 JSON은 보존됩니다. 표를 확장하세요.');
      const e = report.experiment ?? {};
      const baseline = e.baseline || '';
      if (baseline && !ids.includes(baseline)) throw Error(`기준 실험 ID가 표에 없습니다: ${baseline}`);
      // 사람이 적는 해석/다음 실험은 빈칸으로 남깁니다. 실행 조건만 사실대로 입력합니다.
      const cells = {A:report.name, B:new Date(report.created_at), C:e.author || '',
        D:report.reason, E:baseline, F:report.hypothesis, G:e.architecture ? path.basename(e.architecture) : 'KITTI YOLO',
        H:condition.imgsz, I:condition.imgsz, J:condition.precision, K:e.completed_epochs,
        L:report.model.parameters/1e6, M:report.model.flops_g, N:report.model.file_size_mb,
        O:report.per_class.Car.ap40, P:report.per_class.Pedestrian.ap40, Q:report.per_class.Cyclist.ap40,
        T:report.latency_ms.median, U:report.latency_ms.p95, V:report.e2e_ms?.mean,
        X:report.peak_rss_mb, Y:condition.host, Z:'CPU RSS(전체 프로세스), PT 파일크기; 주최 채점기 대조 전', AB:'완료'};
      for (const [col, value] of Object.entries(cells)) set(log, `${col}${row}`, value);
      log.getRange(`B${row}`).setNumberFormat('yyyy-mm-dd');
      log.getRange(`L${row}:Q${row}`).setNumberFormat('0.00');
      log.getRange(`X${row}`).setNumberFormat('0');
      // 불완전한 입력을 평균해서 그럴듯한 숫자로 만들지 않도록 3개 점수가 있을 때만 계산합니다.
      log.getRange(`R${row}`).formulas = [[`=IF(COUNT(O${row}:Q${row})=3,AVERAGE(O${row}:Q${row}),"")`]];
      log.getRange(`S${row}`).formulas = [[`=IF(E${row}="","",IFERROR((INDEX($R$5:$R$64,MATCH(E${row},$A$5:$A$64,0))-R${row})/INDEX($R$5:$R$64,MATCH(E${row},$A$5:$A$64,0))*100,""))`]];
      log.getRange(`W${row}`).formulas = [[`=IFERROR(1000/T${row},"")`]];
      if (summary.getRange('B4').values[0][0] === 'exp001' && log.getRange('AB5').values[0][0] === '예시') {
        set(summary, 'B4', report.name);
      } else if (baseline) {
        set(summary, 'B4', baseline); set(summary, 'B5', report.name);
      }
    }
  }
  if (mode === 'setup') {
    const values = {B6:'OK', B7:payload.dataset_counts.train,
      B8:'KITTI 원본 image_2 / label_2', B12:`${condition.imgsz} x ${condition.imgsz}`,
      B13:'letterbox (rect=False)', B14:114, B15:'RGB / 255 (0~1)', B18:1,
      B19:condition.conf, B20:condition.iou, B21:condition.max_det, B24:condition.warmup,
      B25:condition.benchmark_repetitions, B26:condition.device === 'cpu' ? 'CPU 동기 실행' : 'torch.cuda.synchronize()',
      B30:condition.host, B31:condition.processor, B32:condition.gpu, B33:`${condition.ram_mb.toFixed(0)} MB`,
      B34:condition.platform, B35:condition.cuda || '없음', B36:condition.torch, B39:42,
      C7:`내부 검증 ${payload.dataset_counts.dev}장 별도; 고정 평가 1,000장 제외`,
      C25:`고정 목록 앞 ${condition.benchmark_repetitions}장 순환; 전처리+추론+NMS 포함`,
      C27:'E2E는 디스크/디코딩 포함 평균, Memory는 CPU RSS(decimal MB)',
      C39:'학습 seed=42, 평가 seed=0', C42:'파일 목록·해시는 실험 폴더에 저장; Git 커밋은 별도'};
    for (const [cell, value] of Object.entries(values)) set(conditions,cell,value);
    const usage = workbook.worksheets.getItem('사용법');
    set(usage,'B6','평가 완료 시 자동 입력. 결과 해석·다음 실험은 사람이 보완. 회색 칸은 수식 유지.');
    set(usage,'B14','전체 프로세스 CPU RSS 피크 (decimal MB). GPU VRAM과 다름.');
    set(usage,'B26','KITTI 2D AP40 구현을 사용. 주최 측 채점기 최종 대조 전. model.val() mAP와 다름.');
  }
  // 같은 실험 재시도는 파일을 다시 쓰지 않아 수동 입력도 보존합니다.
  if (mode === 'setup' || row !== null) {
    const backup = path.join(path.dirname(filename), 'workbook_backups');
    await fs.mkdir(backup,{recursive:true});
    await fs.writeFile(path.join(backup, `${path.basename(filename,'.xlsx')}_${Date.now()}.xlsx`), original);
    const errors = await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',
      options:{useRegex:true,maxResults:20},summary:'formula check'});
    if (errors.ndjson.split('\n').some(line => line && JSON.parse(line).kind === 'match')) {
      throw Error('엑셀 수식 오류가 있습니다. 원본은 보존했습니다: ' + errors.ndjson);
    }
    console.log(errors.ndjson);
    const temporary = filename.replace(/\.xlsx$/i, '.pending.xlsx');
    await (await SpreadsheetFile.exportXlsx(workbook)).save(temporary);
    // 쓰기 완료 후 교체. Excel이 열려 있어 교체 실패하면 원본은 그대로 남습니다.
    await fs.rename(temporary,filename);
    await fs.writeFile(signaturePath,JSON.stringify(signature,null,2));
    const previewDir = path.join(path.dirname(filename),'outputs/workbook_preview');
    await fs.mkdir(previewDir,{recursive:true});
    const preview = await workbook.render({sheetName: mode === 'setup' ? '측정조건' : '실험로그',
      range: mode === 'setup' ? 'A4:C27' : `L4:AB${Math.max(row,7)}`,scale:1.5});
    await fs.writeFile(path.join(previewDir,mode+'.png'),new Uint8Array(await preview.arrayBuffer()));
    console.log(row ? `엑셀 기록 완료: ${report.name}, ${row}행` : '엑셀 측정조건 연결 완료');
  }
} finally {
  await lock.close();
  await fs.unlink(lockPath);
}
