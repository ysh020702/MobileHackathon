// 동일 모델의 FP32/INT8을 한 번에 저장하며 실패 시 원본 Excel을 보존합니다.
import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const [filename,payloadFile] = process.argv.slice(2);
if (!filename) throw Error('node tools/update_quantization.mjs workbook.xlsx [comparison.json]');
const payload = payloadFile ? JSON.parse(await fs.readFile(payloadFile,'utf8')) : null;
const lock = await fs.open(filename+'.lock','wx');
try {
  const original = await fs.readFile(filename);
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(filename));
  const log=wb.worksheets.getItem('실험로그');
  // Import can leave unquoted Korean sheet references as #NAME? in the
  // calculation engine. Re-enter only affected formulas with quoted names,
  // preserving literals, formatting, and the original lookup behavior.
  const summary=wb.worksheets.getItem('보고서요약');
  const summaryFormulas=summary.getRange('B8:D18').formulas;
  for (let row=0;row<summaryFormulas.length;row++) {
    for (let col=0;col<summaryFormulas[row].length;col++) {
      const formula=summaryFormulas[row][col];
      // Also invalidate imported cached deltas in column D.
      if (formula) {
        summary.getCell(row+7,col+1).formulas=[[
          '='+formula.replace(/^=/,'').replace(/(?<!')실험로그!/g,"'실험로그'!")
        ]];
      }
    }
  }
  // The template stores D8:D18 as one shared formula. Import exposes only
  // its anchor; expand the known template formula to avoid stale cached deltas.
  if (summary.getRange('D8').formulas[0][0]?.replace(/^=/,'') === 'IFERROR((C8-B8)/B8*100,"")') {
    summary.getRange('D8:D18').fillDown();
  }
  let comparison;
  try { comparison=wb.worksheets.getItem('양자화비교'); } catch {}
  if (!comparison) {
    comparison=wb.worksheets.add('양자화비교');
    comparison.getRange('A1').values=[['동일 체크포인트의 FP32 / INT8 비교']];
    comparison.getRange('A2').values=[['CPU ONNX Runtime, 동일 1,000장. 유지율=INT8/FP32×100, Drop=(FP32−INT8)/FP32×100. 주최 평가기 대조 전.']];
    comparison.getRange('A3').values=[['파라미터·FLOPs는 원본 FP32 구조 기준이며 INT8 비트 연산량이 아닙니다. FP32 AP40=0이면 비율은 정의되지 않습니다.']];
    comparison.getRange('A4:AD4').values=[[
      '비교 ID','FP32 평가 ID','INT8 평가 ID','FP32 mAP40','INT8 mAP40','유지율 (%)','Drop Rate (%)',
      'Car ΔAP40 (pp)','Pedestrian ΔAP40 (pp)','Cyclist ΔAP40 (pp)',
      'FP32 p50 (ms)','INT8 p50 (ms)','Latency 감소 (%)','FP32 RSS (MB)','INT8 RSS (MB)','RSS 감소 (%)',
      'FP32 Params (M)','INT8 Params (M)','FP32 GFLOPs','INT8 GFLOPs','FP32 ONNX (MB)','INT8 ONNX (MB)',
      '파일 감소 (%)','FP32 p95 (ms)','INT8 p95 (ms)','측정 PC','변환 SHA256','측정 조건','Δ Params (M)','Δ GFLOPs']];
    comparison.getRange('A4:AD4').format={fill:'#203864',font:{bold:true,color:'#FFFFFF'},rowHeight:42,wrapText:true};
    comparison.getRange('A:AD').format.columnWidth=17;
    comparison.getRange('A:C').format.columnWidth=25;
    comparison.getRange('Z:AB').format.columnWidth=30;
    comparison.getRange('D5:Y204').setNumberFormat('0.00');
    comparison.freezePanes.freezeRows(4);
    comparison.showGridLines=false;
  }
  let recordRow=null;
  if (payload) {
    if (payload.status !== 'evaluation_complete' || payload.organizer_verified !== false) throw Error('완료된 비교 JSON 필요');
    const reports=[payload.fp32,payload.int8];
    for (const r of reports) {
      if (!r.full_fixed_set || r.conditions.count!==1000 || r.experiment?.smoke_test) throw Error('부분/smoke 결과 금지');
      for (const c of ['Car','Pedestrian','Cyclist']) if (!Number.isFinite(r.per_class[c]?.ap40)) throw Error('AP40 누락');
    }
    const signature=JSON.parse(await fs.readFile(filename+'.conditions.json','utf8'));
    for (const r of reports) for (const [k,v] of Object.entries(signature)) {
      if (JSON.stringify(r.conditions[k])!==JSON.stringify(v)) throw Error(`기존 Excel 측정조건 불일치: ${k}`);
    }
    const conditions={...reports[0].conditions}; delete conditions.precision;
    const afterConditions={...reports[1].conditions}; delete afterConditions.precision;
    if (JSON.stringify(conditions)!==JSON.stringify(afterConditions)) throw Error('쌍의 측정조건 불일치');
    if (reports[0].quantization.bundle_sha256!==reports[1].quantization.bundle_sha256 ||
        reports[0].quantization.source_checkpoint_sha256!==reports[1].quantization.source_checkpoint_sha256) throw Error('모델 쌍 불일치');
    const ids=comparison.getRange('A5:A204').values.flat();
    const existing=ids.indexOf(payload.name);
    if (existing>=0) {
      if (comparison.getRange(`AA${existing+5}`).values[0][0]!==reports[0].quantization.bundle_sha256) throw Error('동일 ID의 다른 결과');
      console.log('이미 기록된 동일 양자화 비교입니다.');
      process.exitCode=0;
    } else {
      const free=ids.findIndex(v=>v===null || v==='');
      if (free<0) throw Error('양자화 비교 표가 가득 찼습니다.');
      recordRow=free+5;
      const logIds=log.getRange('A5:A64').values.flat();
      if (reports.some(r=>logIds.includes(r.name))) throw Error('기존 평가 ID와 충돌합니다.');
      const empty=logIds.map((v,i)=>!v?i+5:null).filter(v=>v!==null);
      if (empty.length<2) throw Error('실험로그에 빈 행 2개가 필요합니다.');
      const literal=v=>typeof v==='string' && v.startsWith('=') ? "'"+v : v;
      for (let i=0;i<2;i++) {
        const r=reports[i],row=empty[i],c=r.conditions,e=r.experiment;
        const cells={A:r.name,B:new Date(r.created_at),C:e.author||'',D:r.reason,E:i?reports[0].name:'',F:r.hypothesis,
          G:path.basename(e.architecture),H:c.imgsz,I:c.imgsz,J:c.precision,K:e.completed_epochs,
          L:r.model.parameters/1e6,M:r.model.flops_g,N:r.model.file_size_mb,
          O:r.per_class.Car.ap40,P:r.per_class.Pedestrian.ap40,Q:r.per_class.Cyclist.ap40,
          T:r.latency_ms.median,U:r.latency_ms.p95,V:r.e2e_ms.mean,X:r.peak_rss_mb,Y:c.host,
          Z:`ONNX Runtime ${c.onnxruntime}; ONNX 파일 크기; 원본 구조 Params/FLOPs; CPU RSS; 주최 대조 전`,AB:'평가 완료'};
        for (const [col,value] of Object.entries(cells)) log.getRange(`${col}${row}`).values=[[literal(value??null)]];
        log.getRange(`B${row}`).setNumberFormat('yyyy-mm-dd');
        log.getRange(`R${row}`).formulas=[[`=AVERAGE(O${row}:Q${row})`]];
        log.getRange(`S${row}`).formulas=[[i?`=IF(R${empty[0]}=0,"",(R${empty[0]}-R${row})/R${empty[0]}*100)`:'=""']];
        log.getRange(`W${row}`).formulas=[[`=IF(T${row}=0,"",1000/T${row})`]];
      }
      const [a,b]=reports,r=recordRow;
      comparison.getRange(`A${r}:AB${r}`).values=[[
        payload.name,a.name,b.name,a.mean_ap40,b.mean_ap40,null,null,
        b.per_class.Car.ap40-a.per_class.Car.ap40,b.per_class.Pedestrian.ap40-a.per_class.Pedestrian.ap40,
        b.per_class.Cyclist.ap40-a.per_class.Cyclist.ap40,a.latency_ms.median,b.latency_ms.median,null,
        a.peak_rss_mb,b.peak_rss_mb,null,a.model.parameters/1e6,b.model.parameters/1e6,a.model.flops_g,b.model.flops_g,
        a.model.file_size_mb,b.model.file_size_mb,null,a.latency_ms.p95,b.latency_ms.p95,a.conditions.host,
        a.quantization.bundle_sha256,JSON.stringify(conditions)]];
      for (const [col,formula] of Object.entries({F:`IF(D${r}=0,"",E${r}/D${r}*100)`,G:`IF(D${r}=0,"",(D${r}-E${r})/D${r}*100)`,
          M:`IF(K${r}=0,"",(K${r}-L${r})/K${r}*100)`,P:`IF(N${r}=0,"",(N${r}-O${r})/N${r}*100)`,
          W:`IF(U${r}=0,"",(U${r}-V${r})/U${r}*100)`,AC:`R${r}-Q${r}`,AD:`IF(OR(S${r}="",T${r}=""),"",T${r}-S${r})`})) comparison.getRange(`${col}${r}`).formulas=[['='+formula]];
      const summary=wb.worksheets.getItem('보고서요약');
      summary.getRange('B4').values=[[a.name]]; summary.getRange('B5').values=[[b.name]];
    }
  }
  if (!payload || recordRow!==null) {
    const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100}});
    if(errors.ndjson.split('\n').some(x=>x && JSON.parse(x).kind==='match')) throw Error(errors.ndjson);
    console.log((await wb.inspect({kind:'table',range:`양자화비교!A4:M${recordRow||5}`,tableMaxRows:3,tableMaxCols:13})).ndjson);
    const backup=path.join(path.dirname(filename),'workbook_backups');
    await fs.mkdir(backup,{recursive:true});
    await fs.writeFile(path.join(backup,`${path.basename(filename,'.xlsx')}_${Date.now()}.xlsx`),original);
    const temp=filename.replace(/\.xlsx$/i,'.quantization.pending.xlsx');
    await (await SpreadsheetFile.exportXlsx(wb)).save(temp);
    try {
      await fs.rename(temp,filename);
    } catch (error) {
      if (!['EPERM','EACCES','EBUSY'].includes(error.code)) throw error;
      console.error(`엑셀 교체가 잠금/권한 문제로 보류됐습니다. 파일을 닫고 기록만 재시도하세요: ${filename}`);
      process.exitCode=75;
    }
    if (process.exitCode!==75) {
    const preview=await wb.render({sheetName:'양자화비교',range:`A4:G${recordRow||6}`,scale:1.5});
    const previewPath=path.join(path.dirname(filename),'outputs/quantization_preview.png');
    await fs.mkdir(path.dirname(previewPath),{recursive:true});
    await fs.writeFile(previewPath,new Uint8Array(await preview.arrayBuffer()));
    console.log('양자화 비교 Excel 저장 완료');
    }
  }
} finally { await lock.close(); await fs.unlink(filename+'.lock'); }
