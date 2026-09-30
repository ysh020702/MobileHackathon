import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(process.argv[2]));
console.log((await wb.inspect({kind:'table',range:'실험로그!A4:AB7',tableMaxRows:4,tableMaxCols:28,maxChars:4500})).ndjson);
console.log(JSON.stringify(wb.worksheets.getItem('실험로그').getRange('R5:S7').formulas));
const image=await wb.render({sheetName:'실험로그',range:'L4:AB7',scale:1});
await fs.writeFile('outputs/quantization_test/workbook_before.png',new Uint8Array(await image.arrayBuffer()));
