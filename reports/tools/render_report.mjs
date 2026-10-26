// Render the report with an isolated headless Chrome profile; no user session.
// Uses Node's built-in WebSocket/fetch and the Chrome DevTools protocol.
import {spawn} from 'node:child_process';
import {readFile,writeFile,mkdir,mkdtemp,rm} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {tmpdir} from 'node:os';
import {pathToFileURL} from 'node:url';

const root=resolve(import.meta.dirname,'../..');
const html=join(root,'reports/vlm-multiturn-rl-tutorial.html');
const pdf=join(root,'output/pdf/vlm-multiturn-rl-tutorial.pdf');
const qa=join(root,'tmp/pdfs');
const chrome=process.env.REPORT_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
await mkdir(qa,{recursive:true});await mkdir(join(root,'output/pdf'),{recursive:true});
const profile=await mkdtemp(join(tmpdir(),'vlm-report-render-'));
const proc=spawn(chrome,['--headless=new','--disable-gpu','--no-first-run','--disable-extensions',
  '--disable-background-networking','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],
  {stdio:['ignore','ignore','pipe']});
let logs='';proc.stderr.on('data',b=>{logs+=b.toString()});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));let ws;
try {
  let port;
  for(let i=0;i<100;i++){
    try{port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];break}catch{}
    if(proc.exitCode!==null)throw Error('Chrome exited: '+logs.slice(-1500));
    await sleep(100);
  }
  if(!port)throw Error('Chrome debugging endpoint did not start');
  const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true})});
  let counter=0;const pending=new Map();const errors=[];
  ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown'){errors.push(m.params.exceptionDetails.text)}});
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{const id=++counter;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
  const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
  await cdp('Page.enable');await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:pathToFileURL(html).href});
  for(let i=0;i<100;i++){
    if(await evaluate('document.readyState==="complete" && Array.from(document.images).every(i=>i.complete)'))break;
    await sleep(100);
  }
  await evaluate('document.fonts.ready.then(()=>true)');
  const audit=await evaluate(`({title:document.title,images:document.images.length,brokenImages:Array.from(document.images).filter(i=>!i.naturalWidth).length,headings:document.querySelectorAll('h2').length,externalAssets:Array.from(document.querySelectorAll('[src],link[rel="stylesheet"]')).map(e=>e.src||e.href).filter(u=>u&&!u.startsWith('data:')&&!u.startsWith('file:')),screenOverflow:document.documentElement.scrollWidth>innerWidth})`);
  const capture=async(name,selector)=>{
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start'});`);await sleep(100);
    const image=await cdp('Page.captureScreenshot',{format:'png'});await writeFile(join(qa,name+'.png'),Buffer.from(image.data,'base64'));
  };
  await capture('html-cover','#top');
  await capture('html-task-gallery','.triptych');
  await capture('html-math','#trajectory');
  await evaluate(`document.getElementById('policy').value='after';document.getElementById('policy').dispatchEvent(new Event('change'));for(let i=0;i<6;i++)document.getElementById('next').click();`);
  audit.replay=await evaluate(`({label:document.getElementById('frame-label').textContent,action:document.getElementById('action').textContent,feedback:document.getElementById('feedback').textContent,nextDisabled:document.getElementById('next').disabled})`);
  if(audit.replay.label!=='Action 7 of 7'||!audit.replay.feedback.includes('Reward: 1'))throw Error('Replay controls failed');
  await capture('html-replay','.replay');
  await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
  audit.mobileOverflow=await evaluate('document.documentElement.scrollWidth>innerWidth');
  audit.mobileOffenders=await evaluate(`Array.from(document.querySelectorAll('main *')).filter(e=>e.getBoundingClientRect().right>innerWidth+1&&getComputedStyle(e).position!=='fixed').slice(0,15).map(e=>({tag:e.tagName,class:e.className,right:Math.round(e.getBoundingClientRect().right)}))`);
  await capture('html-mobile','#case');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Emulation.setEmulatedMedia',{media:'print'});
  audit.printMathWidths=await evaluate(`Array.from(document.querySelectorAll('.equation')).map(e=>({width:e.clientWidth,content:e.scrollWidth}))`);
  const result=await cdp('Page.printToPDF',{printBackground:true,preferCSSPageSize:true,
      displayHeaderFooter:false,generateDocumentOutline:true,generateTaggedPDF:true});
  await writeFile(pdf,Buffer.from(result.data,'base64'));
  audit.jsErrors=errors;
  await writeFile(join(qa,'html-audit.json'),JSON.stringify(audit,null,2)+'\n');
  if(audit.brokenImages||audit.externalAssets.length||audit.jsErrors.length||audit.screenOverflow||audit.mobileOverflow)throw Error('HTML audit failed: '+JSON.stringify(audit));
  console.log(JSON.stringify({pdf,audit},null,2));
  await cdp('Browser.close');
} finally {
  ws?.close();proc.kill('SIGTERM');await sleep(300);await rm(profile,{recursive:true,force:true}).catch(()=>{});
}
