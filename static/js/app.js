'use strict';
const $=id=>document.getElementById(id);
const CLASSES=['Person','Luggage','Hurry Up/Busy','Pet'];
const DB_CLS={'Person':'db-person','Luggage':'db-luggage','Hurry Up/Busy':'db-hurry','Pet':'db-pet'};
const CB_KEY={'Person':'Person','Luggage':'Luggage','Hurry Up/Busy':'Hurry','Pet':'Pet'};
const CC=['#1A56DB','#18A558','#E31837','#7C3AED'];
let running=false,videoPath=null,pollTimer=null,dashTimer=null;
let barC=null,pieC=null,lineC=null;

function reconnectVideoFeed(){
  const img = $('vfeed');
  if(!img) return;
  // Cache-buster forces the browser to open a fresh MJPEG connection.
  // This prevents the stream from staying broken after tab switching or Stop.
  img.src = '/video_feed?t=' + Date.now();
}

const vfeed = $('vfeed');
if(vfeed){
  vfeed.onerror = () => {
    // Do not show a fatal error. Reconnect softly while the app is running.
    if(running) setTimeout(reconnectVideoFeed, 800);
  };
}

document.querySelectorAll('.tab').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    $('panel-'+btn.dataset.target).classList.add('active');
    if(btn.dataset.target==='stats'){
      refreshDash();
      startDashTimer();
    }else{
      stopDashTimer();
      if(btn.dataset.target==='live') reconnectVideoFeed();
    }
  });
});

$('op-mode').addEventListener('change',function(){
  const hints={
    detection:          'Phát hiện object, hiển thị bounding box',
    counting:           'Đếm object theo vùng IN / OUT',
    detection_counting: 'Vừa phát hiện vừa đếm IN / OUT theo vùng',
  };
  const badges={
    detection:          '✈ DETECTION MODE',
    counting:           '✈ COUNTING MODE',
    detection_counting: '✈ DETECTION & COUNTING',
  };
  $('op-hint').textContent = hints[this.value]||'';
  $('mode-badge').textContent = badges[this.value]||'✈ DETECTION MODE';
  const cnt = this.value==='counting'||this.value==='detection_counting';
  ['lbc-in','lbc-out','lbc-net'].forEach(id=>$(id).classList.toggle('hidden',!cnt));
});

$('model-src').addEventListener('change',function(){
  const p=this.value==='pretrained';
  $('pretrained-block').classList.toggle('hidden',!p);
  $('custom-block').classList.toggle('hidden',p);
});

$('video-src').addEventListener('change',function(){
  $('video-upload-block').classList.toggle('hidden',this.value!=='video');
});

[['conf-s','conf-v',v=>(+v).toFixed(2)],['iou-s','iou-v',v=>(+v).toFixed(2)],['skip-s','skip-v',v=>v]]
.forEach(([s,v,f])=>$(s).addEventListener('input',function(){$(v).textContent=f(this.value);}));

// ── Danh sách model trong models/ ─────────────────────────
const DEFAULT_MODELS = [
  // YOLO26 checkpoints trained/used with Open Images V7 labels.
  'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt',
  // Keep YOLO11 COCO models as fallback/demo options.
  'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt',
];

async function loadModelList() {
  const sel = $('pretrained-name');
  const info = $('model-info');
  try {
    const models = await (await fetch('/api/list_models')).json();
    const all = [...new Set([...models, ...DEFAULT_MODELS])];
    const cur = sel.value;
    sel.innerHTML = '<option value="">— Chọn checkpoint —</option>'
      + all.map(m => {
          const inFolder = models.includes(m);
          const label = inFolder ? `✓ ${m}` : `↓ ${m} (auto-download)`;
          return `<option value="${m}" ${m===cur?'selected':''}>${label}</option>`;
        }).join('');
    info.innerHTML = models.length
      ? `<span class="dot"></span>${models.length} model trong thư mục models/`
      : '<span style="color:var(--amber)">⚠ Chưa có model nào — sẽ tự tải khi Load</span>';
  } catch(e) {
    sel.innerHTML = '<option value="">— Không kết nối được server —</option>';
  }
}

$('btn-refresh-models').addEventListener('click', loadModelList);

$('btn-load').addEventListener('click',async()=>{
  const p = $('pretrained-name').value.trim();
  if (!p) { setMsg('Chọn một checkpoint trước', false); return; }
  $('btn-load').disabled = true;
  setMsg('Loading…', null);
  const r = await api('/api/load_model', { model_type:'pretrained', model_path:p });
  setMsg(r.message, r.success);
  if (r.success) {
    $('model-info').innerHTML = `<span class="dot"></span>Đang dùng: <strong>${p}</strong>`;
    loadModelList();   // refresh để cập nhật ✓ marker
  }
  $('btn-load').disabled = false;
});

$('custom-file').addEventListener('change',function(){
  $('custom-name').textContent=this.files[0]?.name||'No file chosen';
});
$('btn-upload-model').addEventListener('click',async()=>{
  const f=$('custom-file').files[0];
  if(!f){setMsg('Choose a .pt file',false);return;}
  $('btn-upload-model').disabled=true; setMsg('Uploading…',null);
  const fd=new FormData();fd.append('file',f);
  const u=await(await fetch('/api/upload_model',{method:'POST',body:fd})).json();
  if(!u.success){setMsg(u.message,false);$('btn-upload-model').disabled=false;return;}
  const r=await api('/api/load_model',{model_type:'custom',model_path:u.path});
  setMsg(r.message,r.success); $('btn-upload-model').disabled=false;
});

$('video-file').addEventListener('change',function(){
  $('video-fname').textContent=this.files[0]?.name||'No file chosen';
});
$('btn-upload-video').addEventListener('click',async()=>{
  const f=$('video-file').files[0];if(!f)return;
  $('btn-upload-video').disabled=true;
  const fd=new FormData();fd.append('file',f);
  const r=await(await fetch('/api/upload_video',{method:'POST',body:fd})).json();
  if(r.success){videoPath=r.path;$('video-ok').classList.remove('hidden');}
  $('btn-upload-video').disabled=false;
});

$('btn-start').addEventListener('click',async()=>{
  const cls=[...document.querySelectorAll('.cls-row input:checked')].map(e=>e.value);
  if(!cls.length){alert('Select at least one class.');return;}
  if($('video-src').value==='video'&&!videoPath){alert('Upload a video first.');return;}

  // 6 presets: 3 split ratios × 2 IN/OUT directions.
  // Format: "ratio:in_side", e.g. "0.30:left" means
  // line at 30% width and LEFT side is considered IN.
  const preset = $('count-preset') ? $('count-preset').value : '0.50:right';
  const [countFracRaw, countInSideRaw] = preset.split(':');

  const r=await api('/api/start',{
    mode:$('op-mode').value, source:$('video-src').value,
    video_path:videoPath||'', classes:cls,
    count_axis: $('count-axis') ? $('count-axis').value : 'x',
    count_frac: +(countFracRaw || 0.50),
    count_in_side: countInSideRaw || 'right',
    confidence:+$('conf-s').value, iou:+$('iou-s').value, frame_skip:+$('skip-s').value,
  });
  if(r.success){running=true;setUI(true);reconnectVideoFeed();startPoll();}else alert('Error: '+r.message);
});

$('btn-stop').addEventListener('click',async()=>{
  $('btn-stop').disabled = true;
  try{
    await api('/api/stop',{});
  }catch(e){
    console.warn('Soft stop warning:', e);
  }
  running=false;
  setUI(false);
  stopPoll();
  // Keep the video endpoint alive. Stop is only a pause/interruption, not a full disconnect.
  setTimeout(reconnectVideoFeed, 250);
});

function setUI(on){
  $('btn-start').disabled=on; $('btn-stop').disabled=!on;
  $('feed-overlay').classList.toggle('gone',on);
  $('status-dot').className='status-dot'+(on?' live':'');
  $('status-lbl').textContent=on?'Running':'Idle';
}

function setMsg(msg,ok){
  const el=$('model-msg');
  el.textContent=msg;
  el.className='msg'+(ok===true?' ok':ok===false?' err':' loading');
  el.classList.remove('hidden');
}

function startPoll(){ if(pollTimer) clearInterval(pollTimer); pollTimer=setInterval(fetchLive,500); }
function stopPoll(){ if(pollTimer){ clearInterval(pollTimer); pollTimer=null; } }

async function fetchLive(){
  try{
    const d=await(await fetch('/api/stats')).json();
    if(!d.running&&running){running=false;setUI(false);stopPoll();return;}
    $('lv-fps').textContent=d.fps;
    if(d.mode==='counting'||d.mode==='detection_counting'){
      $('lv-in').textContent  = d.total_in  ?? 0;
      $('lv-out').textContent = d.total_out ?? 0;
      $('lv-net').textContent = d.net       ?? 0;
    }
    const cur=d.current||{};
    CLASSES.forEach(c=>{const k=CB_KEY[c];const el=$('cb-'+k);if(el)el.textContent=cur[c]||0;});
    const keys=Object.keys(cur).filter(k=>cur[k]>0);
    const badges=$('det-badges');
    badges.innerHTML=keys.length
      ?keys.map(k=>`<span class="det-badge ${DB_CLS[k]||'db-person'}"><span class="dot"></span>${k}: ${cur[k]}</span>`).join('')
      :'<span class="det-empty">No detections yet</span>';
  }catch(e){}
}

function startDashTimer(){ if(dashTimer) clearInterval(dashTimer); dashTimer=setInterval(refreshDash,3000); }
function stopDashTimer(){ if(dashTimer){ clearInterval(dashTimer); dashTimer=null; } }

async function refreshDash(){
  try{
    const[d,sess]=await Promise.all([
      fetch('/api/stats').then(r=>r.json()),
      fetch('/api/sessions').then(r=>r.json()),
    ]);
    const aIn={},aOut={};
    const sources = [...sess];
    // Only add the live session while it is still running.
    // After Stop, the session has already been saved into logs/sessions.json,
    // so adding d.in_counts/d.out_counts again would double-count it.
    if (d.running) {
      sources.push({in_counts:d.in_counts||{}, out_counts:d.out_counts||{}});
    }
    sources.forEach(s=>{
      Object.entries(s.in_counts||{}).forEach(([k,v])=>aIn[k]=(aIn[k]||0)+v);
      Object.entries(s.out_counts||{}).forEach(([k,v])=>aOut[k]=(aOut[k]||0)+v);
    });
    const tI=Object.values(aIn).reduce((a,b)=>a+b,0);
    const tO=Object.values(aOut).reduce((a,b)=>a+b,0);
    $('kpi-in').textContent=tI; $('kpi-out').textContent=tO;
    $('kpi-net').textContent=tI-tO; $('kpi-fps').textContent=d.avg_fps||0;
    $('last-update').textContent=new Date().toLocaleTimeString();
    renderBar(aIn); renderPie(d.current||{}); renderLine(d.timeline||[]); renderTable(aIn,aOut);
  }catch(e){console.error(e);}
}

function renderBar(inC){
  if(typeof Chart === 'undefined') return;
  if(barC)barC.destroy();
  barC=new Chart($('chart-bar'),{type:'bar',data:{labels:CLASSES,datasets:[{
    label:'IN count',data:CLASSES.map(c=>inC[c]||0),
    backgroundColor:CC.map(c=>c+'cc'),borderRadius:5,
  }]},options:{responsive:true,animation:false,plugins:{legend:{display:false}},
    scales:{x:{grid:{display:false},ticks:{font:{size:11}}},
            y:{beginAtZero:true,ticks:{font:{size:11},stepSize:1},grid:{color:'rgba(0,0,0,.05)'}}}
  }});
}

function renderPie(cur){
  if(typeof Chart === 'undefined') return;
  if(pieC)pieC.destroy();
  const lbls=CLASSES.filter(c=>(cur[c]||0)>0);
  if(!lbls.length)return;
  pieC=new Chart($('chart-pie'),{type:'doughnut',data:{labels:lbls,datasets:[{
    data:lbls.map(c=>cur[c]||0),
    backgroundColor:lbls.map(l=>CC[CLASSES.indexOf(l)]+'dd'),borderWidth:1,
  }]},options:{responsive:true,animation:false,
    plugins:{legend:{position:'bottom',labels:{font:{size:11}}}}}});
}

function renderLine(tl){
  if(typeof Chart === 'undefined') return;
  if(lineC)lineC.destroy();
  const last=tl.slice(-50);
  lineC=new Chart($('chart-line'),{type:'line',data:{labels:last.map((_,i)=>i+1),
    datasets:CLASSES.map((c,i)=>({label:c,data:last.map(d=>(d.cur||{})[c]||0),
      borderColor:CC[i],backgroundColor:CC[i]+'20',tension:0.3,borderWidth:1.5,pointRadius:0,fill:true,
    }))},options:{responsive:true,animation:false,plugins:{legend:{labels:{font:{size:11}}}},
      scales:{x:{display:false},y:{beginAtZero:true,ticks:{font:{size:11}},grid:{color:'rgba(0,0,0,.05)'}}}
  }});
}

function renderTable(aIn,aOut){
  const tbody=$('detail-tbody');
  const keys=[...new Set([...Object.keys(aIn),...Object.keys(aOut)])];
  if(!keys.length){
    tbody.innerHTML='<tr><td colspan="5" style="color:#9399b2;text-align:center;padding:20px">No data recorded yet</td></tr>';return;
  }
  tbody.innerHTML=keys.map(k=>{const i=aIn[k]||0,o=aOut[k]||0;
    return`<tr><td>${k}</td><td>${i}</td><td>${o}</td><td>${i+o}</td><td>${i-o}</td></tr>`;}).join('');
}

$('btn-refresh').addEventListener('click',refreshDash);
$('btn-clear').addEventListener('click',async()=>{
  if(!confirm('Clear all session statistics?'))return;
  await fetch('/api/clear_sessions',{method:'POST'});refreshDash();
});

async function api(url,body){
  const res = await fetch(url,{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  });
  if(!res.ok){
    throw new Error(`HTTP ${res.status}`);
  }
  return await res.json();
}

// ── Khởi động ─────────────────────────────────────────────
loadModelList();