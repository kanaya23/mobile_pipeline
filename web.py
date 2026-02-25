"""FastAPI application: endpoints, HTML pages, server launcher."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .config import STORAGE, JOB_STATUS
from .utils import parse_zoom_crop, read_json, metadata_path
from .jobs import (
    JobRequest, queue_job, process_job,
    gallery_items, image_paths,
    set_status,
)
from .config import CONFIG, Mode, Look


# ══════════════════════════════════════════════════════════════════════════════
# HTML pages
# ══════════════════════════════════════════════════════════════════════════════

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Photo Pipeline</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,Arial,sans-serif;margin:0;padding:16px;max-width:800px;background:#f4f4f4;color:#222}
h1{font-size:1.2rem;margin:0 0 14px}h3{font-size:.95rem;margin:0 0 10px;color:#444}
.card{background:#fff;border:1px solid #ddd;border-radius:14px;padding:16px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:640px){.row{grid-template-columns:1fr}}
label{font-size:.82rem;color:#666;display:block;margin-top:10px}
select,input[type=text]{width:100%;padding:9px 10px;border:1px solid #ccc;border-radius:9px;margin-top:4px;font-size:.92rem;background:#fafafa}
input[type=file]{width:100%;padding:7px;border:1px solid #ccc;border-radius:9px;margin-top:4px;font-size:.86rem;background:#fafafa}
.chk-row{display:flex;align-items:center;gap:8px;margin-top:12px;font-size:.9rem}
button{width:100%;padding:11px;margin-top:12px;border:none;border-radius:10px;font-size:.97rem;cursor:pointer;font-weight:600}
.btn-p{background:#1a73e8;color:#fff}.btn-p:hover{background:#1558c0}
.btn-s{background:#5f6368;color:#fff}.btn-s:hover{background:#444}
#statusBox{font-family:monospace;font-size:.82rem;background:#f0f4ff;border-radius:9px;padding:10px;margin-top:10px;min-height:32px;color:#333}
video{width:100%;border-radius:10px;background:#000;max-height:240px;display:block;margin-bottom:4px}
a.gl{display:inline-block;margin-top:10px;color:#1a73e8;font-size:.9rem;text-decoration:none;font-weight:500}
</style>
</head>
<body>
<h1>&#128247; Mobile Photography Pipeline</h1>

<div class="card">
<h3>&#9881;&#65039; Options</h3>
<div class="row">
  <div>
    <label>Mode</label>
    <select id="mode">
      <option value="fast">&#9889; Fast (~15 s)</option>
      <option value="quality">&#127912; Quality (~90 s)</option>
    </select>
  </div>
  <div>
    <label>Look</label>
    <select id="look">
      <option value="blend">&#10024; Blend</option>
      <option value="leica">&#128308; Leica</option>
      <option value="pixel">&#128309; Pixel</option>
    </select>
  </div>
</div>
<div class="chk-row"><input type="checkbox" id="portrait"/><label style="margin:0">Portrait / bokeh</label></div>
<label>Zoom crop x,y,w,h (0-1, leave blank for full frame)</label>
<input type="text" id="zoom" placeholder="e.g. 0.25,0.20,0.50,0.60"/>
</div>

<div class="card">
<h3>&#128194; Upload RAW or JPEG</h3>
<label>Primary file (.dng / .jpg / .jpeg / .png)</label>
<input type="file" id="fileInput" accept=".dng,.jpg,.jpeg,.png"/>
<label>Burst frames for HDR (select 5-7 extras, optional)</label>
<input type="file" id="burstInput" accept=".dng,.jpg,.jpeg,.png" multiple/>
<button class="btn-p" onclick="uploadFile()">Upload &amp; Process</button>
</div>

<div class="card">
<h3>&#128247; Quick camera (JPEG)</h3>
<video id="video" autoplay playsinline></video>
<canvas id="canvas" style="display:none"></canvas>
<button class="btn-s" onclick="startCam()">Open camera</button>
<button class="btn-p" onclick="captureCam()">Capture &amp; Process</button>
</div>

<div class="card">
<h3>&#128202; Status</h3>
<div id="statusBox">No active job.</div>
<a class="gl" href="/gallery_page">&#8594; Open Gallery</a>
</div>

<script>
let job=null,timer=null;
function stopPoll(){if(timer){clearInterval(timer);timer=null;}}
function show(m){document.getElementById('statusBox').textContent=m;}
function opts(){return{
  mode:document.getElementById('mode').value,
  look:document.getElementById('look').value,
  portrait:document.getElementById('portrait').checked?'true':'false',
  zoom:document.getElementById('zoom').value.trim()
};}
function startPoll(){
  stopPoll();
  timer=setInterval(async()=>{
    try{
      const r=await fetch('/status/'+job);
      if(!r.ok){stopPoll();show('Poll error '+r.status);return;}
      const s=await r.json();
      const lbl=(s.stage||'').replace(/_/g,' ')+' '+(s.percent||0)+'%';
      show(s.status==='failed'?'Failed: '+(s.detail||'?'):lbl);
      if(s.status==='done'){stopPoll();show('Done! Opening gallery...');setTimeout(()=>{location.href='/gallery_page';},1200);}
      if(s.status==='failed') stopPoll();
    }catch(e){show('Network error: '+e.message);}
  },2000);
}
async function uploadFile(){
  const f=document.getElementById('fileInput').files[0];
  if(!f){alert('Select a file first');return;}
  const o=opts(),fd=new FormData();
  fd.append('file',f);
  const burst=document.getElementById('burstInput').files;
  for(let i=0;i<burst.length;i++) fd.append('burst_files',burst[i]);
  fd.append('mode',o.mode);fd.append('look',o.look);
  fd.append('portrait',o.portrait);
  if(o.zoom) fd.append('zoom_crop',o.zoom);
  show('Uploading...');stopPoll();
  const r=await fetch('/upload',{method:'POST',body:fd});
  if(!r.ok){show('Error: '+await r.text());return;}
  job=(await r.json()).job_id;startPoll();
}
async function startCam(){
  try{const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
  document.getElementById('video').srcObject=s;}catch(e){alert('Camera: '+e.message);}
}
async function captureCam(){
  const v=document.getElementById('video');
  if(!v.srcObject){alert('Open camera first');return;}
  const c=document.getElementById('canvas');
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  const blob=await new Promise(res=>c.toBlob(res,'image/jpeg',0.95));
  const o=opts(),fd=new FormData();
  fd.append('file',blob,'capture.jpg');
  fd.append('mode',o.mode);fd.append('look',o.look);fd.append('portrait',o.portrait);
  if(o.zoom) fd.append('zoom_crop',o.zoom);
  show('Uploading capture...');stopPoll();
  const r=await fetch('/capture',{method:'POST',body:fd});
  if(!r.ok){show('Error: '+await r.text());return;}
  job=(await r.json()).job_id;startPoll();
}
</script>
</body></html>"""

GALLERY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gallery</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,Arial,sans-serif;margin:0;padding:16px;background:#f4f4f4;color:#222}
h1{font-size:1.2rem;margin:0 0 14px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
.card{background:#fff;border:1px solid #ddd;border-radius:14px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.card img{width:100%;display:block;cursor:pointer;transition:opacity .15s}
.card img:hover{opacity:.85}
.info{padding:10px 12px 12px}
.badges{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px}
.badge{padding:2px 9px;border-radius:999px;background:#e8eaf6;font-size:.72rem;color:#3949ab;font-weight:600}
.ts{font-size:.72rem;color:#999;margin-bottom:8px}
.actions{display:flex;gap:6px}
.actions a,.actions button{flex:1;text-align:center;padding:7px 4px;border-radius:8px;font-size:.8rem;text-decoration:none;cursor:pointer;font-weight:500;border:none}
.dl{background:#e8f0fe;color:#1a73e8}.orig{background:#f1f3f4;color:#444}.del{background:#fde8e8;color:#c5221f}
/* Comparison overlay */
#ov{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.92);z-index:999;align-items:center;justify-content:center;flex-direction:column;gap:10px}
#ov.open{display:flex}
#ov p{color:#bbb;margin:0;font-size:.8rem}
#cmp{display:flex;gap:8px;max-width:96vw;max-height:82vh}
#cmp figure{margin:0;display:flex;flex-direction:column;align-items:center;gap:4px;max-width:46vw}
#cmp figcaption{color:#aaa;font-size:.74rem}
#cmp img{max-width:100%;max-height:78vh;object-fit:contain;border-radius:8px}
a.back{display:inline-block;margin-bottom:12px;color:#1a73e8;font-size:.9rem;text-decoration:none;font-weight:500}
</style>
</head>
<body>
<a class="back" href="/">&#8592; Upload / Capture</a>
<h1>&#128247; Gallery</h1>
<div id="grid" class="grid"></div>
<div id="ov">
  <p>Tap outside to close</p>
  <div id="cmp">
    <figure><figcaption>Processed</figcaption><img id="cP"/></figure>
    <figure><figcaption>Original</figcaption><img id="cO"/></figure>
  </div>
</div>
<script>
const ov=document.getElementById('ov');
ov.addEventListener('click',e=>{if(e.target===ov)ov.classList.remove('open');});
function compare(id){
  document.getElementById('cP').src='/image/'+id+'?t='+Date.now();
  document.getElementById('cO').src='/image/'+id+'/original?t='+Date.now();
  ov.classList.add('open');
}
async function del(id){
  if(!confirm('Delete?')) return;
  await fetch('/image/'+id,{method:'DELETE'});
  load();
}
async function load(){
  const r=await fetch('/gallery');
  if(!r.ok) return;
  const g=document.getElementById('grid');
  g.innerHTML='';
  for(const item of (await r.json()).items){
    const id=item.job_id;
    const mode=item.request.mode==='quality'?'&#127912; Quality':'&#9889; Fast';
    const port=item.request.portrait?'<span class="badge">portrait</span>':'';
    const ts=item.completed_at?new Date(item.completed_at).toLocaleString():'';
    const c=document.createElement('div');
    c.className='card';
    c.innerHTML=`<img src="/image/${id}/thumb" loading="lazy" title="Tap to compare" onclick="compare('${id}')"/>
<div class="info">
  <div class="badges"><span class="badge">${mode}</span><span class="badge">${item.request.look}</span>${port}</div>
  <div class="ts">${ts}</div>
  <div class="actions">
    <a class="dl" href="/image/${id}" target="_blank">Download</a>
    <a class="orig" href="/image/${id}/original" target="_blank">Original</a>
    <button class="del" onclick="del('${id}')">Delete</button>
  </div>
</div>`;
    g.appendChild(c);
  }
}
load();
setInterval(load,5000);
</script>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Mobile Computational Photography Pipeline")


def _parse_req(mode: str, look: str, portrait: str, zoom_crop: Optional[str]) -> JobRequest:
    try:
        return JobRequest(
            mode=Mode(mode),
            look=Look(look),
            portrait=str(portrait).lower() in {"1", "true", "yes", "on"},
            zoom_crop=parse_zoom_crop(zoom_crop),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _save_upload(upload: UploadFile, forced_suffix: Optional[str] = None) -> Path:
    suffix = forced_suffix or Path(upload.filename or "upload.bin").suffix.lower()
    tmp = CONFIG.base_dir / "_tmp_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / f"{uuid.uuid4()}{suffix}"
    dst.write_bytes(await upload.read())
    return dst


@app.get("/",            response_class=HTMLResponse)
def index():   return INDEX_HTML

@app.get("/gallery_page", response_class=HTMLResponse)
def gallery_page(): return GALLERY_HTML

@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file:             UploadFile        = File(...),
    burst_files:      Optional[List[UploadFile]] = File(None),
    mode:             str               = Form("fast"),
    look:             str               = Form("blend"),
    portrait:         str               = Form("false"),
    zoom_crop:        Optional[str]     = Form(None),
):
    req = _parse_req(mode, look, portrait, zoom_crop)
    temp = await _save_upload(file)
    temp_bursts: List[Path] = []
    for bf in burst_files or []:
        if bf and bf.filename:
            temp_bursts.append(await _save_upload(bf))
    try:
        payload = queue_job(temp, req, burst_paths=temp_bursts, run_sync=False)
        if not CONFIG.enable_celery:
            inp  = Path(payload.pop("saved_input"))
            brst = [Path(x) for x in payload.pop("saved_bursts", [])]
            background_tasks.add_task(process_job, payload["job_id"], inp, req, brst)
        return payload
    finally:
        if temp.exists(): temp.unlink()
        for bp in temp_bursts:
            if bp.exists(): bp.unlink()


@app.post("/capture")
async def capture(
    background_tasks: BackgroundTasks,
    file:             UploadFile    = File(...),
    mode:             str           = Form("fast"),
    look:             str           = Form("blend"),
    portrait:         str           = Form("false"),
    zoom_crop:        Optional[str] = Form(None),
):
    req = _parse_req(mode, look, portrait, zoom_crop)
    temp = await _save_upload(file, ".jpg")
    try:
        payload = queue_job(temp, req, run_sync=False)
        if not CONFIG.enable_celery:
            inp  = Path(payload.pop("saved_input"))
            brst = [Path(x) for x in payload.pop("saved_bursts", [])]
            background_tasks.add_task(process_job, payload["job_id"], inp, req, brst)
        return payload
    finally:
        if temp.exists(): temp.unlink()


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id in JOB_STATUS:
        return JOB_STATUS[job_id]
    mf = metadata_path(job_id)
    if not mf.exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    return read_json(mf).get("status", {"status": "unknown"})


@app.get("/gallery")
def gallery(): return {"items": gallery_items()}


@app.get("/image/{job_id}")
def image(job_id: str):
    try:    p = image_paths(job_id)
    except FileNotFoundError: raise HTTPException(404, "Unknown job")
    if not p["processed"] or not p["processed"].exists():
        raise HTTPException(404, "Processed image not ready yet")
    return FileResponse(p["processed"])


@app.get("/image/{job_id}/thumb")
def image_thumb(job_id: str):
    try:    p = image_paths(job_id)
    except FileNotFoundError: raise HTTPException(404, "Unknown job")
    if not p["thumbnail"] or not p["thumbnail"].exists():
        raise HTTPException(404, "Thumbnail missing")
    return FileResponse(p["thumbnail"])


@app.get("/image/{job_id}/original")
def image_original(job_id: str):
    try:    p = image_paths(job_id)
    except FileNotFoundError: raise HTTPException(404, "Unknown job")
    if not p["original"].exists():
        raise HTTPException(404, "Original missing")
    return FileResponse(p["original"])


@app.delete("/image/{job_id}")
def image_delete(job_id: str):
    try:    p = image_paths(job_id)
    except FileNotFoundError: raise HTTPException(404, "Unknown job")
    for key in ["original", "processed", "thumbnail", "meta"]:
        path = p.get(key)
        if path and path.exists(): path.unlink()
    JOB_STATUS.pop(job_id, None)
    return {"deleted": job_id}


# ══════════════════════════════════════════════════════════════════════════════
# Server launcher (used by notebook)
# ══════════════════════════════════════════════════════════════════════════════

_server_thread: Optional[threading.Thread] = None
_uvicorn_server = None


def start_server(host: str = "0.0.0.0", port: int = 8000, log_level: str = "warning") -> None:
    global _server_thread, _uvicorn_server

    if _server_thread is not None and _server_thread.is_alive():
        print(f"Server already running on port {port}")
        return

    import nest_asyncio
    nest_asyncio.apply()

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    _uvicorn_server = uvicorn.Server(config)
    _server_thread = threading.Thread(target=_uvicorn_server.run, daemon=True)
    _server_thread.start()

    # Wait until actually ready (poll /health instead of blind sleep)
    import urllib.request as _ur
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            _ur.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            print(f"✓ Server ready → http://0.0.0.0:{port}")
            return
        except Exception:
            time.sleep(0.3)

    print("WARN: Server did not respond to /health within 15 s — check logs.")


def stop_server() -> None:
    global _uvicorn_server
    if _uvicorn_server:
        _uvicorn_server.should_exit = True
        print("Server stop signal sent.")
