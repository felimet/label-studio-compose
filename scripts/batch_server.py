#!/usr/bin/env python3
"""Web UI companion service for SAM3/SAM2.1 batch annotation.

Exposes a browser-accessible form at http://<host>:8085/ to trigger
batch_annotate.py without requiring terminal access.

Usage:
    uvicorn scripts.batch_server:app --host 0.0.0.0 --port 8085
    make batch-server
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import uuid
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Batch Annotation Server", version="1.0.0")

# In-memory job registry (lost on restart — acceptable, jobs complete in minutes)
_jobs: dict[str, dict] = {}


# ── Output collector ─────────────────────────────────────────────────────


async def _collect_output(job_id: str, proc: subprocess.Popen) -> None:
    """Collect subprocess stdout/stderr without blocking the event loop."""
    loop = asyncio.get_event_loop()
    while True:
        line: str = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        _jobs[job_id]["log"].append(line.rstrip())
        _jobs[job_id]["log"] = _jobs[job_id]["log"][-100:]  # keep last 100 lines
    rc = await loop.run_in_executor(None, proc.wait)
    job = _jobs.get(job_id)
    if job and job["status"] == "running":
        job["status"] = "done" if rc == 0 else "failed"
        job["exit_code"] = rc


# ── HTML ─────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Batch Annotation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root{
      --font:'Inter',system-ui,-apple-system,sans-serif;
      --mono:'SF Mono','Fira Code','Cascadia Code',Consolas,monospace;
      --r-s:8px; --r-m:12px; --r-l:16px;
      --tr:0.18s ease;
    }
    /* ── Dark (default) — charcoal + amber ────────────── */
    [data-theme="dark"]{
      --bg:#191919; --bg-c:#232323; --bg-i:#2C2C2C; --bg-log:#1E1E1E;
      --tx:#E8E4E0; --tx2:#A09890; --txm:#6B6360;
      --bd:#333; --bdf:#D4914E;
      --ac:#D4914E; --ac-h:#E8A55E; --ac-tx:#191919;
      --dg:#D06060; --dg-h:#E07070;
      --ok:#5CB85C;
      --b-run:#D4914E; --b-ok:#5CB85C; --b-fail:#D06060; --b-stop:#777;
      --sh:0 1px 4px rgba(0,0,0,.4);
    }
    /* ── Light ────────────────────────────────────────── */
    [data-theme="light"]{
      --bg:#F8F5F0; --bg-c:#FFFFFF; --bg-i:#F0ECE6; --bg-log:#F5F1EB;
      --tx:#2C2420; --tx2:#6B5E52; --txm:#9E9080;
      --bd:#DDD5CB; --bdf:#C07A3E;
      --ac:#C07A3E; --ac-h:#A96830; --ac-tx:#FFF;
      --dg:#C45050; --dg-h:#A63D3D;
      --ok:#4A8F4A;
      --b-run:#D4914E; --b-ok:#4A8F4A; --b-fail:#C45050; --b-stop:#999;
      --sh:0 1px 3px rgba(60,40,20,.08),0 4px 12px rgba(60,40,20,.04);
    }

    *,*::before,*::after{box-sizing:border-box}
    body{font-family:var(--font);background:var(--bg);color:var(--tx);margin:0;padding:0;min-height:100vh;
         transition:background var(--tr),color var(--tr)}
    .shell{max-width:1180px;margin:0 auto;padding:1.75rem 1.5rem 3rem}

    /* header */
    .hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.25rem}
    .hdr h1{font-size:1.25rem;font-weight:700;margin:0;letter-spacing:-.02em}
    .hdr h1 em{font-style:normal;color:var(--ac)}
    .thm{background:var(--bg-i);border:1px solid var(--bd);border-radius:var(--r-s);
         color:var(--tx);padding:5px 9px;cursor:pointer;font-size:1rem;line-height:1;
         transition:background var(--tr),border-color var(--tr)}
    .thm:hover{border-color:var(--bdf)}

    /* grid */
    .g{display:grid;grid-template-columns:400px 1fr;gap:1.25rem;align-items:start}
    @media(max-width:860px){.g{grid-template-columns:1fr}}

    /* card */
    .c{background:var(--bg-c);border:1px solid var(--bd);border-radius:var(--r-l);
       padding:1.35rem;box-shadow:var(--sh);transition:background var(--tr),border-color var(--tr),box-shadow var(--tr)}

    /* field */
    .f{margin-bottom:.85rem}.f:last-child{margin-bottom:0}
    .f>label{display:block;font-size:.72rem;font-weight:600;color:var(--tx2);margin-bottom:.25rem;
             text-transform:uppercase;letter-spacing:.05em}
    .f input[type="number"],.f input[type="url"],.f input[type="text"],.f input[type="password"],
    .f select,.f textarea{
      width:100%;padding:.48rem .65rem;font-family:var(--font);font-size:.85rem;color:var(--tx);
      background:var(--bg-i);border:1px solid var(--bd);border-radius:var(--r-s);outline:none;
      transition:border-color var(--tr),box-shadow var(--tr)}
    .f input:focus,.f select:focus,.f textarea:focus{
      border-color:var(--bdf);box-shadow:0 0 0 3px rgba(212,145,78,.18)}
    .f textarea{resize:vertical;min-height:2.8rem}
    .f .sub{font-size:.7rem;color:var(--txm);margin-top:.15rem;line-height:1.35}
    .pair{display:flex;gap:.7rem}.pair>.f{flex:1}

    /* checks */
    .ck{display:flex;gap:1rem;flex-wrap:wrap}
    .ck label{display:flex;align-items:center;gap:.3rem;font-size:.8rem;font-weight:500;
              color:var(--tx);cursor:pointer;text-transform:none;letter-spacing:normal}
    .ck input[type="checkbox"]{width:14px;height:14px;accent-color:var(--ac)}

    /* buttons */
    .br{display:flex;gap:.55rem;margin-top:1rem}
    .bt{padding:.5rem 1.2rem;font-family:var(--font);font-size:.82rem;font-weight:600;border:none;
        border-radius:var(--r-s);cursor:pointer;transition:background var(--tr),transform .08s ease}
    .bt:active{transform:scale(.97)}
    .bt-go{background:var(--ac);color:var(--ac-tx)}
    .bt-go:hover{background:var(--ac-h)}
    .bt-go:disabled{opacity:.45;cursor:not-allowed}
    .bt-st{background:var(--dg);color:#fff}
    .bt-st:hover{background:var(--dg-h)}
    .bt-st:disabled{opacity:.45;cursor:not-allowed}

    /* collapse */
    .cb{background:none;border:none;color:var(--tx2);font-family:var(--font);font-size:.72rem;
        font-weight:600;cursor:pointer;padding:0;margin-bottom:.55rem;display:flex;
        align-items:center;gap:.25rem;text-transform:uppercase;letter-spacing:.04em}
    .cb:hover{color:var(--ac)}
    .cb .a{transition:transform var(--tr);display:inline-block;font-size:.65rem}
    .cb.o .a{transform:rotate(90deg)}
    .cx{display:none}.cx.o{display:block}

    /* right panel */
    .oc{opacity:.5;transition:opacity .3s ease}.oc.on{opacity:1}
    .oh{display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem}
    .oh h2{font-size:.88rem;font-weight:600;margin:0}
    .pill{display:inline-block;padding:.15rem .5rem;font-size:.65rem;font-weight:700;
          text-transform:uppercase;letter-spacing:.06em;border-radius:999px;color:#fff}
    .pill-idle{background:var(--txm)}.pill-running{background:var(--b-run)}
    .pill-done{background:var(--b-ok)}.pill-failed{background:var(--b-fail)}
    .pill-stopped{background:var(--b-stop)}

    .log{background:var(--bg-log);border:1px solid var(--bd);border-radius:var(--r-m);
         padding:.7rem .85rem;font-family:var(--mono);font-size:.72rem;line-height:1.6;
         white-space:pre-wrap;word-break:break-all;min-height:180px;max-height:calc(100vh - 180px);
         overflow-y:auto;color:var(--tx);transition:background var(--tr),border-color var(--tr)}
    .log-ph{color:var(--txm);font-family:var(--font);font-size:.8rem;text-align:center;padding:3rem 1rem}

    .X{display:none!important}

    /* ref table */
    .ref{margin-top:1.25rem}
    .ref h2{font-size:.92rem;font-weight:600;margin:0 0 .7rem}
    .ref table{width:100%;border-collapse:collapse;font-size:.78rem}
    .ref th{text-align:left;font-weight:600;color:var(--tx2);padding:.4rem .6rem;border-bottom:2px solid var(--bd);text-transform:uppercase;letter-spacing:.04em;font-size:.7rem}
    .ref td{padding:.38rem .6rem;border-bottom:1px solid var(--bd);vertical-align:top}
    .ref td:first-child{font-weight:500;white-space:nowrap;color:var(--ac)}
    .ref tr:last-child td{border-bottom:none}

    /* footer */
    .ft{margin-top:2.5rem;text-align:center;font-size:.75rem;color:var(--txm)}
    .ft a{color:var(--txm);text-decoration:none;transition:color var(--tr)}
    .ft a:hover{color:var(--ac)}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hdr">
      <div style="display:flex;align-items:center;gap:.6rem;">
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--ac)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
          <polyline points="2 12 12 17 22 12"></polyline>
          <polyline points="2 17 12 22 22 17"></polyline>
        </svg>
        <h1><em>SAM</em> Batch Annotation</h1>
      </div>
      <button class="thm" id="thm" title="Toggle theme" aria-label="Toggle theme">&#9790;</button>
    </div>
    <div class="g">

      <!-- LEFT -->
      <form id="fm" class="c">
        <div class="pair">
          <div class="f"><label for="project_id">Project ID *</label>
            <input type="number" id="project_id" name="project_id" required min="1" placeholder="1"></div>
          <div class="f"><label for="backend">Backend</label>
            <select id="backend" name="backend">
              <option value="sam3">SAM3</option>
              <option value="sam21">SAM2.1 (exp)</option>
            </select></div>
        </div>
        <div class="f"><label for="ml_backend_url">ML Backend URL *</label>
          <input type="url" id="ml_backend_url" name="ml_backend_url" required
                 value="http://sam3-image-backend:9090"></div>
        <div class="f" id="tpR"><label for="text_prompt">Text Prompt *</label>
          <textarea id="text_prompt" name="text_prompt" rows="2"
                    placeholder="e.g. cow, grass, fence"></textarea>
          <div class="sub">Describes what to segment. Required for SAM3.</div></div>
        <div class="f X" id="s2R"><label for="sam21_mode">SAM2.1 Mode</label>
          <select id="sam21_mode" name="sam21_mode">
            <option value="">--</option><option value="grid">grid (3x3)</option>
          </select></div>
        <div class="pair">
          <div class="f"><label for="confidence">Confidence</label>
            <input type="number" id="confidence" name="confidence" value="0.5"
                   min="0" max="1" step="0.05"></div>
          <div class="f"><label for="max_tasks">Max tasks</label>
            <input type="number" id="max_tasks" name="max_tasks" placeholder="all"></div>
        </div>
        <button type="button" class="cb" id="tB"><span class="a">&#9654;</span> Specific Tasks</button>
        <div class="cx" id="tX">
          <div class="f"><label for="task_ids">Task IDs</label>
            <input type="text" id="task_ids" name="task_ids" placeholder="e.g. 1, 3, 17">
            <div class="sub">Specify specific tasks. E.g. <code>1, 3, 17</code>. Filter applies before 'Max tasks'.</div></div>
        </div>
        <div class="f"><label>Options</label>
          <div class="ck">
            <label><input type="checkbox" name="dry_run" value="1"> Dry run</label>
            <label><input type="checkbox" name="force" value="1"> Force overwrite</label>
          </div></div>
        <button type="button" class="cb" id="aB"><span class="a">&#9654;</span> Auth</button>
        <div class="cx" id="aX">
          <div class="pair">
            <div class="f"><label for="basic_auth_user">Username</label>
              <input type="text" id="basic_auth_user" name="basic_auth_user" placeholder="optional"></div>
            <div class="f"><label for="basic_auth_pass">Password</label>
              <input type="password" id="basic_auth_pass" name="basic_auth_pass"></div>
          </div></div>
        <div class="br">
          <button type="submit" class="bt bt-go" id="bGo">Start Batch</button>
          <button type="button" class="bt bt-st X" id="bSt">Stop</button>
        </div>
      </form>

      <!-- RIGHT -->
      <div class="c oc" id="oC">
        <div class="oh"><h2>Output</h2><span class="pill pill-idle" id="pill">idle</span></div>
        <div class="log" id="lB"><div class="log-ph">No jobs running.<br>Submit a batch to see output here.</div></div>
      </div>

    </div>

    <!-- Field reference -->
    <div class="c ref">
      <h2>Field Reference</h2>
      <table>
        <thead><tr><th>Field</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td>Project ID</td><td>Label Studio project ID (visible in the URL: <code>/projects/{id}</code>). Required.</td></tr>
          <tr><td>Backend</td><td>ML backend type. <b>SAM3</b> uses text prompts for segmentation (recommended). <b>SAM2.1</b> uses geometric grid points (experimental).</td></tr>
          <tr><td>ML Backend URL</td><td>Full URL of the ML backend's <code>/predict</code> endpoint. Default points to the Docker Compose service name.</td></tr>
          <tr><td>Text Prompt</td><td>Free-form text describing objects to segment (e.g. <code>cow, grass, fence</code>). SAM3 uses this as the sole input context. Required for SAM3; ignored for SAM2.1.</td></tr>
          <tr><td>SAM2.1 Mode</td><td>Only visible when SAM2.1 is selected. <code>grid</code> sends a 3&times;3 grid of keypoints per image.</td></tr>
          <tr><td>Confidence</td><td>Score threshold (0&ndash;1). Predictions below this value are discarded. Lower values yield more masks but may include noise.</td></tr>
          <tr><td>Max tasks</td><td>Maximum number of tasks to process. Leave empty to process all tasks in the project.</td></tr>
          <tr><td>Task IDs</td><td>Comma-separated list of specific task IDs to process (e.g. <code>1, 3, 17</code>).</td></tr>
          <tr><td>Dry run</td><td>Simulates the batch without calling the ML backend or writing predictions. Useful for verifying task counts and config.</td></tr>
          <tr><td>Force overwrite</td><td>Process tasks even if they already have human annotations. Predictions <b>coexist</b> with existing annotations (nothing is deleted).</td></tr>
          <tr><td>Auth (Username / Password)</td><td>Optional HTTP Basic Auth credentials for the ML backend. Maps to <code>BASIC_AUTH_USER</code> / <code>BASIC_AUTH_PASS</code> in <code>.env.ml</code>.</td></tr>
          <tr><td>Stop</td><td>Terminates a running batch job (sends SIGTERM, then SIGKILL after 5 s). Only visible while a job is running.</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Footer -->
    <div class="ft">
      <a href="https://github.com/felimet/label-anything-sam" target="_blank" rel="noopener noreferrer">label-anything-sam</a> &copy; 2026 Jia-Ming Zhou &middot; Apache-2.0 License
    </div>

  </div>
  <script>
    const $=id=>document.getElementById(id);
    const fm=$('fm'),oC=$('oC'),lB=$('lB'),pill=$('pill'),
          bGo=$('bGo'),bSt=$('bSt'),be=$('backend'),
          tpR=$('tpR'),tp=$('text_prompt'),s2R=$('s2R'),
          thm=$('thm'),aB=$('aB'),aX=$('aX'),tB=$('tB'),tX=$('tX');
    let po=null,cj=null;

    /* theme */
    const pr=()=>localStorage.getItem('bt')||(matchMedia('(prefers-color-scheme:light)').matches?'light':'dark');
    const st=t=>{document.documentElement.setAttribute('data-theme',t);thm.innerHTML=t==='dark'?'&#9788;':'&#9790;';localStorage.setItem('bt',t)};
    st(pr());
    thm.onclick=()=>st(document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark');

    /* collapse */
    aB.onclick=()=>{aB.classList.toggle('o');aX.classList.toggle('o')};
    tB.onclick=()=>{tB.classList.toggle('o');tX.classList.toggle('o')};

    /* backend toggle */
    const sy=()=>{const s3=be.value==='sam3';tpR.classList.toggle('X',!s3);tp.required=s3;s2R.classList.toggle('X',s3)};
    be.onchange=sy;sy();

    /* badge */
    const bg=s=>{pill.textContent=s;pill.className='pill pill-'+(s||'idle')};

    /* start */
    fm.onsubmit=async e=>{
      e.preventDefault();
      oC.classList.add('on');lB.textContent='Starting batch\u2026';bg('running');
      bGo.disabled=true;bSt.classList.remove('X');bSt.disabled=false;
      try{
        const r=await fetch('/batch',{method:'POST',body:new FormData(fm)});
        if(!r.ok){const e=await r.json().catch(()=>({}));lB.textContent='ERROR '+r.status+': '+JSON.stringify(e.detail||e);bg('failed');bGo.disabled=false;bSt.classList.add('X');return}
        const{job_id}=await r.json();cj=job_id;lB.textContent='Job '+job_id+' started.\n';dp(job_id);
      }catch(x){lB.textContent='Network error: '+x.message;bg('failed');bGo.disabled=false;bSt.classList.add('X')}
    };

    /* poll */
    function dp(j){
      po=setInterval(async()=>{
        try{
          const s=await(await fetch('/batch/'+j+'/status')).json();
          bg(s.status);lB.textContent=(s.log||[]).join('\n');lB.scrollTop=lB.scrollHeight;
          if(s.status!=='running'){clearInterval(po);po=null;bGo.disabled=false;bSt.classList.add('X');cj=null}
        }catch(_){}
      },1500);
    }

    /* stop */
    bSt.onclick=async()=>{if(!cj)return;bSt.disabled=true;try{await fetch('/batch/'+cj+'/stop',{method:'POST'})}catch(_){}};
  </script>
</body>
</html>"""


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


@app.post("/batch")
async def start_batch(
    project_id: Annotated[int, Form()],
    ml_backend_url: Annotated[str, Form()],
    backend: Annotated[str, Form()] = "sam3",
    sam21_mode: Annotated[str, Form()] = "",
    confidence: Annotated[float, Form()] = 0.5,
    max_tasks: Annotated[int | None, Form()] = None,
    dry_run: Annotated[str, Form()] = "",
    force: Annotated[str, Form()] = "",
    basic_auth_user: Annotated[str, Form()] = "",
    basic_auth_pass: Annotated[str, Form()] = "",
    text_prompt: Annotated[str, Form()] = "",
    task_ids: Annotated[str, Form()] = "",
) -> JSONResponse:
    # Validate LABEL_STUDIO_API_KEY present before spawning
    if not os.environ.get("LABEL_STUDIO_API_KEY"):
        raise HTTPException(
            status_code=422,
            detail="LABEL_STUDIO_API_KEY environment variable is not set on the server",
        )

    # SAM3 requires text_prompt
    if backend == "sam3" and not text_prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="Text Prompt is required for SAM3 backend",
        )

    cmd = [
        sys.executable,
        "scripts/batch_annotate.py",
        "--project-id", str(project_id),
        "--backend", backend,
        "--ls-url", os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        "--backend-url", ml_backend_url,
    ]

    if text_prompt.strip():
        cmd += ["--text-prompt", text_prompt.strip()]
    if task_ids.strip():
        cmd += ["--task-ids", task_ids.strip()]
    if backend == "sam21" and sam21_mode == "grid":
        cmd += ["--sam21-mode", "grid"]
    cmd += ["--confidence", str(confidence)]
    if max_tasks:
        cmd += ["--max-tasks", str(max_tasks)]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")
        cmd.append("--confirm-force")
    if basic_auth_user:
        cmd += ["--basic-auth-user", basic_auth_user, "--basic-auth-pass", basic_auth_pass]

    # LABEL_STUDIO_API_KEY inherited from env — never exposed in cmd args
    env = {**os.environ}

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "log": [], "exit_code": None, "proc": proc}
    asyncio.create_task(_collect_output(job_id, proc))

    return JSONResponse({"job_id": job_id})


@app.get("/batch/{job_id}/status")
async def job_status(job_id: str) -> JSONResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "log": job["log"],
            "exit_code": job.get("exit_code"),
        }
    )


@app.post("/batch/{job_id}/stop")
async def stop_job(job_id: str) -> JSONResponse:
    """Terminate a running batch job.

    Sends SIGTERM first (graceful); child process has 5 s to exit before
    SIGKILL. On Windows, terminate() is equivalent to kill().
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "running":
        return JSONResponse({"job_id": job_id, "status": job["status"], "detail": "not running"})

    proc: subprocess.Popen | None = job.get("proc")
    if proc is None:
        return JSONResponse({"job_id": job_id, "status": job["status"], "detail": "no process ref"})

    # Graceful -> hard kill fallback
    try:
        proc.terminate()  # SIGTERM on Unix, TerminateProcess on Windows
    except OSError:
        pass

    # Give 5 s for graceful exit, then force kill
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, proc.wait),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()  # SIGKILL
        except OSError:
            pass

    job["status"] = "stopped"
    job["exit_code"] = proc.returncode
    job["log"].append("[batch-server] Job stopped by user.")

    return JSONResponse({"job_id": job_id, "status": "stopped"})
