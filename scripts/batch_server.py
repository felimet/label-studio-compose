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
import hmac
import os
import subprocess
import sys
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Batch Annotation Server", version="1.0.0")

# In-memory job registry (lost on restart — acceptable, jobs complete in minutes)
_jobs: dict[str, dict] = {}


# ── Auth ─────────────────────────────────────────────────────────────────


def _check_api_key(request: Request) -> None:
    required_key = os.environ.get("BATCH_SERVER_API_KEY", "")
    if not required_key:
        return  # No key configured — open for local dev
    provided = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(provided, required_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )


AuthDep = Annotated[None, Depends(_check_api_key)]


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
    _jobs[job_id]["status"] = "done" if rc == 0 else "failed"
    _jobs[job_id]["exit_code"] = rc


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
async def index(_auth: AuthDep = None) -> HTMLResponse:
    api_key_required = bool(os.environ.get("BATCH_SERVER_API_KEY"))
    key_row = ""
    if api_key_required:
        key_row = """
        <tr>
          <td><label for="api_key">X-API-Key</label></td>
          <td><input type="password" id="api_key" name="api_key" style="width:100%"></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Batch Annotation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.4rem; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
    td {{ padding: .4rem .6rem; vertical-align: top; }}
    td:first-child {{ width: 170px; font-weight: 500; padding-top: .6rem; }}
    input, select {{ width: 100%; box-sizing: border-box; padding: .3rem; }}
    input[type=checkbox] {{ width: auto; }}
    button {{ padding: .5rem 1.5rem; font-size: 1rem; cursor: pointer; }}
    #status {{ margin-top: 1.5rem; white-space: pre-wrap; font-family: monospace;
               font-size: .85rem; background: #f4f4f4; padding: 1rem;
               max-height: 400px; overflow-y: auto; display: none; }}
  </style>
</head>
<body>
  <h1>SAM3/SAM2.1 Batch Annotation</h1>
  <form id="batchForm">
    <table>
      <tr>
        <td><label for="project_id">Project ID *</label></td>
        <td><input type="number" id="project_id" name="project_id" required min="1"></td>
      </tr>
      <tr>
        <td><label for="backend">Backend</label></td>
        <td>
          <select id="backend" name="backend">
            <option value="sam3">SAM3 (recommended)</option>
            <option value="sam21">SAM2.1 (experimental)</option>
          </select>
        </td>
      </tr>
      <tr>
        <td><label for="sam21_mode">SAM2.1 Mode</label></td>
        <td>
          <select id="sam21_mode" name="sam21_mode">
            <option value="">— (SAM3 only)</option>
            <option value="grid">grid (3×3)</option>
          </select>
        </td>
      </tr>
      <tr>
        <td><label for="confidence">Confidence</label></td>
        <td><input type="number" id="confidence" name="confidence" value="0.5"
                   min="0" max="1" step="0.05"></td>
      </tr>
      <tr>
        <td><label for="max_tasks">Max tasks</label></td>
        <td><input type="number" id="max_tasks" name="max_tasks" placeholder="(all)"></td>
      </tr>
      <tr>
        <td>Options</td>
        <td>
          <label><input type="checkbox" name="dry_run" value="1"> Dry run</label>
          &nbsp;
          <label><input type="checkbox" name="force" value="1"> Force</label>
          &nbsp;
          <label><input type="checkbox" name="confirm_force" value="1"> Confirm-force</label>
        </td>
      </tr>
      {key_row}
    </table>
    <button type="submit">Start Batch</button>
  </form>

  <div id="status"></div>

  <script>
    const form = document.getElementById('batchForm');
    const statusBox = document.getElementById('status');
    let pollInterval = null;

    form.addEventListener('submit', async (e) => {{
      e.preventDefault();
      const data = new FormData(form);
      const apiKey = data.get('api_key') || '';
      const headers = {{}};
      if (apiKey) headers['X-API-Key'] = apiKey;

      statusBox.style.display = 'block';
      statusBox.textContent = 'Starting batch…';

      const resp = await fetch('/batch', {{
        method: 'POST',
        body: data,
        headers,
      }});

      if (!resp.ok) {{
        const err = await resp.json().catch(() => ({{}}));
        statusBox.textContent = 'ERROR ' + resp.status + ': ' + JSON.stringify(err);
        return;
      }}

      const {{job_id}} = await resp.json();
      statusBox.textContent = 'Job ' + job_id + ' started.\\n';

      pollInterval = setInterval(async () => {{
        const sr = await fetch('/batch/' + job_id + '/status', {{headers}});
        const s = await sr.json();
        statusBox.textContent =
          'Status: ' + s.status + '\\n' +
          (s.log || []).join('\\n');
        statusBox.scrollTop = statusBox.scrollHeight;
        if (s.status === 'done' || s.status === 'failed') {{
          clearInterval(pollInterval);
        }}
      }}, 2000);
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/batch")
async def start_batch(
    project_id: Annotated[int, Form()],
    backend: Annotated[str, Form()] = "sam3",
    sam21_mode: Annotated[str, Form()] = "",
    confidence: Annotated[float, Form()] = 0.5,
    max_tasks: Annotated[int | None, Form()] = None,
    dry_run: Annotated[str, Form()] = "",
    force: Annotated[str, Form()] = "",
    confirm_force: Annotated[str, Form()] = "",
    _auth: AuthDep = None,
) -> JSONResponse:
    # Validate LABEL_STUDIO_API_KEY present before spawning
    if not os.environ.get("LABEL_STUDIO_API_KEY"):
        raise HTTPException(
            status_code=422,
            detail="LABEL_STUDIO_API_KEY environment variable is not set on the server",
        )

    cmd = [
        sys.executable,
        "scripts/batch_annotate.py",
        "--project-id", str(project_id),
        "--backend", backend,
        "--ls-url", os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        "--backend-url", os.environ.get("ML_BACKEND_URL", "http://localhost:9090"),
    ]

    if backend == "sam21" and sam21_mode == "grid":
        cmd += ["--sam21-mode", "grid"]
    if confidence != 0.5:
        cmd += ["--confidence", str(confidence)]
    if max_tasks:
        cmd += ["--max-tasks", str(max_tasks)]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")
    if confirm_force:
        cmd.append("--confirm-force")

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
    _jobs[job_id] = {"status": "running", "log": [], "exit_code": None}
    asyncio.create_task(_collect_output(job_id, proc))

    return JSONResponse({"job_id": job_id})


@app.get("/batch/{job_id}/status")
async def job_status(job_id: str, _auth: AuthDep = None) -> JSONResponse:
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
