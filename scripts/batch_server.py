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
from pathlib import Path
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

_UI_HTML = Path(__file__).parent / "batch_ui.html"

def _load_ui() -> str:
    return _UI_HTML.read_text(encoding="utf-8")


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_load_ui())


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
    use_agent: Annotated[str, Form()] = "",
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
    if backend == "sam3":
        cmd.append("--use-agent" if use_agent == "1" else "--no-agent")

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
