from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RESUME_FILE = "scripts/.batch_resume.json"


def load_resume(resume_file: str = DEFAULT_RESUME_FILE) -> dict | None:
    """Load resume state from file. Returns None if file does not exist."""
    path = Path(resume_file)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_resume(
    task_id: int,
    project_id: int,
    resume_file: str = DEFAULT_RESUME_FILE,
) -> None:
    """Record the last successfully processed task_id."""
    state = {
        "last_task_id": task_id,
        "project_id": project_id,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
    path = Path(resume_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def clear_resume(resume_file: str = DEFAULT_RESUME_FILE) -> None:
    """Remove resume state file after a successful full run."""
    path = Path(resume_file)
    if path.exists():
        path.unlink()
