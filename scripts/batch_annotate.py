#!/usr/bin/env python3
"""SAM3/SAM2.1 batch annotation CLI for Label Studio.

Usage:
    python scripts/batch_annotate.py --project-id 1 --backend sam3
    make batch-annotate PROJECT_ID=1
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import requests

# Ensure scripts/ is importable when invoked directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.constants import (
    CLI_MODEL_VERSION_SAM21,
    CLI_MODEL_VERSION_SAM3,
)
from scripts.utils.context_builder import build_context, estimate_local_vram_gb
from scripts.utils.label_parser import extract_label_names, has_brush_labels
from scripts.utils.ls_api import LabelStudioAPI
from scripts.utils.resume import load_resume, write_resume


# ── Auth ─────────────────────────────────────────────────────────────────


def load_api_key() -> str:
    key = os.environ.get("LABEL_STUDIO_API_KEY", "")
    if not key:
        print(
            "ERROR: LABEL_STUDIO_API_KEY not set. "
            "Export it before running this script.",
            file=sys.stderr,
        )
        sys.exit(3)
    return key


# ── ML Backend ───────────────────────────────────────────────────────────


def call_predict(
    ml_backend_url: str,
    task: dict,
    context: dict,
    basic_auth: tuple[str, str] | None = None,
) -> tuple[str, list, float]:
    """Call ML backend /predict endpoint.

    Returns (status, result, score) where status is one of:
      'success'       — HTTP 200, non-empty predictions
      'zero_match'    — HTTP 200, predictions=[]
      'protocol_fail' — HTTP 4xx/5xx or network error
    score defaults to 0.0 on failure or missing field.
    """
    payload = {
        "tasks": [{"id": task["id"], "data": task.get("data", {})}],
        "context": context,
    }
    try:
        resp = requests.post(
            f"{ml_backend_url.rstrip('/')}/predict",
            json=payload,
            timeout=(5, 120),
            auth=basic_auth or None,
        )
    except Exception as exc:
        return "protocol_fail", [], 0.0

    if resp.status_code >= 400:
        return "protocol_fail", [], 0.0

    data = resp.json()
    predictions = data.get("results", []) or []
    if not predictions:
        return "zero_match", [], 0.0

    first = predictions[0]
    result = first.get("result", [])
    score = float(first.get("score", 0.0))
    return "success", result, score


# ── TOCTOU guard ─────────────────────────────────────────────────────────


def safe_write_prediction(
    task_id: int,
    result: list,
    score: float,
    ls: LabelStudioAPI,
    model_version: str,
    force: bool = False,
) -> str:
    """Re-fetch task before writing; abort if human annotation appeared.

    Returns one of: 'success', 'skip_race', 'error_fetch'.

    TOCTOU scope: total_annotations only counts submitted annotations.
    Draft annotations and concurrent submits in the re-GET→write window
    are NOT protected. Do NOT describe this as 'fully prevents race conditions'.

    When force=True: skip the annotation guard but still re-GET for fresh state.
    Human annotations are NEVER deleted regardless of force flag.
    """
    try:
        fresh = ls.get_task(task_id)
    except Exception:
        return "error_fetch"  # re-GET failed; caller counts as protocol_fail

    if not force and fresh.get("total_annotations", 0) > 0:
        return "skip_race"

    ls.delete_cli_predictions(task_id, model_version=model_version)
    ls.create_prediction(task_id, result, score=score, model_version=model_version)
    return "success"


# ── Pre-flight ───────────────────────────────────────────────────────────


def pre_flight_check(
    ls: LabelStudioAPI,
    ml_backend_url: str,
    project_id: int,
    args,
) -> tuple[list[str], str]:
    """Validate environment and return (label_names, label_config).

    Exits with appropriate exit codes on unrecoverable errors.
    """
    # LS reachable
    if not ls.health_check():
        print(
            "ERROR: Cannot reach Label Studio. "
            "Check --ls-url and LABEL_STUDIO_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(4)

    # Project exists
    try:
        project = ls.get_project(project_id)
    except Exception as exc:
        print(f"ERROR: Cannot fetch project {project_id}: {exc}", file=sys.stderr)
        sys.exit(3)

    label_config: str = project.get("label_config", "") or ""
    label_names = extract_label_names(label_config)

    if not label_names:
        print(
            "WARN: No <Label value> found in project labeling config. "
            "SAM3 context will be empty.",
            file=sys.stderr,
        )

    # SAM2.1 validation: must have --sam21-mode grid
    if args.backend == "sam21":
        if not args.sam21_mode:
            print(
                "ERROR: --backend sam21 requires --sam21-mode grid. "
                "SAM2.1 has no text-prompt batch path.",
                file=sys.stderr,
            )
            sys.exit(3)
        if not has_brush_labels(label_config):
            print(
                "WARN: No <BrushLabels> found in labeling config. "
                "SAM2.1 mask output may not be stored correctly.",
                file=sys.stderr,
            )

    # SAM3: BrushLabels check
    if args.backend == "sam3" and not has_brush_labels(label_config):
        print(
            "WARN: No <BrushLabels> found in labeling config. "
            "SAM3 mask predictions may not display correctly.",
            file=sys.stderr,
        )

    # VRAM hint (advisory, not used for concurrency)
    vram = estimate_local_vram_gb()
    if vram > 0:
        print(f"INFO: Local GPU VRAM ≈ {vram} GB (hint only; backend VRAM may differ)")

    if args.concurrency > 1:
        print(
            f"WARN: --concurrency {args.concurrency} requires backend "
            f"WORKERS≥{args.concurrency}. Verify SAM3_IMAGE_WORKERS in .env.ml.",
            file=sys.stderr,
        )

    return label_names, label_config


# ── Main ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM3/SAM2.1 batch annotation CLI for Label Studio"
    )
    p.add_argument("--project-id", type=int, required=True, help="Label Studio project ID")
    p.add_argument(
        "--backend",
        choices=["sam3", "sam21"],
        default="sam3",
        help="ML backend type (default: sam3)",
    )
    p.add_argument(
        "--backend-url",
        default=os.environ.get("ML_BACKEND_URL", "http://localhost:9090"),
        help="ML backend URL (default: $ML_BACKEND_URL or http://localhost:9090)",
    )
    p.add_argument(
        "--ls-url",
        default=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
        help="Label Studio URL (default: $LABEL_STUDIO_URL or http://localhost:8080)",
    )
    p.add_argument("--dry-run", action="store_true", help="List tasks without annotating")
    p.add_argument(
        "--force",
        action="store_true",
        help="Write predictions even on tasks with human annotations (annotations never deleted)",
    )
    p.add_argument(
        "--confirm-force",
        action="store_true",
        help="Required second flag when using --force",
    )
    p.add_argument("--concurrency", type=int, default=1, help="Parallel HTTP requests (default: 1)")
    p.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks processed")
    p.add_argument("--resume", action="store_true", help="Resume from last successful task")
    p.add_argument(
        "--resume-file",
        default="scripts/.batch_resume.json",
        help="Resume state file path (default: scripts/.batch_resume.json)",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold (default: 0.5, SAM3 only)",
    )
    p.add_argument(
        "--sam21-mode",
        choices=["grid"],
        default=None,
        help="[EXPERIMENTAL] SAM2.1 annotation mode (currently only 'grid')",
    )
    p.add_argument(
        "--grid-n",
        type=int,
        default=3,
        help="SAM2.1 grid edge length (default: 3, i.e. 3×3=9 points)",
    )
    p.add_argument(
        "--basic-auth-user",
        default="",
        help="HTTP Basic Auth username for ML backend (leave empty to disable)",
    )
    p.add_argument(
        "--basic-auth-pass",
        default="",
        help="HTTP Basic Auth password for ML backend (leave empty to disable)",
    )
    return p.parse_args()


def process_task(
    task: dict,
    args,
    label_names: list[str],
    model_version: str,
    ls: LabelStudioAPI,
    basic_auth: tuple[str, str] | None = None,
) -> tuple[int, str]:
    """Process a single task. Returns (task_id, status)."""
    task_id = task["id"]

    # Skip tasks with human annotations (unless --force + --confirm-force)
    if task.get("total_annotations", 0) > 0:
        if not (args.force and args.confirm_force):
            return task_id, "skip_human"

    context = build_context(args.backend, label_names, args)
    status, result, score = call_predict(args.backend_url, task, context, basic_auth)

    if status == "protocol_fail":
        return task_id, "fail"

    if status == "zero_match":
        return task_id, "zero"

    # status == "success"
    write_status = safe_write_prediction(
        task_id, result, score, ls, model_version, force=args.force
    )
    if write_status == "error_fetch":
        return task_id, "fail"
    return task_id, write_status  # "success" or "skip_race"


def main() -> None:
    args = parse_args()

    # --force requires --confirm-force
    if args.force and not args.confirm_force:
        print(
            "ERROR: --force requires --confirm-force as a second confirmation flag.",
            file=sys.stderr,
        )
        sys.exit(3)

    api_key = load_api_key()
    ls = LabelStudioAPI(args.ls_url, api_key)
    basic_auth: tuple[str, str] | None = (
        (args.basic_auth_user, args.basic_auth_pass)
        if args.basic_auth_user
        else None
    )

    label_names, _ = pre_flight_check(ls, args.backend_url, args.project_id, args)

    model_version = (
        CLI_MODEL_VERSION_SAM3 if args.backend == "sam3" else CLI_MODEL_VERSION_SAM21
    )

    # SAM2.1 EXPERIMENTAL banner
    if args.backend == "sam21" and args.sam21_mode == "grid":
        print(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  [EXPERIMENTAL] SAM2.1 Grid Mode                             ║\n"
            "║  • Returns 1 mask per image (argmax of scores)               ║\n"
            "║  • Not suitable for multi-object or empty scenes             ║\n"
            "║  • Recommended: use --backend sam3 for production use        ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )

    # Resume: find last processed task_id
    resume_from: int | None = None
    if args.resume:
        state = load_resume(args.resume_file)
        if state and state.get("project_id") == args.project_id:
            resume_from = state.get("last_task_id")
            print(f"INFO: Resuming from task_id > {resume_from}")

    # Fetch tasks
    try:
        all_tasks = list(ls.list_tasks(args.project_id))
    except Exception as exc:
        print(f"ERROR: Cannot list tasks: {exc}", file=sys.stderr)
        sys.exit(1)

    # Apply resume filter
    if resume_from is not None:
        all_tasks = [t for t in all_tasks if t["id"] > resume_from]

    # Apply max-tasks limit
    if args.max_tasks is not None:
        all_tasks = all_tasks[: args.max_tasks]

    if args.dry_run:
        print(f"\nDRY RUN — {len(all_tasks)} tasks would be processed:")
        for t in all_tasks:
            ann = t.get("total_annotations", 0)
            pred = t.get("total_predictions", 0)
            flag = "[human]" if ann > 0 else ("[pred]" if pred > 0 else "[empty]")
            print(f"  task {t['id']:>6}  {flag}")
        return

    # Counters
    counts: dict[str, int] = {
        "success": 0,
        "skip_human": 0,
        "skip_race": 0,
        "zero": 0,
        "fail": 0,
    }

    print(f"\nProcessing {len(all_tasks)} tasks with {args.concurrency} worker(s)…\n")

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(all_tasks), unit="task", dynamic_ncols=True)
    except ImportError:
        progress = None

    def _update_progress(task_id: int, status: str) -> None:
        counts[status] = counts.get(status, 0) + 1
        if progress is not None:
            progress.set_postfix(
                success=counts["success"],
                skip=counts["skip_human"] + counts["skip_race"],
                zero=counts["zero"],
                fail=counts["fail"],
            )
            progress.update(1)
        else:
            total_done = sum(counts.values())
            pct = int(total_done / len(all_tasks) * 100)
            print(
                f"[{total_done}/{len(all_tasks)}] {pct}% | "
                f"✓ success={counts['success']} | "
                f"⊘ skip={counts['skip_human']+counts['skip_race']} | "
                f"○ zero={counts['zero']} | ✗ fail={counts['fail']}",
                end="\r",
            )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_task, task, args, label_names, model_version, ls, basic_auth
            ): task["id"]
            for task in all_tasks
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                tid, status = future.result()
            except Exception as exc:
                print(f"\nERROR task {task_id}: {exc}", file=sys.stderr)
                status = "fail"
                tid = task_id

            _update_progress(tid, status)

            if status in ("success", "skip_race"):
                write_resume(tid, args.project_id, args.resume_file)

            if status == "skip_human":
                print(f"\n  [SKIP human] task {tid}", end="")
            elif status == "skip_race":
                print(f"\n  [SKIP race]  task {tid}", end="")
            elif status == "fail":
                print(f"\n  [FAIL]       task {tid}", file=sys.stderr, end="")
            elif status == "zero":
                print(f"\n  [ZERO]       task {tid}", end="")

    if progress is not None:
        progress.close()

    # Summary
    print(
        f"\n\n─── Summary ───────────────────────────────────────────────\n"
        f"  ✓ success    : {counts['success']}\n"
        f"  ⊘ skip_human : {counts['skip_human']}\n"
        f"  ⊘ skip_race  : {counts['skip_race']}\n"
        f"  ○ zero_match : {counts['zero']}\n"
        f"  ✗ fail       : {counts['fail']}\n"
        f"───────────────────────────────────────────────────────────"
    )

    # Zero-rate warning
    total_attempted = counts["success"] + counts["zero"] + counts["fail"]
    if total_attempted > 0 and counts["zero"] / total_attempted > 0.8:
        print(
            "\nWARN: >80% of tasks returned zero predictions. "
            "Check label names match labeling config, or lower --confidence.",
            file=sys.stderr,
        )

    # Exit codes
    if counts["fail"] > 0 and counts["success"] == 0:
        sys.exit(2)  # all failed
    elif counts["fail"] > 0:
        sys.exit(1)  # partial failures
    sys.exit(0)


if __name__ == "__main__":
    main()
