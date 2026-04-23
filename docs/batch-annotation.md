# Batch Annotation Guide

Use `scripts/batch_annotate.py` (or the Web UI at port 8085) to run
SAM3/SAM2.1 inference on an entire Label Studio project without opening each
task manually.

---

## Prerequisites

| Requirement | Check |
|---|---|
| ML backend running | `curl http://localhost:9090/health` |
| Label Studio running | `curl http://localhost:8080/health` |
| `LABEL_STUDIO_API_KEY` set | `echo $LABEL_STUDIO_API_KEY` |
| Project labeling config has `<BrushLabels>` | Label Studio UI → Settings → Labeling |

---

## Quick Start

### CLI

```bash
# Dry run — list tasks without annotating
python scripts/batch_annotate.py --project-id 1 --backend sam3 --dry-run

# Annotate all tasks
python scripts/batch_annotate.py --project-id 1 --backend sam3

# Makefile shortcut
make batch-annotate PROJECT_ID=1
```

### Web UI (no terminal required)

Start the companion server:

```bash
uvicorn scripts.batch_server:app --host 0.0.0.0 --port 8085
# or
make batch-server
```

Then open `http://<your-server>:8085` in a browser and fill in the form.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--project-id INT` | (required) | Label Studio project ID |
| `--backend sam3\|sam21` | `sam3` | ML backend type |
| `--backend-url URL` | `$ML_BACKEND_URL` or `http://localhost:9090` | ML backend endpoint |
| `--ls-url URL` | `$LABEL_STUDIO_URL` or `http://localhost:8080` | Label Studio endpoint |
| `--dry-run` | off | List tasks without calling ML backend |
| `--force` | off | Write predictions even on tasks with human annotations |
| `--confirm-force` | off | Required second flag when using `--force` |
| `--concurrency N` | `1` | Parallel HTTP requests |
| `--max-tasks INT` | (all) | Limit number of tasks processed |
| `--resume` | off | Continue from last successful task |
| `--resume-file PATH` | `scripts/.batch_resume.json` | Resume state file path |
| `--confidence FLOAT` | `0.5` | SAM3 confidence threshold (SAM3 only) |
| `--sam21-mode grid` | off | Enable SAM2.1 Grid mode (EXPERIMENTAL) |
| `--grid-n INT` | `3` | SAM2.1 grid edge length (NxN points) |

---

## Authentication

API key is read **exclusively** from the `LABEL_STUDIO_API_KEY` environment
variable. It is never logged, never passed as a CLI argument.

```bash
export LABEL_STUDIO_API_KEY=your-token-here
python scripts/batch_annotate.py --project-id 1
```

---

## SAM3 vs SAM2.1

| Feature | SAM3 (`--backend sam3`) | SAM2.1 (`--backend sam21`) |
|---|---|---|
| Context type | Text prompt (label names) | Grid of keypoints (geometry) |
| Multi-object | Yes (one mask per label) | No (1 mask per image, argmax) |
| Accuracy | Higher | Lower (geometry heuristic) |
| Activation | Default | `--sam21-mode grid` required |
| Recommended | Production | Single-dominant-object scenes |

> **SAM2.1 is EXPERIMENTAL.** It returns exactly **one mask per image**
> (the highest-scoring point). It is not suitable for multi-object or empty scenes.

---

## Concurrency

```bash
# Run 4 parallel requests (requires SAM3_IMAGE_WORKERS=4 in .env.ml)
python scripts/batch_annotate.py --project-id 1 --concurrency 4
```

Backend `WORKERS` controls actual parallelism. If `WORKERS=1` (the default),
setting `--concurrency > 1` queues requests without throughput gain.

Check your `.env.ml`:
```
SAM3_IMAGE_WORKERS=4
```

---

## Task Status Classification

| Status | Meaning |
|---|---|
| `success` | Prediction written to Label Studio |
| `skip_human` | Task has human annotations; skipped (use `--force` to override) |
| `skip_race` | Annotation appeared between list and write (TOCTOU window) |
| `zero_match` | ML backend returned no predictions (check label names / confidence) |
| `fail` | HTTP error or network failure |

> **`--force` safety**: When `--force --confirm-force` is used, predictions are
> written alongside existing human annotations. **Human annotations are never
> deleted.** This is a parallel write, not an overwrite.

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | All tasks succeeded (zero_match counts as success) |
| `1` | Partial failures |
| `2` | All tasks failed |
| `3` | Configuration error (missing args, API key not set) |
| `4` | Authentication failure (invalid API key) |

---

## Resume

If a batch is interrupted:

```bash
# Resume from last successfully processed task
python scripts/batch_annotate.py --project-id 1 --resume
```

State is stored in `scripts/.batch_resume.json` (excluded from git commits
but retained locally across sessions).

---

## Web UI (batch_server)

The companion FastAPI server exposes a browser form at `http://<host>:8085`:

```
GET  /                      → HTML form
POST /batch                 → Start batch, returns {"job_id": "..."}
GET  /batch/{job_id}/status → Poll progress and log
GET  /health                → {"status": "ok"}
```

**Authentication** (optional): Set `BATCH_SERVER_API_KEY` environment variable.
All requests then require an `X-API-Key: <key>` header.

**Docker Compose integration** (optional — append to existing
`docker-compose.override.yml`):

```yaml
services:
  batch-server:
    build:
      context: .
      dockerfile: scripts/Dockerfile.batch-server
    ports:
      - "8085:8085"
    environment:
      - LABEL_STUDIO_URL=http://label-studio:8080
      - ML_BACKEND_URL=http://sam3-image:9090
    env_file: [.env]
```

---

## Troubleshooting

### All tasks return `zero_match`

1. Check label names match `<Label value="...">` in the labeling config exactly
2. Lower `--confidence` (e.g. `--confidence 0.3`)
3. Verify the ML backend is running: `curl http://localhost:9090/health`
4. Run with a single task: `--max-tasks 1 --dry-run` then without `--dry-run`

### `skip_race` on every task

Another process or user is annotating the same project concurrently.
**Only one batch CLI process should run per project at a time.**

### Connection refused

- `--ls-url` or `--backend-url` points to wrong host/port
- Ensure both services are running before starting the batch

---

## Warnings

- **Do not run multiple batch processes against the same project simultaneously.**
  `delete_cli_predictions()` uses `model_version` scoping, but concurrent deletes
  can race and remove each other's freshly written predictions.
- **Do not run on a project where users are actively annotating** unless using
  `--force`. The TOCTOU window (between list-tasks and write-prediction) is small
  but non-zero. Draft annotations (not yet submitted) are not protected.
