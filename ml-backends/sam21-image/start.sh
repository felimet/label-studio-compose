#!/bin/bash
# SAM2.1 image backend entrypoint
#
# GPU assignment:
#   SAM21_IMAGE_GPU_INDEX (set in .env.ml) — comma-separated nvidia-smi GPU index(es)
#   to expose to this container (e.g. "0", "0,1").  Empty = all GPUs visible.
#
# Worker count:
#   SAM21_IMAGE_WORKERS (set in .env.ml) — gunicorn workers for this service.
#   Falls back to WORKERS (Dockerfile default 1) if SAM21_IMAGE_WORKERS is unset.
#
# NOTE: Do NOT use --preload. CUDA must be initialised inside each worker
# process (after fork), not in the gunicorn master.

# ── GPU pinning ───────────────────────────────────────────────────────────────
if [ -n "${SAM21_IMAGE_GPU_INDEX:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SAM21_IMAGE_GPU_INDEX}"
fi

exec gunicorn \
  --config /app/gunicorn.conf.py \
  --bind ":${PORT:-9090}" \
  --workers "${SAM21_IMAGE_WORKERS:-${WORKERS:-1}}" \
  --threads "${THREADS:-8}" \
  --worker-class gthread \
  --timeout "${TIMEOUT:-120}" \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  _wsgi:app
