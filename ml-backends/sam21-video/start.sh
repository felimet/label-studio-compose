#!/bin/bash
# SAM2.1 video backend entrypoint
#
# GPU assignment:
#   SAM21_VIDEO_GPU_INDEX (set in .env.ml) — comma-separated nvidia-smi GPU index(es).
#
# Worker count:
#   SAM21_VIDEO_WORKERS — gunicorn workers for this service.
#   Falls back to WORKERS (Dockerfile default 1) if SAM21_VIDEO_WORKERS is unset.
#
# NOTE: Do NOT use --preload.

# ── GPU pinning ───────────────────────────────────────────────────────────────
if [ -n "${SAM21_VIDEO_GPU_INDEX:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${SAM21_VIDEO_GPU_INDEX}"
fi

exec gunicorn \
  --config /app/gunicorn.conf.py \
  --bind ":${PORT:-9090}" \
  --workers "${SAM21_VIDEO_WORKERS:-${WORKERS:-1}}" \
  --threads "${THREADS:-8}" \
  --worker-class gthread \
  --timeout "${TIMEOUT:-300}" \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  _wsgi:app
