#!/bin/bash
# SAM3 image backend entrypoint
#
# Tuning guide:
#   WORKERS  — 1 per GPU. Each worker loads the model independently.
#               Multiple workers on a single GPU cause duplicate VRAM usage.
#               Raise to N for N-GPU nodes, e.g. WORKERS=2 for dual-GPU.
#   THREADS  — threads share the same worker process and model weights;
#               safe to raise on modern CPUs. 8 is a good baseline for 8-core
#               machines; raise to 16 for 16+ core servers.
#
# NOTE: Do NOT use --preload. CUDA must be initialised inside each worker
# process (after fork), not in the gunicorn master. --preload causes the
# master to import the app first, which can trigger CUDA init via transitive
# imports, making all forked workers fail with:
#   "Cannot re-initialize CUDA in forked subprocess"
#
# Override via environment: ML_WORKERS / ML_THREADS in .env or docker-compose.
exec gunicorn \
  --config /app/gunicorn.conf.py \
  --bind ":${PORT:-9090}" \
  --workers "${WORKERS:-1}" \
  --threads "${THREADS:-8}" \
  --worker-class gthread \
  --timeout "${TIMEOUT:-120}" \
  --graceful-timeout 30 \
  --log-level "${LOG_LEVEL:-info}" \
  _wsgi:app
