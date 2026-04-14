# ============================================================
# SAM2.1 Video Backend
# Base: pytorch/pytorch (includes torch + CUDA + Python 3.12)
# Build: single-stage with BuildKit pip cache mounts
#
# Requirements: NVIDIA driver >= 535.x (CUDA 12.6); minimum GPU: Turing (sm_75+)
# Build: DOCKER_BUILDKIT=1 docker compose build sam21-video-backend
#
# Checkpoints (~10 GB total for all 4 models):
#   Primary:  huggingface_hub (facebook/sam2.1-hiera-*)
#   Fallback: download_ckpts.sh (Meta CDN) when HF is unreachable
#
# Note: sam21-image and sam21-video share the same model-cache volume.
#       The first service to build will download all checkpoints; the
#       second will skip (files already exist in the volume).
# ============================================================

# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    PORT=9090 \
    WORKERS=1 \
    THREADS=8 \
    TIMEOUT=300 \
    DEVICE=cuda \
    LOG_LEVEL=INFO \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    MODEL_DIR=/data/models \
    MODEL_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml \
    MODEL_CHECKPOINT=sam2.1_hiera_large.pt \
    MAX_FRAMES_TO_TRACK=10 \
    MAX_FRAME_LONG_SIDE=1024

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ \
        ffmpeg \
        libglib2.0-0 libgl1 libsm6 libxrender1 libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Base deps: gunicorn + label-studio-ml (git) ──────────────
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# ── App deps (includes opencv for frame extraction) ───────────
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# ── SAM2 from source ─────────────────────────────────────────
ARG SAM2_COMMIT=main
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    git clone --depth 1 --branch ${SAM2_COMMIT} \
        https://github.com/facebookresearch/sam2.git /sam2_src \
    && pip install -e /sam2_src \
    && rm -rf /sam2_src/.git /sam2_src/notebooks /sam2_src/demo

# ── Download all 4 SAM2.1 checkpoints ────────────────────────
# Primary path:  huggingface_hub → /data/models/
# Fallback path: download_ckpts.sh (Meta CDN) when HF is unreachable
#
# HF_TOKEN is optional — SAM2.1 repos are public.
# Providing a token enables higher HF rate limits and avoids the
# "unauthenticated request" notice.
#
# Recommended: use `make build-sam21-video` — it auto-reads HF_TOKEN
# from .env.ml so you don't need to pass --build-arg manually.
# Manual override: docker compose build --build-arg HF_TOKEN=<token> sam21-video-backend
COPY download_models.py .
ARG HF_TOKEN=""
# HF_HUB_VERBOSITY=error suppresses the unauthenticated-request notice.
# SAM2.1 repos are public; a token is not required but raises rate limits.
RUN mkdir -p /data/models && \
    HF_TOKEN="${HF_TOKEN}" HF_HUB_VERBOSITY=error python /app/download_models.py || \
    (echo "[fallback] HuggingFace failed, using download_ckpts.sh" && \
     cd /sam2_src && bash download_ckpts.sh && \
     mv *.pt /data/models/ && \
     echo "[fallback] checkpoints moved to /data/models/")

# ── App files ─────────────────────────────────────────────────
COPY . .
RUN chmod +x start.sh

# Non-root user
RUN useradd -m -u 1001 appuser \
    && mkdir -p /data/models /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /data /home/appuser/.cache

USER appuser

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
    CMD curl -sf http://localhost:9090/health || exit 1

CMD ["/app/start.sh"]
