# ============================================================
# SAM3 ML Backend — Multi-stage CUDA build
# Base: NVIDIA CUDA 12.4 + cuDNN (matches torch cu124 wheels)
# ============================================================

# ── Stage 1: dependency builder ─────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    WORKERS=1 \
    THREADS=8 \
    DEVICE=cuda \
    # Hugging Face cache location inside container
    HF_HOME=/hf-cache \
    TRANSFORMERS_CACHE=/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/dist-packages \
                    /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App files
WORKDIR /app
COPY . .

# Non-root user for security
RUN useradd -m -u 1001 appuser \
    && mkdir -p /hf-cache \
    && chown -R appuser:appuser /app /hf-cache

USER appuser

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=5 \
    CMD curl -sf http://localhost:9090/health || exit 1

# Single worker (model is not thread-safe for concurrent inference)
# Threads handle concurrent non-inference requests (health, setup)
CMD ["gunicorn", \
     "--preload", \
     "--bind", "0.0.0.0:9090", \
     "--workers", "1", \
     "--threads", "8", \
     "--worker-class", "gthread", \
     "--timeout", "0", \
     "--graceful-timeout", "30", \
     "--log-level", "info", \
     "_wsgi:app"]
