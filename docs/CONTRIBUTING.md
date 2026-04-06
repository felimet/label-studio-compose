# Contributing Guide

## Prerequisites

| Tool | Min version | Notes |
|------|-------------|-------|
| Docker Engine | 26.x | With Compose v2 plugin |
| NVIDIA driver | 535.x | CUDA 12.6 support (SAM3 only) |
| nvidia-container-toolkit | latest | GPU passthrough to containers |
| Python | 3.12 | For running tests locally |
| git | 2.x | — |

## Local Development Setup

```bash
# Clone
git clone https://github.com/felimet/label-studio-compose
cd label-studio-compose

# Configure environment
cp .env.example .env
# Edit .env — fill in all <PLACEHOLDER> values
# Minimum for core stack (no SAM3):
#   POSTGRES_PASSWORD, REDIS_PASSWORD, MINIO_ROOT_USER,
#   MINIO_ROOT_PASSWORD, LABEL_STUDIO_SECRET_KEY,
#   LABEL_STUDIO_PASSWORD, LABEL_STUDIO_USER_TOKEN
# NOTE: LABEL_STUDIO_USER_TOKEN must be ≤40 chars — use: openssl rand -hex 20

# For SAM3 ML backends (optional — GPU required):
cp .env.ml.example .env.ml
# Edit .env.ml — fill in LABEL_STUDIO_API_KEY and HF_TOKEN at minimum

# Start core stack (exposed on dev ports — see docker-compose.override.yml)
make up
make init-minio       # first time only

# Verify
make health
```

Dev override ports ([docker-compose.override.yml](../docker-compose.override.yml)):

<!-- AUTO-GENERATED from docker-compose.override.yml -->
| Service | Host port |
|---------|-----------|
| nginx | 18090 |
| label-studio | 18086 |
| minio API | 19000 |
| minio console | 19001 |
| postgres | 5433 |
| redis | 6380 |
<!-- END AUTO-GENERATED -->

> **Windows 注意**：8000–9000 附近的 port 常被 Hyper-V 保留；若 bind 失敗改用 18000+ 範圍。

## Available Commands

<!-- AUTO-GENERATED from Makefile -->
| Command | Description |
|---------|-------------|
| `make up` | Start core stack (postgres · redis · minio · label-studio · nginx · cloudflared) |
| `make down` | Stop core stack |
| `make restart` | Restart all core services |
| `make logs` | Follow logs (last 100 lines) |
| `make ps` | Show container status |
| `make ml-up` | Start core stack + SAM3 image + video backends (GPU required) |
| `make ml-down` | Stop all services (core + SAM3) |
| `make build-sam3-image` | Build SAM3 image backend Docker image |
| `make build-sam3-video` | Build SAM3 video backend Docker image |
| `make test-sam3-image` | Run pytest inside sam3-image-backend container |
| `make test-sam3-video` | Run pytest inside sam3-video-backend container |
| `make init-minio` | One-time: create S3 bucket + CORS policy (minio-init container) |
| `make create-admin` | Create Label Studio superuser (interactive) |
| `make health` | Run full stack health check |
| `make push` | git add -A + interactive commit + push origin main |
<!-- END AUTO-GENERATED -->

## Testing

### SAM3 backend tests (no GPU required)

Tests mock all model weights — runnable on CPU locally:

```bash
# Image backend
cd ml-backends/sam3-image
python -m pytest tests/ --tb=short -v

# Video backend
cd ml-backends/sam3-video
python -m pytest tests/ --tb=short -v
```

Install test dependencies:

```bash
pip install label-studio-ml label-studio-sdk label-studio-converter \
            Pillow numpy torch pytest
```

### Running inside containers

```bash
make test-sam3-image   # requires ml-up to be running
make test-sam3-video
```

## Code Style

No enforced linter is configured at repo level. Follow:

- **Python**: PEP 8, `from __future__ import annotations`, type hints on all public methods
- **Single function ≤ 50 lines**
- **Comments explain WHY, not WHAT**
- No hardcoded secrets or magic numbers

## Commit Format

```
<type>(<scope>): <subject>
```

Types: `feat` · `fix` · `refactor` · `perf` · `test` · `docs` · `chore` · `ci`

Examples:
```
feat(sam3): add text concept prompt support
fix(compose): correct redis healthcheck auth flag
docs(architecture): update volume table for dual backends
```

## Pull Request Checklist

- [ ] `make health` passes on a clean stack
- [ ] No secrets committed (check `.env` not staged)
- [ ] Tests pass: `pytest ml-backends/sam3-image/tests ml-backends/sam3-video/tests`
- [ ] `.env.example` updated if new core env vars added
- [ ] `.env.ml.example` updated if new SAM3 env vars added
- [ ] `docs/configuration.md` updated if new env vars added
