# Contributing Guide

> Audience: developers and contributors
>
> Covers: local setup, testing workflow, commit and PR expectations
>
> Fast task recipes: [cookbook/developer-cookbook.md](cookbook/developer-cookbook.md)
>
> Full docs map: [README.md](README.md)

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
git clone https://github.com/felimet/label-anything-sam
cd label-anything-sam

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

# Optional local tools (RedisInsight)
cp .env.tools.example .env.tools

# Supabase management (default in this branch)
# Pairing: docker-compose.supabase.yml + .env.supabase
cp .env.supabase.example .env.supabase
# IMPORTANT: keep POSTGRES_PASSWORD in .env and .env.supabase the same.

# Supabase minimal example mode for Label Studio integration (not in this branch runtime flow)
# Pairing: docker-compose.supabase.sample.yml + .env.supabase.sample
cp .env.supabase.sample.template .env.supabase.sample

# Start Supabase standalone first (default Label Studio DB target uses supavisor)
make supabase-up SUPABASE_STANDALONE_ENV=.env.supabase

# Start core stack (exposed on dev ports — see docker-compose.override.yml)
make up
make init-minio       # first time only
make tools-up         # optional: RedisInsight local GUI

# Verify
make health
```

### Direct `docker compose` (without Make)

For contributors who run Compose manually, keep both protections:

1. Interpolation: always set project name and explicit env files.
2. Runtime: keep service-level `env_file` and `${VAR:?}` required checks.

PowerShell example:

```powershell
$env:COMPOSE_PROJECT_NAME = "label-anything-sam"
docker compose --project-name label-anything-sam --env-file .env --env-file .env.supabase -f docker-compose.supabase.yml config -q
docker compose --project-name label-anything-sam --env-file .env --env-file .env.supabase -f docker-compose.supabase.yml up -d
```

Optional fallback when not passing `--env-file`:

```powershell
$env:COMPOSE_ENV_FILES = ".env,.env.supabase"
docker compose -f docker-compose.supabase.yml config -q
```

`COMPOSE_ENV_FILES` is ignored if `--env-file` is provided.

Dev override ports ([docker-compose.override.yml](../docker-compose.override.yml)):

<!-- AUTO-GENERATED from docker-compose.override.yml -->
| Service | Host port | Notes |
|---------|-----------|-------|
| nginx | 18090 | Label Studio reverse proxy entry point |
| label-studio | 18086 | Django app direct access (bypass nginx) |
| minio API | 19000 | S3 endpoint (`aws s3`, SDK, presigned URL) |
| minio console | 19001 | MinIO admin UI (`http://localhost:19001`) |
| postgres | 5433 | Avoid conflict with local PostgreSQL |
| redis | 16380 | Avoid conflict with local Redis |
<!-- END AUTO-GENERATED -->

> **Windows 注意**：8000–9000 附近的 port 常被 Hyper-V 保留；若 bind 失敗改用 18000+ 範圍。

Optional example-mode ports:

| Service | Host port | Notes |
|---------|-----------|-------|
| redisinsight | 127.0.0.1:15540 (default) | Redis GUI overlay (`make tools-up`) |
| supabase-studio | 127.0.0.1:18091 (default) | Supabase Studio 管理 UI（僅示例模式） |
| supabase-meta | 127.0.0.1:18087 (default) | Supabase Postgres Meta REST API（僅示例模式） |

## Available Commands

<!-- AUTO-GENERATED from Makefile -->
| Command | Description |
|---------|-------------|
| `make up` | Start core stack (postgres · redis · minio · label-studio · nginx · cloudflared) |
| `make down` | Stop core stack |
| `make restart` | Restart all core services |
| `make logs` | Follow logs (last 100 lines) |
| `make ps` | Show container status |
| `make ml-up` | Start core stack + SAM3/SAM2.1 image/video backends (GPU required) |
| `make ml-down` | Stop all services (core + ML overlays) |
| `make tools-up` | Start RedisInsight local GUI overlay |
| `make tools-down` | Stop RedisInsight local GUI overlay |
| `make tools-logs` | Follow RedisInsight logs |
| `make supabase-up SUPABASE_STANDALONE_ENV=...` | Start Supabase management stack (standalone default) |
| `make supabase-down SUPABASE_STANDALONE_ENV=...` | Stop Supabase management stack |
| `make supabase-logs SUPABASE_STANDALONE_ENV=...` | Follow logs for Supabase management stack |
| `make supabase-standalone-up SUPABASE_STANDALONE_ENV=...` | Start Supabase standalone management stack (without native pg-db) |
| `make supabase-standalone-down SUPABASE_STANDALONE_ENV=...` | Stop Supabase standalone management stack |
| `make supabase-standalone-logs SUPABASE_STANDALONE_ENV=...` | Follow logs for Supabase standalone management stack |
| `make supabase-sample-up SUPABASE_SAMPLE_ENV=...` | Start Supabase minimal example mode (supabase-studio + supabase-meta) |
| `make supabase-sample-down` | Stop Supabase minimal example mode |
| `make supabase-sample-logs SUPABASE_SAMPLE_ENV=...` | Follow logs for Supabase minimal example mode |
| `make build-sam3-image` | Build SAM3 image backend Docker image |
| `make build-sam3-video` | Build SAM3 video backend Docker image |
| `make test-sam3-image` | Run pytest inside sam3-image-backend container |
| `make test-sam3-video` | Run pytest inside sam3-video-backend container |
| `make init-minio` | One-time: create S3 bucket + service account policy (minio-init container) |
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
- [ ] `.env.supabase.sample.template` updated if new Supabase example-mode env vars added
- [ ] `.env.supabase.example` updated if new Supabase standalone stack env vars added
- [ ] `.env.tools.example` updated if new local tools env vars added
- [ ] `docs/configuration.md` updated if new env vars added
