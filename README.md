# label-anything-sam

[Label Studio](https://labelstud.io) SAM3/3.1 Full Deployment Setup: PostgreSQL · Redis · MinIO (S3) · Nginx · Cloudflare Tunnel · SAM3/3.1 Interactive Image Segmentation

> As of April 11, 2026, [Label Studio ml-backend](https://github.com/HumanSignal/label-studio-ml-backend) has not yet been updated to include [Meta (Facebook) SAM 3: Segment Anything with Concepts](https://github.com/facebookresearch/sam3) as a backend segmentation model. This repo therefore provides a custom Label Studio ML Backend implementation based on SAM 3 for users who need direct integration, following the folder structure of [Label Studio ml-backend](https://github.com/HumanSignal/label-studio-ml-backend) as closely as possible. See `./ml-backends` for details.

> **繁體中文說明** → [README.zh-TW.md](README.zh-TW.md)

## Stack

| Service | Image | Role | Data Explain Link |
|---------|-------|------|------|
| `label-studio` | `heartexlabs/label-studio:latest` | Labeling UI + API | [`./ls-data/`](docs/configuration.md#pg-db-vs-ls-data--資料分層說明) — exports, local files |
| `pg-db` | `postgres:17` | Metadata store | [`./postgres-data/`](docs/configuration.md#pg-db-vs-ls-data--資料分層說明) — tasks, annotations, users |
| `redis` | `redis:8.6.2` | Task queue / cache | [`./redis-data/`](docs/configuration.md#資料目錄說明) — transient queue state |
| `minio` | `firstfinger/minio:latest` | S3-compatible object storage + full Admin UI (port 9002) | [`./minio-data/`](docs/configuration.md#minio) — media files |
| `minio-init` | `minio/mc:RELEASE.2025-08-13T08-35-41Z` | One-shot bucket init + service account + quota | — |
| `nginx` | `nginx:1.28.3-alpine3.23` | Reverse proxy | — |
| `cloudflared` | `cloudflare/cloudflared:2026.3.0` | Zero Trust tunnel | — |
| `sam3-image-backend` | (custom build) | SAM3 image segmentation → BrushLabels *(GPU, optional)* | `hf-cache` (shared volume) — model weights |
| `sam3-video-backend` | (custom build) | SAM3 video object tracking → VideoRectangle *(GPU, optional)* | `hf-cache` (shared volume) — model weights |

> **MinIO CE note**: MinIO removed all Admin UI from Community Edition on 2025-05-24 and stopped pushing CE images to Docker Hub after 2025-09-07. This stack uses `firstfinger/minio` — a daily build from upstream source that restores the full Admin Console (ports 9001 Console, 9002 Full Admin UI). Ref: [Harsh-2002/MinIO](https://github.com/Harsh-2002/MinIO)

## Prerequisites

- Docker Engine ≥ 26 + Docker Compose v2
- NVIDIA GPU + `nvidia-container-toolkit` (SAM3 backend only)
- Cloudflare account with Zero Trust enabled
- HuggingFace account — Meta `facebook/sam3.1` license accepted

## Quick Start

```bash
git clone https://github.com/felimet/label-anything-sam
cd label-anything-sam

# 1. Core stack
cp .env.example .env
$EDITOR .env           # fill every <PLACEHOLDER>
                       # LABEL_STUDIO_USER_TOKEN: openssl rand -hex 20  (must be ≤40 chars)

make up                # start core stack (admin account auto-created on first boot)
make init-minio        # create S3 bucket + policies

# 2. Get the Label Studio API token (needed for SAM3 backends)
#    Login → Avatar (top-right) → Account & Settings → Legacy Token → Copy
#    ⚠ Must use Legacy Token (NOT Personal Access Token) — ML SDK sends
#      "Authorization: Token <key>"; PAT uses JWT Bearer → 401 Unauthorized.

# 3. SAM3 ML backends (optional, requires NVIDIA GPU)
cp .env.ml.example .env.ml
$EDITOR .env.ml        # set LABEL_STUDIO_API_KEY (from step 2) and HF_TOKEN

make ml-up
```

Connect MinIO storage in Label Studio:
**Project → Settings → Cloud Storage → Add Source Storage → S3**
(endpoint: `http://minio:9000`, use `MINIO_LS_ACCESS_ID` / `MINIO_LS_SECRET_KEY` — the least-privilege service account created by `make init-minio`. Do **not** use root credentials here).

> **⚠️ After first deployment:** rotate the service account secret immediately.
> Admin UI (`http://localhost:19002`) → top-right avatar → **Access Keys → Change Password**
> Update `MINIO_LS_SECRET_KEY` in `.env` and in the LS Cloud Storage settings.

## Makefile Reference

| Target | Description |
|--------|-------------|
| `up / down / restart / logs / ps` | Core stack lifecycle |
| `ml-up / ml-down` | SAM3 ML overlay (image + video) |
| `build-sam3-image / build-sam3-video` | Build ML backend images |
| `test-sam3-image / test-sam3-video` | Run pytest in containers |
| `init-minio` | One-time bucket initialisation |
| `create-admin` | Create superuser |
| `health` | Check all services |
| `push` | git add + commit + push |

## Documentation

| Guide | Contents |
|-------|----------|
| [docs/configuration.md](docs/configuration.md) | `.env` variable reference · [MinIO Access Policy](docs/configuration.md#minio-bucket-access-policy) · [Bucket Encryption](docs/configuration.md#bucket-encryptionsse-s3--sse-kms) · [pg-db vs ls-data](docs/configuration.md#pg-db-vs-ls-data--資料分層說明) |
| [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md) | Zero Trust setup + WAF rules + alternatives (ngrok) |
| [docs/sam3-backend.md](docs/sam3-backend.md) | SAM3 model setup + annotation workflow |
| [docs/architecture.md](docs/architecture.md) | Service topology, volumes, networking |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Operations guide (health checks, upgrades, troubleshooting) |

## License

Apache-2.0 © 2026 Jia-Ming Zhou
