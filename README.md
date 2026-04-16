# label-anything-sam

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Supabase Standalone](https://img.shields.io/badge/Supabase-Standalone-3ECF8E?logo=supabase&logoColor=white)
![Label Studio](https://img.shields.io/badge/Label%20Studio-Production%20Stack-7F52FF)
![SAM Backends](https://img.shields.io/badge/SAM-SAM3%20%2B%20SAM2.1-FF6B35)

Traditional Chinese version: [README.zh-TW.md](README.zh-TW.md)

## Why this repository exists

As of 2026-04, the upstream [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend) does not provide a maintained SAM3 integration path for production deployment. This repository provides a practical, deployment-focused stack:

- Core services: Label Studio + PostgreSQL + Redis + MinIO + Nginx + Cloudflare Tunnel
- Optional GPU overlays: SAM3 image/video backends and SAM2.1 image/video backends
- Security-first defaults for S3 access, token usage, and network exposure

> [!NOTE]
> Version guidance:
>
> - `main` and release `v1.1.1` include the SAM3 image Point Prompt fix:
>   native SAM3 point embeddings are used first; tiny-box fallback is used only when runtime support is missing.
> - If you prefer native PostgreSQL mode (no Supabase), use release `v1.0.1` (hotfix line based on pre-Supabase baseline).
> - You can get each line in any of these ways:
>   1. Git checkout (recommended for local dev)
>   2. Download `Source code (zip)` from the corresponding Release
>   3. Switch Branch/Tag in GitHub UI
>
> ```bash
> git fetch --tags
> git checkout tags/v1.1.1 -b local-main-v1.1.1
> # or
> git checkout tags/v1.0.1 -b local-v1-native-pg
> ```
>
> In `v1.0.1`, Label Studio data is stored in native PostgreSQL (`pg-db`) and does not require `.env.supabase` or `make supabase-up`.

## Quick Start

```bash
git clone https://github.com/felimet/label-anything-sam
cd label-anything-sam

# 1) Core stack
cp .env.example .env
# Supabase standalone is required for default Label Studio DB path in this branch
cp .env.supabase.example .env.supabase
# Fill all <PLACEHOLDER> values
# LABEL_STUDIO_USER_TOKEN must be <= 40 chars (use: openssl rand -hex 20)
# IMPORTANT: keep POSTGRES_PASSWORD in .env and .env.supabase the same

make supabase-up SUPABASE_STANDALONE_ENV=.env.supabase
make up
make init-minio

# 2) Optional ML backends (GPU)
cp .env.ml.example .env.ml
# Set LABEL_STUDIO_API_KEY (Legacy Token) and HF_TOKEN

make ml-up

# 3) Optional RedisInsight (Redis GUI)
cp .env.tools.example .env.tools
make tools-up

# 4) Supabase management command aliases
# (already started in step 1 for default DB route)
# make supabase-up / make supabase-down / make supabase-logs
```

Overlay minimal example for Label Studio integration (NOT part of this branch runtime flow):

```bash
# Example pairing only:
# docker-compose.supabase.sample.yml + .env.supabase.sample.template
cp .env.supabase.sample.template .env.supabase.sample
make supabase-sample-up SUPABASE_SAMPLE_ENV=.env.supabase.sample
```

Optional Cloudflare Tunnel admin routes are configured in Cloudflare UI (not via env vars), for example:

```text
supabase-studio.example.com -> http://supabase-studio:3000
supabase-meta.example.com   -> http://supabase-meta:8080
redisinsight.example.com    -> http://redisinsight:5540
```

If `SUPABASE_META_CONTAINER_PORT` is changed, update the `supabase-meta` target port in Cloudflare accordingly.

See [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md) for the complete mapping table and CF Access requirements.

Open:

- Label Studio: `http://localhost:18090`
- MinIO Console: `http://localhost:19001`
- MinIO Full Admin UI: `http://localhost:19002`

Verify stack health:

```bash
make health
```

## Direct Compose (Without Make)

If you prefer typing `docker compose -f ... up` directly, keep two safeguards enabled:

1. Interpolation safeguard: always provide a fixed project name plus explicit env files.
2. Runtime safeguard: keep service-level `env_file` (for ML backends) and required vars (`${VAR:?}`) in compose files.

PowerShell session defaults (recommended):

```powershell
$env:COMPOSE_PROJECT_NAME = "label-anything-sam"
```

Standalone Supabase (default branch runtime):

```bash
docker compose --project-name label-anything-sam \
	--env-file .env --env-file .env.supabase \
	-f docker-compose.supabase.yml up -d
```

Supabase sample mode:

```bash
docker compose --project-name label-anything-sam \
	--env-file .env --env-file .env.supabase.sample \
	-f docker-compose.yml -f docker-compose.override.yml -f docker-compose.supabase.sample.yml up -d
```

ML overlays:

```bash
docker compose --project-name label-anything-sam \
	--env-file .env \
	-f docker-compose.yml -f docker-compose.override.yml -f docker-compose.ml.yml up -d
```

Optional one-liner fallback (`--env-file` omitted):

```powershell
$env:COMPOSE_ENV_FILES = ".env,.env.supabase"
docker compose -f docker-compose.supabase.yml config -q
```

`COMPOSE_ENV_FILES` is only used when CLI `--env-file` is not provided.

## Critical Notes Before You Continue

- Use **Legacy Token** for ML backends, not Personal Access Token.
- Use `MINIO_LS_ACCESS_ID` / `MINIO_LS_SECRET_KEY` for Label Studio S3 storage, never root credentials.
- Rotate MinIO service-account credentials immediately after first deployment.
- If changing `.env`, recreate containers (`down` + `up`) instead of only `restart`.

## Environment Profiles

To avoid one oversized env file, variables are split by scope:

- `.env.example` → `.env`: Core runtime stack (required)
- `.env.ml.example` → `.env.ml`: SAM3/SAM2.1 backends (optional)
- `.env.tools.example` → `.env.tools`: Local dev tools such as RedisInsight (optional)
- `.env.supabase.example` → `.env.supabase`: Supabase standalone management stack (required)
- `.env.supabase.sample.template` → `.env.supabase.sample`: Supabase minimal example mode (documentation/demo only)

Supabase mode boundaries:

- Runtime mode (this branch): `docker-compose.supabase.yml` + `.env.supabase`
- Example mode only: `docker-compose.supabase.sample.yml` + `.env.supabase.sample`

`.env.example` is the single complete core template.

## Start Here By Role

| Role | Start Here | Cookbook | Deep Dive |
|------|------------|----------|-----------|
| End User / Project Admin | [docs/README.md](docs/README.md) | [docs/cookbook/user-cookbook.md](docs/cookbook/user-cookbook.md) | [docs/user-guide.md](docs/user-guide.md) |
| Developer | [docs/README.md](docs/README.md) | [docs/cookbook/developer-cookbook.md](docs/cookbook/developer-cookbook.md) | [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) |
| Operator / SRE | [docs/README.md](docs/README.md) | [docs/cookbook/ops-cookbook.md](docs/cookbook/ops-cookbook.md) | [docs/RUNBOOK.md](docs/RUNBOOK.md) |

## Documentation Map

- [docs/README.md](docs/README.md): Documentation hub and reading paths
- [docs/user-guide.md](docs/user-guide.md): User workflows and admin operations
- [docs/configuration.md](docs/configuration.md): Single source of truth for environment variables
- [docs/architecture.md](docs/architecture.md): Topology, data flow, and security design
- [docs/cloudflare-tunnel.md](docs/cloudflare-tunnel.md): Public exposure, tunnel, and WAF setup
- [docs/sam3-backend.md](docs/sam3-backend.md): SAM3 backend behavior and constraints
- [docs/sam21-backend.md](docs/sam21-backend.md): SAM2.1 backend behavior and constraints
- [docs/RUNBOOK.md](docs/RUNBOOK.md): Operations, incident response, backup and restore
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md): Development workflow and contribution policy

## Make Targets (Short List)

- `make up / down / restart / logs / ps`: Core stack lifecycle
- `make ml-up / ml-down`: Core stack with ML overlays
- `make tools-up / tools-down / tools-logs`: RedisInsight local GUI overlay
- `make supabase-up / supabase-down / supabase-logs`: Supabase management (standalone stack, default)
- `make supabase-standalone-up / supabase-standalone-down / supabase-standalone-logs`: Explicit standalone aliases
- `make supabase-sample-up / supabase-sample-down / supabase-sample-logs`: Supabase minimal example mode (studio + meta)
- `make build-sam3-image / build-sam3-video / build-sam21-image / build-sam21-video`: Build ML images
- `make test-sam3-image / test-sam3-video / test-sam21-image / test-sam21-video`: Run ML backend tests
- `make init-minio`: One-time bucket and service-account initialization
- `make health`: End-to-end health checks

## License

Apache-2.0 © 2026 Jia-Ming Zhou
