# label-anything-sam

Production-ready Label Studio deployment stack with optional SAM3 and SAM2.1 ML backends.

Traditional Chinese version: [README.zh-TW.md](README.zh-TW.md)

## Why this repository exists

As of 2026-04, the upstream [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend) does not provide a maintained SAM3 integration path for production deployment. This repository provides a practical, deployment-focused stack:

- Core services: Label Studio + PostgreSQL + Redis + MinIO + Nginx + Cloudflare Tunnel
- Optional GPU overlays: SAM3 image/video backends and SAM2.1 image/video backends
- Security-first defaults for S3 access, token usage, and network exposure

> [!NOTE]
> Version guidance:
>
> - `v1.0.3` is the latest stable release of the native PostgreSQL line (no Supabase required).
>   It includes all SAM3 fixes and enhancements: native point embeddings (image + video),
>   mask selection modes (`adaptive`/`top1`/`topk`/`threshold`/`all`), runtime threshold and
>   selection-mode UI overrides, bidirectional video tracking, multi-object track merging,
>   and dual text-prompt fields (pure vs mixed-use).
> - For the Supabase-integrated line, use `main` or release `v1.1.3`.
>
> ```bash
> git fetch --tags
> git checkout tags/v1.0.3 -b local-v1-native-pg
> ```

## 5-Minute Quick Start

```bash
git clone https://github.com/felimet/label-anything-sam
cd label-anything-sam

# 1) Core stack
cp .env.example .env
# Fill all <PLACEHOLDER> values
# LABEL_STUDIO_USER_TOKEN must be <= 40 chars (use: openssl rand -hex 20)

make up
make init-minio

# 2) Optional ML backends (GPU)
cp .env.ml.example .env.ml
# Set LABEL_STUDIO_API_KEY (Legacy Token) and HF_TOKEN

make ml-up
```

Open:

- Label Studio: `http://localhost:18090`
- MinIO Console: `http://localhost:19001`
- MinIO Full Admin UI: `http://localhost:19002`

Verify stack health:

```bash
make health
```

## Critical Notes Before You Continue

- Use **Legacy Token** for ML backends, not Personal Access Token.
- Use `MINIO_LS_ACCESS_ID` / `MINIO_LS_SECRET_KEY` for Label Studio S3 storage, never root credentials.
- Rotate MinIO service-account credentials immediately after first deployment.
- If changing `.env`, recreate containers (`down` + `up`) instead of only `restart`.

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
- `make build-sam3-image / build-sam3-video / build-sam21-image / build-sam21-video`: Build ML images
- `make test-sam3-image / test-sam3-video / test-sam21-image / test-sam21-video`: Run ML backend tests
- `make init-minio`: One-time bucket and service-account initialization
- `make health`: End-to-end health checks

## License

Apache-2.0 © 2026 Jia-Ming Zhou
