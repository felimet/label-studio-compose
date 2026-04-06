# Runbook

Operations reference for the Label Studio production stack.

## Health Checks

```bash
make health          # full stack check (PostgreSQL · Redis · MinIO · LS · Nginx · SAM3)
```

Individual service health endpoints:

| Service | Endpoint | Expected |
|---------|----------|---------- |
| Label Studio | `http://localhost:8085/health` *(dev port)* | HTTP 200 |
| nginx | `http://localhost:8090/health` *(dev port)* | `OK` |
| sam3-image-backend | `http://sam3-image-backend:9090/health` *(internal)* | HTTP 200 |
| sam3-video-backend | `http://sam3-video-backend:9090/health` *(internal)* | HTTP 200 |
| MinIO | `mc ready local` *(via docker exec)* | `The cluster is ready` |

## Deployment

### First-time deployment

```bash
cp .env.example .env
# Fill in all <PLACEHOLDER> values — see docs/configuration.md
# Generate secrets:
openssl rand -hex 32    # LABEL_STUDIO_SECRET_KEY, LABEL_STUDIO_USER_TOKEN
openssl rand -base64 24 # POSTGRES_PASSWORD, REDIS_PASSWORD, MINIO_ROOT_PASSWORD

docker compose up -d
make init-minio         # create bucket + CORS (one-time)
make health
```

### Upgrade services

```bash
# 1. Update version pins in .env (LABEL_STUDIO_VERSION, POSTGRES_VERSION, etc.)
# 2. Pull new images
docker compose pull

# 3. Rolling restart (LS → nginx → cloudflared)
docker compose up -d --no-deps label-studio
docker compose up -d --no-deps nginx
docker compose up -d --no-deps cloudflared

# 4. Verify
make health
```

### Start ML backends (GPU)

```bash
make ml-up             # core stack + sam3-image-backend + sam3-video-backend
make health            # includes SAM3 checks
```

After startup, register each backend in Label Studio:

1. **Project → Settings → Machine Learning → Add Model**
2. Image backend URL: `http://sam3-image-backend:9090`
3. Video backend URL: `http://sam3-video-backend:9090`
4. Click **Validate and Save**, enable **Auto-Annotation**

## Rollback

### Application rollback

```bash
# Revert LABEL_STUDIO_VERSION in .env to previous value, then:
docker compose up -d --no-deps label-studio
```

> PostgreSQL schema migrations are not automatically reversed. If the new version ran `migrate`, rolling back may require a database restore.

### Database restore from backup

```bash
docker compose stop label-studio
docker compose exec db psql -U labelstudio -c "DROP DATABASE labelstudio;"
docker compose exec db psql -U labelstudio -c "CREATE DATABASE labelstudio;"
docker compose exec -T db psql -U labelstudio labelstudio < backup.sql
docker compose start label-studio
```

## Common Issues

### Label Studio won't start — "database does not exist"

```bash
# Check postgres health
docker compose exec db pg_isready -U labelstudio

# Inspect init log
docker compose logs db | grep -i error
```

If the database was never created, re-run init:
```bash
docker compose down
docker volume rm label-studio_postgres-data   # ⚠️ deletes all data
docker compose up -d db
docker compose logs -f db  # wait for "database system is ready"
```

### Label Studio 登入 403 CSRF verification failed

原因：nginx 的 `X-Forwarded-Proto` 送出 `http`（Cloudflare Tunnel 以 HTTP 連到 nginx），Django CSRF 與瀏覽器的 HTTPS Origin 不符。

確認 nginx 設定是否已套用：
```bash
docker compose exec nginx grep -r "X-Forwarded-Proto" /etc/nginx/conf.d/
# 應顯示: proxy_set_header   X-Forwarded-Proto https;
```

若未套用（`$scheme` 而非 `https`）：
```bash
docker compose up -d --no-deps nginx
```

同時確認 `LABEL_STUDIO_HOST` 含 `https://` 前綴（影響 `CSRF_TRUSTED_ORIGINS`）。

### 首次登入帳號不存在 / 登入後 500 "No memberships found"

`DEFAULT_USERNAME`/`DEFAULT_USER_PASSWORD` 只在 Postgres DB **首次初始化**時生效。若 DB 已存在，需手動建立：

```bash
docker compose exec label-studio bash -c '
cd /label-studio
python label_studio/manage.py shell -c "
from django.contrib.auth import get_user_model
from organizations.models import Organization
U = get_user_model()
user = U.objects.create_superuser(email=\"admin@example.com\", password=\"your_password\")
Organization.create_organization(created_by=user, title=\"Default\")
print(\"Done\")
"'
```

> 若 DB 已存在但帳號不存在（如重建容器未 `down -v`），執行上述指令即可；Organization 建立後才能正常登入。

### MinIO bucket not found

```bash
make init-minio
# Or manually:
docker compose run --rm minio-init
```

### SAM3 backend — model download fails

Common causes:
- `HF_TOKEN` not set or expired
- Meta license not accepted at https://huggingface.co/facebook/sam3.1

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  logs sam3-image-backend | grep -i error
```

Fix: update `HF_TOKEN` in `.env`, then:
```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  restart sam3-image-backend sam3-video-backend
```

### SAM3 backend — CUDA out of memory

Reduce concurrent workers (already set to 1 in Dockerfile CMD). If OOM still occurs:

1. Close other GPU workloads
2. Use `DEVICE=cpu` in `.env` (very slow, no GPU required)
3. Try a smaller model: `SAM3_IMAGE_MODEL_ID=facebook/sam3`

### Redis connection refused

```bash
docker compose exec redis redis-cli -a "$REDIS_PASSWORD" ping
# Expected: PONG
docker compose logs redis | tail -20
```

### nginx 502 Bad Gateway

```bash
docker compose ps label-studio          # check healthy status
docker compose logs label-studio --tail=50
```

Label Studio typically takes 60–90 s to start on first boot.

## Log Access

```bash
make logs                                     # all services, last 100 lines
docker compose logs -f label-studio           # LS only
docker compose logs -f db redis               # infra only

# ML backends
docker compose -f docker-compose.yml -f docker-compose.ml.yml \
  logs -f sam3-image-backend sam3-video-backend
```

Structured JSON logs are enabled (`JSON_LOG=1`). Use `jq` to filter:

```bash
docker compose logs label-studio 2>&1 | jq 'select(.level=="ERROR")'
```

## Backup

### Data volumes

所有資料目錄已使用 bind mount，直接存在專案根目錄，可用一般檔案工具備份。

```bash
# Label Studio 標注資料 + Local files
tar -czf ls-data-$(date +%Y%m%d).tar.gz ./label-studio-data/

# PostgreSQL — 必須用 pg_dump（postgres-data/ 是內部格式，直接複製無法還原）
docker compose exec db pg_dump -U labelstudio labelstudio \
  > backup-$(date +%Y%m%d).sql

# MinIO 媒體檔案
tar -czf minio-data-$(date +%Y%m%d).tar.gz ./minio-data/
```

### Exclude from backup

- `redis-data/` — 任務佇列暫存，重啟後自動恢復，不需備份
- `hf-cache`, `sam3-image-models`, `sam3-video-models` — 可從 HuggingFace 重新下載

## Monitoring

No bundled monitoring stack. Recommended additions:

| Tool | Purpose |
|------|---------|
| Prometheus + cAdvisor | Container resource metrics |
| Loki + Promtail | Log aggregation (forward docker json logs) |
| Grafana | Dashboard for above |
| Uptime Kuma | External endpoint health checks |

Cloudflare provides basic WAF analytics and request metrics for public endpoints.
